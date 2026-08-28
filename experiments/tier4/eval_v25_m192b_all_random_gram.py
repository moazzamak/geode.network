"""M192b — all-random field-based Gram sharing (2-of-3 replicated
Shamir over q = 2**31 - 1).

Registered in ``RESEARCH_IMPLEMENTATION_PLAN_v25.md`` section 6
(19 Aug 2026, before the build). Unlike the M192 additive protocol,
every share is individually uniform over the field — no residual
share — so a single party's view carries no row information at all.

The Gram protocol: with share pairs {s_p, s_{p+1}} and public
Lagrange coefficients λ, each party computes

    local_p = ½λ_p² s_pᵀs_p + λ_pλ_q (s_pᵀs_q + s_qᵀs_p)
              + ½λ_q² s_qᵀs_q

mod q (the diagonal is halved because each s_pᵀs_p appears in two
parties' locals); every index pair appears in exactly one party's
local set, so the sum of locals equals the field Gram of the
quantized rows exactly.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.data_cache import (
    configure_external_cache_environment,
    data_cache_root,
)
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from experiments.tier4.eval_v15_m104_experts import RidgeAccumulator
from experiments.tier4.eval_v16_m109_trunk import _load_corpus
from geode.privacy.secret_sharing import PRIME, shamir_reconstruct

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m192b_all_random_gram.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m192b_all_random_gram")

CLASSES = 345
BLOCK = 4096
PENALTY = 1.0
ANCHOR_V_MS = 0.24214492753623187
TOL = 1e-9

FIELD = PRIME  # 2**61 - 1: entries <= 2^24, products <= 2^48,
# Gram sums <= 2^57 < 2^61 — no wraps (registered repair)


def _lagrange_coeffs(x_points: list[int], target: int) -> list[int]:
    """Lagrange coefficients λ_i with sum_i λ_i share_i(x) == f(target)."""
    coeffs = []
    for i, xi in enumerate(x_points):
        num, den = 1, 1
        for j, xj in enumerate(x_points):
            if i == j:
                continue
            num = (num * (target - xj)) % FIELD
            den = (den * (xi - xj)) % FIELD
        coeffs.append((num * pow(den, FIELD - 2, FIELD)) % FIELD)
    return coeffs


def _shamir_vector_share(values: np.ndarray, k: int, n: int, rng
                         ) -> list[np.ndarray]:
    """Per-entry Shamir shares of an integer vector mod q."""
    shares = [np.zeros(len(values), dtype=np.int64) for _ in range(n)]
    for i, v in enumerate(values):
        coeffs = [int(v) % FIELD] + [int(rng.integers(0, FIELD))
                                     for _ in range(k - 1)]
        for x in range(1, n + 1):
            shares[x - 1][i] = sum(
                c * pow(x, d, FIELD) for d, c in enumerate(coeffs)
            ) % FIELD
    return shares


def _mod_outer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """outer(a, b) mod FIELD with Python-integer arithmetic
    (correctness-first; values < 2^61 so int64 products would
    overflow)."""
    m = len(a)
    out = [[0] * m for _ in range(m)]
    ai = [int(v) for v in a]
    bi = [int(v) for v in b]
    for i in range(m):
        row = out[i]
        ai_v = ai[i]
        for j in range(m):
            row[j] = (ai_v * bi[j]) % FIELD
    return np.asarray(out, dtype=object)


def _party_local(sp: np.ndarray, sq: np.ndarray, lp: int, lq: int
                 ) -> np.ndarray:
    """½λ_p² s_pᵀs_p + λ_pλ_q (s_pᵀs_q + s_qᵀs_p) + ½λ_q² s_qᵀs_q mod q
    (the diagonal is halved because each s_pᵀs_p appears in two
    parties' locals)."""
    half = (FIELD + 1) // 2
    lpp = (lp * lp) % FIELD * half % FIELD
    lpq = (lp * lq) % FIELD
    lqq = (lq * lq) % FIELD * half % FIELD
    diag_p = (_mod_outer(sp, sp) * lpp) % FIELD
    diag_q = (_mod_outer(sq, sq) * lqq) % FIELD
    cross = ((_mod_outer(sp, sq) + _mod_outer(sq, sp)) * lpq) % FIELD
    return (diag_p + cross + diag_q) % FIELD


def run_m192b(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")
    started = time.time()

    configure_external_cache_environment()
    corpus, _ti, _tei = _load_corpus(config)
    test_labels = corpus["test_labels"]
    ms_cache = data_cache_root() / config["artifacts"]["cache_relpath"]
    ms_test_cache = data_cache_root() \
        / config["artifacts"]["test_cache_relpath"]
    labels = np.load(data_cache_root()
                     / config["artifacts"]["labels_file"])["labels"]
    mem_train = np.load(ms_cache / config["artifacts"]["train_file"],
                        mmap_mode="r")
    mem_test = np.load(ms_test_cache / config["artifacts"]["test_file"],
                       mmap_mode="r")

    # ---- environment anchor -------------------------------------------------
    acc = RidgeAccumulator(mem_train.shape[1], CLASSES)
    for start in range(0, len(labels), BLOCK):
        stop = min(start + BLOCK, len(labels))
        acc.add(mem_train[start:stop], labels[start:stop])
    weights = acc.solve_many([PENALTY])[PENALTY]
    standardise = acc.standardiser()
    preds = np.empty(len(test_labels), dtype=np.int64)
    for start in range(0, len(test_labels), BLOCK):
        stop = min(start + BLOCK, len(test_labels))
        s = (standardise(np.asarray(mem_test[start:stop],
                                    dtype=np.float64))
             @ weights[:-1] + weights[-1])
        preds[start:stop] = np.argmax(s, axis=1)
    anchor_value = float((preds == test_labels).mean())
    anchor = {"measured": anchor_value, "sealed": ANCHOR_V_MS,
              "delta": anchor_value - ANCHOR_V_MS, "tolerance": TOL,
              "ok": abs(anchor_value - ANCHOR_V_MS) <= TOL}
    print(f"anchor V_ms {anchor_value:.15f} delta "
          f"{anchor['delta']:+.3e}", flush=True)

    cell = config["cell"]
    cols = int(cell["columns"])
    rows = int(cell["rows"])
    scale = int(cell["fixed_point_scale"])
    k, n = int(cell["threshold_k"]), int(cell["parties"])
    rng = np.random.default_rng(int(cell["share_seed"]))
    x_points = list(range(1, n + 1))
    lam = _lagrange_coeffs(x_points, 0)

    block = np.asarray(mem_train[:rows, :cols], dtype=np.float64)
    block_int = np.rint(block * scale).astype(np.int64)
    # field Gram of the quantized plaintext (the exact reference)
    gram_ref = np.zeros((cols, cols), dtype=object)
    for r in range(rows):
        row = block_int[r] % FIELD
        gram_ref = (gram_ref + _mod_outer(row, row)) % FIELD

    # share every row; accumulate party locals
    party_grams = [np.zeros((cols, cols), dtype=object)
                   for _ in range(n)]
    for r in range(rows):
        shares = _shamir_vector_share(block_int[r], k, n, rng)
        for p in range(n):
            q = (p + 1) % n
            party_grams[p] = (party_grams[p]
                              + _party_local(shares[p], shares[q],
                                             lam[p], lam[q])) % FIELD
    reconstructed = np.zeros((cols, cols), dtype=object)
    for p in range(n):
        reconstructed = (reconstructed + party_grams[p]) % FIELD
    gram_exact = bool(np.all(np.asarray(reconstructed, dtype=object)
                             == np.asarray(gram_ref, dtype=object)))
    print(f"g2 field Gram bit-exact vs quantized plaintext: "
          f"{gram_exact}", flush=True)

    # g1: any 2-of-3 pair reconstructs the quantized row exactly
    row0_shares = _shamir_vector_share(block_int[0], k, n, rng)
    pairs_ok = True
    for pair in ((1, 2), (2, 3), (1, 3)):
        rec = np.zeros(cols, dtype=object)
        for i in range(cols):
            vals = [(x, int(row0_shares[x - 1][i])) for x in pair]
            rec[i] = shamir_reconstruct(vals, modulus=FIELD)
        if not np.all(np.asarray(rec, dtype=object)
                      == (block_int[0] % FIELD).astype(object)):
            pairs_ok = False
    print(f"g1 all 2-of-3 pairs reconstruct row bit-exact: {pairs_ok}",
          flush=True)

    # g3: quantization fidelity vs float Gram
    gram_float = (block.T @ block)
    gram_dequant = (np.asarray(reconstructed, dtype=np.float64)
                    / (scale * scale))
    fidelity = float(np.abs(gram_dequant - gram_float).max()
                     / max(np.abs(gram_float).max(), 1e-300))
    print(f"g3 quantization fidelity rel {fidelity:.3e}", flush=True)

    # g4: share uniformity over 4,096 independent share values
    n_samples = 4096
    sample_rows = 64
    vals = []
    for r in range(sample_rows):
        vals.extend(int(v) for v in
                    _shamir_vector_share(block_int[r], k, n, rng)[1])
    vals = np.asarray(vals[:n_samples], dtype=np.float64)
    corr_source_row = np.tile(block_int[:sample_rows].reshape(-1),
                              int(np.ceil(n_samples
                                          / (sample_rows * cols))))[:n_samples]
    corr = float(np.corrcoef(vals, corr_source_row)[0, 1])
    mean_frac = float(np.abs(vals.mean() - FIELD / 2) / FIELD)
    std_frac = float(np.abs(vals.std() - FIELD / np.sqrt(12)) / FIELD)
    print(f"g4 share corr {corr:.4f} mean-frac {mean_frac:.4f} "
          f"std-frac {std_frac:.4f}", flush=True)

    gates = {
        "g1_pair_reconstruction": pairs_ok,
        "g2_field_gram_exact": gram_exact,
        "g3_fidelity": {"rel": fidelity,
                        "ok": fidelity <= float(cell["fidelity_tol"])},
        "g4_uniformity": {"corr": abs(float(corr)),
                          "mean_frac": mean_frac,
                          "std_frac": std_frac,
                          "ok": abs(float(corr)) < float(cell["corr_tol"])
                          and mean_frac <= float(cell["band_tol"])
                          and std_frac <= float(cell["band_tol"])},
    }
    gates_ok = anchor["ok"] and pairs_ok and gram_exact \
        and gates["g3_fidelity"]["ok"] and gates["g4_uniformity"]["ok"]

    evidence: dict[str, Any] = {
        "milestone": "M192b",
        "cell": "all-random field-based Gram sharing (2-of-3 Shamir, "
                "q = 2^61-1, 64 cols x 512 rows)",
        "admissible_as_evidence": not inadmissible,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "anchor": anchor,
        "gates": gates,
        "reading": ("every share is uniform over the field (the M192 "
                    "residual-share limitation removed); the field Gram "
                    "reconstructs exactly from the party locals"),
        "void": not gates_ok,
        "void_reason": "" if gates_ok else "one or more M192b gates failed",
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"anchor_ok": anchor["ok"], "gates": gates,
                      "gates_ok": gates_ok}, indent=1), flush=True)
    print(f"M192b complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m192b(args.config, args.output)


if __name__ == "__main__":
    main()
