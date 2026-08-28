"""M192 — secret-shared ridge fit + Byzantine threshold score
reconstruction (the day-one cryptographic privacy tier).

Registered in ``RESEARCH_IMPLEMENTATION_PLAN_v25.md`` section 6
(19 Aug 2026, before the build). Operands: the sealed ms codes
(width 1428). Environment anchor: the plaintext full-data ms ridge
reproduces V_ms at 1e-9 before any sharing measurement.

Cell A (contribution privacy): additive float64 splitting of 20,000
train rows across 3 parties, share-wise Gram accumulation, then
reconstruction. Gates: Gram rel error <= 1e-8 vs the plaintext
RidgeAccumulator Gram; a single party's share is statistically
independent of the row (|pearson r| < 0.05).

Cell B (Byzantine threshold): Shamir 3-of-5 over 2**61-1 on
fixed-point class scores (scale 2**20) of the anchored full-data head
for 8 test rows. Gates: every 3-of-5 subset reconstructs the
plaintext scores bit-exactly; a corrupted share is DETECTED by
subset-consistency disagreement.
"""
from __future__ import annotations

import argparse
import itertools
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
from geode.privacy.secret_sharing import (
    PRIME,
    recombine_additive,
    replicated_gram_shares,
    shamir_reconstruct,
    shamir_split,
    signed_from_field,
    split_additive,
    to_field,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m192_secret_shared.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m192_secret_shared")

CLASSES = 345
BLOCK = 4096
PENALTY = 1.0
ANCHOR_V_MS = 0.24214492753623187
TOL = 1e-9


def _cell_a(mem_train: np.ndarray, labels: np.ndarray,
            cfg: dict[str, Any]) -> dict[str, Any]:
    rows = int(cfg["rows"])
    parties = int(cfg["parties"])
    rng = np.random.default_rng(int(cfg["share_seed"]))

    plain = RidgeAccumulator(mem_train.shape[1], CLASSES)
    for start in range(0, rows, BLOCK):
        stop = min(start + BLOCK, rows)
        plain.add(mem_train[start:stop], labels[start:stop])

    party_grams = [np.zeros((mem_train.shape[1], mem_train.shape[1]),
                             dtype=np.float64) for _ in range(parties)]
    # Block-wise Z-resharing accumulation (the registered repair): each
    # party's share of the Gram includes its local cross terms and the
    # random Z mask, so sum_p C_p == block^T block. Same registered
    # tolerances as the original registration.
    for start in range(0, rows, BLOCK):
        stop = min(start + BLOCK, rows)
        block = np.asarray(mem_train[start:stop], dtype=np.float64)
        shares = replicated_gram_shares(block, parties, rng)
        for p in range(parties):
            party_grams[p] += shares[p]
    reconstructed = np.zeros_like(plain.gram)
    for p in range(parties):
        reconstructed += party_grams[p]
    rel = float(np.abs(reconstructed - plain.gram).max()
                / max(np.abs(plain.gram).max(), 1e-300))
    print(f"  cell A: gram rel {rel:.3e} (tol "
          f"{cfg['gram_reconstruction_rel_tolerance']})", flush=True)

    # privacy: a single share carries no row information
    corrs = []
    for _ in range(int(cfg["privacy_rows"])):
        row = np.asarray(mem_train[rng.integers(0, rows)],
                         dtype=np.float64)
        shares = split_additive(row, parties, rng)
        view = shares[1]
        if np.std(view) > 0 and np.std(row) > 0:
            corrs.append(float(np.corrcoef(view, row)[0, 1]))
    max_abs_corr = max(abs(c) for c in corrs) if corrs else 0.0
    print(f"  cell A: max |corr(share, row)| {max_abs_corr:.4f} "
          f"(tol {cfg['privacy_corr_tolerance']})", flush=True)
    return {
        "gram_reconstruction_rel": rel,
        "gram_ok": rel <= float(cfg["gram_reconstruction_rel_tolerance"]),
        "max_abs_corr_share_row": max_abs_corr,
        "privacy_ok": max_abs_corr < float(cfg["privacy_corr_tolerance"]),
    }


def _cell_b(weights: np.ndarray, standardise, mem_test: np.ndarray,
            test_labels: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    rows = int(cfg["score_rows"])
    k, n = int(cfg["threshold_k"]), int(cfg["parties"])
    scale = int(cfg["fixed_point_scale"])
    rng = np.random.default_rng(int(cfg["shamir_seed"]))

    scores = (standardise(np.asarray(mem_test[:rows], dtype=np.float64))
              @ weights[:-1] + weights[-1])
    scores_int = np.rint(scores * scale).astype(np.int64)

    # split every score scalar into n Shamir shares
    shares = np.empty((n, rows, CLASSES), dtype=np.int64)
    for i in range(rows):
        for c in range(CLASSES):
            for x, value in enumerate(shamir_split(
                    to_field(int(scores_int[i, c])), k, n, rng)):
                shares[x, i, c] = value

    x_values = list(range(1, n + 1))
    subsets = list(itertools.combinations(x_values, k))
    reconstructed = np.zeros((len(subsets), rows, CLASSES), dtype=np.int64)
    for si, subset in enumerate(subsets):
        for i in range(rows):
            for c in range(CLASSES):
                pairs = [(x, int(shares[x - 1, i, c])) for x in subset]
                reconstructed[si, i, c] = signed_from_field(
                    shamir_reconstruct(pairs))
    all_equal = bool(np.all(reconstructed == reconstructed[0]))
    matches_plain = bool(np.all(reconstructed[0] == scores_int))
    b1_ok = all_equal and matches_plain
    print(f"  cell B: all subsets identical={all_equal} "
          f"match plaintext={matches_plain}", flush=True)

    # corrupt one share of one scalar; subset consistency must disagree
    shares[int(cfg["corrupt_party_index"]) - 1,
           int(cfg["corrupt_row"]), int(cfg["corrupt_class"])] = (
               shares[int(cfg["corrupt_party_index"]) - 1,
                      int(cfg["corrupt_row"]),
                      int(cfg["corrupt_class"])]
               + int(rng.integers(1, PRIME))) % PRIME
    distinct: set[int] = set()
    for subset in subsets:
        pairs = [(x, int(shares[x - 1, int(cfg["corrupt_row"]),
                                  int(cfg["corrupt_class"])]))
                 for x in subset]
        distinct.add(signed_from_field(shamir_reconstruct(pairs)))
    detected = len(distinct) > 1
    print(f"  cell B: corrupted-share detection={detected} "
          f"(distinct reconstructions {len(distinct)})", flush=True)
    return {"all_subsets_identical": all_equal,
            "matches_plaintext_scores": matches_plain,
            "b1_ok": b1_ok,
            "corrupted_share_detected": detected,
            "b2_ok": detected}


def run_m192(config_path: Path, output_dir: Path) -> dict[str, Any]:
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
    ms_test_cache = data_cache_root() / config["artifacts"]["test_cache_relpath"]
    labels = np.load(data_cache_root()
                     / config["artifacts"]["labels_file"])["labels"]
    mem_train = np.load(ms_cache / config["artifacts"]["train_file"],
                        mmap_mode="r")
    mem_test = np.load(ms_test_cache / config["artifacts"]["test_file"],
                       mmap_mode="r")
    if len(mem_train) != len(labels):
        raise SystemExit("M192 premise failure: ms train rows != labels")

    # ---- environment anchor: the plaintext full-data head ---------------
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
    if not anchor["ok"]:
        print("M192 VOID: environment anchor failed", flush=True)

    cell_a = _cell_a(mem_train, labels, config["cell_a"])
    cell_b = _cell_b(weights, standardise, mem_test, test_labels,
                     config["cell_b"])
    gates_ok = anchor["ok"] and cell_a["gram_ok"] and cell_a["privacy_ok"] \
        and cell_b["b1_ok"] and cell_b["b2_ok"]

    evidence: dict[str, Any] = {
        "milestone": "M192",
        "cell": "secret-shared ridge fit + Byzantine threshold score "
                "reconstruction (ms codes)",
        "admissible_as_evidence": not inadmissible,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "anchor": anchor,
        "cell_a": cell_a,
        "cell_b": cell_b,
        "prior_art": ["FSS logistic regression (arXiv:2309.09486)",
                      "Shamir-secret regression (arXiv:2109.11200)"],
        "reading": ("cell A: contribution privacy with reconstruction-"
                    "grade Gram equality; cell B: 3-of-5 threshold with "
                    "corruption DETECTION (no error correction claimed)"),
        "void": not gates_ok,
        "void_reason": "" if gates_ok else
        "one or more M192 gates failed",
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"anchor_ok": anchor["ok"], "cell_a": cell_a,
                      "cell_b": cell_b, "gates_ok": gates_ok}, indent=1),
          flush=True)
    print(f"M192 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m192(args.config, args.output)


if __name__ == "__main__":
    main()
