"""M138 — score-level seed ensemble at matched total atoms.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v21.md`` (M138, amended
13 Aug 2026 before measurement) and
``experiments/configs/v16/m138_seed_ensemble.json``.

Question. M126 closed the CONCATENATION variant (fresh atom draws past the pool
cap: flat accuracy, flat eff-rank ~7.8). The unsealed variant is score-level
ensembling: two independently seeded (3072-atom dictionary, ridge head) pairs,
per-row L2-normalised 345-score vectors averaged, at matched total atoms
(2 x 3072 = 6144). Does it beat Q(6144, 138000) = 0.22487?

Members: seed 11 (M117 exact - its accuracy must reproduce the sealed 0.21528
within 0.002) and seed 22 (a fresh permutation of the SAME sealed 8192 pool;
disclosed in the config - the concatenation of this construction family was
measured at M126 and was flat).

Also reported: the effective rank of the joint 24576-width standardised Gram
(participation ratio, the M128 definition) - whether the ensemble's accuracy
moves is read against the rank it achieves.

Reproduce with::

    $env:GEODE_CACHE_DIR="F:\\geode-ml\\data\\cache"
    $env:HIP_VISIBLE_DEVICES="1"
    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v16_m138_seed_ensemble
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.common.data_cache import (
    configure_external_cache_environment,
    data_cache_root,
)
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from experiments.tier4.eval_v15_m107_dense import _solve_and_score, _verify_pixel_identity
from experiments.tier4.eval_v16_m108_dictionary import _verify_device
from experiments.tier4.eval_v16_m109_trunk import _load_corpus
from experiments.tier4.eval_v16_m113_learned import (
    _build_whitener_and_candidates,
    _random_dictionary,
)
from experiments.tier4.eval_v16_m115_lofi import _write_frozen_codes
from experiments.tier4.eval_v16_m128_diagnostics import _spectrum, _standardised_gram
from experiments.tier4.eval_v16_m136_margin_head import _fit_ridge, _test_blocks

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "m138_seed_ensemble.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m138_seed_ensemble"

T1_TOLERANCE = 0.002
KS_MARGIN = 0.005
MEMBER1_REFERENCE = 0.2152753623188406
ENSEMBLE_REFERENCE = 0.2248695652173913
CLASSES = 345


def _score_matrix(acc, mem_test: np.ndarray, block: int) -> np.ndarray:
    """(rows, 345) score matrix of a fitted ridge on the full test set."""
    standardise = acc.standardiser()
    weights = acc.solve_many([1.0])[1.0]
    out: list[np.ndarray] = []
    for start in range(0, len(mem_test), block):
        out.append(standardise(np.asarray(mem_test[start:start + block]))
                   @ weights[:-1] + weights[-1])
    return np.vstack(out)


def _ensemble_accuracy(accs, mem_tests, block: int, test_labels: np.ndarray,
                       test_domains: np.ndarray) -> dict[str, Any]:
    """Score-level ensemble: per-row L2-normalised member scores, averaged."""
    members = [_score_matrix(acc, mem, block) for acc, mem in zip(accs, mem_tests)]
    rows = members[0].shape[0]
    fused = np.zeros_like(members[0])
    for score in members:
        fused += score / np.linalg.norm(score, axis=1, keepdims=True)
    predictions = np.argmax(fused, axis=1).astype(np.int64)
    correct = predictions == test_labels
    per_domain = [float(correct[test_domains == d].mean())
                  if (test_domains == d).any() else 0.0 for d in range(6)]
    return {"accuracy": float(correct.mean()), "per_domain": per_domain}


def run_m138(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if "_smoke_note" in config and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")

    torch.set_num_threads(int(config["numerics"]["torch_threads"]))
    torch.manual_seed(int(config["numerics"]["seed"]))
    configure_external_cache_environment()
    _verify_device(torch)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    smoke = bool(config.get("_smoke_skip_gates", False))
    block = int(config["numerics"]["block"])
    throttle = float(config["numerics"]["encode_throttle_seconds"])
    started = time.time()

    print("loading corpus", flush=True)
    corpus, train_index, test_index = _load_corpus(config)
    size = int(config["corpus"]["image_size"])
    for split, idx in (("train", train_index), ("test", test_index)):
        _verify_pixel_identity(split, idx, corpus[f"{split}_images"], size,
                               int(config["corpus"]["pixel_identity_rows"]))

    atoms = int(config["sparse"]["member_atoms"])
    pool_grid = int(config["sparse"]["pool_grid"])
    seeds = [int(s) for s in config["sparse"]["member_seeds"]]
    n_full = int(config["ensemble"]["n"])
    train_cap = int(config.get("_smoke_train_cap", n_full))
    test_cap = int(config.get("_smoke_test_rows", 10 ** 9))

    print("building whitener + candidate pool (M108 exact)", flush=True)
    whitener, candidates = _build_whitener_and_candidates(config, corpus)

    cache = data_cache_root() / "v16" / "m138"
    cache.mkdir(parents=True, exist_ok=True)
    train_rows = np.arange(len(corpus["train_labels"]))
    test_rows = np.arange(len(corpus["test_labels"]))

    members: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"member seed={seed}: dictionary + encode", flush=True)
        dictionary = _random_dictionary(candidates, len(candidates), seed, atoms)
        mem_train = _write_frozen_codes(
            corpus, dictionary, whitener, pool_grid, device, train_rows,
            cache / f"f{atoms}_s{seed}_train.npy", split="train",
            throttle_seconds=throttle)
        mem_test = _write_frozen_codes(
            corpus, dictionary, whitener, pool_grid, device, test_rows,
            cache / f"f{atoms}_s{seed}_test.npy", split="test",
            throttle_seconds=throttle)
        acc = _fit_ridge(mem_train, corpus["train_labels"], train_cap, block)
        result = _solve_and_score(
            acc, [1.0],
            _test_blocks(mem_test[:test_cap], corpus["test_labels"][:test_cap],
                         corpus["test_domains"][:test_cap], block))
        members.append({"seed": seed, "acc": acc,
                        "mem_train": mem_train, "mem_test": mem_test,
                        "accuracy": result["accuracy_by_penalty"]["1.0"],
                        "per_domain": [
                            c / r if r else 0.0 for c, r in zip(
                                result["per_domain_correct"]["1.0"],
                                result["per_domain_rows"]["1.0"])]})
        print(f"  member seed={seed}: {members[-1]['accuracy']:.4f}", flush=True)

    evidence: dict[str, Any] = {
        "milestone": "M138",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "members": [{"seed": m["seed"], "accuracy": m["accuracy"],
                     "per_domain": m["per_domain"]} for m in members],
        "ensemble_rule": config["ensemble"]["rule"],
    }
    if not smoke:
        t1_delta = members[0]["accuracy"] - MEMBER1_REFERENCE
        evidence["t1"] = {"measured": members[0]["accuracy"],
                          "reference": MEMBER1_REFERENCE,
                          "delta": t1_delta,
                          "tolerance": T1_TOLERANCE}
        if abs(t1_delta) > T1_TOLERANCE:
            evidence["void"] = True
            evidence["void_reason"] = "member 1 t1 anchor reproduction failed"
            output_dir.mkdir(parents=True, exist_ok=True)
            write_canonical_json(output_dir / "evidence.json", evidence)
            build_artifact_index(output_dir)
            return evidence
        print(f"  t1 delta {t1_delta:+.6f} (<= {T1_TOLERANCE})", flush=True)

    ensemble = _ensemble_accuracy(
        [m["acc"] for m in members],
        [m["mem_test"][:test_cap] for m in members], block,
        corpus["test_labels"][:test_cap], corpus["test_domains"][:test_cap])
    evidence["ensemble"] = ensemble
    print(f"  ensemble: {ensemble['accuracy']:.4f} per_domain="
          f"{[round(p, 4) for p in ensemble['per_domain']]}", flush=True)

    # ---- joint standardised Gram + effective rank ---------------------------
    print("joint Gram + effective rank", flush=True)
    width = members[0]["acc"].width
    g11 = np.zeros((width, width), dtype=np.float64)
    g22 = np.zeros((width, width), dtype=np.float64)
    c12 = np.zeros((width, width), dtype=np.float64)
    colsum = np.zeros(2 * width, dtype=np.float64)
    sqsum = np.zeros(2 * width, dtype=np.float64)
    for start in range(0, train_cap, block):
        stop = min(start + block, train_cap)
        x1 = np.asarray(members[0]["mem_train"][start:stop], dtype=np.float64)
        x2 = np.asarray(members[1]["mem_train"][start:stop], dtype=np.float64)
        g11 += x1.T @ x1
        g22 += x2.T @ x2
        c12 += x1.T @ x2
        colsum[:width] += x1.sum(axis=0)
        colsum[width:] += x2.sum(axis=0)
        sqsum[:width] += np.square(x1).sum(axis=0)
        sqsum[width:] += np.square(x2).sum(axis=0)
    joint = np.block([[g11, c12], [c12.T, g22]])
    del g11, g22, c12
    gstd, _, _ = _standardised_gram(joint, colsum, sqsum, train_cap)
    vals, _ = _spectrum(gstd)
    keep = vals > max(float(vals.max()) * 1e-10, 1e-12)
    positive = vals[keep]
    trace = float(positive.sum())
    eff_rank = float(trace ** 2 / (positive ** 2).sum()) if trace > 0 else 0.0
    evidence["eff_rank"] = {
        "joint_width": int(2 * width),
        "n_eigenvalues": int(len(positive)),
        "effective_rank": eff_rank,
        "top1_share": float(positive[0] / trace) if trace > 0 else 0.0,
        "method": "participation ratio on the joint standardised Gram (M128 definition)",
    }
    print(f"  joint eff-rank {eff_rank:.2f}", flush=True)

    if not smoke:
        fired = ensemble["accuracy"] - ENSEMBLE_REFERENCE < KS_MARGIN
        evidence["gate"] = {
            "registered": config["gate"]["kill_switch_ensemble"],
            "ensemble_accuracy": ensemble["accuracy"],
            "reference": ENSEMBLE_REFERENCE,
            "gain": ensemble["accuracy"] - ENSEMBLE_REFERENCE,
            "required": KS_MARGIN,
            "fired": fired,
            "consequence": config["gate"]["kill_switch_ensemble"].split("Fired: ")[1]
            if fired else "the score-level ensemble beats the single-pool head "
                         "at matched MACs; the seed direction re-opens.",
        }
    evidence["runtime_seconds"] = round(time.time() - started, 2)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"wrote {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    run_m138(Path(args.config), Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
