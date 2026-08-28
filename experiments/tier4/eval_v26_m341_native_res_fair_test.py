"""M341 - the native-resolution fair test of the alignment mechanism
(H26-4 as it should have been run).

Registered in ``analysis/SCIENCE_LAYER_PLAN_2026-08-28.md`` (M341
REGISTRATION AMENDMENT, 28 Aug 2026, before the build). The M301
H26-4 cell ran CCA on the M228 hybrid block where the DINOv2 side
carried zero class signal (dino-only 0.00435 ~ chance) - alignment
had nothing to preserve. The clean block already exists in cache
(M230 native-224 DINOv2-small, dino-only 0.4845), and the M230
readings make the second chain link already-measured on a clean
cell (hybrid 0.5479 > ms-only 0.2421). M341 measures the FIRST
link: aligned vs concatenated, on the clean cell.

Arms (all the sealed ridge recipe - RidgeAccumulator, penalty 1.0,
LU solve, the M301 convention):
- ms_only: the sealed anchor 0.24214492753623187 (reproduction).
- dino_only: report-only (the clean block's own signal).
- concat: ms + native-dino raw concatenation; the M230 hybrid anchor
  0.5478550724637681 at penalty 1.0 is the instrument-identity
  reproduction.
- cca_aligned: k = 384 (the full dino block width), streaming
  ``cca_from_moments`` (ridge 1e-8), canonical variates concatenated,
  sealed standardisation + LU solve at penalty 1.0, scored once.

Gates: g1 premise (row counts, cache presence); g2 both anchor
reproductions at 1e-9; g3 the CCA instrument (components = 384,
nonnegative correlations, test-side decorrelation < 0.05); g4
accuracies valid.

Registered reading: the chain ``aligned > concatenated >
single-encoder`` is evaluated in order; the second link is already
true on this cell (0.5479 > 0.2421), so the thesis question reduces
to the first link - aligned > concatenated. A pass supports the
bridge mechanism specifically; a fail records that alignment loses
to raw concatenation even on a clean cell, and the federation
thesis carries bridges as optional (measured, priced) rather than
load-bearing. Both outcomes publishable.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

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
from geode.core.alignment import cca_from_moments

REPO_ROOT = Path(__file__).resolve().parents[2]
M299_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v26"
               / "m299_hybrid_blocks.json")
DINO_TRAIN = (REPO_ROOT / "logs" / "results" / "v25"
              / "m230_native_res_dinov2" / "features"
              / "native224_train_dino.npy")
DINO_TEST = (REPO_ROOT / "logs" / "results" / "v25"
             / "m230_native_res_dinov2" / "features"
             / "native224_test_dino.npy")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v26"
                  / "m341_native_res_fair_test")

CLASSES = 345
FULL_TRAIN_ROWS = 409832
SEALED_TEST_ROWS = 34500
BLOCK = 4096
MS_ANCHOR = 0.24214492753623187
HYBRID_ANCHOR = 0.5478550724637681     # M230 hybrid at penalty 1.0
ANCHOR_TOL = 1e-9
PENALTY = 1.0
CCA_RIDGE = 1e-8
DECORR_TOL = 0.05


def _fit_lu(features_fn: Callable[[int, int], np.ndarray],
            labels: np.ndarray, n: int, width: int
            ) -> tuple[np.ndarray, Any]:
    """The sealed RidgeAccumulator path over a streaming feature
    provider: float64 Gram accumulation in 4096-row chunks from fp32
    blocks, fp32-rounded standardiser, LU solve at penalty 1.0 (the
    M301 convention)."""
    acc = RidgeAccumulator(width, CLASSES)
    for start in range(0, n, BLOCK):
        stop = min(start + BLOCK, n)
        block = np.asarray(features_fn(start, stop), dtype=np.float64)
        acc.add(block.astype(np.float32), labels[start:stop])
    standardiser = acc.standardiser()
    centred, cross, intercept = acc._standardised_system()
    system = centred.copy()
    system.flat[:: system.shape[0] + 1] += PENALTY
    weights = np.vstack([np.linalg.solve(system, cross),
                         intercept[None, :]])
    return weights, standardiser


def _score(weights: np.ndarray, features_fn: Callable[[int, int],
                                                      np.ndarray],
           labels: np.ndarray, n: int, standardiser: Any) -> float:
    hits = 0
    for start in range(0, n, BLOCK):
        stop = min(start + BLOCK, n)
        xs = standardiser(features_fn(start, stop)).astype(np.float64)
        scores = xs @ weights[:-1] + weights[-1]
        hits += int((np.argmax(scores, axis=1)
                     == labels[start:stop]).sum())
    return hits / n


def _stream_moments(ms_mem: np.ndarray, dino: np.ndarray,
                    n: int) -> dict[str, np.ndarray]:
    """Float64 sufficient statistics for CCA over the ms block and
    the native-res DINOv2 block, in 4096-row chunks. Never
    materialises the design matrix (the M301 pattern)."""
    w_ms = ms_mem.shape[1]
    w_dino = dino.shape[1]
    gram_ms = np.zeros((w_ms, w_ms), dtype=np.float64)
    gram_dino = np.zeros((w_dino, w_dino), dtype=np.float64)
    cross = np.zeros((w_ms, w_dino), dtype=np.float64)
    sum_ms = np.zeros(w_ms, dtype=np.float64)
    sum_dino = np.zeros(w_dino, dtype=np.float64)
    for start in range(0, n, BLOCK):
        stop = min(start + BLOCK, n)
        b_ms = np.asarray(ms_mem[start:stop], dtype=np.float64)
        b_dino = np.asarray(dino[start:stop], dtype=np.float64)
        gram_ms += b_ms.T @ b_ms
        gram_dino += b_dino.T @ b_dino
        cross += b_ms.T @ b_dino
        sum_ms += b_ms.sum(axis=0)
        sum_dino += b_dino.sum(axis=0)
    mean_ms = sum_ms / n
    mean_dino = sum_dino / n
    cov_ms = gram_ms - np.outer(sum_ms, mean_ms)
    cov_dino = gram_dino - np.outer(sum_dino, mean_dino)
    cross_c = cross - np.outer(sum_ms, mean_dino)
    return {"cov_ms": cov_ms, "cov_dino": cov_dino,
            "cross": cross_c, "mean_ms": mean_ms,
            "mean_dino": mean_dino}


def run_m341(output_dir: Path) -> dict[str, Any]:
    started = time.time()
    config = json.loads(M299_CONFIG.read_text(encoding="utf-8"))

    configure_external_cache_environment()
    corpus, train_index, test_index = _load_corpus(config)
    test_labels = corpus["test_labels"]
    root = data_cache_root()

    # the schedule permutation (the M230/M233 convention): part1
    # subsample + ext600 + rest, reconstructing the labels file
    from experiments.tier4.eval_v15_m104_experts import _load_domainnet
    from experiments.tier4.eval_v16_m140_data_extension import (
        _extension_indices,
    )
    from experiments.tier4.eval_v16_m141_data_full import (
        _rest_extension_indices,
    )
    raw = _load_domainnet(32)
    ext600_indices, _ = _extension_indices(raw["train_labels"],
                                           train_index, 600, CLASSES)
    rest_indices = _rest_extension_indices(raw["train_labels"],
                                           train_index, CLASSES,
                                           per_class_take=200)
    perm = np.concatenate([train_index, ext600_indices, rest_indices])
    labels = np.load(root / config["artifacts"]["labels_file"])["labels"]
    g0_schedule = (len(perm) == FULL_TRAIN_ROWS
                   and np.array_equal(raw["train_labels"][perm], labels))
    del raw
    import gc
    gc.collect()

    ms_cache = root / config["artifacts"]["cache_relpath"]
    ms_test_cache = root / config["artifacts"]["test_cache_relpath"]
    mem_train = np.load(ms_cache / config["artifacts"]["train_file"],
                        mmap_mode="r")
    mem_test = np.load(ms_test_cache / config["artifacts"]["test_file"],
                       mmap_mode="r")

    # the clean block: M230 native-224 DINOv2-small. The SAVED train
    # file is in RAW parquet file order (the M230 harness applies perm
    # in-memory after loading); map it to the schedule order here. The
    # saved test file is raw file order -> test_index (the M230
    # convention, identical to M236/M237).
    dino_train_raw = np.asarray(np.load(DINO_TRAIN, mmap_mode="r"))
    dino_train = np.ascontiguousarray(dino_train_raw[perm])
    del dino_train_raw
    gc.collect()
    dino_test_raw = np.asarray(np.load(DINO_TEST, mmap_mode="r"))
    dino_test = np.ascontiguousarray(dino_test_raw[test_index])
    del dino_test_raw
    gc.collect()

    g1 = (g0_schedule
          and len(mem_train) == FULL_TRAIN_ROWS
          and len(mem_test) == SEALED_TEST_ROWS
          and len(labels) == FULL_TRAIN_ROWS
          and len(dino_train) == FULL_TRAIN_ROWS
          and len(dino_test) == SEALED_TEST_ROWS
          and dino_train.shape[1] == 384)
    print(f"g1 premise: {g1} (ms {mem_train.shape}, "
          f"dino {dino_train.shape}, test {len(dino_test)})",
          flush=True)

    # ---- instrument identity: the two sealed anchors -----------------
    ms_weights, ms_std = _fit_lu(
        lambda s, e: mem_train[s:e], labels, len(mem_train),
        mem_train.shape[1])
    ms_accuracy = _score(ms_weights,
                         lambda s, e: mem_test[s:e],
                         test_labels[:SEALED_TEST_ROWS],
                         SEALED_TEST_ROWS, ms_std)
    print(f"ms_only: {ms_accuracy:.17f}", flush=True)

    concat_width = mem_train.shape[1] + dino_train.shape[1]
    concat_weights, concat_std = _fit_lu(
        lambda s, e: np.concatenate(
            [mem_train[s:e], dino_train[s:e]], axis=1),
        labels, len(mem_train), concat_width)
    concat_accuracy = _score(
        concat_weights,
        lambda s, e: np.concatenate(
            [mem_test[s:e], dino_test[s:e]], axis=1),
        test_labels[:SEALED_TEST_ROWS], SEALED_TEST_ROWS, concat_std)
    print(f"concat: {concat_accuracy:.17f}", flush=True)

    dino_weights, dino_std = _fit_lu(
        lambda s, e: dino_train[s:e], labels, len(dino_train),
        dino_train.shape[1])
    dino_accuracy = _score(dino_weights,
                           lambda s, e: dino_test[s:e],
                           test_labels[:SEALED_TEST_ROWS],
                           SEALED_TEST_ROWS, dino_std)
    print(f"dino_only: {dino_accuracy:.17f}", flush=True)

    g2 = (abs(ms_accuracy - MS_ANCHOR) <= ANCHOR_TOL
          and abs(concat_accuracy - HYBRID_ANCHOR) <= ANCHOR_TOL)
    print(f"g2 anchors: {g2}", flush=True)

    # ---- the aligned arm: streaming CCA over the clean blocks --------
    moments = _stream_moments(mem_train, dino_train, FULL_TRAIN_ROWS)
    components = int(dino_train.shape[1])   # k = the dino block width
    cca = cca_from_moments(moments["cov_ms"], moments["cov_dino"],
                           moments["cross"], components,
                           ridge=CCA_RIDGE)
    mean_ms = moments["mean_ms"]
    mean_dino = moments["mean_dino"]

    # decorrelation instrument on the TEST rows (train-fit
    # projections; the property is measured on held-out data)
    za_test = np.zeros((SEALED_TEST_ROWS, components), dtype=np.float64)
    zb_test = np.zeros((SEALED_TEST_ROWS, components), dtype=np.float64)
    for start in range(0, SEALED_TEST_ROWS, BLOCK):
        stop = min(start + BLOCK, SEALED_TEST_ROWS)
        za_test[start:stop] = (np.asarray(mem_test[start:stop],
                                          dtype=np.float64)
                               - mean_ms) @ cca.projection_a
        zb_test[start:stop] = (np.asarray(dino_test[start:stop],
                                          dtype=np.float64)
                               - mean_dino) @ cca.projection_b
    off_a = float(np.abs(np.triu(np.corrcoef(za_test.T), k=1)).max())
    off_b = float(np.abs(np.triu(np.corrcoef(zb_test.T), k=1)).max())
    cca_report = dict(cca.report)
    cca_report["decorrelated"] = bool(max(off_a, off_b) < DECORR_TOL)
    cca_report["max_within_offdiag"] = float(max(off_a, off_b))
    g3 = (cca_report["components"] == components
          and cca_report["all_nonnegative"]
          and cca_report["decorrelated"])
    print(f"g3 cca instrument: {g3} "
          f"(max_offdiag {max(off_a, off_b):.4f})", flush=True)

    aligned_width = 2 * components
    aligned_weights, aligned_std = _fit_lu(
        lambda s, e: np.concatenate(
            [(np.asarray(mem_train[s:e], dtype=np.float64) - mean_ms)
             @ cca.projection_a,
             (np.asarray(dino_train[s:e], dtype=np.float64) - mean_dino)
             @ cca.projection_b], axis=1),
        labels, len(mem_train), aligned_width)
    aligned_accuracy = _score(
        aligned_weights,
        lambda s, e: np.concatenate(
            [(np.asarray(mem_test[s:e], dtype=np.float64) - mean_ms)
             @ cca.projection_a,
             (np.asarray(dino_test[s:e], dtype=np.float64) - mean_dino)
             @ cca.projection_b], axis=1),
        test_labels[:SEALED_TEST_ROWS], SEALED_TEST_ROWS, aligned_std)
    print(f"cca_aligned: {aligned_accuracy:.17f}", flush=True)

    g4 = all(0.0 <= acc <= 1.0 for acc in
             (ms_accuracy, dino_accuracy, concat_accuracy,
              aligned_accuracy))

    # ---- the registered reading --------------------------------------
    arms = {
        "single_encoder_ms": ms_accuracy,
        "dino_only": dino_accuracy,
        "concatenated": concat_accuracy,
        "aligned_cca": aligned_accuracy,
    }
    second_link = bool(concat_accuracy > ms_accuracy)
    first_link = bool(aligned_accuracy > concat_accuracy)
    full_chain = first_link and second_link
    if full_chain:
        reading = ("the full chain holds on the clean cell: aligned > "
                   "concatenated > single-encoder - the bridge "
                   "mechanism is supported at this scale")
    elif second_link and not first_link:
        reading = ("the second link holds (concatenated > "
                   "single-encoder, carried by the M230 anchor) but "
                   "the first link fails: alignment loses to raw "
                   "concatenation even on a clean cell - the "
                   "federation thesis carries bridges as optional "
                   "(measured, priced) rather than load-bearing")
    else:
        reading = ("the second link fails on this cell - unexpected "
                   "against the M230 anchor; treat as an instrument "
                   "defect and VOID")
    aligned_beats_both = bool(
        aligned_accuracy > ms_accuracy
        and aligned_accuracy > concat_accuracy)

    gates = {
        "g1_premise": {"ok": bool(g1),
                       "schedule_alignment": bool(g0_schedule),
                       "train_rows": len(mem_train),
                       "test_rows": len(mem_test),
                       "ms_width": int(mem_train.shape[1]),
                       "dino_width": int(dino_train.shape[1]),
                       "dino_source": ("M230 native-224 DINOv2-small, "
                                       "raw file order mapped to "
                                       "schedule order via perm")},
        "g2_anchor_reproductions": {
            "ok": bool(g2), "ms_measured": ms_accuracy,
            "ms_sealed": MS_ANCHOR,
            "concat_measured": concat_accuracy,
            "concat_sealed": HYBRID_ANCHOR, "tolerance": ANCHOR_TOL},
        "g3_cca_instrument": {"ok": bool(g3), **cca_report},
        "g4_accuracies_valid": {"ok": bool(g4)},
    }
    gates_ok = all(g["ok"] for g in gates.values())

    evidence: dict[str, Any] = {
        "milestone": "M341",
        "cell": ("H26-4 fair test: streaming CCA alignment before "
                 "fusion on the CLEAN native-resolution cell "
                 "(ms-13244 + native-224 dino-s-384)"),
        "arms": arms,
        "chain_links": {"first_link_aligned_gt_concat": first_link,
                        "second_link_concat_gt_single": second_link,
                        "full_chain": bool(full_chain)},
        "aligned_beats_both": aligned_beats_both,
        "reading": reading,
        "gates": gates,
        "gates_ok": bool(gates_ok),
        "void": (not gates_ok),
        "configuration_hash": payload_hash({
            "penalty": PENALTY, "cca_components": components,
            "cca_ridge": CCA_RIDGE,
            "dino_source": str(DINO_TRAIN),
            "alignment": ("cca_from_moments, streaming float64 "
                          "chunks, eig-based inverse square root "
                          "(alignment.py)")}),
        "scope": ("full 409,832-row train, sealed 34,500-row test; "
                  "cached ms-13244 + M230 native-224 DINOv2-small "
                  "features (no re-extraction); the design matrix is "
                  "never materialised"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"gates_ok": bool(gates_ok), "arms": arms,
                      "chain_links": evidence["chain_links"],
                      "reading": reading}, indent=1))
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m341(args.output)


if __name__ == "__main__":
    main()
