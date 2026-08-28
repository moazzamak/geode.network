"""H26-4 - the alignment measurement on the sealed M228 hybrid cell.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md``
§8.40 (27 Aug 2026, before dispatch; premise amendment registered
after the first dispatch crashed on a wrong block-width premise).
The corrected sealed cell: ms-13244 + DINOv2-small-384, 409,832
train / 34,500 sealed test rows, penalty 1.0.

Arms:
- ms-only: the sealed anchor 0.24214492753623187 (LU reproduction).
- dino-only: report-only.
- raw-concat: the sealed 0.19434782608695653 (LU reproduction).
- cca-aligned: train-side CCA of the two blocks (k = 384
  components - the full DINOv2 block width), computed STREAMING
  from the memmap via ``cca_from_moments`` (float64 4096-row
  chunks, never a materialised design matrix); the canonical
  variates concatenated; the SEALED standardisation + LU solve at
  penalty 1.0; scored once.

Registered reading: the chain ``aligned > concatenated >
single-encoder`` is evaluated in order; the sealed concatenated
arm sits below the sealed single-encoder (0.1943 < 0.2421), so
the second link is already measured false at this scale and the
shared-space thesis is recorded unsupported whatever the aligned
arm reads. The aligned arm is still measured: beating BOTH sealed
arms would support the alignment mechanism specifically.
"""
from __future__ import annotations

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
from experiments.tier4.eval_v26_m299_hybrid_blocks import (
    DINO_TEST_DIR,
    DINO_TRAIN_DIR,
    _load_cached_dino,
)
from geode.core.alignment import cca_from_moments

REPO_ROOT = Path(__file__).resolve().parents[2]
M299_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v26"
               / "m299_hybrid_blocks.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v26"
                  / "m301_alignment_h26_4")

CLASSES = 345
FULL_TRAIN_ROWS = 409832
SEALED_TEST_ROWS = 34500
BLOCK = 4096
MS_ANCHOR = 0.24214492753623187
HYBRID_ANCHOR = 0.19434782608695653
ANCHOR_TOL = 1e-9
PENALTY = 1.0
CCA_RIDGE = 1e-8
DECORR_TOL = 0.05


def _score(weights: np.ndarray, features_fn: Any, labels: np.ndarray,
           n: int, standardiser: Any) -> float:
    hits = 0
    for start in range(0, n, BLOCK):
        stop = min(start + BLOCK, n)
        xs = standardiser(features_fn(start, stop)).astype(np.float64)
        scores = xs @ weights[:-1] + weights[-1]
        hits += int((np.argmax(scores, axis=1)
                     == labels[start:stop]).sum())
    return hits / n


def _fit_lu(features_fn: Any, labels: np.ndarray, n: int,
            width: int) -> tuple[np.ndarray, Any]:
    """The sealed RidgeAccumulator path over a streaming feature
    provider: float64 Gram accumulation in 4096-row chunks from
    fp32 blocks, fp32-rounded standardiser, LU solve at penalty
    1.0."""
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


def _stream_moments(ms_mem: np.ndarray, dino: np.ndarray,
                    n: int) -> dict[str, np.ndarray]:
    """Float64 sufficient statistics for CCA over the ms block and
    the DINOv2 block, in 4096-row chunks. Never materialises the
    design matrix."""
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


def main() -> int:
    started = time.time()
    config = json.loads(M299_CONFIG.read_text(encoding="utf-8"))

    configure_external_cache_environment()
    corpus, _train_index, test_index = _load_corpus(config)
    test_labels = corpus["test_labels"]
    root = data_cache_root()

    ms_cache = root / config["artifacts"]["cache_relpath"]
    ms_test_cache = root / config["artifacts"]["test_cache_relpath"]
    mem_train = np.load(ms_cache / config["artifacts"]["train_file"],
                        mmap_mode="r")
    mem_test = np.load(ms_test_cache / config["artifacts"]["test_file"],
                       mmap_mode="r")
    labels = np.load(root / config["artifacts"]["labels_file"])["labels"]

    dino_train = _load_cached_dino(
        DINO_TRAIN_DIR, "fulltrain",
        np.arange(FULL_TRAIN_ROWS, dtype=np.int64))
    dino_test = _load_cached_dino(DINO_TEST_DIR, "test",
                                  test_index.astype(np.int64))

    g1 = (len(mem_train) == FULL_TRAIN_ROWS
          and len(mem_test) == SEALED_TEST_ROWS
          and len(labels) == FULL_TRAIN_ROWS
          and len(dino_train) == FULL_TRAIN_ROWS
          and len(dino_test) == SEALED_TEST_ROWS)

    # ---- instrument identity: the two sealed LU anchors --------------
    ms_weights, ms_std = _fit_lu(
        lambda s, e: mem_train[s:e], labels, len(mem_train),
        mem_train.shape[1])
    ms_accuracy = _score(ms_weights,
                         lambda s, e: mem_test[s:e],
                         test_labels[:SEALED_TEST_ROWS],
                         SEALED_TEST_ROWS, ms_std)

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

    dino_weights, dino_std = _fit_lu(
        lambda s, e: dino_train[s:e], labels, len(dino_train),
        dino_train.shape[1])
    dino_accuracy = _score(dino_weights,
                           lambda s, e: dino_test[s:e],
                           test_labels[:SEALED_TEST_ROWS],
                           SEALED_TEST_ROWS, dino_std)

    g2 = (abs(ms_accuracy - MS_ANCHOR) <= ANCHOR_TOL
          and abs(concat_accuracy - HYBRID_ANCHOR) <= ANCHOR_TOL)

    # ---- the aligned arm: streaming CCA over the train blocks --------
    moments = _stream_moments(mem_train, dino_train,
                              FULL_TRAIN_ROWS)
    components = int(dino_train.shape[1])   # k = the DINOv2 width
    cca = cca_from_moments(moments["cov_ms"], moments["cov_dino"],
                           moments["cross"], components,
                           ridge=CCA_RIDGE)
    mean_ms = moments["mean_ms"]
    mean_dino = moments["mean_dino"]

    # decorrelation instrument on the TEST rows (the projections are
    # train-fit; the property is measured on held-out data)
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

    g4 = all(0.0 <= acc <= 1.0 for acc in
             (ms_accuracy, concat_accuracy, dino_accuracy,
              aligned_accuracy))

    # ---- the registered reading -------------------------------------
    chain = {
        "single_encoder_ms": ms_accuracy,
        "dino_only": dino_accuracy,
        "concatenated": concat_accuracy,
        "aligned_cca": aligned_accuracy,
    }
    second_link = bool(concat_accuracy > ms_accuracy)
    first_link = bool(aligned_accuracy > concat_accuracy)
    full_chain = first_link and second_link
    if not second_link:
        reading = ("the second link of the gate chain is measured "
                   "false at sealed scale (concatenated below the "
                   "single-encoder anchor): the shared-space thesis "
                   "is recorded UNSUPPORTED at this scale, "
                   "regardless of the aligned arm")
    elif full_chain:
        reading = ("aligned > concatenated > single-encoder: the "
                   "chain holds")
    else:
        reading = ("the aligned arm does not beat the concatenated "
                   "arm: the chain does not hold")
    aligned_beats_both = bool(
        aligned_accuracy > ms_accuracy
        and aligned_accuracy > concat_accuracy)

    gates = {
        "g1_premise": {"ok": bool(g1),
                       "train_rows": len(mem_train),
                       "test_rows": len(mem_test),
                       "ms_width": int(mem_train.shape[1]),
                       "dino_width": int(dino_train.shape[1]),
                       "dino_train_rows": len(dino_train),
                       "dino_test_rows": len(dino_test)},
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
        "milestone": "M301",
        "cell": ("H26-4: streaming CCA alignment before fusion on "
                 "the sealed M228 hybrid cell (ms-13244 + "
                 "dino-s-384; the corrected premise per the §8.40 "
                 "amendment)"),
        "arms": chain,
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
            "alignment": ("cca_from_moments, streaming float64 "
                          "chunks, eig-based inverse square root "
                          "(alignment.py)")}),
        "scope": ("full 409,832-row train, sealed 34,500-row test, "
                  "cached ms-13244 + DINOv2-small features (no "
                  "re-extraction); the design matrix is never "
                  "materialised"),
        "runtime_seconds": round(time.time() - started, 2),
    }
    DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)
    write_canonical_json(DEFAULT_OUTPUT / "evidence.json", evidence)
    build_artifact_index(DEFAULT_OUTPUT)
    print(json.dumps({"gates_ok": bool(gates_ok),
                      "arms": chain, "reading": reading}))
    return 0 if gates_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
