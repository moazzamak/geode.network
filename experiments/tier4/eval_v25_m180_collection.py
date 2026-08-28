"""M180 data collection — the complete 3-arm coalition game from cached
codes (spm / ms / pool, raw, penalty 1.0).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` section 6
(18 Aug 2026) before building. No re-encoding: every code family's
full-train and full-test memmaps already exist. The four sealed anchors
pin the machinery before any missing coalition value is read:

    V(spm)     = 0.2604927536231884   (M151 anchor, raw full)
    V(ms)      = 0.24214492753623187  (M151 anchor, raw full)
    V(pool)    = 0.22753623188405797  (M142 C2 gate pool_penalty1_full)
    V(spm,ms)  = 0.2831304347826087   (M151 raw_lambda1.0)

Missing and measured here (same read): V(spm,pool), V(ms,pool), V(all).
Column-concatenated ridge, penalty 1.0, 34,500-row sealed test.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m180_collection.json")
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v25" / "m180_collection"

CLASSES = 345
TOL = 1e-9
T1_TOL = 0.002   # registered environment tolerance for the wide concats
PENALTY = 1.0
BLOCK = 4096     # stream chunk (must match the sealed M142/M151 path)
RESID_TOL = 1e-10        # exact closed-form residual (blocks path)
STREAM_RESID_TOL = 1e-5  # data-streamed residual (V_all, convention gap)


def _build_blocks(parts_train: list[np.ndarray],
                  labels: np.ndarray) -> tuple[list[list[np.ndarray]],
                                               list[Any], list[np.ndarray],
                                               np.ndarray, int,
                                               list[np.ndarray]]:
    """ONE pass over the raw train parts (repair 5): all (i, j) block
    Grams, column sums, crosses and class counts; then the M142/M151
    closed-form standardisation IN PLACE (fp32-rounded centres/scales,
    penalty 1.0 on the diagonal blocks). ``blocks[i][j]`` (i <= j) is
    the standardised Gram block afterwards."""
    from experiments.tier4.eval_v15_m104_experts import Standardiser

    k = len(parts_train)
    widths = [p.shape[1] for p in parts_train]
    grams = [[None] * k for _ in range(k)]
    colsums = [np.zeros(w, dtype=np.float64) for w in widths]
    crosses = [np.zeros((w, CLASSES), dtype=np.float64) for w in widths]
    class_count = np.zeros(CLASSES, dtype=np.float64)
    rows = 0
    n = len(labels)
    for start in range(0, n, BLOCK):
        stop = min(start + BLOCK, n)
        blocks = [np.asarray(part[start:stop]).astype(np.float64)
                  for part in parts_train]
        targets = np.zeros((stop - start, CLASSES), dtype=np.float64)
        targets[np.arange(stop - start), labels[start:stop]] = 1.0
        for i in range(k):
            for j in range(i, k):
                prod = blocks[i].T @ blocks[j]
                if grams[i][j] is None:
                    grams[i][j] = prod
                else:
                    grams[i][j] += prod
            colsums[i] += blocks[i].sum(axis=0)
            crosses[i] += blocks[i].T @ targets
        class_count += targets.sum(axis=0)
        rows += stop - start
    stds = []
    for i in range(k):
        centre = colsums[i] / rows
        variance = grams[i][i].diagonal() / rows - np.square(centre)
        scale = np.sqrt(np.maximum(variance, 0.0)) + 1e-8
        stds.append(Standardiser(centre.astype(np.float32),
                                 scale.astype(np.float32)))
    intercept = class_count / rows
    # ---- the standardised ridge block system, in place ---------------------
    for i in range(k):
        centre64 = stds[i].centre.astype(np.float64)
        inv64 = 1.0 / stds[i].scale.astype(np.float64)
        for j in range(i, k):
            centre_j = stds[j].centre.astype(np.float64)
            inv_j = 1.0 / stds[j].scale.astype(np.float64)
            G = grams[i][j]
            G -= np.outer(colsums[i], centre_j)
            G *= inv64[:, None]
            G *= inv_j[None, :]
            if i == j:
                G.flat[:: widths[i] + 1] += PENALTY
        crosses[i] = (crosses[i]
                      - np.outer(centre64, class_count)) * inv64[:, None]
    return grams, stds, crosses, intercept, rows, colsums


def _norm_g_inf(blocks: list[list[np.ndarray]], selected: list[int]
                ) -> float:
    """Infinity norm of the coalition's assembled system, from the
    cached blocks (must be called while the blocks still exist)."""
    widths = [blocks[p][p].shape[0] for p in selected]
    row_max = np.zeros(sum(widths), dtype=np.float64)
    offsets = np.concatenate([[0], np.cumsum(widths)])
    for a, p in enumerate(selected):
        total = np.zeros(widths[a], dtype=np.float64)
        for b, q in enumerate(selected[a:], start=a):
            total += np.abs(blocks[p][q]).sum(axis=1)
        row_max[offsets[a]:offsets[a + 1]] = total
    return float(row_max.max())


def _assemble_and_solve(blocks: list[list[np.ndarray]],
                        selected: list[int], crosses: list[np.ndarray],
                        intercept: np.ndarray, stds: list[Any],
                        colsums: list[np.ndarray],
                        free_blocks: bool = False) -> np.ndarray:
    """Assemble the coalition's standardised system into ONE Fortran-
    order array and factor IN PLACE (dgetrf + dgetrs == LAPACK dgesv,
    bit-exact with the sealed solve path). The lower-left block of each
    off-diagonal pair is built with its OWN centring convention
    (sealed: centred[q, p] = gram − colsum_q·centre_p, NOT the
    transpose of the (p, q) block — the convention is asymmetric,
    measured 1.6e-4 relative; using the transpose cost 0.66 points of
    holdout accuracy). With ``free_blocks`` each block (and cross) is
    released as soon as it is copied — the V(all) memory schedule
    (registered repair 5). Returns weights [w; b]."""
    from scipy import linalg as scipy_linalg

    widths = [blocks[p][p].shape[0] for p in selected]
    total = sum(widths)
    arr = np.empty((total, total), dtype=np.float64, order="F")
    offsets = np.concatenate([[0], np.cumsum(widths)])
    for a, p in enumerate(selected):
        r0, r1 = int(offsets[a]), int(offsets[a + 1])
        for b, q in enumerate(selected[a:], start=a):
            c0, c1 = int(offsets[b]), int(offsets[b + 1])
            arr[r0:r1, c0:c1] = blocks[p][q]
            if p != q:
                # sealed lower-left convention, chunk-wise (registered
                # repair 5): G_qp[c, r] = G_pq[r, c] +
                #   (colsum_p[r]·centre_q[c] − colsum_q[c]·centre_p[r])
                #   · inv_q[c] · inv_p[r]
                inv_p = 1.0 / stds[p].scale.astype(np.float64)
                inv_q = 1.0 / stds[q].scale.astype(np.float64)
                centre_p = stds[p].centre.astype(np.float64)
                centre_q = stds[q].centre.astype(np.float64)
                for cc0 in range(0, widths[b], 2048):
                    cc1 = min(cc0 + 2048, widths[b])
                    delta = (np.outer(colsums[p], centre_q[cc0:cc1])
                             - np.outer(centre_p, colsums[q][cc0:cc1]))
                    delta *= inv_p[:, None]
                    delta *= inv_q[cc0:cc1][None, :]
                    block_qp = blocks[p][q][:, cc0:cc1].T + delta.T
                    arr[c0 + cc0:c0 + cc1, r0:r1] = block_qp
                    del block_qp, delta
            if free_blocks:
                blocks[p][q] = None
    cross = np.vstack([crosses[p] for p in selected])
    lu_piv = scipy_linalg.lu_factor(arr, overwrite_a=True,
                                    check_finite=False)
    weights = scipy_linalg.lu_solve(lu_piv, cross)
    del arr, lu_piv
    return np.vstack([weights, intercept[None, :]])


def _residual_certificate(parts_train: list[np.ndarray], labels: np.ndarray,
                          stds: list[Any], selected: list[int],
                          weights: np.ndarray, crosses: list[np.ndarray],
                          norm_g_inf: float,
                          blocks: list[list[np.ndarray]] | None = None,
                          colsums: list[np.ndarray] | None = None
                          ) -> dict[str, Any]:
    """Backward error of the fitted weights.

    With ``blocks``: the EXACT closed-form residual r = C − G w on the
    stored standardised system (what the assembled LU actually solved),
    lower-left blocks in the sealed convention, tolerance 1e-10.
    Without (V_all, whose blocks die during assembly):
    r = C − X~^T (X~ w) − w re-streamed from the data, which carries the
    registered fp32-centre convention difference (~1e-7), tolerance
    1e-5. Both paths catch solver breakage (the void block-Schur path
    measured 2.9e-4)."""
    tol = RESID_TOL if blocks is not None else STREAM_RESID_TOL
    if blocks is not None:
        parts_r = []
        split = [len(crosses[p]) for p in selected]
        w_split = np.split(weights[:-1], np.cumsum(split)[:-1])
        for a, p in enumerate(selected):
            r = crosses[p].copy()
            for b, q in enumerate(selected):
                if p <= q:
                    r -= blocks[p][q] @ w_split[b]
                else:
                    # reconstruct centred[p, q] from the stored (q, p)
                    # block: the centring convention is asymmetric.
                    delta = ((np.outer(colsums[q],
                                       stds[p].centre.astype(np.float64)).T
                              - np.outer(colsums[p],
                                         stds[q].centre
                                         .astype(np.float64)))
                             * (1.0 / stds[p].scale.astype(np.float64))
                             [:, None]
                             * (1.0 / stds[q].scale.astype(np.float64))
                             [None, :])
                    r -= (blocks[q][p].T + delta) @ w_split[b]
                    del delta
            parts_r.append(r)
        r = np.concatenate(parts_r)
    else:
        cross = np.vstack([crosses[p] for p in selected])
        v = np.zeros_like(weights[:-1])
        n = len(labels)
        for start in range(0, n, BLOCK):
            stop = min(start + BLOCK, n)
            parts = [np.asarray(part[start:stop]) for part in parts_train]
            xs = np.concatenate(
                [stds[p](parts[p]).astype(np.float64)
                 for p in selected], axis=1)
            v += xs.T @ (xs @ weights[:-1])
        r = cross - v - weights[:-1]
    cross_flat = np.vstack([crosses[p] for p in selected])
    denom = (norm_g_inf * float(np.max(np.abs(weights[:-1])))
             + float(np.max(np.abs(cross_flat))))
    backward = float(np.max(np.abs(r))) / max(denom, 1e-300)
    return {"backward_error": backward, "tolerance": tol,
            "passed": backward <= tol}


def _score_coalition(parts_test: list[np.ndarray], test_labels: np.ndarray,
                     stds: list[Any], selected: list[int],
                     weights: np.ndarray) -> float:
    hits = 0
    n_test = len(test_labels)
    for start in range(0, n_test, BLOCK):
        stop = min(start + BLOCK, n_test)
        parts = [np.asarray(part[start:stop]) for part in parts_test]
        xs = np.concatenate([stds[p](parts[p]) for p in selected], axis=1)
        scores = xs.astype(np.float64) @ weights[:-1] + weights[-1]
        hits += int((np.argmax(scores, axis=1)
                     == test_labels[start:stop]).sum())
    return hits / n_test


def _fit_coalition(blocks: list[list[np.ndarray]], stds: list[Any],
                   crosses: list[np.ndarray], intercept: np.ndarray,
                   colsums: list[np.ndarray], selected: list[int],
                   parts_train: list[np.ndarray], labels: np.ndarray,
                   parts_test: list[np.ndarray], test_labels: np.ndarray,
                   norm_g: float | None = None,
                   free_blocks: bool = False) -> dict[str, Any]:
    """Repair-5 coalition fit: assemble -> in-place LU -> certify ->
    score. The residual certificate is the wide-fit validity gate.
    ``free_blocks`` (V_all only, last coalition) releases each block as
    it is copied; ``norm_g`` is precomputed in that case because the
    blocks die during assembly."""
    if norm_g is None:
        norm_g = _norm_g_inf(blocks, selected)
    weights = _assemble_and_solve(blocks, selected, crosses, intercept,
                                  stds, colsums, free_blocks=free_blocks)
    certificate = _residual_certificate(
        parts_train, labels, stds, selected, weights, crosses, norm_g,
        blocks=None if free_blocks else blocks, colsums=colsums)
    accuracy = _score_coalition(parts_test, test_labels, stds, selected,
                                weights)
    return {"accuracy": accuracy, "residual": certificate,
            "weights_shape": [int(w) for w in weights.shape]}


def _equivalence_gate(parts_train: list[np.ndarray], labels: np.ndarray,
                      cols: int, rows_gate: int) -> dict[str, Any]:
    """The registered gate (repair 5): on a two-block capped concat of
    REAL data,
    (a) the assemble+LU path must match np.linalg.solve ON THE SAME
        standardised matrix (weights rel <= 1e-9; measured ~1e-16 — the
        two LAPACK routes are the same routine), and
    (b) the convention end-to-end must agree with the sealed
        RidgeAccumulator reference at decision level on a holdout slice
        (accuracy diff <= 1e-3; last-ulp dgemm-shape differences can
        flip fp32 centres at ~1e-6 in weights, which the sealed anchors
        then pin on the real run)."""
    cols_per = cols // 2
    capped = [np.asarray(part[:rows_gate, :cols_per])
              for part in parts_train]
    blocks, stds, crosses, intercept, _rows, colsums = _build_blocks(
        capped, labels[:rows_gate])
    # (a) same-matrix solver equivalence
    weights = _assemble_and_solve(blocks, [0, 1], crosses, intercept,
                                  stds, colsums)
    arr = np.empty((cols, cols), dtype=np.float64, order="F")
    arr[:cols_per, :cols_per] = blocks[0][0]
    arr[:cols_per, cols_per:] = blocks[0][1]
    # sealed lower-left convention for the reference matrix
    inv_0 = 1.0 / stds[0].scale.astype(np.float64)
    inv_1 = 1.0 / stds[1].scale.astype(np.float64)
    centre_0 = stds[0].centre.astype(np.float64)
    centre_1 = stds[1].centre.astype(np.float64)
    for cc0 in range(0, cols_per, 2048):
        cc1 = min(cc0 + 2048, cols_per)
        delta = (np.outer(colsums[0], centre_1[cc0:cc1])
                 - np.outer(centre_0, colsums[1][cc0:cc1]))
        delta *= inv_0[:, None]
        delta *= inv_1[cc0:cc1][None, :]
        arr[cols_per + cc0:cols_per + cc1, :cols_per] = \
            blocks[0][1][:, cc0:cc1].T + delta.T
        del delta
    arr[cols_per:, cols_per:] = blocks[1][1]
    direct = np.vstack([np.linalg.solve(arr, np.vstack(crosses)),
                        intercept[None, :]])
    del arr
    rel = float(np.max(np.abs(direct - weights))
                / max(float(np.max(np.abs(direct))), 1e-12))
    # (b) decision-level convention check vs the sealed reference
    acc = RidgeAccumulator(cols, CLASSES)
    for start in range(0, rows_gate, BLOCK):
        stop = min(start + BLOCK, rows_gate)
        xs = np.concatenate(
            [np.asarray(part[start:stop, :cols_per])
             for part in parts_train], axis=1)
        acc.add(xs, labels[start:stop])
    ref_w = acc.solve(PENALTY)
    ref_std = acc.standardiser()
    hold = min(rows_gate // 4, len(labels) - rows_gate)
    if hold > 0:
        hold_labels = labels[rows_gate:rows_gate + hold]
        hold_xs = np.concatenate(
            [np.asarray(part[rows_gate:rows_gate + hold, :cols_per])
             for part in parts_train], axis=1)
        # the build pipeline on the holdout
        xs_build = np.concatenate(
            [stds[0](hold_xs[:, :cols_per]),
             stds[1](hold_xs[:, cols_per:])], axis=1)
        scores_build = (xs_build.astype(np.float64) @ weights[:-1]
                        + weights[-1])
        # the sealed pipeline on the holdout
        scores_ref = (ref_std(hold_xs).astype(np.float64) @ ref_w[:-1]
                      + ref_w[-1])
        acc_build = float((np.argmax(scores_build, axis=1)
                           == hold_labels).mean())
        acc_ref = float((np.argmax(scores_ref, axis=1)
                         == hold_labels).mean())
        holdout = {"acc_build": acc_build, "acc_ref": acc_ref,
                   "delta": abs(acc_build - acc_ref), "hold_rows": hold,
                   "tolerance": 1e-3}
    else:
        holdout = {"acc_build": None, "acc_ref": None, "delta": 0.0,
                   "hold_rows": 0, "tolerance": 1e-3,
                   "skipped": True}
    passed = rel <= 1e-9 and holdout["delta"] <= holdout["tolerance"]
    return {"weights_rel_delta": rel, "solver_tolerance": 1e-9,
            "gate_cols": cols, "gate_rows": rows_gate, "holdout": holdout,
            "passed": passed}


def _open_memmap(relpath: str, filename: str) -> np.ndarray:
    return np.load(data_cache_root() / relpath / filename, mmap_mode="r")


def run_m180_collection(config_path: Path,
                        output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")
    started = time.time()
    smoke = inadmissible

    configure_external_cache_environment()
    codes = config["codes"]
    spm_tr = _open_memmap(codes["spm"]["cache_relpath"],
                          codes["spm"]["train_file"])
    spm_te = _open_memmap(codes["spm"]["cache_relpath"],
                          codes["spm"]["test_file"])
    ms_tr = _open_memmap(codes["ms"]["cache_relpath"],
                         codes["ms"]["train_file"])
    ms_te = _open_memmap(codes["ms"]["test_cache_relpath"],
                         codes["ms"]["test_file"])
    pool_tr = _open_memmap(codes["pool"]["cache_relpath"],
                           codes["pool"]["train_file"])
    pool_te = _open_memmap(codes["pool"]["cache_relpath"],
                           codes["pool"]["test_file"])
    labels = np.load(data_cache_root() / codes["labels"]["relpath"]
                     / codes["labels"]["file"])["labels"]
    corpus, _ti, _tei = _load_corpus(config)
    test_labels = corpus["test_labels"]
    print(f"loaded codes: spm {spm_tr.shape} ms {ms_tr.shape} "
          f"pool {pool_tr.shape}; labels {labels.shape}", flush=True)

    # ---- the assemble+LU equivalence gate (before any wide fit) -----------
    gate = _equivalence_gate(
        [spm_tr, ms_tr], labels, cols=int(config["gate"]["cols"]),
        rows_gate=int(config["gate"]["rows"]))
    print(f"assemble+LU equivalence gate: rel "
          f"{gate['weights_rel_delta']:.3e} passed={gate['passed']}",
          flush=True)
    if not gate["passed"]:
        raise SystemExit("M180 VOID: assemble+LU equivalence gate failed")

    # ---- ONE build pass: all six standardised block Grams -----------------
    parts_train = [spm_tr, ms_tr, pool_tr]
    parts_test = [spm_te, ms_te, pool_te]
    print("building the six standardised block Grams (one pass) ...",
          flush=True)
    blocks, stds, crosses, intercept, _n_rows, colsums = _build_blocks(
        parts_train, labels)
    print("blocks built", flush=True)

    # ---- the four sealed anchors (machinery pin) ---------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    progress: dict[str, Any] = {
        "status": "running",
        "note": ("recovery record written after every completed fit; the "
                 "sealed evidence is evidence.json in this directory"),
        "anchors": {},
        "missing": {},
    }

    def _write_progress() -> None:
        write_canonical_json(output_dir / "progress.json", progress)

    anchors: dict[str, Any] = {}
    anchor_specs = {
        "V_spm": [0],
        "V_ms": [1],
        "V_pool": [2],
        "V_spm_ms": [0, 1],
    }
    sealed_values = config["sealed_anchors"]
    for name, selected in anchor_specs.items():
        fit = _fit_coalition(blocks, stds, crosses, intercept, colsums,
                             selected, parts_train, labels, parts_test,
                             test_labels)
        measured = fit["accuracy"]
        if not fit["residual"]["passed"]:
            raise SystemExit(f"M180 VOID: residual certificate failed for "
                             f"{name}: {fit['residual']}")
        tolerance = T1_TOL if name == "V_spm_ms" else TOL
        delta = abs(measured - sealed_values[name])
        anchors[name] = {"measured": measured,
                         "sealed": sealed_values[name],
                         "delta": delta,
                         "tolerance": tolerance,
                         "ok": delta <= tolerance,
                         "residual": fit["residual"]}
        progress["anchors"][name] = anchors[name]
        _write_progress()
        print(f"  anchor {name}: {measured:.10f} vs sealed "
              f"{sealed_values[name]} delta {delta:.3e} "
              f"ok={anchors[name]['ok']} "
              f"backward={fit['residual']['backward_error']:.2e}",
              flush=True)
        if not anchors[name]["ok"]:
            break

    missing: dict[str, Any] = {}
    if all(a["ok"] for a in anchors.values()):
        missing_specs = {
            "V_spm_pool": [0, 2],
            "V_ms_pool": [1, 2],
            "V_spm_ms_pool": [0, 1, 2],
        }
        for name, selected in missing_specs.items():
            free_blocks = name == "V_spm_ms_pool"
            # V(spm,ms,pool) runs LAST and frees each block as it is
            # copied, keeping its peak at ~40 GB (registered repair 5);
            # its infinity norm is precomputed because the blocks die.
            norm_g = (_norm_g_inf(blocks, selected) if free_blocks
                      else None)
            fit = _fit_coalition(blocks, stds, crosses, intercept, colsums,
                                 selected, parts_train, labels, parts_test,
                                 test_labels, norm_g=norm_g,
                                 free_blocks=free_blocks)
            if not fit["residual"]["passed"]:
                raise SystemExit(
                    f"M180 VOID: residual certificate failed for {name}: "
                    f"{fit['residual']}")
            missing[name] = {"accuracy": fit["accuracy"],
                             "residual": fit["residual"]}
            progress["missing"][name] = missing[name]
            _write_progress()
            print(f"  measured {name} = {fit['accuracy']:.10f} "
                  f"backward={fit['residual']['backward_error']:.2e}",
                  flush=True)
    progress["status"] = "complete"
    _write_progress()

    game_values = {name: anchors[name]["measured"]
                   for name in anchors if anchors[name]["ok"]}
    game_values.update({name: missing[name]["accuracy"]
                        for name in missing})
    evidence: dict[str, Any] = {
        "milestone": "M180",
        "cell": "coalition data collection (cached codes, penalty 1.0)",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "equivalence_gate": gate,
        "anchors": anchors,
        "missing_coalitions_measured": missing,
        "complete_game": game_values if len(game_values) == 7 else None,
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"anchors_ok": all(a["ok"] for a in anchors.values()),
                      "game": evidence["complete_game"]}, indent=1),
          flush=True)
    print(f"M180 collection complete -> {output_dir / 'evidence.json'}",
          flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m180_collection(args.config, args.output)


if __name__ == "__main__":
    main()
