"""M151 — SPM x MS interaction: the column-concatenated code.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v23.md`` (section 4
M151 + the section 6 amendment; 16 Aug 2026). The separability test: the
column-concatenated SPM+MS code (optionally + signed sqrt + L2) vs the
best single construction at the disclosed 53,627-column width.

Amendment (registered before measurement): the C3 run persisted only the
MS TRAIN codes; the MS test codes are encoded here once and persisted as
``v16/m151/ms357_fulltest.npy`` (34,500 rows against the three C3 scale
dictionaries). The encode is anchored by the MS anchor itself: after the
encode, the train-side penalty-1.0 refit scored on the new test codes
must reproduce the sealed MS full-data read 0.24214492753623187 (tol
1e-9). The anchor reproductions ARE the row-identity premise check for
the concatenation.

Anchors (before any concat number is read): SPM half penalty-1.0 refit
reproduces 0.2604927536231884; MS half reproduces 0.24214492753623187
(tol 1e-9 each). Gate: best concat cell (ridge read, penalty ladder
{0.1, 1.0, 10.0} x power {raw, 0.5}, full data) >= 0.2604927536231884
+ 0.005. Trained-head read at 138k (the M146 r2 protocol) is the
co-adaptation control; a cell failing both reads closes as a scoped
negative.

Smoke encodes its tiny slice to RAM (the C3 smoke pattern) and never
touches the persisted artifact; smoke refuses the sealed output
directory.

Reproduce with::

    $env:GEODE_CACHE_DIR="F:\\geode-ml\\data\\cache"
    $env:HIP_VISIBLE_DEVICES="1"
    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v23_m151_interaction
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch

from scipy import linalg as scipy_linalg

from experiments.common.data_cache import (
    configure_external_cache_environment,
    data_cache_root,
)
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    sha256_file,
    write_canonical_json,
)
from experiments.tier4.eval_v15_m104_experts import (
    RidgeAccumulator,
    Standardiser,
)
from experiments.tier4.eval_v16_m108_dictionary import _verify_device
from experiments.tier4.eval_v16_m109_trunk import (
    _load_corpus,
    _train_with_schedule,
)
from experiments.tier4.eval_v16_m142_c3 import (
    _append_scale_encode,
    _build_scale_whitener,
    _scale_dictionary,
)
from experiments.tier4.eval_v16_m142_factorial import power_norm
from experiments.tier4.eval_v16_m146_arbiter import HeadOnly

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v23"
                  / "m151_interaction.json")
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v23" / "m151_interaction"

CLASSES = 345
TOLERANCE = 1e-9
MARGIN = 0.005
SPM_ANCHOR = 0.2604927536231884
MS_ANCHOR = 0.24214492753623187
SPM_WIDTH = 40383
MS_WIDTH = 13244
CONCAT_WIDTH = SPM_WIDTH + MS_WIDTH
SCALES = (3, 5, 7)
ATOMS_BY_SCALE = {3: 1950, 5: 850, 7: 511}


def _concat_block(spm_mem: np.ndarray, ms_mem: np.ndarray, start: int,
                  stop: int) -> np.ndarray:
    """One block of the row-matched concatenation (SPM first, MS second)."""
    return np.concatenate(
        [np.asarray(spm_mem[start:stop]), np.asarray(ms_mem[start:stop])],
        axis=1)


def _fit_concat(spm_mem: np.ndarray, ms_mem: np.ndarray, labels: np.ndarray,
                n_rows: int, penalties: list[float], block: int,
                power: float | None = None) -> tuple[dict[str, np.ndarray], Any]:
    acc = RidgeAccumulator(CONCAT_WIDTH, CLASSES)
    for start in range(0, n_rows, block):
        stop = min(start + block, n_rows)
        xs = _concat_block(spm_mem, ms_mem, start, stop)
        acc.add(power_norm(xs, power) if power is not None else xs,
                labels[start:stop])
    solved = acc.solve_many(penalties)
    return ({str(q): w for q, w in solved.items()}, acc.standardiser())


# ---------------------------------------------------------------------------
# concat fitter (the section 6 amendment + correction): the 53,627-dim systems
# cannot use the sealed np.linalg.solve on this machine (it copies the 23 GB
# Gram and the smoke died at ~70 GB peak). Same accumulation arithmetic, the
# centred system built IN PLACE from the FLOAT32-ROUNDED standardiser
# statistics (exactly as _standardised_system reads them back), spilled to
# disk, per-penalty in-RAM copy factorized IN PLACE by LU (gesv, the same
# LAPACK family as the sealed solve). Equivalence-gated in-run.
# ---------------------------------------------------------------------------
def _fit_concat_inplace(spm_mem: np.ndarray, ms_mem: np.ndarray,
                        labels: np.ndarray, n_rows: int,
                        penalties: list[float], block: int,
                        scratch_dir: Path,
                        power: float | None = None,
                        max_cols: int | None = None,
                        gram_chunk: int = 2048
                        ) -> tuple[dict[str, np.ndarray], Standardiser]:
    width = CONCAT_WIDTH if max_cols is None else int(max_cols)
    gram = np.zeros((width, width), dtype=np.float64)
    colsum = np.zeros(width, dtype=np.float64)
    cross = np.zeros((width, CLASSES), dtype=np.float64)
    class_count = np.zeros(CLASSES, dtype=np.float64)
    rows = 0
    for start in range(0, n_rows, block):
        stop = min(start + block, n_rows)
        xs = _concat_block(spm_mem, ms_mem, start, stop)
        if power is not None:
            xs = power_norm(xs, power)
        if max_cols is not None:
            xs = xs[:, :max_cols]
        block_f = np.asarray(xs, dtype=np.float64)
        targets = np.zeros((len(block_f), CLASSES), dtype=np.float64)
        targets[np.arange(len(block_f)), labels[start:stop]] = 1.0
        # column-chunked accumulation: each gram entry is the SAME dot
        # product whether dgemm's output width is `width` or a chunk, so
        # the values are bitwise identical to `gram += block.T @ block`
        # while the matmul temporary stays width x chunk (the section 6
        # correction; the full-width temp is 23 GB and kills the machine).
        for c0 in range(0, width, gram_chunk):
            c1 = min(c0 + gram_chunk, width)
            gram[:, c0:c1] += block_f.T @ block_f[:, c0:c1]
        colsum += block_f.sum(axis=0)
        cross += block_f.T @ targets
        class_count += targets.sum(axis=0)
        rows += len(block_f)
    # the RidgeAccumulator._standardised_system closed form, IN PLACE,
    # from the float32-rounded statistics (the section 6 correction).
    diag = gram.diagonal().copy()
    centre = colsum / rows
    variance = diag / rows - np.square(centre)
    scale = np.sqrt(np.maximum(variance, 0.0)) + 1e-8
    std32 = Standardiser(centre.astype(np.float32),
                         scale.astype(np.float32))
    centre64 = std32.centre.astype(np.float64)
    inv64 = 1.0 / std32.scale.astype(np.float64)
    intercept = class_count / rows
    gram -= np.outer(colsum, centre64)
    gram *= inv64[:, None]
    gram *= inv64[None, :]
    cross_std = (cross - np.outer(centre64, class_count)) * inv64[:, None]
    scratch_dir.mkdir(parents=True, exist_ok=True)
    centred_mem = np.lib.format.open_memmap(
        scratch_dir / "centred.npy", mode="w+", dtype=np.float64,
        shape=(width, width))
    centred_mem[:] = gram
    del gram
    out: dict[str, np.ndarray] = {}
    for penalty in penalties:
        # F-order copy so the LAPACK gesv factors IN PLACE (a C-order
        # input would be internally copied to F-order = another 23 GB).
        buf = np.array(centred_mem, order="F")
        buf.flat[:: width + 1] += penalty
        # in-place LU (gesv), the same LAPACK family as the sealed solve
        w = scipy_linalg.solve(buf, cross_std, overwrite_a=True,
                               check_finite=False)
        out[str(penalty)] = np.vstack([w, intercept[None, :]])
        del buf
    return out, std32


def _inplace_equivalence_check(spm_mem: np.ndarray, ms_mem: np.ndarray,
                               labels: np.ndarray, n_rows: int,
                               cols: int, block: int, scratch_dir: Path
                               ) -> dict[str, Any]:
    """The registered gate: in-place LU path vs the sealed solve path."""
    acc = RidgeAccumulator(cols, CLASSES)
    for start in range(0, n_rows, block):
        stop = min(start + block, n_rows)
        xs = _concat_block(spm_mem, ms_mem, start, stop)[:, :cols]
        acc.add(xs, labels[start:stop])
    ref = acc.solve_many([1.0])[1.0]
    std_ref = acc.standardiser()
    del acc
    w_inp, std_inp = _fit_concat_inplace(
        spm_mem, ms_mem, labels, n_rows, [1.0], block, scratch_dir,
        power=None, max_cols=cols)
    w_inp_1 = w_inp["1.0"]
    rel = float(np.max(np.abs(ref - w_inp_1))
                / max(float(np.max(np.abs(ref))), 1e-12))
    centre_ok = bool(np.allclose(std_ref.centre, std_inp.centre,
                                 rtol=1e-9, atol=1e-12))
    scale_ok = bool(np.allclose(std_ref.scale, std_inp.scale,
                                rtol=1e-9, atol=1e-12))
    return {"weights_rel_delta": rel,
            "weights_tolerance": 1e-9,
            "standardiser_ok": bool(centre_ok and scale_ok),
            "check_rows": int(n_rows), "check_cols": int(cols),
            "passed": bool(rel <= 1e-9 and centre_ok and scale_ok)}


def _score_concat(weights: np.ndarray, standardise, spm_test: np.ndarray,
                  ms_test: np.ndarray, labels: np.ndarray, block: int,
                  power: float | None = None) -> float:
    hits = 0
    n = len(labels)
    for start in range(0, n, block):
        stop = min(start + block, n)
        xs = _concat_block(spm_test, ms_test, start, stop)
        if power is not None:
            xs = power_norm(xs, power)
        scores = standardise(xs) @ weights[:-1] + weights[-1]
        hits += int((np.argmax(scores, axis=1)
                     == labels[start:stop]).sum())
    return hits / n


def _encode_ms_test(config: dict[str, Any], corpus: dict[str, np.ndarray],
                    device: torch.device, n_test: int, throttle: float
                    ) -> np.ndarray:
    """The C3 multi-scale encode of the first n_test test rows (to RAM)."""
    out = np.empty((n_test, MS_WIDTH), dtype=np.float32)
    col_start = 0
    for patch in SCALES:
        whitener, candidates = _build_scale_whitener(config, corpus, patch)
        dictionary = _scale_dictionary(
            candidates, len(candidates), int(config["sparse"]
                                             ["dictionary_seed"]),
            ATOMS_BY_SCALE[patch])
        _append_scale_encode(corpus["test_images"], np.arange(n_test),
                             dictionary, whitener, device, out, 0, col_start,
                             throttle)
        col_start += 4 * ATOMS_BY_SCALE[patch]
    return out


def _concat_batches(spm_mem, ms_mem, labels, rows, power, batch, device
                    ) -> Callable[[], Iterator]:
    def gen() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for start in range(0, len(rows), batch):
            take = rows[start:start + batch]
            # gather the ROWS named by `take` (a permutation in the
            # trained-head split), not the contiguous range between its
            # endpoints (registered defect, 16 Aug)
            block = np.concatenate(
                [np.asarray(spm_mem[take]), np.asarray(ms_mem[take])],
                axis=1)
            if power is not None:
                block = power_norm(block, power)
            yield (torch.from_numpy(np.ascontiguousarray(
                block.astype(np.float32))).to(device),
                torch.from_numpy(labels[take]).to(device))
    return gen


def run_m151(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")

    started = time.time()
    smoke = inadmissible
    skip_anchors = bool(config.get("_smoke_skip_anchors", False))
    smoke_train = int(config.get("_smoke_train_rows", 10 ** 9))
    smoke_test = int(config.get("_smoke_test_rows", 10 ** 9))
    block = int(config["numerics"]["block"])

    torch.set_num_threads(int(config["numerics"]["torch_threads"]))
    torch.manual_seed(109)
    configure_external_cache_environment()
    _verify_device(torch)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    m142_cache = data_cache_root() / config["artifacts"]["m142_cache_relpath"]
    ms_cache = data_cache_root() / config["artifacts"]["m142_c3_cache_relpath"]
    evidence: dict[str, Any] = {
        "milestone": "M151",
        "cell": "SPM x MS interaction (separability test)",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "question": config["question"],
    }

    print("loading corpus + cached labels", flush=True)
    corpus, _train_index, _test_index = _load_corpus(config)
    test_labels = corpus["test_labels"][:smoke_test]
    test_domains = corpus["test_domains"][:smoke_test]
    labels = np.load(m142_cache / config["artifacts"]["labels_file"])["labels"]
    spm_train = np.load(m142_cache / config["artifacts"]["spm_train_file"],
                        mmap_mode="r")
    spm_test = np.load(m142_cache / config["artifacts"]["spm_test_file"],
                       mmap_mode="r")[:smoke_test]
    ms_train = np.load(ms_cache / config["artifacts"]["ms_train_file"],
                       mmap_mode="r")
    n_train = min(len(labels), len(spm_train), len(ms_train), smoke_train)
    n_test = len(test_labels)
    if spm_train.shape[0] < n_train or ms_train.shape[0] < n_train:
        raise SystemExit("M151 premise failure: train row counts disagree")

    # ---- MS test codes (encode once; anchor validates the encode) ----------
    ms_path = data_cache_root() / "v16" / "m151" / "ms357_fulltest.npy"
    if smoke:
        print(f"smoke: encoding {n_test} MS test rows to RAM", flush=True)
        ms_test = _encode_ms_test(config, corpus, device, n_test,
                                  throttle=0.0)
    elif ms_path.exists():
        ms_test = np.load(ms_path, mmap_mode="r")[:n_test]
        print(f"reusing persisted MS test codes {ms_path}", flush=True)
    else:
        print(f"encoding {n_test} MS test rows (persisting)", flush=True)
        ms_test = _encode_ms_test(config, corpus, device, n_test,
                                  throttle=0.05)
        ms_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(ms_path, ms_test)
        ms_test = np.load(ms_path, mmap_mode="r")[:n_test]
    evidence["ms_test_codes"] = {"path": str(ms_path), "rows": int(n_test),
                                 "smoke_ram_only": smoke}

    # ---- anchors: each half alone ------------------------------------------
    anchors: dict[str, Any] = {}
    print("anchors: each half alone (penalty 1.0)", flush=True)
    acc_spm = RidgeAccumulator(SPM_WIDTH, CLASSES)
    acc_ms = RidgeAccumulator(MS_WIDTH, CLASSES)
    for start in range(0, n_train, block):
        stop = min(start + block, n_train)
        acc_spm.add(np.asarray(spm_train[start:stop]), labels[start:stop])
        acc_ms.add(np.asarray(ms_train[start:stop]), labels[start:stop])
    w_spm = acc_spm.solve_many([1.0])[1.0]
    w_ms = acc_ms.solve_many([1.0])[1.0]
    std_spm = acc_spm.standardiser()
    std_ms = acc_ms.standardiser()
    hits = 0
    for start in range(0, n_test, block):
        stop = min(start + block, n_test)
        scores = std_spm(np.asarray(spm_test[start:stop])) @ w_spm[:-1] \
            + w_spm[-1]
        hits += int((np.argmax(scores, axis=1)
                     == test_labels[start:stop]).sum())
    spm_acc = hits / n_test
    hits = 0
    for start in range(0, n_test, block):
        stop = min(start + block, n_test)
        scores = std_ms(np.asarray(ms_test[start:stop])) @ w_ms[:-1] \
            + w_ms[-1]
        hits += int((np.argmax(scores, axis=1)
                     == test_labels[start:stop]).sum())
    ms_acc = hits / n_test
    anchors["spm_half"] = {"measured": spm_acc, "sealed": SPM_ANCHOR,
                           "delta": spm_acc - SPM_ANCHOR,
                           "tolerance": TOLERANCE}
    anchors["ms_half"] = {"measured": ms_acc, "sealed": MS_ANCHOR,
                          "delta": ms_acc - MS_ANCHOR,
                          "tolerance": TOLERANCE}
    print(f"  SPM {spm_acc:.6f} (delta {spm_acc - SPM_ANCHOR:+.3e}); "
          f"MS {ms_acc:.6f} (delta {ms_acc - MS_ANCHOR:+.3e})", flush=True)
    if not skip_anchors and (abs(spm_acc - SPM_ANCHOR) > TOLERANCE
                             or abs(ms_acc - MS_ANCHOR) > TOLERANCE):
        evidence.update({"void": True,
                         "void_reason": "half-alone anchor reproduction failed",
                         "anchors": anchors})
        _write(output_dir, evidence)
        return evidence

    # ---- concat cells ------------------------------------------------------
    penalties = [float(q) for q in config["cell"]["penalty_ladder"]]
    powers = [None if p is None else float(p)
              for p in config["cell"]["power_ladder"]]
    scratch = (data_cache_root() / "v23" / "m151_solver_scratch"
               / output_dir.name)
    eq = _inplace_equivalence_check(
        spm_train, ms_train, labels,
        int(config["cell"].get("_equiv_check_rows", 20000)),
        int(config["cell"].get("_equiv_check_cols", 8192)), block, scratch)
    print(f"inplace equivalence: rel {eq['weights_rel_delta']:.3e} "
          f"(tol 1e-9), standardiser {eq['standardiser_ok']} -> "
          f"passed={eq['passed']}", flush=True)
    if not eq["passed"]:
        evidence.update({"void": True,
                         "void_reason": "concat-solver equivalence gate "
                                        "failed",
                         "inplace_equivalence": eq})
        _write(output_dir, evidence)
        return evidence
    evidence["inplace_equivalence"] = eq

    cells: dict[str, Any] = {}
    best_key, best_acc = None, -1.0
    for power in powers:
        tag = "raw" if power is None else f"p{power}"
        print(f"concat {tag}: fit + ladder", flush=True)
        solved, std = _fit_concat_inplace(
            spm_train, ms_train, labels, n_train, penalties, block,
            scratch, power=power)
        for q in penalties:
            acc = _score_concat(solved[str(q)], std, spm_test, ms_test,
                                test_labels, block, power=power)
            key = f"{tag}_lambda{q}"
            cells[key] = {"accuracy": acc, "power": tag, "penalty": q}
            print(f"  {key}: {acc:.4f}", flush=True)
            if acc > best_acc:
                best_acc, best_key = acc, key

    # ---- 138k read ---------------------------------------------------------
    n_138 = min(int(config["cell"]["n_138k"]), n_train)
    cells_138: dict[str, Any] = {}
    for power in powers:
        tag = "raw" if power is None else f"p{power}"
        solved, std = _fit_concat_inplace(
            spm_train, ms_train, labels, n_138, penalties, block, scratch,
            power=power)
        for q in penalties:
            acc = _score_concat(solved[str(q)], std, spm_test, ms_test,
                                test_labels, block, power=power)
            cells_138[f"{tag}_lambda{q}"] = acc

    # ---- trained-head read at 138k -----------------------------------------
    print("trained-head read (138k, concat codes)", flush=True)
    model = HeadOnly(CONCAT_WIDTH, CLASSES, device)
    order = np.random.default_rng(11).permutation(n_138)
    val_count = int(round(n_138 * 0.05))
    train_fit = order[val_count:]
    val_rows = order[:val_count]
    batch = 64
    power_t = 0.5
    training = _train_with_schedule(
        model,
        _concat_batches(spm_train, ms_train, labels, train_fit, power_t,
                        batch, device),
        _concat_batches(spm_train, ms_train, labels, val_rows, power_t,
                        batch, device),
        4, 3e-4, 1e-4, device, 2)
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for inputs, lab in _concat_batches(spm_test, ms_test, test_labels,
                                           np.arange(n_test), power_t, batch,
                                           device)():
            logits = model(inputs)
            correct += int((logits.argmax(dim=1) == lab).sum().item())
            total += len(lab)
    trained_acc = correct / total
    print(f"  trained {trained_acc:.6f} (val "
          f"{training['best_validation_accuracy']:.6f})", flush=True)
    del model
    torch.cuda.empty_cache()

    # ---- gate ---------------------------------------------------------------
    gain = best_acc - SPM_ANCHOR
    fired = gain < MARGIN
    both_fail = fired and trained_acc < SPM_ANCHOR + MARGIN
    evidence.update({
        "anchors": anchors,
        "cells": cells,
        "cells_138k": cells_138,
        "best_cell": best_key,
        "best_accuracy": best_acc,
        "trained_head_read": trained_acc,
        "trained_val": training["best_validation_accuracy"],
        "concat_solver": {
            "method": "in-place LU (scipy overwrite_a, the gesv family of "
                      "the sealed solve) on the disk-spilled centred "
                      "system built from the float32-rounded standardiser "
                      "statistics; the section 6 amendment + correction; "
                      "equivalence-gated in-run against the sealed solve "
                      "path",
            "scratch_relpath": str(scratch.relative_to(data_cache_root())),
            "centred_sha256": sha256_file(scratch / "centred.npy"),
        },
        "gate": {
            "registered": config["gate"]["registered"],
            "incumbent": SPM_ANCHOR,
            "gain": gain,
            "required": MARGIN,
            "fired": fired,
            "both_reads_fail": bool(both_fail),
            "consequence": ("scoped negative: the concatenation adds no "
                            "measured value over the best single "
                            "construction" if both_fail
                            else "the interaction is additive-or-better"),
        },
        "runtime_seconds": round(time.time() - started, 2),
    })
    _write(output_dir, evidence)
    print(f"\nM151 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def _write(output_dir: Path, evidence: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m151(args.config, args.output)


if __name__ == "__main__":
    main()
