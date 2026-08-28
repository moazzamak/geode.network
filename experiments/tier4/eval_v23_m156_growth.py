"""M156 — residual growth on the GLOBAL head (full-data scale).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v23.md`` (section 4
M156, section 6; 16 Aug 2026). The construction is registered in full
BEFORE anything runs here; the budgets come from the sealed M155 premise.

Base = the promoted SPM+sqrt full-data head (test read 0.27855072463768116).
Error rows = the M155 premise population (76,670 train rows; exact-match
integrity gate). Budgets = {256, 2048} atoms.

Arms per budget g (identical in everything but the dictionary):

- growth: the first g atoms of the cached f6144 pool (the cached codes
  are the [11,100]-permuted shared pool's first 6,144 atoms, M103
  interleaved column layout: atom a owns {a, 6144+a, 12288+a, 18432+a}),
  extracted from the cache — no new encode.
- control: blind-greedy group-OMP (M108 arm (c)) over the SAME
  error-row features, pool 6,144, budget 2,048, GPU port with the M145
  order-parity check. On parity failure the control arm is VOID and only
  growth-vs-static is adjudicated (section 6 amendment).

Both arms: ridge head (penalty 1.0) on the SAME error rows; scores on
train + test from the cached codes in blocks; 2-arm stacking [base, arm]
with the M143b train protocol (valid seed 55, frac 0.8, ladder
{1,10,100,1000,10000}), evaluated on the test scores. Static = the base
head's test read.

Anchors (before any new number): a1 the base test read
0.27855072463768116 (tol 1e-9); a2 the M145 specialist-path anchor — d0
512-atom own-domain 0.19357142857142856 (tol 0.002); a3 the growth dicts
are nested prefixes and the cached width is 24,576; a4 cached-code
reproduction — a fresh GPU encode of the first 64 train rows with the
rebuilt 6,144-atom dictionary reproduces the cached f6144 codes (tol
1e-5).

Premise gates in-run: n_error > 0 hard-fails; n_error must equal the
sealed 76,670 exactly (else void); ceil(n_error / (4*g)) >= 10 per
budget (else the budget is void, not negative).

Gate per budget: growth_fused >= static + 0.005 AND growth_fused >
control_fused; else scoped negative. Ops ledger disclosed per arm.
Smoke declares inadmissibility and refuses the sealed output directory.
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
from experiments.tier4.eval_v15_m103_atoms import select_discriminative
from experiments.tier4.eval_v15_m104_experts import (
    RidgeAccumulator,
    _inference_macs,
    _load_domainnet,
)
from experiments.tier4.eval_v16_m109_trunk import _load_corpus
from experiments.tier4.eval_v16_m140_data_extension import _extension_indices
from experiments.tier4.eval_v16_m141_data_full import _rest_extension_indices
from experiments.tier4.eval_v16_m142_c4 import _fit_power, _score_power
from experiments.tier4.eval_v16_m142_factorial import power_norm
from experiments.tier4.eval_v16_m143_integration import (
    _select_penalty,
    _stacking_fit,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v23"
                  / "m156_growth.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v23" / "m156_growth")

CLASSES = 345
POOL_GRID = 2
BINS = 4                    # pool_grid**2 pooled columns per atom
TOLERANCE_A1 = 1e-9
TOLERANCE_A2 = 0.002
TOLERANCE_A4 = 1e-5
VALID_FRAC = 0.8
VALID_SEED = 55
LADDER = [1.0, 10.0, 100.0, 1000.0, 10000.0]
MARGIN = 0.005


# ---------------------------------------------------------------------------
# pure helpers (unit-tested)
# ---------------------------------------------------------------------------
def _prefix_columns(atom_count: int, g: int) -> np.ndarray:
    """Columns owned by the first g atoms in the cached pool layout.

    The M103 interleaved layout: atom a owns columns {a, A+a, 2A+a, 3A+a}
    with A = atom_count (the bin-major ``_pool`` output).
    """
    return np.concatenate(
        [np.arange(g, dtype=np.int64) + q * int(atom_count)
         for q in range(BINS)])


def _nested_prefix(cols_small: np.ndarray, cols_big: np.ndarray) -> bool:
    """cols_small is a subset of cols_big (the a3 prefix property)."""
    return bool(np.isin(cols_small, cols_big).all())


def _score_matrix(parts: list[tuple[np.ndarray, np.ndarray]],
                  cols: np.ndarray, weights: np.ndarray, std,
                  block: int, n_total: int) -> np.ndarray:
    """Score rows of the parts (column-extracted) with a fitted head."""
    out = np.empty((n_total, CLASSES), dtype=np.float32)
    offset = 0
    for mem, _part_labels in parts:
        for start in range(0, len(mem), block):
            stop = min(start + block, len(mem))
            xs = std(np.asarray(mem[start:stop])[:, cols])
            out[offset:offset + stop - start] = xs @ weights[:-1] \
                + weights[-1]
            offset += stop - start
    if offset != n_total:
        raise SystemExit("M156 instrument failure: score-matrix row count "
                         f"{offset} != {n_total}")
    return out


def _base_score_matrix(spm_parts: list[tuple[np.ndarray, np.ndarray]],
                       weights: np.ndarray, std, block: int, n_total: int,
                       p: float) -> np.ndarray:
    """Base-head (SPM+sqrt) scores over the parts, in the C4 protocol."""
    out = np.empty((n_total, CLASSES), dtype=np.float32)
    offset = 0
    for mem, _part_labels in spm_parts:
        for start in range(0, len(mem), block):
            stop = min(start + block, len(mem))
            xs = std(power_norm(np.asarray(mem[start:stop]), p))
            out[offset:offset + stop - start] = xs @ weights[:-1] \
                + weights[-1]
            offset += stop - start
    if offset != n_total:
        raise SystemExit("M156 instrument failure: base score-matrix row "
                         f"count {offset} != {n_total}")
    return out


def _floor_ok(n_error_rows: int, atoms: int, floor: float) -> bool:
    """ceil(n_error_rows / (4 * atoms)) >= floor."""
    ratio = (int(n_error_rows) + BINS * atoms - 1) // (BINS * atoms)
    return float(ratio) >= float(floor)


def _extract_error_rows(parts: list[tuple[np.ndarray, np.ndarray]],
                        err_pos: np.ndarray, cols: np.ndarray | None,
                        block: int) -> np.ndarray:
    """(n_error, len(cols)) codes at the error positions (or full width
    when cols is None), across the parts, in err_pos order."""
    n_error = len(err_pos)
    width = (parts[0][0].shape[1] if cols is None else len(cols))
    out = np.empty((n_error, width), dtype=np.float32)
    offset = 0
    filled = 0
    for mem, _part_labels in parts:
        for start in range(0, len(mem), block):
            stop = min(start + block, len(mem))
            take_rows = err_pos[(err_pos >= offset)
                                & (err_pos < offset + stop - start)] \
                - offset
            if len(take_rows):
                block_x = np.asarray(mem[start:stop])[take_rows]
                out[filled:filled + len(take_rows)] = (
                    block_x if cols is None else block_x[:, cols])
                filled += len(take_rows)
            offset += stop - start
    if filled != n_error:
        raise SystemExit(f"instrument failure: extracted {filled} error "
                         f"rows, expected {n_error}")
    return out


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def run_m156(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")

    started = time.time()
    smoke = inadmissible
    skip_anchors = bool(config.get("_smoke_skip_anchors", False))
    skip_control = bool(config.get("_smoke_skip_control", False))
    allow_floor = bool(config.get("_smoke_allow_floor_violation", False))
    smoke_train = int(config.get("_smoke_train_rows", 10 ** 9))
    smoke_test = int(config.get("_smoke_test_rows", 10 ** 9))
    block = int(config["numerics"]["block"])

    configure_external_cache_environment()
    root = data_cache_root()
    f6144_cache = root / config["artifacts"]["f6144_cache_relpath"]
    m142_cache = root / config["artifacts"]["m142_cache_relpath"]
    pool_atoms = int(config["growth"]["pool_atoms"])
    budgets = sorted(int(g) for g in config["growth"]["budgets"])
    head_penalty = float(config["growth"]["head_penalty"])
    floor = float(config["premise"]["fit_floor"])
    evidence: dict[str, Any] = {
        "milestone": "M156",
        "cell": "residual growth on the global head (full-data scale)",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "question": config["question"],
    }

    # ---- corpus + cached parts ----------------------------------------------
    print("loading corpus + cached codes", flush=True)
    corpus, train_index, _test_index = _load_corpus(config)
    labels = np.load(m142_cache / config["artifacts"]["labels_file"])["labels"]
    part1 = np.load(f6144_cache / config["artifacts"]["f6144_train_file"],
                    mmap_mode="r")
    mem_test = np.load(f6144_cache / config["artifacts"]["f6144_test_file"],
                       mmap_mode="r")
    spm_train = np.load(m142_cache / config["artifacts"]["spm_train_file"],
                        mmap_mode="r")
    spm_test = np.load(m142_cache / config["artifacts"]["spm_test_file"],
                       mmap_mode="r")
    test_labels = corpus["test_labels"][:smoke_test]
    test_domains = corpus["test_domains"][:smoke_test]
    n_test = len(test_labels)
    mem_test = mem_test[:n_test]
    spm_test = spm_test[:n_test]
    if smoke:
        parts = [(part1[:smoke_train], labels[:smoke_train])]
        spm_parts = [(spm_train[:smoke_train], labels[:smoke_train])]
        n_train = min(len(labels), smoke_train)
    else:
        raw = _load_domainnet(int(config["corpus"]["image_size"]))
        ext600 = np.load(root / "v16" / "m140" / "f6144_ext600.npy",
                         mmap_mode="r")
        rest = np.load(root / "v16" / "m141" / "f6144_all_rest.npy",
                       mmap_mode="r")
        ext_idx, _ = _extension_indices(raw["train_labels"], train_index,
                                        600, CLASSES)
        rest_idx = _rest_extension_indices(raw["train_labels"], train_index,
                                           CLASSES, per_class_take=200)
        if len(ext600) != 69000 or len(rest) != 202832:
            raise SystemExit("M156 premise failure: full-data part sizes")
        parts = [
            (part1, corpus["train_labels"]),
            (ext600, raw["train_labels"][ext_idx]),
            (rest, raw["train_labels"][rest_idx]),
        ]
        spm_parts = [
            (spm_train, labels),
        ]
        n_train = len(labels)
        if n_train != 409832:
            raise SystemExit("M156 premise failure: full-data row count")
    print(f"rows: train {n_train} / test {n_test}", flush=True)

    # ---- a1 anchor: the base head reproduction ------------------------------
    print("base head: C4 fitter (p=0.5, penalty 0.1)", flush=True)
    p_base = float(config["base"]["p"])
    base_penalty = float(config["base"]["penalty"])
    solved, std_base = _fit_power(spm_train, labels, p_base, [base_penalty],
                                  n_train, block, transform=True)
    base_test_acc = _score_power(spm_test, test_labels, test_domains,
                                 p_base, solved[str(base_penalty)], std_base,
                                 block, transform=True)
    base_train_scores = _base_score_matrix(
        spm_parts, solved[str(base_penalty)], std_base, block, n_train,
        p_base)
    base_test_scores = _base_score_matrix(
        [(spm_test, test_labels)], solved[str(base_penalty)], std_base,
        block, n_test, p_base)
    anchors: dict[str, Any] = {
        "a1_base": {"measured": base_test_acc,
                    "sealed": float(config["anchors"]["base_full_data"]),
                    "delta": base_test_acc
                    - float(config["anchors"]["base_full_data"]),
                    "tolerance": TOLERANCE_A1},
    }
    print(f"  a1 base test {base_test_acc:.6f} (delta "
          f"{anchors['a1_base']['delta']:+.3e})", flush=True)
    if not skip_anchors and abs(anchors["a1_base"]["delta"]) > TOLERANCE_A1:
        evidence.update({"void": True,
                         "void_reason": "a1 base-head reproduction failed",
                         "anchors": anchors})
        _write(output_dir, evidence)
        return evidence

    # ---- error rows + premise ------------------------------------------------
    full_labels = labels[:n_train]
    train_preds = np.argmax(base_train_scores, axis=1)
    err_pos = np.flatnonzero(train_preds != full_labels)
    n_error = len(err_pos)
    expected = int(config["premise"]["expected_n_error_rows"])
    print(f"  error rows {n_error} (sealed premise {expected})", flush=True)
    if n_error == 0:
        raise SystemExit("M156 PREMISE FAILURE: zero error rows; a growth "
                         "population of zero is meaningless. Not waivable.")
    if not smoke and n_error != expected:
        evidence.update({"void": True,
                         "void_reason": "error-row population does not match "
                                        "the sealed M155 premise",
                         "premise": {"n_error_rows": int(n_error),
                                     "expected": expected}})
        _write(output_dir, evidence)
        return evidence
    floor_report: dict[str, Any] = {"n_error_rows": int(n_error)}
    for g in budgets:
        ok = _floor_ok(n_error, g, floor)
        floor_report[str(g)] = {
            "floor_ok": bool(ok),
            "rows_per_dim": (n_error + BINS * g - 1) // (BINS * g),
            "fitted_dims": BINS * g,
        }
        if not ok and not allow_floor:
            raise SystemExit(
                f"M156 PREMISE GATE: budget {g} is below the floor "
                f"({floor_report[str(g)]['rows_per_dim']} rows per fitted "
                f"dimension < {floor}); the budget is void, not negative.")
    evidence["premise"] = {"floor": floor_report}

    # ---- anchors a3/a2/a4 + GPU machinery (full runs) ------------------------
    device = None
    whitener = None
    grid = int(config["numerics"]["grid"])
    dimension = int(config["numerics"]["dimension"])
    if not skip_anchors:
        from experiments.tier4.eval_v16_a5_routed import _domain_candidates
        from experiments.tier4.eval_v16_m108_dictionary import (
            _encode_block_device,
            _verify_device,
        )
        from experiments.tier4.eval_v16_m109_trunk import (
            _build_whitener_dictionary,
        )
        import torch

        torch.set_num_threads(int(config["numerics"]["torch_threads"]))
        _verify_device(torch)
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        batch = int(config["numerics"]["batch"])
        throttle = float(config["numerics"]["encode_throttle_seconds"])

        sparse_tmp = dict(config["sparse"])
        sparse_tmp["atoms"] = int(config["sparse"]["pool_size"])
        config_tmp = dict(config)
        config_tmp["sparse"] = sparse_tmp
        whitener, dictionary, grid, dimension, _pg = \
            _build_whitener_dictionary(config_tmp, corpus)
        if grid != int(config["numerics"]["grid"]) or \
                dimension != int(config["numerics"]["dimension"]):
            raise SystemExit("M156 instrument failure: whitener grid/"
                             "dimension do not match the registered "
                             "numerics")

        # a3: nested prefixes of the permuted pool + cached width
        anchors["a3_prefix"] = {
            "ok": bool(_nested_prefix(_prefix_columns(pool_atoms,
                                                      budgets[0]),
                                      _prefix_columns(pool_atoms,
                                                      budgets[-1]))),
            "cached_width_ok": bool(part1.shape[1] == pool_atoms * BINS),
            "cached_width": int(part1.shape[1]),
        }
        if not anchors["a3_prefix"]["ok"] or \
                not anchors["a3_prefix"]["cached_width_ok"]:
            raise SystemExit("M156 instrument failure: a3 prefix/width "
                             "property failed")

        # a2: the M145 specialist-path anchor (d0 512 atoms)
        print("a2 anchor: domain-0 512-atom specialist reproduction",
              flush=True)
        cand = _domain_candidates(corpus, 0, whitener)
        order = np.random.default_rng([11, 100]).permutation(len(cand))
        d0_dict = cand[order[:512]]
        table = torch.from_numpy(np.ascontiguousarray(d0_dict)
                                 ).to(torch.float32).to(device)
        rows_d = np.where(corpus["train_domains"] == 0)[0]
        acc = RidgeAccumulator(512 * BINS, CLASSES)
        for start in range(0, len(rows_d), batch):
            take = rows_d[start:start + batch]
            block_t = _encode_block_device(corpus["train_images"][take],
                                           table, whitener, POOL_GRID)
            acc.add(block_t, corpus["train_labels"][take])
            if throttle > 0:
                time.sleep(throttle)
        w_d0 = acc.solve_many([1.0])[1.0]
        std_d0 = acc.standardiser()
        own_rows = np.where(corpus["test_domains"] == 0)[0]
        own_rows = own_rows[own_rows < n_test]
        hits = 0
        for start in range(0, len(own_rows), batch):
            take = own_rows[start:start + batch]
            block_t = _encode_block_device(corpus["test_images"][take],
                                           table, whitener, POOL_GRID)
            hits += int((np.argmax(std_d0(block_t) @ w_d0[:-1]
                                   + w_d0[-1], axis=1)
                         == corpus["test_labels"][take]).sum())
            if throttle > 0:
                time.sleep(throttle)
        own_acc = hits / len(own_rows)
        del table
        torch.cuda.empty_cache()
        anchors["a2_d0"] = {"measured": own_acc,
                            "sealed": float(config["anchors"]
                                            ["d0_own_domain"]),
                            "delta": own_acc
                            - float(config["anchors"]["d0_own_domain"]),
                            "tolerance": TOLERANCE_A2}
        print(f"  a2 d0 own-domain {own_acc:.6f} (delta "
              f"{anchors['a2_d0']['delta']:+.6f})", flush=True)
        if abs(anchors["a2_d0"]["delta"]) > TOLERANCE_A2:
            evidence.update({"void": True,
                             "void_reason": "a2 specialist-path anchor "
                                            "failed",
                             "anchors": anchors})
            _write(output_dir, evidence)
            return evidence

        # a4: cached-code reproduction (fresh encode of 64 train rows)
        print("a4 anchor: cached-code reproduction (64 rows)", flush=True)
        a4_rows = np.arange(64)
        table6144 = torch.from_numpy(
            np.ascontiguousarray(dictionary[:pool_atoms])
        ).to(torch.float32).to(device)
        a4_encoded = _encode_block_device(corpus["train_images"][a4_rows],
                                          table6144, whitener, POOL_GRID)
        a4_cached = np.asarray(part1[:64])
        a4_max_delta = float(np.abs(a4_encoded - a4_cached).max())
        del table6144
        torch.cuda.empty_cache()
        anchors["a4_cache"] = {"max_abs_delta": a4_max_delta,
                               "tolerance": TOLERANCE_A4}
        print(f"  a4 max abs delta {a4_max_delta:.3e}", flush=True)
        if a4_max_delta > TOLERANCE_A4:
            evidence.update({"void": True,
                             "void_reason": "a4 cached-code reproduction "
                                            "failed",
                             "anchors": anchors})
            _write(output_dir, evidence)
            return evidence

    # ---- control selection (blind-greedy OMP on the error rows) --------------
    ctrl_order: np.ndarray | None = None
    selection_macs = 0
    control_void = False
    err_features = None
    if not skip_control:
        from experiments.tier4.eval_v16_m108_dictionary import (
            _select_discriminative_gpu,
        )
        import torch

        if device is None:
            raise SystemExit("M156 instrument failure: the control OMP "
                             "requires the GPU anchor setup (device is "
                             "None); skip_control without skip_anchors is "
                             "not a registered configuration.")
        print("control: error-row feature extraction (24,576 cols)",
              flush=True)
        err_features = _extract_error_rows(parts, err_pos, None, block)
        err_labels = full_labels[err_pos]

        parity_rows = int(config["control"]["parity_subset_rows"])
        parity_budget = int(config["control"]["parity_budget"])
        print("control: OMP order-parity check (numpy vs GPU)", flush=True)
        ref_order, _ = select_discriminative(
            err_features[:parity_rows], err_labels[:parity_rows],
            pool_atoms, parity_budget, POOL_GRID)
        gpu_order, _ = _select_discriminative_gpu(
            err_features[:parity_rows], err_labels[:parity_rows],
            pool_atoms, parity_budget, POOL_GRID, device)
        parity_ok = bool(np.array_equal(ref_order, gpu_order))
        print(f"  order parity: {'passed' if parity_ok else 'FAILED'}",
              flush=True)
        if parity_ok:
            print("control: full OMP selection (budget 2048)", flush=True)
            ctrl_order, selection_macs = _select_discriminative_gpu(
                err_features, err_labels, pool_atoms,
                int(config["control"]["budget"]), POOL_GRID, device)
        else:
            control_void = True
            ctrl_order = None
        evidence["control_selection"] = {
            "parity_checked": parity_ok,
            "void": control_void,
            "selection_rows": int(n_error),
            "selection_macs": int(selection_macs),
        }
    else:
        ctrl_order = np.arange(max(budgets), dtype=np.int64)

    # ---- growth + control arms per budget ------------------------------------
    results: dict[str, Any] = {}
    static = float(base_test_acc)
    # 1-arm stacking diagnostic: the M143b protocol over [base] alone.
    diag_p, _diag_ladder = _select_penalty(
        lambda p: _stack_metric(base_train_scores, full_labels, p), LADDER)
    diag_stacking = _stacking_fit(base_train_scores, full_labels, diag_p)
    diag_fused = float((diag_stacking(base_test_scores)
                        == test_labels).mean())
    evidence["diagnostic"] = {
        "one_arm_stack_fused": diag_fused,
        "one_arm_stack_penalty": diag_p,
        "note": "the M143b protocol over the base scores alone; the gate "
                "compares growth against the base read itself (static).",
    }
    print(f"  diagnostic 1-arm stack {diag_fused:.6f} (base read "
          f"{static:.6f})", flush=True)

    for g in budgets:
        print(f"\nbudget g={g}", flush=True)
        cols_g = _prefix_columns(pool_atoms, g)
        arm_dicts: dict[str, np.ndarray] = {
            "growth": cols_g,
        }
        if control_void:
            arm_dicts = {"growth": cols_g}
        elif ctrl_order is not None:
            arm_dicts["control"] = np.concatenate(
                [ctrl_order[:g] + q * pool_atoms for q in range(BINS)])
        for name, cols in arm_dicts.items():
            print(f"  arm {name}: error-row codes + head fit", flush=True)
            if err_features is not None:
                err_codes = err_features[:, cols]
            else:
                err_codes = _extract_error_rows(parts, err_pos, cols, block)
            acc = RidgeAccumulator(len(cols), CLASSES)
            acc.add(err_codes, full_labels[err_pos])
            weights = acc.solve_many([head_penalty])[head_penalty]
            std = acc.standardiser()
            print(f"  arm {name}: train + test scores", flush=True)
            train_scores = _score_matrix(parts, cols, weights, std, block,
                                         n_train)
            test_scores = _score_matrix([(mem_test, test_labels)], cols,
                                        weights, std, block, n_test)
            stack_train = np.concatenate([base_train_scores, train_scores],
                                         axis=1)
            stack_test = np.concatenate([base_test_scores, test_scores],
                                        axis=1)
            pen, _ladder = _select_penalty(
                lambda p: _stack_metric(stack_train, full_labels, p), LADDER)
            stacking = _stacking_fit(stack_train, full_labels, pen)
            fused = float((stacking(stack_test) == test_labels).mean())
            per_image = _inference_macs(g, grid, dimension, POOL_GRID,
                                        CLASSES)["total"]
            results[f"{name}_{g}"] = {
                "fused_accuracy": fused,
                "fusion_penalty_selected": float(pen),
                "specialist_head_penalty": head_penalty,
                "specialist_fit_rows": int(n_error),
                "encode_equivalent_macs": int((n_train + n_test)
                                              * per_image),
                "head_own_accuracy": float(
                    (np.argmax(test_scores, axis=1) == test_labels).mean()),
            }
            if name == "control":
                results[f"{name}_{g}"]["selection_macs"] = \
                    int(selection_macs)
                results[f"{name}_{g}"]["pool_encode_equivalent_macs"] = \
                    int(n_error * _inference_macs(
                        pool_atoms, grid, dimension, POOL_GRID,
                        CLASSES)["total"])
            del train_scores, test_scores, stack_train, stack_test, \
                err_codes
            print(f"    fused {fused:.6f} (static {static:.6f}, delta "
                  f"{fused - static:+.6f})", flush=True)
    if err_features is not None:
        del err_features

    # ---- gate -----------------------------------------------------------------
    gate_report: dict[str, Any] = {}
    fired = False
    for g in budgets:
        growth_fused = results[f"growth_{g}"]["fused_accuracy"]
        control_fused = (results[f"control_{g}"]["fused_accuracy"]
                         if f"control_{g}" in results else None)
        beats_static = growth_fused >= static + MARGIN
        beats_control = (control_fused is None or growth_fused > control_fused)
        passed = bool(beats_static and beats_control and not control_void)
        gate_report[str(g)] = {
            "growth_fused": growth_fused,
            "control_fused": control_fused,
            "control_void": control_void,
            "static": static,
            "beats_static": bool(beats_static),
            "beats_control": bool(beats_control),
            "passed": passed,
        }
        if not passed:
            fired = True
        print(f"  gate g={g}: beats_static={beats_static} "
              f"beats_control={beats_control} control_void={control_void}",
              flush=True)

    evidence.update({
        "anchors": anchors,
        "phase2": {
            "valid_frac": VALID_FRAC,
            "valid_seed": VALID_SEED,
            "penalty_ladder": LADDER,
            "n_train_rows": int(n_train),
            "n_test_rows": int(n_test),
            "static_base_accuracy": static,
        },
        "arms": results,
        "gate": {
            "registered": config["gate"]["registered"],
            "margin": MARGIN,
            "per_budget": gate_report,
            "fired": fired,
            "consequence": (config["gate"].get("consequence_fired", "fired")
                            if fired else config["gate"].get(
                                "consequence_passed", "passed")),
        },
        "runtime_seconds": round(time.time() - started, 2),
    })
    _write(output_dir, evidence)
    print(f"\nM156 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def _stack_metric(features: np.ndarray, labels: np.ndarray, penalty: float
                  ) -> float:
    """M143b train-protocol metric: fit on 80%, measure on the 20%."""
    n = len(labels)
    order = np.random.default_rng(VALID_SEED).permutation(n)
    cut = int(VALID_FRAC * n)
    ft, fv = order[:cut], order[cut:]
    predict = _stacking_fit(features[ft], labels[ft], penalty)
    return float((predict(features[fv]) == labels[fv]).mean())


def _write(output_dir: Path, evidence: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m156(args.config, args.output)


if __name__ == "__main__":
    main()
