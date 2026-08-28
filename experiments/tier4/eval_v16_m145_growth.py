"""M145 — residual-targeted growth on M143's fused system.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v22.md`` (section 9
M145, Remaining-milestones recipe; 15 Aug 2026). The cell design is
registered in full BEFORE any accuracy measurement:

Base system = the M143 sealed stack on the cached score matrices
(``v16/m143/scores.npz``): 6 x 512-atom A5 specialists + the global f6144
head, stacking over 7x345 scores, the M143 split (seed 33: fit 17,250 /
eval 17,250 of the sealed test rows), stacking penalty selected on the
M143 valid slice (valid_seed 55, frac 0.8) over {1,10,100,1000,10000}.

Anchors (t1-style, before any new number is trusted):

- a1: the static-fusion reproduction — recomputed from the SAME cached
  matrices with the SAME M143 phase-2 code, fused eval must reproduce
  0.1462608695652174 and the global arm 0.22510144927536233 (tol 1e-9).
- a2: the shared encode path is re-validated against the M143 sealed d0
  anchor: rebuild the M108 whitener + the domain-0 512-atom dictionary
  and reproduce own-domain 0.19357142857142856 (tol 0.002).
- a3: the growth dictionaries are nested prefixes of the [11,100]-seeded
  permutation of the global pool (g128 subset g256, asserted).

Error rows = fit-half rows the static fusion mispredicts at its selected
penalty. PREMISE GATE: ceil(n_error / (4*g)) >= 10 (the section 5.3
floor) at every budget, checked in-run before any fit — budgets are
{128, 256}; 512 atoms is registered infeasible (would need >= 20,480
error rows).

Arms per budget g (identical in everything but the dictionary):

- growth: dictionary = first g atoms of the global pool in the
  [11,100]-seeded permutation (a prefix of the f6144 dictionary); encode
  all test rows; ridge head (lambda 1.0) fit on the ERROR ROWS ONLY.
- control: dictionary = first g atoms of the M108 arm (c) blind-greedy
  order (group-OMP vs centred one-hot, ``select_discriminative`` on the
  fit-half rows — the E8 prior; GPU port with the M108 order-parity
  check); head fit on the SAME error rows.

Both append as arm 8; the stacking is re-solved with the same
valid-slice penalty protocol; the fused read is on the eval half.

GATE (per budget, eval half): growth_fused >= static_fused + 0.005 AND
growth_fused > control_fused. Control >= growth at any budget means the
gain is not residual targeting's and that budget's growth claim fails.

Ops ledger disclosed per arm: specialist encode MACs; the control
additionally its pool encode + OMP selection MACs. Growth's added ops vs
static fusion are disclosed, not matched away; growth-vs-control is the
matched-cost comparison.

Reproduce with::

    $env:GEODE_CACHE_DIR="F:\\geode-ml\\data\\cache"
    $env:HIP_VISIBLE_DEVICES="1"
    .\\.venv-rocm\\Scripts\\python.exe -m experiments.tier4.eval_v16_m145_growth
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
    _chunk_rows,
    _inference_macs,
    _score,
)
from experiments.tier4.eval_v15_m107_dense import _verify_pixel_identity
from experiments.tier4.eval_v16_a5_routed import _domain_candidates
from experiments.tier4.eval_v16_m108_dictionary import (
    _encode_block_device,
    _select_discriminative_gpu,
    _selection_encode_macs,
    _verify_device,
)
from experiments.tier4.eval_v16_m109_trunk import (
    _build_whitener_dictionary,
    _load_corpus,
)
from experiments.tier4.eval_v16_m143_integration import (
    _select_penalty,
    _split_indices,
    _stacking_fit,
)

import torch  # noqa: E402  (kept after constants to mirror the v16 layout)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v16"
                  / "m145_growth.json")
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m145_growth"

CLASSES = 345
DOMAINS = 6
A1_TOLERANCE = 1e-9
A2_TOLERANCE = 0.002
FLOOR = 10.0


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------
def _error_rows(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Row positions (into the given arrays) where the prediction is wrong."""
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    return np.flatnonzero(preds != labels)


def _floor_ok(n_error_rows: int, atoms: int, pool_grid: int,
              floor: float = FLOOR) -> bool:
    """ceil(n_error_rows / (pool_grid**2 * atoms)) >= floor."""
    width = pool_grid * pool_grid * atoms
    ratio = (int(n_error_rows) + width - 1) // width
    return float(ratio) >= float(floor)


def _growth_dictionaries(pool_permuted: np.ndarray,
                         budgets: list[int]) -> dict[int, np.ndarray]:
    """Nested prefixes of the seeded-permutation order of the pool."""
    return {int(g): np.asarray(pool_permuted[:g]) for g in budgets}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_m145(config_path: Path, output_dir: Path) -> dict[str, Any]:
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
    smoke_rows = int(config.get("_smoke_test_rows", 10 ** 9))

    torch.set_num_threads(int(config["numerics"]["torch_threads"]))
    torch.manual_seed(int(config["numerics"]["seed"]))
    configure_external_cache_environment()
    _verify_device(torch)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    batch = int(config["numerics"]["batch"])
    throttle = float(config["numerics"]["encode_throttle_seconds"])
    block = int(config["numerics"]["block"])

    evidence: dict[str, Any] = {
        "milestone": "M145",
        "cell": "residual-targeted growth on the M143 fused system",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "question": config["question"],
    }
    anchors: dict[str, Any] = {}

    # ---- corpus + sealed score matrices ---------------------------------
    print("loading corpus", flush=True)
    corpus, train_index, test_index = _load_corpus(config)
    size = int(config["corpus"]["image_size"])
    for split, idx in (("train", train_index), ("test", test_index)):
        _verify_pixel_identity(split, idx, corpus[f"{split}_images"], size,
                               int(config["corpus"]["pixel_identity_rows"]))

    scores_path = data_cache_root() / config["score_cache"]["cache_relpath"]
    scores_file = scores_path / "scores.npz"
    if not scores_file.exists():
        raise SystemExit(f"M145 premise failure: no score cache at {scores_file}")
    payload = np.load(scores_file, allow_pickle=False)
    specialist_scores = payload["specialist_scores"]  # (6, n, 345)
    global_scores = payload["global_scores"]          # (n, 345)
    test_labels = payload["test_labels"][:smoke_rows]
    test_domains = payload["test_domains"][:smoke_rows]
    n_test = len(test_labels)
    specialist_scores = specialist_scores[:, :n_test, :]
    global_scores = global_scores[:n_test, :]
    if specialist_scores.shape != (DOMAINS, n_test, CLASSES):
        raise SystemExit(f"M145 instrument failure: score cache shape "
                         f"{specialist_scores.shape} != {(DOMAINS, n_test, CLASSES)}")

    arm_scores = np.concatenate(
        [specialist_scores, global_scores[None, :, :]], axis=0)  # (7, n, 345)
    concat = np.concatenate(
        [specialist_scores.reshape(DOMAINS, n_test, CLASSES)
            .transpose(1, 0, 2).reshape(n_test, -1),
         global_scores], axis=1)  # (n, 2415)

    # ---- a1: static-fusion reproduction (the M143 sealed read) ----------
    split_seed = int(config["phase2"]["split_seed"])
    fit_idx, eval_idx = _split_indices(n_test, split_seed)
    valid_split = int(config["phase2"]["valid_frac"] * len(fit_idx))
    rng = np.random.default_rng(int(config["phase2"]["valid_seed"]))
    fit_order = rng.permutation(len(fit_idx))
    fit_train_idx = fit_idx[fit_order[:valid_split]]
    fit_valid_idx = fit_idx[fit_order[valid_split:]]
    ladder = [float(x) for x in config["phase2"]["penalty_ladder"]]
    print(f"phase2: fit {len(fit_idx)} / eval {len(eval_idx)} rows; "
          f"penalty ladder {ladder}", flush=True)

    def _stack_metric(concat_m, penalty):
        predict = _stacking_fit(concat_m[fit_train_idx],
                                test_labels[fit_train_idx], penalty)
        return float((predict(concat_m[fit_valid_idx])
                      == test_labels[fit_valid_idx]).mean())

    fusion_penalty, fusion_ladder_scores = _select_penalty(
        lambda p: _stack_metric(concat, p), ladder)
    stacking = _stacking_fit(concat[fit_idx], test_labels[fit_idx],
                             fusion_penalty)
    fused_preds = stacking(concat[eval_idx])
    eval_labels = test_labels[eval_idx]
    static_fused = float((fused_preds == eval_labels).mean())
    static_global = float(
        (np.argmax(global_scores[eval_idx], axis=1) == eval_labels).mean())
    print(f"  a1 static: fused {static_fused:.6f} global {static_global:.6f} "
          f"(penalty {fusion_penalty})", flush=True)

    sealed_fused = float(config["anchors"]["static_fused"])
    sealed_global = float(config["anchors"]["static_global"])
    anchors["a1_static"] = {
        "fused": {"measured": static_fused, "sealed": sealed_fused,
                  "delta": static_fused - sealed_fused},
        "global": {"measured": static_global, "sealed": sealed_global,
                   "delta": static_global - sealed_global},
        "tolerance": A1_TOLERANCE,
        "fusion_penalty_selected": fusion_penalty,
    }
    a1_ok = (abs(static_fused - sealed_fused) <= A1_TOLERANCE
             and abs(static_global - sealed_global) <= A1_TOLERANCE)
    if not a1_ok and not skip_anchors:
        evidence.update({"void": True,
                         "void_reason": "a1 static-fusion reproduction failed",
                         "anchors": anchors})
        _write(output_dir, evidence)
        return evidence

    # ---- error rows + floor premise -------------------------------------
    static_fit_preds = stacking(concat[fit_idx])
    err_pos = _error_rows(static_fit_preds, test_labels[fit_idx])
    error_rows = fit_idx[err_pos]
    n_error = len(error_rows)
    print(f"  error rows on the fit half: {n_error} of {len(fit_idx)}",
          flush=True)
    if n_error == 0:
        raise SystemExit(
            "M145 PREMISE FAILURE: the static fusion has zero errors on the "
            "fit half (the stacking interpolates its own rows at this cell "
            "size); a growth population of zero rows is meaningless and no "
            "growth head may be fitted on it. This gate is not waivable, "
            "including in smoke.")

    budgets = sorted(int(g) for g in config["sparse"]["growth_atoms"])
    pool_grid = int(config["sparse"]["pool_grid"])
    floor_report: dict[str, Any] = {"n_error_rows": n_error}
    for g in budgets:
        ok = _floor_ok(n_error, g, pool_grid)
        floor_report[str(g)] = {
            "floor_ok": ok,
            "rows_per_dim": (n_error + pool_grid * pool_grid * g - 1)
            // (pool_grid * pool_grid * g),
            "fitted_dims": pool_grid * pool_grid * g,
        }
        print(f"  floor g={g}: ok={ok} "
              f"({floor_report[str(g)]['rows_per_dim']} rows/dim)",
              flush=True)
        if not ok and not allow_floor:
            raise SystemExit(
                f"M145 PREMISE GATE: budget {g} is below the section 5.3 "
                f"floor ({floor_report[str(g)]['rows_per_dim']} rows per "
                f"fitted dimension < {FLOOR}). The budget is void, not "
                "negative; nothing is measured below the floor.")
    evidence["premise"] = {"floor": floor_report}

    # ---- whitener + global pool in permutation order ---------------------
    print("building M108 whitener + global pool", flush=True)
    pool_size = int(config["sparse"]["pool_size"])
    sparse_tmp = dict(config["sparse"])
    sparse_tmp["atoms"] = pool_size
    config_tmp = dict(config)
    config_tmp["sparse"] = sparse_tmp
    whitener, dictionary, grid, dimension, _ = _build_whitener_dictionary(
        config_tmp, corpus)
    if dictionary.shape != (pool_size, dimension):
        raise SystemExit(f"M145 instrument failure: pool shape "
                         f"{dictionary.shape} != {(pool_size, dimension)}")
    growth_dicts = _growth_dictionaries(dictionary, budgets)
    anchors["a3_prefix"] = {
        "ok": bool(np.array_equal(growth_dicts[budgets[0]],
                                  growth_dicts[budgets[-1]][:budgets[0]])),
        "pool_size": pool_size,
    }
    if not anchors["a3_prefix"]["ok"]:
        raise SystemExit("M145 instrument failure: growth dictionaries are "
                         "not nested prefixes of the permutation order")

    # ---- a2: specialist encode path vs the M143 sealed d0 anchor ---------
    if not skip_anchors:
        print("a2 anchor: domain-0 512-atom specialist reproduction",
              flush=True)
        cand = _domain_candidates(corpus, 0, whitener)
        order = np.random.default_rng([11, 100]).permutation(len(cand))
        d0_dict = cand[order[:512]]
        table = torch.from_numpy(np.ascontiguousarray(d0_dict)
                                 ).to(torch.float32).to(device)
        rows_d = np.where(corpus["train_domains"] == 0)[0]
        acc = RidgeAccumulator(512 * pool_grid * pool_grid, CLASSES)
        for start in range(0, len(rows_d), batch):
            take = rows_d[start:start + batch]
            block_t = _encode_block_device(corpus["train_images"][take],
                                           table, whitener, pool_grid)
            acc.add(block_t, corpus["train_labels"][take])
            if throttle > 0:
                time.sleep(throttle)
        weights = acc.solve_many([1.0])[1.0]
        standardise = acc.standardiser()
        own_rows = np.where(corpus["test_domains"] == 0)[0]
        own_rows = own_rows[own_rows < n_test]
        hits = 0
        for start in range(0, len(own_rows), batch):
            take = own_rows[start:start + batch]
            block_t = _encode_block_device(corpus["test_images"][take],
                                           table, whitener, pool_grid)
            hits += int(_score(weights, standardise(block_t),
                               corpus["test_labels"][take]).sum())
            if throttle > 0:
                time.sleep(throttle)
        own_acc = hits / len(own_rows)
        del table
        torch.cuda.empty_cache()
        sealed_d0 = float(config["anchors"]["d0_own_domain"])
        anchors["a2_d0"] = {"measured": own_acc, "sealed": sealed_d0,
                            "delta": own_acc - sealed_d0,
                            "tolerance": A2_TOLERANCE}
        print(f"  a2 d0 own-domain {own_acc:.6f} vs sealed {sealed_d0:.6f} "
              f"(delta {own_acc - sealed_d0:+.6f})", flush=True)
        if abs(own_acc - sealed_d0) > A2_TOLERANCE:
            evidence.update({"void": True,
                             "void_reason": "a2 specialist-path anchor failed",
                             "anchors": anchors})
            _write(output_dir, evidence)
            return evidence

    # ---- control selection (M108 blind greedy on the fit-half rows) ------
    ctrl_order: np.ndarray | None = None
    selection_macs = 0
    pool_encode_macs = 0
    if not skip_control:
        print("control: pool encode of the fit-half rows", flush=True)
        pool_table = torch.from_numpy(
            np.ascontiguousarray(dictionary)).to(torch.float32).to(device)
        sel_rows = fit_idx
        step = min(4096, _chunk_rows(pool_size, grid, len(sel_rows)))
        selection_features = np.empty(
            (len(sel_rows), pool_grid * pool_grid * pool_size),
            dtype=np.float32)
        for start in range(0, len(sel_rows), step):
            take = sel_rows[start:start + step]
            selection_features[start:start + step] = _encode_block_device(
                corpus["test_images"][take], pool_table, whitener, pool_grid)
            if throttle > 0:
                time.sleep(throttle)
        del pool_table
        torch.cuda.empty_cache()
        pool_encode_macs = _selection_encode_macs(
            len(sel_rows), pool_size, grid, dimension, CLASSES)

        print("control: OMP order-parity check (numpy vs GPU)", flush=True)
        parity_rows = int(config["control"]["parity_subset_rows"])
        parity_budget = int(config["control"]["parity_budget"])
        parity_features = selection_features[:parity_rows]
        parity_labels = test_labels[sel_rows[:parity_rows]]
        reference_order, _ = select_discriminative(
            parity_features, parity_labels, pool_size, parity_budget,
            pool_grid)
        gpu_order, _ = _select_discriminative_gpu(
            parity_features, parity_labels, pool_size, parity_budget,
            pool_grid, device)
        parity_ok = bool(np.array_equal(reference_order, gpu_order))
        print(f"  order parity: {'passed' if parity_ok else 'FAILED'}",
              flush=True)

        print(f"control: full OMP selection (budget {max(budgets)})",
              flush=True)
        if parity_ok:
            ctrl_order, selection_macs = _select_discriminative_gpu(
                selection_features, test_labels[sel_rows], pool_size,
                max(budgets), pool_grid, device)
            backend = "cuda"
        else:
            ctrl_order, selection_macs = select_discriminative(
                selection_features, test_labels[sel_rows], pool_size,
                max(budgets), pool_grid)
            backend = "cpu"
        del selection_features
        torch.cuda.empty_cache()
        evidence["control_selection"] = {
            "backend": backend,
            "selection_rows": int(len(sel_rows)),
            "parity_checked": parity_ok,
            "pool_encode_macs": pool_encode_macs,
            "selection_macs": int(selection_macs),
        }
    else:
        ctrl_order = np.arange(max(budgets), dtype=np.int64)

    # ---- growth + control arms per budget --------------------------------
    results: dict[str, Any] = {}
    head_penalty = float(config["sparse"]["specialist_head_penalty"])
    for g in budgets:
        print(f"\nbudget g={g}", flush=True)
        arm_dicts = {"growth": growth_dicts[g]}
        if not skip_control:
            arm_dicts["control"] = dictionary[ctrl_order[:g]]
        for name, arm_dict in arm_dicts.items():
            print(f"  arm {name}: encode {n_test} test rows", flush=True)
            table = torch.from_numpy(np.ascontiguousarray(arm_dict)
                                     ).to(torch.float32).to(device)
            codes = np.empty((n_test, pool_grid * pool_grid * g),
                             dtype=np.float32)
            step = min(4096, _chunk_rows(g, grid, n_test))
            for start in range(0, n_test, step):
                stop = min(start + step, n_test)
                codes[start:stop] = _encode_block_device(
                    corpus["test_images"][start:stop], table, whitener,
                    pool_grid)
                if throttle > 0:
                    time.sleep(throttle)
            del table
            torch.cuda.empty_cache()

            print(f"  arm {name}: ridge head on {n_error} error rows",
                  flush=True)
            acc = RidgeAccumulator(pool_grid * pool_grid * g, CLASSES)
            for start in range(0, n_error, block):
                stop = min(start + block, n_error)
                acc.add(codes[error_rows[start:stop]],
                        test_labels[error_rows[start:stop]])
            weights = acc.solve_many([head_penalty])[head_penalty]
            standardise = acc.standardiser()
            arm_scores_new = np.empty((n_test, CLASSES), dtype=np.float32)
            for start in range(0, n_test, block):
                stop = min(start + block, n_test)
                xs = standardise(codes[start:stop])
                arm_scores_new[start:stop] = xs @ weights[:-1] + weights[-1]

            concat8 = np.concatenate([concat, arm_scores_new], axis=1)
            fusion_p, ladder_p = _select_penalty(
                lambda p: _stack_metric(concat8, p), ladder)
            stacking8 = _stacking_fit(concat8[fit_idx],
                                      test_labels[fit_idx], fusion_p)
            fused8 = float(
                (stacking8(concat8[eval_idx]) == eval_labels).mean())
            encode_macs = n_test * int(_inference_macs(
                g, grid, dimension, pool_grid, CLASSES)["total"])
            results[f"{name}_{g}"] = {
                "fused_accuracy": fused8,
                "fusion_penalty_selected": fusion_p,
                "fusion_ladder_scores": ladder_p,
                "specialist_head_penalty": head_penalty,
                "specialist_fit_rows": int(acc.rows),
                "encode_macs": encode_macs,
                "selection_macs": (pool_encode_macs + int(selection_macs)
                                   if name == "control" else 0),
                "head_own_accuracy": float(
                    (np.argmax(arm_scores_new, axis=1) == test_labels).mean()),
            }
            del codes, arm_scores_new, concat8
            print(f"    fused {fused8:.6f} (static {static_fused:.6f}, "
                  f"delta {fused8 - static_fused:+.6f})", flush=True)

    # ---- gate -------------------------------------------------------------
    margin = float(config["gate"]["margin"])
    gate_report: dict[str, Any] = {}
    fired = False
    for g in budgets:
        growth_fused = results[f"growth_{g}"]["fused_accuracy"]
        control_fused = (results[f"control_{g}"]["fused_accuracy"]
                         if f"control_{g}" in results else None)
        beats_static = growth_fused >= static_fused + margin
        beats_control = (control_fused is None
                         or growth_fused > control_fused)
        gate_report[str(g)] = {
            "growth_fused": growth_fused,
            "control_fused": control_fused,
            "static_fused": static_fused,
            "beats_static": bool(beats_static),
            "beats_control": bool(beats_control),
            "passed": bool(beats_static and beats_control),
        }
        if not (beats_static and beats_control):
            fired = True
        print(f"  gate g={g}: beats_static={beats_static} "
              f"beats_control={beats_control}", flush=True)

    evidence.update({
        "anchors": anchors,
        "phase2": {
            "split_seed": split_seed,
            "fit_rows": int(len(fit_idx)),
            "eval_rows": int(len(eval_idx)),
            "static_fusion_penalty_selected": fusion_penalty,
            "static_fusion_ladder_scores": fusion_ladder_scores,
            "static_fused_accuracy": static_fused,
            "static_global_accuracy": static_global,
            "n_error_rows": int(n_error),
        },
        "arms": results,
        "score_cache": {"path": str(scores_file),
                        "relpath": config["score_cache"]["cache_relpath"]},
        "gate": {
            "registered": config["gate"]["registered"],
            "margin": margin,
            "per_budget": gate_report,
            "fired": fired,
            "consequence": (config["gate"].get(
                "consequence_fired", "fired") if fired
                else config["gate"].get("consequence_passed", "passed")),
        },
        "runtime_seconds": round(time.time() - started, 2),
    })
    _write(output_dir, evidence)
    print(f"\nM145 complete -> {output_dir / 'evidence.json'}", flush=True)
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
    run_m145(args.config, args.output)


if __name__ == "__main__":
    main()
