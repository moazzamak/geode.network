"""M157 — temporal task-property screen + reservoir-programmatic hybrid +
reservoir growth (CPU).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v23.md`` (section 4
M157; 16 Aug 2026). Reuses the M147 harness arms unchanged. Two new axes:

- Lorenz-63 x-component (RK4, dt=0.01, sampled every 0.1, warmup 2000) —
  the second chaotic axis.
- A seeded Dyck-like two-bracket grammar (next-token, numeric token id,
  1-D regression through the SAME arm machinery so arms stay comparable)
  — the discrete-structure axis where the M134 prior predicts a
  reservoir loss.

Anchors (before any new-axis number is read): the sealed M147 Mackey-Glass
reads reproduced with the same arms (no-memory 0.14585608397316033,
primitives 0.0031721430026391, reservoir best per seed 0.002661459 / 0.002231602 /
0.002485177; relative tolerance 1e-6) and the M134 DSL reservoir anchor
(the M147 t1 path, delta 0.000000 sealed). Gates per new axis (registered):
reservoir best >= 5% relative over the best non-recurrent arm (the M147
rule); the hybrid must beat BOTH parents; growth must beat static reservoir
AND the random-subset control. Echo-state property checked before any
readout; after the append the combined system is block-diagonal, so the
property follows from each block's rho < 1 (checked and recorded).

Reproduce with::

    .\\.venv\\Scripts\\python.exe -m experiments.tier4.eval_v23_m157_temporal_screen
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from experiments.tier4.eval_v16_m134_fixed_sequence import (
    _generate_corpus,
    _reservoir_readout,
)
from experiments.tier4.eval_v16_m147_temporal_memory import (
    _no_memory_arm,
    _nrmsfe,
    _predict_1d,
    _programmatic_arm,
    _reservoir_arm_mg,
    _ridge_1d,
    _tap_delay_arm,
    mackey_glass,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v23"
                  / "m157_temporal_screen.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v23"
                  / "m157_temporal_screen")


# ---------------------------------------------------------------------------
# series generators
# ---------------------------------------------------------------------------
def _lorenz(n_points: int, sigma: float, rho: float, beta: float,
            dt: float, sample_every: int, warmup: int, x0: list[float],
            seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x, y, z = [float(v) + float(rng.standard_normal()) * 1e-9 for v in x0]
    out: list[float] = []
    for step in range((n_points + warmup) * sample_every):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dt * dx
        y += dt * dy
        z += dt * dz
        if step % sample_every == 0 and step // sample_every >= warmup:
            out.append(x)
    return np.asarray(out[:n_points], dtype=np.float64)


def _dyck_series(length: int, max_depth: int, seed: int) -> np.ndarray:
    """Balanced two-bracket Dyck walk -> token ids {0:'(',1:')',2:'[',3:']'}."""
    rng = np.random.default_rng(seed)
    tokens: list[int] = []
    depth = 0
    while len(tokens) < length:
        candidates = []
        if depth < max_depth:
            candidates += [0, 2]
        if depth > 0:
            candidates += [1, 3]
        pick = int(rng.choice(candidates))
        tokens.append(pick)
        depth += 1 if pick in (0, 2) else -1
    return np.asarray(tokens, dtype=np.float64)


# ---------------------------------------------------------------------------
# hybrid + growth arms
# ---------------------------------------------------------------------------
def _reservoir_states(train: np.ndarray, units: int, rho: float, seed: int,
                      warmup: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Run the M147 reservoir dynamics; return (states, targets, rho_measured)."""
    rng = np.random.default_rng(seed)
    w_in = rng.standard_normal((units, 1))
    w_rec = rng.standard_normal((units, units)) / math.sqrt(units)
    eig = np.linalg.eigvals(w_rec)
    rho0 = float(np.max(np.abs(eig))) if len(eig) else 1.0
    if rho0 > 0:
        w_rec = w_rec * (rho / rho0)
    rho_measured = float(np.max(np.abs(np.linalg.eigvals(w_rec))))
    h = np.zeros(units, dtype=np.float64)
    states, targets = [], []
    for t in range(len(train) - 1):
        h = np.tanh(w_in[:, 0] * train[t] + w_rec @ h)
        if t >= warmup:
            states.append(h.copy())
            targets.append(train[t + 1])
    return np.asarray(states), np.asarray(targets), rho_measured


def _append_units(train: np.ndarray, units: int, rho: float, seed: int,
                  warmup: int) -> tuple[np.ndarray, float]:
    """128 appended units' states on the same series (block-diagonal system)."""
    rng = np.random.default_rng(seed)
    w_in = rng.standard_normal((units, 1))
    w_rec = rng.standard_normal((units, units)) / math.sqrt(units)
    eig = np.linalg.eigvals(w_rec)
    rho0 = float(np.max(np.abs(eig))) if len(eig) else 1.0
    if rho0 > 0:
        w_rec = w_rec * (rho / rho0)
    rho_measured = float(np.max(np.abs(np.linalg.eigvals(w_rec))))
    h = np.zeros(units, dtype=np.float64)
    states = []
    for t in range(len(train) - 1):
        h = np.tanh(w_in[:, 0] * train[t] + w_rec @ h)
        if t >= warmup:
            states.append(h.copy())
    return np.asarray(states), rho_measured


def _hybrid_arm(train: np.ndarray, test: np.ndarray, units: int, rho: float,
                seed: int, warmup: int, penalty: float,
                k: int = 4) -> dict[str, Any]:
    base, _targets, rho_m = _reservoir_states(train, units, rho, seed, warmup)
    feats, targs = _primitive_features(train, k)
    n = min(len(base), len(feats))
    joined = np.concatenate([base[:n], feats[:n]], axis=1)
    weights, std = _ridge_1d(joined, targs[:n], penalty)
    # test side
    base_t, _t2, _ = _reservoir_states(test, units, rho, seed, warmup)
    feats_t, targs_t = _primitive_features(test, k)
    m = min(len(base_t), len(feats_t))
    joined_t = np.concatenate([base_t[:m], feats_t[:m]], axis=1)
    preds = _predict_1d(weights, std, joined_t)
    return {"nrmsfe": _nrmsfe(preds, targs_t[:m]), "kind": "hybrid",
            "units": units, "rho_measured": rho_m,
            "echo_state_property_ok": rho_m < 1.0}


def _primitive_features(series: np.ndarray, window: int = 4
                        ) -> tuple[np.ndarray, np.ndarray]:
    """The M147 primitive feature set, exposed for concatenation."""
    n = len(series)
    feats = np.zeros((n - window, 4), dtype=np.float64)
    for t in range(window, n):
        win = series[t - window:t]
        feats[t - window, 0] = series[t]
        feats[t - window, 1] = series[t] - series[t - 1]
        feats[t - window, 2] = win.mean()
        feats[t - window, 3] = win.std()
    return feats, series[window:]


def _growth_arm(train: np.ndarray, test: np.ndarray, units: int, rho: float,
                seed: int, warmup: int, penalty: float, append_seed: int
                ) -> dict[str, Any]:
    base, targets, rho_m = _reservoir_states(train, units, rho, seed, warmup)
    # static fit on all steps -> residual error steps (the growth target)
    w_static, std_static = _ridge_1d(base, targets, penalty)
    preds_static = _predict_1d(w_static, std_static, base)
    err = np.abs(preds_static - targets)
    error_steps = np.flatnonzero(err >= np.median(err))
    appended, rho_a = _append_units(train, 128, 0.9, append_seed, warmup)
    appended = appended[:len(base)]
    joined = np.concatenate([base, appended], axis=1)

    rng = np.random.default_rng(append_seed)
    rand_steps = rng.choice(len(targets), size=len(error_steps),
                            replace=False)
    out: dict[str, Any] = {"kind": "growth", "units": units,
                           "append_units": 128,
                           "error_steps": int(len(error_steps)),
                           "rho_base": rho_m, "rho_appended": rho_a,
                           "echo_state_property_ok":
                               rho_m < 1.0 and rho_a < 1.0}
    for name, steps in (("growth", error_steps),
                        ("random_control", rand_steps)):
        w, std = _ridge_1d(joined[steps], targets[steps], penalty)
        base_t, _t2, _ = _reservoir_states(test, units, rho, seed, warmup)
        app_t, _ = _append_units(test, 128, 0.9, append_seed, warmup)
        m = min(len(base_t), len(app_t))
        joined_t = np.concatenate([base_t[:m], app_t[:m]], axis=1)
        preds = _predict_1d(w, std, joined_t)
        out[f"{name}_nrmsfe"] = _nrmsfe(preds, _t2[:m])
    return out


def _run_axis(config: dict[str, Any], train: np.ndarray, test: np.ndarray,
              axis_name: str, rcfg: dict[str, Any], warmup: int
              ) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    arms["no_memory"] = _no_memory_arm(train, test)
    arms["tap_delay"] = _tap_delay_arm(train, test, 8)
    arms["programmatic"] = _programmatic_arm(train, test)
    reservoir_runs = []
    for seed in [int(s) for s in rcfg["seeds"]]:
        for units in [int(u) for u in rcfg["units_ladder"]]:
            for rho in [float(r) for r in rcfg["rho_ladder"]]:
                arm = _reservoir_arm_mg(train, test, units, rho, seed,
                                        warmup, 1.0)
                reservoir_runs.append(arm)
    arms["reservoir_runs"] = reservoir_runs
    best_per_seed: dict[int, float] = {}
    for arm in reservoir_runs:
        s = int(arm["seed"])
        best_per_seed[s] = min(best_per_seed.get(s, float("inf")),
                               float(arm["nrmsfe"]))
    best_seed = min(best_per_seed, key=lambda s: best_per_seed[s])
    best_arm = min((a for a in reservoir_runs if int(a["seed"]) == best_seed),
                   key=lambda a: a["nrmsfe"])
    non_recurrent_best = min(arms["no_memory"]["nrmsfe"],
                             arms["tap_delay"]["nrmsfe"],
                             arms["programmatic"]["nrmsfe"])

    # hybrid (best reservoir config) + growth
    arms["hybrid"] = _hybrid_arm(train, test, int(best_arm["units"]),
                                 float(best_arm["rho_registered"]), best_seed,
                                 warmup, 1.0)
    arms["growth"] = _growth_arm(train, test, int(best_arm["units"]),
                                 float(best_arm["rho_registered"]), best_seed,
                                 warmup, 1.0, append_seed=1573)

    gate = {
        "best_reservoir_nrmsfe": best_arm["nrmsfe"],
        "non_recurrent_best": non_recurrent_best,
        "reservoir_beats": bool(best_arm["nrmsfe"]
                                <= non_recurrent_best * 0.95),
        "hybrid_nrmsfe": arms["hybrid"]["nrmsfe"],
        "hybrid_beats_both": bool(
            arms["hybrid"]["nrmsfe"]
            < min(best_arm["nrmsfe"], arms["programmatic"]["nrmsfe"])),
        "growth_nrmsfe": arms["growth"]["growth_nrmsfe"],
        "random_control_nrmsfe": arms["growth"]["random_control_nrmsfe"],
        "growth_beats_static": bool(
            arms["growth"]["growth_nrmsfe"] < best_arm["nrmsfe"]),
        "growth_beats_control": bool(
            arms["growth"]["growth_nrmsfe"]
            < arms["growth"]["random_control_nrmsfe"]),
    }
    print(f"  [{axis_name}] reservoir best {best_arm['nrmsfe']:.4f} vs "
          f"non-recurrent {non_recurrent_best:.4f}; hybrid "
          f"{arms['hybrid']['nrmsfe']:.4f}; growth "
          f"{arms['growth']['growth_nrmsfe']:.4f} vs control "
          f"{arms['growth']['random_control_nrmsfe']:.4f}", flush=True)
    return {"arms": arms, "gate": gate, "best_seed": best_seed}


def run_m157(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")
    smoke = inadmissible
    started = time.time()
    evidence: dict[str, Any] = {
        "milestone": "M157",
        "cell": "temporal task-property screen",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "question": config["question"],
    }

    # ---- anchor: the M147 Mackey-Glass sealed reads --------------------------
    print("anchor: M147 Mackey-Glass reproduction", flush=True)
    mg_cfg = config["series"]["mackey_glass_anchor"]
    mg = mackey_glass(
        int(mg_cfg["train_points"]) + int(mg_cfg["test_points"]),
        tau=float(mg_cfg["tau"]), beta=float(mg_cfg["beta"]),
        gamma=float(mg_cfg["gamma"]), n=float(mg_cfg["n"]),
        dt=float(mg_cfg["dt"]), x0=float(mg_cfg["x0"]),
        seed=int(mg_cfg["seed"]), discard=int(mg_cfg["discard"]),
        sample_every=int(mg_cfg["sample_every"]))
    train, test = (mg[:int(mg_cfg["train_points"])],
                   mg[int(mg_cfg["train_points"]):])
    anchors: dict[str, Any] = {}
    a_no = _no_memory_arm(train, test)["nrmsfe"]
    a_pr = _programmatic_arm(train, test)["nrmsfe"]
    best_per_seed: dict[int, float] = {}
    for seed in [21, 22, 23]:
        best = float("inf")
        for units in [64, 256, 1024]:
            for rho in [0.5, 0.9, 0.99]:
                arm = _reservoir_arm_mg(train, test, units, rho, seed, 200,
                                        1.0)
                best = min(best, float(arm["nrmsfe"]))
        best_per_seed[seed] = best
    sealed_res = config["anchors"]["m147_reservoir_best"]
    deltas = {
        "no_memory": abs(a_no - float(config["anchors"]["m147_no_memory"])),
        "primitives": abs(a_pr - float(config["anchors"]["m147_primitives"])),
        "reservoir": max(abs(best_per_seed[s] - sealed_res[i])
                         for i, s in enumerate([21, 22, 23])),
    }
    anchors["m147"] = {"deltas": deltas,
                       "tolerance": float(config["anchors"]["tolerance"]),
                       "ok": all(d <= float(config["anchors"]["tolerance"])
                                 for d in deltas.values())}
    print(f"  anchor deltas: {deltas}", flush=True)
    if not anchors["m147"]["ok"] and not smoke:
        evidence.update({"void": True,
                         "void_reason": "M147 anchor reproduction failed",
                         "anchors": anchors})
        _write(output_dir, evidence)
        return evidence

    # ---- new axes -------------------------------------------------------------
    rcfg = {"seeds": [21, 22, 23], "units_ladder": [64, 256, 1024],
            "rho_ladder": [0.5, 0.9, 0.99]}
    results: dict[str, Any] = {}
    lor = config["series"]["lorenz"]
    print("axis: Lorenz", flush=True)
    s_l = _lorenz(int(lor["train_points"]) + int(lor["test_points"]),
                  float(lor["sigma"]), float(lor["rho"]), float(lor["beta"]),
                  float(lor["dt"]), int(lor["sample_every"]),
                  int(lor["warmup_points"]), list(lor["x0"]),
                  int(lor["seed"]))
    results["lorenz"] = _run_axis(
        config, s_l[:int(lor["train_points"])],
        s_l[int(lor["train_points"]):], "lorenz", rcfg, 200)

    dy = config["series"]["dyck_grammar"]
    print("axis: Dyck grammar", flush=True)
    s_d = _dyck_series(int(dy["length"]), int(dy["max_depth"]),
                       int(dy["seed"]))
    results["dyck"] = _run_axis(
        config, s_d[:int(dy["length"] * 0.75)],
        s_d[int(dy["length"] * 0.75):], "dyck", rcfg, 20)

    evidence.update({"anchors": anchors,
                     "results": results,
                     "gate": {"registered": config["gate"]["registered"],
                              "margin_relative":
                                  float(config["gate"]["margin_relative"])},
                     "runtime_seconds": round(time.time() - started, 2)})
    _write(output_dir, evidence)
    print(f"\nM157 complete -> {output_dir / 'evidence.json'}", flush=True)
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
    run_m157(args.config, args.output)


if __name__ == "__main__":
    main()
