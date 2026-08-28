"""M147 — temporal-memory screen: reservoir vs additive tap-delay vs programmatic.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v22.md`` (section 6
Phase T, section 9 M147, 14 Aug 2026).

Question (registered before measurement). On one-step-ahead prediction of a
chaotic series (Mackey-Glass), does a fixed random reservoir (delay + feedback)
earn its recurrence over (a) a no-memory ridge baseline, (b) the purely
additive tap-delay line (concatenated last-k values + ridge), and (c)
hand-written programmatic primitives? The reservoir's loss on the DSL
next-token task is already sealed (M134: r128 ppl 29.72 / r512 27.54 vs count
w4 3.32) and is carried as a registered PRIOR on that axis only; this screen
re-tests on the chaotic-series axis where that prior does not bind (rule 2).

Arms (all registered before measurement):
- no_memory: ridge on x(t) only.
- tap_delay(k): ridge on [x(t), ..., x(t-k+1)], k in {2, 4, 8}; best reported.
- programmatic: zero-parameter rules (extrapolator x_hat = 2x(t)-x(t-1); EWMA)
  plus ridge on hand-written primitive features [x(t), delta, rolling mean,
  rolling std]; best of the three reported. No feedback anywhere.
- reservoir(units, rho): fixed random W_in/W_rec with W_rec scaled to spectral
  radius rho < 1 (echo-state property verified numerically and recorded),
  tanh state, warmup discarded, ridge readout. Multi-seed.
- mlp_baseline: a tiny gradient-trained MLP (the trained measuring stick).

t1 anchor: the M134 DSL reservoir (r128) is reproduced exactly via the sealed
M134 code paths; ppl must match the sealed 29.7193 within tolerance.

Gates (kill switches, registered): the reservoir's best test NRMSE must beat
the best of the three non-recurrent arms by the registered relative margin on
>= half the seeds, else M147 closes as a scoped negative on this axis.

Reproduce with::

    .\\.venv\\Scripts\\python.exe -m experiments.tier4.eval_v16_m147_temporal_memory
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
    write_canonical_json,
)
from experiments.tier4.eval_v16_m131_additive_next_token import _generate_corpus
from experiments.tier4.eval_v16_m134_fixed_sequence import _reservoir_readout

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v16" / "m147_temporal_memory.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v16" / "m147_temporal_memory"


# ---------------------------------------------------------------------------
# Mackey-Glass series (fixed registered parameters, RK4 integration)
# ---------------------------------------------------------------------------

def mackey_glass(
    length: int,
    tau: float = 17.0,
    beta: float = 0.2,
    gamma: float = 0.1,
    n: float = 10.0,
    dt: float = 0.1,
    x0: float = 1.2,
    seed: int = 7,
    discard: int = 1000,
    sample_every: int = 1,
) -> np.ndarray:
    """Sample *length* points of the continuous-time Mackey-Glass delay
    equation, integrated with RK4 at step *dt*, keeping every
    *sample_every*-th step (tau = 17 time units, sample_every = 10 -> 1.0
    time-unit resolution, the canonical chaotic benchmark).

    The delay ring holds tau/dt + 1 slots: with exactly tau/dt slots the
    delayed index would wrap to the current value and the DDE would collapse
    to a stable ODE (registered defect, caught by the premise check).
    """
    rng = np.random.default_rng(seed)
    tau = float(tau)
    tau_slots = int(round(tau / dt))
    ring_len = tau_slots + 1
    ring = np.empty(ring_len, dtype=np.float64)
    ring[:] = x0 + 0.05 * rng.standard_normal(ring_len)

    def deriv(x: float, x_tau: float) -> float:
        return beta * x_tau / (1.0 + x_tau ** n) - gamma * x

    steps = discard + length * int(sample_every)
    points: list[float] = []
    idx = 0
    for step in range(steps):
        x = ring[idx]
        x_tau = ring[(idx - tau_slots) % ring_len]
        k1 = deriv(x, x_tau)
        k2 = deriv(x + 0.5 * dt * k1, x_tau)
        k3 = deriv(x + 0.5 * dt * k2, x_tau)
        k4 = deriv(x + dt * k3, x_tau)
        x_new = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        idx = (idx + 1) % ring_len
        ring[idx] = x_new
        if step >= discard and (step - discard) % int(sample_every) == 0:
            points.append(x_new)
    return np.asarray(points, dtype=np.float64)


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def _ridge_1d(features: np.ndarray, targets: np.ndarray, penalty: float = 1.0):
    """Closed-form ridge for regression: standardise, solve, return (w, standardiser)."""
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    centre = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-12] = 1.0
    xs = (features - centre) / scale
    d = xs.shape[1]
    gram = xs.T @ xs
    cross = xs.T @ targets
    w = np.linalg.solve(gram + penalty * np.eye(d), cross)
    b = float(targets.mean())

    def standardise(block: np.ndarray) -> np.ndarray:
        return (np.asarray(block, dtype=np.float64) - centre) / scale

    return np.concatenate([w, [b]]), standardise


def _predict_1d(weights: np.ndarray, standardise, features: np.ndarray) -> np.ndarray:
    xs = standardise(features)
    return (xs @ weights[:-1] + weights[-1]).reshape(-1)


def _nrmsfe(pred: np.ndarray, target: np.ndarray) -> float:
    """Normalised root-mean-square forecast error (RMSE / std(target))."""
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    return rmse / float(np.std(target))


def _delay_matrix(series: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Features [x(t),...,x(t-k+1)] with target x(t+1)."""
    n = len(series)
    features = np.empty((n - k, k), dtype=np.float64)
    for j in range(k):
        features[:, j] = series[k - 1 - j: n - 1 - j]
    targets = series[k:]
    return features, targets


def _no_memory_arm(train: np.ndarray, test: np.ndarray) -> dict[str, Any]:
    ftr, tgt = train[:-1, None], train[1:]
    weights, standardise = _ridge_1d(ftr, tgt)
    pred = _predict_1d(weights, standardise, test[:-1, None])
    return {"nrmsfe": _nrmsfe(pred, test[1:]), "kind": "no_memory"}


def _tap_delay_arm(train: np.ndarray, test: np.ndarray, k: int) -> dict[str, Any]:
    ftr, tgt = _delay_matrix(train, k)
    weights, standardise = _ridge_1d(ftr, tgt)
    tftr, ttgt = _delay_matrix(test, k)
    pred = _predict_1d(weights, standardise, tftr)
    return {"nrmsfe": _nrmsfe(pred, ttgt), "k": k, "kind": "tap_delay"}


def _programmatic_arm(train: np.ndarray, test: np.ndarray) -> dict[str, Any]:
    """Zero-parameter rules + ridge on hand-written primitive features."""
    results: dict[str, float] = {}

    # (i) linear extrapolation: x_hat(t+1) = 2x(t) - x(t-1)
    ext = 2.0 * test[1:-1] - test[:-2]
    results["extrapolator"] = _nrmsfe(ext, test[2:])

    # (ii) EWMA of past values (alpha registered = 0.3)
    alpha = 0.3
    ewma = test[0]
    preds: list[float] = []
    for x in test[:-1]:
        ewma = alpha * x + (1.0 - alpha) * ewma
        preds.append(ewma)
    results["ewma"] = _nrmsfe(np.asarray(preds), test[1:])

    # (iii) ridge on hand-written primitive features: x, delta, mean4, std4
    def primitives(series: np.ndarray, window: int = 4):
        n = len(series)
        feats = np.empty((n - window, 4), dtype=np.float64)
        for t in range(window, n):
            win = series[t - window: t]
            feats[t - window, 0] = series[t]
            feats[t - window, 1] = series[t] - series[t - 1]
            feats[t - window, 2] = float(np.mean(win))
            feats[t - window, 3] = float(np.std(win))
        return feats, series[window:]

    ptr, ptg = primitives(train)
    weights, standardise = _ridge_1d(ptr, ptg)
    pte, ptt = primitives(test)
    results["primitives_ridge"] = _nrmsfe(
        _predict_1d(weights, standardise, pte), ptt)

    best_name = min(results, key=results.get)
    return {"nrmsfe": results[best_name], "kind": "programmatic",
            "winner": best_name, "all": results}


def _reservoir_arm_mg(
    train: np.ndarray, test: np.ndarray, units: int, rho: float, seed: int,
    warmup: int, penalty: float,
) -> dict[str, Any]:
    """Fixed random reservoir (tanh, spectral radius rho) + ridge readout."""
    rng = np.random.default_rng(seed)
    w_in = rng.standard_normal((units, 1)) / math.sqrt(1)
    w_rec = rng.standard_normal((units, units)) / math.sqrt(units)
    eig = np.linalg.eigvals(w_rec)
    rho0 = float(np.max(np.abs(eig))) if len(eig) else 1.0
    if rho0 > 0:
        w_rec = w_rec * (rho / rho0)
    eig_scaled = np.linalg.eigvals(w_rec)
    rho_measured = float(np.max(np.abs(eig_scaled)))

    h = np.zeros(units, dtype=np.float64)
    states: list[np.ndarray] = []
    targets: list[float] = []
    for t in range(len(train) - 1):
        h = np.tanh(w_in[:, 0] * train[t] + w_rec @ h)
        if t >= warmup:
            states.append(h.copy())
            targets.append(train[t + 1])
    weights, standardise = _ridge_1d(
        np.asarray(states), np.asarray(targets), penalty)

    h = np.zeros(units, dtype=np.float64)
    preds: list[float] = []
    for t in range(len(test) - 1):
        h = np.tanh(w_in[:, 0] * test[t] + w_rec @ h)
        if t >= warmup:
            preds.append(float(_predict_1d(weights, standardise, h[None, :])[0]))
    target = test[warmup + 1:]
    return {"nrmsfe": _nrmsfe(np.asarray(preds), target),
            "kind": "reservoir", "units": units, "rho_registered": rho,
            "seed": seed,
            "rho_measured": rho_measured,
            "echo_state_property_ok": rho_measured < 1.0}


def _mlp_baseline_arm(train: np.ndarray, test: np.ndarray, seed: int,
                      hidden: int = 32, epochs: int = 200,
                      lr: float = 1e-2) -> dict[str, Any]:
    """Tiny gradient-trained MLP (the trained measuring stick), CPU torch."""
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(4)
    model = torch.nn.Sequential(
        torch.nn.Linear(1, hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, 1),
    ).double()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    x_tr = torch.from_numpy(train[:-1, None])
    y_tr = torch.from_numpy(train[1:, None])
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(x_tr), y_tr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = model(torch.from_numpy(test[:-1, None])).numpy().reshape(-1)
    return {"nrmsfe": _nrmsfe(pred, test[1:]), "kind": "mlp",
            "hidden": hidden, "epochs": epochs}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_m147(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")

    started = time.time()
    mg_cfg = config["mackey_glass"]
    smoke = bool(config.get("_smoke_note", False))
    train_n = int(mg_cfg.get("_smoke_train", mg_cfg["train_points"]))
    test_n = int(mg_cfg.get("_smoke_test", mg_cfg["test_points"]))

    series = mackey_glass(
        train_n + test_n, tau=float(mg_cfg["tau"]), beta=float(mg_cfg["beta"]),
        gamma=float(mg_cfg["gamma"]), n=float(mg_cfg["n"]),
        dt=float(mg_cfg["dt"]), x0=float(mg_cfg["x0"]),
        seed=int(mg_cfg["seed"]), discard=int(mg_cfg["discard"]),
        sample_every=int(mg_cfg.get("sample_every", 10)))
    train, test = series[:train_n], series[train_n:]
    print(f"Mackey-Glass: train {train_n} / test {test_n} points", flush=True)

    # t1 anchor: M134 DSL reservoir reproduction (exact sealed code paths)
    anchor = config["t1_anchor"]
    print("t1 anchor: M134 DSL reservoir r128 reproduction", flush=True)
    dsl = _generate_corpus({"corpus": config["dsl_corpus"]},
                            limit_programs=int(config.get("_smoke_programs", 0))
                            or None)
    vocab_list = sorted(set(dsl["train"]))
    token_to_id = {t: i for i, t in enumerate(vocab_list)}
    dsl_train_ids = np.asarray([token_to_id[t] for t in dsl["train"]],
                               dtype=np.int64)
    dsl_test_ids = np.asarray([token_to_id[t] for t in dsl["test"]],
                              dtype=np.int64)
    anchor_result = _reservoir_readout(
        dsl_train_ids, dsl_test_ids, len(vocab_list), int(anchor["units"]),
        int(anchor["seed"]), float(anchor["spectral_radius"]),
        int(anchor["warmup_steps"]), float(anchor["ridge_penalty"]))
    anchor_delta = float(anchor_result["test_perplexity"]) - float(
        anchor["sealed_ppl"])
    print(f"  anchor ppl {anchor_result['test_perplexity']} "
          f"(sealed {anchor['sealed_ppl']}, delta {anchor_delta:+.6f})",
          flush=True)

    # ---- arms on Mackey-Glass ---------------------------------------------
    arms: dict[str, Any] = {}
    arms["no_memory"] = _no_memory_arm(train, test)
    print(f"  no_memory  NRMSFE {arms['no_memory']['nrmsfe']:.4f}", flush=True)

    tap = {}
    for k in [int(k) for k in config["tap_delay"]["k_ladder"]]:
        tap[k] = _tap_delay_arm(train, test, k)
        print(f"  tap_delay(k={k}) NRMSFE {tap[k]['nrmsfe']:.4f}", flush=True)
    arms["tap_delay"] = tap
    arms["tap_delay_best"] = min(tap.values(), key=lambda a: a["nrmsfe"])

    arms["programmatic"] = _programmatic_arm(train, test)
    print(f"  programmatic ({arms['programmatic']['winner']}) "
          f"NRMSFE {arms['programmatic']['nrmsfe']:.4f}", flush=True)

    reservoir_runs = []
    rcfg = config["reservoir"]
    for seed in [int(s) for s in rcfg["seeds"]]:
        for units in [int(u) for u in rcfg["units_ladder"]]:
            for rho in [float(r) for r in rcfg["rho_ladder"]]:
                arm = _reservoir_arm_mg(
                    train, test, units, rho, seed,
                    int(rcfg["warmup_steps"]), float(rcfg["ridge_penalty"]))
                reservoir_runs.append(arm)
                print(f"  reservoir(u={units},rho={rho},seed={seed}) "
                      f"NRMSFE {arm['nrmsfe']:.4f} "
                      f"rho_measured={arm['rho_measured']:.6f} "
                      f"esp_ok={arm['echo_state_property_ok']}", flush=True)
    arms["reservoir_runs"] = reservoir_runs

    if not smoke:
        arms["mlp"] = _mlp_baseline_arm(
            train, test, int(config["mlp"]["seed"]),
            hidden=int(config["mlp"]["hidden"]),
            epochs=int(config["mlp"]["epochs"]), lr=float(config["mlp"]["lr"]))
        print(f"  mlp        NRMSFE {arms['mlp']['nrmsfe']:.4f}", flush=True)

    # ---- gates --------------------------------------------------------------
    non_recurrent_best = min(
        arms["no_memory"]["nrmsfe"],
        arms["tap_delay_best"]["nrmsfe"],
        arms["programmatic"]["nrmsfe"],
    )
    gate_margin = float(config["gate"]["relative_margin"])
    best_per_seed: dict[int, float] = {}
    for arm in reservoir_runs:
        seed = int(arm["seed"])
        best_per_seed[seed] = min(best_per_seed.get(seed, float("inf")),
                                  float(arm["nrmsfe"]))

    if smoke:
        evidence_gate: dict[str, Any] | None = None
        admissible = False
    else:
        anchor_ok = abs(anchor_delta) <= float(anchor["tolerance"])
        winning_seeds = [
            s for s, best in best_per_seed.items()
            if best <= non_recurrent_best * (1.0 - gate_margin)
        ]
        required = int(config["gate"]["min_seeds"])
        fired = not anchor_ok or len(winning_seeds) < required
        evidence_gate = {
            "registered": config["gate"]["registered"],
            "anchor_delta": anchor_delta,
            "anchor_ok": anchor_ok,
            "non_recurrent_best_nrmsfe": non_recurrent_best,
            "best_per_reservoir_seed": best_per_seed,
            "winning_seeds": winning_seeds,
            "min_seeds": required,
            "fired": fired,
            "consequence": (config["gate"]["consequence_fired"] if fired
                            else config["gate"]["consequence_passed"]),
        }
        admissible = True
        print(f"  gate fired={fired} winning_seeds={winning_seeds}",
              flush=True)

    evidence: dict[str, Any] = {
        "milestone": "M147",
        "admissible_as_evidence": admissible,
        "config_file": Path(config_path).name,
        "config": config,
        "question": ("on one-step-ahead Mackey-Glass prediction, does a fixed "
                     "random reservoir earn its recurrence over the no-memory "
                     "baseline, the additive tap-delay line, and programmatic "
                     "primitives?"),
        "series": {"train_points": train_n, "test_points": test_n,
                   "test_std": float(np.std(test))},
        "t1_anchor": {"sealed_ppl": float(anchor["sealed_ppl"]),
                      "measured_ppl": anchor_result["test_perplexity"],
                      "delta": anchor_delta,
                      "tolerance": float(anchor["tolerance"])},
        "arms": arms,
        "gate": evidence_gate,
        "runtime_seconds": round(time.time() - started, 2),
        "notes": {
            "m134_prior": ("reservoir loss on the DSL token task is sealed "
                           "(M134 r128 29.72 / r512 27.54); this screen tests "
                           "a NEW axis where that prior does not bind"),
            "no_backprop_except_mlp": True,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"\nM147 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m147(args.config, args.output)


if __name__ == "__main__":
    main()
