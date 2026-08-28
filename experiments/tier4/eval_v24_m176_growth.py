"""M176 — registry-growth: at what registry size does routing
measurably beat the global fallback?

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v24.md`` (section 7
Phase D M176). Simulation cell (all synthetic, all regression): K
families with DISTINCT descriptors drawn from the registered axis grid
(temporal_structure {iid, sequential, delayed} x noise_regime {low,
medium, high} x stationarity {stationary, non-stationary}), one
lag-window ridge specialist per family (the M171b intercept fit) plus
the mean-predictor global fallback. K ladder 2 / 4 / 8 / 16.

Per K: the full competence matrix is measured (every specialist on
every family — all share the window provider and regression contract),
the router routes each family by fingerprint with the registered
eps-advance rule (eps = 0), and the margin routed - fallback is
recorded per family. Registered exclusion (applied uniformly): a
family is excluded from the margin where the BEST measured arm scores
below 0.1 R2 (routing cannot beat a fallback on pure noise; measuring
there tests nothing).

Gate: K* = the smallest registry size where the mean margin over
included families >= 0.05 AND at least 2/3 of included families
improve. K* is the registered learned-router trigger. The raw
nearest-arm margin (no eps-advance) is reported alongside.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.data_cache import configure_external_cache_environment
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from experiments.tier4.eval_v24_m171_router import (
    _build_encoder,
    _ridge_fit,
    _ridge_predict,
)
from geode.core.descriptor import normalise
from geode.core.router import Router

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v24"
                  / "m176_growth.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v24" / "m176_growth")

WINDOW = 10
K_LADDER = [2, 4, 8, 16]
N_ROWS = 800


def _family(k: int, n: int, temporal: str, noise: str,
            stationary: bool) -> np.ndarray:
    """Deterministic synthetic family k with the registered axis combo."""
    rng = np.random.default_rng(1000 + k)
    sigma = {"low": 0.05, "medium": 0.15, "high": 0.5}[noise]
    x = np.zeros(n + 10)
    x[:5] = 0.1 * rng.standard_normal(5)
    for t in range(5, n + 10):
        if temporal == "iid":
            x[t] = 0.3 * rng.standard_normal()
        elif temporal == "sequential":
            x[t] = (0.6 * x[t - 1] + 0.2 * np.sin(2.0 * x[t - 2])
                    + sigma * rng.standard_normal())
        else:  # delayed
            x[t] = (0.35 * x[t - 1] + 0.30 * x[t - 4]
                    + 0.15 * x[t - 1] * x[t - 2] + 0.10 * np.sin(
                        2.5 * x[t - 1]) + sigma * rng.standard_normal())
        if not stationary:
            x[t] += 0.01 * (t / n)  # slow registered drift
    return x[10:]


def _combo(k: int) -> tuple[str, str, bool]:
    """Deterministic INTERLEAVED assignment of axis values to family k so
    small registries span the grid (registered): temporal cycles fastest,
    then noise, then stationarity."""
    temporals = ["iid", "sequential", "delayed"]
    noises = ["low", "medium", "high"]
    temporal = temporals[k % 3]
    noise = noises[(k // 3) % 3]
    stationary = bool((k // 9) % 2 == 0)
    return temporal, noise, stationary


def _desc_of(temporal: str, noise: str, stationary: bool):
    return normalise({
        "input.modality": "numeric-series",
        "input.value_kind": "continuous",
        "input.temporal_structure": temporal,
        "output.kind": "regression",
        "output.ordinality": "cardinal",
        "latent.recurrence": "none",
        "latent.stationarity": "stationary" if stationary
        else "non-stationary",
        "latent.noise_regime": noise,
        "latent.label_cardinality": 1,
        "latent.sample_regime": "small",
        "coupling": "single-task",
    })


def run_m176(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")
    started = time.time()
    smoke = inadmissible
    ladder = list(K_LADDER)
    n_rows = N_ROWS
    if smoke:
        ladder = [k for k in ladder if k <= int(config.get("_smoke_max", 4))]
        n_rows = min(n_rows, int(config.get("_smoke_rows", 200)))

    configure_external_cache_environment()
    enc, _ = _build_encoder(config)

    ladder_rows: dict[str, Any] = {}
    k_star = None
    for K in ladder:
        combos = {k: _combo(k) for k in range(K)}
        descs = {k: _desc_of(*combos[k]) for k in range(K)}
        fps = {k: [float(v) for v in
                   enc.fingerprint(descs[k]).detach().cpu().numpy()]
               for k in range(K)}
        # data + specialist fits
        series = {k: _family(k, n_rows, *combos[k]) for k in range(K)}
        n_tr = int(n_rows * 0.6)
        fits = {}
        for k in range(K):
            idx = np.arange(WINDOW, len(series[k]) - 1)
            is_tr = idx < n_tr
            xw = np.stack([series[k][t - WINDOW + 1:t + 1]
                           for t in idx])
            y = series[k][idx + 1]
            w, norm, lam = _ridge_fit(xw[is_tr], y[is_tr])
            fits[k] = (w, norm)
        # competence matrix K x K
        matrix = {}
        for k in range(K):
            idx = np.arange(WINDOW, len(series[k]) - 1)
            is_tr = idx < n_tr
            xw = np.stack([series[k][t - WINDOW + 1:t + 1]
                           for t in idx])
            y = series[k][idx + 1]
            y_ho, x_ho = y[~is_tr], xw[~is_tr]
            ss_tot = float(((y_ho - y_ho.mean()) ** 2).sum())
            for j in range(K):
                pred = _ridge_predict(x_ho, fits[j][0], fits[j][1])
                pred = pred[:, 0] if pred.ndim == 2 else pred
                matrix.setdefault(k, {})[j] = float(
                    1 - ((y_ho - pred) ** 2).sum() / ss_tot)
        fallback = {}
        for k in range(K):
            idx = np.arange(WINDOW, len(series[k]) - 1)
            is_tr = idx < n_tr
            y = series[k][idx + 1]
            y_ho = y[~is_tr]
            ss_tot = float(((y_ho - y_ho.mean()) ** 2).sum())
            pred = np.full(len(y_ho), float(y[is_tr].mean()))
            fallback[k] = float(1 - ((y_ho - pred) ** 2).sum() / ss_tot)
        # router with eps-advance
        router = Router()
        for k in range(K):
            router.add_arm({
                "arm_id": f"fam{k}-ridge", "task_id": str(k),
                "fingerprint": fps[k],
                "output_contract": {"kind": "regression", "dim": 1},
                "held_out_accuracy": {str(j): matrix[j][k]
                                      for j in range(K)},
                "selection_accuracy": float(matrix[k][k]),
                "availability": {"contract_hash": "sim",
                                 "payload_hash": "sim", "healthy": True},
                "price": 0.0, "general": False, "primitive": False,
            })
        margins, raw_margins, bests = [], [], []
        for k in range(K):
            best = max(matrix[k].values())
            bests.append(best)
            chain = router.chain(fps[k], task_id=str(k))
            routed = next(c for c in chain if c["output_contract"]["kind"]
                          == "regression")["arm_id"]
            routed_acc = matrix[k][int(routed.replace("fam", "")
                                       .split("-")[0])]
            raw_head = int(router.route(fps[k], k=1)[0]["arm_id"]
                           .replace("fam", "").split("-")[0])
            raw_acc = matrix[k][raw_head]
            margins.append(routed_acc - fallback[k])
            raw_margins.append(raw_acc - fallback[k])
        included = [k for k in range(K) if bests[k] >= 0.1]
        inc_margins = [margins[k] for k in included]
        inc_raw = [raw_margins[k] for k in included]
        mean_margin = float(np.mean(inc_margins)) if inc_margins else 0.0
        frac_improved = float(np.mean([m >= 0.0 for m in inc_margins])) \
            if inc_margins else 0.0
        passed = bool(mean_margin >= 0.05 and frac_improved >= 2 / 3)
        if passed and k_star is None:
            k_star = K
        ladder_rows[str(K)] = {
            "families": K, "included_families": len(included),
            "mean_margin_eps_advance": mean_margin,
            "fraction_improved": frac_improved,
            "mean_margin_raw_nearest": float(np.mean(inc_raw))
            if inc_raw else 0.0,
            "gate_passed": passed,
            "margins_eps_advance": [margins[k] for k in included],
        }

    evidence: dict[str, Any] = {
        "milestone": "M176",
        "cell": "registry-growth vs global fallback (simulation)",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "question": config["question"],
        "interpretation_registered_before_running":
            config["interpretation_registered_before_running"],
        "ladder": ladder_rows,
        "learned_router_trigger_k_star": k_star,
        "void": False,
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"M176 complete -> {output_dir / 'evidence.json'}", flush=True)
    print(json.dumps({"trigger_k_star": k_star, "ladder":
                      {k: {kk: vv for kk, vv in v.items()
                           if kk != "margins_eps_advance"}
                       for k, v in ladder_rows.items()}}, indent=1),
          flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m176(args.config, args.output)


if __name__ == "__main__":
    main()
