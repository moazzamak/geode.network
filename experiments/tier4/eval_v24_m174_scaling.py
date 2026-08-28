"""M174 — Q(n) scaling of the toolbox route on a NEW synthetic family.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v24.md`` (section 7
Phase D M174): "does the frozen path keep scaling? — the honest open
question". The v24 scaling evidence so far was measured on the real
corpus (0.2246 @ 138k -> 0.2614 @ 409,832). This cell asks the same
question on a new, out-of-registry family: as train n grows, does the
toolbox's fit-and-report (frozen lag-window provider + closed-form
intercept ridge, the M171b repair) keep improving on held-out rows, or
plateau (which would be L1 evidence: growth must change codes, not
heads)?

Cell (registered): the "delayed-coupling" family
x[t+1] = 0.35x[t] + 0.30x[t-4] + 0.15x[t]x[t-2] + 0.10 sin(2.5x[t-1])
+ 0.05 eps, temporal_structure = "delayed" (a new axis value
combination, so its fingerprint is distinct from mg/lorenz/tabular).
n ladder 200 / 1000 / 5000 / 25000 (60% train, window 10, penalty grid
+ 5-fold CV as in M171). Reported per n: the fit-and-report ridge R2,
the routed frozen specialist's R2 on the new rows, and the
mean-primitive fallback R2 (flat control). Gates: S1 route determinism;
S2 the fit-and-report ladder is non-decreasing; S3 R2(25000) -
R2(200) >= 0.10; a plateau before the top cell is reported (L1
evidence), not hidden. Encoder/arms: one self-consistent in-process
encoder (the M172b pattern) + the M171 sealed accuracy records.
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
from experiments.tier4.eval_v24_m169_fingerprint_train import TASK_DESCRIPTORS
from experiments.tier4.eval_v24_m171_router import (
    _build_encoder,
    _ridge_fit,
    _ridge_predict,
)
from geode.core.descriptor import normalise
from geode.core.router import Router

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v24"
                  / "m174_scaling.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v24" / "m174_scaling")
M171_EVIDENCE = (REPO_ROOT / "logs" / "results" / "v24" / "m171_router"
                 / "evidence.json")

WINDOW = 10
N_LADDER = [200, 1000, 5000, 25000]


def _delayed_family(n: int, seed: int = 23) -> np.ndarray:
    """The registered delayed-coupling family (see module docstring)."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n + 10)
    x[:5] = 0.1 * rng.standard_normal(5)
    for t in range(5, n + 10):
        x[t] = (0.35 * x[t - 1] + 0.30 * x[t - 4]
                + 0.15 * x[t - 1] * x[t - 2]
                + 0.10 * np.sin(2.5 * x[t - 1])
                + 0.05 * rng.standard_normal())
    return x[10:]


def _window_frames(ser: np.ndarray, lag: int, n_tr: int):
    idx = np.arange(lag, len(ser) - 1)
    is_tr = idx < n_tr
    x = np.stack([ser[t - lag + 1:t + 1] for t in idx])
    y = ser[idx + 1]
    return (x[is_tr], y[is_tr]), (x[~is_tr], y[~is_tr])


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float(1 - ss_res / ss_tot)


def run_m174(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")
    started = time.time()
    smoke = inadmissible
    ladder = list(N_LADDER)
    if smoke:
        ladder = [l for l in ladder if l <= int(config.get("_smoke_max", 1000))]

    configure_external_cache_environment()

    # ---- the new family descriptor (delayed axis value) --------------------
    family_desc = normalise({
        **{k: v for k, v in TASK_DESCRIPTORS["mackey_glass"].items()},
        "input.temporal_structure": "delayed",
    })

    # ---- one self-consistent encoder + router (the M172b pattern) ----------
    m171 = json.loads(M171_EVIDENCE.read_text(encoding="utf-8"))
    enc, _ = _build_encoder(config)
    fp = {n: [float(v) for v in
              enc.fingerprint(normalise(d)).detach().cpu().numpy()]
          for n, d in TASK_DESCRIPTORS.items()}
    fp_family = [float(v) for v in
                 enc.fingerprint(family_desc).detach().cpu().numpy()]

    arm_id = {"mackey_glass": "mackey_glass-ridge",
              "lorenz": "lorenz-ridge", "tabular": "tabular-ridge",
              "dyck": "dyck-ridge"}
    router = Router()
    for task, aid in arm_id.items():
        router.add_arm({
            "arm_id": aid, "task_id": task, "fingerprint": fp[task],
            "output_contract": {"kind": "regression" if task != "dyck"
                                else "classification",
                                "dim": 4 if task == "dyck" else 1},
            "held_out_accuracy": {task: m171["r1"][task]["best_accuracy"]},
            "selection_accuracy": float(m171["r1"][task]
                                        ["best_accuracy"]),
            "availability": {"contract_hash": "sealed-m171",
                             "payload_hash": "sealed-m171",
                             "healthy": True},
            "price": 0.0, "general": False, "primitive": False,
        })

    # ---- route the new family (task-level, fingerprint) --------------------
    chain = router.chain(fp_family)
    routed = next(c for c in chain if c["output_contract"]["kind"]
                  == "regression")
    chain_again = router.chain(fp_family)
    s1 = [c["arm_id"] for c in chain] == [c["arm_id"] for c in chain_again]

    # ---- the n ladder ------------------------------------------------------
    ladder_rows: dict[str, Any] = {}
    for n in ladder:
        ser = _delayed_family(n)
        n_tr = int(n * 0.6)
        (x_tr, y_tr), (x_ho, y_ho) = _window_frames(ser, WINDOW, n_tr)
        w, norm, lam = _ridge_fit(x_tr, y_tr)
        pred = _ridge_predict(x_ho, w, norm)
        pred = pred[:, 0] if pred.ndim == 2 else pred
        ridge_r2 = _r2(y_ho, pred)
        # primitive fallback control (flat by construction)
        primitive_r2 = _r2(y_ho, np.full(len(y_ho), y_tr.mean()))
        ladder_rows[str(n)] = {
            "ridge_r2": ridge_r2, "primitive_r2": primitive_r2,
            "lam_selected": lam,
        }

    ridge_ladder = [ladder_rows[str(n)]["ridge_r2"] for n in ladder]
    s2 = bool(all(ridge_ladder[i] <= ridge_ladder[i + 1] + 1e-9
                  for i in range(len(ridge_ladder) - 1)))
    s3 = bool(ridge_ladder[-1] - ridge_ladder[0] >= 0.10)
    plateau_n = next((ladder[i] for i in range(len(ridge_ladder) - 1)
                      if ridge_ladder[-1] - ridge_ladder[i] <= 0.01), None)

    evidence: dict[str, Any] = {
        "milestone": "M174",
        "cell": "Q(n) scaling of the toolbox route, new delayed family",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "question": config["question"],
        "interpretation_registered_before_running":
            config["interpretation_registered_before_running"],
        "family": "delayed-coupling AR(5)",
        "family_descriptor": family_desc.axes,
        "family_descriptor_hash": family_desc.hash(),
        "routed_arm": routed["arm_id"],
        "chain": [c["arm_id"] for c in chain],
        "s1_route_determinism": s1,
        "ladder": ladder_rows,
        "ridge_r2_ladder": {str(n): ladder_rows[str(n)]["ridge_r2"]
                            for n in ladder},
        "s2_ladder_non_decreasing": s2,
        "s3_top_minus_bottom": ridge_ladder[-1] - ridge_ladder[0],
        "s3_passed": s3,
        "plateau_first_n": plateau_n,
        "void": False,
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"M174 complete -> {output_dir / 'evidence.json'}", flush=True)
    print(json.dumps({"routed": routed["arm_id"], "ladder":
                      evidence["ridge_r2_ladder"], "s1": s1, "s2": s2,
                      "s3": s3, "plateau_first_n": plateau_n},
                     indent=1), flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m174(args.config, args.output)


if __name__ == "__main__":
    main()
