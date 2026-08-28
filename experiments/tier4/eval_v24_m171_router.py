"""M171 — router + registry integration and the R1 gate.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v24.md`` (section 7
Phase C M171; section 5 router contract). What this cell measures:

- four registered tasks on the measured axes (the M147/M157 series
  families): mackey_glass, lorenz, dyck, tabular;
- arms: one closed-form ridge specialist per task (frozen providers:
  lag-window for series, raw (index, value) pairs for tabular, next-token
  for dyck) plus the programmatic primitives (mean/mode predictor,
  dyck-bigram) as the always-available bottom tier;
- R1 per task: the routed arm's held-out accuracy must be >= the best
  single applicable arm on the same held-out rows. Routed = the first
  contract-matching, healthy arm in the registered failover chain.
- capability 9 (cold start): a task with no registered arm gets a
  fingerprint and falls back to the strongest general arm (the
  mean/mode primitive); measured, not gated on R1.

Registered interpretation (before running): R1 failure on a task is a
scoped negative on nearest-arm routing FOR THIS fingerprint — the
router and registry still ship as the MVP mechanism; the gates are the
measurement. A smoke run declares inadmissibility and refuses the
sealed output directory.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from experiments.common.data_cache import configure_external_cache_environment
from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    write_canonical_json,
)
from experiments.tier4.eval_v16_m147_temporal_memory import mackey_glass
from experiments.tier4.eval_v23_m157_temporal_screen import (
    _dyck_series,
    _lorenz,
)
from experiments.tier4.eval_v24_m169_fingerprint_train import (
    SIMILAR_PAIRS,
    TASK_DESCRIPTORS,
    _axis_key,
    _train,
)
from geode.core.descriptor import AXES, normalise
from geode.core.fingerprint import FingerprintEncoder
from geode.core.registry import TaskRegistry
from geode.core.router import Router

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v24"
                  / "m171_router.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v24" / "m171_router")

R1_TASKS = ["mackey_glass", "lorenz", "dyck", "tabular"]
WINDOW = 10
PENALTIES = (1e-6, 1e-4, 1e-2, 1e0)


# ---------------------------------------------------------------------------
# generators (registered provenance: mg = M147 canonical; lorenz/dyck = M157)
# ---------------------------------------------------------------------------
def _tabular_series(n_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Tabular-style rows: x1 = ordered index, x2 = sin(x1) + noise,
    target y = x1*x2 + x1^2 + noise. Registered for M171."""
    rng = np.random.default_rng(seed)
    x1 = np.arange(n_points, dtype=np.float64) / 50.0
    x2 = np.sin(x1) + 0.05 * rng.standard_normal(n_points)
    y = x1 * x2 + x1 ** 2 + 0.1 * rng.standard_normal(n_points)
    return x2, y


def _audio_series(n_points: int, seed: int) -> np.ndarray:
    """A novel 'audio-like' series for the capability-9 cold-start demo:
    two sinusoids + noise, low-pass coupled. NOT in the registry."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_points, dtype=np.float64) / 100.0
    s = np.sin(2 * np.pi * 0.37 * t) + 0.4 * np.sin(2 * np.pi * 0.11 * t)
    return s + 0.05 * rng.standard_normal(n_points)


# ---------------------------------------------------------------------------
# providers (frozen feature contracts) and the closed-form fit
# ---------------------------------------------------------------------------
def _ridge_fit(x_tr: np.ndarray, y_tr: np.ndarray,
               classify: bool = False) -> tuple[np.ndarray, float]:
    """Closed-form ridge with an explicit intercept column and
    deterministic 5-fold CV over the penalty grid.

    Registered repair (M171a, 17 Aug 2026): the M171 build omitted the
    intercept, so standardized (zero-mean) windows regressed against the
    uncentered target had to fake an offset through collinear columns
    (cond(X'X) ~ 1e17 on Mackey-Glass) and every arm on smooth series
    collapsed. The intercept column is appended to the standardized
    features; y one-hot encoded when classify (argmax at predict time).
    """
    mu, sd = x_tr.mean(axis=0), x_tr.std(axis=0)
    sd[sd == 0] = 1.0
    xs = np.hstack([(x_tr - mu) / sd, np.ones((len(x_tr), 1))])
    if classify:
        y_tr = np.eye(int(y_tr.max()) + 1)[y_tr.astype(int)]
    n = len(xs)
    folds = np.arange(n) % 5
    best_lam, best_score = PENALTIES[0], -np.inf
    for lam in PENALTIES:
        scores = []
        for f in range(5):
            tr, va = folds != f, folds == f
            xtx = xs[tr].T @ xs[tr] + lam * np.eye(xs.shape[1])
            w = np.linalg.solve(xtx, xs[tr].T @ y_tr[tr])
            pred = xs[va] @ w
            if classify:
                scores.append(float((pred.argmax(1) == y_tr[va].argmax(1))
                                    .mean()))
            else:
                ss_res = ((y_tr[va] - pred) ** 2).sum()
                ss_tot = ((y_tr[va] - y_tr[va].mean()) ** 2).sum()
                scores.append(float(1 - ss_res / ss_tot))
        score = float(np.mean(scores))
        if score > best_score:
            best_lam, best_score = lam, score
    xtx = xs.T @ xs + best_lam * np.eye(xs.shape[1])
    w = np.linalg.solve(xtx, xs.T @ y_tr)
    return w, (mu, sd), best_lam


def _ridge_predict(x, w, norm) -> np.ndarray:
    mu, sd = norm
    xs = np.hstack([(x - mu) / sd, np.ones((len(x), 1))])
    return xs @ w


# ---------------------------------------------------------------------------
# the M169 fingerprint encoder, rebuilt deterministically from the sealed
# config (G1 determinism makes the rebuild equal the sealed fingerprints)
# ---------------------------------------------------------------------------
def _build_encoder(config: dict[str, Any]) -> FingerprintEncoder:
    f_dim = int(config["model"]["f_dim"])
    mlp_hidden = int(config["model"]["mlp_hidden"])
    seed = int(config["model"]["seed"])
    steps = int(config["model"]["steps"])
    lr = float(config["model"]["lr"])
    descs = {n: normalise(d) for n, d in TASK_DESCRIPTORS.items()}
    enc = FingerprintEncoder(f_dim=f_dim, mlp_hidden=mlp_hidden, seed=seed)
    enc.attr_heads = torch.nn.ModuleDict({
        _axis_key(axis): torch.nn.Linear(f_dim, len(vocab))
        for axis, vocab in AXES.items()})
    _train(enc, descs, SIMILAR_PAIRS, steps, lr)
    return enc, descs


# ---------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------
def run_m171(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    inadmissible = "_smoke_note" in config
    if inadmissible and Path(output_dir).resolve() == DEFAULT_OUTPUT.resolve():
        raise SystemExit(
            f"REFUSING TO RUN: {Path(config_path).name} declares itself "
            "inadmissible and would write to the SEALED output directory.")

    started = time.time()
    smoke = inadmissible
    n_rows = int(config["cell"]["n_rows"])
    train_frac = float(config["cell"]["train_frac"])
    if smoke:
        n_rows = min(n_rows, int(config.get("_smoke_rows", 120)))

    configure_external_cache_environment()

    # ---- data ------------------------------------------------------------
    series = {
        "mackey_glass": mackey_glass(n_rows, seed=7),
        "lorenz": _lorenz(n_points=n_rows, sigma=10.0, rho=28.0,
                          beta=8.0 / 3.0, dt=0.01, sample_every=10,
                          warmup=100, x0=[1.0, 1.0, 1.0], seed=11),
        "dyck": _dyck_series(n_rows, max_depth=4, seed=13),
        "audio": _audio_series(n_rows, seed=19),
    }
    tab_x2, tab_y = _tabular_series(n_rows, seed=17)
    series["tabular"] = tab_y  # the tabular task's ordered value column

    # ---- data: one canonical frame per task (registered) -----------------
    # rows t in [lag, n-2): window features ser[t-lag+1..t], pairs features
    # [t/n, ser[t]]; target ser[t+1] (next value/token). train: t < n_tr,
    # held-out: t >= n_tr. Held-out windows may include train-history values
    # (standard sliding-window evaluation; no target leakage).
    def _frames(ser: np.ndarray, lag: int, n_tr: int):
        idx = np.arange(lag, len(ser) - 1)
        is_tr = idx < n_tr
        x_w = np.stack([ser[t - lag + 1:t + 1] for t in idx])
        x_p = np.stack([idx / len(ser), ser[idx]], axis=1)
        y = ser[idx + 1]
        return (x_w[is_tr], y[is_tr]), (x_w[~is_tr], y[~is_tr]), \
            (x_p[is_tr], y[is_tr]), (x_p[~is_tr], y[~is_tr])

    tasks: dict[str, dict[str, Any]] = {}
    n_tr = int(n_rows * train_frac)
    for name in ["mackey_glass", "lorenz", "dyck", "tabular", "audio"]:
        kind = "classification" if name == "dyck" else "regression"
        tr_w, ho_w, tr_p, ho_p = _frames(series[name], WINDOW, n_tr)
        tasks[name] = {"kind": kind, "train_w": tr_w, "heldout_w": ho_w,
                       "train_p": tr_p, "heldout_p": ho_p,
                       "series": series[name]}

    # ---- fingerprints (the sealed M169 encoder, rebuilt deterministically)
    enc, descs = _build_encoder(config)
    fingerprints = {n: enc.fingerprint(descs[n]).detach().cpu().numpy()
                    for n in descs}
    # the capability-9 cold-start descriptor (registered; submodality is OOV
    # on purpose, exercising the I4 no-refusal fallback)
    desc_audio = normalise({
        "input.modality": "audio", "input.submodality": "spectrogram",
        "input.value_kind": "continuous",
        "input.temporal_structure": "sequential",
        "output.kind": "regression", "output.ordinality": "cardinal",
        "latent.recurrence": "markov",
        "latent.stationarity": "stationary",
        "latent.noise_regime": "medium", "latent.label_cardinality": 1,
        "latent.sample_regime": "small", "coupling": "single-task",
    })
    fingerprints["audio"] = (enc.fingerprint(desc_audio).detach()
                              .cpu().numpy())
    fp_lists = {n: [float(v) for v in fp] for n, fp in fingerprints.items()}

    # ---- fit specialist arms (closed form, per-task providers) -----------
    arm_task = {"mackey_glass-ridge": "mackey_glass",
                "lorenz-ridge": "lorenz", "tabular-ridge": "tabular",
                "dyck-ridge": "dyck"}
    fits: dict[str, dict[str, Any]] = {}
    for name in ["mackey_glass", "lorenz"]:
        w, norm, lam = _ridge_fit(*tasks[name]["train_w"])
        fits[name] = {"w": w, "norm": norm, "lam": lam}
    # the tabular specialist uses its (index, value) pairs provider
    w, norm, lam = _ridge_fit(*tasks["tabular"]["train_p"])
    fits["tabular"] = {"w": w, "norm": norm, "lam": lam}
    w, norm, lam = _ridge_fit(*tasks["dyck"]["train_w"], classify=True)
    fits["dyck"] = {"w": w, "norm": norm, "lam": lam}
    # dyck-bigram primitive: P(next | current) from train
    nxt = {}
    y_tr_dyck = tasks["dyck"]["train_w"][1]
    for cur, nxt_tok in zip(y_tr_dyck[:-1].astype(int),
                            y_tr_dyck[1:].astype(int)):
        nxt.setdefault(cur, np.zeros(4))
        nxt[cur][nxt_tok] += 1
    bigram = {c: v / v.sum() for c, v in nxt.items()}
    tr_mean = {n: float(tasks[n]["train_w"][1].mean()) for n in
               ["mackey_glass", "lorenz", "tabular", "audio"]}
    dyck_mode = int(np.bincount(y_tr_dyck.astype(int)).argmax())

    def predict(arm: str, task: str) -> np.ndarray:
        if arm == "mean-mode-primitive":
            y_ho = tasks[task]["heldout_w"][1]
            return (np.full(len(y_ho), tr_mean[task]) if
                    tasks[task]["kind"] == "regression"
                    else np.full(len(y_ho), dyck_mode))
        if arm == "dyck-bigram":
            x_ho = tasks[task]["heldout_w"][0]
            out = [int(np.argmax(bigram.get(int(row[-1]),
                                            np.ones(4) / 4)))
                   for row in x_ho]
            return np.asarray(out)
        if arm == "tabular-ridge":
            x_ho = tasks[task]["heldout_p"][0]
        else:
            x_ho = tasks[task]["heldout_w"][0]
        w, norm = fits[arm_task[arm]]["w"], fits[arm_task[arm]]["norm"]
        pred = _ridge_predict(x_ho, w, norm)
        if tasks[task]["kind"] == "classification":
            return pred.argmax(1)
        return pred[:, 0] if pred.ndim == 2 else pred

    def score(task: str, pred: np.ndarray) -> float:
        y = tasks[task]["heldout_w"][1]
        if tasks[task]["kind"] == "classification":
            return float((pred == y.astype(int)).mean())
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        return float(1 - ss_res / ss_tot)

    # ---- competence matrix (every applicable arm on every task) ----------
    APPLICABLE = {
        "mackey_glass": ["mackey_glass-ridge", "lorenz-ridge",
                         "tabular-ridge", "mean-mode-primitive"],
        "lorenz": ["mackey_glass-ridge", "lorenz-ridge",
                   "tabular-ridge", "mean-mode-primitive"],
        "tabular": ["mackey_glass-ridge", "lorenz-ridge",
                    "tabular-ridge", "mean-mode-primitive"],
        "dyck": ["dyck-ridge", "dyck-bigram", "mean-mode-primitive"],
        "audio": ["mackey_glass-ridge", "lorenz-ridge", "tabular-ridge",
                  "mean-mode-primitive"],
    }
    matrix: dict[str, dict[str, float]] = {}
    for task in APPLICABLE:
        matrix[task] = {}
        for arm in APPLICABLE[task]:
            matrix[task][arm] = score(task, predict(arm, task))

    # ---- router + registry integration -----------------------------------
    router = Router()
    registry = TaskRegistry()
    arm_id = {"mackey_glass": "mackey_glass-ridge",
              "lorenz": "lorenz-ridge", "tabular": "tabular-ridge",
              "dyck": "dyck-ridge"}
    for task in R1_TASKS:
        registry.add(TASK_DESCRIPTORS[task])
    tx_hashes = {}
    for i, task in enumerate(R1_TASKS):
        tx_hashes[f"after_{i + 1}_{task}"] = {
            tid: registry.content_hash(tid) for tid in registry.list_ids()}

    for task in R1_TASKS:
        router.add_arm({
            "arm_id": arm_id[task],
            "task_id": task,
            "fingerprint": fp_lists[task],
            "output_contract": {"kind": tasks[task]["kind"],
                                "dim": 4 if task == "dyck" else 1},
            "held_out_accuracy": {t: matrix[t][arm_id[task]]
                                   for t in matrix if arm_id[task]
                                   in matrix[t]},
            "selection_accuracy": float(matrix[task][arm_id[task]]),
            "availability": {"contract_hash": payload_hash(
                {"provider": "window" if task != "tabular" else "pairs",
                 "window": WINDOW, "penalties": list(PENALTIES),
                 "lam": fits[task]["lam"]}),
                "payload_hash": payload_hash(fits[task]["w"]), "healthy": True},
            "price": 0.0,
            "general": False,
            "primitive": False,
        })
    router.add_arm({
        "arm_id": "dyck-bigram", "task_id": "dyck", "fingerprint": [],
        "output_contract": {"kind": "classification", "dim": 4},
        "held_out_accuracy": {"dyck": matrix["dyck"]["dyck-bigram"]},
        "selection_accuracy": float(matrix["dyck"]["dyck-bigram"]),
        "availability": {"contract_hash": payload_hash({"provider":
                                                        "bigram"}),
                         "payload_hash": payload_hash(bigram),
                         "healthy": True},
        "price": 0.0, "general": False, "primitive": True,
    })
    router.add_arm({
        "arm_id": "mean-mode-primitive", "task_id": None, "fingerprint": [],
        "output_contract": {"kind": "regression", "dim": 1},
        "held_out_accuracy": {t: matrix[t]["mean-mode-primitive"]
                              for t in APPLICABLE},
        "selection_accuracy": float(np.mean(
            [matrix[t]["mean-mode-primitive"] for t in APPLICABLE])),
        "availability": {"contract_hash": payload_hash({"provider":
                                                        "mean/mode"}),
                         "payload_hash": payload_hash(
                             {"tr_mean": tr_mean, "dyck_mode": dyck_mode}),
                         "healthy": True},
        "price": 0.0, "general": True, "primitive": True,
    })

    # ---- R1 per task ------------------------------------------------------
    r1 = {}
    chains = {}
    for task in R1_TASKS:
        chain = router.chain(fp_lists[task], task_id=task)
        chains[task] = [{"arm_id": c["arm_id"], "route_cos": c["route_cos"],
                         "contract": c["output_contract"],
                         "selection_accuracy": c["selection_accuracy"]}
                        for c in chain]
        routed = next(c for c in chain
                      if c["output_contract"]["kind"] == tasks[task]["kind"])
        routed_acc = matrix[task][routed["arm_id"]]
        best_acc = max(matrix[task].values())
        r1[task] = {"routed_arm": routed["arm_id"],
                    "routed_accuracy": routed_acc,
                    "best_accuracy": best_acc,
                    "passed": bool(routed_acc + 1e-12 >= best_acc)}

    # ---- the registered section-5 eps-advance failover rule --------------
    # "the chosen arm must stay within a registered eps of the best measured
    # arm on held-out rows, or the chain advances" (eps = 0.0 registered).
    # Simulated from the measured matrix; deployment reads the same records
    # from the registry.
    EPS = 0.0
    r1_advance = {}
    for task in R1_TASKS:
        chain = router.chain(fp_lists[task], task_id=task)
        best_acc = max(matrix[task].values())
        chosen = None
        for c in chain:
            if (c["output_contract"]["kind"] == tasks[task]["kind"]
                    and c["arm_id"] in matrix[task]):
                if matrix[task][c["arm_id"]] + EPS + 1e-12 >= best_acc:
                    chosen = c["arm_id"]
                    break
        r1_advance[task] = {"chosen_after_advance": chosen,
                            "accuracy": matrix[task].get(chosen, None)
                            if chosen else None,
                            "best_accuracy": best_acc,
                            "passed": bool(chosen is not None)}

    # ---- capability 9: cold start for unknown tasks -----------------------
    cold = {}
    for task in ["audio", "cifar10"]:
        if task == "cifar10":
            d = np.load(REPO_ROOT / "data" / "tier4" / "cifar10_features.npz",
                        mmap_mode="r")
            labels = d["labels"][:400]
            y_ho = labels.astype(int)
            pred = np.full(len(y_ho), int(np.bincount(labels[:240]).argmax()))
            acc = float((pred == y_ho).mean())
            cold["cifar10"] = {
                "fallback_arm": "mean-mode-primitive",
                "accuracy_on_heldout_sample": acc,
                "note": "no registry arm for cifar10; strongest general arm "
                        "= mode predictor on a 400-row label sample",
            }
            continue
        fb = router.cold_start("regression")
        pred = np.full(len(tasks["audio"]["heldout_w"][1]), tr_mean["audio"])
        acc = score("audio", pred)
        nearest = router.route(fp_lists["audio"], k=1)[0]["arm_id"]
        cold["audio"] = {
            "fallback_arm": fb["arm_id"],
            "nearest_specialist_by_cos": nearest,
            "fallback_r2": acc,
            "best_applicable_r2": max(matrix["audio"].values()),
            "note": "novel descriptor, no registry entry -> strongest "
                    "general arm (I4 no-refusal)",
        }

    # ---- determinism + hashes ----------------------------------------------
    chain_again = {t: [c["arm_id"] for c in router.chain(fp_lists[t],
                                                            task_id=t)]
                   for t in R1_TASKS}
    determinism = all(chain_again[t] == [c["arm_id"] for c in chains[t]]
                      for t in R1_TASKS)
    router_hash = router.content_hash()

    evidence: dict[str, Any] = {
        "milestone": "M171",
        "cell": "router + registry integration, R1 gate (v0)",
        "admissible_as_evidence": not smoke,
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "question": config["question"],
        "interpretation_registered_before_running":
            config["interpretation_registered_before_running"],
        "cell_tasks": R1_TASKS,
        "window": WINDOW,
        "penalty_grid": list(PENALTIES),
        "competence_matrix": matrix,
        "r1": r1,
        "r1_all_passed": bool(all(v["passed"] for v in r1.values())),
        "r1_eps_advance": r1_advance,
        "eps_registered": EPS,
        "r1_eps_advance_all_passed": bool(
            all(v["passed"] for v in r1_advance.values())),
        "route_chains": chains,
        "cold_start": cold,
        "determinism": determinism,
        "router_content_hash": router_hash,
        "registry_transactional_hashes": tx_hashes,
        "mg_lorenz_descriptor_identical": bool(
            descs["mackey_glass"].canonical() == descs["lorenz"].canonical()),
        "fingerprints": {n: fp_lists[n] for n in R1_TASKS + ["audio"]},
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(f"M171 complete -> {output_dir / 'evidence.json'}", flush=True)
    print(json.dumps({"r1": r1, "r1_all_passed":
                      evidence["r1_all_passed"], "cold_start": cold},
                     indent=1)[:1600], flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m171(args.config, args.output)


if __name__ == "__main__":
    main()
