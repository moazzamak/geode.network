"""M344 - the RFF reading on the sealed text axis (M262).

Registered in ``analysis/SCIENCE_LAYER_PLAN_2026-08-28.md`` (M344
REGISTRATION AMENDMENT, 28 Aug 2026, before the build). M262 sealed
the text recipe's base readings (frozen bert-base-uncased mean-pooled
features + closed-form ridge, alpha=1.0: SST-2 validation 0.8567,
IMDb test 0.8282, MNLI-m/mm 0.5374/0.5458). M344 asks whether the
M300 RFF map (D=16384, sigma=0.5, seed 20260828) lifts those
readings - i.e., is the text axis's remaining gap a linearity gap
the same way quickdraw's was?

Arms per task, both on the SAME re-derived frozen features (the M262
feature caches were not retained; extraction is deterministic - same
checkpoint, same tokenizer, same batch order, max_length 128, eval
mode):
- linear_reproduction: the sealed M262 ridge RE-FIT on the
  re-derived train features; its weights+bias sha256 must equal the
  sealed evidence weights_hash and its accuracies must equal the
  sealed accuracies (the g1 instrument-identity gate, strengthened
  by the g1 amendment: the probe caches are first-run logistic
  relics, recorded not used).
- rff: [features, phi(features)] with the same closed-form ridge
  head and the same alpha=1.0 (the g2 reading, scored once).

Gates: g1 premise (the three M262 probe caches load; the sealed
weights reproduce the sealed accuracies on the re-derived features
within 1e-9); g2 the RFF reading per task is scored once on the same
held-out splits.

Registered readings, written before the run: (a) RFF beats linear
by >= 0.01 on any task -> the breadth claim gains its second
modality and the M300 map is modality-portable; (b) a uniform null
-> the text features are already linear-sufficient at this scale and
the breadth claim rests on the recipe, not the map.
"""
from __future__ import annotations

import argparse
import hashlib
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
from experiments.tier4.eval_v25_m262_language_arm import (
    _extract_features,
    _load_split,
)
from experiments.tier4.eval_v26_m300_rff_quickdraw import (
    build_design,
    rff_params,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
M262_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
               / "m262_language_arm.json")
M262_CACHE_REL = "v25/m262_language_arm"
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v26"
                  / "m344_text_rff")

# the registered M300 map, applied to the text axis
RFF_DIM = 16384
RFF_SIGMA = 0.5
RFF_SEED = 20260828

RIDGE_ALPHA = 1.0          # the M262 sealed probe's alpha
REPRO_TOL = 1e-9           # g1: bitwise-scale reproduction

# the sealed M262 readings (logs/results/v25/m262_language_arm/
# evidence.json, 21 Aug 2026) - the g1 targets
M262_SEALED = {
    "sst2": {"validation": 0.856651376146789},
    "imdb": {"test": 0.8282},
    "nli": {"validation_matched": 0.5374426897605705,
            "validation_mismatched": 0.5457689178193653},
}
# the sealed weights+bias sha256 per task (the g1 hash targets);
# the M262 probe caches are first-run logistic relics and are
# recorded, not used (the g1 amendment)
M262_WEIGHTS_HASH = {
    "nli": "53ea3bfd21d13f43ab05df43fcdeeeb307665bb675eb12f31c9b1949a99a0de7",
    "sst2": "71439ef6a2027083ff0cd2859967d797fe00af0b9c274c7cd0049230fa07a847",
    "imdb": "219f9ce30773e4ca28e3550a397d0ba7044864d9a008ed4dfd0a6e32b74ee917",
}


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ridge_probe(tr_feat: np.ndarray, train_labels: list[int],
                 alpha: float) -> dict[str, Any]:
    """The M262 closed-form ridge least-squares probe on one-hot
    targets, verbatim (Gram solve; convergence-free, deterministic;
    per-column z-score fitted on the train split only)."""
    classes = sorted(set(train_labels))
    n_classes = len(classes)
    y = np.zeros((len(train_labels), n_classes), dtype=np.float64)
    for i, label in enumerate(train_labels):
        y[i, classes.index(label)] = 1.0
    mean = tr_feat.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(tr_feat.std(axis=0, dtype=np.float64), 1e-6)
    tr_norm = (tr_feat - mean) / std
    gram = tr_norm.T @ tr_norm
    rhs = tr_norm.T @ y
    w = np.linalg.solve(gram + alpha * np.eye(tr_norm.shape[1]),
                        rhs).astype(np.float32)
    b = (y.mean(axis=0) - (tr_norm.mean(axis=0) @ w))
    return {"weights": w, "bias": b, "mean": mean, "std": std,
            "classes": classes}


def _ridge_probe_design(tr_feat: np.ndarray,
                        train_labels: list[int], alpha: float,
                        omega: np.ndarray, phase: np.ndarray,
                        chunk: int = 8192) -> dict[str, Any]:
    """The same closed-form ridge head on the design form
    [features, phi(features)], fitted WITHOUT materialising the full
    design matrix: the per-column mean/std are computed on the
    features block only (the RFF block is a deterministic function
    of the features, so its statistics are accumulated in the same
    streaming pass), and the Gram/rhs are accumulated over row
    chunks of the design. The M300b memory lesson: at MNLI scale the
    full design is (392702, 17152) float32 = 27 GiB and the
    normalised copy doubles it - the streaming form needs one
    (chunk, 17152) buffer instead."""
    classes = sorted(set(train_labels))
    n_classes = len(classes)
    n, d = tr_feat.shape
    D = omega.shape[1]
    total = d + D

    # pass 1: streaming mean/std over the design columns
    s1 = np.zeros(total, dtype=np.float64)
    s2 = np.zeros(total, dtype=np.float64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        block = build_design(tr_feat[s:e], omega, phase)
        s1 += block.sum(axis=0, dtype=np.float64)
        s2 += (block.astype(np.float64) ** 2).sum(axis=0)
    mean = (s1 / n).astype(np.float32)
    std = np.maximum(np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2,
                                        0.0)), 1e-6).astype(np.float32)

    # pass 2: streaming Gram/rhs on the standardised design
    y = np.zeros((n, n_classes), dtype=np.float64)
    for i, label in enumerate(train_labels):
        y[i, classes.index(label)] = 1.0
    gram = np.zeros((total, total), dtype=np.float64)
    rhs = np.zeros((total, n_classes), dtype=np.float64)
    col_sum = np.zeros(total, dtype=np.float64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        block = ((build_design(tr_feat[s:e], omega, phase) - mean)
                 / std).astype(np.float64)
        gram += block.T @ block
        rhs += block.T @ y[s:e]
        col_sum += block.sum(axis=0)
    w = np.linalg.solve(gram + alpha * np.eye(total), rhs
                        ).astype(np.float32)
    b = (y.mean(axis=0) - (col_sum / n) @ w)
    return {"weights": w, "bias": b, "mean": mean, "std": std,
            "classes": classes}


def _score_design(probe: dict[str, Any], feats: np.ndarray,
                  omega: np.ndarray, phase: np.ndarray,
                  labels: list[int], chunk: int = 8192) -> float:
    """Score the design-form probe in row chunks (the same streaming
    discipline as the fit)."""
    hits = 0
    for s in range(0, len(feats), chunk):
        e = min(s + chunk, len(feats))
        block = build_design(feats[s:e], omega, phase)
        norm = (block - probe["mean"]) / probe["std"]
        scores = norm @ probe["weights"] + probe["bias"]
        preds = [probe["classes"][int(i)]
                 for i in scores.argmax(axis=1)]
        hits += int((np.asarray(preds, dtype=np.int64)
                     == np.asarray(labels[s:e])).sum())
    return hits / len(feats)


def _predict(probe: dict[str, Any], feats: np.ndarray) -> np.ndarray:
    norm = (feats - probe["mean"]) / probe["std"]
    scores = norm @ probe["weights"] + probe["bias"]
    return np.asarray([probe["classes"][int(i)]
                       for i in scores.argmax(axis=1)], dtype=np.int64)


def _accuracy(probe: dict[str, Any], feats: np.ndarray,
              labels: list[int]) -> float:
    preds = _predict(probe, feats)
    return float((preds == np.asarray(labels)).mean())


def _texts_of(ds) -> list[str]:
    return [r["premise"] + " [SEP] " + r["hypothesis"]
            if "premise" in r else (r.get("text") or r["sentence"])
            for r in ds]


def run_m344(output_dir: Path, smoke: bool = False) -> dict[str, Any]:
    config = json.loads(M262_CONFIG.read_text(encoding="utf-8"))
    started = time.time()
    configure_external_cache_environment()
    cache_root = data_cache_root() / M262_CACHE_REL

    import torch
    import transformers
    from datasets import load_dataset as _hf_load

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}; transformers {transformers.__version__}",
          flush=True)

    # ---- the frozen encoder, M262-identical --------------------------
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["encoder"]["checkpoint"])
    model = AutoModel.from_pretrained(config["encoder"]["checkpoint"])
    model.to(device)

    batch = int(config["extraction"]["batch"])
    throttle = float(config["extraction"]["throttle_seconds"])

    # smoke mode: the class-balanced M262 row caps (a pipeline check,
    # never a reading; the caps can never poison the full-run cache
    # because the cache key includes the row count)
    sm = config["smoke"] if smoke else {}
    train_rows = sm.get("train_rows")
    eval_rows = sm.get("eval_rows")

    def extract(name: str, split: str, texts: list[str]) -> np.ndarray:
        """Re-derive the M262 features, cached to disk under the M262
        convention ({task}_{split}_{nrows}_feat.npy) so a crashed run
        never pays for extraction twice. The M262 feature caches were
        not retained; extraction is deterministic (same checkpoint,
        same tokenizer, same batch order, max_length 128, eval mode),
        so the re-derived features are the M262 features."""
        safe = split.replace("/", "_")
        feat_path = cache_root / f"{name}_{safe}_{len(texts)}_feat.npy"
        if feat_path.exists():
            return np.load(feat_path, mmap_mode="r").copy()
        feats = _extract_features(model, tokenizer, texts, device,
                                  batch, throttle)
        np.save(feat_path, feats)
        return feats

    tasks: dict[str, Any] = {}

    def run_task(name: str, hf_id: str, train_split: str,
                 eval_splits: list[str], label_from_row,
                 config_name: str | None = None) -> dict[str, Any]:
        t0 = time.time()
        train_ds = _load_split(_hf_load, hf_id, train_split,
                               train_rows, config_name=config_name)
        train_texts = _texts_of(train_ds)
        train_labels = [label_from_row(r) for r in train_ds]
        tr_feat = extract(name, train_split, train_texts)

        # each eval split is extracted ONCE and held for both arms
        eval_feats: dict[str, np.ndarray] = {}
        eval_labels: dict[str, list[int]] = {}
        for split in eval_splits:
            ev_ds = _load_split(_hf_load, hf_id, split, eval_rows,
                                config_name=config_name)
            eval_feats[split] = extract(name, split, _texts_of(ev_ds))
            eval_labels[split] = [label_from_row(r) for r in ev_ds]

        # ---- g1: the sealed ridge, re-fit and re-scored end-to-end ---
        # (the g1 amendment: the probe caches are first-run logistic
        # relics; the sealed ridge is reproduced by RE-FITTING on the
        # re-derived features and hash-comparing weights+bias)
        sealed = M262_SEALED[name]
        refit_probe = _ridge_probe(tr_feat, train_labels, RIDGE_ALPHA)
        weights_hash = _sha256_hex(
            refit_probe["weights"].astype(np.float32).tobytes()
            + np.asarray(refit_probe["bias"],
                         dtype=np.float32).tobytes())
        hash_ok = weights_hash == M262_WEIGHTS_HASH[name]
        repro: dict[str, float] = {}
        g1_task = hash_ok
        print(f"  {name}: refit weights hash "
              f"{'MATCH' if hash_ok else 'MISMATCH'} "
              f"({weights_hash[:16]}...)", flush=True)
        for split in eval_splits:
            acc = _accuracy(refit_probe, eval_feats[split],
                            eval_labels[split])
            repro[split] = acc
            ok = abs(acc - sealed[split]) <= REPRO_TOL
            g1_task = g1_task and ok
            print(f"  {name}/{split}: linear repro {acc:.6f} "
                  f"(sealed {sealed[split]:.6f}, "
                  f"{'OK' if ok else 'DRIFT'})", flush=True)

        # ---- g2: the RFF arm, scored once per split -------------------
        omega, phase = rff_params(tr_feat.shape[1], RFF_DIM,
                                  RFF_SIGMA, RFF_SEED)
        rff_probe = _ridge_probe_design(tr_feat, train_labels,
                                        RIDGE_ALPHA, omega, phase)
        rff_accs: dict[str, float] = {}
        for split in eval_splits:
            acc = _score_design(rff_probe, eval_feats[split], omega,
                                phase, eval_labels[split])
            rff_accs[split] = acc
            print(f"  {name}/{split}: rff {acc:.6f} "
                  f"(linear {repro[split]:.6f}, "
                  f"delta {acc - repro[split]:+.6f})", flush=True)
        del eval_feats

        return {
            "train_rows": len(train_labels),
            "n_classes": len(rff_probe["classes"]),
            "linear_reproduction": repro,
            "rff": rff_accs,
            "g1_reproduction_ok": bool(g1_task),
            "seconds": round(time.time() - t0, 1),
        }

    nli = config["datasets"]["nli"]
    tasks["nli"] = run_task(
        "nli", nli["hf_id"], "train",
        ["validation_matched", "validation_mismatched"],
        lambda r: int(r["label"]))
    sst = config["datasets"]["sst2"]
    tasks["sst2"] = run_task(
        "sst2", sst["hf_id"], "train", ["validation"],
        lambda r: int(r["label"]), config_name=sst.get("config_name"))
    imdb = config["datasets"]["imdb"]
    tasks["imdb"] = run_task(
        "imdb", imdb["hf_id"], "train", ["test"],
        lambda r: int(r["label"]))

    # ---- the registered reading ---------------------------------------
    deltas = {f"{t}/{s}": tasks[t]["rff"][s]
              - tasks[t]["linear_reproduction"][s]
              for t in tasks for s in tasks[t]["rff"]}
    lifts = {k: v for k, v in deltas.items() if v >= 0.01}
    g1 = all(tasks[t]["g1_reproduction_ok"] for t in tasks)
    if lifts:
        reading = ("RFF lifts the sealed text readings by >= 0.01 on "
                   + ", ".join(sorted(lifts))
                   + ": the breadth claim gains its second modality "
                     "and the M300 map is recorded as "
                     "modality-portable")
    else:
        reading = ("a uniform null: no task lifts by >= 0.01 - the "
                   "text features are already linear-sufficient at "
                   "this scale and the breadth claim rests on the "
                   "recipe, not the map")

    gates = {
        "g1_sealed_ridge_reproduction": {
            "ok": bool(g1), "tolerance": REPRO_TOL,
            "form": ("re-fit on re-derived features; weights+bias "
                     "sha256 must equal the sealed evidence hash "
                     "(the g1 amendment: the probe caches are "
                     "first-run logistic relics, recorded not used)"),
            "sealed": M262_SEALED,
            "measured": {t: tasks[t]["linear_reproduction"]
                         for t in tasks}},
        "g2_rff_scored_once": {"ok": True},
    }
    gates_ok = all(g["ok"] for g in gates.values())

    evidence: dict[str, Any] = {
        "milestone": "M344",
        "cell": ("the RFF reading on the sealed text axis: does the "
                 "M300 map (D=16384, sigma=0.5) lift the M262 "
                 "readings?"),
        "admissible_as_evidence": not smoke,
        "smoke": smoke,
        "rff_map": {"D": RFF_DIM, "sigma": RFF_SIGMA,
                    "seed": RFF_SEED,
                    "source": "the registered M300 selection"},
        "ridge_alpha": RIDGE_ALPHA,
        "deltas": deltas,
        "lifts": lifts,
        "reading": reading,
        "m262_probe_cache_note": (
            "the M262 probe caches (F:/geode-ml/data/cache/v25/"
            "m262_language_arm/*_probe.npz) hold the FIRST run's "
            "logistic probes (sklearn coef/intercept); they "
            "hash-mismatch the sealed ridge evidence and are "
            "recorded, not used"),
        "gates": gates,
        "gates_ok": bool(gates_ok),
        "void": not gates_ok,
        "configuration_hash": payload_hash({
            "m262_config": config,
            "rff": {"D": RFF_DIM, "sigma": RFF_SIGMA,
                    "seed": RFF_SEED},
            "ridge_alpha": RIDGE_ALPHA}),
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"gates_ok": bool(gates_ok), "deltas": deltas,
                      "reading": reading}, indent=1), flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true",
                        help="class-balanced row caps (pipeline check, "
                             "never a reading)")
    args = parser.parse_args()
    run_m344(args.output, smoke=args.smoke)


if __name__ == "__main__":
    main()
