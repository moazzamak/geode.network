"""M262 diagnostic — reproduce the solver defect mechanically.

At the full 392k-row MNLI fit, the first full run's logistic probe
(lbfgs, max_iter=500) read 0.5346 matched / 0.5422 mismatched —
barely above chance. The registered hypothesis: the iteration cap
left the fit unconverged (the standing lesson: an unconverged
optimiser is a one-sided instrument). This diagnostic fits BOTH
solvers on the SAME features at a scale where the effect shows,
captures the convergence state directly (n_iter_, warnings), and
prints the comparison. Scratch diagnostic, never evidence.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from experiments.common.data_cache import configure_external_cache_environment
from experiments.tier4.eval_v25_m262_language_arm import (
    _extract_features,
    _load_split,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    configure_external_cache_environment()
    import torch
    from datasets import load_dataset as _hf_load
    from sklearn.linear_model import LogisticRegression, Ridge
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").to(device)

    n_train = 30000
    n_eval = 500
    train_ds = _load_split(_hf_load, "multi_nli", "train", None)
    train_ds = train_ds.select(range(n_train))
    texts = [r["premise"] + " [SEP] " + r["hypothesis"] for r in train_ds]
    labels = [int(r["label"]) for r in train_ds]
    ev_ds = _load_split(_hf_load, "multi_nli", "validation_matched", None)
    ev_ds = ev_ds.select(range(n_eval))
    ev_texts = [r["premise"] + " [SEP] " + r["hypothesis"]
                for r in ev_ds]
    ev_labels = [int(r["label"]) for r in ev_ds]

    print("extracting features...", flush=True)
    tr_feat = _extract_features(model, tokenizer, texts, device, 64, 0.01)
    ev_feat = _extract_features(model, tokenizer, ev_texts, device, 64, 0.01)

    # ---- logistic, the first-run configuration ----------------------------
    mean = tr_feat.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(tr_feat.std(axis=0, dtype=np.float64), 1e-6)
    tr_norm = (tr_feat - mean) / std
    ev_norm = (ev_feat - mean) / std
    clf = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs",
                             random_state=20260821)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clf.fit(tr_norm, np.asarray(labels, dtype=np.int64))
    logistic_acc = float((clf.predict(ev_norm)
                          == np.asarray(ev_labels)).mean())
    conv_warnings = [str(w.message) for w in caught
                     if "CONVERG" in str(w.message).upper()
                     or "ITERATION" in str(w.message).upper()]
    print(json.dumps({
        "n_train": n_train,
        "logistic_n_iter_": int(clf.n_iter_[0]),
        "logistic_max_iter": 500,
        "logistic_hit_cap": bool(clf.n_iter_[0] >= 500),
        "convergence_warnings": conv_warnings,
        "logistic_matched_acc_500": logistic_acc,
    }, indent=1), flush=True)

    # ---- ridge, the replacement (closed-form, same features) ------------
    classes = sorted(set(labels))
    y = np.zeros((len(labels), len(classes)), dtype=np.float64)
    for i, label in enumerate(labels):
        y[i, classes.index(label)] = 1.0
    reg = Ridge(alpha=1.0)
    reg.fit(tr_norm, y)
    ridge_preds = reg.predict(ev_norm).argmax(axis=1)
    ridge_acc = float((ridge_preds == np.asarray(ev_labels)).mean())
    print(json.dumps({
        "ridge_matched_acc_500": ridge_acc,
        "reading": ("ridge closed-form beats the capped logistic on "
                    "identical features" if ridge_acc > logistic_acc
                    else "capped logistic not outperformed — revisit"),
    }, indent=1), flush=True)


if __name__ == "__main__":
    main()
