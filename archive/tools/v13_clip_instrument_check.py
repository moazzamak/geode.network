"""Validate the hand-written CLIP preprocessing by zero-shot retrieval.

If the transform is wrong, CLIP still emits vectors and naming still emits
words; only the words are meaningless. Zero-shot accuracy against the corpus's
own labels is the cheapest instrument check available.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from experiments.tier4.prepare_v13_clip_embeddings import (  # noqa: E402
    embed_images,
    embed_vocabulary,
    load_clip,
)

REPO = Path(__file__).resolve().parents[0]
config = json.loads((REPO / "experiments/configs/v13/m82_clip_embeddings.json").read_text())
vocabulary = json.loads((REPO / "logs/results/v13/naming_vocabulary/vocabulary.json").read_text())
manifest = json.loads((REPO / "logs/results/v13/domainnet_large/selection_manifest.json").read_text())
rows = manifest["selection"]

labels = np.array([r["class_label"] for r in rows])
domains = np.array([r["domain"] for r in rows])

# Stratified: 4 rows from each of the 128 classes.
generator = np.random.default_rng(20260729)
picked: list[int] = []
for label in range(128):
    candidates = np.flatnonzero(labels == label)
    picked.extend(generator.choice(candidates, size=4, replace=False).tolist())
picked = sorted(int(index) for index in picked)
sampled = [rows[index] for index in picked]

model, tokenizer, torch = load_clip(config, "cuda:0")
objects, styles = embed_vocabulary(
    model, tokenizer, torch, vocabulary, device="cuda:0", batch_size=256
)
embeddings = embed_images(model, torch, sampled, config, device="cuda:0", report_every=1 << 30)

truth = labels[picked]
truth_domain = domains[picked]

# In-corpus terms only: the 128 the corpus actually depicts.
in_corpus = objects[:128]
predicted = np.argmax(embeddings @ in_corpus.T, axis=1)
print(f"zero-shot over 128 in-corpus terms : {np.mean(predicted == truth):.4f}  (chance {1/128:.4f})")

# Full 345-term vocabulary, including the 217 far-field distractors.
predicted_full = np.argmax(embeddings @ objects.T, axis=1)
print(f"zero-shot over all 345 terms       : {np.mean(predicted_full == truth):.4f}")
print(f"far-field leakage (picks absent)   : {np.mean(predicted_full >= 128):.4f}")

# Style channel: does the style vocabulary recover the rendering domain?
style_predicted = np.argmax(embeddings @ styles.T, axis=1)
print(f"style term recovers domain         : {np.mean(style_predicted == truth_domain):.4f}  (chance {1/6:.4f})")

for domain, name in enumerate(["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]):
    mask = truth_domain == domain
    if mask.sum():
        print(
            f"  {name:10s} n={int(mask.sum()):4d}  object acc {np.mean(predicted[mask] == truth[mask]):.4f}"
            f"  style acc {np.mean(style_predicted[mask] == domain):.4f}"
        )
