"""M237 - quickdraw MLP probe on CLIP + dino-b concatenated features.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` (M237
REGISTERED, 20 Aug). The last cheap arm before the quickdraw wall is
declared: one hidden layer (1536 -> 512 -> 345), fixed AdamW, scored
once on the quickdraw sealed test.
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
from experiments.tier4.eval_v15_m104_experts import _load_domainnet
from experiments.tier4.eval_v16_m109_trunk import _load_corpus
from experiments.tier4.eval_v16_m140_data_extension import _extension_indices
from experiments.tier4.eval_v16_m141_data_full import _rest_extension_indices

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m237_quickdraw_mlp.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m237_quickdraw_mlp")

FULL_TRAIN_ROWS = 409832
CLASSES = 345
BLOCK = 4096
QUICKDRAW = 3


def _train_mlp(train_feat: np.ndarray, labels: np.ndarray, epochs: int,
               lr: float, wd: float, batch: int, hidden: int, seed: int,
               device) -> tuple[np.ndarray, ...]:
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    d = train_feat.shape[1]
    model = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                          nn.Linear(hidden, CLASSES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    n = len(train_feat)
    x = torch.from_numpy(np.asarray(train_feat, dtype=np.float32))
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, batch):
            e = min(s + batch, n)
            idx = perm[s:e]
            opt.zero_grad()
            loss = lossf(model(x[idx].to(device)), y[idx].to(device))
            loss.backward()
            opt.step()
    return tuple(p.detach().cpu().numpy() for p in model.parameters())


def _mlp_score(params, feat: np.ndarray, labels: np.ndarray) -> float:
    import torch
    import torch.nn as nn
    d, hidden = params[0].shape[1], params[2].shape[1]
    model = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                          nn.Linear(hidden, CLASSES))
    with torch.no_grad():
        for p, q in zip(model.parameters(), params):
            p.copy_(torch.from_numpy(q))
    hits = 0
    with torch.no_grad():
        for s in range(0, len(feat), BLOCK):
            e = min(s + BLOCK, len(feat))
            out = model(torch.from_numpy(
                np.asarray(feat[s:e], dtype=np.float32)))
            hits += int((out.argmax(dim=1)
                         == torch.from_numpy(labels[s:e])).sum())
    return hits / len(feat)


def run_m237(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    started = time.time()
    configure_external_cache_environment()
    root = data_cache_root()

    corpus, train_index, test_index = _load_corpus(config)
    raw = _load_domainnet(32)
    ext600_indices, _ = _extension_indices(raw["train_labels"], train_index,
                                           600, CLASSES)
    rest_indices = _rest_extension_indices(raw["train_labels"], train_index,
                                           CLASSES, per_class_take=200)
    perm = np.concatenate([train_index, ext600_indices, rest_indices])
    labels = np.load(root / config["artifacts"]["labels_file"])["labels"]
    g2_ok = (len(perm) == FULL_TRAIN_ROWS
             and np.array_equal(raw["train_labels"][perm], labels))
    train_domains = raw["train_domains"][perm]
    test_labels = corpus["test_labels"]
    test_domains = corpus["test_domains"]

    f = config["features"]
    clip_tr = np.load(REPO_ROOT / f["clip_train"], mmap_mode="r")
    clip_te = np.load(REPO_ROOT / f["clip_test"], mmap_mode="r")
    dino_tr = np.load(REPO_ROOT / f["dino_train"], mmap_mode="r")
    dino_te = np.load(REPO_ROOT / f["dino_test"], mmap_mode="r")
    tr = np.concatenate([np.asarray(clip_tr[perm]),
                         np.asarray(dino_tr[perm])], axis=1)
    te = np.concatenate([np.asarray(clip_te[test_index]),
                         np.asarray(dino_te[test_index])], axis=1)
    del clip_tr, clip_te, dino_tr, dino_te, raw
    import gc
    gc.collect()

    import torch
    device = torch.device("cuda")
    t = config["training"]
    epochs, lr, wd, batch, hidden, seed = (int(t["epochs"]), float(t["lr"]),
                                           float(t["weight_decay"]),
                                           int(t["batch"]),
                                           int(t["hidden"]), int(t["seed"]))

    rows = np.flatnonzero(train_domains == QUICKDRAW)
    trows = np.flatnonzero(test_domains == QUICKDRAW)
    params = _train_mlp(tr[rows], labels[rows], epochs, lr, wd, batch,
                        hidden, seed, device)
    score = _mlp_score(params, te[trows], test_labels[trows])
    print(f"quickdraw MLP probe: {score:.4f}", flush=True)

    params2 = _train_mlp(tr[rows], labels[rows], epochs, lr, wd, batch,
                         hidden, seed, device)
    score2 = _mlp_score(params2, te[trows], test_labels[trows])
    g3_ok = bool(all(np.array_equal(a, b) for a, b in zip(params, params2))
                 and score == score2)
    print(f"g3: {g3_ok}", flush=True)

    ladder = config["ladder"]
    met = score >= ladder["easy_bar"]
    gates = {"g2_schedule_alignment": {"ok": bool(g2_ok)},
             "g3_reproducibility": {"ok": bool(g3_ok)}}
    gates_ok = all(g["ok"] for g in gates.values())

    evidence: dict[str, Any] = {
        "milestone": "M237",
        "cell": "quickdraw MLP probe (CLIP + dino-b concat)",
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "quickdraw_probe_accuracy": score,
        "ladder_verdict": {"quickdraw": {"best": round(score, 4),
                                         "bar": ladder["easy_bar"],
                                         "met": bool(met)}},
        "wall_note": ("the quickdraw wall across backbones: dino-s "
                      "0.6040, dino-b 0.6302, CLIP-L 0.6267, "
                      f"MLP-concat {score:.4f}"),
        "gates": gates,
        "gates_ok": bool(gates_ok),
        "void": not gates_ok,
        "verdict": {
            "passes": bool(gates_ok),
            "reading": ("quickdraw >= 0.8 closes the ladder; otherwise "
                        "the wall is declared and the ladder closes at "
                        "5/6 easy bars") if gates_ok
            else "a gate failed — VOID",
        },
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"gates_ok": gates_ok, "quickdraw": score,
                      "met": met}, indent=1), flush=True)
    print(f"M237 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m237(args.config, args.output)


if __name__ == "__main__":
    main()
