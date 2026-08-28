"""M233 - per-domain trained linear probes on the cached native
features (the training-cost lever the user authorised).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` (M233
REGISTERED, 20 Aug). The closed-form ridge plateaus; this cell trains
softmax probes (dino 384-d and hybrid 13,628-d inputs) per domain with
a FIXED Adam schedule - no test-set tuning, no early stopping - and
evaluates each on its own domain's sealed test rows exactly once.
Gates: g1 the ms anchor (probes do not touch it), g2 alignment,
g3 same-seed reproducibility (VOID on failure).
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
from experiments.tier4.eval_v15_m104_experts import (
    RidgeAccumulator,
    _load_domainnet,
)
from experiments.tier4.eval_v16_m109_trunk import _load_corpus
from experiments.tier4.eval_v16_m140_data_extension import _extension_indices
from experiments.tier4.eval_v16_m141_data_full import _rest_extension_indices

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m233_trained_probes.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m233_trained_probes")

FULL_TRAIN_ROWS = 409832
CLASSES = 345
BLOCK = 4096


def _train_probe(train_feat: np.ndarray, labels: np.ndarray,
                 epochs: int, lr: float, wd: float, batch: int,
                 seed: int, device) -> np.ndarray:
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    d = train_feat.shape[1]
    probe = nn.Linear(d, CLASSES).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
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
            out = probe(x[idx].to(device))
            loss = lossf(out, y[idx].to(device))
            loss.backward()
            opt.step()
    with torch.no_grad():
        return probe.weight.detach().cpu().numpy(), \
            probe.bias.detach().cpu().numpy()


def _probe_score(w: np.ndarray, b: np.ndarray, feat: np.ndarray,
                 labels: np.ndarray) -> float:
    hits = 0
    for s in range(0, len(feat), BLOCK):
        e = min(s + BLOCK, len(feat))
        scores = np.asarray(feat[s:e], dtype=np.float64) @ w.T + b
        hits += int((np.argmax(scores, axis=1) == labels[s:e]).sum())
    return hits / len(feat)


def run_m233(config_path: Path, output_dir: Path) -> dict[str, Any]:
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

    feat_src = REPO_ROOT / config["features_source"]
    train_feat_raw = np.load(feat_src / "native224_train_dino.npy",
                             mmap_mode="r")
    test_feat_raw = np.load(feat_src / "native224_test_dino.npy",
                            mmap_mode="r")
    train_dino = np.ascontiguousarray(train_feat_raw[perm])
    test_dino = np.ascontiguousarray(test_feat_raw[test_index])
    ms_cache = root / config["artifacts"]["cache_relpath"]
    ms_test_cache = root / config["artifacts"]["test_cache_relpath"]
    train_ms = np.asarray(np.load(
        ms_cache / config["artifacts"]["train_file"], mmap_mode="r"))
    test_ms = np.asarray(np.load(
        ms_test_cache / config["artifacts"]["test_file"], mmap_mode="r"))

    # g1: the ms global ridge anchor (probes do not touch it)
    acc = RidgeAccumulator(train_ms.shape[1], CLASSES)
    for s in range(0, FULL_TRAIN_ROWS, BLOCK):
        e = min(s + BLOCK, FULL_TRAIN_ROWS)
        acc.add(np.asarray(train_ms[s:e]), labels[s:e])
    w_ms = acc.solve_many([1.0])[1.0]
    std = acc.standardiser()
    hits = 0
    for s in range(0, len(test_ms), BLOCK):
        e = min(s + BLOCK, len(test_ms))
        scores = (std(np.asarray(test_ms[s:e])).astype(np.float64)
                  @ w_ms[:-1] + w_ms[-1])
        hits += int((np.argmax(scores, axis=1) == test_labels[s:e]).sum())
    anchor_measured = hits / len(test_ms)
    anchor = float(config["anchor"]["value"])
    tol = float(config["anchor"]["tolerance"])
    g1_ok = abs(anchor_measured - anchor) <= tol
    print(f"g1: {anchor_measured:.17f} ok={g1_ok}", flush=True)

    import torch
    device = torch.device("cuda")
    t = config["training"]
    epochs, lr, wd, batch, seed = (int(t["epochs"]), float(t["lr"]),
                                   float(t["weight_decay"]),
                                   int(t["batch"]), int(t["seed"]))

    arms = {"dino": (train_dino, test_dino),
            "hybrid": (np.concatenate([train_ms, train_dino], axis=1),
                       np.concatenate([test_ms, test_dino], axis=1))}
    tables: dict[str, dict[str, float]] = {}
    for arm_name, (tr_f, te_f) in arms.items():
        tables[arm_name] = {}
        for d, name in enumerate(config["domain_names"]):
            rows = np.flatnonzero(train_domains == d)
            trows = np.flatnonzero(test_domains == d)
            w, b = _train_probe(tr_f[rows], labels[rows], epochs, lr, wd,
                                batch, seed, device)
            tables[arm_name][name] = _probe_score(
                w, b, te_f[trows], test_labels[trows])
            print(f"{arm_name} probe {name}: {tables[arm_name][name]:.4f}",
                  flush=True)

    # g3: same-seed reproducibility on one domain probe (dino, real)
    rows = np.flatnonzero(train_domains == 4)
    trows = np.flatnonzero(test_domains == 4)
    w1, b1 = _train_probe(train_dino[rows], labels[rows], epochs, lr, wd,
                          batch, seed, device)
    w2, b2 = _train_probe(train_dino[rows], labels[rows], epochs, lr, wd,
                          batch, seed, device)
    s1 = _probe_score(w1, b1, test_dino[trows], test_labels[trows])
    s2 = _probe_score(w2, b2, test_dino[trows], test_labels[trows])
    g3_ok = bool(np.array_equal(w1, w2) and np.array_equal(b1, b2)
                 and s1 == s2)
    print(f"g3 reproducibility: {g3_ok}", flush=True)

    ladder = config["ladder"]
    verdicts: dict[str, dict[str, Any]] = {}
    for arm_name in arms:
        verdicts[arm_name] = {}
        for name in config["domain_names"]:
            v = tables[arm_name][name]
            if name in ladder["easy"]:
                verdicts[arm_name][name] = {"best": round(v, 4),
                                            "bar": ladder["easy_bar"],
                                            "met": v >= ladder["easy_bar"]}
            elif name in ladder["middle"]:
                verdicts[arm_name][name] = {
                    "best": round(v, 4), "bar": ladder["middle_bar"],
                    "met": ladder["middle_bar"][0] <= v}
            else:
                verdicts[arm_name][name] = {
                    "best": round(v, 4), "bar": ladder["hard_first_bar"],
                    "met": ladder["hard_first_bar"][0] <= v}

    gates = {
        "g1_ms_global_anchor": {"ok": bool(g1_ok),
                                "measured": anchor_measured,
                                "sealed": anchor,
                                "delta": anchor_measured - anchor,
                                "tolerance": tol},
        "g2_schedule_alignment": {"ok": bool(g2_ok)},
        "g3_reproducibility": {"ok": bool(g3_ok), "probe_scores": [s1, s2]},
    }
    gates_ok = all(g["ok"] for g in gates.values())

    evidence: dict[str, Any] = {
        "milestone": "M233",
        "cell": "per-domain trained linear probes",
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "per_domain_probe_accuracies": tables,
        "ladder_verdicts": verdicts,
        "gates": gates,
        "gates_ok": bool(gates_ok),
        "void": not gates_ok,
        "verdict": {
            "passes": bool(gates_ok),
            "reading": ("trained per-domain probes scored on their own "
                        "domain against the M229 ladder") if gates_ok
            else "a gate failed — VOID",
        },
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"gates_ok": gates_ok, "tables": tables,
                      "verdicts": verdicts}, indent=1), flush=True)
    print(f"M233 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m233(args.config, args.output)


if __name__ == "__main__":
    main()
