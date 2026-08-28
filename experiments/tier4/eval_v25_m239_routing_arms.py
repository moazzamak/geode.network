"""M239 - deployable per-domain arms + routing measurement.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` (M239
REGISTERED, 20 Aug, dispatched on user direction). Makes the winning
recipe product-shaped: re-trains the per-domain CLIP probes (fixed
recipe) and PERSISTS the weights as arm artifacts, trains the 6-way
domain router, and measures the M229 routing bar on the sealed test:
routing accuracy, oracle-routed accuracy, router-routed accuracy.
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
from experiments.tier4.eval_v15_m104_experts import _load_domainnet
from experiments.tier4.eval_v16_m109_trunk import _load_corpus
from experiments.tier4.eval_v16_m140_data_extension import _extension_indices
from experiments.tier4.eval_v16_m141_data_full import _rest_extension_indices
from experiments.tier4.eval_v25_m233_trained_probes import (
    _probe_score,
    _train_probe,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (REPO_ROOT / "experiments" / "configs" / "v25"
                  / "m239_routing_arms.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25" / "m239_routing_arms")

FULL_TRAIN_ROWS = 409832
CLASSES = 345
DOMAINS = 6


def run_m239(config_path: Path, output_dir: Path) -> dict[str, Any]:
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
    clip_tr = np.asarray(np.load(REPO_ROOT / f["clip_train"], mmap_mode="r"))
    clip_te = np.asarray(np.load(REPO_ROOT / f["clip_test"], mmap_mode="r"))
    tr = np.ascontiguousarray(clip_tr[perm])
    te = np.ascontiguousarray(clip_te[test_index])
    del clip_tr, clip_te, raw
    import gc
    gc.collect()

    import torch
    device = torch.device("cuda")
    t = config["training"]
    epochs, lr, wd, batch, seed = (int(t["epochs"]), float(t["lr"]),
                                   float(t["weight_decay"]),
                                   int(t["batch"]), int(t["seed"]))
    names = config["domain_names"]

    # ---- the per-domain arms (weights persisted as artifacts) -----------
    output_dir.mkdir(parents=True, exist_ok=True)
    arms_dir = output_dir / "arms"
    arms_dir.mkdir(exist_ok=True)
    arm_weights: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    arm_digests: dict[str, str] = {}
    for d, name in enumerate(names):
        rows = np.flatnonzero(train_domains == d)
        w, b = _train_probe(tr[rows], labels[rows], epochs, lr, wd, batch,
                            seed, device)
        arm_weights[name] = (w, b)
        np.savez(arms_dir / f"arm_{name}.npz", weight=w, bias=b)
        arm_digests[name] = hashlib.sha256(
            (arms_dir / f"arm_{name}.npz").read_bytes()).hexdigest()
        print(f"arm {name}: {w.shape}, digest "
              f"{arm_digests[name][:12]}...", flush=True)

    # ---- the router: a 6-way domain classifier ---------------------------
    w_r, b_r = _train_probe(tr, train_domains.astype(np.int64), epochs, lr,
                            wd, batch, seed, device)
    pred_domains = np.empty(len(te), dtype=np.int64)
    for s in range(0, len(te), 4096):
        e = min(s + 4096, len(te))
        scores = np.asarray(te[s:e], dtype=np.float64) @ w_r.T + b_r
        pred_domains[s:e] = np.argmax(scores, axis=1)
    routing_acc = float((pred_domains == test_domains).mean())
    per_domain_recall = {
        name: float((pred_domains[test_domains == d]
                     == d).mean())
        for d, name in enumerate(names)}

    # ---- oracle vs router accuracy ---------------------------------------
    def _arm_score(name: str, feat: np.ndarray, lbl: np.ndarray) -> float:
        w, b = arm_weights[name]
        return _probe_score(w, b, feat, lbl)

    oracle = sum(
        _arm_score(name, te[test_domains == d],
                   test_labels[test_domains == d])
        * int((test_domains == d).sum()) for d, name in enumerate(names)
    ) / len(te)
    routed = sum(
        _arm_score(name, te[pred_domains == d],
                   test_labels[pred_domains == d])
        * int((pred_domains == d).sum()) for d, name in enumerate(names)
    ) / len(te)
    print(f"routing {routing_acc:.4f} | oracle {oracle:.4f} | "
          f"routed {routed:.4f}", flush=True)

    # ---- g3 reproducibility (router) -------------------------------------
    w_r2, b_r2 = _train_probe(tr, train_domains.astype(np.int64), epochs,
                              lr, wd, batch, seed, device)
    g3_ok = bool(np.array_equal(w_r, w_r2) and np.array_equal(b_r, b_r2))

    bar = float(config["routing_bar"])
    gates = {
        "g2_schedule_alignment": {"ok": bool(g2_ok)},
        "g3_reproducibility": {"ok": bool(g3_ok)},
        "g4_routing_bar": {"ok": bool(routing_acc >= bar),
                           "routing_accuracy": routing_acc,
                           "bar": bar},
    }
    gates_ok = all(g["ok"] for g in gates.values())

    evidence: dict[str, Any] = {
        "milestone": "M239",
        "cell": "deployable per-domain arms + routing measurement",
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "arm_artifacts": arm_digests,
        "routing_accuracy": routing_acc,
        "per_domain_routing_recall": per_domain_recall,
        "oracle_routed_accuracy": oracle,
        "router_routed_accuracy": routed,
        "routed_minus_oracle": routed - oracle,
        "gates": gates,
        "gates_ok": bool(gates_ok),
        "void": not gates_ok,
        "verdict": {
            "passes": bool(gates_ok),
            "reading": ("routing >= 0.8 meets the last unmeasured M229 "
                        "bar; the routed-vs-oracle gap is the realised "
                        "cost of routing") if gates_ok
            else "a gate failed — VOID",
        },
        "runtime_seconds": round(time.time() - started, 2),
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"gates_ok": gates_ok, "routing": routing_acc,
                      "oracle": oracle, "routed": routed,
                      "recall": per_domain_recall}, indent=1), flush=True)
    print(f"M239 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m239(args.config, args.output)


if __name__ == "__main__":
    main()
