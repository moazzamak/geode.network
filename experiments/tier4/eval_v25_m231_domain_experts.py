"""M231 - per-domain expert ridge heads on the native-resolution
features cached by M230 (no re-extraction).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` (M231
REGISTERED, 20 Aug). The M229 ladder bars are for per-domain EXPERTS;
M230 scored a single global model per domain. This cell fits, for
each of the six domains, a ridge on THAT domain's train rows only
(ms / dino / hybrid arms, fixed penalties) and scores it on that
domain's sealed test rows.

Gates: g1 the ms GLOBAL fit reproduces the anchor at 1e-9 (the
instrument contract); g2 the schedule permutation reconstructs the
labels byte-exactly (alignment).
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
                  / "m231_domain_experts.json")
DEFAULT_OUTPUT = (REPO_ROOT / "logs" / "results" / "v25"
                  / "m231_domain_experts")

FULL_TRAIN_ROWS = 409832
CLASSES = 345
BLOCK = 4096


def _fit_one(rows: np.ndarray, features: np.ndarray, labels: np.ndarray,
             penalties: list[float]):
    acc = RidgeAccumulator(features.shape[1], CLASSES)
    for s in range(0, len(rows), BLOCK):
        e = min(s + BLOCK, len(rows))
        acc.add(np.asarray(features[rows[s:e]]), labels[rows[s:e]])
    weights = acc.solve_many(penalties)
    return acc.standardiser(), weights


def _score(weights: np.ndarray, std, test_feat: np.ndarray,
           test_labels: np.ndarray) -> float:
    hits = 0
    for s in range(0, len(test_feat), BLOCK):
        e = min(s + BLOCK, len(test_feat))
        scores = (std(np.asarray(test_feat[s:e])).astype(np.float64)
                  @ weights[:-1] + weights[-1])
        hits += int((np.argmax(scores, axis=1)
                     == test_labels[s:e]).sum())
    return hits / len(test_feat)


def run_m231(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    started = time.time()
    configure_external_cache_environment()
    root = data_cache_root()

    print("sealed corpus + schedule permutation", flush=True)
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

    print("loading cached M230 native features (raw order -> schedule)",
          flush=True)
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
    penalties = [float(p) for p in config["penalties"]]

    # g1: the ms GLOBAL fit reproduces the anchor at 1e-9
    std_ms, w_ms = _fit_one(np.arange(FULL_TRAIN_ROWS), train_ms, labels,
                            penalties)
    anchor = float(config["anchor"]["value"])
    tol = float(config["anchor"]["tolerance"])
    anchor_measured = _score(w_ms[1.0], std_ms, test_ms, test_labels)
    g1_ok = abs(anchor_measured - anchor) <= tol
    print(f"g1: {anchor_measured:.17f} delta "
          f"{anchor_measured - anchor:.3e} ok={g1_ok}", flush=True)
    del std_ms, w_ms

    arms = {"ms": (train_ms, test_ms),
            "dino": (train_dino, test_dino),
            "hybrid": (np.concatenate([train_ms, train_dino], axis=1),
                       np.concatenate([test_ms, test_dino], axis=1))}

    tables: dict[str, dict[str, Any]] = {}
    ladder = config["ladder"]
    for arm_name, (tr_f, te_f) in arms.items():
        tables[arm_name] = {}
        for d, name in enumerate(config["domain_names"]):
            rows = np.flatnonzero(train_domains == d)
            trows = np.flatnonzero(test_domains == d)
            std, weights = _fit_one(rows, tr_f, labels, penalties)
            tables[arm_name][name] = {
                str(p): _score(weights[p], std, te_f[trows],
                               test_labels[trows]) for p in penalties}
            print(f"{arm_name} {name}: "
                  f"{ {k: round(v, 4) for k, v in tables[arm_name][name].items()} }",
                  flush=True)

    def _verdict(tables: dict[str, dict[str, float]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in config["domain_names"]:
            vals = tables[name]
            best = max(vals.values())
            if name in ladder["easy"]:
                out[name] = {"best": round(best, 4),
                             "bar": ladder["easy_bar"],
                             "met": best >= ladder["easy_bar"]}
            elif name in ladder["middle"]:
                out[name] = {"best": round(best, 4),
                             "bar": ladder["middle_bar"],
                             "met": ladder["middle_bar"][0] <= best}
            else:
                out[name] = {"best": round(best, 4),
                             "bar": ladder["hard_first_bar"],
                             "met": ladder["hard_first_bar"][0] <= best}
        return out

    verdicts = {arm: _verdict(tables[arm]) for arm in arms}
    gates = {
        "g1_ms_global_anchor": {"ok": bool(g1_ok),
                                "measured": anchor_measured,
                                "sealed": anchor,
                                "delta": anchor_measured - anchor,
                                "tolerance": tol},
        "g2_schedule_alignment": {"ok": bool(g2_ok)},
    }
    gates_ok = all(g["ok"] for g in gates.values())

    evidence: dict[str, Any] = {
        "milestone": "M231",
        "cell": "per-domain expert ridge heads (native features)",
        "configuration_hash": payload_hash(config),
        "config_file": Path(config_path).name,
        "config": config,
        "per_domain_expert_accuracies": tables,
        "ladder_verdicts": verdicts,
        "gates": gates,
        "gates_ok": bool(gates_ok),
        "void": not gates_ok,
        "verdict": {
            "passes": bool(gates_ok),
            "reading": ("per-domain experts scored on their own domain "
                        "against the M229 ladder") if gates_ok
            else "a gate failed — VOID",
        },
        "runtime_seconds": round(time.time() - started, 2),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    print(json.dumps({"gates_ok": gates_ok, "tables": tables,
                      "verdicts": verdicts}, indent=1), flush=True)
    print(f"M231 complete -> {output_dir / 'evidence.json'}", flush=True)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_m231(args.config, args.output)


if __name__ == "__main__":
    main()
