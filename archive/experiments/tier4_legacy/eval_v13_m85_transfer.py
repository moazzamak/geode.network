"""M85: does the v13 dictionary transfer, and how much of the answer is resolution?

The registered transfer corpus is CIFAR-100 and the v13 corpus is DomainNet at
256 px or better. Those differ in two ways at once, and N85.2 measured that the
second one is not small: degrading corpus images to CIFAR-100's 32x32 costs 41 %
of nearest-class-mean accuracy and displaces a row further than the distance to
its own class mean. So the evaluation runs three sources, not two:

* **native** — the corpus's held-out evaluation rows.
* **degraded** — the identical images at 32x32 through the identical graph.
* **cifar100** — the registered transfer corpus.

The operand is **retention** (N85.8): sparse-probe accuracy over dense-probe
accuracy on the same rows, split and budget. A 128-way DomainNet accuracy and a
20-way CIFAR-100 accuracy are not comparable and R7 forbids reading them as if
they were; what can be compared is how much of the raw features' value each
dictionary coding keeps. Width is matched as well as budget (N85.9), and every
cell carries its label-shuffled null and the free k-NN bar (N85.10).
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    sha256_file,
    write_canonical_json,
)
from experiments.common.v13_boundary import (
    domain_matched_partition,
    domain_stratified_halves,
)
from experiments.common.v13_linear_probe import (
    balanced_accuracy,
    dense_probe_accuracy,
    sparse_probe_accuracy,
)
from experiments.common.v13_sparse_dictionary import (
    SparseCodes,
    fit_sparse_dictionary,
    random_dictionary,
)
from experiments.tier4.eval_v13_m80_sparse_dictionary import (
    _load_corpus,
    _verify_corpus,
)
from experiments.tier4.eval_v13_m84_exposure_ladder import _resolve

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments" / "configs" / "v13" / "m85_transfer_eval.json"
)


def _verify_transfer(specification: dict[str, Any]) -> Path:
    index_path = _resolve(specification["path"])
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for artifact in index["artifacts"]:
        artifact_path = index_path.parent / artifact["path"]
        if sha256_file(artifact_path) != artifact["sha256"]:
            raise ValueError(f"M85 transfer artifact hash mismatch: {artifact_path}")
    evidence = json.loads(
        (index_path.parent / "evidence.json").read_text(encoding="utf-8")
    )
    if evidence["feature_hash"] != specification["feature_hash"]:
        raise ValueError("M85 transfer features are not the sealed ones")
    for control in ("reproduction_control", "shard_invariance_control"):
        if not evidence[control]["passes"]:
            raise ValueError(f"M85 transfer artifact failed its {control}")
    return index_path


def _split(
    labels: np.ndarray,
    *,
    class_count: int,
    train_per_class: int,
    test_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class train and test rows, drawn without replacement at this seed.

    Classes outside ``class_count`` are dropped rather than remapped, which is
    what makes the 20-way DomainNet cell a genuine subset of the 128-way one
    instead of a relabelled version of it.
    """
    generator = np.random.default_rng(seed)
    train: list[np.ndarray] = []
    test: list[np.ndarray] = []
    for label in range(class_count):
        rows = np.flatnonzero(labels == label)
        needed = train_per_class + test_per_class
        if len(rows) < needed:
            raise ValueError(
                f"class {label} holds {len(rows)} rows against the registered {needed}"
            )
        drawn = generator.permutation(rows)[:needed]
        train.append(drawn[:train_per_class])
        test.append(drawn[train_per_class:])
    return np.concatenate(train), np.concatenate(test)


def _knn_accuracy(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    *,
    k: int,
) -> float:
    """The free-composability bar: majority vote over the k nearest train rows."""
    neighbours = NearestNeighbors(n_neighbors=min(k, len(train_features)))
    neighbours.fit(train_features)
    _, indices = neighbours.kneighbors(test_features)
    voted = train_labels[indices]
    predicted = np.array(
        [np.bincount(row).argmax() for row in voted], dtype=np.int64
    )
    return balanced_accuracy(predicted, test_labels)


def _subset(codes: SparseCodes, rows: np.ndarray) -> SparseCodes:
    return SparseCodes(
        indices=codes.indices[rows],
        values=codes.values[rows],
        dictionary_size=codes.dictionary_size,
    )


def _cell(
    cell: dict[str, Any],
    *,
    features: np.ndarray,
    labels: np.ndarray,
    codes: SparseCodes,
    random_codes: SparseCodes,
    probe: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    class_count = int(cell["class_count"])
    train_rows, test_rows = _split(
        labels,
        class_count=class_count,
        train_per_class=int(cell["train_per_class"]),
        test_per_class=int(cell["test_per_class"]),
        seed=seed,
    )
    budget = {
        "epochs": int(probe["epochs"]),
        "batch_size": int(probe["batch_size"]),
        "learning_rate": float(probe["learning_rate"]),
        "seed": seed,
        "class_count": class_count,
    }
    train_labels = labels[train_rows]
    test_labels = labels[test_rows]
    shuffled = np.random.default_rng(seed + 1).permutation(train_labels)

    dense = dense_probe_accuracy(
        features[train_rows], train_labels, features[test_rows], test_labels, **budget
    )
    sparse = sparse_probe_accuracy(
        _subset(codes, train_rows),
        train_labels,
        _subset(codes, test_rows),
        test_labels,
        **budget,
    )
    sparse_random = sparse_probe_accuracy(
        _subset(random_codes, train_rows),
        train_labels,
        _subset(random_codes, test_rows),
        test_labels,
        **budget,
    )
    dense_null = dense_probe_accuracy(
        features[train_rows], shuffled, features[test_rows], test_labels, **budget
    )
    sparse_null = sparse_probe_accuracy(
        _subset(codes, train_rows),
        shuffled,
        _subset(codes, test_rows),
        test_labels,
        **budget,
    )
    knn = _knn_accuracy(
        features[train_rows],
        train_labels,
        features[test_rows],
        test_labels,
        k=int(probe["knn_k"]),
    )
    return {
        "train_rows": int(len(train_rows)),
        "test_rows": int(len(test_rows)),
        "chance": 1.0 / class_count,
        "dense_probe": dense,
        "sparse_probe": sparse,
        "sparse_probe_random_dictionary": sparse_random,
        "knn": knn,
        "dense_probe_shuffled_null": dense_null,
        "sparse_probe_shuffled_null": sparse_null,
        "retention": sparse / dense if dense > 0 else None,
        "retention_random_dictionary": sparse_random / dense if dense > 0 else None,
    }


def _mean(values: list[float]) -> float:
    return float(np.mean(values))


def _gate(evidence: dict[str, Any]) -> dict[str, Any]:
    """Nulls first, then the resolution control, then the transfer reading."""
    cells = evidence["cells"]
    chance_failures = [
        name
        for name, cell in cells.items()
        if cell["sparse_probe_shuffled_null"]["mean"] > 4.0 * cell["chance"]
        or cell["dense_probe_shuffled_null"]["mean"] > 4.0 * cell["chance"]
    ]
    if chance_failures:
        return {
            "verdict": "null_not_at_chance",
            "reason": (
                "a label-shuffled probe scored well above chance in "
                f"{sorted(chance_failures)}, so the split leaks and no figure "
                "above it is evidence"
            ),
            "transfer_holds": False,
        }

    native = cells["native_20"]["retention"]["mean"]
    degraded = cells["degraded_20"]["retention"]["mean"]
    cifar = cells["cifar100_20_matched"]["retention"]["mean"]
    margin = float(evidence["reporting"]["decisive_margin"])

    resolution_cost = native - degraded
    corpus_cost = degraded - cifar
    if abs(native - cifar) <= margin:
        verdict = "retention_transfers"
    elif resolution_cost > margin and abs(corpus_cost) <= margin:
        verdict = "loss_is_resolution_not_corpus"
    elif corpus_cost > margin:
        verdict = "loss_is_corpus_beyond_resolution"
    else:
        verdict = "loss_is_resolution_and_corpus"
    return {
        "verdict": verdict,
        "width_matched_retention": {
            "native_20": native,
            "degraded_20": degraded,
            "cifar100_20_matched": cifar,
        },
        "resolution_cost": resolution_cost,
        "corpus_cost_beyond_resolution": corpus_cost,
        "decisive_margin": margin,
        "transfer_holds": bool(verdict == "retention_transfers"),
        "reason": (
            "N85.8. Retention is the operand because a 128-way DomainNet "
            "accuracy and a 20-way CIFAR-100 accuracy are not comparable. The "
            "degraded arm carries the resolution effect, so the corpus effect "
            "is what remains after it."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    started = time.time()
    torch.set_num_threads(1)

    corpus_index = _verify_corpus(config["corpus"])
    features, labels = _load_corpus(corpus_index)
    manifest = json.loads(
        (corpus_index.parent / "selection_manifest.json").read_text(encoding="utf-8")
    )
    domains = np.asarray(
        [int(record["domain"]) for record in manifest["selection"]], dtype=np.int64
    )
    partition = config["partition"]
    fit_rows, evaluation_rows, partition_report = domain_matched_partition(
        labels,
        domains,
        quota=tuple(int(value) for value in partition["evaluation_domain_quota"]),
        fit_per_class=int(partition["fit_per_class"]),
        domain_count=int(config["corpus"]["domain_count"]),
    )
    domain_stratified_halves(labels, domains, evaluation_rows)

    transfer_index = _verify_transfer(config["transfer"])
    arrays = transfer_index.parent / "arrays"
    degraded_features = np.load(arrays / "degraded_features.npy")
    degraded_rows = np.load(arrays / "degraded_corpus_rows.npy")
    cifar_features = np.load(arrays / "cifar100_features.npy")
    cifar_labels = np.load(arrays / "cifar100_labels.npy").astype(np.int64)
    if not np.array_equal(degraded_rows, evaluation_rows):
        raise ValueError(
            "the degraded artifact was built on different evaluation rows than "
            "this run computes, so the resolution control is not paired"
        )

    sources = {
        "native": (features[evaluation_rows], labels[evaluation_rows]),
        "degraded": (degraded_features, labels[evaluation_rows]),
        "cifar100": (cifar_features, cifar_labels),
    }

    dictionary_config = config["dictionary"]
    collected: dict[str, dict[str, list[float]]] = {}
    for seed in config["seeds"]:
        print(f"seed {seed}: fitting the dictionary on {len(fit_rows)} rows ...")
        dictionary, _ = fit_sparse_dictionary(
            features[fit_rows],
            dictionary_size=int(dictionary_config["dictionary_size"]),
            active_atoms=int(dictionary_config["active_atoms"]),
            epochs=int(dictionary_config["epochs"]),
            batch_size=int(dictionary_config["batch_size"]),
            learning_rate=float(dictionary_config["learning_rate"]),
            seed=seed,
        )
        untrained = random_dictionary(
            features[fit_rows],
            dictionary_size=int(dictionary_config["dictionary_size"]),
            active_atoms=int(dictionary_config["active_atoms"]),
            seed=seed,
        )
        coded = {
            name: (dictionary.codes(rows), untrained.codes(rows))
            for name, (rows, _) in sources.items()
        }
        for cell in config["cells"]:
            source_features, source_labels = sources[cell["source"]]
            codes, random_codes = coded[cell["source"]]
            result = _cell(
                cell,
                features=source_features,
                labels=source_labels,
                codes=codes,
                random_codes=random_codes,
                probe=config["probe"],
                seed=seed,
            )
            bucket = collected.setdefault(cell["name"], {})
            for key, value in result.items():
                bucket.setdefault(key, []).append(value)
            print(
                f"  {cell['name']:<22} dense {result['dense_probe']:.4f}"
                f"  sparse {result['sparse_probe']:.4f}"
                f"  retention {result['retention']:.4f}"
                f"  knn {result['knn']:.4f}"
            )

    cells: dict[str, Any] = {}
    for name, bucket in collected.items():
        entry: dict[str, Any] = {}
        for key, values in bucket.items():
            if key in ("train_rows", "test_rows", "chance"):
                entry[key] = values[0]
            else:
                entry[key] = {
                    "mean": _mean(values),
                    "per_seed": [float(value) for value in values],
                    "spread": float(max(values) - min(values)),
                }
        cells[name] = entry

    evidence: dict[str, Any] = {
        "milestone": "M85",
        "component": "transfer_evaluation",
        "generated_at": datetime.now(UTC).isoformat(),
        "registered_question": config["registered_question"],
        "registration_notes": config["registration_notes"],
        "corpus": config["corpus"],
        "transfer": config["transfer"],
        "partition": partition_report,
        "dictionary": dictionary_config,
        "probe": config["probe"],
        "seeds": config["seeds"],
        "reporting": config["reporting"],
        "cells": cells,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "runtime_seconds": None,
    }
    evidence["gate"] = _gate(evidence)
    evidence["runtime_seconds"] = round(time.time() - started, 2)
    evidence["evidence_hash"] = payload_hash(
        {key: value for key, value in evidence.items() if key != "generated_at"}
    )

    output_dir = _resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)

    gate = evidence["gate"]
    print(f"\nverdict         {gate['verdict']}")
    print(f"evidence_hash   {evidence['evidence_hash']}")
    print(f"runtime         {evidence['runtime_seconds'] / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
