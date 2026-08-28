"""M80 — sparse concept dictionary (GEODE v13, breakthrough arm stage 1).

Registered hypothesis H80: frozen DINOv2 embeddings admit an overcomplete sparse
decomposition whose atoms are substantially monosemantic, at a reconstruction
fidelity sufficient to preserve downstream accuracy.

Registered gate (`RESEARCH_IMPLEMENTATION_PLAN_v13.md` Section 7): advance if
linear-probe balanced accuracy on codes is within 3.0 points of the raw-feature
probe at mean active atoms <= 64. Otherwise sweep `m` and `k` once, then close
the arm.

Registration note N80.1 — an operand the registered gate does not cover. H80
asserts the atoms are "substantially monosemantic", but the four registered
operands (reconstruction R^2, mean active atoms, dead fraction, probe accuracy)
do not measure monosemanticity at all. Mean atom label entropy is therefore
reported here against a shuffled-label control, as a **reported, non-gating**
operand. The gate is left exactly as registered; the hypothesis is simply no
longer half-tested.

Threading contract: cells train single-threaded in parallel worker processes.
This was measured, not assumed. At 8 threads two identical fits of the largest
cell produced different parameters; at 1 thread they were bit-identical. Gated
evidence in this milestone therefore may not use intra-op parallelism.

Reproduce with::

    .\\.venv\\Scripts\\python.exe -m experiments.tier4.eval_v13_m80_sparse_dictionary
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    sha256_file,
    write_canonical_json,
)
from experiments.common.v13_linear_probe import (
    dense_probe_accuracy,
    sparse_probe_accuracy,
)
from experiments.common.v13_sparse_dictionary import (
    atom_label_entropy,
    fit_sparse_dictionary,
    random_dictionary,
    reconstruction_r2,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments" / "configs" / "v13" / "m80_sparse_dictionary.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v13" / "m80_sparse_dictionary"

_WORKER_STATE: dict[str, Any] = {}


def _resolve(path: str) -> Path:
    resolved = (REPO_ROOT / Path(path)).resolve()
    if REPO_ROOT.resolve() not in resolved.parents:
        raise ValueError("M80 paths must remain inside the repository")
    return resolved


def _verify_corpus(specification: dict[str, str]) -> Path:
    index_path = _resolve(specification["path"])
    if sha256_file(index_path) != specification["sha256"]:
        raise ValueError(f"M80 corpus index hash mismatch: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for artifact in index["artifacts"]:
        artifact_path = index_path.parent / artifact["path"]
        if sha256_file(artifact_path) != artifact["sha256"]:
            raise ValueError(f"M80 corpus artifact hash mismatch: {artifact_path}")
    return index_path


def _load_corpus(index_path: Path) -> tuple[np.ndarray, np.ndarray]:
    root = index_path.parent
    features = np.load(root / "arrays" / "features.npy").astype(np.float32)
    labels = np.load(root / "arrays" / "labels.npy").astype(np.int64)
    return features, labels


def _partition(
    labels: np.ndarray, *, fit_per_class: int, evaluation_per_class: int
) -> tuple[np.ndarray, np.ndarray]:
    """Class-stratified split. The corpus is class-major and uniform at 576."""
    fit_rows: list[np.ndarray] = []
    evaluation_rows: list[np.ndarray] = []
    for label in np.unique(labels):
        rows = np.flatnonzero(labels == label)
        if len(rows) < fit_per_class + evaluation_per_class:
            raise ValueError(f"M80 partition exceeds available rows for class {label}")
        fit_rows.append(rows[:fit_per_class])
        evaluation_rows.append(
            rows[fit_per_class : fit_per_class + evaluation_per_class]
        )
    return np.concatenate(fit_rows), np.concatenate(evaluation_rows)


def _cell_operands(
    fit_features: np.ndarray,
    fit_labels: np.ndarray,
    evaluation_features: np.ndarray,
    evaluation_labels: np.ndarray,
    *,
    config: dict[str, Any],
    dictionary_size: int,
    active_atoms: int,
    seed: int,
) -> dict[str, Any]:
    torch.set_num_threads(int(config["threading"]["torch_threads_per_worker"]))
    training = config["training"]
    probe = config["probe"]
    controls = config["controls"]
    class_count = int(len(np.unique(fit_labels)))

    dictionary, diagnostics = fit_sparse_dictionary(
        fit_features,
        dictionary_size=dictionary_size,
        active_atoms=active_atoms,
        epochs=int(training["epochs"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        seed=seed,
    )
    control = random_dictionary(
        fit_features,
        dictionary_size=dictionary_size,
        active_atoms=active_atoms,
        seed=int(controls["random_dictionary_seed"]),
    )

    fit_codes = dictionary.codes(fit_features)
    evaluation_codes = dictionary.codes(evaluation_features)
    control_fit_codes = control.codes(fit_features)
    control_evaluation_codes = control.codes(evaluation_features)

    usage = evaluation_codes.atom_usage()
    entropy = atom_label_entropy(
        evaluation_codes, evaluation_labels, class_count=class_count
    )
    shuffled = np.random.default_rng(
        int(controls["shuffled_label_seed"])
    ).permutation(evaluation_labels)
    shuffled_entropy = atom_label_entropy(
        evaluation_codes, shuffled, class_count=class_count
    )

    probe_arguments = {
        "class_count": class_count,
        "epochs": int(probe["epochs"]),
        "batch_size": int(probe["batch_size"]),
        "learning_rate": float(probe["learning_rate"]),
        "seed": seed,
    }
    codes_accuracy = sparse_probe_accuracy(
        fit_codes, fit_labels, evaluation_codes, evaluation_labels, **probe_arguments
    )
    control_accuracy = sparse_probe_accuracy(
        control_fit_codes,
        fit_labels,
        control_evaluation_codes,
        evaluation_labels,
        **probe_arguments,
    )

    operands = {
        "dictionary_size": int(dictionary_size),
        "active_atoms": int(active_atoms),
        "seed": int(seed),
        "reconstruction_r2": reconstruction_r2(
            evaluation_features, dictionary.reconstruct(evaluation_features)
        ),
        "random_control_reconstruction_r2": reconstruction_r2(
            evaluation_features, control.reconstruct(evaluation_features)
        ),
        "mean_active_atoms": float(evaluation_codes.active_atom_count().mean()),
        "dead_atom_fraction": float(np.mean(usage == 0)),
        "codes_probe_balanced_accuracy": codes_accuracy,
        "random_control_probe_balanced_accuracy": control_accuracy,
        "mean_atom_label_entropy_bits": entropy["mean_bits"],
        "shuffled_label_entropy_bits": shuffled_entropy["mean_bits"],
        "live_atoms": entropy["live_atoms"],
        "final_train_loss": diagnostics["final_train_loss"],
        "loss_decreased": diagnostics["loss_decreased"],
        "epoch_loss_trace": diagnostics["epoch_loss_trace"],
    }
    operands["state_hash"] = payload_hash(
        {key: value for key, value in operands.items() if key != "epoch_loss_trace"}
    )
    return operands


def _worker_initializer(index_path: str, config_text: str) -> None:
    torch.use_deterministic_algorithms(True)
    config = json.loads(config_text)
    torch.set_num_threads(int(config["threading"]["torch_threads_per_worker"]))
    features, labels = _load_corpus(Path(index_path))
    partition = config["partition"]
    fit_rows, evaluation_rows = _partition(
        labels,
        fit_per_class=int(partition["fit_per_class"]),
        evaluation_per_class=int(partition["evaluation_per_class"]),
    )
    _WORKER_STATE["config"] = config
    _WORKER_STATE["fit_features"] = features[fit_rows]
    _WORKER_STATE["fit_labels"] = labels[fit_rows]
    _WORKER_STATE["evaluation_features"] = features[evaluation_rows]
    _WORKER_STATE["evaluation_labels"] = labels[evaluation_rows]


def _worker_run(task: tuple[int, int, int]) -> dict[str, Any]:
    dictionary_size, active_atoms, seed = task
    return _cell_operands(
        _WORKER_STATE["fit_features"],
        _WORKER_STATE["fit_labels"],
        _WORKER_STATE["evaluation_features"],
        _WORKER_STATE["evaluation_labels"],
        config=_WORKER_STATE["config"],
        dictionary_size=dictionary_size,
        active_atoms=active_atoms,
        seed=seed,
    )


def _build_gate(
    cells: list[dict[str, Any]], raw_accuracy: float, config: dict[str, Any]
) -> dict[str, Any]:
    gate_config = config["gate"]
    tolerance = float(gate_config["probe_tolerance_points"])
    maximum_active = float(gate_config["maximum_mean_active_atoms"])

    eligible = [
        cell for cell in cells if cell["mean_active_atoms"] <= maximum_active
    ]
    best = max(eligible, key=lambda cell: cell["codes_probe_balanced_accuracy"])
    deficit = 100.0 * (raw_accuracy - best["codes_probe_balanced_accuracy"])

    controls_valid = all(
        cell["reconstruction_r2"] > cell["random_control_reconstruction_r2"]
        for cell in cells
    ) and all(
        cell["shuffled_label_entropy_bits"] > cell["mean_atom_label_entropy_bits"]
        for cell in cells
    )

    return {
        "raw_feature_probe_balanced_accuracy": raw_accuracy,
        "best_cell": {
            "dictionary_size": best["dictionary_size"],
            "active_atoms": best["active_atoms"],
            "codes_probe_balanced_accuracy": best["codes_probe_balanced_accuracy"],
            "mean_active_atoms": best["mean_active_atoms"],
            "reconstruction_r2": best["reconstruction_r2"],
        },
        "probe_deficit_points": deficit,
        "probe_tolerance_points": tolerance,
        "maximum_mean_active_atoms": maximum_active,
        "controls_discriminate": controls_valid,
        "all_cells_beat_random_dictionary": all(
            cell["codes_probe_balanced_accuracy"]
            > cell["random_control_probe_balanced_accuracy"]
            for cell in cells
        ),
        "h80_gate_passed": bool(deficit <= tolerance and controls_valid),
        "final_labels_opened": False,
    }


def run_m80(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    workers: int | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    config_text = config_path.read_text(encoding="utf-8")
    config = json.loads(config_text)

    index_path = _verify_corpus(config["corpus_index"])
    features, labels = _load_corpus(index_path)
    partition = config["partition"]
    fit_rows, evaluation_rows = _partition(
        labels,
        fit_per_class=int(partition["fit_per_class"]),
        evaluation_per_class=int(partition["evaluation_per_class"]),
    )

    torch.set_num_threads(int(config["threading"]["torch_threads_per_worker"]))
    probe = config["probe"]
    raw_accuracy = dense_probe_accuracy(
        features[fit_rows],
        labels[fit_rows],
        features[evaluation_rows],
        labels[evaluation_rows],
        class_count=int(len(np.unique(labels))),
        epochs=int(probe["epochs"]),
        batch_size=int(probe["batch_size"]),
        learning_rate=float(probe["learning_rate"]),
        seed=int(config["seeds"][0]),
    )

    tasks = [
        (int(dictionary_size), int(active_atoms), int(seed))
        for dictionary_size in config["grid"]["dictionary_size"]
        for active_atoms in config["grid"]["active_atoms"]
        for seed in config["seeds"]
    ]
    worker_count = workers or min(len(tasks), max(1, (os.cpu_count() or 2) - 2))
    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_worker_initializer,
        initargs=(str(index_path), config_text),
    ) as pool:
        cells = list(pool.map(_worker_run, tasks))

    evidence = {
        "schema_version": 1,
        "milestone": "M80",
        "program": "v13",
        "registered_hypothesis": "H80",
        "registration_notes": ["N80.1"],
        "configuration_hash": sha256_file(config_path),
        "corpus": {
            "name": "v13 DomainNet large",
            "index_sha256": config["corpus_index"]["sha256"],
            "rows": int(len(features)),
            "dimension": int(features.shape[1]),
            "classes": int(len(np.unique(labels))),
            "fit_rows": int(len(fit_rows)),
            "evaluation_rows": int(len(evaluation_rows)),
        },
        "worker_count": int(worker_count),
        "cells": cells,
        "gate": _build_gate(cells, raw_accuracy, config),
        "final_labels_opened": False,
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    build_artifact_index(output_dir)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=None)
    arguments = parser.parse_args()
    evidence = run_m80(arguments.config, arguments.output, workers=arguments.workers)

    header = (
        f"{'m':>6} {'k':>4} {'R2':>7} {'rndR2':>7} {'act':>6} "
        f"{'dead':>6} {'probe':>8} {'rndprobe':>9} {'bits':>6}"
    )
    print(header)
    print("-" * len(header))
    for cell in evidence["cells"]:
        print(
            f"{cell['dictionary_size']:>6} {cell['active_atoms']:>4} "
            f"{cell['reconstruction_r2']:>7.4f} "
            f"{cell['random_control_reconstruction_r2']:>7.4f} "
            f"{cell['mean_active_atoms']:>6.1f} "
            f"{cell['dead_atom_fraction'] * 100:>5.1f}% "
            f"{cell['codes_probe_balanced_accuracy'] * 100:>7.3f}% "
            f"{cell['random_control_probe_balanced_accuracy'] * 100:>8.3f}% "
            f"{cell['mean_atom_label_entropy_bits']:>6.2f}"
        )
    print()
    print(json.dumps(evidence["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
