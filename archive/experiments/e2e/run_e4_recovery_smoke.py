"""Qualify E4 interruption recovery on real CIFAR-100 cached features."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from experiments.common.experiment_manifest import canonical_json
from experiments.e2e.e4_cifar_protocol import (
    build_id_partitions,
    load_config,
    load_e4_data,
)
from experiments.e2e.run_tier4_smoke import (
    InjectedStageFailure,
    run_resumable_tier4_smoke,
)
from src.runtime import PretrainingLane


def run_recovery(config_path: Path, runtime_root: Path) -> dict:
    config = load_config(config_path)
    data = load_e4_data(config)
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    feature_loads = 0

    def load_features():
        nonlocal feature_loads
        feature_loads += 1
        return data.id_features, data.id_labels

    def official_partitions(labels, seed):
        if len(labels) != len(data.id_labels) or not (labels == data.id_labels).all():
            raise ValueError("runner labels do not match the E4 data contract")
        partitions, _ = build_id_partitions(data, config, seed)
        return partitions

    common = {
        "runtime_root": runtime_root,
        "dataset_fingerprint": hashlib.sha256(
            canonical_json(data.fingerprints).encode("utf-8")
        ).hexdigest(),
        "feature_loader": load_features,
        "seed": int(config["deployment_seed"]),
        "pca_components": int(config["model"]["pca_components"]),
        "alpha": float(config["model"]["alpha"]),
        "max_iterations": int(config["model"]["max_iterations"]),
        "nudge_iterations": int(config["model"]["nudge_iterations"]),
        "pretraining_lane": PretrainingLane.CONTROLLED,
        "pretraining_source": "ImageNet-1K MobileNetV2 IMAGENET1K_V1",
        "partition_builder": official_partitions,
        "partition_strategy": "cifar100_official_test_e4_v1",
    }
    baseline = run_resumable_tier4_smoke(
        **common, attempt_id="uninterrupted",
    )
    loads_after_baseline = feature_loads
    try:
        run_resumable_tier4_smoke(
            **common, attempt_id="recovered", fail_during="class-10",
        )
    except InjectedStageFailure:
        pass
    else:
        raise AssertionError("E4 recovery qualification did not inject a failure")
    partial_path = (
        runtime_root / "runs" / baseline["run_id"] / "attempts"
        / "recovered" / "class-10.partial"
    )
    partial_visible = partial_path.is_dir()
    loads_before_resume = feature_loads
    recovered = run_resumable_tier4_smoke(
        **common, attempt_id="recovered",
    )
    summary = {
        "schema_version": 1,
        "milestone": "E4",
        "run_id": baseline["run_id"],
        "failure_stage": "class-10",
        "partial_stage_visible": partial_visible,
        "feature_loads": {
            "uninterrupted": loads_after_baseline,
            "recovered_before_resume": loads_before_resume - loads_after_baseline,
            "recovered_during_resume": feature_loads - loads_before_resume,
        },
        "assembly_hashes_match": (
            baseline["assembly_output_hashes"]
            == recovered["assembly_output_hashes"]
        ),
        "validation_metrics_match": (
            baseline["summary"]["validation_accuracy"]
            == recovered["summary"]["validation_accuracy"]
        ),
        "final_test_observed": recovered["summary"]["final_test_observed"],
        "recovered_complete": recovered["runtime_status"]["complete"],
    }
    summary["passed"] = (
        summary["partial_stage_visible"]
        and summary["feature_loads"]["recovered_during_resume"] == 0
        and summary["assembly_hashes_match"]
        and summary["validation_metrics_match"]
        and not summary["final_test_observed"]
        and summary["recovered_complete"]
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("experiments/configs/e4_cifar_qualification.json"),
    )
    parser.add_argument(
        "--runtime-root", type=Path,
        default=Path("data/e4_recovery_runtime"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("logs/results/e4_cifar_recovery.json"),
    )
    arguments = parser.parse_args()
    summary = run_recovery(arguments.config, arguments.runtime_root)
    if not summary["passed"]:
        raise RuntimeError(f"E4 recovery qualification failed: {summary}")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()