"""Run the three-seed M30 S2 directional-geometry confirmation."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    sha256_file,
    write_canonical_json,
)
from experiments.common.v5_frozen_representations import RepresentationManifest
from experiments.common.v5_protocol import DataStage, seeds_for_stage
from experiments.common.v5_statistics import (
    paired_prediction_interval,
    paired_seed_t_interval,
)
from experiments.common.v6_boundary_distillation import (
    fit_distilled_candidate_fields,
)
from experiments.common.v6_directional_distillation import (
    directional_field_matrix,
    generate_paired_directional_candidates,
    normalized_representation_hash,
    predict_directional_student,
    serialize_directional_student,
)
from experiments.common.v6_protocol import select_boundary_cohort
from experiments.tier4.eval_v5_frozen_space_heads import (
    fit_rbf_head,
    fit_weighted_knn_head,
)
from experiments.tier4.eval_v5_native_dinov2_sphere import _load_bound_cache
from experiments.tier4.eval_v6_directional_primitives import _candidate_labels
from experiments.tier4.eval_v6_subspace_primitives import _metrics
from src.directional_primitive import l2_normalize


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v6" / "m30_directional_s2.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v6" / "m30_directional_s2"


def _resolve(path: str) -> Path:
    return REPO_ROOT / Path(path)


def _load_seed_data(seed_input: dict[str, Any]) -> dict[str, Any]:
    feature_dir = _resolve(seed_input["feature_dir"])
    extraction_path = feature_dir / "extraction_summary.json"
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    extracted = extraction["representations"]["dinov2-small"]
    manifest = RepresentationManifest.from_dict(extracted["manifest"])
    if manifest.representation_hash != seed_input["parent_representation_hash"]:
        raise ValueError("S2 parent representation lineage mismatch.")
    if normalized_representation_hash(manifest.representation_hash) != seed_input[
        "directional_representation_hash"
    ]:
        raise ValueError("S2 directional representation lineage mismatch.")
    datasets = {
        split: _load_bound_cache(
            feature_dir,
            "dinov2-small",
            manifest,
            extracted["cache_metadata"][split],
            split,
        )
        for split in ("train", "dev")
    }
    return {
        "feature_dir": feature_dir,
        "extraction_path": extraction_path,
        "extraction": extraction,
        "manifest": manifest,
        "datasets": datasets,
    }


def _fit_seed(
    seed: int,
    seed_input: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    loaded = _load_seed_data(seed_input)
    train_features, train_labels = loaded["datasets"]["train"]
    dev_features, dev_labels = loaded["datasets"]["dev"]
    classes = np.unique(train_labels)

    started = time.perf_counter()
    teacher = fit_rbf_head(train_features, train_labels, seed)
    teacher_fit_seconds = time.perf_counter() - started
    teacher_model = teacher["model"]
    if not np.array_equal(teacher_model.classes_, classes):
        raise ValueError("S2 teacher class order mismatch.")
    teacher_probabilities = {
        "train": teacher_model.predict_proba(train_features),
        "development": teacher_model.predict_proba(dev_features),
    }

    knn_config = config["weighted_knn"]
    knn = fit_weighted_knn_head(
        train_features,
        train_labels,
        n_neighbors=int(knn_config["n_neighbors"]),
        temperature=float(knn_config["temperature"]),
        query_batch_size=int(knn_config["query_batch_size"]),
    )
    knn_model = knn["model"]
    knn_probabilities = knn_model.predict_proba(dev_features)
    knn_predictions = knn_model.predict(dev_features)
    knn_balanced_accuracy = _metrics(
        dev_labels,
        knn_predictions,
        knn_probabilities,
        classes,
        teacher_probabilities["development"],
    )["balanced_accuracy"]

    normalized_train = l2_normalize(train_features)
    objective = config["objective"]
    cohort = select_boundary_cohort(
        teacher_probabilities["train"],
        fraction=float(objective["cohort_fraction"]),
        minimum_count=int(objective["cohort_minimum_count"]),
    )
    cohort_indices = np.asarray(cohort["selected_indices"], dtype=np.int64)
    candidate_config = config["candidate_generation"]
    started = time.perf_counter()
    spheres, caps = generate_paired_directional_candidates(
        normalized_train,
        train_labels,
        teacher_probabilities["train"],
        classes,
        candidates_per_class=int(candidate_config["candidates_per_class"]),
        seed_size=int(candidate_config["seed_size"]),
        anchor_fraction=float(candidate_config["anchor_fraction"]),
    )
    candidate_generation_seconds = time.perf_counter() - started
    if [candidate.anchor_index for candidate in spheres] != [
        candidate.anchor_index for candidate in caps
    ] or any(
        sphere.support_size != len(cap.support_indices)
        for sphere, cap in zip(spheres, caps)
    ):
        raise RuntimeError("S2 paired candidate construction diverged.")

    variants: dict[str, Any] = {}
    students: dict[str, dict[str, Any]] = {}
    predictions: dict[str, np.ndarray] = {}
    probabilities: dict[str, np.ndarray] = {}
    candidates_by_geometry = {
        "euclidean_sphere": spheres,
        "cosine_cap": caps,
    }
    component_count = int(config["budget"]["component_count"])
    for geometry in config["geometries"]:
        candidates = candidates_by_geometry[geometry]
        fields = directional_field_matrix(
            candidates, normalized_train[cohort_indices], geometry
        )
        started = time.perf_counter()
        selection = fit_distilled_candidate_fields(
            fields,
            _candidate_labels(candidates),
            teacher_probabilities["train"][cohort_indices],
            train_labels[cohort_indices],
            classes,
            component_limit=component_count,
            teacher_weight=float(objective["teacher_weight"]),
            ground_truth_weight=float(objective["ground_truth_weight"]),
            complexity_penalty=float(objective["complexity_penalty"]),
            minimum_improvement=float(objective["minimum_improvement"]),
            initial_components_per_class=int(
                objective["initial_components_per_class"]
            ),
            exact_component_count=True,
        )
        student = serialize_directional_student(
            geometry=geometry,
            classes=classes,
            candidates=candidates,
            selection=selection,
            parent_representation_hash=seed_input["parent_representation_hash"],
            directional_representation_hash=seed_input[
                "directional_representation_hash"
            ],
            cohort_indices=cohort_indices,
            configuration={
                "seed": seed,
                "candidate_generation": candidate_config,
                "objective": objective,
                "budget": config["budget"],
            },
        )
        fit_seconds = time.perf_counter() - started
        geometry_predictions, geometry_probabilities = predict_directional_student(
            student,
            dev_features,
            parent_representation_hash=seed_input["parent_representation_hash"],
        )
        selected = [
            candidates[index] for index in selection["selected_candidate_indices"]
        ]
        variants[geometry] = {
            "selected_component_count": len(selected),
            "component_counts": selection["component_counts"],
            "parameter_count": int(
                sum(candidate.parameter_count for candidate in selected)
            ),
            "array_bytes": int(sum(candidate.array_bytes for candidate in selected)),
            "objective_initial": selection["objective_trajectory"][0],
            "objective_final": selection["objective_trajectory"][-1],
            "student_fit_seconds": fit_seconds,
            "development": _metrics(
                dev_labels,
                geometry_predictions,
                geometry_probabilities,
                classes,
                teacher_probabilities["development"],
            ),
        }
        students[geometry] = student
        predictions[geometry] = geometry_predictions
        probabilities[geometry] = geometry_probabilities

    return {
        "seed": seed,
        "parent_representation_hash": seed_input["parent_representation_hash"],
        "directional_representation_hash": seed_input[
            "directional_representation_hash"
        ],
        "split_hashes": loaded["extraction"]["split_hashes"],
        "extraction_summary_hash": sha256_file(loaded["extraction_path"]),
        "teacher_fit_seconds": teacher_fit_seconds,
        "candidate_generation_seconds": candidate_generation_seconds,
        "cohort": cohort,
        "weighted_knn_development_balanced_accuracy": knn_balanced_accuracy,
        "variants": variants,
        "students": students,
        "predictions": predictions,
        "probabilities": probabilities,
        "teacher_probabilities": teacher_probabilities,
        "development_labels": dev_labels,
    }


def run_evaluation(
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("schema_version") != 1
        or config.get("milestone") != "M30"
        or config.get("stage") != "S2"
        or config.get("geometries") != ["euclidean_sphere", "cosine_cap"]
        or config.get("teacher")
        != {
            "family": "rbf_svm",
            "C": 1.0,
            "kernel": "rbf",
            "calibration": "sigmoid",
            "calibration_folds": 3,
        }
        or config.get("test_labels_opened") is not False
    ):
        raise ValueError("Unsupported or test-open M30 S2 configuration.")
    seeds = seeds_for_stage(
        DataStage.S2, tuple(int(seed) for seed in config["seeds"])
    )
    if set(config["seed_inputs"]) != {str(seed) for seed in seeds}:
        raise ValueError("S2 seed inputs do not match the frozen stage seeds.")

    seed_results = [
        _fit_seed(seed, config["seed_inputs"][str(seed)], config) for seed in seeds
    ]
    cosine_values = np.asarray(
        [
            result["variants"]["cosine_cap"]["development"]["balanced_accuracy"]
            for result in seed_results
        ]
    )
    euclidean_values = np.asarray(
        [
            result["variants"]["euclidean_sphere"]["development"][
                "balanced_accuracy"
            ]
            for result in seed_results
        ]
    )
    seed_interval = paired_seed_t_interval(
        cosine_values,
        euclidean_values,
        confidence=float(config["statistics"]["confidence"]),
    )
    pooled_labels = np.concatenate(
        [result["development_labels"] for result in seed_results]
    )
    pooled_cosine = np.concatenate(
        [result["predictions"]["cosine_cap"] for result in seed_results]
    )
    pooled_euclidean = np.concatenate(
        [result["predictions"]["euclidean_sphere"] for result in seed_results]
    )
    pooled_interval = paired_prediction_interval(
        pooled_labels,
        pooled_cosine,
        pooled_euclidean,
        metric="balanced_accuracy",
        confidence=float(config["statistics"]["confidence"]),
        n_resamples=int(config["statistics"]["bootstrap_resamples"]),
        seed=int(config["statistics"]["bootstrap_seed"]),
    )
    cosine_nll = np.asarray(
        [
            result["variants"]["cosine_cap"]["development"]["nll"]
            for result in seed_results
        ]
    )
    euclidean_nll = np.asarray(
        [
            result["variants"]["euclidean_sphere"]["development"]["nll"]
            for result in seed_results
        ]
    )
    knn_values = np.asarray(
        [
            result["weighted_knn_development_balanced_accuracy"]
            for result in seed_results
        ]
    )
    mean_improvement = float(np.mean(cosine_values - euclidean_values))
    mean_euclidean_gap = float(np.mean(knn_values - euclidean_values))
    gap_closure_fraction = (
        mean_improvement / mean_euclidean_gap if mean_euclidean_gap > 0.0 else 0.0
    )
    nll_non_regression = float(np.mean(cosine_nll)) <= float(np.mean(euclidean_nll))
    parameter_matched = all(
        result["variants"]["cosine_cap"]["parameter_count"]
        == result["variants"]["euclidean_sphere"]["parameter_count"]
        for result in seed_results
    )
    gate = config["advancement_gate"]
    direct_path = (
        mean_improvement >= float(gate["minimum_mean_accuracy_improvement"])
        and (
            seed_interval["lower"] > 0.0
            if gate["paired_interval_must_exclude_zero"]
            else True
        )
    )
    gap_path = gap_closure_fraction >= float(
        gate["minimum_gap_closure_fraction"]
    ) and (
        nll_non_regression
        if gate["require_nll_non_regression_for_gap_closure"]
        else True
    )
    predictive_gate_passed = (
        (direct_path or gap_path)
        and (parameter_matched if gate["require_parameter_match"] else True)
    )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    serializable_seed_results: dict[str, Any] = {}
    deterministic_paths: list[str] = []
    for result in seed_results:
        seed = result["seed"]
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True)
        for geometry, student in result["students"].items():
            path = seed_dir / f"{geometry}_student.json"
            write_canonical_json(path, student)
            deterministic_paths.append(path.relative_to(output_dir).as_posix())
        arrays = {
            "development_labels": result["development_labels"],
            "teacher_train_probabilities": result["teacher_probabilities"]["train"],
            "teacher_development_probabilities": result["teacher_probabilities"][
                "development"
            ],
            **{
                f"{geometry}_development_predictions": result["predictions"][geometry]
                for geometry in config["geometries"]
            },
            **{
                f"{geometry}_development_probabilities": result["probabilities"][
                    geometry
                ]
                for geometry in config["geometries"]
            },
        }
        array_artifacts = {}
        for name, values in arrays.items():
            path = seed_dir / f"{name}.npy"
            np.save(path, values, allow_pickle=False)
            relative = path.relative_to(output_dir).as_posix()
            deterministic_paths.append(relative)
            array_artifacts[name] = {
                "path": relative,
                "sha256": sha256_file(path),
            }
        serializable_seed_results[str(seed)] = {
            key: value
            for key, value in result.items()
            if key
            not in {
                "students",
                "predictions",
                "probabilities",
                "teacher_probabilities",
                "development_labels",
            }
        }
        serializable_seed_results[str(seed)]["array_artifacts"] = array_artifacts

    evidence = {
        "schema_version": 1,
        "milestone": "M30",
        "stage": "S2",
        "configuration_hash": payload_hash(config),
        "seeds": list(seeds),
        "seed_results": serializable_seed_results,
        "mean_metrics": {
            "euclidean_development_balanced_accuracy": float(
                np.mean(euclidean_values)
            ),
            "cosine_development_balanced_accuracy": float(np.mean(cosine_values)),
            "weighted_knn_development_balanced_accuracy": float(np.mean(knn_values)),
            "euclidean_development_nll": float(np.mean(euclidean_nll)),
            "cosine_development_nll": float(np.mean(cosine_nll)),
        },
        "mean_accuracy_improvement": mean_improvement,
        "seed_paired_t_interval": seed_interval,
        "pooled_per_example_bootstrap_interval": pooled_interval,
        "mean_euclidean_to_knn_gap": mean_euclidean_gap,
        "gap_closure_fraction": gap_closure_fraction,
        "nll_non_regression": nll_non_regression,
        "parameter_matched": parameter_matched,
        "predictive_gate_operands": {
            "direct_mean_and_interval_path": direct_path,
            "gap_closure_and_nll_path": gap_path,
        },
        "predictive_gate_passed": predictive_gate_passed,
        "exact_replay_verified": False,
        "advancement_passed": False,
        "test_labels_opened": False,
        "deterministic_paths": deterministic_paths,
    }
    write_canonical_json(output_dir / "evidence.json", evidence)
    index = build_artifact_index(output_dir)
    return {
        "mean_accuracy_improvement": mean_improvement,
        "seed_interval_lower": seed_interval["lower"],
        "gap_closure_fraction": gap_closure_fraction,
        "predictive_gate_passed": predictive_gate_passed,
        "advancement_passed": False,
        "artifact_count": len(index["artifacts"]),
    }


def verify_replay(
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary:
        first = Path(temporary) / "first"
        second = Path(temporary) / "second"
        first_summary = run_evaluation(config_path, first)
        run_evaluation(config_path, second)
        evidence = json.loads((first / "evidence.json").read_text(encoding="utf-8"))
        if any(
            (first / relative).read_bytes() != (second / relative).read_bytes()
            for relative in evidence["deterministic_paths"]
        ):
            raise RuntimeError("M30 S2 student or prediction replay was not exact.")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(first, output_dir)
    evidence_path = output_dir / "evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["exact_replay_verified"] = True
    replay_required = bool(
        json.loads(config_path.read_text(encoding="utf-8"))["advancement_gate"][
            "require_exact_replay"
        ]
    )
    evidence["advancement_passed"] = bool(
        evidence["predictive_gate_passed"]
        and (evidence["exact_replay_verified"] if replay_required else True)
    )
    write_canonical_json(evidence_path, evidence)
    build_artifact_index(output_dir)
    return {
        **first_summary,
        "exact_replay_verified": True,
        "advancement_passed": evidence["advancement_passed"],
        "evidence_hash": sha256_file(evidence_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-replay", action="store_true")
    args = parser.parse_args()
    runner = verify_replay if args.verify_replay else run_evaluation
    print(json.dumps(runner(args.config, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
