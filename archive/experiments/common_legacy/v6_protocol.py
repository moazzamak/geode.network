from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.common.experiment_manifest import canonical_json
from experiments.common.v5_artifacts import sha256_file
from experiments.common.v5_protocol import require_sha256


@dataclass(frozen=True)
class TeacherLineage:
    family: str
    representation_hash: str
    training_split_hash: str
    development_split_hash: str
    checkpoint_hash: str
    prediction_hash: str
    selection_metric: str
    test_labels_used_for_selection: bool = False

    def validate(self) -> None:
        if not self.family or not self.selection_metric:
            raise ValueError("Teacher family and selection_metric are required.")
        for field in (
            "representation_hash",
            "training_split_hash",
            "development_split_hash",
            "checkpoint_hash",
            "prediction_hash",
        ):
            require_sha256(getattr(self, field), field)
        if self.test_labels_used_for_selection:
            raise PermissionError("Teacher selection must not use final-test labels.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "family": self.family,
            "representation_hash": self.representation_hash,
            "training_split_hash": self.training_split_hash,
            "development_split_hash": self.development_split_hash,
            "checkpoint_hash": self.checkpoint_hash,
            "prediction_hash": self.prediction_hash,
            "selection_metric": self.selection_metric,
            "test_labels_used_for_selection": self.test_labels_used_for_selection,
        }


def require_teacher_compatibility(
    teacher: TeacherLineage,
    *,
    representation_hash: str,
    training_split_hash: str,
    development_split_hash: str,
) -> None:
    teacher.validate()
    expected = {
        "representation_hash": require_sha256(
            representation_hash, "representation_hash"
        ),
        "training_split_hash": require_sha256(
            training_split_hash, "training_split_hash"
        ),
        "development_split_hash": require_sha256(
            development_split_hash, "development_split_hash"
        ),
    }
    for field, value in expected.items():
        if getattr(teacher, field) != value:
            raise ValueError(
                f"Teacher {field} {getattr(teacher, field)} does not match {value}."
            )


@dataclass(frozen=True)
class PrimitiveMetadata:
    family: str
    minimum_seed_rule: str
    score_semantics: str
    local_rank: int | None = None
    residual_scale: str | None = None
    direction: str | None = None
    angular_radius: str | None = None

    def validate(self) -> None:
        if self.family not in {"sphere", "subspace", "cosine_cap"}:
            raise ValueError(f"Unsupported primitive family {self.family!r}.")
        if not self.minimum_seed_rule or not self.score_semantics:
            raise ValueError("Primitive seed rule and score semantics are required.")
        if self.family == "subspace":
            if self.local_rank is None or self.local_rank < 1:
                raise ValueError("Subspace primitives require a positive local_rank.")
            if self.residual_scale not in {"isotropic", "diagonal"}:
                raise ValueError(
                    "Subspace primitives require isotropic or diagonal residual scale."
                )
        elif self.local_rank is not None or self.residual_scale is not None:
            raise ValueError(
                f"{self.family} primitives cannot declare subspace-only metadata."
            )
        if self.family == "cosine_cap":
            if self.direction != "unit_mean" or self.angular_radius != "radians":
                raise ValueError(
                    "Cosine caps require unit_mean direction and radians radius."
                )
        elif self.direction is not None or self.angular_radius is not None:
            raise ValueError(
                f"{self.family} primitives cannot declare directional metadata."
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "family": self.family,
            "minimum_seed_rule": self.minimum_seed_rule,
            "score_semantics": self.score_semantics,
            "local_rank": self.local_rank,
            "residual_scale": self.residual_scale,
            "direction": self.direction,
            "angular_radius": self.angular_radius,
        }


@dataclass(frozen=True)
class BudgetSpec:
    mode: str
    component_limit: int | None
    parameter_limit: int | None

    def validate(self) -> None:
        if self.mode not in {"component_matched", "parameter_matched"}:
            raise ValueError(f"Unsupported budget mode {self.mode!r}.")
        if self.mode == "component_matched":
            if self.component_limit is None or self.component_limit < 1:
                raise ValueError("component_matched requires a positive component_limit.")
            if self.parameter_limit is not None:
                raise ValueError("component_matched cannot set parameter_limit.")
        else:
            if self.parameter_limit is None or self.parameter_limit < 1:
                raise ValueError("parameter_matched requires a positive parameter_limit.")
            if self.component_limit is not None:
                raise ValueError("parameter_matched cannot set component_limit.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "mode": self.mode,
            "component_limit": self.component_limit,
            "parameter_limit": self.parameter_limit,
        }


def enumerate_budget_table(
    budgets: Sequence[BudgetSpec],
    primitive_ids: Sequence[str],
) -> list[dict[str, Any]]:
    if not budgets or not primitive_ids or len(set(primitive_ids)) != len(primitive_ids):
        raise ValueError("Budgets and unique primitive identifiers are required.")
    rows = [
        {"primitive_id": primitive_id, **budget.to_dict()}
        for primitive_id in primitive_ids
        for budget in budgets
    ]
    return sorted(rows, key=canonical_json)


def select_boundary_cohort(
    teacher_probabilities: np.ndarray,
    *,
    fraction: float,
    minimum_count: int,
) -> dict[str, Any]:
    probabilities = np.asarray(teacher_probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("Teacher probabilities must have shape (samples, classes>=2).")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Teacher probabilities must be finite.")
    if np.any(probabilities < 0.0) or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-8
    ):
        raise ValueError("Teacher probabilities must be normalized and non-negative.")
    if not 0.0 < fraction <= 1.0 or minimum_count < 1:
        raise ValueError("Boundary cohort fraction and minimum_count must be positive.")

    ordered = np.sort(probabilities, axis=1)
    margins = ordered[:, -1] - ordered[:, -2]
    count = min(
        len(probabilities),
        max(minimum_count, int(math.ceil(fraction * len(probabilities)))),
    )
    sample_indices = np.arange(len(probabilities), dtype=np.int64)
    boundary_order = np.lexsort((sample_indices, margins))
    selected = boundary_order[:count]
    return {
        "fraction": float(fraction),
        "minimum_count": int(minimum_count),
        "selected_count": int(count),
        "selected_indices": selected.tolist(),
        "selected_margins": margins[selected].tolist(),
    }


def _lookup(payload: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"Artifact does not contain expected path {dotted_path!r}.")
        value = value[part]
    return value


def validate_baseline_locks(
    locks: Sequence[Mapping[str, Any]],
    repository_root: str | Path,
) -> list[dict[str, Any]]:
    root = Path(repository_root).resolve()
    if not locks:
        raise ValueError("At least one frozen baseline lock is required.")
    verified = []
    for lock in locks:
        if set(lock) != {"artifact_id", "path", "sha256", "expected_values"}:
            raise ValueError("Baseline locks have an unsupported schema.")
        artifact_id = str(lock["artifact_id"])
        relative = Path(str(lock["path"]))
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError("Baseline lock paths must remain inside the repository.")
        expected_hash = require_sha256(str(lock["sha256"]), "sha256")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Frozen artifact {artifact_id!r} hash mismatch: "
                f"expected {expected_hash}, got {actual_hash}."
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected_values = lock["expected_values"]
        if not isinstance(expected_values, Mapping):
            raise ValueError("expected_values must be a mapping.")
        observed = {}
        for dotted_path, expected in expected_values.items():
            actual = _lookup(payload, str(dotted_path))
            if actual != expected:
                raise ValueError(
                    f"Frozen artifact {artifact_id!r} value mismatch at "
                    f"{dotted_path!r}: expected {expected!r}, got {actual!r}."
                )
            observed[str(dotted_path)] = actual
        verified.append(
            {
                "artifact_id": artifact_id,
                "path": relative.as_posix(),
                "sha256": actual_hash,
                "observed_values": observed,
            }
        )
    return verified


def validate_prediction_baseline(
    metadata_path: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    path = (root / metadata_path).resolve()
    if root not in path.parents:
        raise ValueError("Prediction baseline metadata must remain inside the repository.")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "milestone",
        "representation_hash",
        "split_hashes",
        "feature_hashes",
        "source_evidence_hash",
        "labels",
        "heads",
        "teacher_checkpoint",
    }
    if set(metadata) != required or metadata["schema_version"] != 1:
        raise ValueError("Unsupported prediction-baseline schema.")
    if metadata["milestone"] != "M27":
        raise ValueError("Prediction baseline must identify M27.")
    for field in ("representation_hash", "source_evidence_hash"):
        require_sha256(str(metadata[field]), field)

    labels_by_split: dict[str, np.ndarray] = {}
    for split in ("development", "test"):
        item = metadata["labels"][split]
        label_path = (root / item["path"]).resolve()
        if root not in label_path.parents:
            raise ValueError("Prediction baseline arrays must remain inside the repository.")
        if sha256_file(label_path) != require_sha256(item["sha256"], "labels.sha256"):
            raise ValueError(f"{split} label hash mismatch.")
        labels = np.load(label_path, allow_pickle=False)
        if labels.ndim != 1:
            raise ValueError(f"{split} labels must be one-dimensional.")
        labels_by_split[split] = labels

    observed_metrics: dict[str, dict[str, float]] = {}
    for head, splits in metadata["heads"].items():
        observed_metrics[head] = {}
        for split in ("development", "test"):
            item = splits[split]
            prediction_path = (root / item["predictions_path"]).resolve()
            probability_path = (root / item["probabilities_path"]).resolve()
            if root not in prediction_path.parents or root not in probability_path.parents:
                raise ValueError(
                    "Prediction baseline arrays must remain inside the repository."
                )
            if sha256_file(prediction_path) != require_sha256(
                item["predictions_sha256"], "predictions_sha256"
            ):
                raise ValueError(f"{head}/{split} prediction hash mismatch.")
            if sha256_file(probability_path) != require_sha256(
                item["probabilities_sha256"], "probabilities_sha256"
            ):
                raise ValueError(f"{head}/{split} probability hash mismatch.")
            predictions = np.load(prediction_path, allow_pickle=False)
            probabilities = np.load(probability_path, allow_pickle=False)
            labels = labels_by_split[split]
            if predictions.shape != labels.shape:
                raise ValueError(f"{head}/{split} predictions do not match labels.")
            if probabilities.ndim != 2 or probabilities.shape[0] != len(labels):
                raise ValueError(f"{head}/{split} probabilities have invalid shape.")
            if not np.all(np.isfinite(probabilities)) or not np.allclose(
                probabilities.sum(axis=1), 1.0, atol=1e-8
            ):
                raise ValueError(f"{head}/{split} probabilities are invalid.")
            classes = np.unique(labels)
            recalls = [
                np.mean(predictions[labels == class_id] == class_id)
                for class_id in classes
            ]
            balanced_accuracy = float(np.mean(recalls))
            if not np.isclose(
                balanced_accuracy,
                float(item["balanced_accuracy"]),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(f"{head}/{split} balanced accuracy mismatch.")
            observed_metrics[head][split] = balanced_accuracy

    checkpoint = metadata["teacher_checkpoint"]
    checkpoint_path = (root / checkpoint["path"]).resolve()
    if root not in checkpoint_path.parents:
        raise ValueError("Teacher checkpoint must remain inside the repository.")
    if sha256_file(checkpoint_path) != require_sha256(
        checkpoint["sha256"], "teacher_checkpoint.sha256"
    ):
        raise ValueError("Teacher checkpoint hash mismatch.")
    return {
        "metadata_path": Path(metadata_path).as_posix(),
        "metadata_sha256": sha256_file(path),
        "observed_metrics": observed_metrics,
        "teacher_checkpoint_hash": checkpoint["sha256"],
    }


def validate_v6_protocol_config(payload: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "milestone",
        "parent_protocol",
        "active_representation",
        "split_hashes",
        "feature_hashes",
        "baseline_locks",
        "prediction_baseline",
        "teacher",
        "boundary_cohort",
        "primitive_candidates",
        "budgets",
    }
    if set(payload) != required or payload.get("schema_version") != 1:
        raise ValueError("Unsupported v6 protocol schema.")
    if payload["milestone"] != "M27":
        raise ValueError("The initial v6 protocol must identify milestone M27.")

    parent = payload["parent_protocol"]
    if not isinstance(parent, Mapping) or set(parent) != {"path", "sha256"}:
        raise ValueError("parent_protocol must contain path and sha256.")
    require_sha256(str(parent["sha256"]), "parent_protocol.sha256")

    active = payload["active_representation"]
    if not isinstance(active, Mapping) or set(active) != {
        "id",
        "hash",
        "interface",
        "output_dimension",
    }:
        raise ValueError("active_representation has an unsupported schema.")
    if active["id"] != "dinov2-small" or active["interface"] != "identity":
        raise ValueError("M27 locks DINOv2-small with the identity interface.")
    require_sha256(str(active["hash"]), "active_representation.hash")
    if int(active["output_dimension"]) != 384:
        raise ValueError("M27 expects native 384-dimensional DINOv2 features.")

    split_hashes = payload["split_hashes"]
    feature_hashes = payload["feature_hashes"]
    if set(split_hashes) != {"train", "development", "test"}:
        raise ValueError("split_hashes must contain train, development, and test.")
    if set(feature_hashes) != {"train", "development", "test"}:
        raise ValueError("feature_hashes must contain train, development, and test.")
    for field, values in (("split_hashes", split_hashes), ("feature_hashes", feature_hashes)):
        for name, digest in values.items():
            require_sha256(str(digest), f"{field}.{name}")

    if not isinstance(payload["prediction_baseline"], str) or not payload[
        "prediction_baseline"
    ]:
        raise ValueError("prediction_baseline must identify its metadata artifact.")

    teacher = payload["teacher"]
    if not isinstance(teacher, Mapping) or set(teacher) != {
        "primary_family",
        "secondary_family",
        "selection_metric",
        "test_labels_used_for_selection",
    }:
        raise ValueError("teacher has an unsupported schema.")
    if teacher["primary_family"] != "rbf_svm":
        raise ValueError("M27 requires RBF SVM as the primary teacher.")
    if bool(teacher["test_labels_used_for_selection"]):
        raise PermissionError("Teacher selection must not use final-test labels.")

    cohort = payload["boundary_cohort"]
    if not isinstance(cohort, Mapping) or set(cohort) != {
        "fraction",
        "minimum_count",
        "ranking",
        "tie_break",
    }:
        raise ValueError("boundary_cohort has an unsupported schema.")
    if cohort["ranking"] != "ascending_top2_probability_margin":
        raise ValueError("Unsupported boundary ranking.")
    if cohort["tie_break"] != "sample_index":
        raise ValueError("Boundary ties must use sample_index.")
    if not 0.0 < float(cohort["fraction"]) <= 1.0 or int(cohort["minimum_count"]) < 1:
        raise ValueError("Invalid boundary cohort size.")

    primitives = payload["primitive_candidates"]
    if not isinstance(primitives, list) or not primitives:
        raise ValueError("primitive_candidates must be a non-empty list.")
    primitive_ids = []
    for item in primitives:
        if not isinstance(item, Mapping) or "id" not in item:
            raise ValueError("Every primitive candidate requires an id.")
        primitive_ids.append(str(item["id"]))
        PrimitiveMetadata(
            family=str(item["family"]),
            minimum_seed_rule=str(item["minimum_seed_rule"]),
            score_semantics=str(item["score_semantics"]),
            local_rank=item.get("local_rank"),
            residual_scale=item.get("residual_scale"),
            direction=item.get("direction"),
            angular_radius=item.get("angular_radius"),
        ).validate()
    if len(primitive_ids) != len(set(primitive_ids)):
        raise ValueError("Primitive candidate identifiers must be unique.")

    budgets = payload["budgets"]
    if not isinstance(budgets, list) or not budgets:
        raise ValueError("budgets must be a non-empty list.")
    parsed_budgets = [
        BudgetSpec(
            mode=str(item["mode"]),
            component_limit=item.get("component_limit"),
            parameter_limit=item.get("parameter_limit"),
        )
        for item in budgets
    ]
    enumerate_budget_table(parsed_budgets, primitive_ids)
