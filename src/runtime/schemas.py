"""Versioned, dependency-free schemas for durable GEODE execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping


SCHEMA_VERSION = 1


class LifecycleState(str, Enum):
    CREATED = "CREATED"
    DATA_READY = "DATA_READY"
    REPRESENTATION_READY = "REPRESENTATION_READY"
    FEATURES_READY = "FEATURES_READY"
    GEOMETRY_READY = "GEOMETRY_READY"
    CALIBRATED = "CALIBRATED"
    GRAPH_VALIDATED = "GRAPH_VALIDATED"
    EVALUATED = "EVALUATED"
    PACKAGED = "PACKAGED"
    STAGED = "STAGED"
    PUBLISHED = "PUBLISHED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    INTERRUPTED = "INTERRUPTED"


class PretrainingLane(str, Enum):
    CONTROLLED = "controlled"
    EXTERNAL = "external"


class ReproducibilityLevel(str, Enum):
    REPLAY_IDENTITY = "replay_identity"
    SCIENTIFIC_EQUIVALENCE = "scientific_equivalence"


GEOMETRY_FAMILIES = {
    "sphere",
    "axis_aligned",
    "shrinkage",
    "low_rank_diagonal",
    "full",
}


def _validate_keys(
    payload: Mapping[str, Any],
    required: set[str],
    *,
    context: str,
) -> None:
    missing = required - payload.keys()
    unknown = payload.keys() - required
    if missing:
        raise ValueError(f"{context} missing required fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{context} has unknown fields: {sorted(unknown)}")


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_strings(value: Any, name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list of strings")
    values = tuple(_require_string(item, name) for item in value)
    if not allow_empty and not values:
        raise ValueError(f"{name} must not be empty")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def _require_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_positive_float(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be positive")
    return float(value)


def _require_nonnegative_float(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be non-negative")
    return float(value)


def _require_finite_float(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _require_sha256(value: Any, name: str) -> str:
    digest = _require_string(value, name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _require_probability(value: Any, name: str) -> float:
    probability = _require_finite_float(value, name)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return probability


def _require_hash_pairs(value: Any, name: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object mapping names to hashes")
    pairs = tuple(sorted(
        (_require_string(key, name), _require_string(item, name))
        for key, item in value.items()
    ))
    return pairs


def _validate_hash_pairs(value: Any, name: str) -> None:
    if not isinstance(value, tuple):
        raise ValueError(f"{name} must be a canonical tuple of hash pairs")
    names: list[str] = []
    for pair in value:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(f"{name} must contain (name, hash) pairs")
        names.append(_require_string(pair[0], name))
        _require_string(pair[1], name)
    if len(names) != len(set(names)):
        raise ValueError(f"{name} must not contain duplicate names")
    if value != tuple(sorted(value)):
        raise ValueError(f"{name} must use canonical name ordering")


@dataclass(frozen=True)
class DatasetEpisodeContract:
    geometry_split: str
    readout_calibration_split: str
    risk_control_split: str
    validation_split: str
    final_test_split: str

    def __post_init__(self) -> None:
        values = (
            self.geometry_split,
            self.readout_calibration_split,
            self.risk_control_split,
            self.validation_split,
            self.final_test_split,
        )
        for value in values:
            _require_string(value, "dataset episode split")
        if len(values) != len(set(values)):
            raise ValueError("dataset episode splits must be pairwise distinct")

    def to_dict(self) -> dict[str, Any]:
        return {
            "geometry_split": self.geometry_split,
            "readout_calibration_split": self.readout_calibration_split,
            "risk_control_split": self.risk_control_split,
            "validation_split": self.validation_split,
            "final_test_split": self.final_test_split,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DatasetEpisodeContract":
        required = {
            "geometry_split", "readout_calibration_split", "risk_control_split",
            "validation_split", "final_test_split",
        }
        _validate_keys(payload, required, context=cls.__name__)
        return cls(**{name: _require_string(payload[name], name) for name in required})


@dataclass(frozen=True)
class PretrainingProvenance:
    lane: PretrainingLane
    source_datasets: tuple[str, ...]
    objective: str
    checkpoint_hash: str
    license: str
    access_date: str
    overlap_risk: str

    def __post_init__(self) -> None:
        if not isinstance(self.lane, PretrainingLane):
            raise ValueError("lane must be a PretrainingLane")
        _require_strings(self.source_datasets, "source_datasets", allow_empty=True)
        for name in ("objective", "checkpoint_hash", "license", "access_date", "overlap_risk"):
            _require_string(getattr(self, name), name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane": self.lane.value,
            "source_datasets": list(self.source_datasets),
            "objective": self.objective,
            "checkpoint_hash": self.checkpoint_hash,
            "license": self.license,
            "access_date": self.access_date,
            "overlap_risk": self.overlap_risk,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PretrainingProvenance":
        required = {
            "lane", "source_datasets", "objective", "checkpoint_hash",
            "license", "access_date", "overlap_risk",
        }
        _validate_keys(payload, required, context=cls.__name__)
        try:
            lane = PretrainingLane(payload["lane"])
        except (TypeError, ValueError) as error:
            raise ValueError("lane must be 'controlled' or 'external'") from error
        return cls(
            lane=lane,
            source_datasets=_require_strings(
                payload["source_datasets"], "source_datasets", allow_empty=True,
            ),
            objective=_require_string(payload["objective"], "objective"),
            checkpoint_hash=_require_string(payload["checkpoint_hash"], "checkpoint_hash"),
            license=_require_string(payload["license"], "license"),
            access_date=_require_string(payload["access_date"], "access_date"),
            overlap_risk=_require_string(payload["overlap_risk"], "overlap_risk"),
        )


@dataclass(frozen=True)
class GeometryCapacityContract:
    allowed_families: tuple[str, ...]
    max_condition_number: float
    max_parameter_sample_ratio: float
    min_effective_rank: float

    def __post_init__(self) -> None:
        families = _require_strings(self.allowed_families, "allowed_families")
        unknown = set(families) - GEOMETRY_FAMILIES
        if unknown:
            raise ValueError(f"unsupported geometry families: {sorted(unknown)}")
        _require_positive_float(self.max_condition_number, "max_condition_number")
        _require_positive_float(
            self.max_parameter_sample_ratio, "max_parameter_sample_ratio",
        )
        _require_positive_float(self.min_effective_rank, "min_effective_rank")

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_families": list(self.allowed_families),
            "max_condition_number": self.max_condition_number,
            "max_parameter_sample_ratio": self.max_parameter_sample_ratio,
            "min_effective_rank": self.min_effective_rank,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GeometryCapacityContract":
        required = {
            "allowed_families", "max_condition_number",
            "max_parameter_sample_ratio", "min_effective_rank",
        }
        _validate_keys(payload, required, context=cls.__name__)
        return cls(
            allowed_families=_require_strings(payload["allowed_families"], "allowed_families"),
            max_condition_number=_require_positive_float(
                payload["max_condition_number"], "max_condition_number",
            ),
            max_parameter_sample_ratio=_require_positive_float(
                payload["max_parameter_sample_ratio"], "max_parameter_sample_ratio",
            ),
            min_effective_rank=_require_positive_float(
                payload["min_effective_rank"], "min_effective_rank",
            ),
        )


@dataclass(frozen=True)
class ModelSelectionContract:
    validation_domains: tuple[str, ...]
    final_domains: tuple[str, ...]
    selection_rule: str
    primary_metric: str

    def __post_init__(self) -> None:
        validation = _require_strings(self.validation_domains, "validation_domains")
        final = _require_strings(self.final_domains, "final_domains")
        overlap = set(validation) & set(final)
        if overlap:
            raise ValueError(f"validation and final domains overlap: {sorted(overlap)}")
        _require_string(self.selection_rule, "selection_rule")
        _require_string(self.primary_metric, "primary_metric")

    def to_dict(self) -> dict[str, Any]:
        return {
            "validation_domains": list(self.validation_domains),
            "final_domains": list(self.final_domains),
            "selection_rule": self.selection_rule,
            "primary_metric": self.primary_metric,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModelSelectionContract":
        required = {
            "validation_domains", "final_domains", "selection_rule", "primary_metric",
        }
        _validate_keys(payload, required, context=cls.__name__)
        return cls(
            validation_domains=_require_strings(
                payload["validation_domains"], "validation_domains",
            ),
            final_domains=_require_strings(payload["final_domains"], "final_domains"),
            selection_rule=_require_string(payload["selection_rule"], "selection_rule"),
            primary_metric=_require_string(payload["primary_metric"], "primary_metric"),
        )


@dataclass(frozen=True)
class ReproducibilityContract:
    level: ReproducibilityLevel
    environment_fingerprint: str
    absolute_tolerance: float
    relative_tolerance: float

    def __post_init__(self) -> None:
        if not isinstance(self.level, ReproducibilityLevel):
            raise ValueError("level must be a ReproducibilityLevel")
        _require_string(self.environment_fingerprint, "environment_fingerprint")
        absolute = _require_nonnegative_float(self.absolute_tolerance, "absolute_tolerance")
        relative = _require_nonnegative_float(self.relative_tolerance, "relative_tolerance")
        if self.level is ReproducibilityLevel.REPLAY_IDENTITY and (absolute or relative):
            raise ValueError("replay_identity requires zero numeric tolerances")

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "environment_fingerprint": self.environment_fingerprint,
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReproducibilityContract":
        required = {
            "level", "environment_fingerprint", "absolute_tolerance", "relative_tolerance",
        }
        _validate_keys(payload, required, context=cls.__name__)
        try:
            level = ReproducibilityLevel(payload["level"])
        except (TypeError, ValueError) as error:
            raise ValueError("unsupported reproducibility level") from error
        return cls(
            level=level,
            environment_fingerprint=_require_string(
                payload["environment_fingerprint"], "environment_fingerprint",
            ),
            absolute_tolerance=_require_nonnegative_float(
                payload["absolute_tolerance"], "absolute_tolerance",
            ),
            relative_tolerance=_require_nonnegative_float(
                payload["relative_tolerance"], "relative_tolerance",
            ),
        )


@dataclass(frozen=True)
class RunContract:
    run_id: str
    config_hash: str
    episode: DatasetEpisodeContract
    pretraining: PretrainingProvenance
    geometry: GeometryCapacityContract
    model_selection: ModelSelectionContract
    reproducibility: ReproducibilityContract
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        _require_string(self.run_id, "run_id")
        _require_string(self.config_hash, "config_hash")
        nested_contracts = (
            (self.episode, DatasetEpisodeContract, "episode"),
            (self.pretraining, PretrainingProvenance, "pretraining"),
            (self.geometry, GeometryCapacityContract, "geometry"),
            (self.model_selection, ModelSelectionContract, "model_selection"),
            (self.reproducibility, ReproducibilityContract, "reproducibility"),
        )
        for value, expected_type, name in nested_contracts:
            if not isinstance(value, expected_type):
                raise ValueError(f"{name} must be a {expected_type.__name__}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "episode": self.episode.to_dict(),
            "pretraining": self.pretraining.to_dict(),
            "geometry": self.geometry.to_dict(),
            "model_selection": self.model_selection.to_dict(),
            "reproducibility": self.reproducibility.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunContract":
        required = {
            "schema_version", "run_id", "config_hash", "episode", "pretraining",
            "geometry", "model_selection", "reproducibility",
        }
        _validate_keys(payload, required, context=cls.__name__)
        version = _require_nonnegative_int(payload["schema_version"], "schema_version")
        return cls(
            schema_version=version,
            run_id=_require_string(payload["run_id"], "run_id"),
            config_hash=_require_string(payload["config_hash"], "config_hash"),
            episode=DatasetEpisodeContract.from_dict(payload["episode"]),
            pretraining=PretrainingProvenance.from_dict(payload["pretraining"]),
            geometry=GeometryCapacityContract.from_dict(payload["geometry"]),
            model_selection=ModelSelectionContract.from_dict(payload["model_selection"]),
            reproducibility=ReproducibilityContract.from_dict(payload["reproducibility"]),
        )


@dataclass(frozen=True)
class StageManifest:
    run_id: str
    attempt_id: str
    stage_name: str
    state: LifecycleState
    created_at: str
    input_hashes: tuple[tuple[str, str], ...]
    output_hashes: tuple[tuple[str, str], ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        for name in ("run_id", "attempt_id", "stage_name", "created_at"):
            _require_string(getattr(self, name), name)
        if not isinstance(self.state, LifecycleState):
            raise ValueError("state must be a LifecycleState")
        _validate_hash_pairs(self.input_hashes, "input_hashes")
        _validate_hash_pairs(self.output_hashes, "output_hashes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "stage_name": self.stage_name,
            "state": self.state.value,
            "created_at": self.created_at,
            "input_hashes": dict(self.input_hashes),
            "output_hashes": dict(self.output_hashes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StageManifest":
        required = {
            "schema_version", "run_id", "attempt_id", "stage_name", "state",
            "created_at", "input_hashes", "output_hashes",
        }
        _validate_keys(payload, required, context=cls.__name__)
        try:
            state = LifecycleState(payload["state"])
        except (TypeError, ValueError) as error:
            raise ValueError("unsupported lifecycle state") from error
        return cls(
            schema_version=_require_nonnegative_int(payload["schema_version"], "schema_version"),
            run_id=_require_string(payload["run_id"], "run_id"),
            attempt_id=_require_string(payload["attempt_id"], "attempt_id"),
            stage_name=_require_string(payload["stage_name"], "stage_name"),
            state=state,
            created_at=_require_string(payload["created_at"], "created_at"),
            input_hashes=_require_hash_pairs(payload["input_hashes"], "input_hashes"),
            output_hashes=_require_hash_pairs(payload["output_hashes"], "output_hashes"),
        )


@dataclass(frozen=True)
class CheckpointMetadata:
    run_id: str
    attempt_id: str
    stage_name: str
    epoch: int
    global_step: int
    created_at: str
    artifact_hashes: tuple[tuple[str, str], ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        for name in ("run_id", "attempt_id", "stage_name", "created_at"):
            _require_string(getattr(self, name), name)
        _require_nonnegative_int(self.epoch, "epoch")
        _require_nonnegative_int(self.global_step, "global_step")
        _validate_hash_pairs(self.artifact_hashes, "artifact_hashes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "stage_name": self.stage_name,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "created_at": self.created_at,
            "artifact_hashes": dict(self.artifact_hashes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CheckpointMetadata":
        required = {
            "schema_version", "run_id", "attempt_id", "stage_name", "epoch",
            "global_step", "created_at", "artifact_hashes",
        }
        _validate_keys(payload, required, context=cls.__name__)
        version = _require_nonnegative_int(payload["schema_version"], "schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {version}")
        return cls(
            schema_version=version,
            run_id=_require_string(payload["run_id"], "run_id"),
            attempt_id=_require_string(payload["attempt_id"], "attempt_id"),
            stage_name=_require_string(payload["stage_name"], "stage_name"),
            epoch=_require_nonnegative_int(payload["epoch"], "epoch"),
            global_step=_require_nonnegative_int(payload["global_step"], "global_step"),
            created_at=_require_string(payload["created_at"], "created_at"),
            artifact_hashes=_require_hash_pairs(payload["artifact_hashes"], "artifact_hashes"),
        )


@dataclass(frozen=True)
class MetricEvent:
    event_id: str
    run_id: str
    attempt_id: str
    stage_name: str
    split: str
    metric_name: str
    value: float
    sample_count: int
    created_at: str
    epoch: int = 0
    global_step: int = 0
    namespace: str = "exploratory"
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        for name in (
            "event_id", "run_id", "attempt_id", "stage_name", "split",
            "metric_name", "created_at",
        ):
            _require_string(getattr(self, name), name)
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise ValueError("value must be numeric")
        if not math.isfinite(self.value):
            raise ValueError("value must be finite")
        _require_nonnegative_int(self.sample_count, "sample_count")
        _require_nonnegative_int(self.epoch, "epoch")
        _require_nonnegative_int(self.global_step, "global_step")
        if self.namespace not in {"exploratory", "selection", "final"}:
            raise ValueError("namespace must be exploratory, selection, or final")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "stage_name": self.stage_name,
            "split": self.split,
            "metric_name": self.metric_name,
            "value": self.value,
            "sample_count": self.sample_count,
            "created_at": self.created_at,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "namespace": self.namespace,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MetricEvent":
        required = {
            "schema_version", "event_id", "run_id", "attempt_id", "stage_name",
            "split", "metric_name", "value", "sample_count", "created_at",
            "epoch", "global_step", "namespace",
        }
        _validate_keys(payload, required, context=cls.__name__)
        version = _require_nonnegative_int(payload["schema_version"], "schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {version}")
        namespace = _require_string(payload["namespace"], "namespace")
        if namespace not in {"exploratory", "selection", "final"}:
            raise ValueError("namespace must be exploratory, selection, or final")
        return cls(
            schema_version=version,
            event_id=_require_string(payload["event_id"], "event_id"),
            run_id=_require_string(payload["run_id"], "run_id"),
            attempt_id=_require_string(payload["attempt_id"], "attempt_id"),
            stage_name=_require_string(payload["stage_name"], "stage_name"),
            split=_require_string(payload["split"], "split"),
            metric_name=_require_string(payload["metric_name"], "metric_name"),
            value=float(payload["value"]),
            sample_count=_require_nonnegative_int(payload["sample_count"], "sample_count"),
            created_at=_require_string(payload["created_at"], "created_at"),
            epoch=_require_nonnegative_int(payload["epoch"], "epoch"),
            global_step=_require_nonnegative_int(payload["global_step"], "global_step"),
            namespace=namespace,
        )


V8_EPISODE_PARTITIONS = (
    "stream",
    "anchor",
    "review_candidates",
    "adaptation_support",
    "validation",
    "final_test",
)

V8_INTERFACES = (
    "rejector_to_buffer",
    "buffer_to_clusterer",
    "clusterer_to_review",
    "review_to_adapter",
    "adapter_to_router",
)


@dataclass(frozen=True)
class AdaptationUtilityEndpoint:
    review_budget: int
    known_regression_ceiling: float
    unknown_recall_drop_ceiling: float
    minimum_mean_gain_over_baseline: float
    bootstrap_confidence: float
    minimum_positive_cells: int
    total_cells: int
    primary_metric: str = "post_integration_balanced_accuracy"
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        if self.review_budget <= 0:
            raise ValueError("review_budget must be positive")
        for name in (
            "known_regression_ceiling",
            "unknown_recall_drop_ceiling",
            "minimum_mean_gain_over_baseline",
        ):
            _require_nonnegative_float(getattr(self, name), name)
        confidence = _require_probability(self.bootstrap_confidence, "bootstrap_confidence")
        if confidence <= 0.0 or confidence >= 1.0:
            raise ValueError("bootstrap_confidence must be strictly between zero and one")
        if (
            self.minimum_positive_cells <= 0
            or self.total_cells <= 0
            or self.minimum_positive_cells > self.total_cells
        ):
            raise ValueError("positive-cell requirement is invalid")
        if self.primary_metric != "post_integration_balanced_accuracy":
            raise ValueError("adaptation utility is the only registered primary metric")

    def utility(self, parent_balanced_accuracy: float, child_balanced_accuracy: float) -> float:
        parent = _require_probability(parent_balanced_accuracy, "parent_balanced_accuracy")
        child = _require_probability(child_balanced_accuracy, "child_balanced_accuracy")
        return child - parent

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "review_budget": self.review_budget,
            "known_regression_ceiling": self.known_regression_ceiling,
            "unknown_recall_drop_ceiling": self.unknown_recall_drop_ceiling,
            "minimum_mean_gain_over_baseline": self.minimum_mean_gain_over_baseline,
            "bootstrap_confidence": self.bootstrap_confidence,
            "minimum_positive_cells": self.minimum_positive_cells,
            "total_cells": self.total_cells,
            "primary_metric": self.primary_metric,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AdaptationUtilityEndpoint":
        required = {
            "schema_version",
            "review_budget",
            "known_regression_ceiling",
            "unknown_recall_drop_ceiling",
            "minimum_mean_gain_over_baseline",
            "bootstrap_confidence",
            "minimum_positive_cells",
            "total_cells",
            "primary_metric",
        }
        _validate_keys(payload, required, context=cls.__name__)
        return cls(
            schema_version=_require_nonnegative_int(payload["schema_version"], "schema_version"),
            review_budget=_require_nonnegative_int(payload["review_budget"], "review_budget"),
            known_regression_ceiling=_require_nonnegative_float(
                payload["known_regression_ceiling"], "known_regression_ceiling"
            ),
            unknown_recall_drop_ceiling=_require_nonnegative_float(
                payload["unknown_recall_drop_ceiling"], "unknown_recall_drop_ceiling"
            ),
            minimum_mean_gain_over_baseline=_require_nonnegative_float(
                payload["minimum_mean_gain_over_baseline"],
                "minimum_mean_gain_over_baseline",
            ),
            bootstrap_confidence=_require_probability(
                payload["bootstrap_confidence"], "bootstrap_confidence"
            ),
            minimum_positive_cells=_require_nonnegative_int(
                payload["minimum_positive_cells"], "minimum_positive_cells"
            ),
            total_cells=_require_nonnegative_int(payload["total_cells"], "total_cells"),
            primary_metric=_require_string(payload["primary_metric"], "primary_metric"),
        )


@dataclass(frozen=True)
class EpisodeReplayContract:
    episode_id: str
    seed: int
    arrival_class: str
    parent_class_order: tuple[str, ...]
    child_class_order: tuple[str, ...]
    partition_hashes: tuple[tuple[str, str], ...]
    parent_bundle_hash: str
    acceptance_policy_hash: str
    anchor_set_hash: str
    review_budget: int
    final_test_sealed: bool
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        for name in ("episode_id", "arrival_class"):
            _require_string(getattr(self, name), name)
        _require_nonnegative_int(self.seed, "seed")
        parent = _require_strings(self.parent_class_order, "parent_class_order")
        child = _require_strings(self.child_class_order, "child_class_order")
        if self.arrival_class in parent:
            raise ValueError("arrival_class must be unknown to the parent")
        if child != (*parent, self.arrival_class):
            raise ValueError("child_class_order must append exactly the arrival class")
        _validate_hash_pairs(self.partition_hashes, "partition_hashes")
        if tuple(name for name, _ in self.partition_hashes) != tuple(
            sorted(V8_EPISODE_PARTITIONS)
        ):
            raise ValueError("episode partition hashes are incomplete")
        for name, digest in self.partition_hashes:
            _require_sha256(digest, f"partition_hashes[{name}]")
        if len({digest for _, digest in self.partition_hashes}) != len(
            self.partition_hashes
        ):
            raise ValueError("episode partitions must be pairwise disjoint")
        for name in ("parent_bundle_hash", "acceptance_policy_hash", "anchor_set_hash"):
            _require_sha256(getattr(self, name), name)
        if self.review_budget <= 0:
            raise ValueError("review_budget must be positive")
        if self.final_test_sealed is not True:
            raise PermissionError("episode final test must remain sealed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "seed": self.seed,
            "arrival_class": self.arrival_class,
            "parent_class_order": list(self.parent_class_order),
            "child_class_order": list(self.child_class_order),
            "partition_hashes": dict(self.partition_hashes),
            "parent_bundle_hash": self.parent_bundle_hash,
            "acceptance_policy_hash": self.acceptance_policy_hash,
            "anchor_set_hash": self.anchor_set_hash,
            "review_budget": self.review_budget,
            "final_test_sealed": self.final_test_sealed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EpisodeReplayContract":
        required = {
            "schema_version",
            "episode_id",
            "seed",
            "arrival_class",
            "parent_class_order",
            "child_class_order",
            "partition_hashes",
            "parent_bundle_hash",
            "acceptance_policy_hash",
            "anchor_set_hash",
            "review_budget",
            "final_test_sealed",
        }
        _validate_keys(payload, required, context=cls.__name__)
        return cls(
            schema_version=_require_nonnegative_int(payload["schema_version"], "schema_version"),
            episode_id=_require_string(payload["episode_id"], "episode_id"),
            seed=_require_nonnegative_int(payload["seed"], "seed"),
            arrival_class=_require_string(payload["arrival_class"], "arrival_class"),
            parent_class_order=_require_strings(
                payload["parent_class_order"], "parent_class_order"
            ),
            child_class_order=_require_strings(
                payload["child_class_order"], "child_class_order"
            ),
            partition_hashes=_require_hash_pairs(
                payload["partition_hashes"], "partition_hashes"
            ),
            parent_bundle_hash=_require_sha256(
                payload["parent_bundle_hash"], "parent_bundle_hash"
            ),
            acceptance_policy_hash=_require_sha256(
                payload["acceptance_policy_hash"], "acceptance_policy_hash"
            ),
            anchor_set_hash=_require_sha256(payload["anchor_set_hash"], "anchor_set_hash"),
            review_budget=_require_nonnegative_int(payload["review_budget"], "review_budget"),
            final_test_sealed=bool(payload["final_test_sealed"]),
        )


@dataclass(frozen=True)
class InterfaceContractAudit:
    interface_name: str
    producer_schema: str
    consumer_schema: str
    producer_artifact_hash: str
    required_statistics: tuple[str, ...]
    supplied_statistics: tuple[str, ...]
    approximated_statistics: tuple[str, ...] = ()
    unsupported_diagnostics: tuple[str, ...] = ()
    class_order_version: str = ""
    calibration_version: str = ""
    downstream_utility_impact: float | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        if self.interface_name not in V8_INTERFACES:
            raise ValueError("unsupported v8 interface")
        for name in (
            "producer_schema",
            "consumer_schema",
            "class_order_version",
            "calibration_version",
        ):
            _require_string(getattr(self, name), name)
        _require_sha256(self.producer_artifact_hash, "producer_artifact_hash")
        _require_strings(self.required_statistics, "required_statistics")
        supplied = _require_strings(
            self.supplied_statistics, "supplied_statistics", allow_empty=True
        )
        approximated = _require_strings(
            self.approximated_statistics, "approximated_statistics", allow_empty=True
        )
        _require_strings(
            self.unsupported_diagnostics, "unsupported_diagnostics", allow_empty=True
        )
        if set(approximated) - set(supplied):
            raise ValueError("approximated statistics must also be supplied")
        if self.downstream_utility_impact is not None:
            _require_finite_float(
                self.downstream_utility_impact, "downstream_utility_impact"
            )

    @property
    def complete(self) -> bool:
        missing = set(self.required_statistics) - set(self.supplied_statistics)
        unsupported_required = set(self.unsupported_diagnostics) & set(
            self.required_statistics
        )
        return not missing and not unsupported_required

    @property
    def missing_statistics(self) -> tuple[str, ...]:
        return tuple(
            sorted(set(self.required_statistics) - set(self.supplied_statistics))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "interface_name": self.interface_name,
            "producer_schema": self.producer_schema,
            "consumer_schema": self.consumer_schema,
            "producer_artifact_hash": self.producer_artifact_hash,
            "required_statistics": list(self.required_statistics),
            "supplied_statistics": list(self.supplied_statistics),
            "approximated_statistics": list(self.approximated_statistics),
            "unsupported_diagnostics": list(self.unsupported_diagnostics),
            "class_order_version": self.class_order_version,
            "calibration_version": self.calibration_version,
            "downstream_utility_impact": self.downstream_utility_impact,
        }


@dataclass(frozen=True)
class ThresholdTransferRecord:
    episode_id: str
    class_order_before: tuple[str, ...]
    class_order_after: tuple[str, ...]
    anchor_set_hash: str
    threshold_before: float
    threshold_after: float
    rule: str
    stale_action: str
    review_labels_consumed: int = 0
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        _require_string(self.episode_id, "episode_id")
        before = _require_strings(self.class_order_before, "class_order_before")
        after = _require_strings(self.class_order_after, "class_order_after")
        if len(after) != len(before) + 1 or after[:-1] != before:
            raise ValueError("threshold transfer requires one appended class")
        _require_sha256(self.anchor_set_hash, "anchor_set_hash")
        _require_finite_float(self.threshold_before, "threshold_before")
        _require_finite_float(self.threshold_after, "threshold_after")
        _require_string(self.rule, "rule")
        if self.stale_action not in {"frozen_parent", "exhaustive_fallback"}:
            raise ValueError("threshold transfer must fail closed when stale")
        if self.review_labels_consumed != 0:
            raise ValueError("threshold transfer cannot consume review labels")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "class_order_before": list(self.class_order_before),
            "class_order_after": list(self.class_order_after),
            "anchor_set_hash": self.anchor_set_hash,
            "threshold_before": self.threshold_before,
            "threshold_after": self.threshold_after,
            "rule": self.rule,
            "stale_action": self.stale_action,
            "review_labels_consumed": self.review_labels_consumed,
        }


@dataclass(frozen=True)
class ReviewSelectionEvidence:
    episode_id: str
    selector: str
    candidate_ids: tuple[str, ...]
    selected_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    review_budget: int
    selection_frozen_before_validation: bool
    expected_utility: float | None = None
    realized_utility: float | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        _require_string(self.episode_id, "episode_id")
        _require_string(self.selector, "selector")
        candidates = _require_strings(self.candidate_ids, "candidate_ids")
        selected = _require_strings(self.selected_ids, "selected_ids")
        validation = _require_strings(self.validation_ids, "validation_ids")
        if self.review_budget <= 0 or len(selected) > self.review_budget:
            raise ValueError("review selection exceeds the registered budget")
        if set(selected) - set(candidates):
            raise ValueError("selected review samples must come from candidates")
        if set(selected) & set(validation):
            raise ValueError("review support and validation examples overlap")
        if self.selection_frozen_before_validation is not True:
            raise PermissionError("selection must be frozen before validation scoring")
        for name in ("expected_utility", "realized_utility"):
            value = getattr(self, name)
            if value is not None:
                _require_finite_float(value, name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "selector": self.selector,
            "candidate_ids": list(self.candidate_ids),
            "selected_ids": list(self.selected_ids),
            "validation_ids": list(self.validation_ids),
            "review_budget": self.review_budget,
            "selection_frozen_before_validation": self.selection_frozen_before_validation,
            "expected_utility": self.expected_utility,
            "realized_utility": self.realized_utility,
        }


@dataclass(frozen=True)
class LocalizedResidualScope:
    episode_id: str
    parent_bundle_hash: str
    affected_sample_ids: tuple[str, ...]
    unaffected_sample_ids: tuple[str, ...]
    activated_component_ids: tuple[str, ...]
    responsibility_threshold: float
    minimum_unaffected_preservation: float = 0.999
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        _require_string(self.episode_id, "episode_id")
        _require_sha256(self.parent_bundle_hash, "parent_bundle_hash")
        affected = _require_strings(self.affected_sample_ids, "affected_sample_ids")
        unaffected = _require_strings(self.unaffected_sample_ids, "unaffected_sample_ids")
        _require_strings(self.activated_component_ids, "activated_component_ids")
        if set(affected) & set(unaffected):
            raise ValueError("affected and unaffected evaluation regions overlap")
        _require_probability(self.responsibility_threshold, "responsibility_threshold")
        minimum = _require_probability(
            self.minimum_unaffected_preservation,
            "minimum_unaffected_preservation",
        )
        if minimum < 0.999:
            raise ValueError("v8 locality requires at least 99.9% preservation")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "parent_bundle_hash": self.parent_bundle_hash,
            "affected_sample_ids": list(self.affected_sample_ids),
            "unaffected_sample_ids": list(self.unaffected_sample_ids),
            "activated_component_ids": list(self.activated_component_ids),
            "responsibility_threshold": self.responsibility_threshold,
            "minimum_unaffected_preservation": self.minimum_unaffected_preservation,
        }


@dataclass(frozen=True)
class IntegratedRoutingDecision:
    episode_id: str
    child_bundle_hash: str
    child_class_order: tuple[str, ...]
    threshold_lineage_hash: str
    routing_profile_hash: str
    exhaustive_winner_included: bool
    final_prediction_agreement: float
    unknown_fallback_rate: float
    evaluation_reduction: float
    utility_difference_from_exhaustive: float
    stale_profile_action: str
    authoritative: bool
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        _require_string(self.episode_id, "episode_id")
        _require_strings(self.child_class_order, "child_class_order")
        for name in (
            "child_bundle_hash",
            "threshold_lineage_hash",
            "routing_profile_hash",
        ):
            _require_sha256(getattr(self, name), name)
        agreement = _require_probability(
            self.final_prediction_agreement, "final_prediction_agreement"
        )
        fallback = _require_probability(self.unknown_fallback_rate, "unknown_fallback_rate")
        reduction = _require_probability(self.evaluation_reduction, "evaluation_reduction")
        utility_difference = _require_finite_float(
            self.utility_difference_from_exhaustive,
            "utility_difference_from_exhaustive",
        )
        if self.stale_profile_action != "exhaustive_fallback":
            raise ValueError("stale routing profiles must fall back exhaustively")
        if self.authoritative and (
            not self.exhaustive_winner_included
            or agreement != 1.0
            or fallback != 1.0
            or reduction < 0.25
            or abs(utility_difference) > 1e-12
        ):
            raise PermissionError("authoritative routing gates are not satisfied")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "child_bundle_hash": self.child_bundle_hash,
            "child_class_order": list(self.child_class_order),
            "threshold_lineage_hash": self.threshold_lineage_hash,
            "routing_profile_hash": self.routing_profile_hash,
            "exhaustive_winner_included": self.exhaustive_winner_included,
            "final_prediction_agreement": self.final_prediction_agreement,
            "unknown_fallback_rate": self.unknown_fallback_rate,
            "evaluation_reduction": self.evaluation_reduction,
            "utility_difference_from_exhaustive": self.utility_difference_from_exhaustive,
            "stale_profile_action": self.stale_profile_action,
            "authoritative": self.authoritative,
        }


V9_SURFACE_SCORE_VARIANTS = {"normalized", "metric_corrected"}
V9_SURFACE_STRATA = ("near_surface", "deep_interior", "exterior")


@dataclass(frozen=True)
class SurfaceSupportDiagnostic:
    component_hash: str
    representation_hash: str
    score_variant: str
    score_direction: str
    class_label: int
    seed: int
    partition_id: str
    normalized_signed_depth_quantiles: tuple[float, ...]
    metric_signed_depth_quantiles: tuple[float, ...]
    stratum_counts: tuple[tuple[str, int], ...]
    own_class_precision: tuple[tuple[str, float], ...]
    competing_class_occupancy: tuple[tuple[str, float], ...]
    unknown_occupancy: tuple[tuple[str, float], ...]
    width_selection_provenance: str
    selected_ids: tuple[str, ...]
    replay_hash: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        for name in ("component_hash", "representation_hash", "replay_hash"):
            _require_sha256(getattr(self, name), name)
        if self.score_variant not in V9_SURFACE_SCORE_VARIANTS:
            raise ValueError("unsupported v9 surface score variant")
        if self.score_direction != "lower_is_stronger_support":
            raise ValueError("surface scores must use the registered direction")
        _require_nonnegative_int(self.class_label, "class_label")
        _require_nonnegative_int(self.seed, "seed")
        _require_string(self.partition_id, "partition_id")
        _require_string(self.width_selection_provenance, "width_selection_provenance")
        _require_strings(self.selected_ids, "selected_ids", allow_empty=True)
        for name in (
            "normalized_signed_depth_quantiles",
            "metric_signed_depth_quantiles",
        ):
            values = tuple(
                _require_finite_float(value, name) for value in getattr(self, name)
            )
            if len(values) != 5 or tuple(sorted(values)) != values:
                raise ValueError(f"{name} must contain five ordered quantiles")
        self._validate_stratum_pairs(self.stratum_counts, "stratum_counts", integer=True)
        for name in (
            "own_class_precision",
            "competing_class_occupancy",
            "unknown_occupancy",
        ):
            self._validate_stratum_pairs(getattr(self, name), name, integer=False)

    @staticmethod
    def _validate_stratum_pairs(
        pairs: tuple[tuple[str, Any], ...],
        name: str,
        *,
        integer: bool,
    ) -> None:
        if tuple(key for key, _ in pairs) != V9_SURFACE_STRATA:
            raise ValueError(f"{name} must use the registered stratum order")
        for key, value in pairs:
            if integer:
                _require_nonnegative_int(value, f"{name}[{key}]")
            else:
                _require_probability(value, f"{name}[{key}]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "component_hash": self.component_hash,
            "representation_hash": self.representation_hash,
            "score_variant": self.score_variant,
            "score_direction": self.score_direction,
            "class_label": self.class_label,
            "seed": self.seed,
            "partition_id": self.partition_id,
            "normalized_signed_depth_quantiles": list(
                self.normalized_signed_depth_quantiles
            ),
            "metric_signed_depth_quantiles": list(
                self.metric_signed_depth_quantiles
            ),
            "stratum_counts": dict(self.stratum_counts),
            "own_class_precision": dict(self.own_class_precision),
            "competing_class_occupancy": dict(self.competing_class_occupancy),
            "unknown_occupancy": dict(self.unknown_occupancy),
            "width_selection_provenance": self.width_selection_provenance,
            "selected_ids": list(self.selected_ids),
            "replay_hash": self.replay_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SurfaceSupportDiagnostic":
        required = {
            "schema_version",
            "component_hash",
            "representation_hash",
            "score_variant",
            "score_direction",
            "class_label",
            "seed",
            "partition_id",
            "normalized_signed_depth_quantiles",
            "metric_signed_depth_quantiles",
            "stratum_counts",
            "own_class_precision",
            "competing_class_occupancy",
            "unknown_occupancy",
            "width_selection_provenance",
            "selected_ids",
            "replay_hash",
        }
        _validate_keys(payload, required, context=cls.__name__)

        def pairs(name: str, cast: Any) -> tuple[tuple[str, Any], ...]:
            value = payload[name]
            if not isinstance(value, Mapping):
                raise ValueError(f"{name} must be an object")
            if set(value) != set(V9_SURFACE_STRATA):
                raise ValueError(f"{name} must contain exactly the registered strata")
            return tuple((key, cast(value[key])) for key in V9_SURFACE_STRATA)

        return cls(
            schema_version=_require_nonnegative_int(payload["schema_version"], "schema_version"),
            component_hash=_require_sha256(payload["component_hash"], "component_hash"),
            representation_hash=_require_sha256(
                payload["representation_hash"], "representation_hash"
            ),
            score_variant=_require_string(payload["score_variant"], "score_variant"),
            score_direction=_require_string(payload["score_direction"], "score_direction"),
            class_label=_require_nonnegative_int(payload["class_label"], "class_label"),
            seed=_require_nonnegative_int(payload["seed"], "seed"),
            partition_id=_require_string(payload["partition_id"], "partition_id"),
            normalized_signed_depth_quantiles=tuple(
                float(value) for value in payload["normalized_signed_depth_quantiles"]
            ),
            metric_signed_depth_quantiles=tuple(
                float(value) for value in payload["metric_signed_depth_quantiles"]
            ),
            stratum_counts=pairs("stratum_counts", int),
            own_class_precision=pairs("own_class_precision", float),
            competing_class_occupancy=pairs("competing_class_occupancy", float),
            unknown_occupancy=pairs("unknown_occupancy", float),
            width_selection_provenance=_require_string(
                payload["width_selection_provenance"], "width_selection_provenance"
            ),
            selected_ids=_require_strings(
                payload["selected_ids"], "selected_ids", allow_empty=True
            ),
            replay_hash=_require_sha256(payload["replay_hash"], "replay_hash"),
        )


V10_PROBE_FAMILIES = (
    "axis_tangent",
    "corner_tangent",
    "normal",
    "mixed",
    "bridge",
    "cross_class_bridge",
    "random_direction",
)


@dataclass(frozen=True)
class TubeCalibrationRecord:
    geometry_hash: str
    representation_hash: str
    partition_hash: str
    rank: int
    patch_count: int
    extent_quantile: float
    outer_scale_policy: str
    penalty_grid: tuple[float, ...]
    selected_penalty: float
    known_coverage_target: float
    calibrated_threshold: float
    calibration_known_coverage: float
    selected_before_development: bool
    final_labels_opened: bool
    replay_hash: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        for name in (
            "geometry_hash",
            "representation_hash",
            "partition_hash",
            "replay_hash",
        ):
            _require_sha256(getattr(self, name), name)
        if self.rank not in {8, 16, 32}:
            raise ValueError("unsupported v10 tube rank")
        if self.patch_count not in {1, 2, 4}:
            raise ValueError("unsupported v10 patch count")
        if not 0.0 < self.extent_quantile < 1.0:
            raise ValueError("extent_quantile must be strictly between zero and one")
        if self.outer_scale_policy not in {"median_overshoot", "interquantile_range"}:
            raise ValueError("unsupported outer_scale_policy")
        penalties = tuple(
            _require_positive_float(value, "penalty_grid")
            for value in self.penalty_grid
        )
        if not penalties or tuple(sorted(set(penalties))) != penalties:
            raise ValueError("penalty_grid must be strictly increasing")
        selected = _require_positive_float(self.selected_penalty, "selected_penalty")
        if selected not in penalties:
            raise ValueError("selected_penalty must come from penalty_grid")
        target = _require_probability(
            self.known_coverage_target, "known_coverage_target"
        )
        if target != 0.92:
            raise ValueError("v10 known coverage target is frozen at 92%")
        _require_nonnegative_float(self.calibrated_threshold, "calibrated_threshold")
        coverage = _require_probability(
            self.calibration_known_coverage, "calibration_known_coverage"
        )
        if coverage < 0.90:
            raise PermissionError("calibration known coverage is below the safety floor")
        if self.selected_before_development is not True:
            raise PermissionError("tube calibration must precede development evaluation")
        if self.final_labels_opened:
            raise PermissionError("final labels must remain sealed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "geometry_hash": self.geometry_hash,
            "representation_hash": self.representation_hash,
            "partition_hash": self.partition_hash,
            "rank": self.rank,
            "patch_count": self.patch_count,
            "extent_quantile": self.extent_quantile,
            "outer_scale_policy": self.outer_scale_policy,
            "penalty_grid": list(self.penalty_grid),
            "selected_penalty": self.selected_penalty,
            "known_coverage_target": self.known_coverage_target,
            "calibrated_threshold": self.calibrated_threshold,
            "calibration_known_coverage": self.calibration_known_coverage,
            "selected_before_development": self.selected_before_development,
            "final_labels_opened": self.final_labels_opened,
            "replay_hash": self.replay_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TubeCalibrationRecord":
        required = {
            "schema_version",
            "geometry_hash",
            "representation_hash",
            "partition_hash",
            "rank",
            "patch_count",
            "extent_quantile",
            "outer_scale_policy",
            "penalty_grid",
            "selected_penalty",
            "known_coverage_target",
            "calibrated_threshold",
            "calibration_known_coverage",
            "selected_before_development",
            "final_labels_opened",
            "replay_hash",
        }
        _validate_keys(payload, required, context=cls.__name__)
        penalty_grid = payload["penalty_grid"]
        if not isinstance(penalty_grid, (list, tuple)):
            raise ValueError("penalty_grid must be a list")
        return cls(
            schema_version=_require_nonnegative_int(payload["schema_version"], "schema_version"),
            geometry_hash=_require_sha256(payload["geometry_hash"], "geometry_hash"),
            representation_hash=_require_sha256(
                payload["representation_hash"], "representation_hash"
            ),
            partition_hash=_require_sha256(payload["partition_hash"], "partition_hash"),
            rank=_require_nonnegative_int(payload["rank"], "rank"),
            patch_count=_require_nonnegative_int(payload["patch_count"], "patch_count"),
            extent_quantile=float(payload["extent_quantile"]),
            outer_scale_policy=_require_string(
                payload["outer_scale_policy"], "outer_scale_policy"
            ),
            penalty_grid=tuple(float(value) for value in penalty_grid),
            selected_penalty=float(payload["selected_penalty"]),
            known_coverage_target=float(payload["known_coverage_target"]),
            calibrated_threshold=float(payload["calibrated_threshold"]),
            calibration_known_coverage=float(payload["calibration_known_coverage"]),
            selected_before_development=bool(payload["selected_before_development"]),
            final_labels_opened=bool(payload["final_labels_opened"]),
            replay_hash=_require_sha256(payload["replay_hash"], "replay_hash"),
        )


@dataclass(frozen=True)
class TubeSafetyEvidence:
    calibration_replay_hash: str
    probe_generator_hash: str
    probe_counts: tuple[tuple[str, int], ...]
    source_patch_acceptance: tuple[tuple[str, float], ...]
    system_acceptance: tuple[tuple[str, float], ...]
    tangent_acceptance_by_multiplier: tuple[tuple[str, float], ...]
    parameter_count: int
    fit_work_units: int
    latency_seconds: float
    peak_temporary_bytes: int
    exact_replay: bool
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        _require_sha256(self.calibration_replay_hash, "calibration_replay_hash")
        _require_sha256(self.probe_generator_hash, "probe_generator_hash")
        for name, pairs, probability in (
            ("probe_counts", self.probe_counts, False),
            ("source_patch_acceptance", self.source_patch_acceptance, True),
            ("system_acceptance", self.system_acceptance, True),
        ):
            if tuple(key for key, _ in pairs) != V10_PROBE_FAMILIES:
                raise ValueError(f"{name} must use the registered probe-family order")
            for key, value in pairs:
                if probability:
                    _require_probability(value, f"{name}[{key}]")
                else:
                    _require_nonnegative_int(value, f"{name}[{key}]")
        expected_multipliers = ("0.5", "1", "2", "4", "8")
        if tuple(key for key, _ in self.tangent_acceptance_by_multiplier) != expected_multipliers:
            raise ValueError("tangent acceptance multipliers are incomplete")
        for key, value in self.tangent_acceptance_by_multiplier:
            _require_probability(value, f"tangent_acceptance_by_multiplier[{key}]")
        _require_nonnegative_int(self.parameter_count, "parameter_count")
        _require_nonnegative_int(self.fit_work_units, "fit_work_units")
        _require_nonnegative_float(self.latency_seconds, "latency_seconds")
        _require_nonnegative_int(self.peak_temporary_bytes, "peak_temporary_bytes")
        if self.exact_replay is not True:
            raise PermissionError("tube safety evidence must replay exactly")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "calibration_replay_hash": self.calibration_replay_hash,
            "probe_generator_hash": self.probe_generator_hash,
            "probe_counts": dict(self.probe_counts),
            "source_patch_acceptance": dict(self.source_patch_acceptance),
            "system_acceptance": dict(self.system_acceptance),
            "tangent_acceptance_by_multiplier": dict(
                self.tangent_acceptance_by_multiplier
            ),
            "parameter_count": self.parameter_count,
            "fit_work_units": self.fit_work_units,
            "latency_seconds": self.latency_seconds,
            "peak_temporary_bytes": self.peak_temporary_bytes,
            "exact_replay": self.exact_replay,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TubeSafetyEvidence":
        required = {
            "schema_version",
            "calibration_replay_hash",
            "probe_generator_hash",
            "probe_counts",
            "source_patch_acceptance",
            "system_acceptance",
            "tangent_acceptance_by_multiplier",
            "parameter_count",
            "fit_work_units",
            "latency_seconds",
            "peak_temporary_bytes",
            "exact_replay",
        }
        _validate_keys(payload, required, context=cls.__name__)

        def ordered_pairs(
            name: str, keys: tuple[str, ...], cast: Any
        ) -> tuple[tuple[str, Any], ...]:
            value = payload[name]
            if not isinstance(value, Mapping) or set(value) != set(keys):
                raise ValueError(f"{name} must contain exactly the registered keys")
            return tuple((key, cast(value[key])) for key in keys)

        return cls(
            schema_version=_require_nonnegative_int(payload["schema_version"], "schema_version"),
            calibration_replay_hash=_require_sha256(
                payload["calibration_replay_hash"], "calibration_replay_hash"
            ),
            probe_generator_hash=_require_sha256(
                payload["probe_generator_hash"], "probe_generator_hash"
            ),
            probe_counts=ordered_pairs("probe_counts", V10_PROBE_FAMILIES, int),
            source_patch_acceptance=ordered_pairs(
                "source_patch_acceptance", V10_PROBE_FAMILIES, float
            ),
            system_acceptance=ordered_pairs(
                "system_acceptance", V10_PROBE_FAMILIES, float
            ),
            tangent_acceptance_by_multiplier=ordered_pairs(
                "tangent_acceptance_by_multiplier",
                ("0.5", "1", "2", "4", "8"),
                float,
            ),
            parameter_count=_require_nonnegative_int(
                payload["parameter_count"], "parameter_count"
            ),
            fit_work_units=_require_nonnegative_int(
                payload["fit_work_units"], "fit_work_units"
            ),
            latency_seconds=_require_nonnegative_float(
                payload["latency_seconds"], "latency_seconds"
            ),
            peak_temporary_bytes=_require_nonnegative_int(
                payload["peak_temporary_bytes"], "peak_temporary_bytes"
            ),
            exact_replay=bool(payload["exact_replay"]),
        )


V11_EXTENT_POLICIES = {
    "quantile",
    "negative_guided",
    "negative_guided_iqr",
}
V11_PROBE_FAMILIES = V10_PROBE_FAMILIES + ("masking",)


@dataclass(frozen=True)
class DirectionalGeometryRecord:
    geometry_hash: str
    representation_hash: str
    partition_hash: str
    rank: int
    patch_count: int
    extent_policy: str
    extent_quantile: float
    parameter_count: int
    fit_work_units: int
    replay_hash: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        for name in (
            "geometry_hash",
            "representation_hash",
            "partition_hash",
            "replay_hash",
        ):
            _require_sha256(getattr(self, name), name)
        if self.rank not in {8, 16, 32}:
            raise ValueError("unsupported v11 directional rank")
        if self.patch_count not in {1, 2, 4}:
            raise ValueError("unsupported v11 patch count")
        if self.extent_policy not in V11_EXTENT_POLICIES:
            raise ValueError("unsupported v11 extent policy")
        if self.extent_quantile not in {0.95, 0.99}:
            raise ValueError("unsupported v11 extent quantile")
        _require_nonnegative_int(self.parameter_count, "parameter_count")
        _require_nonnegative_int(self.fit_work_units, "fit_work_units")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "geometry_hash": self.geometry_hash,
            "representation_hash": self.representation_hash,
            "partition_hash": self.partition_hash,
            "rank": self.rank,
            "patch_count": self.patch_count,
            "extent_policy": self.extent_policy,
            "extent_quantile": self.extent_quantile,
            "parameter_count": self.parameter_count,
            "fit_work_units": self.fit_work_units,
            "replay_hash": self.replay_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DirectionalGeometryRecord":
        required = {
            "schema_version",
            "geometry_hash",
            "representation_hash",
            "partition_hash",
            "rank",
            "patch_count",
            "extent_policy",
            "extent_quantile",
            "parameter_count",
            "fit_work_units",
            "replay_hash",
        }
        _validate_keys(payload, required, context=cls.__name__)
        return cls(
            schema_version=_require_nonnegative_int(payload["schema_version"], "schema_version"),
            geometry_hash=_require_sha256(payload["geometry_hash"], "geometry_hash"),
            representation_hash=_require_sha256(
                payload["representation_hash"], "representation_hash"
            ),
            partition_hash=_require_sha256(payload["partition_hash"], "partition_hash"),
            rank=_require_nonnegative_int(payload["rank"], "rank"),
            patch_count=_require_nonnegative_int(payload["patch_count"], "patch_count"),
            extent_policy=_require_string(payload["extent_policy"], "extent_policy"),
            extent_quantile=float(payload["extent_quantile"]),
            parameter_count=_require_nonnegative_int(
                payload["parameter_count"], "parameter_count"
            ),
            fit_work_units=_require_nonnegative_int(
                payload["fit_work_units"], "fit_work_units"
            ),
            replay_hash=_require_sha256(payload["replay_hash"], "replay_hash"),
        )


@dataclass(frozen=True)
class ConformalCalibrationRecord:
    geometry_replay_hash: str
    delegated_head_hash: str
    miscoverage: float
    class_counts: tuple[tuple[str, int], ...]
    class_thresholds: tuple[tuple[str, float], ...]
    selected_before_development: bool
    final_labels_opened: bool
    replay_hash: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        for name in ("geometry_replay_hash", "delegated_head_hash", "replay_hash"):
            _require_sha256(getattr(self, name), name)
        if self.miscoverage != 0.08:
            raise ValueError("v11 conformal miscoverage is frozen at 0.08")
        count_names = tuple(name for name, _ in self.class_counts)
        threshold_names = tuple(name for name, _ in self.class_thresholds)
        if (
            not count_names
            or count_names != threshold_names
            or count_names != tuple(sorted(set(count_names)))
        ):
            raise ValueError("class calibration entries must use canonical matching keys")
        for name, count in self.class_counts:
            if _require_nonnegative_int(count, f"class_counts[{name}]") < 1:
                raise ValueError("every conformal class requires observations")
        for name, threshold in self.class_thresholds:
            _require_positive_float(threshold, f"class_thresholds[{name}]")
        if self.selected_before_development is not True:
            raise PermissionError("conformal calibration must precede development evaluation")
        if self.final_labels_opened:
            raise PermissionError("final labels must remain sealed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "geometry_replay_hash": self.geometry_replay_hash,
            "delegated_head_hash": self.delegated_head_hash,
            "miscoverage": self.miscoverage,
            "class_counts": dict(self.class_counts),
            "class_thresholds": dict(self.class_thresholds),
            "selected_before_development": self.selected_before_development,
            "final_labels_opened": self.final_labels_opened,
            "replay_hash": self.replay_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConformalCalibrationRecord":
        required = {
            "schema_version",
            "geometry_replay_hash",
            "delegated_head_hash",
            "miscoverage",
            "class_counts",
            "class_thresholds",
            "selected_before_development",
            "final_labels_opened",
            "replay_hash",
        }
        _validate_keys(payload, required, context=cls.__name__)
        counts = payload["class_counts"]
        thresholds = payload["class_thresholds"]
        if not isinstance(counts, Mapping) or not isinstance(thresholds, Mapping):
            raise ValueError("class calibration entries must be objects")
        return cls(
            schema_version=_require_nonnegative_int(payload["schema_version"], "schema_version"),
            geometry_replay_hash=_require_sha256(
                payload["geometry_replay_hash"], "geometry_replay_hash"
            ),
            delegated_head_hash=_require_sha256(
                payload["delegated_head_hash"], "delegated_head_hash"
            ),
            miscoverage=float(payload["miscoverage"]),
            class_counts=tuple(sorted((str(key), int(value)) for key, value in counts.items())),
            class_thresholds=tuple(
                sorted((str(key), float(value)) for key, value in thresholds.items())
            ),
            selected_before_development=bool(payload["selected_before_development"]),
            final_labels_opened=bool(payload["final_labels_opened"]),
            replay_hash=_require_sha256(payload["replay_hash"], "replay_hash"),
        )


@dataclass(frozen=True)
class ContrastAcceptanceRecord:
    calibration_replay_hash: str
    margin_grid: tuple[float, ...]
    selected_margin: float
    probe_counts: tuple[tuple[str, int], ...]
    source_patch_acceptance: tuple[tuple[str, float], ...]
    system_acceptance: tuple[tuple[str, float], ...]
    endpoint_count: int
    latency_seconds: float
    peak_temporary_bytes: int
    exact_replay: bool
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        _require_sha256(self.calibration_replay_hash, "calibration_replay_hash")
        expected_grid = (0.0, 0.05, 0.1, 0.2)
        if self.margin_grid != expected_grid or self.selected_margin not in expected_grid:
            raise ValueError("contrast margins must use the frozen v11 grid")
        for name, pairs, probability in (
            ("probe_counts", self.probe_counts, False),
            ("source_patch_acceptance", self.source_patch_acceptance, True),
            ("system_acceptance", self.system_acceptance, True),
        ):
            if tuple(key for key, _ in pairs) != V11_PROBE_FAMILIES:
                raise ValueError(f"{name} must use the registered v11 probe order")
            for key, value in pairs:
                if probability:
                    _require_probability(value, f"{name}[{key}]")
                else:
                    _require_nonnegative_int(value, f"{name}[{key}]")
        _require_nonnegative_int(self.endpoint_count, "endpoint_count")
        _require_nonnegative_float(self.latency_seconds, "latency_seconds")
        _require_nonnegative_int(self.peak_temporary_bytes, "peak_temporary_bytes")
        if self.exact_replay is not True:
            raise PermissionError("contrast acceptance evidence must replay exactly")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "calibration_replay_hash": self.calibration_replay_hash,
            "margin_grid": list(self.margin_grid),
            "selected_margin": self.selected_margin,
            "probe_counts": dict(self.probe_counts),
            "source_patch_acceptance": dict(self.source_patch_acceptance),
            "system_acceptance": dict(self.system_acceptance),
            "endpoint_count": self.endpoint_count,
            "latency_seconds": self.latency_seconds,
            "peak_temporary_bytes": self.peak_temporary_bytes,
            "exact_replay": self.exact_replay,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContrastAcceptanceRecord":
        required = {
            "schema_version",
            "calibration_replay_hash",
            "margin_grid",
            "selected_margin",
            "probe_counts",
            "source_patch_acceptance",
            "system_acceptance",
            "endpoint_count",
            "latency_seconds",
            "peak_temporary_bytes",
            "exact_replay",
        }
        _validate_keys(payload, required, context=cls.__name__)

        def pairs(name: str, cast: Any) -> tuple[tuple[str, Any], ...]:
            value = payload[name]
            if not isinstance(value, Mapping) or set(value) != set(V11_PROBE_FAMILIES):
                raise ValueError(f"{name} must contain every v11 probe family")
            return tuple((key, cast(value[key])) for key in V11_PROBE_FAMILIES)

        return cls(
            schema_version=_require_nonnegative_int(payload["schema_version"], "schema_version"),
            calibration_replay_hash=_require_sha256(
                payload["calibration_replay_hash"], "calibration_replay_hash"
            ),
            margin_grid=tuple(float(value) for value in payload["margin_grid"]),
            selected_margin=float(payload["selected_margin"]),
            probe_counts=pairs("probe_counts", int),
            source_patch_acceptance=pairs("source_patch_acceptance", float),
            system_acceptance=pairs("system_acceptance", float),
            endpoint_count=_require_nonnegative_int(
                payload["endpoint_count"], "endpoint_count"
            ),
            latency_seconds=_require_nonnegative_float(
                payload["latency_seconds"], "latency_seconds"
            ),
            peak_temporary_bytes=_require_nonnegative_int(
                payload["peak_temporary_bytes"], "peak_temporary_bytes"
            ),
            exact_replay=bool(payload["exact_replay"]),
        )