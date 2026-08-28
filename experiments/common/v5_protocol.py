from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from experiments.common.experiment_manifest import canonical_json


class DataStage(str, Enum):
    S0 = "S0"
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    S4 = "S4"
    S5 = "S5"


class SplitRole(str, Enum):
    TRAIN = "train"
    DEVELOPMENT = "development"
    CALIBRATION = "calibration"
    TEST = "test"


class LabelUse(str, Enum):
    FIT = "fit"
    SELECT = "select"
    CALIBRATE = "calibrate"
    OBSERVE = "observe"


STAGE_SEEDS: dict[DataStage, tuple[int, ...]] = {
    DataStage.S0: (11,),
    DataStage.S1: (11,),
    DataStage.S2: (11, 23, 37),
    DataStage.S3: (11, 23, 37, 53, 71),
    DataStage.S4: (11, 23, 37),
    DataStage.S5: (11, 23, 37),
}

_ALLOWED_LABEL_USES: dict[SplitRole, frozenset[LabelUse]] = {
    SplitRole.TRAIN: frozenset({LabelUse.FIT}),
    SplitRole.DEVELOPMENT: frozenset({LabelUse.SELECT}),
    SplitRole.CALIBRATION: frozenset({LabelUse.CALIBRATE}),
    SplitRole.TEST: frozenset({LabelUse.OBSERVE}),
}


def seeds_for_stage(stage: DataStage, declared: tuple[int, ...] | None = None) -> tuple[int, ...]:
    expected = STAGE_SEEDS[stage]
    if declared is not None and tuple(declared) != expected:
        raise ValueError(
            f"{stage.value} seeds must be {expected}, got {tuple(declared)}."
        )
    return expected


def require_label_use(role: SplitRole, use: LabelUse) -> None:
    if use not in _ALLOWED_LABEL_USES[role]:
        raise PermissionError(
            f"{role.value} labels cannot be used for {use.value}; "
            f"allowed uses are {sorted(value.value for value in _ALLOWED_LABEL_USES[role])}."
        )


def require_sha256(value: str, field: str) -> str:
    raw = str(value)
    normalized = raw.lower()
    if raw != normalized:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest.")
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest.")
    return normalized


@dataclass(frozen=True)
class DatasetManifest:
    dataset: str
    stage: DataStage
    split_protocol: str
    feature_protocol: str
    split_hash: str | None = None
    feature_hash: str | None = None

    def validate(self, *, require_materialized: bool = False) -> None:
        if not self.dataset or not self.split_protocol or not self.feature_protocol:
            raise ValueError("Dataset manifests require dataset and protocol identifiers.")
        if require_materialized and (self.split_hash is None or self.feature_hash is None):
            raise ValueError("Materialized datasets require split_hash and feature_hash.")
        if self.split_hash is not None:
            require_sha256(self.split_hash, "split_hash")
        if self.feature_hash is not None:
            require_sha256(self.feature_hash, "feature_hash")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "dataset": self.dataset,
            "stage": self.stage.value,
            "split_protocol": self.split_protocol,
            "feature_protocol": self.feature_protocol,
            "split_hash": self.split_hash,
            "feature_hash": self.feature_hash,
        }


@dataclass(frozen=True)
class RepresentationLineage:
    backbone_id: str
    weights_hash: str
    preprocessing_hash: str
    output_dimension: int
    interface_id: str | None = None
    interface_hash: str | None = None
    parent_hash: str | None = None

    def validate(self) -> None:
        if not self.backbone_id:
            raise ValueError("backbone_id is required.")
        require_sha256(self.weights_hash, "weights_hash")
        require_sha256(self.preprocessing_hash, "preprocessing_hash")
        if self.output_dimension < 1:
            raise ValueError("output_dimension must be positive.")
        if (self.interface_id is None) != (self.interface_hash is None):
            raise ValueError("interface_id and interface_hash must be specified together.")
        if self.interface_hash is not None:
            require_sha256(self.interface_hash, "interface_hash")
        if self.parent_hash is not None:
            require_sha256(self.parent_hash, "parent_hash")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "backbone_id": self.backbone_id,
            "weights_hash": self.weights_hash,
            "preprocessing_hash": self.preprocessing_hash,
            "output_dimension": self.output_dimension,
            "interface_id": self.interface_id,
            "interface_hash": self.interface_hash,
            "parent_hash": self.parent_hash,
        }

    @property
    def digest(self) -> str:
        import hashlib

        return hashlib.sha256(canonical_json(self.to_dict()).encode("utf-8")).hexdigest()


_OPERATORS = {
    "eq": lambda value, threshold: value == threshold,
    "ge": lambda value, threshold: value >= threshold,
    "gt": lambda value, threshold: value > threshold,
    "le": lambda value, threshold: value <= threshold,
    "lt": lambda value, threshold: value < threshold,
}


@dataclass(frozen=True)
class GateOperand:
    name: str
    value: float | int | bool | str
    operator: str
    threshold: float | int | bool | str

    @property
    def passed(self) -> bool:
        try:
            comparator = _OPERATORS[self.operator]
        except KeyError as error:
            raise ValueError(f"Unsupported gate operator {self.operator!r}.") from error
        try:
            return bool(comparator(self.value, self.threshold))
        except TypeError as error:
            raise ValueError(f"Gate operand {self.name!r} has incompatible values.") from error

    def to_dict(self) -> dict[str, Any]:
        if not self.name:
            raise ValueError("Gate operand name is required.")
        return {
            "name": self.name,
            "value": self.value,
            "operator": self.operator,
            "threshold": self.threshold,
            "passed": self.passed,
        }


def validate_protocol_config(payload: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "milestone",
        "stages",
        "required_heads",
        "representations",
        "heads",
        "readouts",
        "datasets",
    }
    if set(payload) != required or payload.get("schema_version") != 1:
        raise ValueError("Unsupported v5 protocol schema.")
    if payload["milestone"] != "M16":
        raise ValueError("The initial v5 protocol must identify milestone M16.")
    stages = payload["stages"]
    if not isinstance(stages, Mapping):
        raise ValueError("stages must be a mapping.")
    expected_stages = {stage.value for stage in DataStage}
    if set(stages) != expected_stages:
        raise ValueError(f"stages must contain exactly {sorted(expected_stages)}.")
    for stage_name, stage_payload in stages.items():
        stage = DataStage(stage_name)
        declared = tuple(int(seed) for seed in stage_payload["seeds"])
        seeds_for_stage(stage, declared)
    for field in ("required_heads", "representations", "heads", "readouts"):
        values = payload[field]
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f"{field} must be a non-empty list of unique identifiers.")
    if not set(payload["required_heads"]).issubset(payload["heads"]):
        raise ValueError("All required_heads must appear in heads.")
    datasets = payload["datasets"]
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("datasets must be a non-empty list.")
    dataset_stages = [item.get("stage") for item in datasets]
    if len(dataset_stages) != len(set(dataset_stages)):
        raise ValueError("datasets must contain exactly one manifest per stage.")
    if set(dataset_stages) != expected_stages:
        raise ValueError(
            f"datasets must cover exactly the stages {sorted(expected_stages)}."
        )
    for item in datasets:
        manifest = DatasetManifest(
            dataset=str(item["dataset"]),
            stage=DataStage(item["stage"]),
            split_protocol=str(item["split_protocol"]),
            feature_protocol=str(item["feature_protocol"]),
            split_hash=item.get("split_hash"),
            feature_hash=item.get("feature_hash"),
        )
        manifest.validate()
