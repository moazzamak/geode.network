"""Preflight checks for matching representation capacity to GEODE geometry."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from sklearn.covariance import LedoitWolf

from src.runtime.schemas import GeometryCapacityContract


FAMILY_ORDER = (
    "sphere",
    "axis_aligned",
    "shrinkage",
    "low_rank_diagonal",
    "full",
)


@dataclass(frozen=True)
class FamilyFeasibility:
    family: str
    parameter_count: int
    parameter_sample_ratio: float
    condition_number: float
    eligible: bool
    rejection_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "parameter_count": self.parameter_count,
            "parameter_sample_ratio": self.parameter_sample_ratio,
            "condition_number": self.condition_number,
            "eligible": self.eligible,
            "rejection_reasons": list(self.rejection_reasons),
        }


@dataclass(frozen=True)
class ClassGeometryFeasibility:
    class_id: int | str
    sample_count: int
    dimension: int
    effective_rank: float
    families: tuple[FamilyFeasibility, ...]

    @property
    def eligible_families(self) -> tuple[str, ...]:
        return tuple(item.family for item in self.families if item.eligible)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "sample_count": self.sample_count,
            "dimension": self.dimension,
            "effective_rank": self.effective_rank,
            "eligible_families": list(self.eligible_families),
            "families": [item.to_dict() for item in self.families],
        }


@dataclass(frozen=True)
class GeometryFeasibilityReport:
    classes: tuple[ClassGeometryFeasibility, ...]

    @property
    def supportable(self) -> bool:
        return bool(self.classes) and all(item.eligible_families for item in self.classes)

    @property
    def unsupported_classes(self) -> tuple[int | str, ...]:
        return tuple(item.class_id for item in self.classes if not item.eligible_families)

    def to_dict(self) -> dict[str, Any]:
        return {
            "supportable": self.supportable,
            "unsupported_classes": list(self.unsupported_classes),
            "classes": [item.to_dict() for item in self.classes],
        }

    def require_supportable(self) -> None:
        if not self.supportable:
            raise ValueError(
                "no authorized geometry family is supportable for classes "
                f"{list(self.unsupported_classes)}"
            )


def _effective_rank(eigenvalues: np.ndarray) -> float:
    nonnegative = np.maximum(np.asarray(eigenvalues, dtype=np.float64), 0.0)
    total = float(np.sum(nonnegative))
    squared_total = float(np.sum(nonnegative ** 2))
    if total <= 0.0 or squared_total <= 0.0:
        return 0.0
    return total ** 2 / squared_total


def _condition_number(matrix: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvalsh(np.asarray(matrix, dtype=np.float64))
    largest = float(eigenvalues[-1])
    smallest = float(eigenvalues[0])
    if smallest <= np.finfo(np.float64).eps or not math.isfinite(largest):
        return math.inf
    return largest / smallest


def _family_parameter_counts(dimension: int, low_rank: int) -> dict[str, int]:
    return {
        "sphere": dimension + 1,
        "axis_aligned": 2 * dimension,
        "shrinkage": dimension * (dimension + 3) // 2,
        "low_rank_diagonal": dimension * (low_rank + 2),
        "full": dimension * (dimension + 3) // 2,
    }


def _family_condition_numbers(
    points: np.ndarray,
    covariance: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    low_rank: int,
) -> dict[str, float]:
    variances = np.diag(covariance)
    axis_condition = (
        math.inf
        if np.min(variances) <= np.finfo(np.float64).eps
        else float(np.max(variances) / np.min(variances))
    )
    shrinkage_condition = _condition_number(LedoitWolf().fit(points).covariance_)

    order = np.argsort(eigenvalues)[::-1]
    top_indices = order[:low_rank]
    top_vectors = eigenvectors[:, top_indices]
    top_values = eigenvalues[top_indices]
    residual = covariance - (top_vectors * top_values) @ top_vectors.T
    residual_diagonal = np.maximum(
        np.diag(residual), np.finfo(np.float64).eps,
    )
    regularized = np.diag(residual_diagonal) + (top_vectors * top_values) @ top_vectors.T

    return {
        "sphere": 1.0,
        "axis_aligned": axis_condition,
        "shrinkage": shrinkage_condition,
        "low_rank_diagonal": _condition_number(regularized),
        "full": _condition_number(covariance),
    }


def evaluate_geometry_feasibility(
    features: np.ndarray,
    labels: np.ndarray,
    contract: GeometryCapacityContract,
    *,
    low_rank: int = 8,
) -> GeometryFeasibilityReport:
    """Evaluate per-class geometry families without fitting a GEODE model."""
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError("features must be (n, d) and align with labels")
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("features must be non-empty")
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if not np.all(np.isfinite(features)):
        raise ValueError("features must contain only finite values")
    if isinstance(low_rank, bool) or not isinstance(low_rank, int) or low_rank <= 0:
        raise ValueError("low_rank must be a positive integer")

    class_reports: list[ClassGeometryFeasibility] = []
    dimension = features.shape[1]
    selected_rank = min(low_rank, max(1, dimension - 1))
    parameter_counts = _family_parameter_counts(dimension, selected_rank)

    for class_id in np.unique(labels):
        points = features[labels == class_id]
        sample_count = len(points)
        if sample_count < 2:
            covariance = np.zeros((dimension, dimension), dtype=np.float64)
            eigenvalues = np.zeros(dimension, dtype=np.float64)
            eigenvectors = np.eye(dimension, dtype=np.float64)
            condition_numbers = {family: math.inf for family in FAMILY_ORDER}
        else:
            covariance = np.atleast_2d(np.cov(points, rowvar=False, ddof=1))
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            condition_numbers = _family_condition_numbers(
                points, covariance, eigenvalues, eigenvectors, selected_rank,
            )
        effective_rank = _effective_rank(eigenvalues)

        family_reports: list[FamilyFeasibility] = []
        for family in FAMILY_ORDER:
            parameter_count = parameter_counts[family]
            ratio = parameter_count / sample_count
            condition_number = condition_numbers[family]
            reasons: list[str] = []
            if family not in contract.allowed_families:
                reasons.append("family_not_authorized")
            if sample_count < 2:
                reasons.append("insufficient_samples")
            if effective_rank < contract.min_effective_rank:
                reasons.append("effective_rank_below_minimum")
            if ratio > contract.max_parameter_sample_ratio:
                reasons.append("parameter_sample_ratio_exceeded")
            if condition_number > contract.max_condition_number:
                reasons.append("condition_number_exceeded")
            family_reports.append(FamilyFeasibility(
                family=family,
                parameter_count=parameter_count,
                parameter_sample_ratio=ratio,
                condition_number=condition_number,
                eligible=not reasons,
                rejection_reasons=tuple(reasons),
            ))

        normalized_class_id = class_id.item() if isinstance(class_id, np.generic) else class_id
        class_reports.append(ClassGeometryFeasibility(
            class_id=normalized_class_id,
            sample_count=sample_count,
            dimension=dimension,
            effective_rank=effective_rank,
            families=tuple(family_reports),
        ))

    return GeometryFeasibilityReport(classes=tuple(class_reports))