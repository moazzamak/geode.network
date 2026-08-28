"""Probabilistic interpretation of covariance-fitted GEODE primitives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from src.sdf_engine import EllipsoidExpert, Expert


def _negative_log_mean_exp(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("mixture values must have shape (components, samples)")
    logits = -values
    maximum = np.max(logits, axis=0)
    return -(maximum + np.log(np.mean(np.exp(logits - maximum), axis=0)))


def gaussian_primitive_nll(
    primitive: EllipsoidExpert,
    points: np.ndarray,
    covariance_temperature: float = 1.0,
) -> np.ndarray:
    """Evaluate the Gaussian NLL implied by covariance-scaled primitive radii.

    Covariance fitters construct ``r_i = sqrt(d * lambda_i)``. Therefore the
    implied covariance eigenvalues are ``lambda_i = r_i**2 / d``.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != len(primitive.center):
        raise ValueError("points must have shape (samples, primitive dimension)")
    if primitive.polarity < 0:
        raise ValueError("subtractive primitives do not define Gaussian densities")
    if np.any(~np.isfinite(primitive.radii)) or np.any(primitive.radii <= 0.0):
        raise ValueError("primitive radii must be finite and positive")
    if not np.isfinite(covariance_temperature) or covariance_temperature <= 0.0:
        raise ValueError("covariance_temperature must be finite and positive")
    dimension = points.shape[1]
    local = (points - primitive.center) @ primitive.orientation
    mahalanobis = dimension * np.sum(np.square(local / primitive.radii), axis=1)
    log_determinant = np.sum(np.log(np.square(primitive.radii) / dimension))
    return 0.5 * (
        mahalanobis / covariance_temperature
        + log_determinant
        + dimension * np.log(covariance_temperature)
        + dimension * np.log(2.0 * np.pi)
    )


class ProbabilisticInferenceEngine:
    """Hierarchical uniform Gaussian-mixture scoring for GEODE class models."""

    def __init__(self, class_models: Mapping[int, Sequence[Expert]]) -> None:
        self.class_ids = np.asarray(sorted(class_models), dtype=np.int32)
        self.class_models = {
            int(class_id): list(class_models[int(class_id)])
            for class_id in self.class_ids
        }
        for class_id, experts in self.class_models.items():
            if not experts:
                raise ValueError(f"class {class_id} has no probability model")
            for expert in experts:
                if not any(primitive.polarity > 0 for primitive in expert.ellipsoids):
                    raise ValueError(
                        f"class {class_id} contains an expert with no probability model"
                    )
                if any(primitive.polarity < 0 for primitive in expert.ellipsoids):
                    raise ValueError(
                        "probabilistic inference does not support subtractive primitives"
                    )

    @staticmethod
    def _expert_nll(
        expert: Expert,
        points: np.ndarray,
        covariance_temperature: float,
    ) -> np.ndarray:
        primitives = [
            primitive for primitive in expert.ellipsoids
            if primitive.polarity > 0
        ]
        if not primitives:
            return np.full(len(points), np.inf)
        primitive_nlls = np.asarray([
            gaussian_primitive_nll(
                primitive, points, covariance_temperature=covariance_temperature,
            )
            for primitive in primitives
        ])
        return _negative_log_mean_exp(primitive_nlls)

    def class_nlls(
        self,
        points: np.ndarray,
        covariance_temperature: float | Sequence[float] = 1.0,
    ) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2:
            raise ValueError("points must have shape (samples, dimensions)")
        temperatures = np.asarray(covariance_temperature, dtype=np.float64)
        if temperatures.ndim == 0:
            temperatures = np.full(len(self.class_ids), temperatures.item())
        if temperatures.shape != (len(self.class_ids),):
            raise ValueError("covariance_temperature must be scalar or class-width")
        if np.any(~np.isfinite(temperatures)) or np.any(temperatures <= 0.0):
            raise ValueError("covariance_temperature must be finite and positive")
        columns = []
        for class_column, class_id in enumerate(self.class_ids):
            experts = self.class_models[int(class_id)]
            expert_nlls = np.asarray([
                self._expert_nll(expert, points, temperatures[class_column])
                for expert in experts
            ])
            columns.append(_negative_log_mean_exp(expert_nlls))
        return np.column_stack(columns)