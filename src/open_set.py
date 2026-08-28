from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


UNKNOWN_LABEL = "__unknown__"


class OpenSetReason(StrEnum):
    ACCEPTED = "accepted"
    LOW_CONFIDENCE = "low_confidence"
    OUTSIDE_SUPPORT = "outside_support"
    NO_COMPATIBLE_CANDIDATE = "no_compatible_candidate"
    STALE_SUPPORT_PROFILE = "stale_support_profile"


@dataclass(frozen=True)
class SupportProfile:
    """Versioned empirical support metadata for one fitted model.

    This profile is deliberately separate from ``ModelFingerprint``: the
    fingerprint identifies a model's role, while this object records the
    learned support and frozen novelty policy for one fitted version.
    """

    model_signature: str
    feature_transform_fingerprint: str
    training_dataset_fingerprint: str
    calibration_dataset_fingerprint: str
    class_ids: tuple[Any, ...]
    score_scales: tuple[float, ...]
    novelty_score: str
    global_threshold: float
    version: str
    fit_seed: int
    created_at: str
    class_thresholds: tuple[float, ...] = ()
    routing_centroids: tuple[tuple[float, ...], ...] = ()
    routing_radii: tuple[float, ...] = ()
    class_order_version: str = ""
    threshold_lineage_hash: str = ""
    anchor_set_hash: str = ""

    def __post_init__(self) -> None:
        required = {
            "model_signature": self.model_signature,
            "feature_transform_fingerprint": self.feature_transform_fingerprint,
            "training_dataset_fingerprint": self.training_dataset_fingerprint,
            "calibration_dataset_fingerprint": self.calibration_dataset_fingerprint,
            "novelty_score": self.novelty_score,
            "version": self.version,
            "created_at": self.created_at,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"Support profile fields must be non-empty: {missing}")
        if not self.class_ids or len(set(self.class_ids)) != len(self.class_ids):
            raise ValueError("class_ids must be non-empty and unique.")
        if len(self.score_scales) != len(self.class_ids):
            raise ValueError("score_scales must align with class_ids.")
        if any(not math.isfinite(scale) or scale <= 0.0 for scale in self.score_scales):
            raise ValueError("score_scales must be finite and positive.")
        if not math.isfinite(self.global_threshold):
            raise ValueError("global_threshold must be finite.")
        if self.class_thresholds and len(self.class_thresholds) != len(self.class_ids):
            raise ValueError("class_thresholds must align with class_ids.")
        if any(not math.isfinite(value) for value in self.class_thresholds):
            raise ValueError("class_thresholds must be finite.")
        if bool(self.routing_centroids) != bool(self.routing_radii):
            raise ValueError("routing_centroids and routing_radii must be provided together.")
        if self.routing_centroids:
            if len(self.routing_centroids) != len(self.class_ids):
                raise ValueError("routing_centroids must align with class_ids.")
            if len(self.routing_radii) != len(self.class_ids):
                raise ValueError("routing_radii must align with class_ids.")
            dimensions = {len(center) for center in self.routing_centroids}
            if len(dimensions) != 1 or not next(iter(dimensions)):
                raise ValueError("routing_centroids must share a positive dimension.")
            if any(
                not math.isfinite(value)
                for center in self.routing_centroids
                for value in center
            ):
                raise ValueError("routing_centroids must be finite.")
            if any(
                not math.isfinite(radius) or radius < 0.0
                for radius in self.routing_radii
            ):
                raise ValueError("routing_radii must be finite and non-negative.")
        lineage = (
            self.class_order_version,
            self.threshold_lineage_hash,
            self.anchor_set_hash,
        )
        if any(lineage) and not all(lineage):
            raise ValueError("support-profile lineage must be supplied as a complete set.")
        for name in ("threshold_lineage_hash", "anchor_set_hash"):
            value = getattr(self, name)
            if value and (
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest.")

    @property
    def profile_id(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def threshold_for(self, class_id: Any) -> float:
        if not self.class_thresholds:
            return self.global_threshold
        try:
            index = self.class_ids.index(class_id)
        except ValueError as error:
            raise KeyError(f"Class {class_id!r} is absent from the support profile.") from error
        return self.class_thresholds[index]

    def assert_compatible(
        self,
        *,
        model_signature: str,
        class_ids: tuple[Any, ...],
        feature_transform_fingerprint: str,
    ) -> None:
        mismatches = []
        if model_signature != self.model_signature:
            mismatches.append("model_signature")
        if tuple(class_ids) != self.class_ids:
            mismatches.append("class_ids")
        if feature_transform_fingerprint != self.feature_transform_fingerprint:
            mismatches.append("feature_transform_fingerprint")
        if mismatches:
            raise ValueError(f"Stale or incompatible support profile: {mismatches}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_signature": self.model_signature,
            "feature_transform_fingerprint": self.feature_transform_fingerprint,
            "training_dataset_fingerprint": self.training_dataset_fingerprint,
            "calibration_dataset_fingerprint": self.calibration_dataset_fingerprint,
            "class_ids": list(self.class_ids),
            "score_scales": list(self.score_scales),
            "novelty_score": self.novelty_score,
            "global_threshold": self.global_threshold,
            "version": self.version,
            "fit_seed": self.fit_seed,
            "created_at": self.created_at,
            "class_thresholds": list(self.class_thresholds),
            "routing_centroids": [list(center) for center in self.routing_centroids],
            "routing_radii": list(self.routing_radii),
            "class_order_version": self.class_order_version,
            "threshold_lineage_hash": self.threshold_lineage_hash,
            "anchor_set_hash": self.anchor_set_hash,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SupportProfile":
        return cls(
            model_signature=str(payload["model_signature"]),
            feature_transform_fingerprint=str(payload["feature_transform_fingerprint"]),
            training_dataset_fingerprint=str(payload["training_dataset_fingerprint"]),
            calibration_dataset_fingerprint=str(payload["calibration_dataset_fingerprint"]),
            class_ids=tuple(payload["class_ids"]),
            score_scales=tuple(float(value) for value in payload["score_scales"]),
            novelty_score=str(payload["novelty_score"]),
            global_threshold=float(payload["global_threshold"]),
            version=str(payload["version"]),
            fit_seed=int(payload["fit_seed"]),
            created_at=str(payload["created_at"]),
            class_thresholds=tuple(
                float(value) for value in payload.get("class_thresholds", [])
            ),
            routing_centroids=tuple(
                tuple(float(value) for value in center)
                for center in payload.get("routing_centroids", [])
            ),
            routing_radii=tuple(
                float(value) for value in payload.get("routing_radii", [])
            ),
            class_order_version=str(payload.get("class_order_version", "")),
            threshold_lineage_hash=str(payload.get("threshold_lineage_hash", "")),
            anchor_set_hash=str(payload.get("anchor_set_hash", "")),
        )


@dataclass(frozen=True)
class OpenSetPrediction:
    """Auditable open-set decision for one sample."""

    label: Any
    accepted: bool
    candidate_model_signature: str
    candidate_class_id: Any | None
    raw_novelty_score: float
    calibrated_novelty_score: float
    threshold: float
    decision_margin: float
    support_profile_version: str
    reason_code: OpenSetReason

    def __post_init__(self) -> None:
        numeric = (
            self.raw_novelty_score,
            self.calibrated_novelty_score,
            self.threshold,
            self.decision_margin,
        )
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("Open-set decision scores must be finite.")
        expected_margin = self.calibrated_novelty_score - self.threshold
        if not math.isclose(self.decision_margin, expected_margin, abs_tol=1e-12):
            raise ValueError("decision_margin must equal novelty score minus threshold.")
        if self.accepted:
            if self.label == UNKNOWN_LABEL or self.candidate_class_id is None:
                raise ValueError("Accepted predictions require a known candidate label.")
            if self.label != self.candidate_class_id:
                raise ValueError("Accepted label must equal the candidate class ID.")
            if self.reason_code != OpenSetReason.ACCEPTED:
                raise ValueError("Accepted predictions require the accepted reason code.")
            if self.decision_margin >= 0.0:
                raise ValueError("Accepted predictions must be below the threshold.")
        else:
            if self.label != UNKNOWN_LABEL:
                raise ValueError("Rejected predictions must use the explicit unknown label.")
            if self.reason_code == OpenSetReason.ACCEPTED:
                raise ValueError("Rejected predictions require a rejection reason code.")
            if self.decision_margin < 0.0:
                raise ValueError("Rejected predictions must meet or exceed the threshold.")


@dataclass(frozen=True)
class RoutingStageCounters:
    """Operation counts for exhaustive or shortlisted open-set routing."""

    sample_count: int
    nodes_executed: int
    compatible_candidate_pairs: int
    shortlisted_candidate_pairs: int
    exact_class_sdf_pairs: int
    primitive_sdf_pairs: int
    score_values_materialized: int

    def __post_init__(self) -> None:
        values = (
            self.sample_count,
            self.nodes_executed,
            self.compatible_candidate_pairs,
            self.shortlisted_candidate_pairs,
            self.exact_class_sdf_pairs,
            self.primitive_sdf_pairs,
            self.score_values_materialized,
        )
        if any(value < 0 for value in values):
            raise ValueError("Routing stage counters must be non-negative.")
        if self.shortlisted_candidate_pairs > self.compatible_candidate_pairs:
            raise ValueError("Shortlisted candidates cannot exceed compatible candidates.")
        if self.exact_class_sdf_pairs > self.shortlisted_candidate_pairs:
            raise ValueError("Exact class scoring cannot exceed the shortlist.")


@dataclass(frozen=True)
class OpenSetBatchResult:
    """Batch wrapper that leaves per-sample decisions explicit and immutable."""

    predictions: tuple[OpenSetPrediction, ...]
    counters: RoutingStageCounters

    def __post_init__(self) -> None:
        if len(self.predictions) != self.counters.sample_count:
            raise ValueError("Prediction count must match the routing sample count.")

    @property
    def candidates_evaluated(self) -> int:
        """Compatibility alias for shortlisted class-sample candidate pairs."""
        return self.counters.shortlisted_candidate_pairs

    @property
    def exact_sdf_evaluations(self) -> int:
        """Compatibility alias for exact class-SDF sample pairs."""
        return self.counters.exact_class_sdf_pairs


def novelty_scores(
    class_scores: Any,
    score_name: str,
    *,
    probabilities: Any | None = None,
    temperature: float = 1.0,
) -> Any:
    """Compute a larger-is-more-novel score from precomputed model outputs."""
    import numpy as np

    scores = np.asarray(class_scores, dtype=np.float64)
    if scores.ndim != 2 or not scores.shape[1]:
        raise ValueError("class_scores must have shape (samples, classes).")
    if not np.all(np.isfinite(scores)):
        raise ValueError("class_scores must be finite.")
    if score_name in ("minimum_sdf", "minimum_metric_corrected_sdf"):
        return np.min(scores, axis=1)
    if score_name == "sdf_energy":
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        logits = -scores / temperature
        maximum = np.max(logits, axis=1)
        return -temperature * (
            maximum
            + np.log(np.sum(np.exp(logits - maximum[:, None]), axis=1))
        )
    if score_name == "maximum_probability":
        if probabilities is None:
            raise ValueError("maximum_probability requires probabilities.")
        values = np.asarray(probabilities, dtype=np.float64)
        if values.shape != scores.shape or not np.all(np.isfinite(values)):
            raise ValueError("probabilities must be finite and align with class_scores.")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("probabilities must lie in [0, 1].")
        return 1.0 - np.max(values, axis=1)
    raise ValueError(f"Unsupported novelty score: {score_name!r}")


def apply_rejection_policy(
    class_scores: Any,
    closed_set_labels: Any,
    class_ids: tuple[Any, ...],
    profile: SupportProfile,
    *,
    model_signature: str,
    feature_transform_fingerprint: str,
    probabilities: Any | None = None,
    calibrated_novelty_scores: Any | None = None,
) -> OpenSetBatchResult:
    """Apply a frozen support profile without fitting or selecting thresholds."""
    import numpy as np

    scores = np.asarray(class_scores, dtype=np.float64)
    labels = np.asarray(closed_set_labels)
    if scores.ndim != 2 or scores.shape[1] != len(profile.class_ids):
        raise ValueError("class_scores must align with support-profile classes.")
    if labels.ndim != 1 or len(labels) != len(scores):
        raise ValueError("closed_set_labels must align with class_scores rows.")
    profile.assert_compatible(
        model_signature=model_signature,
        class_ids=tuple(class_ids),
        feature_transform_fingerprint=feature_transform_fingerprint,
    )
    raw = novelty_scores(
        scores,
        profile.novelty_score,
        probabilities=probabilities,
    )
    if calibrated_novelty_scores is None:
        calibrated = raw
    else:
        calibrated = np.asarray(calibrated_novelty_scores, dtype=np.float64)
        if calibrated.shape != raw.shape or not np.all(np.isfinite(calibrated)):
            raise ValueError("calibrated_novelty_scores must be finite and aligned.")

    decisions = []
    known_classes = set(profile.class_ids)
    for index, candidate in enumerate(labels.tolist()):
        if candidate not in known_classes:
            raise ValueError(f"Closed-set candidate {candidate!r} is absent from profile.")
        threshold = profile.threshold_for(candidate)
        margin = float(calibrated[index] - threshold)
        accepted = margin < 0.0
        reason = (
            OpenSetReason.ACCEPTED
            if accepted
            else OpenSetReason.LOW_CONFIDENCE
            if profile.novelty_score == "maximum_probability"
            else OpenSetReason.OUTSIDE_SUPPORT
        )
        decisions.append(OpenSetPrediction(
            label=candidate if accepted else UNKNOWN_LABEL,
            accepted=accepted,
            candidate_model_signature=model_signature,
            candidate_class_id=candidate,
            raw_novelty_score=float(raw[index]),
            calibrated_novelty_score=float(calibrated[index]),
            threshold=float(threshold),
            decision_margin=margin,
            support_profile_version=profile.version,
            reason_code=reason,
        ))
    return OpenSetBatchResult(
        predictions=tuple(decisions),
        counters=RoutingStageCounters(
            sample_count=len(scores),
            nodes_executed=0,
            compatible_candidate_pairs=len(scores) * scores.shape[1],
            shortlisted_candidate_pairs=len(scores) * scores.shape[1],
            exact_class_sdf_pairs=0,
            primitive_sdf_pairs=0,
            score_values_materialized=scores.size,
        ),
    )