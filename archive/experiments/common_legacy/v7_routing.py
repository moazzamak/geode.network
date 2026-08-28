from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from experiments.common.v5_artifacts import payload_hash
from experiments.common.v7_protocol import EmpiricalRoutingProfile
from src.subspace_primitive import fit_subspace_primitive


@dataclass(frozen=True)
class RoutingProfileModel:
    metadata: EmpiricalRoutingProfile
    family: str
    class_centers: np.ndarray
    class_labels: np.ndarray
    threshold: float
    state: tuple[Any, ...]

    def score(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if self.family == "centroid_radius":
            distances = np.linalg.norm(
                values[:, None, :] - self.class_centers[None, :, :], axis=2
            )
            return np.min(distances, axis=1)
        if self.family == "compact_prototypes":
            prototypes = np.asarray(self.state[0], dtype=np.float64)
            distances = np.linalg.norm(
                values[:, None, :] - prototypes[None, :, :], axis=2
            )
            return np.min(distances, axis=1)
        if self.family == "low_rank_gaussian":
            primitives = self.state[0]
            likelihoods = np.column_stack(
                [primitive.log_likelihood(values) for primitive in primitives]
            )
            return -np.max(likelihoods, axis=1)
        if self.family == "autoencoder_reconstruction":
            center = np.asarray(self.state[0], dtype=np.float64)
            basis = np.asarray(self.state[1], dtype=np.float64)
            centered = values - center
            residual = centered - (centered @ basis) @ basis.T
            return np.sum(residual * residual, axis=1)
        raise ValueError(f"Unsupported profile family: {self.family}")

    @property
    def state_hash(self) -> str:
        arrays = [self.class_centers, self.class_labels]
        for value in self.state:
            if isinstance(value, np.ndarray):
                arrays.append(value)
            elif isinstance(value, tuple):
                for item in value:
                    arrays.extend(
                        [
                            item.center,
                            item.basis,
                            item.tangent_variances,
                            np.asarray([item.residual_variance]),
                        ]
                    )
        return payload_hash(
            {
                "family": self.family,
                "threshold": self.threshold,
                "arrays": [
                    {
                        "shape": list(array.shape),
                        "dtype": str(array.dtype),
                        "bytes": array.tobytes().hex(),
                    }
                    for array in arrays
                ],
            }
        )


def fit_routing_profile(
    family: str,
    features: np.ndarray,
    labels: np.ndarray,
    calibration: np.ndarray,
    *,
    model_signature: str,
    representation_hash: str,
    rank: int,
    prototypes_per_class: int,
    quantile: float,
    seed: int,
) -> RoutingProfileModel:
    classes = np.unique(labels)
    centers = np.vstack([features[labels == label].mean(axis=0) for label in classes])
    if family == "centroid_radius":
        state: tuple[Any, ...] = ()
    elif family == "compact_prototypes":
        rng = np.random.default_rng(seed)
        prototypes = np.concatenate(
            [
                features[
                    rng.choice(
                        np.flatnonzero(labels == label),
                        min(prototypes_per_class, np.sum(labels == label)),
                        replace=False,
                    )
                ]
                for label in classes
            ]
        )
        state = (prototypes,)
    elif family == "low_rank_gaussian":
        primitives = tuple(
            fit_subspace_primitive(
                features[labels == label],
                min(rank, features.shape[1] - 1, np.sum(labels == label) - 2),
                class_label=int(label),
            )
            for label in classes
        )
        state = (primitives,)
    elif family == "autoencoder_reconstruction":
        center = features.mean(axis=0)
        _, _, vectors = np.linalg.svd(features - center, full_matrices=False)
        basis = vectors[: min(rank, len(vectors))].T
        state = (center, basis)
    else:
        raise ValueError(f"Unsupported profile family: {family}")
    provisional = RoutingProfileModel(
        metadata=EmpiricalRoutingProfile(
            model_signature=model_signature,
            representation_hash=representation_hash,
            class_order=tuple(str(value) for value in classes),
            profile_family=family,
            fit_data_hash=payload_hash(
                {"shape": list(features.shape), "bytes": features.tobytes().hex()}
            ),
            calibration_data_hash=payload_hash(
                {"shape": list(calibration.shape), "bytes": calibration.tobytes().hex()}
            ),
            score_direction="lower_is_match",
            threshold=0.0,
            dimension=features.shape[1],
        ),
        family=family,
        class_centers=centers,
        class_labels=classes,
        threshold=0.0,
        state=state,
    )
    threshold = float(np.quantile(provisional.score(calibration), quantile, method="higher"))
    metadata = EmpiricalRoutingProfile(
        model_signature=model_signature,
        representation_hash=representation_hash,
        class_order=tuple(str(value) for value in classes),
        profile_family=family,
        fit_data_hash=provisional.metadata.fit_data_hash,
        calibration_data_hash=provisional.metadata.calibration_data_hash,
        score_direction="lower_is_match",
        threshold=threshold,
        dimension=features.shape[1],
    )
    return RoutingProfileModel(
        metadata=metadata,
        family=family,
        class_centers=centers,
        class_labels=classes,
        threshold=threshold,
        state=state,
    )


def route_profiles(
    profiles: tuple[RoutingProfileModel, ...],
    features: np.ndarray,
    *,
    shortlist_size: int,
) -> tuple[np.ndarray, list[tuple[int, ...]], np.ndarray]:
    scores = np.column_stack([profile.score(features) for profile in profiles])
    normalized = scores / np.asarray([profile.threshold for profile in profiles])[None, :]
    top1 = np.argmin(normalized, axis=1)
    shortlists = []
    fallbacks = np.zeros(len(features), dtype=bool)
    for row in range(len(features)):
        confident = np.flatnonzero(scores[row] <= np.asarray(
            [profile.threshold for profile in profiles]
        ))
        if len(confident) == 0:
            shortlists.append(tuple(range(len(profiles))))
            fallbacks[row] = True
        else:
            ordered = np.argsort(normalized[row])
            selected = tuple(
                int(index)
                for index in ordered
                if index in set(confident)
            )[:shortlist_size]
            shortlists.append(selected)
    return top1, shortlists, fallbacks
