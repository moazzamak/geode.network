from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
from scipy.stats import weibull_min
from sklearn.covariance import LedoitWolf
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


def minimum_sdf_score(class_sdfs: np.ndarray) -> np.ndarray:
    """Return an OOD score where larger minimum class SDF is more OOD-like."""
    scores = np.asarray(class_sdfs, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] == 0:
        raise ValueError("class_sdfs must have shape (samples, classes).")
    return np.min(scores, axis=1)


def sdf_energy_score(class_sdfs: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Return negative SDF energy, oriented so larger values indicate OOD."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    scores = np.asarray(class_sdfs, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] == 0:
        raise ValueError("class_sdfs must have shape (samples, classes).")
    return -temperature * logsumexp(-scores / temperature, axis=1)


def maximum_probability_score(probabilities: np.ndarray) -> np.ndarray:
    """Return one minus maximum class probability as an OOD score."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] == 0:
        raise ValueError("probabilities must have shape (samples, classes).")
    return 1.0 - np.max(probabilities, axis=1)


@dataclass
class FeatureOODScorers:
    mahalanobis: LedoitWolf
    gmm: GaussianMixture
    neighbors: NearestNeighbors
    knn_k: int

    def score(self, features: np.ndarray) -> dict[str, np.ndarray]:
        features = np.asarray(features, dtype=np.float64)
        centered = features - self.mahalanobis.location_
        squared_distance = np.einsum(
            "ni,ij,nj->n", centered, self.mahalanobis.precision_, centered,
        )
        neighbor_distances, _ = self.neighbors.kneighbors(features)
        return {
            "mahalanobis": np.sqrt(np.maximum(squared_distance, 0.0)),
            "gmm_nll": -self.gmm.score_samples(features),
            "knn_distance": neighbor_distances[:, self.knn_k - 1],
        }


@dataclass
class ClassConditionalOODScorers:
    classes: np.ndarray
    estimators: tuple[LedoitWolf, ...]
    log_priors: np.ndarray
    tail_thresholds: np.ndarray
    tail_shapes: np.ndarray
    tail_scales: np.ndarray

    def score(self, features: np.ndarray) -> dict[str, np.ndarray]:
        features = np.asarray(features, dtype=np.float64)
        squared_distances = []
        gaussian_nlls = []
        for estimator in self.estimators:
            centered = features - estimator.location_
            squared_distance = np.einsum(
                "ni,ij,nj->n", centered, estimator.precision_, centered,
            )
            _, log_determinant = np.linalg.slogdet(estimator.covariance_)
            squared_distances.append(np.maximum(squared_distance, 0.0))
            gaussian_nlls.append(0.5 * (
                squared_distance
                + log_determinant
                + features.shape[1] * np.log(2.0 * np.pi)
            ))
        squared_distance_matrix = np.column_stack(squared_distances)
        nll_matrix = np.column_stack(gaussian_nlls)
        nearest_classes = np.argmin(squared_distance_matrix, axis=1)
        nearest_distances = np.sqrt(
            squared_distance_matrix[np.arange(len(features)), nearest_classes],
        )
        excess = np.maximum(
            nearest_distances - self.tail_thresholds[nearest_classes], 0.0,
        )
        tail_probability = weibull_min.cdf(
            excess,
            self.tail_shapes[nearest_classes],
            loc=0.0,
            scale=self.tail_scales[nearest_classes],
        )
        return {
            "class_mahalanobis": np.sqrt(
                np.min(squared_distance_matrix, axis=1),
            ),
            "gaussian_mixture_nll": -logsumexp(
                self.log_priors - nll_matrix, axis=1,
            ),
            "class_tail_probability": tail_probability,
        }


def fit_feature_ood_scorers(
    geometry_features: np.ndarray,
    *,
    gmm_components: int = 5,
    knn_k: int = 5,
    seed: int = 42,
) -> FeatureOODScorers:
    """Fit matched OOD baselines using in-distribution geometry data only."""
    features = np.asarray(geometry_features, dtype=np.float64)
    if features.ndim != 2 or len(features) < 2:
        raise ValueError("geometry_features must contain at least two samples.")
    if knn_k < 1 or knn_k > len(features):
        raise ValueError("knn_k must be between one and the geometry sample count.")
    components = min(max(1, gmm_components), len(features))
    return FeatureOODScorers(
        mahalanobis=LedoitWolf().fit(features),
        gmm=GaussianMixture(
            n_components=components,
            covariance_type="full",
            reg_covar=1e-6,
            random_state=seed,
        ).fit(features),
        neighbors=NearestNeighbors(n_neighbors=knn_k).fit(features),
        knn_k=knn_k,
    )


def fit_class_conditional_ood_scorers(
    geometry_features: np.ndarray,
    geometry_labels: np.ndarray,
    *,
    tail_fraction: float = 0.1,
) -> ClassConditionalOODScorers:
    """Fit class-conditional Gaussian and Weibull-tail OOD controls."""
    features = np.asarray(geometry_features, dtype=np.float64)
    labels = np.asarray(geometry_labels)
    if features.ndim != 2 or len(features) != len(labels):
        raise ValueError("Geometry features and labels must have matching rows.")
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must be in (0, 1].")
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2 or np.any(counts < 3):
        raise ValueError("At least two classes with three samples each are required.")

    estimators = tuple(LedoitWolf().fit(features[labels == class_id]) for class_id in classes)
    tail_thresholds = []
    tail_shapes = []
    tail_scales = []
    for class_id, estimator in zip(classes, estimators):
        class_features = features[labels == class_id]
        centered = class_features - estimator.location_
        distances = np.sqrt(np.maximum(np.einsum(
            "ni,ij,nj->n", centered, estimator.precision_, centered,
        ), 0.0))
        tail_count = min(len(distances), max(3, int(np.ceil(
            tail_fraction * len(distances),
        ))))
        threshold = float(np.partition(distances, len(distances) - tail_count)[
            len(distances) - tail_count
        ])
        excess = np.maximum(np.sort(distances)[-tail_count:] - threshold, 0.0)
        shape, _, scale = weibull_min.fit(excess + np.finfo(float).eps, floc=0.0)
        tail_thresholds.append(threshold)
        tail_shapes.append(shape)
        tail_scales.append(max(float(scale), np.finfo(float).eps))

    return ClassConditionalOODScorers(
        classes=classes,
        estimators=estimators,
        log_priors=np.log(counts / counts.sum()),
        tail_thresholds=np.asarray(tail_thresholds),
        tail_shapes=np.asarray(tail_shapes),
        tail_scales=np.asarray(tail_scales),
    )