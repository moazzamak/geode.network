from collections.abc import Callable

import numpy as np
from sklearn.covariance import LedoitWolf, MinCovDet
from sklearn.mixture import GaussianMixture

from src.greedy_constructor import GreedyConstructor
from src.sdf_engine import EllipsoidExpert


def _ellipsoid_from_covariance(
    center: np.ndarray,
    covariance: np.ndarray,
    surface_scale: float,
) -> EllipsoidExpert:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if not np.all(np.isfinite(eigenvalues)) or np.any(eigenvalues <= 1e-12):
        raise ValueError("Covariance must be finite and positive definite.")
    order = np.argsort(eigenvalues)[::-1]
    radii = np.sqrt(eigenvalues[order] * surface_scale)
    return EllipsoidExpert(center, radii, eigenvectors[:, order])


def fit_quadric_svd(points: np.ndarray, seed: int = 0) -> EllipsoidExpert:
    return GreedyConstructor(
        seed=seed,
        primitive_family="ellipsoid",
    )._generate_candidate(points)


def fit_full_covariance(points: np.ndarray, seed: int = 0) -> EllipsoidExpert:
    del seed
    dimension = points.shape[1]
    return _ellipsoid_from_covariance(
        np.mean(points, axis=0), np.cov(points, rowvar=False), dimension,
    )


def fit_diagonal_covariance(points: np.ndarray, seed: int = 0) -> EllipsoidExpert:
    del seed
    dimension = points.shape[1]
    covariance = np.diag(np.var(points, axis=0, ddof=1))
    return _ellipsoid_from_covariance(
        np.mean(points, axis=0), covariance, dimension,
    )


def fit_spherical_covariance(points: np.ndarray, seed: int = 0) -> EllipsoidExpert:
    del seed
    dimension = points.shape[1]
    variance = float(np.mean(np.var(points, axis=0, ddof=1)))
    radius = float(np.sqrt(variance * dimension))
    if not np.isfinite(radius) or radius <= 1e-6:
        raise ValueError("Spherical variance must be finite and positive.")
    return EllipsoidExpert(
        np.mean(points, axis=0),
        np.full(dimension, radius),
        np.eye(dimension),
    )


def fit_shrinkage_covariance(points: np.ndarray, seed: int = 0) -> EllipsoidExpert:
    del seed
    estimator = LedoitWolf().fit(points)
    return _ellipsoid_from_covariance(
        estimator.location_, estimator.covariance_, points.shape[1],
    )


def fit_minimum_covariance_determinant(
    points: np.ndarray, seed: int = 0,
) -> EllipsoidExpert:
    estimator = MinCovDet(random_state=seed).fit(points)
    return _ellipsoid_from_covariance(
        estimator.location_, estimator.covariance_, points.shape[1],
    )


def fit_low_rank_covariance(points: np.ndarray, seed: int = 0) -> EllipsoidExpert:
    del seed
    dimension = points.shape[1]
    covariance = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    rank = min(5, dimension - 1)
    residual = max(float(np.mean(eigenvalues[rank:])), 1e-12)
    low_rank = eigenvectors[:, :rank] @ np.diag(
        np.maximum(eigenvalues[:rank] - residual, 0.0)
    ) @ eigenvectors[:, :rank].T
    regularized = low_rank + residual * np.eye(dimension)
    return _ellipsoid_from_covariance(
        np.mean(points, axis=0), regularized, dimension,
    )


def fit_gmm_covariance(points: np.ndarray, seed: int = 0) -> EllipsoidExpert:
    estimator = GaussianMixture(
        n_components=1,
        covariance_type="full",
        reg_covar=1e-6,
        random_state=seed,
    ).fit(points)
    return _ellipsoid_from_covariance(
        estimator.means_[0], estimator.covariances_[0], points.shape[1],
    )


ELLIPSOID_FITTERS: dict[str, Callable[[np.ndarray, int], EllipsoidExpert]] = {
    "quadric_svd": fit_quadric_svd,
    "full_covariance": fit_full_covariance,
    "diagonal_covariance": fit_diagonal_covariance,
    "spherical_covariance": fit_spherical_covariance,
    "shrinkage_covariance": fit_shrinkage_covariance,
    "minimum_covariance_determinant": fit_minimum_covariance_determinant,
    "low_rank_covariance": fit_low_rank_covariance,
    "gmm_covariance": fit_gmm_covariance,
}

FITTER_PRIMITIVE_FAMILIES = {
    "full_covariance": "ellipsoid",
    "diagonal_covariance": "diagonal_ellipsoid",
    "spherical_covariance": "sphere",
}

GPU_CANDIDATE_FITTERS = frozenset(FITTER_PRIMITIVE_FAMILIES)