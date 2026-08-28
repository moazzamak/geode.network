import numpy as np
from scipy.optimize import minimize

from src.sdf_engine import EllipsoidExpert


def numerical_closest_point(
    ellipsoid: EllipsoidExpert,
    point: np.ndarray,
) -> np.ndarray:
    """Return a deterministic numerical closest point for a 2D/3D ellipsoid."""
    point = np.asarray(point, dtype=np.float64)
    dimensions = len(ellipsoid.center)
    if dimensions not in (2, 3) or point.shape != (dimensions,):
        raise ValueError("The numerical reference supports one 2D or 3D point.")

    local_point = (point - ellipsoid.center) @ ellipsoid.orientation
    radii = ellipsoid.radii
    normalized_radius = np.linalg.norm(local_point / radii)
    if normalized_radius > 1e-12:
        radial_start = local_point / normalized_radius
    else:
        radial_start = np.zeros(dimensions)
        radial_start[np.argmin(radii)] = np.min(radii)

    starts = [radial_start]
    for axis in range(dimensions):
        endpoint = np.zeros(dimensions)
        endpoint[axis] = radii[axis]
        starts.extend((endpoint, -endpoint))

    def objective(surface_point: np.ndarray) -> float:
        difference = surface_point - local_point
        return float(difference @ difference)

    def surface_constraint(surface_point: np.ndarray) -> float:
        return float(np.sum((surface_point / radii) ** 2) - 1.0)

    candidates = []
    for start in starts:
        result = minimize(
            objective,
            start,
            method="SLSQP",
            constraints={"type": "eq", "fun": surface_constraint},
            options={"ftol": 1e-13, "maxiter": 500},
        )
        if result.success and abs(surface_constraint(result.x)) <= 1e-7:
            candidates.append(result.x)
    if not candidates:
        raise RuntimeError("Numerical closest-point optimization did not converge.")

    closest_local = min(candidates, key=objective)
    return ellipsoid.center + closest_local @ ellipsoid.orientation.T


def numerical_signed_distance(
    ellipsoid: EllipsoidExpert,
    points: np.ndarray,
) -> np.ndarray:
    """Compute research-reference signed Euclidean distances in 2D or 3D."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError("points must have shape (N, d).")
    distances = np.array([
        np.linalg.norm(point - numerical_closest_point(ellipsoid, point))
        for point in points
    ])
    signs = np.where(ellipsoid.compute_sdf(points) < 0.0, -1.0, 1.0)
    return signs * distances