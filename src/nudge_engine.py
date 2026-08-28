import numpy as np
from typing import List, Dict
from src.sdf_engine import EllipsoidExpert, Expert


def _nudge_ellipsoid_from_points(
    e: EllipsoidExpert,
    pts: np.ndarray,
    learning_rate: float,
) -> None:
    """Update one ellipsoid's center, radii, and orientation from assigned points.

    Uses the covariance eigendecomposition of the assigned points rather than
    world-frame per-axis standard deviation:

    - center  ← lerp toward sample mean
    - radii   ← lerp toward √(λᵢ · d)  (matches constructor's covariance fallback)
    - orientation ← lerp toward eigenvectors (columns of cov eigenvector matrix)

    This is consistent with the covariance fallback in ``GreedyConstructor``
    and, unlike the world-frame std update, respects the ellipsoid's orientation
    so anisotropic experts do not drift toward axis-aligned shapes.
    """
    d = pts.shape[1]
    centroid = np.mean(pts, axis=0)
    e.center += learning_rate * (centroid - e.center)

    if len(pts) >= 2:
        cov = np.cov(pts, rowvar=False) if d > 1 else np.var(pts, axis=0, keepdims=True)
        try:
            eigvals, eigvecs = np.linalg.eigh(np.atleast_2d(cov))
        except np.linalg.LinAlgError:
            # Fallback to axis-aligned std if eigh fails
            eigvals = np.var(pts, axis=0)
            eigvecs = np.eye(d)

        # Clamp negative eigenvalues (numerical noise on near-degenerate clusters)
        eigvals = np.maximum(eigvals, 1e-12)

        # Sort descending so the first axis is the principal direction
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        target_radii = np.sqrt(eigvals * d)
        target_orientation = eigvecs  # columns are eigenvectors

        e.radii += learning_rate * (target_radii - e.radii)
        e.radii = np.maximum(e.radii, 1e-6)

        # Interpolate orientation via column-wise lerp then re-orthogonalise
        blended = (1.0 - learning_rate) * e.orientation + learning_rate * target_orientation
        u, _, vt = np.linalg.svd(blended)
        e.orientation = u @ vt  # nearest orthonormal matrix


class NudgeEngine:
    """
    Implements the "Local Nudge" feedback mechanism.
    Refines the parameters (center, radii, and orientation) of locked experts
    based on the points assigned to them using covariance-eigendecomposition
    updates, consistent with the constructor's covariance fallback.
    """

    def __init__(self, learning_rate: float = 0.1, iterations: int = 10):
        """
        Initialize the nudge engine.

        :param learning_rate: The step size for the update (lerp factor ∈ (0, 1]).
        :param iterations: The number of refinement passes to perform.
        """
        self.learning_rate = learning_rate
        self.iterations = iterations

    def assign_points_to_experts(
        self, experts: List[Expert], points: np.ndarray
    ) -> Dict[int, List[np.ndarray]]:
        """Assigns each point to the Expert that yields the minimum fused SDF value."""
        if not experts:
            return {}

        assignments: Dict[int, List[np.ndarray]] = {i: [] for i in range(len(experts))}
        sdf_matrix = np.array([expert.compute_sdf(points) for expert in experts])
        min_indices = np.argmin(sdf_matrix, axis=0)
        for i, min_idx in enumerate(min_indices):
            assignments[min_idx].append(points[i])
        return assignments

    def apply_nudge(
        self, experts: List[Expert], assignments: Dict[int, List[np.ndarray]]
    ) -> None:
        """Refine each Expert's ellipsoids based on their assigned points.

        Uses covariance eigendecomposition to update center, radii, and
        orientation consistently, avoiding the orientation-ignoring world-frame
        std update.  For multi-ellipsoid experts, points are sub-assigned to
        the nearest ellipsoid before the per-ellipsoid update.
        """
        for i, pts in assignments.items():
            if not pts:
                continue

            expert = experts[i]
            pts_arr = np.array(pts)

            if not expert.ellipsoids:
                continue

            if len(expert.ellipsoids) == 1:
                _nudge_ellipsoid_from_points(
                    expert.ellipsoids[0], pts_arr, self.learning_rate
                )
            else:
                # Sub-assign each point to the nearest ellipsoid within this expert
                inner_sdf = np.array([e.compute_sdf(pts_arr) for e in expert.ellipsoids])
                nearest = np.argmin(inner_sdf, axis=0)
                for ei, e in enumerate(expert.ellipsoids):
                    sub = pts_arr[nearest == ei]
                    if len(sub) == 0:
                        continue
                    _nudge_ellipsoid_from_points(e, sub, self.learning_rate)

    def refine(self, experts: List[Expert], points: np.ndarray) -> List[Expert]:
        """Refines the parameters of all Expert objects iteratively."""
        for _ in range(self.iterations):
            assignments = self.assign_points_to_experts(experts, points)
            self.apply_nudge(experts, assignments)
        return experts


def test_nudge_engine():
    """Basic tests for the Nudge Engine."""
    from src.sdf_engine import Expert

    # Wrap EllipsoidExpert in an Expert so apply_nudge has .ellipsoids to work with
    ellipsoid = EllipsoidExpert(center=np.array([0.0, 0.0]), radii=np.array([1.0, 1.0]),
                                orientation=np.eye(2))
    expert = Expert(alpha=1.0)
    expert.add_ellipsoid(ellipsoid)

    points = np.array([[2.0, 2.0], [2.1, 2.1], [1.9, 1.9]])

    engine = NudgeEngine(learning_rate=0.5, iterations=1)

    assignments = engine.assign_points_to_experts([expert], points)
    engine.apply_nudge([expert], assignments)

    # Check if center moved towards (2,2)
    assert ellipsoid.center[0] > 0, f"Expected x to increase, got {ellipsoid.center[0]}"
    assert ellipsoid.center[1] > 0, f"Expected y to increase, got {ellipsoid.center[1]}"

    print("Nudge Engine tests passed!")


if __name__ == "__main__":
    test_nudge_engine()
