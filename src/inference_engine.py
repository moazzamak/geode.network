import numpy as np
from typing import List
from src.sdf_engine import EllipsoidExpert, SoftminFusion, Expert

class InferenceEngine:
    """
    Implements the "Ray Marching Inference" mechanism.
    Calculates the 'depth' of a query point within the assembled manifold
    by marching along the gradient of the fused normalized radial field.
    """
    def __init__(self, experts: List[Expert], alpha: float = 1.0):
        """
        Initialize the inference engine.
        
        :param experts: A list of EllipsoidExpert objects that form the model.
        :param alpha: The concentration parameter for Softmin fusion.
        """
        self.experts = experts
        self.alpha = alpha
        self.fusion = SoftminFusion(alpha=self.alpha)

    def get_fused_sdf(self, points: np.ndarray) -> np.ndarray:
        """
        Computes the fused SDF for a set of points.
        
        :param points: A 2D numpy array of shape (N, d).
        :return: A 1D numpy array of shape (N,) containing fused SDF values.
        """
        return self.fusion.fuse(self.experts, points)

    def get_metric_corrected_sdf(self, points: np.ndarray) -> np.ndarray:
        """Apply the first-order Euclidean correction to the fused SDF."""
        points = np.asarray(points, dtype=np.float64)
        sdf = self.get_fused_sdf(points)
        gradient_norms = np.array([
            np.linalg.norm(self.get_gradient(point)) for point in points
        ])
        return sdf / np.maximum(gradient_norms, 1e-12)

    def get_gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the fused SDF at point x.
        Each Expert handles its own internal gradient (over its ellipsoid group);
        this method fuses those gradients with the top-level Softmin weights.

        :param x: A 1D numpy array of shape (d,).
        :return: A 1D numpy array of shape (d,) representing the gradient.
        """
        sdf_values = np.array([expert.compute_sdf(x.reshape(1, -1))[0] for expert in self.experts])
        dists = sdf_values + 1
        if self.alpha == 0.0:
            weights = np.ones_like(sdf_values)
        else:
            weights = np.exp(-self.alpha * (sdf_values - sdf_values.min()))

        gradients = []
        for i, expert in enumerate(self.experts):
            if dists[i] < 1e-6:
                gradients.append(np.zeros_like(x))
            else:
                gradients.append(expert.compute_gradient(x))

        gradients = np.array(gradients)  # (M, d)
        weighted_grads = gradients * weights.reshape(-1, 1)
        return np.sum(weighted_grads, axis=0) / np.sum(weights)

    def ray_march_depth(self, x: np.ndarray, max_steps: int = 100, step_size: float = 0.1) -> float:
        """Estimate query depth using adaptive gradient ray marching.

        Each step advances by ``|SDF(x)| / ‖∇SDF(x)‖``, the first-order
        Euclidean distance estimate. Unlike a conservative metric SDF, the
        normalized radial field does not guarantee a safe sphere-tracing step.
        Sign changes are therefore detected and interpolated. Relative to a
        fixed step size, the adaptive rule provides:

        1. **Speed** — large steps when far away, tiny steps near the surface.
        2. **Scale consistency** — corrects for the non-metric (Mahalanobis-style)
           normalized SDF whose gradient norm is not 1 for anisotropic ellipsoids.
           Using ``|f| / ‖∇f‖`` converts the SDF value to an approximate
           Euclidean distance before accumulating depth.

        The *step_size* parameter is kept for API compatibility but is only used
        as a minimum step guard to avoid stalling near degenerate points.

        :param x: Query point (d,).
        :param max_steps: Maximum number of marching steps.
        :param step_size: Minimum step guard (prevents stalling near flat regions).
        :return: Approximate Euclidean distance to the nearest surface.
        """
        current_x = x.copy().astype(np.float64)
        current_sdf = self.get_fused_sdf(current_x.reshape(1, -1))[0]

        if abs(current_sdf) < 1e-4:
            return 0.0

        total_distance = 0.0
        for _ in range(max_steps):
            grad = self.get_gradient(current_x)
            grad_norm = np.linalg.norm(grad)

            if grad_norm < 1e-6:
                break

            # First-order distance estimate; sign changes are checked below.
            # Clamp from below by step_size / 10 to avoid stalling.
            sphere_step = max(abs(current_sdf) / grad_norm, step_size * 0.1)

            direction = -np.sign(current_sdf) * grad / grad_norm

            next_x = current_x + direction * sphere_step
            next_sdf = self.get_fused_sdf(next_x.reshape(1, -1))[0]

            # Surface crossed: interpolate for a precise distance estimate.
            if (current_sdf > 0 and next_sdf <= 0) or (current_sdf < 0 and next_sdf >= 0):
                fraction = abs(current_sdf) / (abs(current_sdf) + abs(next_sdf))
                return total_distance + fraction * sphere_step

            current_x = next_x
            total_distance += sphere_step
            current_sdf = next_sdf

        return total_distance

def test_inference_engine():
    """Basic tests for the Inference Engine."""
    from src.sdf_engine import Expert, EllipsoidExpert

    # Wrap EllipsoidExpert in an Expert (InferenceEngine expects Expert objects)
    ellipsoid = EllipsoidExpert(center=np.array([0.0, 0.0]), radii=np.array([1.0, 1.0]))
    expert = Expert(alpha=1.0)
    expert.add_ellipsoid(ellipsoid)

    engine = InferenceEngine(experts=[expert], alpha=1.0)

    # Test 1: Point outside (2,0) should have a positive depth
    # Distance to circle x^2 + y^2 = 1 from (2,0) is 1.0
    x_outside = np.array([2.0, 0.0])
    depth_outside = engine.ray_march_depth(x_outside)
    print(f"Outside depth (expected ~1.0): {depth_outside}")
    assert np.isclose(depth_outside, 1.0, atol=0.15)

    # Test 2: Point inside (0.5, 0) should have a positive depth
    # Distance to circle x^2 + y^2 = 1 from (0.5, 0) is 0.5
    x_inside = np.array([0.5, 0.0])
    depth_inside = engine.ray_march_depth(x_inside)
    print(f"Inside depth (expected ~0.5): {depth_inside}")
    assert np.isclose(depth_inside, 0.5, atol=0.15)

    # Test 3: Point on surface (1, 0) should have 0 depth
    x_surface = np.array([1.0, 0.0])
    depth_surface = engine.ray_march_depth(x_surface)
    print(f"Surface depth (expected 0.0): {depth_surface}")
    assert np.isclose(depth_surface, 0.0, atol=1e-3)

if __name__ == "__main__":
    test_inference_engine()