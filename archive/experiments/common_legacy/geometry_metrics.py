import numpy as np
from scipy.spatial import cKDTree

from src.inference_engine import InferenceEngine


def sample_fused_surface(
    experts,
    samples_per_ellipsoid: int = 256,
    seed: int = 42,
    projection_steps: int = 20,
) -> np.ndarray:
    """Sample additive primitives and project them onto the fused zero level set."""
    additive = [
        ellipsoid
        for expert in experts
        for ellipsoid in expert.ellipsoids
        if ellipsoid.polarity > 0
    ]
    if not additive:
        return np.empty((0, 0), dtype=np.float64)

    rng = np.random.default_rng(seed)
    initial = []
    for ellipsoid in additive:
        directions = rng.normal(
            size=(samples_per_ellipsoid, ellipsoid.center.size),
        )
        directions /= np.maximum(
            np.linalg.norm(directions, axis=1, keepdims=True), 1e-12,
        )
        local = directions * ellipsoid.radii
        initial.append(ellipsoid.center + local @ ellipsoid.orientation.T)

    samples = np.concatenate(initial, axis=0)
    engine = InferenceEngine(experts, alpha=experts[0].alpha)
    for sample_index in range(len(samples)):
        point = samples[sample_index]
        for _ in range(projection_steps):
            sdf = float(engine.get_fused_sdf(point.reshape(1, -1))[0])
            if abs(sdf) < 1e-7:
                break
            gradient = engine.get_gradient(point)
            squared_norm = float(gradient @ gradient)
            if squared_norm < 1e-12:
                break
            point = point - sdf * gradient / squared_norm
        samples[sample_index] = point
    return samples


def symmetric_chamfer_distance(
    experts,
    reference_points: np.ndarray,
    samples_per_ellipsoid: int = 256,
    seed: int = 42,
    projection_steps: int = 20,
) -> float:
    """Return symmetric mean-squared Chamfer distance to the fused model surface."""
    reference = np.asarray(reference_points, dtype=np.float64)
    generated = sample_fused_surface(
        experts,
        samples_per_ellipsoid=samples_per_ellipsoid,
        seed=seed,
        projection_steps=projection_steps,
    )
    if len(reference) == 0 or len(generated) == 0:
        return float("nan")
    if reference.ndim != 2 or generated.shape[1] != reference.shape[1]:
        raise ValueError("Reference and generated points must have matching dimensions.")

    reference_to_generated = cKDTree(generated).query(reference, k=1)[0]
    generated_to_reference = cKDTree(reference).query(generated, k=1)[0]
    return float(
        np.mean(reference_to_generated ** 2)
        + np.mean(generated_to_reference ** 2)
    )