import numpy as np
import os

def generate_sphere_data(center, radius, num_points=1000, noise_std=0.05):
    """Generates points on the surface of a sphere with added noise."""
    phi = np.random.uniform(0, 2*np.pi, num_points)
    costheta = np.random.uniform(-1, 1, num_points)
    theta = np.arccos(costheta)
    
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    
    points = np.stack([x, y, z], axis=-1) + center
    points += np.random.normal(0, noise_std, points.shape)
    return points

def generate_ellipsoid_data(center, radii, num_points=1000, noise_std=0.05):
    """Generates points on the surface of an ellipsoid with added noise."""
    phi = np.random.uniform(0, 2*np.pi, num_points)
    costheta = np.random.uniform(-1, 1, num_points)
    theta = np.arccos(costheta)
    
    x = radii[0] * np.sin(theta) * np.cos(phi)
    y = radii[1] * np.sin(theta) * np.sin(phi)
    z = radii[2] * np.cos(theta)
    
    points = np.stack([x, y, z], axis=-1) + center
    points += np.random.normal(0, noise_std, points.shape)
    return points

def main():
    os.makedirs("data/tier1", exist_ok=True)
    rng = np.random.default_rng(42)

    def save_train_test(points: np.ndarray, base_name: str, test_fraction: float = 0.2):
        indices = np.arange(len(points))
        rng.shuffle(indices)
        test_size = int(round(len(points) * test_fraction))
        test_size = min(max(test_size, 1), len(points) - 1)
        test_idx = indices[:test_size]
        train_idx = indices[test_size:]

        train_points = points[train_idx]
        test_points = points[test_idx]
        np.save(f"data/tier1/{base_name}_train.npy", train_points)
        np.save(f"data/tier1/{base_name}_test.npy", test_points)
        np.save(f"data/tier1/{base_name}.npy", points)
        print(
            f"Generated {base_name}: train={train_points.shape[0]}, "
            f"test={test_points.shape[0]}, total={points.shape[0]}"
        )
    
    # 1. Sphere
    center1 = np.array([0.0, 0.0, 0.0])
    radius1 = np.array([1.0, 1.0, 1.0])
    points1 = generate_sphere_data(center1, radius1[0], num_points=1500, noise_std=0.02)
    save_train_test(points1, "sphere")

    # 2. Ellipsoid (slightly rotated/shifted)
    center2 = np.array([3.0, 0.0, 0.0])
    radii2 = np.array([1.5, 0.8, 1.2])
    points2 = generate_ellipsoid_data(center2, radii2, num_points=1500, noise_std=0.03)
    save_train_test(points2, "ellipsoid")

if __name__ == "__main__":
    main()