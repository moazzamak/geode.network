import sys
from pathlib import Path

import numpy as np

reference_dir = Path("logs/results/v12/m70_native_domainnet/arrays")
candidate_dir = Path(sys.argv[1])

reference_features = np.load(reference_dir / "features.npy")
reference_labels = np.load(reference_dir / "labels.npy")
candidate_features = np.load(candidate_dir / "features.npy")
candidate_labels = np.load(candidate_dir / "labels.npy")

print("reference", reference_features.shape, "candidate", candidate_features.shape)

left = reference_features[reference_labels == 0][:100]
right = candidate_features[candidate_labels == 0][:100]
print("class 0 max absolute difference:", np.abs(left - right).max())
print("per-row max difference, first 10:", np.round(np.abs(left - right).max(axis=1)[:10], 4))

distances = np.linalg.norm(right[:, None, :] - left[None, :, :], axis=2)
nearest = distances.argmin(axis=1)
print("nearest reference row for candidate rows 0..9:", nearest[:10])
print("is a pure permutation:", sorted(nearest.tolist()) == list(range(100)))
print("matched distances, first 10:", np.round(distances[np.arange(100), nearest][:10], 6))
print("reference row norms, first 5:", np.round(np.linalg.norm(left[:5], axis=1), 4))
print("candidate row norms, first 5:", np.round(np.linalg.norm(right[:5], axis=1), 4))
