"""
Dataset preparation utilities.
Each function is idempotent: it returns immediately if the output NPZ already exists.
All downloads are cached locally so subsequent runs are instant.
"""
import os

import numpy as np

from experiments.common.data_cache import configure_external_cache_environment

MODELNET10_NPZ = "data/tier2/modelnet10_pointclouds.npz"
CIFAR10_NPZ = "data/tier4/cifar10_features.npz"
CIFAR100_NPZ = "data/tier5/cifar100_superclass.npz"


# ---------------------------------------------------------------------------
# OFF mesh parser (ModelNet10)
# ---------------------------------------------------------------------------

def _parse_off_vertices(filepath: str):
    """Return (N, 3) float64 array of vertex coordinates from an OFF file, or None."""
    try:
        with open(filepath, "r", errors="replace") as fh:
            # Strip blank lines and comments, keep original order
            lines = [
                ln.rstrip("\n")
                for ln in fh
                if ln.strip() and not ln.strip().startswith("#")
            ]

        if not lines:
            return None

        first = lines[0].strip()
        if not first.upper().startswith("OFF"):
            return None

        # Counts may be on the same line as OFF ("OFF 12 20 0") or on the next line
        inline = first[3:].strip()
        if inline:
            parts = inline.split()
            num_v = int(parts[0])
            data_start = 1
        else:
            parts = lines[1].split()
            num_v = int(parts[0])
            data_start = 2

        verts = []
        for ln in lines[data_start : data_start + num_v]:
            coords = ln.split()
            verts.append([float(coords[0]), float(coords[1]), float(coords[2])])

        if len(verts) < 3:
            return None
        return np.array(verts, dtype=np.float64)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Tier 2: ModelNet10
# ---------------------------------------------------------------------------

def prepare_modelnet10(
    output_path: str = MODELNET10_NPZ,
    points_per_mesh: int = 1024,
    max_shapes: int = 200,
    seed: int = 42,
):
    """
    Download ModelNet10 via kagglehub (first run only), parse OFF vertex clouds,
    and save a single NPZ cache at *output_path* with key 'pointclouds' shaped
    (N, points_per_mesh, 3).

    Subsequent calls are no-ops if the file already exists.
    """
    if os.path.exists(output_path):
        return

    configure_external_cache_environment()
    try:
        import kagglehub
    except ImportError:
        raise ImportError(
            "kagglehub is required for ModelNet10 download. "
            "Install with: pip install kagglehub"
        )

    print("  Downloading ModelNet10 via kagglehub (first run only)...")
    raw_path = kagglehub.dataset_download(
        "balraj98/modelnet10-princeton-3d-object-dataset"
    )
    print(f"  Dataset cached at: {raw_path}")

    # Collect all OFF files
    off_files = []
    for root, _, files in os.walk(raw_path):
        for fname in files:
            if fname.lower().endswith(".off"):
                off_files.append(os.path.join(root, fname))

    if not off_files:
        raise RuntimeError(f"No .off files found under {raw_path}")

    print(f"  Found {len(off_files)} OFF files. Sampling up to {max_shapes}...")

    rng = np.random.default_rng(seed)
    chosen_idx = rng.permutation(len(off_files))[:max_shapes]
    selected = [off_files[i] for i in chosen_idx]

    pointclouds = []
    skipped = 0
    for fp in selected:
        verts = _parse_off_vertices(fp)
        if verts is None or len(verts) < 3:
            skipped += 1
            continue
        # Sample exactly points_per_mesh points (with replacement if mesh is small)
        replace = len(verts) < points_per_mesh
        idx = rng.choice(len(verts), points_per_mesh, replace=replace)
        pointclouds.append(verts[idx])

    if not pointclouds:
        raise RuntimeError("No valid point clouds could be parsed from the dataset.")

    if skipped:
        print(f"  Skipped {skipped} unreadable OFF files.")

    print(f"  Parsed {len(pointclouds)} point clouds ({points_per_mesh} pts each).")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write to a temporary file first, then rename for atomicity.
    # np.savez_compressed appends .npz if the path doesn't end in it,
    # so keep .npz in the tmp name to match what numpy actually writes.
    tmp_path = output_path.replace(".npz", ".tmp.npz")
    np.savez_compressed(
        tmp_path, pointclouds=np.array(pointclouds, dtype=np.float32)
    )
    os.replace(tmp_path, output_path)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Tier 4: CIFAR-10
# ---------------------------------------------------------------------------

def prepare_cifar10(
    output_path: str = CIFAR10_NPZ,
    seed: int = 42,
):
    """
    Download CIFAR-10 via HuggingFace datasets (first run only — cached automatically),
    and save an NPZ cache at *output_path* with keys:
      'images'  shaped (N, 32, 32, 3)  uint8
      'labels'  shaped (N,)             int32

    Subsequent calls are no-ops if the NPZ already exists.
    """
    if os.path.exists(output_path):
        return

    cache_root = configure_external_cache_environment()
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required for CIFAR-10 download. "
            "Install with: pip install datasets"
        )

    print("  Downloading CIFAR-10 via HuggingFace datasets (first run only)...")
    ds = load_dataset(
        "uoft-cs/cifar10", cache_dir=str(cache_root / "huggingface" / "datasets"),
    )

    all_images, all_labels = [], []
    for split_name in ("train", "test"):
        split = ds[split_name]
        for item in split:
            # Each item: {'img': PIL.Image, 'label': int}
            img = item["img"]
            all_images.append(np.array(img, dtype=np.uint8))      # (32, 32, 3)
            all_labels.append(int(item["label"]))

    images = np.stack(all_images, axis=0)                          # (N, 32, 32, 3)
    labels = np.array(all_labels, dtype=np.int32)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"  Saving {len(images)} images to {output_path}...")

    tmp_path = output_path.replace(".npz", ".tmp.npz")
    np.savez_compressed(tmp_path, images=images, labels=labels)
    os.replace(tmp_path, output_path)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Tier 5: CIFAR-100 (coarse superclass labels)
# ---------------------------------------------------------------------------

def prepare_cifar100(
    output_path: str = CIFAR100_NPZ,
    seed: int = 42,
):
    """
    Download CIFAR-100 via HuggingFace datasets (first run only — cached automatically),
    and save an NPZ cache at *output_path* with keys:
      'images'        shaped (N, 32, 32, 3)  uint8
      'coarse_labels' shaped (N,)             int32 in [0, 19]

    The 20 coarse superclass labels (e.g. "aquatic mammals", "vehicles 1") are
    used instead of the 100 fine-grained labels: they give 3 000 samples/class
    (vs 600) which is sufficient for RANSAC at d=19 (LDA) with realistic sample
    budgets.

    Subsequent calls are no-ops if the file already exists.
    """
    if os.path.exists(output_path):
        return

    cache_root = configure_external_cache_environment()
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required for CIFAR-100 download. "
            "Install with: pip install datasets"
        )

    print("  Downloading CIFAR-100 via HuggingFace datasets (first run only)...")
    ds = load_dataset(
        "uoft-cs/cifar100", cache_dir=str(cache_root / "huggingface" / "datasets"),
    )

    all_images, all_coarse = [], []
    for split_name in ("train", "test"):
        split = ds[split_name]
        for item in split:
            img = item["img"]
            all_images.append(np.array(img, dtype=np.uint8))   # (32, 32, 3)
            all_coarse.append(int(item["coarse_label"]))

    images = np.stack(all_images, axis=0)                       # (N, 32, 32, 3)
    coarse_labels = np.array(all_coarse, dtype=np.int32)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"  Saving {len(images)} images to {output_path}...")

    tmp_path = output_path.replace(".npz", ".tmp.npz")
    np.savez_compressed(tmp_path, images=images, coarse_labels=coarse_labels)
    os.replace(tmp_path, output_path)
    print(f"  Saved: {output_path}")
