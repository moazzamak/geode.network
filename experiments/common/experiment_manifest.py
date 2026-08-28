import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# The canonical shape and the payload hash are PRODUCT primitives;
# the experiments package re-exports them (experiments -> geode only).
from geode.hashing import _json_value, canonical_json


def experiment_id(config: dict, length: int = 16) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()[:length]


def array_fingerprint(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(canonical_json(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _git_metadata(repo_root: Path) -> dict:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=repo_root, check=True,
            capture_output=True, text=True,
        )
        return result.stdout.strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


def build_manifest(
    config: dict,
    seed: int,
    repo_root: str | Path,
    dataset_fingerprint: str,
    split_indices: np.ndarray,
    features: np.ndarray,
    device: str,
    packages: Iterable[str] = ("numpy", "scikit-learn", "pyopencl"),
) -> dict:
    package_versions = {}
    for package in packages:
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None

    normalized_config = _json_value(config)
    return {
        "experiment_id": experiment_id(normalized_config),
        "config": normalized_config,
        "seed": int(seed),
        "git": _git_metadata(Path(repo_root)),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_versions,
            "device": str(device),
        },
        "data": {
            "dataset_fingerprint": str(dataset_fingerprint),
            "split_hash": array_fingerprint(split_indices),
            "feature_hash": array_fingerprint(features),
        },
    }


def append_manifest(path: str | Path, manifest: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical_json(manifest) + "\n")


def read_manifests(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]