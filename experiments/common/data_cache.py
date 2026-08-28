"""Shared paths for large external dataset caches."""

from __future__ import annotations

import os
from pathlib import Path


CACHE_ENVIRONMENT_VARIABLE = "GEODE_CACHE_DIR"


def data_cache_root() -> Path:
    configured = os.environ.get(CACHE_ENVIRONMENT_VARIABLE)
    root = Path(configured).expanduser() if configured else Path.home() / ".cache" / "geode"
    return root.resolve()


def configure_external_cache_environment() -> Path:
    """Point supported third-party downloaders at the GEODE cache.

    Third-party caches live under ``<root parent>/cache`` (F:\\geode-ml\\cache),
    sibling to the GEODE-specific data root, so the corpus/feature caches and
    the downloader caches never mix (20 Aug 2026 relocation: the old defaults
    under C:\\Users\\mak\\.cache were moved there - C: had 0.2 GB free).
    """
    root = data_cache_root()
    # The GEODE data root convention is <top>\data\cache (F:\geode-ml\data\cache);
    # third-party downloader caches live at <top>\cache (F:\geode-ml\cache) -
    # a sibling of `data`, so corpus/feature caches and downloader caches
    # never mix (20 Aug 2026 relocation from C:\Users\mak\.cache).
    third_party = root.parents[1] / "cache"
    mappings = {
        "HF_HOME": third_party / "huggingface",
        "HF_DATASETS_CACHE": third_party / "huggingface" / "datasets",
        "HUGGINGFACE_HUB_CACHE": third_party / "huggingface" / "hub",
        "KAGGLEHUB_CACHE": third_party / "kagglehub",
        "TORCH_HOME": third_party / "torch",
    }
    # FORCED assignment, not setdefault: the shell carries stale D: values
    # from the pre-migration era, and D: is failing hardware (bad sectors,
    # user report 20 Aug) - a stray downloader writing there is a hazard.
    for name, path in mappings.items():
        os.environ[name] = str(path)
    return root