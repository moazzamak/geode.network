"""Product hashing primitives: the canonical JSON shape and the payload
hash.

These were previously defined in the experiments package; they are
product infrastructure, so they live here. The experiments package
re-exports them unchanged (layering: experiments -> geode only — the
product package never imports ``experiments.*``).

The canonical shape is a sealed contract: JSON-safe values, sorted
keys, no separators, no NaN, int dict keys stringified. Every content
hash in the product is computed over this shape, and wall-clock fields
never enter a content hash (the standing reproducibility rule).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported manifest value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """The sealed canonical JSON serialization of a value."""
    return json.dumps(
        _json_value(value), sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )


def payload_hash(payload: Any) -> str:
    """The sha256 hex digest of the payload's canonical JSON."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
