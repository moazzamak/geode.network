"""Task descriptors (v24 section 3): the ontology v0 axis schema and the
deterministic normaliser.

Design invariants honoured here (v24 section 2.1):
- I4 no-refusal: every task gets a descriptor; out-of-vocabulary
  attribute values fall back to a registered token and the event is
  logged.
- I5 repro-hash: the canonical descriptor serialises deterministically
  (fixed axis order, sorted keys) and hashes without volatile fields.

The axis table is PROVISIONAL (v0, code-level); M166 ratifies it and
freezes ``task_ontology_v0.json``.
"""
from __future__ import annotations

import json
from typing import Any

from geode.hashing import payload_hash

FALLBACK_TOKEN = "<oov>"

# Canonical axis order. The axis order IS part of the frozen schema;
# changing it changes every descriptor hash, so it changes only via a
# versioned ontology migration.
AXES: dict[str, list[str]] = {
    "input.modality": ["image", "token-text", "numeric-series", "tabular",
                       "graph", "audio", "control-signal"],
    "input.submodality": ["camera-RGB", "pointcloud", "IR/thermal", "depth",
                          "medical", "none"],
    "input.value_kind": ["discrete", "continuous", "mixed"],
    "input.temporal_structure": ["iid", "sequential", "delayed"],
    "output.kind": ["class", "regression", "next-token", "action",
                    "distribution", "ranking"],
    "output.ordinality": ["nominal", "ordinal", "cardinal"],
    "latent.recurrence": ["markov", "chaotic", "grammar-depth", "none"],
    "latent.stationarity": ["stationary", "non-stationary"],
    "latent.noise_regime": ["low", "medium", "high"],
    "latent.label_cardinality": ["2", "3-10", "11-100", "101-1000", "1001+"],
    "latent.sample_regime": ["tiny", "small", "medium", "large"],
    "coupling": ["single-task", "mixture", "curriculum-position"],
}

# Continuous axes and their registered bins (inclusive upper bounds).
# Values quantise into the bin label of the first threshold they are
# less than or equal to.
_CONTINUOUS_BINS: dict[str, list[tuple[float, str]]] = {
    "latent.label_cardinality": [(2.0, "2"), (10.0, "3-10"),
                                 (100.0, "11-100"), (1000.0, "101-1000")],
}


class NormalisedDescriptor:
    """A normalised, canonical task descriptor plus its OOV event log."""

    def __init__(self, axes: dict[str, str], events: list[dict[str, str]]):
        self.axes = axes
        self.events = events

    def canonical(self) -> str:
        """Deterministic canonical serialisation (axis order fixed, no
        alphabetic re-sort — the axis order is part of the schema)."""
        ordered = {axis: self.axes.get(axis, FALLBACK_TOKEN)
                   for axis in AXES}
        return json.dumps(ordered, sort_keys=False, ensure_ascii=True,
                          separators=(",", ":"))

    def hash(self) -> str:
        return payload_hash(self.canonical())

    def to_dict(self) -> dict[str, Any]:
        return {"axes": self.axes, "events": self.events,
                "canonical_hash": self.hash()}


def _quantise(axis: str, value: float) -> tuple[str, bool]:
    bins = _CONTINUOUS_BINS.get(axis, [])
    for upper, label in bins:
        if value <= upper:
            return label, False
    # above the last closed bin: the registered open bin (the final
    # vocabulary token); logged as out-of-range
    return AXES[axis][-1], True


def normalise(raw: dict[str, Any]) -> NormalisedDescriptor:
    """Normalise a raw task description into the canonical descriptor.

    Rules (v24 section 3): categorical attributes must be registered
    tokens (else the OOV fallback, logged); the continuous axes
    (label cardinality) quantise into registered bins; a value above
    the top bin maps to the top bin and is logged. Unknown axes are
    ignored, not refused (I4).
    """
    axes: dict[str, str] = {}
    events: list[dict[str, str]] = []
    for axis in AXES:
        value = raw.get(axis)
        if axis in _CONTINUOUS_BINS and isinstance(value, (int, float)):
            token, out_of_range = _quantise(axis, float(value))
            axes[axis] = token
            if out_of_range:
                events.append({"axis": axis, "kind": "above-top-bin",
                               "value": str(value), "token": token})
            continue
        token = str(value) if value is not None else FALLBACK_TOKEN
        if token not in AXES[axis]:
            events.append({"axis": axis, "kind": "oov",
                           "value": token, "token": FALLBACK_TOKEN})
            token = FALLBACK_TOKEN
        axes[axis] = token
    return NormalisedDescriptor(axes, events)
