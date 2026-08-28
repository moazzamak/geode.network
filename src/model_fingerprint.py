"""Model fingerprinting for GEODE network composition.

A ModelFingerprint encodes:
  - what task a model is trained for          (task_name)
  - what input it expects                     (InputSpec: source, upstream tasks, dim)
  - what output it produces                   (OutputSpec: type, class labels)

Two models with identical fingerprints are interchangeable — you can swap one
bird-detector for another bird-detector without touching the rest of the network.

Compatibility is checked separately from identity: a model *accepts* upstream
output when the output type matches the expected source and (optionally) the
upstream task name is in the allow-list.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping


# ---------------------------------------------------------------------------
# Input / Output Specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InputSpec:
    """Describes the data format a model expects to receive.

    Attributes
    ----------
    source:
        ``"raw_cnn"``   — 1280-dim MobileNetV2 embeddings fed directly.
        ``"raw_hog"``   — HOG feature vector fed directly.
        ``"sdf_scores"``— column-concatenated normalised SDF score matrices
                          from one or more upstream nodes.  The limited SDF
                          range ( < 0 = inside, ≈ 0 = surface, > 0 = outside)
                          makes these geometrically meaningful features for the
                          next layer of ellipsoid fitting.
        ``"passthrough"``— arbitrary pre-processed array; no transform applied.
    upstream_tasks:
        If non-empty, only outputs from nodes whose ``task_name`` appears here
        are accepted.  Use to enforce that a bird-on-car model only wires to
        ``("bird_detector", "car_detector")``, not arbitrary upstream nodes.
    dim:
        Expected feature dimensionality. ``-1`` means unconstrained.
    """

    source: str
    upstream_tasks: tuple[str, ...] = ()
    dim: int = -1

    def __str__(self) -> str:
        up = ",".join(self.upstream_tasks) if self.upstream_tasks else "*"
        dim_s = f"[{self.dim}d]" if self.dim > 0 else ""
        return f"{self.source}{dim_s}←{up}"


@dataclass(frozen=True)
class OutputSpec:
    """Describes the data a model produces.

    Attributes
    ----------
    type:
        ``"labels"``     — integer class label per sample.
        ``"sdf_scores"`` — (N, n_classes) normalised SDF score matrix.
        ``"probabilities"`` — (N, n_classes) calibrated probability matrix.
    classes:
        Ordered tuple of class identifiers (int IDs or string names).  The
        order matches column order in score matrices.
    """

    type: str
    classes: tuple

    @property
    def dim(self) -> int:
        return len(self.classes)

    def __str__(self) -> str:
        return f"{self.type}[{self.dim}]"


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelFingerprint:
    """Complete identity description of a GEODE model.

    Parameters
    ----------
    task_name:
        Human-readable task identifier, e.g. ``"bird_detector"`` or
        ``"cifar10_10class"``.  Swappability is keyed on this name.
    input_spec:
        What the model expects as input.
    output_spec:
        What the model produces.
    alpha:
        SoftMin sharpness used when this model was trained.
    pca_components:
        Intermediate PCA dimension before LDA (ignored for SDF-score inputs).
    """

    task_name: str
    input_spec: InputSpec
    output_spec: OutputSpec
    alpha: float = 2.0
    pca_components: int = 128

    # ------------------------------------------------------------------
    # Identity & compatibility
    # ------------------------------------------------------------------

    @property
    def signature(self) -> str:
        """12-character SHA-256 prefix uniquely identifying the *role* of this
        model (task + io contract).  Two swappable models share the same
        signature."""
        payload = json.dumps(
            {
                "task": self.task_name,
                "input_source": self.input_spec.source,
                "upstream_tasks": sorted(self.input_spec.upstream_tasks),
                "output_type": self.output_spec.type,
                "classes": sorted(str(c) for c in self.output_spec.classes),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def is_swappable_with(self, other: "ModelFingerprint") -> bool:
        """True when *other* serves the same role and can replace this model.

        Two models are swappable if they share the same task name, expect the
        same input source, and produce the same output type over the same set
        of classes.  Internal hyperparameters (alpha, pca_components) are
        **not** part of swappability — only the IO contract matters.
        """
        return (
            self.task_name == other.task_name
            and self.input_spec.source == other.input_spec.source
            and self.output_spec.type == other.output_spec.type
            and frozenset(self.output_spec.classes) == frozenset(other.output_spec.classes)
        )

    def accepts_from(self, upstream: "ModelFingerprint") -> bool:
        """True when this model can wire to *upstream* as a data source.

        Rules:
        - An ``"sdf_scores"`` input accepts any upstream that outputs
          ``"sdf_scores"``.
        - If ``upstream_tasks`` is non-empty the upstream task_name must be
          listed there.
        - Raw-input models (``"raw_cnn"`` / ``"raw_hog"`` / ``"passthrough"``)
          have no upstream and this method should not be called for them.
        """
        if self.input_spec.source == "sdf_scores":
            if upstream.output_spec.type != "sdf_scores":
                return False
        if self.input_spec.upstream_tasks:
            if upstream.task_name not in self.input_spec.upstream_tasks:
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "task_name": self.task_name,
            "input_spec": {
                "source": self.input_spec.source,
                "upstream_tasks": list(self.input_spec.upstream_tasks),
                "dim": self.input_spec.dim,
            },
            "output_spec": {
                "type": self.output_spec.type,
                "classes": list(self.output_spec.classes),
            },
            "alpha": self.alpha,
            "pca_components": self.pca_components,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModelFingerprint":
        required = {
            "schema_version", "task_name", "input_spec", "output_spec",
            "alpha", "pca_components",
        }
        if set(payload) != required or payload.get("schema_version") != 1:
            raise ValueError("unsupported model fingerprint schema")
        input_payload = payload["input_spec"]
        output_payload = payload["output_spec"]
        if not isinstance(input_payload, Mapping) or set(input_payload) != {
            "source", "upstream_tasks", "dim",
        }:
            raise ValueError("invalid model fingerprint input_spec")
        if not isinstance(output_payload, Mapping) or set(output_payload) != {
            "type", "classes",
        }:
            raise ValueError("invalid model fingerprint output_spec")
        return cls(
            task_name=str(payload["task_name"]),
            input_spec=InputSpec(
                source=str(input_payload["source"]),
                upstream_tasks=tuple(str(value) for value in input_payload["upstream_tasks"]),
                dim=int(input_payload["dim"]),
            ),
            output_spec=OutputSpec(
                type=str(output_payload["type"]),
                classes=tuple(output_payload["classes"]),
            ),
            alpha=float(payload["alpha"]),
            pca_components=int(payload["pca_components"]),
        )

    def __str__(self) -> str:
        return (
            f"ModelFingerprint(task={self.task_name!r}, "
            f"in={self.input_spec}, "
            f"out={self.output_spec}, "
            f"sig={self.signature})"
        )
