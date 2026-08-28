"""Deterministic primitive capability providers for the GEODE network.

Primitives are non-ML, code-defined transformations (math, logic, signal
processing, etc.) that are first-class citizens in the ModelNetwork and
Orchestrator.  They implement the same duck-type interface as FittedModel
so they can occupy any node position in the DAG:

  raw_features  →  [Primitive: l2_normalize]  →  [FittedModel: cat_detector]
  [FittedModel: depth_estimator]  →  [Primitive: threshold(0.5)]  →  [FittedModel: near_object]

Each Primitive carries a PrimitiveSpec that describes its operation, input
and output dimensionality, named parameters, and semantic category.  This
metadata is exposed to the Orchestrator and SemanticRouter so primitives
appear alongside ML models in capability reports and semantic routing.

Built-in factory functions
--------------------------
make_scale(factor)           element-wise multiply by a scalar
make_l2_normalize()          normalise each row to unit L2 length
make_threshold(value)        binarise: 1.0 where X > value, else 0.0
make_clip(low, high)         clip values to [low, high]
make_affine(A, b)            affine transform  Y = X @ A + b
make_select_dims(indices)    keep only the specified column indices
make_logical_and()           column-wise AND on boolean/0-1 arrays
make_logical_or()            column-wise OR  on boolean/0-1 arrays
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from src.model_fingerprint import InputSpec, ModelFingerprint, OutputSpec


# ---------------------------------------------------------------------------
# PrimitiveSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrimitiveSpec:
    """Static description of a primitive operation.

    Attributes
    ----------
    name:
        Short operation identifier, e.g. ``"multiply"``, ``"threshold"``.
    category:
        Broad semantic category for routing and reporting.
        Standard values: ``"math"``, ``"logic"``, ``"signal"``,
        ``"transform"``, ``"custom"``.
    input_dim:
        Expected number of features per sample (columns in X).
        ``-1`` means the primitive accepts any width.
    output_dim:
        Number of features produced per sample.
        ``-1`` means output width equals input width.
    input_dtype:
        NumPy dtype string the primitive expects (e.g. ``"float64"``).
    output_dtype:
        NumPy dtype string the primitive produces.
    params:
        Named parameters that define the operation, e.g.
        ``{"factor": 2.0}`` for a scale primitive.
    description:
        Human-readable explanation of what this primitive does.
    """

    name: str
    category: str
    input_dim: int = -1
    output_dim: int = -1
    input_dtype: str = "float64"
    output_dtype: str = "float64"
    params: dict = field(default_factory=dict)
    description: str = ""

    def resolved_output_dim(self, actual_input_dim: int) -> int:
        """Return the concrete output width given the actual input width."""
        return actual_input_dim if self.output_dim == -1 else self.output_dim

    def __str__(self) -> str:
        i = str(self.input_dim) if self.input_dim != -1 else "*"
        o = str(self.output_dim) if self.output_dim != -1 else "="
        p = ", ".join(f"{k}={v}" for k, v in self.params.items())
        base = f"{self.category}.{self.name}({i}→{o})"
        return f"{base}[{p}]" if p else base


# ---------------------------------------------------------------------------
# Primitive
# ---------------------------------------------------------------------------


class Primitive:
    """A deterministic, non-ML capability provider.

    Implements the same duck-type interface as :class:`~src.model_network.FittedModel`
    so it can be placed in any :class:`~src.model_network.ModelNetwork` node
    without special-casing.

    Parameters
    ----------
    spec:
        Static description of this operation.
    fn:
        Callable ``(X: np.ndarray) -> np.ndarray``.  Receives the node's
        input array and returns the transformed output.  Must be pure
        (no side-effects) and deterministic.
    """

    def __init__(
        self,
        spec: PrimitiveSpec,
        fn: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        self.spec = spec
        self._fn = fn
        self._fingerprint = _make_fingerprint(spec)

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the primitive operation to *X* and return the result."""
        out = self._fn(np.asarray(X, dtype=self.spec.input_dtype))
        return np.asarray(out, dtype=self.spec.output_dtype)

    # ------------------------------------------------------------------
    # Duck-type interface for ModelNetwork / Orchestrator
    # ModelNetwork calls sdf_scores() and _predict_from_scores() on every node.
    # Primitives route both through transform() — the output is a generic
    # feature array, not SDF values, but the network treats it the same way.
    # ------------------------------------------------------------------

    def sdf_scores(self, X: np.ndarray) -> np.ndarray:
        """Return ``transform(X)`` — satisfies the ModelNetwork node interface."""
        return self.transform(X)

    def _predict_from_scores(self, scores: np.ndarray) -> np.ndarray:
        """Pass transformed values through unchanged (primitives don't classify)."""
        return scores

    def is_swappable_with(self, other: object) -> bool:
        """True when *other* is a Primitive with the same name and category."""
        if not isinstance(other, Primitive):
            return False
        return (
            self.spec.name == other.spec.name
            and self.spec.category == other.spec.category
        )

    @property
    def fingerprint(self) -> ModelFingerprint:
        """ModelFingerprint describing this primitive's IO contract."""
        return self._fingerprint

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Primitive({self.spec})"

    def __str__(self) -> str:
        return str(self.spec)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_fingerprint(spec: PrimitiveSpec) -> ModelFingerprint:
    """Build a ModelFingerprint for a primitive.

    source = ``"primitive"`` signals to ModelNetwork.validate() that this
    node is a flexible transformer (not bound to the sdf_scores convention).
    output type = ``"transform"`` distinguishes primitive outputs from
    classification ``"sdf_scores"`` or ``"labels"``.
    """
    return ModelFingerprint(
        task_name=f"{spec.category}.{spec.name}",
        input_spec=InputSpec(source="primitive", dim=spec.input_dim),
        output_spec=OutputSpec(type="transform", classes=(spec.name,)),
    )


# ---------------------------------------------------------------------------
# Built-in factory functions
# ---------------------------------------------------------------------------


def make_scale(factor: float) -> Primitive:
    """Element-wise multiply every feature by *factor*.

    (N, d) → (N, d)
    """
    return Primitive(
        PrimitiveSpec(
            name="scale",
            category="math",
            params={"factor": factor},
            description=f"Multiply every element by {factor}.",
        ),
        fn=lambda X: X * factor,
    )


def make_l2_normalize(eps: float = 1e-12) -> Primitive:
    """Normalise each row to unit L2 length.

    (N, d) → (N, d)  with  ‖row‖₂ = 1
    """
    def _fn(X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(norms, eps)

    return Primitive(
        PrimitiveSpec(
            name="l2_normalize",
            category="transform",
            description="Normalise each sample to unit L2 length.",
        ),
        fn=_fn,
    )


def make_threshold(value: float) -> Primitive:
    """Binarise: 1.0 where X > *value*, 0.0 elsewhere.

    (N, d) → (N, d)  dtype float64, values in {0, 1}
    """
    return Primitive(
        PrimitiveSpec(
            name="threshold",
            category="logic",
            params={"value": value},
            description=f"Return 1.0 where X > {value}, else 0.0.",
        ),
        fn=lambda X: (X > value).astype(np.float64),
    )


def make_clip(low: float, high: float) -> Primitive:
    """Clip all values into [*low*, *high*].

    (N, d) → (N, d)
    """
    return Primitive(
        PrimitiveSpec(
            name="clip",
            category="math",
            params={"low": low, "high": high},
            description=f"Clip values to [{low}, {high}].",
        ),
        fn=lambda X: np.clip(X, low, high),
    )


def make_affine(A: np.ndarray, b: np.ndarray | None = None) -> Primitive:
    """Affine projection  Y = X @ A + b.

    (N, d_in) → (N, d_out)   where  A.shape == (d_in, d_out)
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.zeros(A.shape[1], dtype=np.float64) if b is None else np.asarray(b, dtype=np.float64)

    def _fn(X: np.ndarray) -> np.ndarray:
        return X @ A + b

    return Primitive(
        PrimitiveSpec(
            name="affine",
            category="transform",
            input_dim=A.shape[0],
            output_dim=A.shape[1],
            params={"in": A.shape[0], "out": A.shape[1]},
            description=f"Affine projection {A.shape[0]}→{A.shape[1]}.",
        ),
        fn=_fn,
    )


def make_select_dims(indices: list[int]) -> Primitive:
    """Keep only the columns at *indices*.

    (N, d) → (N, len(indices))
    """
    idx = list(indices)
    return Primitive(
        PrimitiveSpec(
            name="select_dims",
            category="transform",
            output_dim=len(idx),
            params={"indices": idx},
            description=f"Keep {len(idx)} selected feature dimensions.",
        ),
        fn=lambda X: X[:, idx],
    )


def make_logical_and() -> Primitive:
    """Column-wise AND: output[i] = 1.0 iff all columns of row i are > 0.

    (N, d) → (N, 1)
    """
    return Primitive(
        PrimitiveSpec(
            name="logical_and",
            category="logic",
            output_dim=1,
            description="1.0 for rows where ALL columns are positive.",
        ),
        fn=lambda X: np.all(X > 0, axis=1, keepdims=True).astype(np.float64),
    )


def make_logical_or() -> Primitive:
    """Column-wise OR: output[i] = 1.0 iff any column of row i is > 0.

    (N, d) → (N, 1)
    """
    return Primitive(
        PrimitiveSpec(
            name="logical_or",
            category="logic",
            output_dim=1,
            description="1.0 for rows where ANY column is positive.",
        ),
        fn=lambda X: np.any(X > 0, axis=1, keepdims=True).astype(np.float64),
    )


# ---------------------------------------------------------------------------
# DelayPrimitive  — stateful ring-buffer for recurrent feedback loops
# ---------------------------------------------------------------------------


class DelayPrimitive(Primitive):
    """Stateful ring-buffer that closes recurrent feedback loops.

    Each call to :meth:`transform` does two things in order:

    1. **Return** the last *k* inputs concatenated (oldest-first), zero-padded
       before the buffer fills.
    2. **Push** the current input into the ring-buffer and discard the oldest
       entry if the buffer exceeds *k* entries.

    This means the output at step *t* is built from steps *t−k* … *t−1*,
    not from step *t* itself — exactly what is needed to feed previous context
    back into the same node without information leakage.

    Usage pattern (step-by-step inference)::

        delay = make_delay(k=3)
        for t, x_t in enumerate(sequence):
            ctx_t = delay.step(x_t)         # (k*d,) — history before t
            feat_t = np.concatenate([x_t, ctx_t])
            pred_t = model.predict(feat_t)

    Notes
    -----
    * Designed for **single-sample** (N=1) step-by-step use.  Batch mode
      (N>1) is supported but treats all N rows as the *same* time-step, so
      only the first row is stored in the ring-buffer.
    * Call :meth:`reset` at sequence boundaries to clear the buffer.
    * The underlying ``_fn`` passed to the parent is never called; the full
      logic lives in the overridden :meth:`transform`.
    """

    def __init__(
        self,
        spec: PrimitiveSpec,
        fn: Callable[[np.ndarray], np.ndarray],
        k: int,
        d: int,
        fill_value: float = 0.0,
    ) -> None:
        super().__init__(spec, fn)
        self._k          = k
        self._d          = d            # -1 = infer on first call
        self._fill       = fill_value
        self._buffer: list[np.ndarray] = []   # list of (d,) arrays

    # ------------------------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return history of last *k* inputs, then store current input.

        Parameters
        ----------
        X : (N, d) — current input (N=1 recommended).

        Returns
        -------
        (N, k * d) — last *k* inputs concatenated (oldest first).
        """
        X = np.asarray(X, dtype=self.spec.input_dtype)
        N, d = X.shape

        if self._d == -1:
            self._d = d     # infer dimensionality on first call

        k   = self._k
        buf = self._buffer      # list of (d,) arrays

        # Build context from ring-buffer (pad with zeros when not full yet)
        n_have    = len(buf)
        n_missing = max(0, k - n_have)
        parts     = (
            [np.full(self._d, self._fill, dtype=np.float64)] * n_missing
            + buf[-k:]
        )
        context_1d = np.concatenate(parts)          # (k * d,)
        out        = np.broadcast_to(context_1d, (N, len(context_1d))).copy()

        # Update ring-buffer with current input (use first row for N>1)
        self._buffer.append(X[0].copy())
        if len(self._buffer) > k:
            self._buffer.pop(0)

        return out.astype(self.spec.output_dtype)

    def reset(self) -> None:
        """Clear the ring-buffer (call at sequence boundaries)."""
        self._buffer.clear()

    def step(self, x: np.ndarray) -> np.ndarray:
        """Convenience wrapper: process a single vector (d,) → (k*d,)."""
        return self.transform(x.reshape(1, -1)).ravel()

    def __repr__(self) -> str:
        return f"DelayPrimitive(k={self._k}, d={self._d}, buf={len(self._buffer)})"


# ---------------------------------------------------------------------------
# make_delay factory
# ---------------------------------------------------------------------------


def make_delay(
    k: int = 1,
    input_dim: int = -1,
    fill_value: float = 0.0,
) -> DelayPrimitive:
    """Create a stateful delay (ring-buffer) primitive.

    Each call to ``transform(X)`` returns the last *k* inputs concatenated
    (oldest first), then stores the current input.  The first *k* calls
    return zero-padded history.

    Parameters
    ----------
    k :
        Number of previous steps to retain and concatenate.
    input_dim :
        Expected input width per sample.  Use ``-1`` to infer from the first
        call to ``transform``.
    fill_value :
        Value used for padding before the buffer has *k* entries.

    Returns
    -------
    :class:`DelayPrimitive`  with ``output_dim = k * input_dim``
    (or ``-1`` when *input_dim* is unknown).

    Example
    -------
    >>> delay = make_delay(k=2, input_dim=4)
    >>> delay.step(np.array([1, 2, 3, 4]))   # returns zeros (buffer empty)
    array([0., 0., 0., 0., 0., 0., 0., 0.])
    >>> delay.step(np.array([5, 6, 7, 8]))   # returns step-0 padded with zeros
    array([0., 0., 0., 0., 1., 2., 3., 4.])
    >>> delay.step(np.array([9, 10, 11, 12]))
    array([1., 2., 3., 4., 5., 6., 7., 8.])
    """
    out_dim = k * input_dim if input_dim > 0 else -1
    spec = PrimitiveSpec(
        name="delay",
        category="signal",
        input_dim=input_dim,
        output_dim=out_dim,
        params={"k": k, "fill_value": fill_value},
        description=f"Ring-buffer delay: concatenate last {k} step(s).",
    )
    return DelayPrimitive(
        spec,
        fn=lambda X: X,     # overridden by DelayPrimitive.transform()
        k=k,
        d=input_dim,
        fill_value=fill_value,
    )
