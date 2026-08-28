"""Programmatic primitives for the GEODE hybrid router (engineering plan B1).

A :class:`ProgrammaticPrimitive` wraps a plain Python/C function (a math
transform, a known computation, a contract check) behind the SAME fingerprint
interface as a learned primitive (see ``model_fingerprint.ModelFingerprint``).
The router can therefore treat programmatic and learned primitives uniformly:
it matches the fingerprint, checks the typed contract, and dispatches to the
cheapest primitive whose contract accepts the input.

Registered design (``analysis/ENGINEERING_PLAN_v20.md``, B1; literature
technique: typed tool schemas / protocol-agnostic integration, M129-D3):

- Programmatic primitives carry ZERO learned parameters: a pure function plus
  a typed contract. Cost is reported per primitive so the router can order.
- The contract is checked BEFORE any computation, so an out-of-contract input
  is rejected without a single FLOP of the primitive's own work — and, in the
  router, without a learned forward pass (the footprint/energy story).
- ``predict`` validates the output against the primitive's ``OutputSpec``, so
  a broken primitive is caught at the boundary, not downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from src.model_fingerprint import InputSpec, ModelFingerprint, OutputSpec


class OutOfContractError(ValueError):
    """Raised when an input does not satisfy a primitive's typed contract.

    Distinct from a numerical failure: an out-of-contract input is a *routing*
    decision (reject cheaply, or fall back), not a computation error.
    """


# Cost classes are ordered so the router can pick the cheapest contract-
# accepting primitive. "learned" is reserved for learned-model fallbacks.
COST_ORDER: Mapping[str, int] = {
    "constant": 0,
    "log": 1,
    "linear": 2,
    "quadratic": 3,
    "cubic": 4,
    "learned": 5,
}


@dataclass(frozen=True)
class PrimitiveContract:
    """Typed input contract for a programmatic primitive.

    Every field is a *guard*; ``None`` (or empty) means unconstrained. A single
    ``accepts`` method evaluates all guards so the router gets one verdict.
    """

    ndim: int | None = None
    #: Per-axis sizes; None in a position is a wildcard. e.g. (None, 1280)
    shape: tuple[int | None, ...] | None = None
    dtype: str | None = None
    require_finite: bool = True
    #: Inclusive value range (low, high); either bound may be None.
    value_range: tuple[float | None, float | None] | None = None
    #: Named domain checks: "unit_ball", "unit_simplex", "nonnegative".
    domain: str | None = None
    #: When True, ``accepts`` returns the NEGATION of the underlying guards.
    #: Lets a primitive declare "my contract is the out-of-contract set" —
    #: a cheap reject gate that the router dispatches to instead of a learned
    #: model (B3 measurement: zero learned forward passes on out-of-contract).
    negate: bool = False

    def accepts(self, array: np.ndarray) -> tuple[bool, str]:
        """Return ``(accepted, reason)`` for *array* under this contract."""
        ok, reason = self._evaluate(array)
        if self.negate:
            return (not ok, reason + (" (negated)" if not ok else " (negated: in contract)"))
        return ok, reason

    def _evaluate(self, array: np.ndarray) -> tuple[bool, str]:
        values = np.asarray(array)
        if self.ndim is not None and values.ndim != self.ndim:
            return False, f"ndim {values.ndim} != {self.ndim}"
        if self.shape is not None:
            if values.ndim != len(self.shape):
                return False, f"ndim {values.ndim} != shape spec {len(self.shape)}"
            for axis, (size, spec) in enumerate(zip(values.shape, self.shape)):
                if spec is not None and size != spec:
                    return False, f"axis {axis} size {size} != {spec}"
        if self.dtype is not None:
            if np.dtype(values.dtype).name != self.dtype:
                return False, f"dtype {values.dtype} != {self.dtype}"
        if self.require_finite and not np.all(np.isfinite(values)):
            return False, "contains non-finite values"
        if self.value_range is not None:
            low, high = self.value_range
            if low is not None and float(np.min(values)) < low:
                return False, f"value below lower bound {low}"
            if high is not None and float(np.max(values)) > high:
                return False, f"value above upper bound {high}"
        if self.domain is not None:
            ok, reason = _check_domain(values, self.domain)
            if not ok:
                return False, reason
        return True, "accepted"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "ndim": self.ndim,
            "shape": None if self.shape is None else list(self.shape),
            "dtype": self.dtype,
            "require_finite": self.require_finite,
            "value_range": (
                None if self.value_range is None else list(self.value_range)
            ),
            "domain": self.domain,
            "negate": self.negate,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PrimitiveContract":
        required = {
            "schema_version", "ndim", "shape", "dtype", "require_finite",
            "value_range", "domain", "negate",
        }
        if set(payload) != required or payload.get("schema_version") != 1:
            raise ValueError("unsupported primitive contract schema")
        shape = None if payload["shape"] is None else tuple(payload["shape"])
        value_range = (
            None if payload["value_range"] is None else tuple(payload["value_range"])
        )
        return cls(
            ndim=payload["ndim"],
            shape=shape,
            dtype=payload["dtype"],
            require_finite=bool(payload["require_finite"]),
            value_range=value_range,
            domain=payload["domain"],
            negate=bool(payload.get("negate", False)),
        )


def _check_domain(values: np.ndarray, domain: str) -> tuple[bool, str]:
    """Domain guards treat a batch as (N, d): each ROW must satisfy the guard.

    A contract that held per-sample (unit ball, unit simplex) silently fails a
    batch if the guard is computed over the whole array — a 2x3 simplex batch
    sums to 2.0, not 1.0, and a 2x3 unit-ball batch has a norm that is not a
    per-sample verdict. All guards are therefore row-wise.
    """
    batch = values.reshape(values.shape[0], -1) if values.ndim > 1 else values[None, :]
    if domain == "unit_ball":
        norms = np.linalg.norm(batch, axis=1)
        if np.all(norms <= 1.0 + 1e-9):
            return True, "accepted"
        return False, f"max row norm {float(np.max(norms))} > 1"
    if domain == "unit_simplex":
        if np.any(batch < 0.0):
            return False, "negative entry on simplex"
        sums = np.sum(batch, axis=1)
        if np.all(np.isclose(sums, 1.0, atol=1e-9)):
            return True, "accepted"
        return False, f"row sums deviate from 1 (max |dev| {float(np.max(np.abs(sums - 1.0)))})"
    if domain == "nonnegative":
        return (True, "accepted") if np.all(values >= 0.0) else (False, "negative entry")
    raise ValueError(f"unsupported domain guard {domain!r}")


@dataclass(frozen=True)
class ProgrammaticPrimitive:
    """A zero-parameter programmatic module with a fingerprint + contract.

    Parameters
    ----------
    fingerprint:
        The same ``ModelFingerprint`` contract a learned primitive declares:
        task name, input spec (source/dim), output spec (type/classes). This is
        what makes the primitive swappable with a learned one in the router.
    fn:
        The plain function ``fn(array) -> array`` implementing the primitive.
        Must be pure and side-effect free (a numpy/C kernel).
    contract:
        Typed input contract evaluated before any computation.
    cost_class:
        One of ``COST_ORDER`` keys; the router orders dispatch by it.
    description:
        Human-readable note (e.g. "L2 normalise the code block").
    """

    fingerprint: ModelFingerprint
    fn: Callable[[np.ndarray], np.ndarray]
    contract: PrimitiveContract
    cost_class: str = "constant"
    description: str = ""
    _fn_repr: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        if self.cost_class not in COST_ORDER:
            raise ValueError(f"unsupported cost_class {self.cost_class!r}")
        if not callable(self.fn):
            raise ValueError("fn must be callable")
        object.__setattr__(self, "_fn_repr", repr(self.fn))

    @property
    def task_name(self) -> str:
        return self.fingerprint.task_name

    def accepts(self, array: np.ndarray) -> tuple[bool, str]:
        """Contract check only — no computation performed."""
        return self.contract.accepts(array)

    def predict(self, array: np.ndarray) -> np.ndarray:
        """Contract-gated execution.

        Raises :class:`OutOfContractError` before any computation when the
        contract is not satisfied, and validates the output shape against the
        output spec after execution.
        """
        accepted, reason = self.accepts(array)
        if not accepted:
            raise OutOfContractError(
                f"primitive {self.task_name} rejected input: {reason}"
            )
        result = np.asarray(self.fn(np.asarray(array)))
        self._validate_output(result)
        return result

    def _validate_output(self, result: np.ndarray) -> None:
        output_type = self.fingerprint.output_spec.type
        n_classes = self.fingerprint.output_spec.dim
        if output_type == "labels":
            if result.ndim != 1:
                raise OutOfContractError(
                    f"primitive {self.task_name} emitted ndim {result.ndim}, "
                    "expected (N,) labels"
                )
        else:  # sdf_scores / probabilities
            if result.ndim != 2 or result.shape[1] != n_classes:
                raise OutOfContractError(
                    f"primitive {self.task_name} emitted shape {result.shape}, "
                    f"expected (N, {n_classes})"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "fingerprint": self.fingerprint.to_dict(),
            "fn_repr": self._fn_repr,
            "contract": self.contract.to_dict(),
            "cost_class": self.cost_class,
            "description": self.description,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        fn: Callable[[np.ndarray], np.ndarray],
    ) -> "ProgrammaticPrimitive":
        required = {
            "schema_version", "fingerprint", "fn_repr", "contract",
            "cost_class", "description",
        }
        if set(payload) != required or payload.get("schema_version") != 1:
            raise ValueError("unsupported programmatic primitive schema")
        return cls(
            fingerprint=ModelFingerprint.from_dict(payload["fingerprint"]),
            fn=fn,
            contract=PrimitiveContract.from_dict(payload["contract"]),
            cost_class=payload["cost_class"],
            description=payload["description"],
        )

    @staticmethod
    def check(array: np.ndarray, low: float = 0.0, high: float = 1.0) -> np.ndarray:
        """Example programmatic primitive: a pure range/contract check kernel.

        Emits one column per sample: 1 if every entry lies in ``[low, high]``
        and is finite, else 0. Demonstrates a *programmatic* primitive that
        does real work (validation) with zero learned parameters.
        """
        values = np.asarray(array)
        flat = values.reshape(values.shape[0], -1)
        ok = np.all(np.isfinite(flat), axis=1) & np.all(
            (flat >= low) & (flat <= high), axis=1
        )
        return ok.astype(np.int64)
