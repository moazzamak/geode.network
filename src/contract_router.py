"""Contract-gated routing for the GEODE hybrid system (engineering plan B2).

A :class:`ContractGatedRouter` owns a registry of primitives — programmatic
first-class, learned models as an optional fallback — keyed by fingerprint.
For each input it:

1. matches the requested task / output-type fingerprint,
2. selects the CHEAPEST registered primitive whose typed contract accepts the
   input (programmatic primitives are checked before any computation),
3. dispatches and validates the output against the output spec,
4. records a decision log.

Out-of-contract inputs never reach a learned model unless ``enable_fallback``
is set (default True, but the router reports the fallback flag and cost so the
footprint/energy effect is measurable).

Registered design (``analysis/ENGINEERING_PLAN_v20.md``, B2; literature
technique: rule/retrieval-based dispatch, no LLM at inference — M129-D4/M127):

- The decision log is the planner hook: a future HTN-style planner (v19 10)
  can drive the same registry and record its plan in the same log shape.
- The router is rule-based (closed goals): fingerprint + contract match. An
  LLM is never invoked at inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.model_fingerprint import ModelFingerprint
from src.programmatic_primitive import COST_ORDER, ProgrammaticPrimitive


@dataclass(frozen=True)
class RouteResult:
    """Outcome of routing one input (or batch) through the registry."""

    predictions: np.ndarray | None
    primitive_id: str | None
    task_name: str | None
    cost_class: str | None
    accepted: bool
    rejected: bool
    fallback: bool
    reason: str
    decision_log: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "predictions": (
                None if self.predictions is None else self.predictions.tolist()
            ),
            "primitive_id": self.primitive_id,
            "task_name": self.task_name,
            "cost_class": self.cost_class,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "fallback": self.fallback,
            "reason": self.reason,
            "decision_log": list(self.decision_log),
        }


def _primitive_id(primitive: ProgrammaticPrimitive) -> str:
    return f"{primitive.task_name}@{primitive.fingerprint.signature}"


class ContractGatedRouter:
    """Route inputs to the cheapest contract-accepting primitive.

    Parameters
    ----------
    programmatic:
        Registered programmatic primitives (zero learned parameters).
    fallback:
        Optional mapping ``task_name -> object`` for learned fallback. Each
        fallback must expose ``predict(array) -> np.ndarray`` and a
        ``fingerprint`` attribute (a :class:`ModelFingerprint`). Used only when
        no programmatic primitive accepts the input AND ``enable_fallback``.
    enable_fallback:
        Master switch for the learned fallback path.
    """

    def __init__(
        self,
        programmatic: Sequence[ProgrammaticPrimitive],
        fallback: Mapping[str, Any] | None = None,
        enable_fallback: bool = True,
    ) -> None:
        if not programmatic:
            raise ValueError("at least one programmatic primitive is required")
        self._primitives: dict[str, ProgrammaticPrimitive] = {}
        for primitive in programmatic:
            key = _primitive_id(primitive)
            if key in self._primitives:
                raise ValueError(f"duplicate primitive id {key}")
            self._primitives[key] = primitive
        self._fallback = dict(fallback or {})
        self._enable_fallback = bool(enable_fallback)

    @property
    def primitive_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._primitives))

    def route(
        self,
        array: np.ndarray,
        task_name: str | None = None,
        output_type: str | None = None,
    ) -> RouteResult:
        """Route one input (or one batch) to the cheapest accepting primitive."""
        log: list[dict[str, Any]] = []
        candidates = [
            (key, primitive)
            for key, primitive in self._primitives.items()
            if (task_name is None or primitive.task_name == task_name)
            and (
                output_type is None
                or primitive.fingerprint.output_spec.type == output_type
            )
        ]
        if not candidates:
            reason = (
                f"no primitive registered for task={task_name} output={output_type}"
            )
            return self._reject(reason, log)

        # Contract check is pure and cheap: evaluate every candidate, then
        # pick the cheapest that accepts.
        accepted: list[tuple[int, str, ProgrammaticPrimitive]] = []
        for key, primitive in candidates:
            ok, why = primitive.accepts(array)
            log.append(
                {
                    "primitive_id": key,
                    "cost_class": primitive.cost_class,
                    "contract_ok": ok,
                    "reason": why,
                }
            )
            if ok:
                accepted.append(
                    (COST_ORDER[primitive.cost_class], key, primitive)
                )

        if accepted:
            accepted.sort(key=lambda item: (item[0], item[1]))
            _, key, primitive = accepted[0]
            try:
                predictions = primitive.predict(array)
            except ValueError as error:  # OutOfContractError is a ValueError
                return self._reject(f"primitive {key} failed at dispatch: {error}", log)
            log.append(
                {
                    "primitive_id": key,
                    "dispatched": True,
                    "cost_class": primitive.cost_class,
                }
            )
            return RouteResult(
                predictions=predictions,
                primitive_id=key,
                task_name=primitive.task_name,
                cost_class=primitive.cost_class,
                accepted=True,
                rejected=False,
                fallback=False,
                reason="dispatched to cheapest contract-accepting primitive",
                decision_log=tuple(log),
            )

        # No programmatic primitive accepts the input: out of contract.
        reason = "no programmatic primitive accepts the input (out of contract)"
        if self._enable_fallback and task_name in self._fallback:
            fallback_model = self._fallback[task_name]
            try:
                predictions = np.asarray(fallback_model.predict(array))
            except Exception as error:  # noqa: BLE001 - fallback failures are recorded
                return self._reject(f"fallback failed for {task_name}: {error}", log)
            log.append({"fallback": task_name, "dispatched": True, "cost_class": "learned"})
            return RouteResult(
                predictions=predictions,
                primitive_id=f"fallback:{task_name}",
                task_name=task_name,
                cost_class="learned",
                accepted=False,
                rejected=False,
                fallback=True,
                reason=reason,
                decision_log=tuple(log),
            )
        return self._reject(reason, log)

    def _reject(self, reason: str, log: list[dict[str, Any]]) -> RouteResult:
        return RouteResult(
            predictions=None,
            primitive_id=None,
            task_name=None,
            cost_class=None,
            accepted=False,
            rejected=True,
            fallback=False,
            reason=reason,
            decision_log=tuple(log),
        )

    def route_batch(
        self,
        arrays: Sequence[np.ndarray],
        task_name: str | None = None,
        output_type: str | None = None,
    ) -> list[RouteResult]:
        """Route each array in *arrays* independently (contracts are per-input)."""
        return [
            self.route(array, task_name=task_name, output_type=output_type)
            for array in arrays
        ]
