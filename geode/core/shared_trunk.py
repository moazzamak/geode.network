"""GEODE shared-trunk program (v25 M280) — the pooling that exists.

Registered 22 Aug 2026 in
``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` (the M272-M281 wave).
The measured lesson: the only component that pools N data across the
arms is the SHARED frozen trunk; therefore:

- every new arm must REUSE a registered publisher trunk (or be a
  programmatic primitive, which needs no trunk);
- a NEW trunk is admitted only with a measured gap — evidence that
  the existing trunks cannot serve the task — never by preference;
- LoRA adapters on a shared trunk stay behind the §3.1 criterion
  (a measured gap the primitive tier cannot close).

Deterministic; no RNG, no wall clocks.
"""
from __future__ import annotations

from typing import Any


class TrunkRegistry:
    """Append-only registry of publisher trunks."""

    def __init__(self) -> None:
        self._trunks: dict[str, dict[str, Any]] = {}

    def register_trunk(self, trunk_id: str, license: str,
                       params_million: float) -> str:
        if trunk_id in self._trunks:
            raise ValueError(f"trunk {trunk_id!r} already registered")
        if not isinstance(license, str) or not license:
            raise ValueError("a trunk needs a recorded license")
        if params_million <= 0:
            raise ValueError("params_million must be positive")
        self._trunks[trunk_id] = {
            "trunk_id": trunk_id,
            "license": license,
            "params_million": float(params_million),
        }
        return trunk_id

    def trunks(self) -> dict[str, dict[str, Any]]:
        return {k: dict(v) for k, v in self._trunks.items()}

    def admit_trunk(self, trunk_id: str, license: str,
                    params_million: float,
                    gap_evidence: dict[str, Any] | None
                    ) -> dict[str, Any]:
        """A NEW trunk is admitted ONLY with measured gap evidence;
        re-using an existing trunk needs none (M280)."""
        if trunk_id in self._trunks:
            return {"admitted": True, "reason": "trunk_reuse",
                    "trunk": dict(self._trunks[trunk_id])}
        if not gap_evidence:
            return {"admitted": False,
                    "reason": ("new_trunk_without_measured_gap — "
                               "reuse a registered trunk or supply "
                               "gap evidence (M280)")}
        required = {"task", "measured_gap", "evidence_path"}
        if any(k not in gap_evidence for k in required):
            return {"admitted": False,
                    "reason": f"gap_evidence needs {sorted(required)}"}
        self.register_trunk(trunk_id, license, params_million)
        return {"admitted": True, "reason": "gap_measured",
                "trunk": dict(self._trunks[trunk_id])}


def validate_arm_trunk(arm_spec: dict[str, Any],
                       registry: TrunkRegistry) -> list[str]:
    """Every non-primitive arm must reference a REGISTERED trunk;
    primitives need none. Returns reasons ([] = ok)."""
    if arm_spec.get("primitive"):
        return []
    trunk = arm_spec.get("trunk_id")
    if not isinstance(trunk, str) or not trunk:
        return ["arm needs a trunk_id (M280: shared trunks only)"]
    if trunk not in registry.trunks():
        return [f"trunk {trunk!r} not registered (M280)"]
    return []
