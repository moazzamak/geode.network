"""M210 — arm adapters: any artifact (admitted DNN or sealed
closed-form head) becomes a validated, routable arm spec.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` section 6
(19 Aug 2026, before the build). The contract is SIZE-AGNOSTIC by
design: no field or gate depends on parameter counts, byte sizes, or
widths. Size is recorded as metadata for the ledger, never used as a
constraint.
"""
from __future__ import annotations

from typing import Any

from geode.core.dnn_admission import AdmissionResult, DNNSubmission
from geode.core.economics import address_of

# the fields geode.core.router.Router requires of every arm
ROUTER_FIELDS = ("arm_id", "fingerprint", "output_contract",
                 "held_out_accuracy", "availability", "price", "general",
                 "primitive")

# the whitepaper's unified registration fields (24 Aug): every
# registration carries an operator key, a payout address that may
# differ from it, a price per unit of work, and a sealed claim.
# Builders ALWAYS populate them; validation stays permissive for
# pre-existing specs.
REGISTRATION_FIELDS = ("operator_key", "payout_address", "sealed_claim")

# size metadata carried by every arm (informational only — the
# contracts impose NO bound on it)
SIZE_FIELDS = ("param_count", "size_bytes", "kind")

DEFAULT_OUTPUT_CONTRACT = {"kind": "class_scores_345"}

# M270 C6: per-artifact license recorded at registration (audit
# action 1). Empty string = not applicable. SPDX identifiers where
# one exists (https://spdx.org).
LICENSE_FIELDS = ("code", "weights", "data")


def empty_license() -> dict[str, str]:
    """The explicit not-applicable license object every builder
    defaults to (the field must always be present; see C6)."""
    return {"code": "", "weights": "", "data": ""}


def _license_ok(value: Any) -> bool:
    return (isinstance(value, dict)
            and set(value.keys()) == set(LICENSE_FIELDS)
            and all(isinstance(value[key], str) for key in LICENSE_FIELDS))


def _accuracy_ok(value: Any) -> bool:
    """Accuracy is a number in [0, 1], or a dict of per-task numbers
    (the M171 redundant-capability contract)."""
    if isinstance(value, dict):
        return all(isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0
                   for v in value.values())
    return isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0


def _validate_common(spec: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for field in ROUTER_FIELDS:
        if field not in spec:
            reasons.append(f"missing router field {field!r}")
    for field in SIZE_FIELDS:
        if field not in spec:
            reasons.append(f"missing size field {field!r}")
    for field in REGISTRATION_FIELDS:
        if field in spec and not isinstance(spec[field], str):
            reasons.append(f"{field} must be a string when present")
    if not _accuracy_ok(spec.get("held_out_accuracy")):
        reasons.append("held_out_accuracy must be in [0, 1] "
                       "(or a dict of per-task values in [0, 1])")
    if not isinstance(spec.get("price"), (int, float)) \
            or float(spec["price"]) < 0:
        reasons.append("price must be a non-negative number")
    if not isinstance(spec.get("availability"), dict):
        reasons.append("availability must be an object")
    if not _license_ok(spec.get("license")):
        reasons.append("license must be an object with string keys "
                       "code/weights/data (M270 C6)")
    return reasons


def validate_arm_spec(spec: dict[str, Any]) -> list[str]:
    """Gate: an arm spec is routable iff it has every router field,
    size metadata, and sane values. Size is NOT bounded — a 1B-param
    artifact and a 2k-param artifact validate identically."""
    reasons = _validate_common(spec)
    if spec.get("kind") == "dnn":
        if not isinstance(spec.get("replay_hash"), str) \
                or len(spec["replay_hash"]) != 64:
            reasons.append("dnn arms must carry a 64-hex replay_hash")
    if spec.get("kind") == "sealed_head":
        if not isinstance(spec.get("sealed_source"), str):
            reasons.append("sealed_head arms must carry sealed_source")
    return reasons


def _task_accuracies(overall: float,
                     per_task: dict[str, float] | None) -> dict[str, float]:
    """held_out_accuracy as the M171 per-task dict: every per-task
    value must be a sealed measurement for that task; the aggregate is
    carried under 'all'."""
    acc: dict[str, float] = {"all": float(overall)}
    for task, value in (per_task or {}).items():
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"task {task!r} accuracy out of range")
        acc[task] = float(value)
    return acc


def arm_from_admission(sub: DNNSubmission, result: AdmissionResult,
                       arm_id: str, held_out_accuracy: float,
                       availability: dict[str, Any] | None = None,
                       price: float = 0.0,
                       fingerprint: list[float] | None = None,
                       param_count: int = 0,
                       size_bytes: int = 0,
                       per_task: dict[str, float] | None = None,
                       license: dict[str, str] | None = None,
                       operator_key: str | None = None,
                       payout_address: str | None = None
                       ) -> dict[str, Any]:
    """An ADMITTED DNN submission becomes a routable arm. Raises on
    non-admission (a rejected artifact can never route). The unified
    registration fields (operator key, payout address, sealed claim)
    are always populated: defaults are the arm's derived address and
    the admission replay hash."""
    if not result.admitted:
        raise ValueError(f"arm {arm_id!r}: submission not admitted "
                         f"({'; '.join(result.reasons)})")
    license = dict(license or empty_license())
    if not _license_ok(license):
        raise ValueError("license must have string keys code/weights/data")
    operator_key = str(operator_key or address_of(arm_id))
    payout_address = str(payout_address or operator_key)
    return {
        "arm_id": arm_id,
        "kind": "dnn",
        "fingerprint": list(fingerprint or []),
        "output_contract": dict(DEFAULT_OUTPUT_CONTRACT),
        "held_out_accuracy": _task_accuracies(held_out_accuracy, per_task),
        "selection_accuracy": float(held_out_accuracy),
        "availability": dict(availability or {"healthy": True}),
        "price": float(price),
        "general": True,
        "primitive": False,
        "replay_hash": result.replay_hash,
        "architecture_hash": sub.architecture_hash,
        "weights_hash": sub.weights_hash,
        "data_digest": sub.data_digest,
        "license": dict(license),
        "param_count": int(param_count),
        "size_bytes": int(size_bytes),
        "operator_key": operator_key,
        "payout_address": payout_address,
        "sealed_claim": result.replay_hash,
    }


def arm_from_sealed_head(arm_id: str, family: str, width: int,
                         held_out_accuracy: float,
                         sealed_source: str,
                         availability: dict[str, Any] | None = None,
                         price: float = 0.0,
                         fingerprint: list[float] | None = None,
                         size_bytes: int = 0,
                         per_task: dict[str, float] | None = None,
                         license: dict[str, str] | None = None,
                         operator_key: str | None = None,
                         payout_address: str | None = None
                         ) -> dict[str, Any]:
    """A sealed closed-form head becomes a routable arm. The replay
    handle is the sealed evidence source (deterministic by the sealed
    cache digests recorded in that evidence). The unified
    registration fields are always populated: defaults are the arm's
    derived address and the sealed source as the claim."""
    license = dict(license or empty_license())
    if not _license_ok(license):
        raise ValueError("license must have string keys code/weights/data")
    operator_key = str(operator_key or address_of(arm_id))
    payout_address = str(payout_address or operator_key)
    return {
        "arm_id": arm_id,
        "kind": "sealed_head",
        "fingerprint": list(fingerprint or []),
        "output_contract": dict(DEFAULT_OUTPUT_CONTRACT),
        "held_out_accuracy": _task_accuracies(held_out_accuracy, per_task),
        "selection_accuracy": float(held_out_accuracy),
        "availability": dict(availability or {"healthy": True}),
        "price": float(price),
        "general": True,
        "primitive": False,
        "license": dict(license),
        "family": family,
        "width": int(width),
        "sealed_source": sealed_source,
        "param_count": int(width) * 345 + 345,  # head weights + bias
        "size_bytes": int(size_bytes),
        "operator_key": operator_key,
        "payout_address": payout_address,
        "sealed_claim": str(sealed_source),
    }
