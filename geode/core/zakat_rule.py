"""M338 (F10-3) - the zakat recipient-selection rule, registered
mechanically before the trigger can fire.

The M325 clause (v26 plan §8.26, clause 4): "Whoever needs it
most" MUST be defined mechanically in the charter (deterministic
recipient-selection rule) before the trigger; the mechanics were
deferred to a later registration but the immutability constraint
was in force already. This module is that registration.

The rule, in three parts:

1. **Genesis-charter-fixed selection.** The recipient list - a
   frozen sequence of (recipient, fraction) pairs - is set ONCE
   at genesis, like the security floors. No mutator exists; the
   dataclass is frozen and the list is stored as a tuple. The
   quorum cannot add, remove, or reweight a recipient.
2. **Refusal until set.** An unset charter (empty list) makes the
   rule NOT READY: ``ready()`` is False and ``disburse`` refuses.
   The trigger is gated on readiness, so the end-state cannot
   fire before the recipients exist - the registered requirement.
3. **Deterministic disbursement.** The pool is split pro-rata to
   the charter fractions; validation enforces distinct
   recipients, positive fractions, and a sum of exactly one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_ZERO = 0.0


class ZakatRuleError(RuntimeError):
    """The zakat rule cannot execute (unset or malformed charter)."""


@dataclass(frozen=True)
class ZakatRule:
    """The charter-fixed zakat recipient-selection rule. Frozen:
    once constructed, the recipients cannot change (no mutator
    exists on this class; the field is a tuple)."""
    recipients: tuple[tuple[str, float], ...] = field(
        default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.recipients:
            # the unset charter is a legitimate state: ready() is
            # False and every trigger/disbursement path refuses.
            return
        names = [str(r) for r, _ in self.recipients]
        if len(set(names)) != len(names):
            raise ZakatRuleError("recipients must be distinct")
        for name, fraction in self.recipients:
            if not name.strip():
                raise ZakatRuleError("recipient names must be "
                                     "non-empty")
            if not 0.0 < float(fraction) <= 1.0:
                raise ZakatRuleError(
                    "recipient fractions must lie in (0, 1]")
        total = sum(float(f) for _, f in self.recipients)
        if total != 1.0:
            raise ZakatRuleError(
                f"recipient fractions must sum to exactly 1.0, "
                f"got {total}")

    def ready(self) -> bool:
        """The trigger gate: the end-state may fire only when the
        charter names its recipients."""
        return bool(self.recipients)

    def disburse(self, pool: float) -> list[dict[str, Any]]:
        """The deterministic split: pro-rata to the charter
        fractions. Refuses (never silently holds) when the charter
        is unset. Refuses a non-positive pool."""
        if not self.ready():
            raise ZakatRuleError(
                "the zakat charter has no recipients - the trigger "
                "cannot fire until the genesis charter is set")
        amount = float(pool)
        if amount <= 0.0:
            raise ZakatRuleError("the zakat pool must be positive")
        return [{"recipient": name, "amount": amount * float(f)}
                for name, f in self.recipients]


def zakat_trigger_ok(pool: float, rule: ZakatRule) -> dict[str, Any]:
    """The trigger-side check: the end-state may fire only with a
    ready rule and a positive pool. Returns the adjudication; the
    disbursement plan is included only when ready."""
    if not rule.ready():
        return {"ok": False,
                "reason": "zakat charter has no recipients"}
    if float(pool) <= 0.0:
        return {"ok": False, "reason": "the zakat pool is empty"}
    return {"ok": True, "plan": rule.disburse(float(pool))}
