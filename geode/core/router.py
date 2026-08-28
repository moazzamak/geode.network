"""The GEODE router (v24 capability 5, M171).

MVP contract (registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v24.md``
section 5):

- deterministic nearest-arm in fingerprint space over the frozen registry,
  with the M143b-style fusion of the routed arms as a future exploration;
- redundant-capability selection score ordering (measured held-out accuracy,
  measured availability, price) keeps an ordered failover chain per task:
  primary -> next best -> ... -> strongest general arm -> primitives;
- the strongest general arm is the cold-start fallback (I4);
- learned policies are future work; the frozen router stays the incumbent.

Determinism: no RNG anywhere; ties resolve by the registered selection
score, then by arm_id (lexicographic) so the order is reproducible.
"""
from __future__ import annotations

import json
from typing import Any, Sequence

from geode.core.freeze import FreezeError
from geode.hashing import payload_hash


class Router:
    """Deterministic nearest-arm router over an append-only arm registry."""

    def __init__(self) -> None:
        self._arms: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ CRUD
    def add_arm(self, spec: dict[str, Any], freeze: Any = None,
                as_of_index: int | None = None) -> None:
        """Register an arm (append-only). Same arm_id twice is a no-op.

        M248: with a `freeze` registry, admission REJECTS while the
        freeze covers `as_of_index` (raises FreezeError)."""
        if (freeze is not None and as_of_index is not None
                and freeze.is_frozen(as_of_index)):
            raise FreezeError(
                "the registry is frozen at index "
                f"{as_of_index}: admission rejected (M248)")
        required = ("arm_id", "fingerprint", "output_contract",
                    "held_out_accuracy", "availability", "price", "general",
                    "primitive")
        missing = [f for f in required if f not in spec]
        if missing:
            raise ValueError(f"arm spec missing fields: {missing}")
        arm_id = str(spec["arm_id"])
        if arm_id not in self._arms:
            self._arms[arm_id] = dict(spec)

    def list_arms(self) -> list[str]:
        return sorted(self._arms)

    # ------------------------------------------------------------- selection
    def _selection_key(self, arm: dict[str, Any],
                       task_id: str | None = None,
                       best_quality: bool = False
                       ) -> tuple[float, float, float]:
        """Registered ordering (whitepaper, 24 Aug): the router picks
        the capability with the best measured accuracy PER UNIT OF
        POSTED PRICE (efficiency = accuracy / price; a zero price is
        free and reads as infinite efficiency, so among free arms the
        ordering reduces to accuracy). HEALTH gates efficiency: an
        unhealthy arm never outranks a healthy one, whatever its
        price; then efficiency desc, then price asc. ``best_quality``
        (the task's mode) ignores price entirely and orders by
        accuracy."""
        acc = arm.get("selection_accuracy", 0.0)
        if task_id is not None:
            acc = arm.get("held_out_accuracy", {}).get(task_id, acc)
        healthy = 1.0 if arm["availability"].get("healthy", False) else 0.0
        price = float(arm.get("price") or 0.0)
        if best_quality:
            return (healthy, float(acc), -price)
        efficiency = float(acc) / max(price, 1e-9)
        return (healthy, efficiency, -price)

    def _cos(self, fp: Sequence[float],
             arm: dict[str, Any]) -> float | None:
        """Cosine to an arm's task fingerprint; None for fp-less arms."""
        arm_fp = arm.get("fingerprint") or []
        if not arm_fp:
            return None
        dot = sum(a * b for a, b in zip(fp, arm_fp))
        na = sum(a * a for a in fp) ** 0.5
        nb = sum(b * b for b in arm_fp) ** 0.5
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)

    def _matches_contract(self, arm: dict[str, Any],
                          contract_kind: str | None) -> bool:
        """M175 cell C guard: an arm is eligible for a task's contract iff
        its output_contract kind matches (no contract -> everything)."""
        if contract_kind is None:
            return True
        return arm.get("output_contract", {}).get("kind") == contract_kind

    def _emp_cos(self, emp_fp: Sequence[float] | None,
                 arm: dict[str, Any]) -> float | None:
        """Cosine to an arm's EMPIRICAL profile; None when either side
        is absent (the M227 v0 rule: absent -> fall back to the task
        fingerprint)."""
        if emp_fp is None:
            return None
        arm_emp = arm.get("empirical_profile") or []
        if not arm_emp:
            return None
        dot = sum(a * b for a, b in zip(emp_fp, arm_emp))
        na = sum(a * a for a in emp_fp) ** 0.5
        nb = sum(b * b for b in arm_emp) ** 0.5
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)

    def _safety_eligible(self, arm: dict[str, Any],
                         required_tags: Sequence[str] | None,
                         constraints: Any = None) -> bool:
        """M241: hard-constraint admission. Constraint decisions read
        ONLY registry-owned measured fields (`vetted`, `provisional`,
        `measured_tags`); arm-DECLARED fields are never consulted for
        safety admission, so a malicious arm cannot declare itself
        safe. With no safety tags, everything is eligible (the
        backwards-compatible default).

        M252 wiring: with a `constraints` registry, an arm whose
        quorum-measured violations match an ACTIVE prohibition is
        excluded (hard, never down-ranked)."""
        if constraints is not None \
                and constraints.violations(arm):
            return False
        if not required_tags:
            return True
        if not arm.get("vetted", False):
            return False
        if arm.get("provisional", False):
            return False
        measured = set(arm.get("measured_tags") or [])
        return measured.issuperset(set(required_tags))

    def _floor_ok(self, arm_id: str, arm: dict[str, Any],
                  arm_scores: dict[str, float] | None) -> bool:
        """M275: a per-arm abstention floor on the arm's own guard
        score. Fail-closed: an arm WITH a floor but WITHOUT a score
        is excluded (a guarded path is not a safety path). No
        arm_scores -> floors inactive (backwards compatible)."""
        if arm_scores is None:
            return True
        floor = arm.get("abstention_floor")
        if floor is None:
            return True
        score = arm_scores.get(arm_id)
        return score is not None and score <= float(floor)

    def route(self, fp: Sequence[float], k: int = 1,
              task_id: str | None = None,
              contract_kind: str | None = None,
              emp_fp: Sequence[float] | None = None,
              required_tags: Sequence[str] | None = None,
              abstain_below: float | None = None,
              freeze: Any = None,
              as_of_index: int | None = None,
              ood_guard: Any = None,
              input_vec: Sequence[float] | None = None,
              constraints: Any = None,
              arm_scores: dict[str, float] | None = None,
              max_unit_price: float | None = None,
              best_quality: bool = False
              ) -> list[dict[str, Any]]:
        """Top-k arms; the M227 v0 combination rule: the empirical
        profile ranks selection when BOTH the query's empirical
        fingerprint and the arm's profile exist; otherwise the task
        fingerprint ranks (provisional arms - declared but unmeasured -
        are marked in the result).

        M241: with `required_tags` (a safety-flagged task), arms
        without MEASURED coverage of every tag are EXCLUDED (hard
        constraint, never down-ranked); with `abstain_below`, a
        flagged task whose best admissible cosine is below the floor
        returns EMPTY - the caller escalates/refuses (cold_start is
        NOT a safety fallback). Both default to the pre-M241
        behaviour.

        M248: with `freeze` + `as_of_index`, a frozen registry
        returns EMPTY (containment). M251: with `ood_guard`, the
        input is guarded first; an out-of-distribution input (or a
        missing `input_vec`) returns EMPTY - fail-closed, the
        escalation path."""
        if (freeze is not None and as_of_index is not None
                and freeze.is_frozen(as_of_index)):
            return []  # M248 containment: nothing routes while frozen
        if ood_guard is not None:
            if input_vec is None:
                return []  # M251 fail-closed: an unguarded path is
                           # not a safety path
            if not ood_guard.admits(input_vec)["admitted"]:
                return []  # M251: out-of-distribution -> escalate
        scored: list[tuple[float, tuple[float, int, float], str, dict]] = []
        for arm_id, arm in self._arms.items():
            cos = self._cos(fp, arm)
            if cos is None:
                continue  # fp-less arms never match a fingerprint
            if max_unit_price is not None and \
                    float(arm.get("price") or 0.0) > max_unit_price:
                continue  # whitepaper: the task's max unit price
            if not self._floor_ok(arm_id, arm, arm_scores):
                continue  # M275: the arm's own floor abstains this input
            if not self._matches_contract(arm, contract_kind):
                continue  # wrong-modality arms are unreachable
            if not self._safety_eligible(arm, required_tags,
                                         constraints):
                continue  # M241+M252: no measured safety coverage
            emp = self._emp_cos(emp_fp, arm)
            rank = cos if emp is None else emp
            scored.append((rank, self._selection_key(
                arm, task_id, best_quality=best_quality),
                           arm_id, arm))
        # stable two-pass: id asc first, then composite key desc, so exact
        # ties in the composite key keep the lexicographically smallest id
        scored.sort(key=lambda t: t[2])
        scored.sort(key=lambda t: (t[0], t[1][0], t[1][1], t[1][2]),
                    reverse=True)
        if not scored:
            return []
        if abstain_below is not None and scored[0][0] < abstain_below:
            return []  # M241: the registered abstention/escalation path
        out = []
        for rank, _sel, arm_id, arm in scored[:max(k, 1)]:
            rec = dict(arm)
            rec["route_cos"] = rank
            rec["ranked_by"] = ("empirical"
                                if self._emp_cos(emp_fp, arm) is not None
                                else "task")
            rec["provisional"] = bool(arm.get("provisional", False))
            rec["arm_id"] = arm_id
            out.append(rec)
        return out

    def chain(self, fp: Sequence[float],
              task_id: str | None = None,
              contract_kind: str | None = None,
              required_tags: Sequence[str] | None = None,
              abstain_below: float | None = None,
              freeze: Any = None,
              as_of_index: int | None = None,
              ood_guard: Any = None,
              input_vec: Sequence[float] | None = None,
              constraints: Any = None,
              arm_scores: dict[str, float] | None = None,
              max_unit_price: float | None = None,
              best_quality: bool = False
              ) -> list[dict[str, Any]]:
        """The registered failover chain for one task fingerprint.

        Order: fingerprint-matched arms by cosine (unhealthy skipped),
        then general arms by selection score (healthy only), then
        primitives by selection score. With a query task_id, the
        selection score uses the arm's measured held-out accuracy for
        THAT task (the registered redundant-capability ordering). With
        ``contract_kind`` (M175 cell C), every tier admits only
        output-contract-matching arms — no silent cross-modality
        failover.

        M241: with ``required_tags`` (a safety-flagged task) the
        general and primitive tiers are SKIPPED entirely — flagged
        tasks never fail over to arms without measured coverage — and
        ``abstain_below`` applies (an empty chain is the
        escalation/refusal path).

        M255 (the gap-audit fix, completed M270): containment applies
        to EVERY tier — a frozen registry or a guarded
        out-of-distribution input returns EMPTY before any tier is
        assembled, so fingerprint-less general arms cannot bypass the
        freeze.
        """
        if (freeze is not None and as_of_index is not None
                and freeze.is_frozen(as_of_index)):
            return []  # M248 containment: nothing routes while frozen
        if ood_guard is not None:
            if input_vec is None:
                return []  # M251 fail-closed
            if not ood_guard.admits(input_vec)["admitted"]:
                return []  # M251: out-of-distribution -> escalate
        matched = []
        for rec in self.route(fp, k=len(self._arms), task_id=task_id,
                              contract_kind=contract_kind,
                              required_tags=required_tags,
                              abstain_below=abstain_below,
                              freeze=freeze, as_of_index=as_of_index,
                              ood_guard=ood_guard,
                              input_vec=input_vec,
                              constraints=constraints,
                              arm_scores=arm_scores,
                              max_unit_price=max_unit_price,
                              best_quality=best_quality):
            if rec["availability"].get("healthy", True):
                matched.append(rec)
        if required_tags:
            # M241: no unmeasured fallback tiers for flagged tasks.
            return matched
        general = []
        primitives = []
        for arm_id in sorted(self._arms):
            arm = self._arms[arm_id]
            if not arm["availability"].get("healthy", True):
                continue
            if max_unit_price is not None and \
                    float(arm.get("price") or 0.0) > max_unit_price:
                continue  # whitepaper: the task's max unit price
            if not self._floor_ok(arm_id, arm, arm_scores):
                continue  # M275: the arm's own floor abstains this input
            if not self._matches_contract(arm, contract_kind):
                continue
            if not (arm.get("fingerprint") or []):
                tier = primitives if arm["primitive"] else general
                rec = dict(arm)
                rec["route_cos"] = None
                rec["arm_id"] = arm_id
                tier.append(rec)
        general.sort(key=lambda r: self._selection_key(
            r, task_id, best_quality=best_quality), reverse=True)
        primitives.sort(key=lambda r: self._selection_key(
            r, task_id, best_quality=best_quality), reverse=True)
        return matched + general + primitives

    def cold_start(self, output_kind: str | None = None,
                   freeze: Any = None,
                   as_of_index: int | None = None,
                   ood_guard: Any = None,
                   input_vec: Sequence[float] | None = None,
                   constraints: Any = None,
                   arm_scores: dict[str, float] | None = None
                   ) -> dict[str, Any]:
        """Strongest general arm (I4); prefers contract-matched generals.

        Preference order: contract-matching general arms, then any
        general arm, then any arm (specialists last resort), all by
        selection score; empty registry -> {} (the runner supplies the
        programmatic primitive in that case).

        M255 (the gap-audit fix): cold_start is part of the routing
        surface, so it obeys the SAME containment rules as route and
        chain — a frozen registry returns {} (M248) and a guarded
        out-of-distribution (or missing) input returns {} (M251,
        fail-closed). Cold-start is not a containment backdoor.
        """
        if (freeze is not None and as_of_index is not None
                and freeze.is_frozen(as_of_index)):
            return {}  # M248: nothing is served while frozen
        if ood_guard is not None:
            if input_vec is None or not ood_guard.admits(
                    input_vec)["admitted"]:
                return {}  # M251: out-of-distribution -> no arm
        def eligible(pred=None) -> list[dict[str, Any]]:
            tier = []
            for arm_id, arm in self._arms.items():
                if not arm["availability"].get("healthy", True):
                    continue
                if not self._floor_ok(arm_id, arm, arm_scores):
                    continue  # M275: the arm's own floor abstains
                if pred is not None and not pred(arm):
                    continue
                if constraints is not None \
                        and constraints.violations(arm):
                    continue  # M252: active-prohibition violators
                             # never cold-start
                rec = dict(arm)
                rec["arm_id"] = arm_id
                tier.append(rec)
            tier.sort(key=lambda r: self._selection_key(r), reverse=True)
            return tier

        if output_kind is not None:
            kind_match = [r for r in eligible(lambda a: a["general"])
                          if r["output_contract"]["kind"] == output_kind]
            if kind_match:
                return kind_match[0]
        generals = eligible(lambda a: a["general"])
        if generals:
            return generals[0]
        non_primitives = eligible(lambda a: not a["primitive"])
        if non_primitives:
            return non_primitives[0]
        return eligible()[0] if eligible() else {}

    # --------------------------------------------------------------- hashing
    def content_hash(self) -> str:
        """Payload hash of all arm content (deterministic, no wall clocks)."""
        return payload_hash(json.dumps(
            self._arms, sort_keys=True, ensure_ascii=True,
            separators=(",", ":")))
