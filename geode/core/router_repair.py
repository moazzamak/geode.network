"""The repaired router (v26 M303) — price floor, anchor-seeded tie-break,
top-k score-weighted lottery, expected-charge ranking.

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md`` M303
(26 Aug 2026, before any build). This module implements the R-A3
repair WITHOUT touching the sealed ``geode.core.router.Router`` (its
behaviour stays the incumbent until the registered replacement
decision). The safety/containment gates of the sealed Router operate
upstream of selection and are unchanged; this module owns the
selection step only.

Registered selection semantics (written before any measurement):

- **Price floor (R-A3a).** Every axis carries a registered price floor
  (the reference hosting cost). Registration below the floor raises
  ``ValueError``; routing excludes any arm whose posted price is below
  the floor. ``p_a = 0`` can never occur downstream.
- **Expected-charge ranking (R-A3d + M304).** The selection score is
  ``s_a / (p_a * ubar_a)`` - quality per expected total charge of a
  reference query. ``ubar_a`` is the sealed expected unit count on the
  registered reference workload (M304); arms without a measured
  ``ubar_a`` carry the registered default 1.0 and are marked
  ``unmeasured_units`` in the result. Output bloat therefore lowers
  the selection score by exactly the factor it inflates revenue.
- **Top-k lottery (R-A3c).** The qualified arms are ranked by score;
  the route is drawn from the top ``k`` (registered default 5) with
  weights equal to the score. The draw is seeded from
  ``H(anchor, task, registry state root, fingerprint, session id)`` -
  deterministic and replayable per session, unknowable in advance, and
  no host owns the axis. The winner is returned first; the remaining
  pool is the warm failover chain in rank order.

  The session identifier was added by M354 (28 Aug 2026). Without it
  every seed field is constant within an epoch for a given task, and
  the lottery measured as winner-take-all: 1.000 to a single arm in
  all three registered scenarios. See
  ``analysis/m354_route_lottery_entropy.json``.
- **Anchor-seeded tie-break (R-A3b).** Rank ties resolve by
  ``H(artifact_id, anchor)``, never by the artifact hash alone.
"""
from __future__ import annotations

import random
from typing import Any, Sequence

from geode.hashing import payload_hash


def rank_score(accuracy: float, price: float, expected_units: float
               ) -> float:
    """The M303 selection score: quality per expected charge."""
    if price <= 0.0:
        raise ValueError("price must be positive (the price floor "
                         "excludes zero and below-floor prices)")
    if expected_units <= 0.0:
        raise ValueError("expected_units must be positive")
    return float(accuracy) / (price * expected_units)


def tie_key(artifact_id: str, anchor: str) -> str:
    """R-A3b: the tie-break is H(artifact_id, anchor) - unpredictable
    at seal time, deterministic per anchor, never the bare artifact
    hash."""
    return payload_hash({"artifact_id": str(artifact_id),
                         "anchor": str(anchor)})


def draw_seed(anchor: str, task_id: str, state_root: str,
              fp: Sequence[float], session_id: str = "") -> str:
    """The lottery seed: H(anchor, task, registry state root, fp,
    session id). The anchor producer does not control the beacon
    (M311), so nobody can grind the draw. The session id supplies the
    per-session entropy the other four fields cannot: they are all
    constant within an epoch for a given task (M354).
    """
    return payload_hash({"anchor": str(anchor), "task": str(task_id),
                         "state_root": str(state_root),
                         "fingerprint": [float(v) for v in fp],
                         "session": str(session_id)})


class RepairedRouter:
    """Selection-only repaired router over an append-only arm map.

    Arm spec (the harness subset): ``arm_id``, ``held_out_accuracy``,
    ``price``, optional ``expected_units``, optional
    ``availability`` (``{"healthy": bool}``)."""

    def __init__(self, price_floor: float, top_k: int = 5) -> None:
        if price_floor <= 0.0:
            raise ValueError("the price floor must be positive "
                             "(reference hosting cost)")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self.price_floor = float(price_floor)
        self.top_k = int(top_k)
        self._arms: dict[str, dict[str, Any]] = {}

    def add_arm(self, spec: dict[str, Any]) -> None:
        """Register an arm. Below-floor prices are REJECTED here
        (R-A3a), not silently routed around."""
        arm_id = str(spec["arm_id"])
        price = float(spec.get("price") or 0.0)
        if price < self.price_floor:
            raise ValueError(
                f"arm {arm_id!r} price {price} is below the axis floor "
                f"{self.price_floor} (R-A3a: registration rejected)")
        self._arms[arm_id] = dict(spec)

    def list_arms(self) -> list[str]:
        return sorted(self._arms)

    def _healthy(self, arm: dict[str, Any]) -> bool:
        return bool((arm.get("availability") or {}).get("healthy", True))

    def state_root(self) -> str:
        """The registry state root the draw commits against (scores,
        prices, unit expectations)."""
        state = {str(k): {"accuracy": float(v["held_out_accuracy"]),
                          "price": float(v["price"]),
                          "expected_units": float(
                              v.get("expected_units", 1.0))}
                 for k, v in sorted(self._arms.items())}
        return payload_hash(state)

    def route(self, fp: Sequence[float], anchor: str,
              task_id: str = "default",
              session_id: str = "") -> list[dict[str, Any]]:
        """One draw from the top-k lottery. Returns [winner, pool...]
        in the same shape convention as the sealed Router: each entry
        is a copy of the arm dict annotated with ``rank_score``,
        ``share_rank``, ``unmeasured_units``, and the draw's seed and
        winner flag.

        ``session_id`` is assigned at declaration and recorded in the
        ledger, so the route still replays exactly.
        """
        scored: list[tuple[float, str, dict[str, Any]]] = []
        for arm_id, arm in self._arms.items():
            price = float(arm.get("price") or 0.0)
            if price < self.price_floor:
                continue  # R-A3a: below-floor arms are excluded
            if not self._healthy(arm):
                continue  # health gates selection (sealed behaviour)
            ubar = float(arm.get("expected_units", 1.0))
            if ubar <= 0.0:
                continue
            score = rank_score(float(arm["held_out_accuracy"]),
                               price, ubar)
            tie = tie_key(arm_id, anchor)
            scored.append((score, tie, arm))
        if not scored:
            return []
        # score desc; anchor-seeded tie-break asc on exact ties
        scored.sort(key=lambda t: (-t[0], t[1]))
        pool = scored[:self.top_k]
        seed = draw_seed(anchor, task_id, self.state_root(), fp,
                         session_id)
        rng = random.Random(seed)
        weights = [entry[0] for entry in pool]
        total = sum(weights)
        if total <= 0.0:
            return []
        pick = rng.choices(range(len(pool)), weights=weights, k=1)[0]
        out: list[dict[str, Any]] = []
        for idx, (score, _tie, arm) in enumerate(pool):
            rec = dict(arm)
            rec["rank_score"] = score
            rec["share_rank"] = idx + 1
            rec["unmeasured_units"] = "expected_units" not in arm
            rec["draw_seed"] = seed
            rec["lottery_weight"] = score / total
            rec["winner"] = bool(idx == pick)
            out.append(rec)
        # winner first, then the failover pool in rank order
        out.sort(key=lambda r: not r["winner"])
        return out
