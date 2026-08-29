"""M358 — externally-verified voting weight (review-R2 finding G13).

Registered in ``analysis/WHITEPAPER_REVIEW_2026-08-28_R2.md`` §G13
before any build.

The defect. The whitepaper says voting weight "accrues only through
verified work", but the verified-work-only rule (R-A20, shipped in
``geode.core.economic_repairs``) is scoped to *tenure and activity*.
Weight itself is the earned-but-unclaimed credit balance, and the
self-payment exclusion keys on the payout address, so it blocks a
1-cycle only. A ring of three behaviourally distinct artifacts under
one owner cycles payments A -> B -> C -> A: every hop loses the 2.5%
dev-fund dock and credits the recipient with unclaimed balance, i.e.
with voting weight. The ring satisfies the d>=3 diversity floor by
construction. Capital converts into governance weight at a ~5%
haircut.

The repair. Split the balance in two. ``total`` stays the claimable
balance and is untouched — the ring keeps its (net negative) money.
``verified`` is the voting-weight base and counts only revenue from
**external** payers: a payer that has received attribution credit
from the beneficiary within a registered linkage depth is not
external, so cycled revenue accrues no weight.

Why a graph closure and not the payout-address check. The payout
address catches depth 1. The ring is depth ``n``. The closure to a
registered depth ``LINKAGE_DEPTH`` catches every cycle no longer than
the depth, which is what makes the attack unprofitable rather than
merely more expensive.

Honest residual (measured by ``mutual_trade_cost``, not hidden). The
externality test is binary, so two genuinely independent businesses
that trade with each other in both directions lose voting weight on
that revenue. They keep the money; they lose the weight. Prior art
(MeritRank, arXiv:2207.09950) grades this instead of thresholding it,
via transitivity/connectivity/epoch decay. Grading is registered as
the follow-on if the measured false-positive rate matters; the binary
rule is the one the review registered and is strictly conservative.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from geode.core.economics import DEV_FUND_BPS

DEV_FUND_FRACTION = DEV_FUND_BPS / 1000.0

# The registered transitive linkage depth. A 2-cycle and a 3-cycle
# are the affordable ring shapes (each extra hop costs another
# dev-fund dock and another behaviourally distinct artifact), so the
# depth must reach at least 3 to make the cheapest rings weightless.
LINKAGE_DEPTH = 3


@dataclass(frozen=True)
class CreditBalances:
    """The two balances G13 separates.

    ``total`` is claimable credit — unchanged by this module.
    ``verified`` is the voting-weight base: externally-verified
    revenue only.
    """
    total: float
    verified: float

    @property
    def internal(self) -> float:
        """Revenue that earns money but no weight."""
        return self.total - self.verified


def linkage_closure(credit_edges: Mapping[str, Iterable[str]],
                    source: str,
                    depth: int = LINKAGE_DEPTH) -> frozenset[str]:
    """Identities that received attribution credit from ``source``
    within ``depth`` hops, ``source`` included.

    ``credit_edges`` maps an identity to the identities its payments
    credited. A breadth-first walk is used rather than a recursive
    one so that a cyclic graph — which is exactly the case under test
    — terminates.
    """
    if int(depth) < 0:
        raise ValueError("depth must be non-negative")
    reached = {str(source)}
    frontier: deque[tuple[str, int]] = deque([(str(source), 0)])
    while frontier:
        node, hops = frontier.popleft()
        if hops >= int(depth):
            continue
        for nxt in credit_edges.get(node, ()):  # type: ignore[arg-type]
            nxt = str(nxt)
            if nxt not in reached:
                reached.add(nxt)
                frontier.append((nxt, hops + 1))
    return frozenset(reached)


def is_external(payer: str, beneficiary: str,
                credit_edges: Mapping[str, Iterable[str]],
                depth: int = LINKAGE_DEPTH) -> bool:
    """The externality predicate: a payer inside the beneficiary's
    linkage closure is not external, and self-payment is the depth-0
    case."""
    return str(payer) not in linkage_closure(
        credit_edges, beneficiary, depth)


def split_credits(sessions: Sequence[Mapping[str, Any]],
                  beneficiary: str,
                  credit_edges: Mapping[str, Iterable[str]],
                  depth: int = LINKAGE_DEPTH) -> CreditBalances:
    """Split one identity's settled revenue into the claimable total
    and the externally-verified voting-weight base.

    Each session is a mapping with ``payer``, ``beneficiary`` and
    ``credit`` (the settled amount after the dev-fund dock). Sessions
    naming another beneficiary are ignored.
    """
    closure = linkage_closure(credit_edges, beneficiary, depth)
    total = 0.0
    verified = 0.0
    for session in sessions:
        if str(session.get("beneficiary")) != str(beneficiary):
            continue
        credit = float(session.get("credit", 0.0))
        if credit < 0.0:
            raise ValueError("session credit must be non-negative")
        total += credit
        if str(session.get("payer")) not in closure:
            verified += credit
    return CreditBalances(total=total, verified=verified)


def weight_snapshot_weights(
        sessions: Sequence[Mapping[str, Any]],
        identities: Sequence[str],
        credit_edges: Mapping[str, Iterable[str]],
        depth: int = LINKAGE_DEPTH) -> dict[str, float]:
    """The weight map for ``geode.privacy.vote_machinery.WeightSnapshot``.

    This is the wiring point: the snapshot must be built from the
    verified balance, never from the claimable one.
    """
    return {str(i): split_credits(sessions, i, credit_edges,
                                  depth).verified
            for i in identities}


def credit_edges_from_sessions(
        sessions: Sequence[Mapping[str, Any]]
        ) -> dict[str, set[str]]:
    """Derive the attribution-credit graph from settled sessions:
    an edge payer -> beneficiary for every session."""
    edges: dict[str, set[str]] = {}
    for session in sessions:
        payer = str(session.get("payer"))
        beneficiary = str(session.get("beneficiary"))
        edges.setdefault(payer, set()).add(beneficiary)
    return edges


# ---------------------------------------------------------------------------
# The registered M358 gate
# ---------------------------------------------------------------------------
def wash_ring_gate(ring_size: int, spend_per_hop: float,
                   depth: int = LINKAGE_DEPTH,
                   dev_fund_fraction: float = DEV_FUND_FRACTION
                   ) -> dict[str, Any]:
    """M358 gate: a ring of ``ring_size`` distinct identities cycling
    ``spend_per_hop`` acquires zero voting weight and loses the
    dev-fund dock on every hop, while an honest supplier serving
    external payers keeps its weight in full.

    Returns both arms — the pre-repair weight the ring would have
    bought and the post-repair weight it does buy — so the reading is
    a contrast, not a single number.
    """
    if int(ring_size) < 2:
        raise ValueError("a ring needs at least two identities")
    if float(spend_per_hop) <= 0.0:
        raise ValueError("spend_per_hop must be positive")
    n = int(ring_size)
    spend = float(spend_per_hop)
    credit = spend * (1.0 - float(dev_fund_fraction))

    members = [f"ring-{i}" for i in range(n)]
    sessions = [{"payer": members[i],
                 "beneficiary": members[(i + 1) % n],
                 "credit": credit}
                for i in range(n)]
    # the honest control: one supplier, external payers, same volume
    honest_sessions = [{"payer": f"buyer-{i}", "beneficiary": "honest",
                        "credit": credit} for i in range(n)]

    all_sessions = sessions + honest_sessions
    edges = credit_edges_from_sessions(all_sessions)

    ring_weight = sum(
        split_credits(all_sessions, m, edges, depth).verified
        for m in members)
    ring_claimable = sum(
        split_credits(all_sessions, m, edges, depth).total
        for m in members)
    honest = split_credits(all_sessions, "honest", edges, depth)

    pre_repair_weight = ring_claimable   # the shipped rule's base
    capital_lost = n * spend - ring_claimable
    haircut = capital_lost / (n * spend)

    return {
        "ring_size": n,
        "spend_per_hop": spend,
        "linkage_depth": int(depth),
        "ring_weight_post_repair": ring_weight,
        "ring_weight_pre_repair": pre_repair_weight,
        "ring_claimable": ring_claimable,
        "capital_lost": capital_lost,
        "haircut": haircut,
        "honest_weight": honest.verified,
        "honest_claimable": honest.total,
        "passes": (ring_weight == 0.0
                   and pre_repair_weight > 0.0
                   and honest.verified == honest.total),
        "note": ("pre-repair the ring buys weight equal to its "
                 "cycled credit at the dev-fund haircut; post-repair "
                 "it buys none, and the honest supplier is untouched"),
    }


def mutual_trade_cost(spend: float,
                      depth: int = LINKAGE_DEPTH) -> dict[str, Any]:
    """The honest residual, measured rather than asserted: two
    independent businesses that trade in BOTH directions lose voting
    weight on that revenue while keeping the money.

    Reported so the residual appears in the evidence and in Known
    Limits instead of being discovered by an honest supplier.
    """
    if float(spend) <= 0.0:
        raise ValueError("spend must be positive")
    credit = float(spend) * (1.0 - DEV_FUND_FRACTION)
    sessions = [{"payer": "alpha", "beneficiary": "beta",
                 "credit": credit},
                {"payer": "beta", "beneficiary": "alpha",
                 "credit": credit},
                {"payer": "buyer", "beneficiary": "alpha",
                 "credit": credit}]
    edges = credit_edges_from_sessions(sessions)
    alpha = split_credits(sessions, "alpha", edges, depth)
    return {
        "linkage_depth": int(depth),
        "claimable": alpha.total,
        "voting_weight": alpha.verified,
        "weight_lost_to_mutual_trade": alpha.internal,
        "note": ("binary externality: mutual trade between two "
                 "genuine businesses keeps the money and loses the "
                 "weight; MeritRank-style graded decay is the "
                 "registered follow-on"),
    }
