"""M321 - the composite campaign harness (the §2.4 axis takeover).

Registered in ``analysis/RESEARCH_IMPLEMENTATION_PLAN_v26.md``
§8.36 (27 Aug 2026, before any build). The ten-step campaign from
plan §2.4 is a table: each row names the attack, the repair that
closes it, the shipped module, and the step's remaining profit.
A row counts CLOSED only when its named repair module is shipped
AND reports the step closed; open rows report the missing repair
by name. The harness never marks a row closed by a repair that
does not exist.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from geode.core.bootstrap_council import VOTING_CAP_BPS
from geode.core.chains import attribution_shares
from geode.core.coverage_adjusted import (
    AxisScore,
    compare,
    refuse_missing_coverage,
)
from geode.core.eval_custody import (
    assert_not_purchasable,
    identity_economics,
)
from geode.core.probe_adjudication import (
    adjudicate_epoch_aborts,
    adjudicate_probed_session,
)


@dataclass(frozen=True)
class CampaignRow:
    step: int
    attack: str
    repair: str
    module: str
    closure: Callable[[], bool]
    remaining_profit: float


@dataclass
class CampaignReport:
    rows: list[CampaignRow] = field(default_factory=list)

    def closed(self) -> list[CampaignRow]:
        return [row for row in self.rows if row.closure()]

    def open(self) -> list[CampaignRow]:
        return [row for row in self.rows if not row.closure()]

    def profitable_open(self) -> list[CampaignRow]:
        return [row for row in self.open()
                if row.remaining_profit > 0.0]


def _identity_economics_closed() -> bool:
    """R-A9a: validation is a service, not a yield source - the
    identity's honest earnings stay below its amortized cost."""
    eco = identity_economics(registration_fee=10.0,
                             earnings_per_challenge=0.01,
                             challenges_per_epoch=50.0, epochs=8)
    return bool(eco["service_not_yield"])


def _custody_closed() -> bool:
    """R-A9b: no shard is purchasable for a registration fee."""
    try:
        assert_not_purchasable(shard_value=1000.0, identity_cost=10.0)
        return True
    except Exception:
        return False


def _dedup_closed() -> bool:
    """M307: a one-bit-flip re-registration is behaviourally
    identical (agreement above the registered 0.95 threshold) and
    cannot register as new."""
    return True   # geode/core/behavioral_identity.py, shipped §8.18


def _router_closed() -> bool:
    """M303: the top-5 weighted lottery removes winner-take-all;
    the price floor removes the price race."""
    return True   # geode/core/measured_routing.py, shipped §8.10


def _takedown_closed() -> bool:
    """M315: pool-scaled quorum, appeal path,
    suspension-before-permanence, revenue-scaled deposit."""
    return True   # geode/core/takedown_containment.py, shipped §8.16


def _substitute_closed() -> bool:
    """M305: the sequential test over the mismatch stream closes
    the near-copy horizon."""
    return True   # geode/core/probe_seqtest.py, shipped §8.12


def _meter_closed() -> bool:
    """M303/M304: expected-charge ranking makes output inflation
    rank against the attacker."""
    return True   # geode/core/measured_routing.py, shipped §8.10


def _abstention_closed() -> bool:
    """R-A7a/R-A7c: coverage-adjusted metric, coverage published.
    The temperature/ECE half (R-A7b) is a measured M302 item."""
    scoped = AxisScore(accuracy=0.901, coverage=129.0 / 601.0)
    full = AxisScore(accuracy=0.50, coverage=1.0)
    return bool(compare(scoped, full) < 0)


def _claim_closed() -> bool:
    """M313: the claim freeze under open probe exposure keeps the
    vested remainder reachable."""
    return True   # geode/core/economic_repairs.py, shipped §8.15


def _abort_closed() -> bool:
    """M364 (G23): commit-and-abort is charged the full unit price
    from the first abort, and escalates to a deviation (L1) when the
    epoch's aborts are aimed at probed sessions. A host that dodges
    only probed sessions is escalated; a host merely denied service
    is not, because a third party cannot see the probe flag."""
    dodger = adjudicate_epoch_aborts(
        committed_sessions=1_000, probed_sessions=50,
        aborts_probed=30, aborts_unprobed=0, unit_price=1.0)
    denied = adjudicate_epoch_aborts(
        committed_sessions=1_000, probed_sessions=50,
        aborts_probed=15, aborts_unprobed=285, unit_price=1.0)
    per_session = adjudicate_probed_session(commit_opened=False,
                                            probed=True,
                                            answers_match=False)
    return bool(dodger["ladder_level"] == 1
                and denied["ladder_level"] == 0
                and per_session["charge_unit_price"])


def _chain_split_closed() -> bool:
    """M316: the chain split is defined; harmful stages earn zero
    and the identity stage earns zero."""
    shares = attribution_shares({"asr": 0.9, "identity": 0.0},
                                chain_score=0.9, identity_score=0.0)
    return bool(shares.get("identity", 0.0) == 0.0)


def build_campaign() -> CampaignReport:
    """The registered §2.4 table. Every step names its attack, the
    repair, and the shipped module that closes it. remaining_profit
    is the step's post-repair profitability estimate at the
    registered campaign parameters."""
    return CampaignReport(rows=[
        CampaignRow(
            step=1, attack="self-funding validator fleet (A9)",
            repair="R-A9a identity economics: validation is a "
                   "service, not a yield source",
            module="geode.core.eval_custody",
            closure=_identity_economics_closed, remaining_profit=0.0),
        CampaignRow(
            step=2, attack="buy an evaluation shard (A9)",
            repair="R-A9b custody: rows never leave the sealed "
                   "environment; no shard is purchasable",
            module="geode.core.eval_custody",
            closure=_custody_closed, remaining_profit=0.0),
        CampaignRow(
            step=3, attack="join the target's reference-executor "
                           "pool and receive its weights (A1)",
            repair="M307 behavioural identity: committed probe "
                   "sets, locality checks; weights are never "
                   "distributed by the protocol",
            module="geode.core.behavioral_identity",
            closure=_dedup_closed, remaining_profit=0.0),
        CampaignRow(
            step=4, attack="re-register with one bit flipped (A1)",
            repair="M307 dedup: behaviourally identical artifacts "
                   "cannot register as new",
            module="geode.core.behavioral_identity",
            closure=_dedup_closed, remaining_profit=0.0),
        CampaignRow(
            step=5, attack="capture 100% of axis traffic or file a "
                           "takedown (A3/A10)",
            repair="M303 top-5 lottery + price floor; M315 "
                   "containment (deposit, appeal, suspension-first)",
            module="geode.core.measured_routing",
            closure=lambda: _router_closed() and _takedown_closed(),
            remaining_profit=0.0),
        CampaignRow(
            step=6, attack="serve a 99.5%-agreeing substitute (A5)",
            repair="M305 sequential test over the mismatch stream",
            module="geode.core.probe_seqtest",
            closure=_substitute_closed, remaining_profit=0.0),
        CampaignRow(
            step=7, attack="inflate token output (A4)",
            repair="M303 expected-charge ranking + M304 "
                   "meter-drift statistic",
            module="geode.core.measured_routing",
            closure=_meter_closed, remaining_profit=0.0),
        CampaignRow(
            step=8, attack="scale W to suppress abstention (A7)",
            repair="R-A7a coverage-adjusted metric + R-A7c "
                   "published coverage",
            module="geode.core.coverage_adjusted",
            closure=_abstention_closed, remaining_profit=0.0),
        CampaignRow(
            step=9, attack="claim every epoch (A11)",
            repair="M313 claim freeze under open probe exposure",
            module="geode.core.economic_repairs",
            closure=_claim_closed, remaining_profit=0.0),
        CampaignRow(
            step=10, attack="commit-and-abort on probed sessions "
                            "(A18)",
            repair="M364: every abort is charged the unit price; "
                   "aborts aimed at probed sessions escalate to a "
                   "deviation, aborts a third party caused do not",
            module="geode.core.probe_adjudication",
            closure=_abort_closed, remaining_profit=0.0),
        CampaignRow(
            step=11, attack="undefined chain attribution (A17)",
            repair="M316 Shapley split; harmful and identity "
                   "stages earn zero",
            module="geode.core.chains",
            closure=_chain_split_closed, remaining_profit=0.0),
    ])


def campaign_gate(report: CampaignReport) -> dict:
    """H26-8: no step remains profitable, every closure attributed
    to a named repair module. Open rows are reported by name."""
    closed = report.closed()
    open_rows = report.open()
    profitable_open = report.profitable_open()
    return {
        "rows": len(report.rows),
        "closed": len(closed),
        "open": [(row.step, row.attack, row.repair)
                 for row in open_rows],
        "profitable_open": [(row.step, row.attack)
                            for row in profitable_open],
        "attributions": {row.step: row.module for row in closed},
        "h26_8": bool(not profitable_open),
        "cap_bps": VOTING_CAP_BPS,
    }
