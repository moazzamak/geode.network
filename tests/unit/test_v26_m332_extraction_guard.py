"""Unit tests for the M332 extraction-guard machinery (R-A2a/b/c,
registered 28 Aug 2026, before the build). Pins: the bucket edges
form; the bucketed answer carries no raw margin or score vector;
the abstention charge is reduced but nonzero; the budget ledger's
grant/consume/rate/exhaustion semantics."""
from __future__ import annotations

import pytest

from geode.core.extraction_guard import (
    ABSTENTION_PRICE_FRACTION,
    BucketedAnswer,
    BudgetExhausted,
    PayerBudgetLedger,
    abstention_charge,
    confidence_bucket,
)


def test_bucket_edges_validation():
    with pytest.raises(ValueError):
        confidence_bucket(0.5, ())
    with pytest.raises(ValueError):
        confidence_bucket(0.5, (0.1, -0.2))
    with pytest.raises(ValueError):
        confidence_bucket(0.5, (0.2, 0.1))


def test_confidence_bucket_partition():
    edges = (0.1, 0.3, 0.6)
    assert confidence_bucket(0.05, edges) == 0
    assert confidence_bucket(0.1, edges) == 0   # edge inclusive below
    assert confidence_bucket(0.11, edges) == 1
    assert confidence_bucket(0.3, edges) == 1
    assert confidence_bucket(0.31, edges) == 2
    assert confidence_bucket(0.6, edges) == 2
    assert confidence_bucket(0.61, edges) == 3  # top bucket
    assert confidence_bucket(100.0, edges) == 3


def test_confidence_bucket_deterministic():
    edges = (0.1, 0.3)
    assert confidence_bucket(0.2, edges) == confidence_bucket(
        0.2, edges)


def test_bucketed_answer_has_no_raw_margin():
    # structural: the answer type carries the label and the bucket
    # only - no margin field, no score vector field
    fields = {f for f in BucketedAnswer.__dataclass_fields__}
    assert "margin" not in fields
    assert "scores" not in fields
    assert "kappa" not in fields
    ans = BucketedAnswer(label=3, bucket=1, abstained=False)
    assert ans.label == 3 and ans.bucket == 1


def test_abstention_charge_reduced_but_nonzero():
    price = abstention_charge(2.0)
    assert price == 2.0 * ABSTENTION_PRICE_FRACTION
    assert 0.0 < price < 2.0
    with pytest.raises(ValueError):
        abstention_charge(0.0)
    with pytest.raises(ValueError):
        abstention_charge(1.0, fraction=0.0)
    with pytest.raises(ValueError):
        abstention_charge(1.0, fraction=1.0)


def test_budget_grant_and_consume():
    ledger = PayerBudgetLedger()
    ledger.grant("payer1", "vision", 5, cap=10)
    assert ledger.consume("payer1", "vision", 5, queries=4) == 4
    assert ledger.rate("payer1", "vision", 5) == pytest.approx(0.4)
    assert not ledger.exhausted("payer1", "vision", 5)
    ledger.consume("payer1", "vision", 5, queries=6)
    assert ledger.rate("payer1", "vision", 5) == pytest.approx(1.0)
    assert ledger.exhausted("payer1", "vision", 5)


def test_budget_exhaustion_refuses_overage():
    ledger = PayerBudgetLedger()
    ledger.grant("payer1", "vision", 5, cap=3)
    ledger.consume("payer1", "vision", 5, queries=3)
    with pytest.raises(BudgetExhausted):
        ledger.consume("payer1", "vision", 5, queries=1)
    # the overage never counts
    assert ledger.rate("payer1", "vision", 5) == pytest.approx(1.0)


def test_budget_ungranted_key_refused():
    ledger = PayerBudgetLedger()
    with pytest.raises(BudgetExhausted):
        ledger.consume("payer1", "vision", 5)
    with pytest.raises(BudgetExhausted):
        ledger.rate("payer1", "vision", 5)
    assert ledger.exhausted("payer1", "vision", 5)


def test_budget_per_axis_isolation():
    ledger = PayerBudgetLedger()
    ledger.grant("payer1", "vision", 5, cap=10)
    ledger.grant("payer1", "text", 5, cap=10)
    ledger.consume("payer1", "vision", 5, queries=10)
    # the text axis is untouched
    assert ledger.rate("payer1", "text", 5) == pytest.approx(0.0)
    assert not ledger.exhausted("payer1", "text", 5)
