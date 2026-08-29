"""M359 (G14) — voluntary voting escrow: burn base decoupled from
weight base. Registered 29 Aug 2026 before the build. Pins: escrow is
voluntary, term-bounded, burnable at L3; unescrowed vested credits
carry no weight."""
from __future__ import annotations

import pytest

from geode.core.voting_escrow import (
    ESCROW_TERM_EPOCHS,
    EscrowNotFound,
    InsufficientBalance,
    VotingEscrow,
)


def test_unescrowed_credits_carry_no_weight():
    esc = VotingEscrow()
    esc.grant_claimable("voter", 1000.0)
    # no escrow -> no weight, whatever the claimable balance
    assert esc.weight("voter", 0) == 0.0
    assert esc.claimable("voter") == 1000.0


def test_lock_creates_weight_term_bounded():
    esc = VotingEscrow()
    esc.grant_claimable("voter", 1000.0)
    esc.lock("voter", 400.0, current_epoch=10)
    # weight for the term
    assert esc.weight("voter", 10) == 400.0
    assert esc.weight("voter", 10 + ESCROW_TERM_EPOCHS - 1) == 400.0
    # exactly at maturity the slot no longer weighs
    assert esc.weight("voter", 10 + ESCROW_TERM_EPOCHS) == 0.0
    # the unescrowed 600 stays claimable and weightless
    assert esc.claimable("voter") == 600.0


def test_lock_is_voluntary_and_bounded():
    esc = VotingEscrow()
    esc.grant_claimable("voter", 100.0)
    with pytest.raises(InsufficientBalance):
        esc.lock("voter", 101.0, current_epoch=0)
    # partial lock leaves the rest claimable
    esc.lock("voter", 40.0, current_epoch=0)
    assert esc.claimable("voter") == 60.0


def test_unlock_matured_returns_to_claimable():
    esc = VotingEscrow()
    esc.grant_claimable("voter", 100.0)
    esc.lock("voter", 100.0, current_epoch=0)
    assert esc.unlock_matured("voter", ESCROW_TERM_EPOCHS - 1) == 0.0
    assert esc.unlock_matured("voter", ESCROW_TERM_EPOCHS) == 100.0
    assert esc.claimable("voter") == 100.0
    assert esc.weight("voter", ESCROW_TERM_EPOCHS) == 0.0


def test_l3_burn_consumes_weight_not_claimable():
    esc = VotingEscrow()
    esc.grant_claimable("voter", 1000.0)
    esc.lock("voter", 500.0, current_epoch=0)   # matures at 8
    esc.lock("voter", 300.0, current_epoch=2)   # matures at 10
    before = esc.claimable("voter")
    # at epoch 5 both slots are live (800 weight)
    assert esc.weight("voter", 5) == 800.0
    burned = esc.burn("voter", 200.0, current_epoch=5)
    assert burned == 200.0
    # weight falls by exactly the burn
    assert esc.weight("voter", 5) == 600.0
    # the claimable balance is untouched by the L3 burn
    assert esc.claimable("voter") == before
    assert esc.total_burnt("voter") == 200.0


def test_l3_burn_cannot_exceed_escrow():
    esc = VotingEscrow()
    esc.grant_claimable("voter", 1000.0)
    esc.lock("voter", 100.0, current_epoch=0)
    with pytest.raises(EscrowNotFound):
        esc.burn("voter", 101.0, current_epoch=5)


def test_claim_and_weight_are_separate_choices():
    # the rational-actor reading of G14: a participant who claims
    # every epoch simply has no weight; weight is the explicit choice
    esc = VotingEscrow()
    esc.grant_claimable("claimer", 1000.0)
    assert esc.weight("claimer", 0) == 0.0
    # and a participant who wants weight pays the term
    esc2 = VotingEscrow()
    esc2.grant_claimable("voter", 1000.0)
    esc2.lock("voter", 1000.0, current_epoch=0)
    assert esc2.weight("voter", 3) == 1000.0
    assert esc2.claimable("voter") == 0.0
