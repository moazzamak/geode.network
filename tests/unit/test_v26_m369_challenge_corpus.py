"""M369 (G29) — append-only challenge corpus; depletion budget + alarm.
Registered 29 Aug 2026 before the build. Pins: replenishment extends
the commitment, never rewrites it; admissions pause publicly below the
depletion threshold; the depletion gate ratio exceeds the registered
margin."""
from __future__ import annotations

import pytest

from geode.core.challenge_corpus import (
    DEPLETION_MARGIN,
    DEPLETION_PAUSE_FRACTION,
    AppendOnlyCorpus,
    depletion_gate,
)


def test_append_only_extends_never_rewrites():
    corpus = AppendOnlyCorpus()
    i0 = corpus.commit("root-a", 100)
    i1 = corpus.commit("root-b", 100)
    assert i0 == 0 and i1 == 1
    # revealing from root 0 does not touch root 1
    corpus.reveal(0, 40)
    assert corpus.unrevealed() == 160
    assert corpus._roots[1].drawn == 0


def test_depletion_alarm_pauses_publicly():
    corpus = AppendOnlyCorpus()
    corpus.commit("root-a", 100)
    assert not corpus.admissions_paused()
    # reveal down to below 25%
    corpus.reveal(0, 76)
    assert corpus.unrevealed_fraction() == pytest.approx(0.24)
    assert corpus.admissions_paused() is True
    # replenishment extends and unpauses
    corpus.commit("root-b", 100)
    assert not corpus.admissions_paused()
    assert corpus.unrevealed_fraction() == pytest.approx(0.62)


def test_pause_threshold_registered():
    assert DEPLETION_PAUSE_FRACTION == 0.25


def test_depletion_gate_ratio_exceeds_margin():
    # attacker: 200 registrations at 5.0 each = 1000 to exhaust
    # network: 200 registrations x 10 points at 0.4/point = 800
    out = depletion_gate(registration_fee=5.0,
                         points_revealed_per_registration=10,
                         replenish_cost_per_point=0.4,
                         attacker_registrations=200)
    assert out["attacker_cost"] == 1000.0
    assert out["replenish_cost"] == 800.0
    assert out["ratio"] == pytest.approx(1.25)
    assert out["exceeds_margin"] is False    # 1.25 < 2.0 -> fee too low
    # a fee that clears the margin
    out2 = depletion_gate(registration_fee=20.0,
                          points_revealed_per_registration=10,
                          replenish_cost_per_point=0.4,
                          attacker_registrations=200)
    assert out2["ratio"] == pytest.approx(5.0)
    assert out2["exceeds_margin"] is True


def test_margin_registered():
    assert DEPLETION_MARGIN == 2.0


def test_invalid_inputs():
    corpus = AppendOnlyCorpus()
    with pytest.raises(ValueError):
        corpus.commit("r", 0)
    with pytest.raises(IndexError):
        corpus.reveal(0, 1)
    with pytest.raises(ValueError):
        depletion_gate(-1.0, 1, 1.0, 1)
    with pytest.raises(ValueError):
        depletion_gate(1.0, 0, 1.0, 1)
