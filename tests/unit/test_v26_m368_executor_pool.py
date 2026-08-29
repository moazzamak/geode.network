"""M368 (G28) — minimum executor pool size. Registered 29 Aug 2026
before the build. Pins: below PI=8 the operative mechanism falls back
to behavioural identity and the registry shows it; corrupt-pair
probability is the exact hypergeometric form."""
from __future__ import annotations

import pytest

from geode.core.executor_pool import (
    MIN_EXECUTOR_POOL,
    corrupt_sample_probability,
    operative_instrument,
    registry_entry,
)


def test_at_or_above_min_pool_sampled_executors():
    assert operative_instrument(8)["instrument"] == "sampled_executors"
    assert operative_instrument(100)["instrument"] == "sampled_executors"
    assert operative_instrument(8)["fallback"] is False


def test_below_min_pool_falls_back():
    out = operative_instrument(2)
    assert out["instrument"] == "behavioral_identity_fallback"
    assert out["fallback"] is True
    assert operative_instrument(7)["fallback"] is True


def test_registry_shows_the_instrument():
    row = registry_entry("arm-x", 2)
    assert row["executor_pool_size"] == 2
    assert row["operative_instrument"] == \
        "behavioral_identity_fallback"
    assert row["fallback_active"] is True
    row8 = registry_entry("arm-y", 8)
    assert row8["operative_instrument"] == "sampled_executors"


def test_exact_hypergeometric_probability():
    # pool 8, sample 2 (k_e = 2), corrupt fraction 0.25 -> 2 corrupt
    assert corrupt_sample_probability(8, 2, 2) == pytest.approx(
        2 / 8 * 1 / 7)   # C(2,2)/C(8,2) = 1/28
    # corrupt fraction 0.5 -> C(4,2)/C(8,2) = 6/28
    assert corrupt_sample_probability(8, 4, 2) == pytest.approx(6 / 28)
    # a full corrupt pool is certainty
    assert corrupt_sample_probability(8, 8, 2) == 1.0
    # no corrupt executors -> never all-corrupt
    assert corrupt_sample_probability(8, 0, 2) == 0.0
    # the pool-2 case G28 names: certainty or near it
    assert corrupt_sample_probability(2, 2, 2) == 1.0


def test_invalid_inputs():
    with pytest.raises(ValueError):
        corrupt_sample_probability(-1, 0, 0)
    with pytest.raises(ValueError):
        corrupt_sample_probability(8, 9, 2)
    with pytest.raises(ValueError):
        corrupt_sample_probability(8, 2, 9)
    with pytest.raises(ValueError):
        operative_instrument(-1)


def test_min_pool_registered():
    assert MIN_EXECUTOR_POOL == 8
