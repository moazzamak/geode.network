"""M360 (G21) — per-epoch price drift band + currency-risk Known Limit.
Registered 29 Aug 2026 before the build. Pins: a contributor may
register price fixed or with a +/-10% per-epoch drift band; the router
replays against the effective price table of the day; the fixed rule
is the band=0 special case."""
from __future__ import annotations

import pytest

from geode.core.economics import (
    PRICE_DRIFT_BAND_BPS,
    clamp_to_drift,
    effective_price_path,
    price_within_drift,
)


def test_band_zero_is_today_fixed_rule():
    # band 0: only an equal price is inside the band
    assert price_within_drift(100, 100, band_bps=0)
    assert not price_within_drift(100, 101, band_bps=0)
    # and the effective path is the base at every epoch
    assert effective_price_path(100, [100, 100], band_bps=0) \
        == [100, 100, 100]


def test_declaration_inside_band_accepted():
    # +/-10% of 1000 is 100
    assert price_within_drift(1000, 1100, PRICE_DRIFT_BAND_BPS)
    assert price_within_drift(1000, 900, PRICE_DRIFT_BAND_BPS)
    assert not price_within_drift(1000, 1101, PRICE_DRIFT_BAND_BPS)
    assert not price_within_drift(1000, 899, PRICE_DRIFT_BAND_BPS)


def test_clamp_caps_the_move():
    assert clamp_to_drift(1000, 5000, PRICE_DRIFT_BAND_BPS) == 1100
    assert clamp_to_drift(1000, 1, PRICE_DRIFT_BAND_BPS) == 900
    # a large move takes multiple epochs to traverse
    path = effective_price_path(1000, [5000, 5000, 5000])
    assert path == [1000, 1100, 1210, 1331]


def test_no_declaration_carries_previous_price():
    path = effective_price_path(1000, [1100, None, 1200])
    assert path == [1000, 1100, 1100, 1200]


def test_router_replays_the_price_path():
    # the replay reads the deterministic path, never a live feed
    path = effective_price_path(1000, [5000])
    replay = effective_price_path(1000, [5000])
    assert path == replay
    assert path[1] == 1100


def test_invalid_inputs():
    with pytest.raises(ValueError):
        price_within_drift(0, 100)
    with pytest.raises(ValueError):
        price_within_drift(100, 0)
    with pytest.raises(ValueError):
        price_within_drift(100, 100, band_bps=-1)
    with pytest.raises(ValueError):
        effective_price_path(0, [100])
    with pytest.raises(ValueError):
        effective_price_path(100, [0])
