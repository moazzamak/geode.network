"""M371 (G31) — beacon composition required. Registered 29 Aug 2026
before the build. Pins: H(drand || RANDAO-VDF) is the registered
composition; safe as long as either source is honest; the composition
is deterministic."""
from __future__ import annotations

import pytest

from geode.core.beacon import beacon_safe_if_either_honest, composed_beacon


def test_composition_is_deterministic():
    a = composed_beacon("round-100", "vdf-out-42")
    b = composed_beacon("round-100", "vdf-out-42")
    assert a == b


def test_composition_is_order_sensitive():
    assert composed_beacon("x", "y") != composed_beacon("y", "x")


def test_composition_changes_with_either_source():
    base = composed_beacon("r1", "v1")
    assert composed_beacon("r2", "v1") != base
    assert composed_beacon("r1", "v2") != base


def test_safe_if_either_honest():
    assert beacon_safe_if_either_honest("r", "v", True, True)
    assert beacon_safe_if_either_honest("r", "v", True, False)
    assert beacon_safe_if_either_honest("r", "v", False, True)
    # only BOTH compromised breaks the composition
    assert not beacon_safe_if_either_honest("r", "v", False, False)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        composed_beacon("", "v")
    with pytest.raises(ValueError):
        composed_beacon("r", "")
