"""Unit tests for the M337 session TTL repair (R-F9 / A22,
registered 28 Aug 2026). Pins: the lock form; expiry at TTL; the
re-lock rule (past-TTL sessions re-lock at the CURRENT table,
in-TTL sessions hold their lock); the replay rule (the session's
own epoch's table, never the current one); the unit cap."""
from __future__ import annotations

import pytest

from geode.core.session_ttl import (
    SessionLock,
    lock_session,
    relock_on_expiry,
    replay_price,
    session_expired,
    within_unit_cap,
)


def test_lock_form():
    lock = lock_session("s1", current_epoch=10, artifact_id="a1",
                        unit_price=5, ttl_epochs=1)
    assert lock.lock_epoch == 10
    assert lock.expiry_epoch == 11
    assert lock.locked_unit_price == 5


def test_lock_validation():
    with pytest.raises(ValueError):
        lock_session("s1", current_epoch=-1, artifact_id="a1",
                     unit_price=5)
    with pytest.raises(ValueError):
        lock_session("s1", current_epoch=10, artifact_id="a1",
                     unit_price=5, ttl_epochs=0)
    with pytest.raises(ValueError):
        lock_session("s1", current_epoch=10, artifact_id="a1",
                     unit_price=0)


def test_expiry_at_ttl():
    lock = lock_session("s1", 10, "a1", 5, ttl_epochs=2)
    assert not session_expired(lock, 10)
    assert not session_expired(lock, 11)
    assert session_expired(lock, 12)   # expiry epoch reached
    assert session_expired(lock, 99)


def test_relock_on_expiry_uses_current_table():
    # the stale-price drain: locked at 5, the table moved to 50
    lock = lock_session("s1", 10, "a1", 5, ttl_epochs=1)
    relocked = relock_on_expiry(lock, current_epoch=11,
                                current_unit_price=50)
    assert relocked.lock_epoch == 11
    assert relocked.expiry_epoch == 12
    assert relocked.locked_unit_price == 50   # the CURRENT price


def test_relock_inside_ttl_holds_the_lock():
    # a session inside its TTL is NOT repriced mid-session
    lock = lock_session("s1", 10, "a1", 5, ttl_epochs=3)
    same = relock_on_expiry(lock, current_epoch=12,
                            current_unit_price=50)
    assert same is lock
    assert same.locked_unit_price == 5


def test_replay_uses_the_sessions_own_epoch():
    lock = lock_session("s1", 10, "a1", 5, ttl_epochs=1)
    # the replay prices at the lock epoch's table
    assert replay_price(lock, table_epoch=10) == 5
    # a replay against a different table is refused
    with pytest.raises(ValueError):
        replay_price(lock, table_epoch=11)


def test_unit_cap():
    assert within_unit_cap(100, max_units=1000)
    assert within_unit_cap(1000, max_units=1000)
    assert not within_unit_cap(1001, max_units=1000)
    with pytest.raises(ValueError):
        within_unit_cap(-1)
    with pytest.raises(ValueError):
        within_unit_cap(1, max_units=0)


def test_session_lock_validation():
    with pytest.raises(ValueError):
        SessionLock(session_id="s", lock_epoch=5, expiry_epoch=5,
                    artifact_id="a", locked_unit_price=1)
    with pytest.raises(ValueError):
        SessionLock(session_id="s", lock_epoch=5, expiry_epoch=4,
                    artifact_id="a", locked_unit_price=1)
    with pytest.raises(ValueError):
        SessionLock(session_id="s", lock_epoch=5, expiry_epoch=6,
                    artifact_id="a", locked_unit_price=0)
