"""M337 - session TTL and unit cap: the stale-price drain repair
(R-F9 / A22).

Registered in ``analysis/FEASIBILITY_THREAT_REVIEW_2026-08-28.md``
(F9, R-F9) and the review's queue (M337). The gap: A22 was found,
analyzed, and repair-specified - and then dropped: no milestone, no
paper edit. The stale-price drain (open a session immediately
before a timelocked price increase and drain it indefinitely at
the old price; the same trick front-runs an axis-floor change)
stayed open in the paper as written.

The registered repair: a session TTL and a maximum unit count per
session; re-route and re-lock on expiry. One rule, one paragraph,
one unit test. Gate: a session past its TTL re-locks at the
current price table; the replay uses the table of the session's
own epoch.
"""
from __future__ import annotations

from dataclasses import dataclass

# the registered defaults: one epoch TTL, 2**32 units (the cap is
# a safety bound, not the binding constraint for honest sessions)
DEFAULT_SESSION_TTL_EPOCHS = 1
DEFAULT_MAX_UNITS_PER_SESSION = 2 ** 32


@dataclass(frozen=True)
class SessionLock:
    """A serving session's lock: the price table epoch it locked,
    the artifact it routed to, the unit price it locked, and the
    expiry epoch (lock epoch + TTL)."""
    session_id: str
    lock_epoch: int
    expiry_epoch: int
    artifact_id: str
    locked_unit_price: int
    units_served: int = 0

    def __post_init__(self) -> None:
        if self.lock_epoch < 0:
            raise ValueError("the lock epoch must be non-negative")
        if self.expiry_epoch <= self.lock_epoch:
            raise ValueError(
                "the expiry epoch must be after the lock epoch")
        if self.locked_unit_price <= 0:
            raise ValueError("the locked unit price must be positive")
        if self.units_served < 0:
            raise ValueError("units served must be non-negative")


def lock_session(session_id: str, current_epoch: int,
                 artifact_id: str, unit_price: int,
                 ttl_epochs: int = DEFAULT_SESSION_TTL_EPOCHS,
                 ) -> SessionLock:
    """Lock a session at the current epoch's price table. The TTL
    is registered in epochs; the expiry is lock + TTL."""
    if current_epoch < 0:
        raise ValueError("the current epoch must be non-negative")
    if ttl_epochs <= 0:
        raise ValueError("the TTL must be positive")
    if unit_price <= 0:
        raise ValueError("the unit price must be positive")
    return SessionLock(
        session_id=session_id, lock_epoch=current_epoch,
        expiry_epoch=current_epoch + ttl_epochs,
        artifact_id=artifact_id, locked_unit_price=unit_price)


def session_expired(lock: SessionLock, current_epoch: int) -> bool:
    """True iff the session is past its TTL at the current epoch."""
    if current_epoch < 0:
        raise ValueError("the current epoch must be non-negative")
    return current_epoch >= lock.expiry_epoch


def relock_on_expiry(lock: SessionLock, current_epoch: int,
                     current_unit_price: int,
                     ttl_epochs: int = DEFAULT_SESSION_TTL_EPOCHS,
                     ) -> SessionLock:
    """The R-F9 rule: a session past its TTL re-locks at the
    CURRENT price table (a fresh lock epoch, a fresh expiry, the
    current unit price). A session inside its TTL is returned
    unchanged - the lock holds; no mid-session repricing."""
    if not session_expired(lock, current_epoch):
        return lock
    return lock_session(
        session_id=lock.session_id, current_epoch=current_epoch,
        artifact_id=lock.artifact_id,
        unit_price=current_unit_price, ttl_epochs=ttl_epochs)


def replay_price(lock: SessionLock, table_epoch: int) -> int:
    """The replay rule: a replayed session is priced at the table
    of its OWN epoch - the lock epoch's price, never the current
    one. A replay at a later table is refused (the price is part of
    the session's sealed record)."""
    if table_epoch != lock.lock_epoch:
        raise ValueError(
            f"replay must use the session's own table epoch "
            f"({lock.lock_epoch}), got {table_epoch}")
    return lock.locked_unit_price


def within_unit_cap(units: int,
                    max_units: int = DEFAULT_MAX_UNITS_PER_SESSION,
                    ) -> bool:
    """The unit cap: a session may not serve more than the
    registered maximum units, however long it lives."""
    if units < 0:
        raise ValueError("units must be non-negative")
    if max_units <= 0:
        raise ValueError("the unit cap must be positive")
    return units <= max_units
