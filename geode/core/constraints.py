"""GEODE typed constraint language + commitment-based authorship
(v25 M252).

Prohibitions ("never <action> <subject> under <condition>") are
structured, machine-checkable fields. Authorship is
COMMITMENT-BASED (the trustless-world amendment): a prohibition
becomes active only when (a) each author COMMITTED to it before
reveal (commit-reveal: no selective authoring after seeing the
field), and (b) at least ``min_authors`` DISTINCT authors have
revealed it — dual authorship is replaced by a configurable
threshold of committed authors, because two colluding identities
are no protection (Sybil).

Deterministic: hashes are canonical JSON; no RNG, no wall clocks.
The registry consumes these via the M241 constraint tier: an arm
whose quorum-measured violations match an active prohibition is
excluded (hard), never down-ranked.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

ALLOWED_ACTIONS = frozenset({"emit", "route", "store", "admit"})


@dataclass(frozen=True)
class Prohibition:
    """One prohibition. ``condition`` = "" means unconditional."""
    action: str
    subject: str
    condition: str = ""

    def canonical(self) -> str:
        return json.dumps(
            {"action": self.action, "subject": self.subject,
             "condition": self.condition},
            sort_keys=True, ensure_ascii=True, separators=(",", ":"))

    def matches(self, action: str, subject: str,
                condition: str) -> bool:
        if self.action != action or self.subject != subject:
            return False
        return self.condition == "" or self.condition == condition


def commit_hash(author: str, salt: str, prohibition: Prohibition
                ) -> str:
    """The commit: sha256 of author + salt + canonical constraint."""
    material = f"{author}|{salt}|{prohibition.canonical()}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class ConstraintRegistry:
    """Commit-reveal prohibition registry with a minimum author
    threshold."""

    def __init__(self, min_authors: int = 2):
        if min_authors < 1:
            raise ValueError("min_authors must be >= 1")
        self.min_authors = int(min_authors)
        self._commits: dict[str, dict[str, str]] = {}  # commit_id -> {author, hash, canonical}
        self._reveals: dict[str, list[str]] = {}       # canonical -> commit_ids
        self._active: list[Prohibition] = []

    def commit(self, author: str, salt: str,
               prohibition: Prohibition) -> str:
        """Register a commitment; returns the commit id."""
        cid = commit_hash(str(author), str(salt), prohibition)
        if cid not in self._commits:
            self._commits[cid] = {
                "author": str(author),
                "hash": cid,
                "canonical": prohibition.canonical(),
            }
        return cid

    def reveal(self, commit_id: str, author: str, salt: str,
               prohibition: Prohibition) -> bool:
        """Reveal against a PRIOR commit. Returns True iff the
        prohibition became active with this reveal. Raises when the
        commit is missing or the reveal does not hash to it (no
        reveal-without-commit; a mismatched salt fails)."""
        if commit_id not in self._commits:
            raise ValueError("reveal without a prior commit "
                             f"({commit_id!r})")
        expected = self._commits[commit_id]["hash"]
        if commit_hash(author, salt, prohibition) != expected:
            raise ValueError("reveal does not match the committed "
                             "hash (salt or constraint tampered)")
        canonical = prohibition.canonical()
        authors = self._reveals.setdefault(canonical, [])
        if str(author) not in authors:
            authors.append(str(author))
        if len(authors) >= self.min_authors \
                and prohibition not in self._active:
            self._active.append(prohibition)
            return True
        return False

    def active(self) -> list[Prohibition]:
        return list(self._active)

    def violations(self, arm_record: dict[str, Any]
                   ) -> list[Prohibition]:
        """Which active prohibitions the arm's quorum-measured
        violations match (the M241 constraint-tier input)."""
        known = arm_record.get("known_violations") or []
        out = []
        for v in known:
            for p in self._active:
                if p.matches(v.get("action", ""),
                             v.get("subject", ""),
                             v.get("condition", "")):
                    if p not in out:
                        out.append(p)
        return out
