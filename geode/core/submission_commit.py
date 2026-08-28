"""GEODE commit-reveal arm submissions (v25 M282).

A contributor commits to a capability claim + weight digest (salted)
BEFORE probes run. Admission accepts a reveal only if it hashes to a
prior commit; selection metrics are computed against the COMMITTED
claim, so a contributor cannot re-describe an arm after seeing
results — the mismatched reveal fails admission by construction.

The commit primitive reuses the M252 pattern (sha256 over
author|salt|canonical payload). Deterministic: no RNG, no wall
clocks inside hashes or receipts. Append-only: every stage writes a
receipt.

Design contract (registered):
- commit -> (probes run) -> reveal -> admit. Reveal-without-commit,
  admit-without-reveal, and a reveal that does not hash to the
  commit all raise.
- The measured-vs-committed comparison uses the claim AS COMMITTED
  (re-described claims never enter the comparison).
- An arm that measured below its committed claim is rejected with
  the committed claim recorded in the receipt.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


def canonical_claim(arm_id: str, family: str,
                    capability_claim: dict[str, Any],
                    weight_digest: str) -> str:
    """Canonical JSON for the submission payload. Dict fields are
    sorted, so two claims equal in content hash identically."""
    payload: dict[str, Any] = {
        "arm_id": str(arm_id),
        "family": str(family),
        "capability_claim": capability_claim,
        "weight_digest": str(weight_digest),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=True,
                      separators=(",", ":"))


def submission_commit(author: str, salt: str, arm_id: str,
                      family: str, capability_claim: dict[str, Any],
                      weight_digest: str) -> str:
    """The commit: sha256 of author + salt + canonical claim."""
    canonical = canonical_claim(arm_id, family, capability_claim,
                                weight_digest)
    material = f"{author}|{salt}|{canonical}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AdmissionRule:
    """Admission compares measured per-task accuracies against the
    COMMITTED claim (equal or better admits; below rejects)."""
    tolerance: float = 0.0

    def passes(self, committed: dict[str, float],
               measured: dict[str, float]) -> bool:
        return all(
            measured.get(k, -1.0) >= v - self.tolerance
            for k, v in committed.items())


class SubmissionLedger:
    """Append-only commit-reveal registry for arm submissions.

    States per commit id: committed -> revealed -> admitted/rejected.
    Receipts record every stage in order."""

    def __init__(self, rule: AdmissionRule | None = None):
        self.rule = rule or AdmissionRule()
        self._commits: dict[str, dict[str, Any]] = {}
        self._revealed: dict[str, str] = {}      # commit_id -> canonical
        self._admitted: set[str] = set()
        self.receipts: list[dict[str, Any]] = []

    # ---- stages ------------------------------------------------------

    def commit(self, author: str, salt: str, arm_id: str,
               family: str, capability_claim: dict[str, Any],
               weight_digest: str) -> str:
        """Register the commitment BEFORE probes run. Returns the
        commit id."""
        cid = submission_commit(author, salt, arm_id, family,
                                capability_claim, weight_digest)
        canonical = canonical_claim(arm_id, family, capability_claim,
                                    weight_digest)
        if cid in self._commits:
            raise ValueError(f"duplicate commit ({cid})")
        self._commits[cid] = {
            "commit_id": cid,
            "author": str(author),
            "canonical": canonical,
            "arm_id": str(arm_id),
            "family": str(family),
            "capability_claim": capability_claim,
            "weight_digest": str(weight_digest),
        }
        self.receipts.append({"stage": "commit", "commit_id": cid,
                              "author": str(author),
                              "arm_id": str(arm_id),
                              "family": str(family)})
        return cid

    def reveal(self, commit_id: str, author: str, salt: str,
               arm_id: str, family: str,
               capability_claim: dict[str, Any],
               weight_digest: str) -> bool:
        """Reveal against a PRIOR commit. Returns True iff the
        reveal hashes to the committed id. Raises on a missing
        commit or a mismatched reveal (salt/claim tampered) — a
        mismatched reveal fails admission by construction."""
        if commit_id not in self._commits:
            raise ValueError("reveal without a prior commit "
                             f"({commit_id!r})")
        if commit_id in self._revealed:
            raise ValueError(f"already revealed ({commit_id})")
        if submission_commit(author, salt, arm_id, family,
                             capability_claim, weight_digest) != \
                commit_id:
            raise ValueError("reveal does not hash to the committed "
                             "id (salt or claim tampered)")
        self._revealed[commit_id] = self._commits[commit_id][
            "canonical"]
        self.receipts.append({"stage": "reveal", "commit_id": commit_id,
                              "arm_id": arm_id})
        return True

    def admit(self, commit_id: str, measured: dict[str, float]
              ) -> dict[str, Any]:
        """Admit ONLY against a committed + revealed submission. The
        comparison uses the claim AS COMMITTED (re-described claims
        never enter it). Returns the admission receipt."""
        if commit_id not in self._commits:
            raise ValueError("admit without a prior commit "
                             f"({commit_id!r})")
        if commit_id not in self._revealed:
            raise ValueError("admit before reveal "
                             f"({commit_id!r})")
        if commit_id in self._admitted:
            raise ValueError(f"already admitted ({commit_id})")
        entry = self._commits[commit_id]
        committed: dict[str, float] = dict(
            entry["capability_claim"].get("per_task") or {})
        passed = self.rule.passes(committed, measured)
        if passed:
            self._admitted.add(commit_id)
        receipt = {
            "stage": "admit" if passed else "reject",
            "commit_id": commit_id,
            "arm_id": entry["arm_id"],
            "family": entry["family"],
            "committed_per_task": committed,
            "measured_per_task": dict(measured),
            "admitted": passed,
        }
        self.receipts.append(receipt)
        return receipt

    def admitted_ids(self) -> list[str]:
        return sorted(self._admitted)
