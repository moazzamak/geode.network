"""M212 — the settlement wire (whitepaper-aligned, 24 Aug 2026):
the orchestrator's ledger route records become deterministic,
chain-anchored, contract-submittable ``recordCredits`` attribution
batches for the reworked CreditLedger.

Registered rules:

- fee P_REF = 1 credit/query (tokenless bookkeeping unit; the M186
  pricing band is NOT claimed here);
- pool allocation = the contract's own deposit arithmetic (2.5% dev
  cut first, floor division);
- the routed top-1 arm's PAYOUT ADDRESS is credited (default:
  address = sha256('geode:'+arm_id) prefix — no identity; an arm may
  register a payout address that differs from its operator key);
- the self-payment exclusion mirrors the contract EXACTLY: an entry
  whose PAYER is the credited payout address is skipped, because
  ``recordCredits`` skips it (C1 keys on the payout address);
- every entry carries the registration's artifactId (the contract's
  new batch schema: no component masks);
- entries in query order, at most 64 per batch (the contract's
  MAX_BATCH); amounts must fit the pool in post order (a
  would-revert batch is a builder violation);
- every batch carries the M185 anchor_spec fields (ledger tip, record
  count, last record hash).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Sequence

from geode.audit import TIMING_FIELDS
from geode.core.economics import (  # noqa: F401  (re-exported)
    MAX_BATCH,
    artifact_id_of,
    deposit_split,
)
from geode.hashing import payload_hash

DEV_FUND_BPS = 25       # mirrors CreditLedger.DEV_FUND_BPS

BIT_ENCODER = 1 << 0
BIT_HEAD = 1 << 1
BIT_DNN = 1 << 2
BIT_DATA = 1 << 3
BIT_ORCH = 1 << 4


def address_of(name: str) -> str:
    """Deterministic 20-byte address from a name (no identity).
    Kept for the M212-era callers; the canonical implementation is
    ``geode.economics.address_of``."""
    digest = hashlib.sha256(f"geode:{name}".encode("utf-8")).hexdigest()
    return "0x" + digest[:40]


def mask_for(kind: str) -> int:
    """The legacy component mask of an arm kind (kept for the M212
    study; the reworked contract batch schema no longer carries
    masks)."""
    if kind == "dnn":
        return BIT_ENCODER | BIT_DNN | BIT_ORCH
    return BIT_ENCODER | BIT_HEAD | BIT_ORCH


def _route_records(ledger_dict: dict[str, Any]) -> list[dict[str, Any]]:
    return [rec["content"] for rec in ledger_dict.get("records", [])
            if rec["content"].get("kind") == "route"]


def build_credit_batches(
        orchestrator: Any,
        price_per_query: int,
        payer_of: Callable[[str], str],
        payout_of: Callable[[str], str] | None = None,
        registration_fee: int = 0,
) -> dict[str, Any]:
    """Build the settlement report from the orchestrator's sealed
    decision chain (whitepaper-aligned batch schema).

    Every ledger route record becomes one credit entry for its routed
    top-1 arm's PAYOUT address (``payout_of(arm_id)``, default
    ``address_of(arm_id)``), priced at ``price_per_query`` credits
    with the contract's 97.5% pool split, unless the payer IS the
    payout address (self-payment: the contract would skip it).
    Entries stay in query order and are grouped into batches of at
    most MAX_BATCH; each entry carries the registration's artifactId
    (the contract's new schema). The report also carries the unified
    registrations (operator key + payout address + price per unit) so
    the cross-language post script can register them on-chain."""
    ledger_dict = orchestrator.ledger.to_dict()
    arms = orchestrator.router._arms

    def payout_of_arm(arm_id: str) -> str:
        if payout_of is not None:
            override = payout_of(arm_id)
            if override is not None:
                return str(override)
        return str(arms.get(arm_id, {}).get("payout_address")
                   or address_of(arm_id))

    def sealed_claim_of_arm(arm_id: str) -> str:
        """The registration's sealed claim as a 32-byte hex value
        (the contract's bytes32 field): an already-hex claim passes
        through; anything else (evidence path, sealed source) is
        content-hashed deterministically."""
        claim = str(arms.get(arm_id, {}).get("sealed_claim")
                    or arms.get(arm_id, {}).get("replay_hash", "")
                    or arms.get(arm_id, {}).get("sealed_source", "")
                    or "")
        if (claim.startswith("0x") and len(claim) == 66
                and all(c in "0123456789abcdefABCDEF"
                        for c in claim[2:])):
            return claim
        return "0x" + hashlib.sha256(
            claim.encode("utf-8")).hexdigest()

    def proof_hash_of_arm(arm_id: str) -> str:
        """The per-answer computation-proof hash carried by this
        arm's credits (whitepaper: an answer carries a proof of the
        computation behind it). Arms with a registered proof hash
        carry it; arms whose exact component is not yet proved carry
        the sealed claim instead — the paper's own honest boundary."""
        registered = arms.get(arm_id, {}).get("proof_hash")
        if registered and str(registered).startswith("0x") \
                and len(str(registered)) == 66:
            return str(registered)
        return sealed_claim_of_arm(arm_id)

    registrations: list[dict[str, Any]] = []
    for arm_id in sorted(arms):
        registrations.append({
            "artifactId": artifact_id_of(arm_id),
            "operator": str(arms[arm_id].get("operator_key")
                            or address_of(arm_id)),
            "payoutAddress": payout_of_arm(arm_id),
            "pricePerUnit": int(price_per_query),
            "sealedClaim": sealed_claim_of_arm(arm_id),
        })

    entries: list[tuple[str, str, str, int, str]] = []  # payer, id, who, amt, proof
    skipped: list[dict[str, Any]] = []
    deposits: dict[str, int] = {}

    for rec in _route_records(ledger_dict):
        query_id = str(rec["query_id"])
        payer = payer_of(query_id)
        fee = int(price_per_query)
        pool_part, _dev = deposit_split(fee)
        deposits[payer] = deposits.get(payer, 0) + fee
        chosen = rec.get("chosen") or []
        if not chosen:
            continue
        arm_id = str(chosen[0]["arm_id"])
        who = payout_of_arm(arm_id)
        artifact_id = artifact_id_of(arm_id)
        proof_hash = proof_hash_of_arm(arm_id)
        if payer == who:
            # C1: the self-payment exclusion keys on the payout address
            skipped.append({"payer": payer, "artifactId": artifact_id,
                            "who": who, "amount": pool_part,
                            "reason": "self-payment"})
            continue
        entries.append((payer, artifact_id, who, pool_part, proof_hash))

    batches: list[dict[str, Any]] = []
    for start in range(0, len(entries), MAX_BATCH):
        chunk = entries[start:start + MAX_BATCH]
        batch_entries = [{"artifactId": artifact_id, "who": who,
                          "amount": amount, "proofHash": proof_hash}
                         for _p, artifact_id, who, amount, proof_hash
                         in chunk]
        # the whitepaper's "hash of the proofs of the computations it
        # pays for": a digest over the batch's per-entry proof hashes,
        # anchored on-chain with the batch by the post gate.
        batch_proof_hash = "0x" + payload_hash(
            json.dumps([e["proofHash"] for e in batch_entries],
                       sort_keys=True))
        batches.append({
            "payers": [payer for payer, _i, _w, _a, _h in chunk],
            "entries": batch_entries,
            "proof_hash": batch_proof_hash,
        })

    expected_credits: dict[str, int] = {}
    for _p, _i, who, amount, _h in entries:
        expected_credits[who] = expected_credits.get(who, 0) + amount

    pool_expected = sum(deposit_split(amount)[0]
                        for amount in deposits.values())
    anchor = {
        "ledger_tip": orchestrator.ledger.tip(),
        "record_count": ledger_dict["record_count"],
        "last_record_hash": (ledger_dict["records"][-1]["hash"]
                             if ledger_dict["records"]
                             else orchestrator.ledger.anchor_spec()
                             ["values"]["last_record_hash"]),
    }
    report: dict[str, Any] = {
        "milestone": "M212",
        "anchor": anchor,
        "registration_fee": int(registration_fee),
        "registrations": registrations,
        "deposits": [{"payer": payer, "amount": deposits[payer]}
                     for payer in sorted(deposits)],
        "pool_expected": pool_expected,
        "batches": batches,
        "expected": {"credits": expected_credits,
                     "skipped": skipped},
    }
    report["batch_hash"] = content_hash_of(report)
    return report


def content_hash_of(report: dict[str, Any]) -> str:
    """The settlement report's content hash: everything except the
    hash itself and timing fields (the standing reproducibility
    rule)."""
    stripped = {k: v for k, v in report.items()
                if k not in TIMING_FIELDS and k != "batch_hash"}
    return payload_hash(json.dumps(stripped, sort_keys=True,
                                   ensure_ascii=True,
                                   separators=(",", ":")))


def recompute_batch_hash(report: dict[str, Any]) -> str:
    return content_hash_of(report)


def verify_batch_rules(report: dict[str, Any],
                       pool: int | None = None) -> list[str]:
    """Gate: the built batches must be exactly what the reworked
    contract accepts (artifactId-based entries; skip-and-emit for
    malformed credits). Violations are returned; an empty list means
    the payload would post without revert and without a silent
    skip."""
    violations: list[str] = []
    batches = report.get("batches", [])
    if not batches:
        violations.append("no batches built")
    for i, batch in enumerate(batches):
        n = len(batch["payers"])
        if len(batch["entries"]) != n:
            violations.append(f"batch {i}: payer/entry shape mismatch")
        if n == 0:
            violations.append(f"batch {i}: empty")
        if n > MAX_BATCH:
            violations.append(f"batch {i}: {n} entries > MAX_BATCH "
                              f"{MAX_BATCH}")
        for j, entry in enumerate(batch["entries"]):
            if not isinstance(entry.get("artifactId"), str):
                violations.append(f"batch {i} entry {j}: missing "
                                  "artifactId")
            if not isinstance(entry.get("who"), str):
                violations.append(f"batch {i} entry {j}: missing who")
            proof_hash = entry.get("proofHash")
            if not (isinstance(proof_hash, str)
                    and proof_hash.startswith("0x")
                    and len(proof_hash) == 66):
                violations.append(f"batch {i} entry {j}: proofHash "
                                  "must be a 32-byte hex string")
            amount = entry.get("amount")
            if not isinstance(amount, int) or amount <= 0:
                violations.append(f"batch {i} entry {j}: amount must "
                                  f"be a positive integer, got "
                                  f"{amount!r}")
    if pool is not None:
        remaining = int(pool)
        for batch in batches:
            for entry in batch["entries"]:
                if entry["amount"] > remaining:
                    violations.append(f"entry amount {entry['amount']} "
                                      f"exceeds remaining pool "
                                      f"{remaining} (would revert)")
                remaining -= entry["amount"]
    if report.get("pool_expected") is not None:
        credited = sum(e["amount"] for b in batches
                       for e in b["entries"])
        if credited > report["pool_expected"]:
            violations.append(f"total credited {credited} exceeds "
                              f"pool_expected "
                              f"{report['pool_expected']}")
    if report.get("batch_hash") != recompute_batch_hash(report):
        violations.append("batch_hash does not recompute from content")
    return violations


def payer_fees(report: dict[str, Any]) -> dict[str, int]:
    """Per-payer deposit amounts for the cross-language post script."""
    return {d["payer"]: int(d["amount"]) for d in report["deposits"]}
