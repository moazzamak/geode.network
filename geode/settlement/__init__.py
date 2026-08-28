"""Settlement: converting the orchestrator's sealed decision chain into
contract-submittable attribution batches (the CreditLedger wire).

Layering rule: imports only ``geode.core``, ``geode.audit``, and
``geode.hashing``.
"""
from geode.settlement.settlement import (
    BIT_DNN,
    BIT_ENCODER,
    BIT_HEAD,
    BIT_ORCH,
    MAX_BATCH,
    address_of,
    build_credit_batches,
    content_hash_of,
    deposit_split,
    mask_for,
    payer_fees,
    recompute_batch_hash,
    verify_batch_rules,
)
from geode.settlement.slashing import SlashLedger

__all__ = [
    "BIT_DNN",
    "BIT_ENCODER",
    "BIT_HEAD",
    "BIT_ORCH",
    "MAX_BATCH",
    "SlashLedger",
    "address_of",
    "build_credit_batches",
    "content_hash_of",
    "deposit_split",
    "mask_for",
    "payer_fees",
    "recompute_batch_hash",
    "verify_batch_rules",
]
