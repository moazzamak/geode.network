"""GEODE — Generalized Encoders for Open-Domain Expertise: the product
package.

Architecture (see ``docs/ARCHITECTURE.md``): layered packages with a
one-way dependency direction —

    geode.audit        replay / provenance / erasure   (lowest layer)
    geode.hashing      canonical JSON + payload hash
    geode.core         domain model, admission, routing, orchestration
    geode.attribution  coalition games, incentives, pricing
    geode.settlement   the CreditLedger attribution wire
    geode.privacy      secret sharing and zero-knowledge arguments

``experiments`` depends on ``geode``; ``geode`` never imports
``experiments``. This module is the curated public API surface; deeper
imports go through the subpackages.

Built on the sealed discipline: deterministic serialisation, payload
hashes on everything frozen, and no wall-clock fields inside content
hashes.
"""
from geode.audit import AuditAPI, TIMING_FIELDS  # noqa: F401
from geode.attribution import (  # noqa: F401
    Demerit,
    MeasurementClass,
    beta_shapley,
    capped_session_value,
    capture_window_value,
    capture_worth_budget,
    free_rider_report,
    leave_one_out,
    minimum_bond,
    safety_adjusted_value,
    shapley,
    stake_schedule,
    trust_weight,
    trust_weighted_shares,
)
from geode.core import (  # noqa: F401
    AdmissionRegistry,
    AnchorSpec,
    AppendOnlyLedger,
    BehaviorDiffGate,
    ConstraintRegistry,
    DriftGate,
    FreezeError,
    FreezeRegistry,
    OodGate,
    Orchestrator,
    OverrideLedger,
    ProbeSuite,
    Prohibition,
    RefusalCapability,
    Router,
    VerifierRotation,
    anchor_from_ledger,
    arm_from_admission,
    arm_from_sealed_head,
    augment_measured_tags,
    median_vector,
    quorum,
    refusal_admission,
    refusal_measured_tag,
    validate_arm_spec,
    verify_anchor_entry,
)
from geode.core.economics import (  # noqa: F401
    DEV_FUND_BPS,
    MAX_BATCH,
    PRIMITIVE_UNIT,
    REGISTRATION_FIELDS,
    SLASH_LADDER,
    UNIT_TABLE,
    VESTING_EPOCHS,
    address_of,
    artifact_id_of,
    deposit_split,
    served_units,
    unit_of_work,
    within_cap,
)
from geode.hashing import canonical_json, payload_hash  # noqa: F401
from geode.privacy import (  # noqa: F401
    prove,
    recombine_additive,
    shamir_reconstruct,
    shamir_split,
    split_additive,
    verify,
)
from geode.settlement import build_credit_batches, verify_batch_rules  # noqa: F401
from geode.settlement import SlashLedger  # noqa: F401

__version__ = "0.16.0"

__all__ = [
    "AdmissionRegistry",
    "AnchorSpec",
    "AppendOnlyLedger",
    "AuditAPI",
    "BehaviorDiffGate",
    "ConstraintRegistry",
    "DEV_FUND_BPS",
    "Demerit",
    "DriftGate",
    "FreezeError",
    "FreezeRegistry",
    "MAX_BATCH",
    "MeasurementClass",
    "OodGate",
    "Orchestrator",
    "OverrideLedger",
    "PRIMITIVE_UNIT",
    "ProbeSuite",
    "Prohibition",
    "REGISTRATION_FIELDS",
    "RefusalCapability",
    "Router",
    "SLASH_LADDER",
    "SlashLedger",
    "TIMING_FIELDS",
    "UNIT_TABLE",
    "VESTING_EPOCHS",
    "VerifierRotation",
    "address_of",
    "anchor_from_ledger",
    "arm_from_admission",
    "arm_from_sealed_head",
    "artifact_id_of",
    "augment_measured_tags",
    "beta_shapley",
    "build_credit_batches",
    "canonical_json",
    "capped_session_value",
    "capture_window_value",
    "capture_worth_budget",
    "deposit_split",
    "free_rider_report",
    "leave_one_out",
    "median_vector",
    "minimum_bond",
    "payload_hash",
    "prove",
    "quorum",
    "recombine_additive",
    "refusal_admission",
    "refusal_measured_tag",
    "safety_adjusted_value",
    "served_units",
    "shamir_reconstruct",
    "shamir_split",
    "shapley",
    "split_additive",
    "stake_schedule",
    "trust_weight",
    "trust_weighted_shares",
    "unit_of_work",
    "validate_arm_spec",
    "verify",
    "verify_anchor_entry",
    "verify_batch_rules",
    "within_cap",
]
