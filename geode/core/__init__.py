"""The GEODE core: domain model, admission, routing, and orchestration.

Layering rule: modules in this package import only from
``geode.core``, ``geode.audit``, and ``geode.hashing`` — never from
``experiments.*`` and never from higher layers (settlement, privacy,
attribution).
"""
from geode.core.arm import (
    arm_from_admission,
    arm_from_sealed_head,
    validate_arm_spec,
)
from geode.core.anchor import (
    AnchorSpec,
    anchor_from_ledger,
    verify_anchor_entry,
)
from geode.core.artifacts import ArtifactRef, ArtifactStore, verify_artifact
from geode.core.behavior_diff import BehaviorDiffGate
from geode.core.byzantine import median_vector, quorum
from geode.core.constraints import ConstraintRegistry, Prohibition
from geode.core.dnn_admission import (
    AdmissionRegistry,
    AdmissionResult,
    DNNSubmission,
    validate_submission,
)
from geode.core.fingerprint import DriftGate
from geode.core.freeze import FreezeError, FreezeRegistry
from geode.core.audio_primitives import (  # noqa: F401
    mel_spectrogram,
    primitive_replay_hash,
)
from geode.core.auth import (  # noqa: F401
    canonical_signed_bytes,
    generate_keypair,
    sign_request,
    verify_request,
)
from geode.core.ledger import AppendOnlyLedger, record_hash
from geode.core.ood import OodGate
from geode.core.orchestrator import Orchestrator
from geode.core.override import OverrideLedger
from geode.core.probes import ProbeSuite
from geode.core.refusal import (
    RefusalCapability,
    augment_measured_tags,
    refusal_admission,
    refusal_measured_tag,
)
from geode.core.rotation import VerifierRotation
from geode.core.router import Router

__all__ = [
    "AdmissionRegistry",
    "AdmissionResult",
    "AnchorSpec",
    "AppendOnlyLedger",
    "ArtifactRef",
    "ArtifactStore",
    "BehaviorDiffGate",
    "ConstraintRegistry",
    "DNNSubmission",
    "DriftGate",
    "FreezeError",
    "FreezeRegistry",
    "OodGate",
    "Orchestrator",
    "OverrideLedger",
    "ProbeSuite",
    "Prohibition",
    "RefusalCapability",
    "Router",
    "VerifierRotation",
    "anchor_from_ledger",
    "arm_from_admission",
    "arm_from_sealed_head",
    "augment_measured_tags",
    "median_vector",
    "quorum",
    "record_hash",
    "refusal_admission",
    "refusal_measured_tag",
    "validate_arm_spec",
    "validate_submission",
    "verify_anchor_entry",
    "verify_artifact",
]
