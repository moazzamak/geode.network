"""Auditability: deterministic replay, provenance, and concept erasure
with certificates.

Layering rule: imports only ``geode.hashing`` and standard libraries
(the audit layer is the lowest product layer).
"""
from geode.audit.audit import (
    TIMING_FIELDS,
    AuditAPI,
    ProvenanceReport,
    ReplayReport,
    evidence_content_hash,
    sha256_file,
)
from geode.audit.erasure import AffineMap, erasure_certificate, leace_eraser

__all__ = [
    "TIMING_FIELDS",
    "AffineMap",
    "AuditAPI",
    "ProvenanceReport",
    "ReplayReport",
    "erasure_certificate",
    "evidence_content_hash",
    "leace_eraser",
    "sha256_file",
]
