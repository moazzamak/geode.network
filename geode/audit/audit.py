"""GEODE audit API v0 (v25 M177) — L0 deterministic replay + L1 provenance.

Registered contract in ``analysis/v25_audit_ladder_spec.md`` and
``analysis/RESEARCH_IMPLEMENTATION_PLAN_v25.md`` section 6 (18 Aug 2026).

Deterministic by construction: no wall clocks in outputs; timing fields
are excluded from every content hash by the registered set (never
amended after a mismatch). Replays run into a scratch directory; sealed
evidence is never overwritten.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from geode.hashing import payload_hash

# The registered timing-field exclusion set (the standing
# reproducibility-hash rule: wall-clock fields never enter a content hash).
TIMING_FIELDS: frozenset[str] = frozenset({"runtime_seconds"})


def _json_shape(value: Any) -> Any:
    """The JSON round-trip shape of a value (int dict keys -> strings).

    Registered rule 5: BOTH sides of a replay comparison are compared in
    their JSON-normalized shape — an in-memory dict with int keys is a
    DIFFERENT object from its JSON round-trip, even though the stored
    evidence is identical."""
    return json.loads(json.dumps(value, sort_keys=True,
                                 ensure_ascii=True,
                                 separators=(",", ":")))


def evidence_content_hash(evidence: dict[str, Any]) -> str:
    """Content hash of an evidence dict with the registered timing
    fields excluded, serialized canonically (sorted keys, compact)."""
    stripped = {k: v for k, v in evidence.items()
                if k not in TIMING_FIELDS}
    return payload_hash(json.dumps(
        _json_shape(stripped), sort_keys=True, ensure_ascii=True,
        separators=(",", ":")))


def _strip_timing(evidence: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in evidence.items() if k not in TIMING_FIELDS}


class ReplayReport:
    def __init__(self, evidence_hash: str, replayed_hash: str,
                 bit_exact: bool, equal_excluding_timing: bool,
                 excluded_fields: list[str],
                 diffs: list[str]):
        self.evidence_hash = evidence_hash
        self.replayed_hash = replayed_hash
        self.bit_exact = bit_exact
        self.equal_excluding_timing = equal_excluding_timing
        self.excluded_fields = excluded_fields
        self.diffs = diffs

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_hash": self.evidence_hash,
            "replayed_hash": self.replayed_hash,
            "bit_exact": self.bit_exact,
            "equal_excluding_timing": self.equal_excluding_timing,
            "excluded_fields": self.excluded_fields,
            "diffs": self.diffs,
        }


class ProvenanceReport:
    def __init__(self, artifact_dir: Path, chain: dict[str, Any],
                 gaps: list[str]):
        self.artifact_dir = artifact_dir
        self.chain = chain
        self.gaps = gaps

    def to_dict(self) -> dict[str, Any]:
        return {"artifact_dir": str(self.artifact_dir),
                "chain": self.chain, "gaps": self.gaps}


class AuditAPI:
    """v0: L0 replay + L1 provenance. Deterministic, read-only."""

    def replay(self, runner: Callable[[Path, Path], dict[str, Any]],
               config_path: Path, evidence_path: Path,
               scratch_dir: Path) -> ReplayReport:
        """Re-run the milestone's own runner into a scratch directory and
        compare against the sealed evidence.

        ``runner`` = the milestone runner function (same signature as
        the eval_v24 runners); ``config_path`` = the sealed config;
        ``evidence_path`` = the sealed evidence.json; ``scratch_dir`` =
        a fresh directory (created; sealed dirs must not be passed)."""
        scratch_dir = Path(scratch_dir)
        if scratch_dir.exists() and any(scratch_dir.iterdir()):
            raise ValueError(f"scratch dir not empty: {scratch_dir}")
        scratch_dir.mkdir(parents=True, exist_ok=True)
        evidence = json.loads(Path(evidence_path).read_text(
            encoding="utf-8"))
        replayed = runner(Path(config_path), scratch_dir / "replay_out")
        evidence_hash = evidence_content_hash(evidence)
        replayed_hash = evidence_content_hash(replayed)
        diffs: list[str] = []
        stripped_sealed = _json_shape(_strip_timing(evidence))
        stripped_replay = _json_shape(_strip_timing(replayed))
        if stripped_sealed != stripped_replay:
            keys = sorted(set(stripped_sealed) | set(stripped_replay))
            diffs = [k for k in keys
                     if stripped_sealed.get(k) != stripped_replay.get(k)]
        return ReplayReport(
            evidence_hash=evidence_hash,
            replayed_hash=replayed_hash,
            bit_exact=evidence_hash == replayed_hash,
            equal_excluding_timing=not diffs,
            excluded_fields=sorted(TIMING_FIELDS),
            diffs=diffs,
        )

    def provenance(self, artifact_dir: Path) -> ProvenanceReport:
        """L1: resolve the chain data -> code -> weights -> behavior for an
        artifact directory (evidence.json + artifact_index.json)."""
        artifact_dir = Path(artifact_dir)
        evidence_path = artifact_dir / "evidence.json"
        index_path = artifact_dir / "artifact_index.json"
        if not evidence_path.exists():
            raise FileNotFoundError(f"no evidence.json in {artifact_dir}")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        index = json.loads(index_path.read_text(encoding="utf-8")) \
            if index_path.exists() else {}
        gaps: list[str] = []
        chain: dict[str, Any] = {
            "data": evidence.get("corpus", {}),
            "configuration": {
                "config_file": evidence.get("config_file"),
                "configuration_hash": evidence.get("configuration_hash"),
            },
            "code": "recorded-as-milestone-runner (replayable via L0)",
            "weights": "recorded-as-behavior (frozen heads re-fit exactly)",
            "behavior": {
                "evidence_content_hash": evidence_content_hash(evidence),
                "artifact_index": index,
            },
            "fit": evidence.get("head") or evidence.get("fit_and_report")
                   or evidence.get("transfer") or {},
        }
        if not evidence.get("configuration_hash"):
            gaps.append("configuration_hash missing from evidence")
        if not index:
            gaps.append("no artifact_index.json in the artifact directory")
        return ProvenanceReport(artifact_dir, chain, gaps)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
