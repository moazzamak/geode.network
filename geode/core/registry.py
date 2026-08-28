"""The GEODE task registry (v24 capability 7, M165).

Transactional append-only storage of normalised descriptors and their
fits. Operations on one task never touch another task's entry, and
every entry's content hash excludes wall-clock fields (the registered
reproducibility rule: timing/generated_at never sits inside a content
hash).
"""
from __future__ import annotations

import json
from typing import Any

from geode.hashing import payload_hash

from geode.core.descriptor import NormalisedDescriptor, normalise


class TaskRegistry:
    """A deterministic, append-only registry keyed by descriptor hash."""

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, Any]] = {}

    def add(self, raw: dict[str, Any]) -> tuple[str, NormalisedDescriptor]:
        """Register a task; returns (task_id, normalised descriptor).

        The task_id is the first 16 hex characters of the canonical
        descriptor hash — a pure function of the content, so the same
        task registered twice yields the same id (idempotent).
        """
        desc = normalise(raw)
        task_id = desc.hash()[:16]
        if task_id not in self._entries:
            self._entries[task_id] = {
                "descriptor": desc.to_dict(),
                "fingerprint": None,          # set by M168's embedder
                "route": None,                # set by M171's router
                "fits": [],                   # append-only fit records
                "versions": [1],
            }
        return task_id, desc

    def get(self, task_id: str) -> dict[str, Any]:
        """Return a copy of an entry; KeyError if unknown."""
        return json.loads(json.dumps(self._entries[task_id]))

    def list_ids(self) -> list[str]:
        return sorted(self._entries)

    def set_fingerprint(self, task_id: str, fingerprint: Any) -> None:
        """Append a fingerprint version (the M168 contract, planned)."""
        entry = self._entries[task_id]
        entry["fingerprint"] = fingerprint
        entry["versions"].append(entry["versions"][-1] + 1)

    def record_fit(self, task_id: str, fit: dict[str, Any]) -> None:
        """Append a fit record (arm, head hash, accuracy, evidence ref)."""
        self._entries[task_id]["fits"].append(fit)

    def content_hash(self, task_id: str) -> str:
        """Hash of everything that matters, excluding volatile fields."""
        entry = self._entries[task_id]
        return payload_hash(json.dumps(
            {k: v for k, v in entry.items() if k != "versions"},
            sort_keys=True, ensure_ascii=True, separators=(",", ":")))

    def report(self, task_id: str) -> str:
        """Canonical JSON report of one task (deterministic)."""
        return json.dumps(self._entries[task_id], sort_keys=True,
                          ensure_ascii=True, separators=(",", ":"))
