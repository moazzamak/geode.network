"""Ontology v0 loader and consistency check (M166 close-out).

Loads the frozen ``analysis/task_ontology_v0.json`` and verifies it
against the in-code axis schema — the frozen artifact and the running
code must agree, or the normaliser and the ontology have drifted.
"""
from __future__ import annotations

import json
from pathlib import Path

from geode.core.descriptor import AXES

# The frozen ontology artifact lives at the repository root (a
# repo-frozen artifact, not pip-shipped data). From this module's
# location (geode/core/) the repo root is two levels up.
ONTOLOGY_PATH = (Path(__file__).resolve().parents[2] / "analysis"
                 / "task_ontology_v0.json")


def load_ontology() -> dict:
    return json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8"))


def check_consistency() -> list[str]:
    """Return the list of inconsistencies (empty = consistent)."""
    onto = load_ontology()
    problems: list[str] = []
    if onto["axes"] != AXES:
        problems.append("axes mismatch between the frozen ontology and "
                        "geode.core.descriptor.AXES")
    for axis, spec in onto.get("continuous_axes", {}).items():
        if axis not in AXES:
            problems.append(f"continuous axis {axis} not in AXES")
    similar = onto["similarity_positive_controls"]["known_similar"]
    dissimilar = onto["similarity_positive_controls"]["known_dissimilar"]
    if {tuple(p) for p in similar} & {tuple(p) for p in dissimilar}:
        problems.append("a pair appears in both known_similar and "
                        "known_dissimilar")
    return problems
