"""GEODE authored adversarial-router probe suite (v25 M283) — the
adversarial-router release-blocker cell, the M249 pattern applied to
the ROUTER instead of the arms.

Loads the authored probe artifact
(``analysis/router_probes_v0.json``); every probe's commit-reveal
hashes must match its content (tampered content fails by
construction, dual authorship per the M252 rule). A router function
is evaluated deterministically: per-category and overall misroute
rates against the registered bar (0.10).

Deterministic: no RNG, no wall clocks. The real embedding router is
measured in the M283 evidence run; this module ships the authored
suite, the integrity check, and the evaluation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
PROBE_ARTIFACT_PATH = REPO_ROOT / "analysis" / "router_probes_v0.json"


def _commit_hash(author: str, salt: str, probe: dict[str, Any]) -> str:
    material = f"{author}|{salt}|" + json.dumps(
        probe, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class RouterProbeSuite:
    """The authored adversarial-router suite with commit
    verification and deterministic evaluation."""

    def __init__(self, artifact_path: Path | None = None):
        path = artifact_path or PROBE_ARTIFACT_PATH
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.version = str(raw.get("version", "?"))
        self.axis = str(raw.get("axis", "?"))
        self.min_authors = int(raw.get("min_authors", 2))
        self.bar = float(raw.get("bar", 0.10))
        self.probes: dict[str, dict[str, Any]] = {}
        for entry in raw.get("probes", []):
            pid = str(entry["id"])
            if pid in self.probes:
                raise ValueError(f"duplicate probe id {pid!r}")
            self.probes[pid] = entry

    def verify_integrity(self) -> dict[str, Any]:
        """Every probe must carry at least min_authors commit-reveal
        hashes that match its content (tamper detection)."""
        broken: list[str] = []
        for pid, probe in self.probes.items():
            commits = probe.get("commits") or []
            ok = 0
            for c in commits:
                expected = c.get("hash", "")
                actual = _commit_hash(
                    c.get("author", ""), c.get("salt", ""),
                    {k: v for k, v in probe.items()
                     if k not in ("id", "commits")})
                if expected != actual:
                    broken.append(pid)
                    break
                ok += 1
            if ok < self.min_authors:
                broken.append(pid)
        return {"ok": not broken, "broken_probes": sorted(set(broken)),
                "probe_count": len(self.probes)}

    def categories(self) -> list[str]:
        return sorted({p["category"] for p in self.probes.values()})

    def evaluate(self, route_fn: Callable[[str], str]
                 ) -> dict[str, Any]:
        """Route every probe and record the misroute rates. Returns
        per-category and overall rates plus the bar check. A route
        outside the three registered families counts as a misroute
        (the router must answer, deterministically)."""
        per_category: dict[str, list[bool]] = {
            c: [] for c in self.categories()}
        per_probe: dict[str, dict[str, Any]] = {}
        for pid, probe in self.probes.items():
            route = route_fn(probe["text"])
            ok = route == probe["expected_family"]
            per_category[probe["category"]].append(ok)
            per_probe[pid] = {
                "category": probe["category"],
                "expected_family": probe["expected_family"],
                "routed_family": route,
                "misroute": not ok,
            }
        rates: dict[str, float] = {}
        for cat, outcomes in per_category.items():
            rates[cat] = (round(1.0 - sum(outcomes) / len(outcomes), 4)
                          if outcomes else 0.0)
        overall = round(
            sum(rates[c] * len(per_category[c])
                for c in per_category)
            / sum(len(v) for v in per_category.values()), 4) \
            if self.probes else 0.0
        within_bar = all(rates[c] <= self.bar for c in rates) \
            and overall <= self.bar
        return {
            "n_probes": len(self.probes),
            "bar": self.bar,
            "category_misroute_rates": rates,
            "overall_misroute_rate": overall,
            "within_bar": within_bar,
            "per_probe": per_probe,
        }
