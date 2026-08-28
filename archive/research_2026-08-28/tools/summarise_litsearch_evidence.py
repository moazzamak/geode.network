"""Summarise a litsearch evidence.json into a per-family found-work listing.

Read-only: prints, for each family (anchors, probes, claims), every successful
result's top titles so a human can adjudicate claim status. Failed queries are
counted, never listed as findings.
"""

from __future__ import annotations

import json
import sys
from collections import Counter


def _brief(record: dict) -> str:
    title = record.get("title") or "<untitled>"
    year = record.get("year") or ""
    venue = record.get("venue") or ""
    bits = [t for t in (title, str(year), str(venue)) if t]
    return " | ".join(bits)[:140]


def main(path: str) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    payload = json.load(open(path, encoding="utf-8"))
    print(f"evidence: {path}")
    print(f"generated_at: {payload.get('generated_at')}")
    print(f"milestone: {payload.get('milestone')}")
    failures = Counter()

    def show(results, label):
        for result in results:
            if result.get("error"):
                failures[result["source"]] += 1
                continue
            hits = result.get("hits", 0)
            if hits == 0:
                continue
            print(f"\n[{label}] {result['source']} '{result['query']}' -> {hits} hits")
            for record in result.get("records", [])[:8]:
                print("   -", _brief(record))

    print("\n=== ANCHORS ===")
    show(payload.get("anchors", {}).get("results", []), "anchor")
    print("\n=== PROBES ===")
    for probe in payload.get("recall_probes", {}).get("probes", []):
        print(f"\n--- {probe['id']}: must retrieve '{probe['must_retrieve']}'")
        show(probe.get("results", []), f"probe {probe['id']}")
    print("\n=== CLAIMS ===")
    for family, claim in payload.get("claims", {}).items():
        print(f"\n=== {family}: {claim['statement'][:160]}")
        show(claim.get("results", []), family)

    print("\n=== FAILURES (recorded, never empty) ===")
    for source, count in failures.most_common():
        print(f"  {source}: {count}")
    print(f"total failed queries: {sum(failures.values())}")


if __name__ == "__main__":
    main(sys.argv[1])
