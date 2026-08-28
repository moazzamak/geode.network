"""Fresh prior-art sweep for v18 exploitation planning (8 Aug 2026).

Registered BEFORE searching (per the program's recall-testing rule):
- every query string is fixed below,
- anchors (papers known to exist) must be found or the search is broken,
- a rate-limit residual is recorded separately from a genuinely empty result,
- absence from this index never licenses a novelty claim.

Run::

    .\\.venv-rocm\\Scripts\\python.exe tools\\arxiv_sweep_v19.py
"""
from __future__ import annotations

import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

BASE = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}

# ---------------------------------------------------------------------------
# registered queries
# ---------------------------------------------------------------------------
ANCHORS: dict[str, str] = {
    "Bordelon spectrum learning curves": 'ti:"Spectrum dependent learning curves"',
    "Canatar kernel regression stats mech": 'ti:"Statistical mechanics of generalization in kernel regression"',
    "Kaplan scaling laws": 'ti:"Scaling Laws for Neural Language Models"',
    "Hoffmann chinchilla": 'ti:"Training Compute-Optimal Large Language Models"',
    "Mei-Montanari random features": 'ti:"The generalization error of random features regression"',
    "Gong ITQ": 'ti:"Iterative Quantization: A Procrustean Approach to Learning Binary Codes"',
    "Fedus switch transformers": 'ti:"Switch Transformers: Scaling to Trillion Parameter Models"',
    "Clark unified scaling routed": 'ti:"Unified Scaling Laws for Routed Language Models"',
}

TOPICS: dict[str, str] = {
    "kernel-ridge learning curves": 'abs:"learning curve" AND abs:"kernel ridge regression"',
    "kernel regression eigenspectrum": 'abs:"kernel regression" AND abs:"eigenvalue"',
    "random features ridge": 'abs:"random features" AND abs:"ridge regression"',
    "random features scaling": 'abs:"random features" AND abs:"scaling"',
    "neural scaling laws": 'abs:"scaling laws" AND abs:"neural networks"',
    "data scaling": 'abs:"data scaling" AND abs:"neural network"',
    "linear probing scaling laws": 'abs:"linear probing" AND abs:"scaling laws"',
    "linear evaluation transfer": 'abs:"linear evaluation" AND abs:"transfer"',
    "frozen features scaling": 'abs:"frozen features" AND abs:"scaling"',
    "mixture of experts scaling": 'abs:"mixture of experts" AND abs:"scaling laws"',
    "domain experts MoE": 'abs:"domain expert" AND abs:"mixture of experts"',
    "binary hashing image": 'abs:"binary hashing" AND abs:"image"',
    "dictionary learning scaling": 'abs:"dictionary learning" AND abs:"scaling"',
    "sparse coding sample complexity": 'abs:"sparse coding" AND abs:"sample complexity"',
    "feature eigenspectrum transfer": 'abs:"feature eigenspectrum" AND abs:"transfer"',
    "NTK transfer learning": 'abs:"neural tangent kernel" AND abs:"transfer learning"',
    "kernel ridge scaling law": 'abs:"kernel ridge" AND abs:"scaling law"',
}


def query(q: str, max_results: int = 6) -> tuple[list[dict], str | None]:
    params = urllib.parse.urlencode({
        "search_query": q, "start": 0, "max_results": max_results,
        "sortBy": "relevance",
    })
    url = f"{BASE}?{params}"
    for attempt in range(2):
        try:
            with urllib.request.urlopen(url, timeout=40) as resp:
                body = resp.read()
            root = ET.fromstring(body)
            entries = []
            for e in root.findall("a:entry", NS):
                title = " ".join(e.findtext("a:title", "", NS).split())
                ident = e.findtext("a:id", "", NS).split("/abs/")[-1]
                published = e.findtext("a:published", "", NS)
                entries.append({"id": ident, "title": title,
                                "published": published[:10]})
            return entries, None
        except Exception as exc:  # noqa: BLE001 - retry on transient failure
            last = exc
            time.sleep(12 + 12 * attempt)
    return [], f"ERROR after retries: {last}"


def main() -> None:
    print("=" * 78)
    print("ANCHORS (must be found or the search is broken)")
    print("=" * 78)
    anchor_fail = []
    for name, q in ANCHORS.items():
        entries, err = query(q)
        ids = [e["id"] for e in entries]
        found = any(e["id"].startswith(("1906.11320", "2105.03739", "2001.08361",
                                        "2203.15556", "1908.05355", "1203.1581",
                                        "2101.03961", "2202.01169")) for e in entries)
        if not found:
            anchor_fail.append(name)
        print(f"\n[{name}] {'FOUND' if found else 'MISS'} (err={err})")
        for e in entries[:3]:
            print(f"   {e['id']}  {e['title'][:80]}")
        time.sleep(12)

    print()
    print("=" * 78)
    print("TOPIC QUERIES (recall test: which known work do these find?)")
    print("=" * 78)
    empty = []
    for name, q in TOPICS.items():
        entries, err = query(q)
        if err:
            print(f"\n[{name}] RESIDUAL-ERROR: {err}  (record as non-empty!)")
            empty.append((name, err))
        elif not entries:
            print(f"\n[{name}] 0 hits  (genuinely empty or insensitive)")
            empty.append((name, "0 hits"))
        else:
            print(f"\n[{name}] {len(entries)} hits")
            for e in entries[:5]:
                print(f"   {e['id']}  {e['published']}  {e['title'][:72]}")
        time.sleep(12)

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print("anchor failures:", anchor_fail if anchor_fail else "none")
    print("empty-or-error topics:", empty)


if __name__ == "__main__":
    main()
