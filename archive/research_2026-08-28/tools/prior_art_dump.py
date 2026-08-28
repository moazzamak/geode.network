"""Print summaries + links for selected prior-art entries (read-only)."""
import json
from pathlib import Path

path = (Path(__file__).resolve().parents[1] / "logs" / "results"
        / "prior_art_search_2026-08-24" / "arxiv_sweep_run2.json")
data = json.loads(path.read_text(encoding="utf-8"))
WANT = [
    "Dropbear", "Golden Grain", "FL-Market", "SAKSHI", "HadAgent",
    "Token Inflation", "Bitcoin in Decentralized Artificial",
    "TOPLOC", "opML: Optimistic", "opp/ai", "SVIP", "ezDPS",
    "TensorCommitments", "DeServe", "Parallax", "POKT", "IOTA",
    "Permissionless Distributed Learning", "PredictChain",
    "Common Risk Factors", "Peer-to-Peer Intelligence Market",
]
seen = set()
for q in data["queries"]:
    for e in q["entries"]:
        for w in WANT:
            if w.lower() in e["title"].lower() and e["title"] not in seen:
                seen.add(e["title"])
                print(f"TITLE: {e['title']}")
                print(f"  {e['published']}  {e['link']}")
                print(f"  {e['summary'][:350]}")
                print()
