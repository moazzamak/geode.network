"""Locate sealed sources for two plan numbers: the trained-head ~15% accuracy
(E5) and the count-memory optimum window 4 (E12b). Read-only."""
from __future__ import annotations

import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

root = pathlib.Path("logs/results")

print("=== files containing a trained-head-like accuracy (~0.15x) ===")
for path in sorted(root.rglob("evidence.json")):
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        continue
    for marker in ('"accuracy": 0.15', '"accuracy_by_penalty": {"1.0": 0.15', "trained_head"):
        if marker in text:
            print(" ", path, "<-", marker)
            break

print("=== m129 evidence structure (window claims) ===")
m129 = root / "v16" / "m129_programmatic_primitives_litsearch" / "evidence.json"
if m129.exists():
    ev = json.loads(m129.read_text(encoding="utf-8"))
    print(" keys:", list(ev.keys())[:14])
    for key in ("claims", "milestone", "program"):
        print(" ", key, ":", str(ev.get(key))[:200])

print("=== m131 window sweep (all windows) ===")
m131 = root / "v16" / "m131_additive_next_token" / "evidence.json"
ev = json.loads(m131.read_text(encoding="utf-8"))
for w, info in ev.get("per_window", {}).items():
    print(
        f"  w={w}: ppl={info.get('test_perplexity')} valid={info.get('valid_perplexity')} "
        f"footprint={info.get('footprint_bytes')}"
    )

print("=== m125 head exponent (E4 depth/E6 penalty context) ===")
m125 = root / "v16" / "m125_head_exponent" / "evidence.json"
ev = json.loads(m125.read_text(encoding="utf-8"))
print(" keys:", list(ev.keys())[:14])
for key, val in ev.items():
    if key in ("admissible_as_evidence", "configuration_hash", "milestone",
               "runtime_seconds", "question", "registered_in", "source",
               "config", "config_file"):
        continue
    print(" ", key, ":", str(val)[:300])
