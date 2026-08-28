"""Locate sealed sources for E4 (closed-form depth flat) and E5 (trained head
vs closed-form head on sparse codes). Read-only."""
from __future__ import annotations

import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
root = pathlib.Path("logs/results")

print("=== v15 evidence files ===")
for path in sorted((root / "v15").rglob("evidence.json")):
    print(" ", path)

print("\n=== files mentioning trained head / linear probe / SGD ===")
for path in sorted(root.rglob("evidence.json")):
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        continue
    for marker in ("trained_head", "trained head", "linear probe", "sgd", "SGD"):
        if marker in text:
            print(" ", path, "<-", marker)
            break

print("\n=== files mentioning closed-form depth / layers ===")
for path in sorted(root.rglob("evidence.json")):
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        continue
    for marker in ("layers", "depth", "stacked"):
        if marker in text:
            print(" ", path, "<-", marker)
            break

print("\n=== m136 ridge ladder (E6 penalty flatness) ===")
m136 = root / "v16" / "m136_margin_head" / "evidence.json"
ev = json.loads(m136.read_text(encoding="utf-8"))
print(json.dumps(ev.get("ridge_ladder"), default=str)[:600])
print("hinge:", json.dumps(ev.get("hinge"), default=str)[:200])
print("gates:", json.dumps(ev.get("gates"), default=str)[:300])
