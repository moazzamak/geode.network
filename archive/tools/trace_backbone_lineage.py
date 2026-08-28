"""Trace a configuration lineage to the backbone and extraction batch size that produced it.

Defect B (batch-dependent INT8 activation scales) only reaches an experiment if that
experiment's features were produced by a quantized backbone running at batch size
greater than one. This tool walks the ``path`` references of a configuration tree and
reports every backbone token and batch-size declaration it finds, so exposure can be
decided from the recorded lineage rather than from recollection.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BACKBONE_PATTERN = re.compile(
    r"(dinov2[\w./-]*|siglip[\w./-]*|ijepa[\w./-]*|mobilenet[\w./-]*"
    r"|model_int8[\w./-]*|[\w./-]*quantized[\w./-]*)",
    re.IGNORECASE,
)
BATCH_PATTERN = re.compile(r"\"(\w*batch_size)\":\s*(\d+)")
PATH_PATTERN = re.compile(r"\"path\":\s*\"([^\"]+\.json)\"")


def walk(entry: str) -> None:
    seen: set[str] = set()
    stack: list[str] = [entry]
    backbones: dict[str, set[str]] = {}
    batches: dict[str, set[str]] = {}

    while stack:
        relative = stack.pop()
        if relative in seen:
            continue
        candidate = ROOT / relative
        if not candidate.is_file():
            continue
        seen.add(relative)
        text = candidate.read_text(encoding="utf-8")

        found = {match.group(1) for match in BACKBONE_PATTERN.finditer(text)}
        if found:
            backbones[relative] = found
        sizes = {
            f"{match.group(1)}={match.group(2)}"
            for match in BATCH_PATTERN.finditer(text)
        }
        if sizes:
            batches[relative] = sizes

        for match in PATH_PATTERN.finditer(text):
            stack.append(match.group(1))

    print(f"configurations walked: {len(seen)}")
    for name in sorted(seen):
        print(f"  {name}")
    print("\nbackbone tokens:")
    for name in sorted(backbones):
        for token in sorted(backbones[name]):
            print(f"  {name}: {token}")
    if not backbones:
        print("  (none found in configuration tree)")
    print("\nbatch size declarations:")
    for name in sorted(batches):
        for size in sorted(batches[name]):
            print(f"  {name}: {size}")
    if not batches:
        print("  (none declared)")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: trace_backbone_lineage.py <config-relative-path>")
    walk(sys.argv[1])


if __name__ == "__main__":
    main()
