"""Scan the DomainNet shards for out-of-set availability, metadata only.

Reads only ``label`` and ``domain`` — never the image bytes — so the whole
17 GB corpus can be surveyed cheaply. The short-edge filter needs an image
header and is therefore *not* applied here; this is an upper bound used to
decide whether an M83 out-of-set artifact can be domain-stratified at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

import pyarrow.parquet as parquet

REPO_ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]


def main() -> None:
    config = json.loads(
        (REPO_ROOT / "experiments/configs/v13/domainnet_large.json").read_text(
            encoding="utf-8"
        )
    )
    record = json.loads(
        (REPO_ROOT / config["domainnet_download_record"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    repository = Path(record["dataset_root"]) / "repository"
    files = [
        repository / path for path in record["verified_files"] if "train-" in path
    ]
    print(f"{len(files)} training shards")

    per_class: Counter[int] = Counter()
    per_class_domain: Counter[tuple[int, int]] = Counter()
    for path in files:
        source = parquet.ParquetFile(path)
        for batch in source.iter_batches(batch_size=8192, columns=["label", "domain"]):
            labels = batch.column("label").to_pylist()
            domains = batch.column("domain").to_pylist()
            for label, domain in zip(labels, domains, strict=True):
                per_class[int(label)] += 1
                per_class_domain[(int(label), int(domain))] += 1
        print(f"  scanned {path.name}", flush=True)

    unseen = [c for c in sorted(per_class) if c >= 128]
    known = [c for c in sorted(per_class) if c < 128]
    print(f"\nclasses present: {len(per_class)}  known {len(known)}  unseen {len(unseen)}")

    unseen_totals = [per_class[c] for c in unseen]
    unseen_totals.sort()
    print(
        f"unseen-class row counts: min {unseen_totals[0]} "
        f"p10 {unseen_totals[len(unseen_totals) // 10]} "
        f"median {unseen_totals[len(unseen_totals) // 2]} "
        f"max {unseen_totals[-1]}"
    )

    print("\nunseen-class rows by domain (pre-filter):")
    total = sum(per_class_domain[(c, d)] for c in unseen for d in range(6))
    for d in range(6):
        count = sum(per_class_domain[(c, d)] for c in unseen)
        classes_with = sum(1 for c in unseen if per_class_domain[(c, d)] > 0)
        print(
            f"  {DOMAINS[d]:10s} {count:8d}  {count / total:6.2%}  "
            f"present in {classes_with}/{len(unseen)} unseen classes"
        )

    print("\nknown-class rows by domain (pre-filter):")
    ktotal = sum(per_class_domain[(c, d)] for c in known for d in range(6))
    for d in range(6):
        count = sum(per_class_domain[(c, d)] for c in known)
        print(f"  {DOMAINS[d]:10s} {count:8d}  {count / ktotal:6.2%}")

    known_totals = sorted(per_class[c] for c in known)
    print(
        f"\nknown-class row counts: min {known_totals[0]} "
        f"median {known_totals[len(known_totals) // 2]} max {known_totals[-1]}"
    )
    print("(corpus consumed 576 per known class after the 256px filter)")


if __name__ == "__main__":
    main()
