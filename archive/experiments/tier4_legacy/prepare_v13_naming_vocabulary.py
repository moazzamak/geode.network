"""Build and seal the M82 naming vocabulary.

The plan's M82 gate and Amendment R6 both require a vocabulary registered
before any atom is named; a vocabulary assembled after seeing which words score
well is not a vocabulary, it is a result. This script emits that artifact and
its hash, and is run and committed before the naming channel exists.

Amendment R9 requires a gating positive control for the naming instrument. It
is supplied here rather than invented later: DomainNet defines 345 classes, the
v13 corpus uses 128 of them, and the remaining 217 are words of the same kind,
drawn from the same distribution, that nothing in the corpus depicts. They are
the far-field end of the instrument, in the role N1 gave far-field noise for
the detection instrument.

Two properties are asserted here rather than assumed downstream:

* the 128 in-corpus names recovered from the parquet label metadata agree
  exactly, and in order, with the class names recoverable from the corpus's own
  selection manifest, so the vocabulary and the corpus cannot silently drift
  apart;
* the absent names are genuinely absent, checked against every image path in
  the manifest rather than by trusting the label index.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments.common.v5_artifacts import (
    build_artifact_index,
    payload_hash,
    sha256_file,
    write_canonical_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments" / "configs" / "v13" / "m82_naming_vocabulary.json"
)


def _resolve_inside_repo(path: str) -> Path:
    resolved = (REPO_ROOT / Path(path)).resolve()
    if REPO_ROOT.resolve() not in resolved.parents:
        raise ValueError("M82 output paths must remain inside the repository")
    return resolved


def class_names_from_manifest(manifest_path: Path) -> list[str]:
    """Recover the corpus's own class names from its image paths.

    Every DomainNet path is ``<domain>/<class_name>/<file>``, so the corpus
    carries its class names independently of any external label list. This is
    the operand that catches a vocabulary drifting away from the corpus.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed: defaultdict[int, Counter[str]] = defaultdict(Counter)
    for row in manifest["selection"]:
        observed[int(row["class_label"])][row["image_path"].split("/")[1]] += 1

    ambiguous = {label: dict(counts) for label, counts in observed.items() if len(counts) > 1}
    if ambiguous:
        raise ValueError(f"Manifest class labels map to several names: {ambiguous}")

    missing = set(range(manifest["class_count"])) - set(observed)
    if missing:
        raise ValueError(f"Manifest is missing class labels {sorted(missing)}")

    return [observed[label].most_common(1)[0][0] for label in range(manifest["class_count"])]


def all_domainnet_names(parquet_root: Path, metadata_key: str) -> list[str]:
    """Read the full DomainNet label list from the parquet schema metadata."""
    import pyarrow.parquet as pq

    shards = sorted(parquet_root.rglob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No DomainNet parquet shards under {parquet_root}")

    metadata = pq.ParquetFile(shards[0]).schema_arrow.metadata
    if metadata is None or metadata_key.encode("utf-8") not in metadata:
        raise ValueError(f"Parquet schema carries no '{metadata_key}' metadata")

    payload = json.loads(metadata[metadata_key.encode("utf-8")].decode("utf-8"))
    names = payload["info"]["features"]["label"]["names"]
    if len(names) != len(set(names)):
        raise ValueError("DomainNet label names are not unique")
    return list(names)


def _image_path_names(manifest_path: Path) -> set[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {row["image_path"].split("/")[1] for row in manifest["selection"]}


def build_vocabulary(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = _resolve_inside_repo(config["corpus"]["manifest_path"])
    source = config["label_source"]

    corpus_names = class_names_from_manifest(manifest_path)
    every_name = all_domainnet_names(Path(source["parquet_root"]), source["metadata_key"])

    if len(every_name) != source["expected_total_names"]:
        raise ValueError(
            f"Expected {source['expected_total_names']} DomainNet names, read {len(every_name)}"
        )
    if len(corpus_names) != source["expected_in_corpus_names"]:
        raise ValueError(
            f"Expected {source['expected_in_corpus_names']} in-corpus names, read {len(corpus_names)}"
        )

    # The corpus uses labels 0..127, so its names must be the first 128 of the
    # dataset's list, in order. Checking this rather than assuming it is what
    # keeps label indices meaningful downstream.
    leading = every_name[: len(corpus_names)]
    if leading != corpus_names:
        disagreements = [
            {"label": index, "manifest": manifest_name, "parquet": parquet_name}
            for index, (manifest_name, parquet_name) in enumerate(zip(corpus_names, leading))
            if manifest_name != parquet_name
        ]
        raise ValueError(f"Corpus and parquet class names disagree: {disagreements}")

    absent = every_name[len(corpus_names) :]
    depicted = _image_path_names(manifest_path)
    wrongly_absent = sorted(set(absent) & depicted)
    if wrongly_absent:
        raise ValueError(
            f"Names registered as absent are depicted in the corpus: {wrongly_absent}"
        )

    object_terms = [
        {
            "name": name,
            "label": index if index < len(corpus_names) else None,
            "present_in_corpus": index < len(corpus_names),
        }
        for index, name in enumerate(every_name)
    ]

    vocabulary: dict[str, Any] = {
        "schema_version": config["schema_version"],
        "artifact": config["artifact"],
        "registered_before_naming": True,
        "milestone": "M82",
        "name_normalisation": config["name_normalisation"],
        "object_terms": object_terms,
        "object_templates": config["object_templates"]["templates"],
        "style_terms": config["style_terms"]["terms"],
        "style_templates": config["style_templates"]["templates"],
        "counts": {
            "object_terms": len(object_terms),
            "in_corpus_terms": len(corpus_names),
            "absent_terms": len(absent),
            "style_terms": len(config["style_terms"]["terms"]),
            "object_templates": len(config["object_templates"]["templates"]),
            "style_templates": len(config["style_templates"]["templates"]),
        },
        "controls": {
            "absent_terms_are_the_far_field_control": True,
            "absent_terms_verified_undepicted": True,
            "manifest_and_parquet_names_agree": True,
        },
        "provenance": {
            "manifest_path": config["corpus"]["manifest_path"],
            "manifest_sha256": sha256_file(manifest_path),
            "parquet_root": source["parquet_root"],
        },
    }
    vocabulary["vocabulary_hash"] = payload_hash(
        {
            "object_terms": object_terms,
            "object_templates": vocabulary["object_templates"],
            "style_terms": vocabulary["style_terms"],
            "style_templates": vocabulary["style_templates"],
            "name_normalisation": vocabulary["name_normalisation"],
        }
    )
    return vocabulary


def main() -> None:
    parser = argparse.ArgumentParser(description="Seal the M82 naming vocabulary.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()

    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    vocabulary = build_vocabulary(config)

    output_dir = _resolve_inside_repo(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output_dir / "vocabulary.json", vocabulary)
    build_artifact_index(output_dir)

    counts = vocabulary["counts"]
    print(f"vocabulary hash      {vocabulary['vocabulary_hash']}")
    print(f"object terms         {counts['object_terms']}")
    print(f"  in corpus          {counts['in_corpus_terms']}")
    print(f"  absent (far field) {counts['absent_terms']}")
    print(f"style terms          {counts['style_terms']}")
    print(f"object templates     {counts['object_templates']}")
    print(f"written to           {output_dir}")


if __name__ == "__main__":
    main()
