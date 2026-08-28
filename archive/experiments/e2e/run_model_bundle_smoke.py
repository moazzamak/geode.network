"""Qualify immutable model bundles, compatibility gates, and rollback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from src.model_fingerprint import InputSpec, ModelFingerprint, OutputSpec
from src.open_set import SupportProfile
from src.runtime.model_bundle import (
    BundleNode,
    BundleProvenance,
    LocalModelBundleStore,
    assert_node_replacement,
)


def _provenance() -> BundleProvenance:
    return BundleProvenance(
        routing_mode="exhaustive",
        semantic_router_cache_version="cache-v1",
        training_manifest_hash="1" * 64,
        evaluation_manifest_hash="2" * 64,
        metric_summary_hash="3" * 64,
        software_compatibility=">=1.0,<2.0",
        environment_fingerprint="e3-fixed-environment",
        created_at="2026-07-26T00:00:00Z",
        created_by="e3-model-bundle-qualification",
    )


def _node(
    *,
    name: str = "source",
    task: str = "source-task",
    source: str = "passthrough",
    output_type: str = "sdf_scores",
    classes: tuple[int, ...] = (0, 1),
    transform: str = "transform-v1",
    upstream: tuple[str, ...] = (),
    input_dim: int = 2,
) -> BundleNode:
    fingerprint = ModelFingerprint(
        task_name=task,
        input_spec=InputSpec(
            "sdf_scores" if upstream else source,
            ("source-task",) if upstream else (),
            input_dim,
        ),
        output_spec=OutputSpec(output_type, classes),
    )
    profile = SupportProfile(
        model_signature=fingerprint.signature,
        feature_transform_fingerprint=transform,
        training_dataset_fingerprint="train-v1",
        calibration_dataset_fingerprint="calibration-v1",
        class_ids=classes,
        score_scales=tuple(1.0 for _ in classes),
        novelty_score="minimum_sdf",
        global_threshold=0.5,
        version="support-v1",
        fit_seed=7,
        created_at="2026-07-26T00:00:00Z",
    )
    return BundleNode(
        name=name,
        artifact_path=f"{name}.bin",
        fingerprint=fingerprint,
        class_order=classes,
        feature_transform_fingerprint=transform,
        upstream=upstream,
        support_profile=profile,
    )


def _rejected(existing: BundleNode, replacement: BundleNode) -> bool:
    try:
        assert_node_replacement(existing, replacement)
    except ValueError:
        return True
    return False


def run_qualification() -> dict:
    with tempfile.TemporaryDirectory() as directory:
        store = LocalModelBundleStore(directory)
        original_node = _node()
        original = store.publish(
            {"source.bin": b"model-v1"},
            [original_node],
            provenance=_provenance(),
        )
        store.activate(original.bundle_id)

        replacement_node = _node()
        assert_node_replacement(original_node, replacement_node)
        replacement = store.publish(
            {"source.bin": b"model-v2"},
            [replacement_node],
            provenance=_provenance(),
            parent_bundle_id=original.bundle_id,
        )
        store.activate(replacement.bundle_id)
        rolled_back = store.rollback()

        negative_compatibility = {
            "task": _rejected(original_node, _node(task="other-task")),
            "input_source": _rejected(original_node, _node(source="raw_hog")),
            "output_type": _rejected(original_node, _node(output_type="probabilities")),
            "class_set": _rejected(original_node, _node(classes=(0, 2))),
        }
        transform_mismatch = _rejected(
            original_node, _node(transform="transform-v2"),
        )

        expanded_source = _node(classes=(0, 1, 2))
        stale_downstream = _node(
            name="downstream", task="downstream-task", upstream=("source",),
            input_dim=2,
        )
        components = {"source.bin": b"source", "downstream.bin": b"downstream"}
        try:
            store.publish(
                components, [expanded_source, stale_downstream],
                provenance=_provenance(),
            )
            stale_expansion_rejected = False
        except ValueError:
            stale_expansion_rejected = True
        migrated_downstream = _node(
            name="downstream", task="downstream-task", upstream=("source",),
            input_dim=3,
        )
        expanded = store.publish(
            components, [expanded_source, migrated_downstream],
            provenance=_provenance(),
        )

        corrupted = store.publish(
            {"source.bin": b"corruption-target"},
            [original_node],
            provenance=_provenance(),
            parent_bundle_id=original.bundle_id,
        )
        corrupted_path = (
            Path(directory) / "bundles" / corrupted.bundle_id
            / "components" / "source.bin"
        )
        corrupted_path.write_bytes(b"corrupted")
        try:
            store.activate(corrupted.bundle_id)
            corruption_rejected = False
        except ValueError:
            corruption_rejected = True

        outcomes = {
            "positive_swap": True,
            "negative_compatibility": negative_compatibility,
            "transform_mismatch_rejected": transform_mismatch,
            "stale_class_expansion_rejected": stale_expansion_rejected,
            "coordinated_class_expansion": len(expanded.nodes) == 2,
            "corruption_rejected": corruption_rejected,
            "pointer_unchanged_after_corruption": (
                store.current().bundle_id == original.bundle_id
            ),
            "rollback_without_rebuild": rolled_back.bundle_id == original.bundle_id,
        }
        passed = all(
            value if isinstance(value, bool) else all(value.values())
            for value in outcomes.values()
        )
        return {
            "schema_version": 1,
            "milestone": "E3",
            "passed": passed,
            "bundle_ids": {
                "original": original.bundle_id,
                "replacement": replacement.bundle_id,
                "expanded": expanded.bundle_id,
            },
            "original_artifacts": [artifact.to_dict() for artifact in original.artifacts],
            "outcomes": outcomes,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/results/e3_model_bundle_qualification.json"),
    )
    arguments = parser.parse_args()
    summary = run_qualification()
    if not summary["passed"]:
        raise RuntimeError(f"E3 model bundle qualification failed: {summary['outcomes']}")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()