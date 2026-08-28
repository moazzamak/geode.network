from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from src.model_fingerprint import ModelFingerprint, OutputSpec
from src.model_network import FittedModel, ModelNetwork
from src.sdf_engine import Expert


@dataclass(frozen=True)
class GraphMigrationDryRun:
    candidate_network: ModelNetwork
    source_node: str
    old_signature: str
    new_signature: str
    old_classes: tuple[Any, ...]
    new_classes: tuple[Any, ...]
    replaced_downstream_nodes: tuple[str, ...]
    validation_issues: tuple[str, ...]
    published: bool = False

    @property
    def valid(self) -> bool:
        return not self.validation_issues


def dry_run_add_class_migration(
    network: ModelNetwork,
    *,
    source_node: str,
    new_class_id: Any,
    new_class_models: list[Expert],
    score_scale: float,
    replacement_calibrator: Any = None,
    downstream_replacements: dict[str, FittedModel] | None = None,
) -> GraphMigrationDryRun:
    """Build and validate a candidate graph without modifying the live network."""
    if source_node not in network._nodes:
        raise KeyError(f"Node {source_node!r} not in network.")
    live_source = network._nodes[source_node].model
    if not isinstance(live_source, FittedModel):
        raise TypeError("Class migration requires a FittedModel source node.")
    if new_class_id in live_source.class_models:
        raise ValueError(f"Class {new_class_id!r} already exists.")
    if score_scale <= 0.0:
        raise ValueError("score_scale must be positive.")
    if live_source.calibrator is not None and replacement_calibrator is None:
        raise ValueError("A migrated calibrated model requires an explicit recalibrator.")

    candidate = copy.deepcopy(network)
    source = candidate._nodes[source_node].model
    old_classes = tuple(source.class_ids)
    source.class_models[new_class_id] = copy.deepcopy(new_class_models)
    source.score_scales[new_class_id] = float(score_scale)
    new_classes = tuple(source.class_ids)
    source.fingerprint = ModelFingerprint(
        task_name=source.fingerprint.task_name,
        input_spec=source.fingerprint.input_spec,
        output_spec=OutputSpec(source.fingerprint.output_spec.type, new_classes),
        alpha=source.fingerprint.alpha,
        pca_components=source.fingerprint.pca_components,
    )
    source.calibrator = replacement_calibrator
    source._gpu_engine = None

    replacements = downstream_replacements or {}
    direct_dependants = tuple(sorted(
        name for name, node in candidate._nodes.items()
        if source_node in node.upstream
    ))
    missing_replacements = [
        name for name in direct_dependants if name not in replacements
    ]
    if missing_replacements:
        raise ValueError(
            "Downstream replacements required for changed score width: "
            f"{missing_replacements!r}"
        )
    for name in direct_dependants:
        candidate._nodes[name].model = copy.deepcopy(replacements[name])

    old_signature = live_source.fingerprint.signature
    new_signature = source.fingerprint.signature
    issues = tuple(candidate.validate())
    if old_signature == new_signature:
        issues += ("Source fingerprint signature did not change.",)
    return GraphMigrationDryRun(
        candidate_network=candidate,
        source_node=source_node,
        old_signature=old_signature,
        new_signature=new_signature,
        old_classes=old_classes,
        new_classes=new_classes,
        replaced_downstream_nodes=direct_dependants,
        validation_issues=issues,
    )