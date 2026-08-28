from __future__ import annotations

import argparse
import json
from pathlib import Path


FINAL_LABEL_KEYS = (
    "final_labels_used_for_selection",
    "final_labels_used_for_selection_or_constraints",
)


def _assert_false(protocol: dict, keys: tuple[str, ...], artifact: str) -> None:
    present = [key for key in keys if key in protocol]
    if not present or any(protocol[key] is not False for key in present):
        raise ValueError(f"{artifact} does not declare {keys} false.")


def run_m11_advancement_audit(
    *,
    artifacts: list[str],
    retained_automatic_artifact: str,
) -> dict:
    records = []
    for artifact in artifacts:
        payload = json.loads(Path(artifact).read_text(encoding="utf-8"))
        protocol = payload.get("protocol", {})
        _assert_false(protocol, ("mutation_published",), artifact)
        _assert_false(protocol, FINAL_LABEL_KEYS, artifact)
        if "experimental_constraints_persisted" in protocol:
            _assert_false(protocol, ("experimental_constraints_persisted",), artifact)
        if "feature_model_mutation_published" in protocol:
            _assert_false(protocol, ("feature_model_mutation_published",), artifact)
        records.append({
            "artifact": artifact,
            "final_labels_observational": True,
            "constraints_persisted": protocol.get(
                "experimental_constraints_persisted", False,
            ),
            "mutation_published": protocol["mutation_published"],
        })

    if retained_automatic_artifact not in artifacts:
        raise ValueError("retained_automatic_artifact must be audited.")
    return {
        "milestone": "M11.40",
        "audit_passed": True,
        "retained_automatic_policy": {
            "artifact": retained_automatic_artifact,
            "representation": "mobilenetv2",
            "flag_fraction": 0.3,
            "clustering_method": "hdbscan",
            "minimum_cluster_size": 3,
            "selection_scope": "proxy_classes_6_7_only",
            "final_scope": "classes_8_9_observational_only",
        },
        "gate_decisions": {
            "autonomous_grouping_advanced_after_m11_29": False,
            "assisted_review_policy_retained_for_analysis": True,
            "constraint_persistence_allowed": False,
            "semantic_class_creation_allowed": False,
            "representation_replacement_allowed": False,
            "model_mutation_allowed": False,
        },
        "m12_exhaustive_baseline": {
            "counter_schema": [
                "sample_count",
                "nodes_executed",
                "compatible_candidate_pairs",
                "shortlisted_candidate_pairs",
                "exact_class_sdf_pairs",
                "primitive_sdf_pairs",
                "score_values_materialized",
            ],
            "candidate_rule": "all_contract_compatible_classes",
            "exact_fallback": "not_applicable_exhaustive",
            "agreement_reference": "exhaustive_exact_class_sdf_argmin",
            "feedback_derived_routing_state": False,
        },
        "m12_entry_gate": {
            "candidate_growth_target": "sublinear_in_registered_classes",
            "minimum_exhaustive_agreement": 0.99,
            "open_set_tolerance_required": True,
            "resource_budgets_required": True,
            "real_features_before_toy_equivalence": False,
        },
        "artifact_checks": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = run_m11_advancement_audit(**config)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
