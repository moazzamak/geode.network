from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.tier4.eval_real_feature_ood_transfer import (
    run_real_feature_ood_transfer,
)


def run_representation_transfer(
    *,
    representations: list[str],
    **config,
) -> dict:
    if len(set(representations)) != len(representations) or not representations:
        raise ValueError("Representations must be a non-empty unique list.")
    results = {
        representation: run_real_feature_ood_transfer(
            representation=representation,
            **config,
        )
        for representation in representations
    }
    summary = {}
    for representation, result in results.items():
        selected_score = result["selection"]["score"]
        diagnostics = tuple(
            run["representation_diagnostics"] for run in result["runs"]
        )
        summary[representation] = {
            "development_selected_score": selected_score,
            "development_proxy_unknown_recall_mean": result[
                "selection"
            ]["development"]["proxy_unknown_recall_mean"],
            **result["final_summary"][selected_score],
            "closed_set_accuracy_mean": float(np.mean([
                run["closed_set"]["id_test_accuracy"] for run in result["runs"]
            ])),
            "neighborhood_purity_mean": float(np.mean([
                record["neighborhood_purity"] for record in diagnostics
            ])),
            "local_intrinsic_dimension_mean": float(np.mean([
                record["local_intrinsic_dimension"] for record in diagnostics
            ])),
            "compactness_ratio_mean": float(np.mean([
                record["compactness_ratio"] for record in diagnostics
            ])),
        }
    return {
        "protocol": {
            "representations": list(representations),
            "source_identity_rule_shared": True,
            "score_family_selected_per_representation_on_development_only": True,
            "final_unknown_used_for_representation_or_score_selection": False,
            "representation_advance_rule": "all_cells_pass_frozen_gate",
            "mutation_published": False,
        },
        "summary": summary,
        "representations_passing_gate": [
            representation for representation, record in summary.items()
            if record["all_cells_pass_gate"]
        ],
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen real-feature OOD representation transfer study",
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    artifact_path = Path(config.pop("artifact_path"))
    result = run_representation_transfer(**config)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": result["summary"],
        "representations_passing_gate": result["representations_passing_gate"],
    }, indent=2))


if __name__ == "__main__":
    main()