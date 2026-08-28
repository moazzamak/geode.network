from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.v5_artifacts import payload_hash, write_canonical_json
from experiments.common.v7_acceptance import evaluate_acceptance_heads
from experiments.tier4.eval_v6_directional_s2 import _load_seed_data


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v7" / "m39_acceptance_heads.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v7" / "m39_acceptance_heads"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _gate(seed_results: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    names = list(seed_results[0]["heads"])
    summaries: dict[str, Any] = {}
    best_control_precision = max(
        np.mean(
            [result["heads"][name]["review_precision"] for result in seed_results]
        )
        for name in names
        if name != "weighted_affine_sdf"
    )
    for name in names:
        values = [result["heads"][name] for result in seed_results]
        autonomy = all(
            value["known_coverage"] >= config["minimum_evaluation_known_coverage"]
            and value["unknown_recall"] >= config["minimum_unknown_recall"]
            and value["accepted_known_accuracy_loss"]
            <= config["maximum_accepted_known_accuracy_loss"]
            and value["exact_replay"]
            for value in values
        )
        review_precision = float(np.mean([value["review_precision"] for value in values]))
        mean_unknown_recall = float(np.mean([value["unknown_recall"] for value in values]))
        review = bool(
            review_precision
            >= best_control_precision - config["maximum_review_precision_gap"]
            and mean_unknown_recall
            >= config["historical_m11_unknown_recall"]
            + config["minimum_historical_recall_improvement"]
            and all(
                value["known_extension_false_reject_rate"]
                <= 1.0 - config["minimum_evaluation_known_coverage"] + 1e-12
                and value["corruption_false_reject_increase"]
                <= config["maximum_corruption_false_reject_increase"]
                and value["fit_and_score_seconds"]
                <= config["resource_limits"]["maximum_fit_seconds_per_seed"]
                and value["serialized_megabytes"]
                <= config["resource_limits"]["maximum_serialized_megabytes_per_head"]
                for value in values
            )
        )
        summaries[name] = {
            "mean_known_coverage": float(
                np.mean([value["known_coverage"] for value in values])
            ),
            "mean_unknown_recall": mean_unknown_recall,
            "mean_auroc": float(np.mean([value["auroc"] for value in values])),
            "mean_review_precision": review_precision,
            "autonomy_gate": autonomy,
            "review_gate": review,
            "retained": autonomy and review,
        }
    retained = [name for name, value in summaries.items() if value["retained"]]
    return {
        "best_non_geometric_review_precision": float(best_control_precision),
        "head_summaries": summaries,
        "retained_heads": retained,
        "advance_to_m40": bool(retained),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed-limit", type=int)
    args = parser.parse_args()

    config = _load_json(args.config)
    source = _load_json(REPO_ROOT / config["source_config"])
    seeds = config["seeds"][: args.seed_limit]
    seed_results = []
    for seed in seeds:
        loaded = _load_seed_data(source["seed_inputs"][str(seed)])
        train_features, train_labels = loaded["datasets"]["train"]
        dev_features, dev_labels = loaded["datasets"]["dev"]
        seed_results.append(
            evaluate_acceptance_heads(
                train_features,
                train_labels,
                dev_features,
                dev_labels,
                config,
                seed=seed,
            )
        )
    evidence = {
        "schema_version": 1,
        "milestone": "M39",
        "config_sha256": payload_hash(config),
        "source_config_sha256": payload_hash(source),
        "seeds": seeds,
        "final_labels_opened": False,
        "seed_results": seed_results,
        "gate": _gate(seed_results, config),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_canonical_json(args.output / "evidence.json", evidence)
    write_canonical_json(
        args.output / "artifact_index.json",
        {
            "schema_version": 1,
            "milestone": "M39",
            "evidence_sha256": payload_hash(evidence),
            "advance_to_m40": evidence["gate"]["advance_to_m40"],
        },
    )
    print(json.dumps(evidence["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
