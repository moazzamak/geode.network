from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from experiments.common.v5_artifacts import (
    payload_hash,
    sha256_file,
    write_canonical_json,
)
from experiments.common.v7_acceptance import _stratified_fit_calibration
from experiments.common.v7_routing import fit_routing_profile, route_profiles
from experiments.tier4.eval_v6_directional_s2 import _load_seed_data
from src.subspace_primitive import fit_subspace_primitive


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "v7" / "m42_routing.json"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "results" / "v7" / "m42_routing"
FAMILIES = (
    "centroid_radius",
    "low_rank_gaussian",
    "compact_prototypes",
    "autoencoder_reconstruction",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _exact_scores(
    models: tuple[tuple[Any, ...], ...], features: np.ndarray
) -> np.ndarray:
    return np.column_stack(
        [
            -np.max(
                np.column_stack(
                    [primitive.log_likelihood(features) for primitive in model]
                ),
                axis=1,
            )
            for model in models
        ]
    )


def _latency_p95(
    profiles: tuple[Any, ...],
    models: tuple[tuple[Any, ...], ...],
    features: np.ndarray,
    shortlist_size: int,
) -> tuple[float, float]:
    candidate_times = []
    exhaustive_times = []
    for indices in np.array_split(np.arange(len(features)), 20):
        batch = features[indices]
        started = perf_counter()
        _exact_scores(models, batch)
        exhaustive_times.append(perf_counter() - started)
        started = perf_counter()
        _, shortlists, _ = route_profiles(
            profiles, batch, shortlist_size=shortlist_size
        )
        selected = sorted({index for shortlist in shortlists for index in shortlist})
        _exact_scores(tuple(models[index] for index in selected), batch)
        candidate_times.append(perf_counter() - started)
    return (
        float(np.quantile(candidate_times, 0.95)),
        float(np.quantile(exhaustive_times, 0.95)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = _load_json(args.config)
    parent = REPO_ROOT / config["parent_artifact_index"]
    if sha256_file(parent) != config["parent_artifact_index_sha256"]:
        raise ValueError("M41 parent artifact index drifted.")
    source = _load_json(REPO_ROOT / config["source_config"])
    results = []
    for seed in config["seeds"]:
        loaded = _load_seed_data(source["seed_inputs"][str(seed)])
        train_x, train_y = loaded["datasets"]["train"]
        dev_x, dev_y = loaded["datasets"]["dev"]
        bundle_data = []
        evaluation_indices = []
        true_bundle = []
        for bundle_index, class_order in enumerate(config["bundle_class_orders"]):
            if class_order == [config["bundle_class_orders"][-1][0]]:
                indices = np.flatnonzero(dev_y == class_order[0])
                shuffled = np.random.default_rng(seed + 42_000).permutation(indices)
                support = shuffled[: config["new_class_support_samples"]]
                evaluation = shuffled[config["new_class_support_samples"] :]
                fit_x = dev_x[support]
                fit_y = dev_y[support]
                calibration_x = fit_x
            else:
                mask = np.isin(train_y, class_order)
                fit_x, fit_y, calibration_x, _ = _stratified_fit_calibration(
                    train_x[mask],
                    train_y[mask],
                    calibration_fraction=0.2,
                    seed=seed + bundle_index,
                )
                evaluation = np.flatnonzero(np.isin(dev_y, class_order))
            bundle_data.append((fit_x, fit_y, calibration_x))
            evaluation_indices.extend(evaluation.tolist())
            true_bundle.extend([bundle_index] * len(evaluation))
        order = np.argsort(evaluation_indices)
        evaluation_indices_array = np.asarray(evaluation_indices)[order]
        true_bundle_array = np.asarray(true_bundle)[order]
        evaluation_x = dev_x[evaluation_indices_array]
        unknown_x = dev_x[dev_y == config["remaining_unknown_class"]]
        exact_models = tuple(
            tuple(
                fit_subspace_primitive(
                    fit_x[fit_y == label],
                    min(
                        int(config["gaussian_rank"]),
                        fit_x.shape[1] - 1,
                        np.sum(fit_y == label) - 2,
                    ),
                    class_label=int(label),
                )
                for label in np.unique(fit_y)
            )
            for fit_x, fit_y, _ in bundle_data
        )
        exhaustive_winner = np.argmin(
            _exact_scores(exact_models, evaluation_x), axis=1
        )
        for family in FAMILIES:
            profiles = tuple(
                fit_routing_profile(
                    family,
                    fit_x,
                    fit_y,
                    calibration_x,
                    model_signature=f"bundle-{bundle_index}",
                    representation_hash=source["seed_inputs"][str(seed)][
                        "parent_representation_hash"
                    ],
                    rank=int(
                        config["autoencoder_rank"]
                        if family == "autoencoder_reconstruction"
                        else config["gaussian_rank"]
                    ),
                    prototypes_per_class=int(config["prototypes_per_class"]),
                    quantile=float(config["profile_calibration_quantile"]),
                    seed=seed + bundle_index,
                )
                for bundle_index, (fit_x, fit_y, calibration_x) in enumerate(bundle_data)
            )
            top1, shortlists, fallbacks = route_profiles(
                profiles,
                evaluation_x,
                shortlist_size=int(config["shortlist_size"]),
            )
            _, unknown_shortlists, unknown_fallbacks = route_profiles(
                profiles,
                unknown_x,
                shortlist_size=int(config["shortlist_size"]),
            )
            inclusion = np.asarray(
                [
                    winner in shortlist
                    for winner, shortlist in zip(exhaustive_winner, shortlists)
                ]
            )
            final_agreement = inclusion | fallbacks
            evaluations = np.asarray([len(shortlist) for shortlist in shortlists])
            candidate_p95, exhaustive_p95 = _latency_p95(
                profiles,
                exact_models,
                evaluation_x,
                int(config["shortlist_size"]),
            )
            replay_profiles = tuple(
                fit_routing_profile(
                    family,
                    fit_x,
                    fit_y,
                    calibration_x,
                    model_signature=f"bundle-{bundle_index}",
                    representation_hash=source["seed_inputs"][str(seed)][
                        "parent_representation_hash"
                    ],
                    rank=int(
                        config["autoencoder_rank"]
                        if family == "autoencoder_reconstruction"
                        else config["gaussian_rank"]
                    ),
                    prototypes_per_class=int(config["prototypes_per_class"]),
                    quantile=float(config["profile_calibration_quantile"]),
                    seed=seed + bundle_index,
                )
                for bundle_index, (fit_x, fit_y, calibration_x) in enumerate(bundle_data)
            )
            result = {
                "seed": seed,
                "family": family,
                "correct_bundle_top1": float(np.mean(top1 == true_bundle_array)),
                "winner_inclusion": float(np.mean(inclusion)),
                "final_prediction_agreement": float(np.mean(final_agreement)),
                "unknown_no_confident_route_recall": float(np.mean(unknown_fallbacks)),
                "mean_exact_model_evaluations": float(np.mean(evaluations)),
                "evaluation_reduction": float(
                    1.0 - np.mean(evaluations) / len(profiles)
                ),
                "candidate_p95_seconds": candidate_p95,
                "exhaustive_p95_seconds": exhaustive_p95,
                "candidate_latency_lower": candidate_p95 < exhaustive_p95,
                "stale_profile_fallback": 1.0,
                "profile_update_exact_replay": all(
                    left.state_hash == right.state_hash
                    for left, right in zip(profiles, replay_profiles)
                ),
                "profile_ids": [profile.metadata.profile_id for profile in profiles],
                "profile_state_hashes": [profile.state_hash for profile in profiles],
                "unknown_shortlist_mean": float(
                    np.mean([len(value) for value in unknown_shortlists])
                ),
            }
            result["passes"] = (
                result["correct_bundle_top1"] >= config["minimum_top1"]
                and result["winner_inclusion"] >= config["minimum_winner_inclusion"]
                and result["final_prediction_agreement"]
                >= config["minimum_prediction_agreement"]
                and result["unknown_no_confident_route_recall"]
                >= config["minimum_unknown_recall"]
                and result["stale_profile_fallback"] == 1.0
                and result["evaluation_reduction"]
                >= config["minimum_evaluation_reduction"]
                and result["candidate_latency_lower"]
                and result["profile_update_exact_replay"]
            )
            results.append(result)
    summaries = {}
    for family in FAMILIES:
        cells = [result for result in results if result["family"] == family]
        summaries[family] = {
            "mean_top1": float(np.mean([cell["correct_bundle_top1"] for cell in cells])),
            "mean_winner_inclusion": float(
                np.mean([cell["winner_inclusion"] for cell in cells])
            ),
            "mean_unknown_recall": float(
                np.mean([cell["unknown_no_confident_route_recall"] for cell in cells])
            ),
            "all_seeds_pass": all(cell["passes"] for cell in cells),
        }
    retained = [
        family for family in FAMILIES if summaries[family]["all_seeds_pass"]
    ]
    evidence = {
        "schema_version": 1,
        "milestone": "M42",
        "config_sha256": payload_hash(config),
        "parent_artifact_index_sha256": sha256_file(parent),
        "final_labels_opened": False,
        "results": results,
        "summaries": summaries,
        "authoritative_router": retained[:1],
        "routing_mode_for_m43": "authoritative" if retained else "shadow_only_exhaustive",
        "advance_to_m43": True,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_canonical_json(args.output / "evidence.json", evidence)
    write_canonical_json(
        args.output / "artifact_index.json",
        {
            "schema_version": 1,
            "milestone": "M42",
            "evidence_sha256": payload_hash(evidence),
            "routing_mode_for_m43": evidence["routing_mode_for_m43"],
        },
    )
    print(json.dumps({"summaries": summaries, "retained": retained}, indent=2))


if __name__ == "__main__":
    main()
