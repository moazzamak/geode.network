from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from src.discovery_clustering import (
    dbscan_rejections,
    incremental_centroid_rejections,
    no_clustering_rejections,
)
from src.discovery_policy import ClusterProposalPolicy, evaluate_cluster_proposal
from src.open_set import UNKNOWN_LABEL, OpenSetPrediction, OpenSetReason
from src.rejection_buffer import RejectionBuffer


KNOWN = "known"
SHIFTED_KNOWN = "shifted_known"
NEW_KNOWN_MODE = "new_known_mode"
UNSEEN_CLASS = "unseen_class"
STREAM_EVENT_TYPES = (KNOWN, SHIFTED_KNOWN, NEW_KNOWN_MODE, UNSEEN_CLASS)
STREAM_FAMILIES = (
    "baseline",
    "heavy_tailed",
    "anisotropic",
    "abrupt_drift",
    "intermittent_unseen",
)


@dataclass(frozen=True)
class ObservableStream:
    sample_ids: np.ndarray
    embeddings: np.ndarray
    timestamps: np.ndarray
    window_ids: np.ndarray


@dataclass(frozen=True)
class DelayedOracle:
    sample_ids: np.ndarray
    class_ids: np.ndarray
    event_types: tuple[str, ...]


@dataclass(frozen=True)
class ReplayControls:
    known_embeddings: np.ndarray
    known_class_ids: np.ndarray
    ood_embeddings: np.ndarray


@dataclass(frozen=True)
class ClassIncrementalFixture:
    observable: ObservableStream
    oracle: DelayedOracle
    known_centroids: np.ndarray
    replay: ReplayControls


def generate_class_incremental_stream(
    *,
    seed: int,
    dimensions: int = 4,
    window_count: int = 3,
    samples_per_event_per_window: int = 8,
    stream_family: str = "baseline",
) -> ClassIncrementalFixture:
    """Generate recurring event types while keeping oracle labels out of inputs."""
    if dimensions < 2 or window_count < 2 or samples_per_event_per_window <= 0:
        raise ValueError("dimensions/window_count must be >= 2 and sample count positive.")
    if stream_family not in STREAM_FAMILIES:
        raise ValueError(f"Unknown stream family: {stream_family}.")
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(3, dimensions))
    centers *= 4.0 / np.linalg.norm(centers, axis=1, keepdims=True)
    mode_direction = rng.normal(size=dimensions)
    mode_direction /= np.linalg.norm(mode_direction)
    anisotropic_scales = np.linspace(0.5, 1.4, dimensions)
    anisotropic_scales /= np.sqrt(np.mean(anisotropic_scales**2))

    def draw_cluster(center: np.ndarray, noise: float, count: int) -> np.ndarray:
        if stream_family == "heavy_tailed":
            offsets = rng.standard_t(5, size=(count, dimensions)) * np.sqrt(3 / 5)
        else:
            offsets = rng.normal(size=(count, dimensions))
        if stream_family == "anisotropic":
            offsets *= anisotropic_scales
        return center + noise * offsets

    embeddings = []
    window_ids = []
    class_ids = []
    event_types = []
    for window_id in range(window_count):
        if stream_family == "abrupt_drift":
            drift = (0.0 if window_id == 0 else 0.45) * mode_direction
        else:
            drift = 0.1 * window_id * mode_direction
        event_specs = (
            (KNOWN, 0, centers[0] + drift, 0.45),
            (SHIFTED_KNOWN, 0, centers[0] + 1.3 * mode_direction + drift, 0.7),
            (NEW_KNOWN_MODE, 1, centers[1] - 1.6 * mode_direction + drift, 0.4),
            (UNSEEN_CLASS, 2, centers[2] + drift, 0.4),
        )
        window_embeddings = []
        window_classes = []
        window_types = []
        for event_type, class_id, center, noise in event_specs:
            if (
                stream_family == "intermittent_unseen"
                and event_type == UNSEEN_CLASS
                and window_id == 1
            ):
                continue
            window_embeddings.append(draw_cluster(
                center, noise, samples_per_event_per_window,
            ))
            window_classes.extend([class_id] * samples_per_event_per_window)
            window_types.extend([event_type] * samples_per_event_per_window)
        window_embeddings = np.vstack(window_embeddings)
        order = rng.permutation(len(window_embeddings))
        embeddings.append(window_embeddings[order])
        class_ids.extend(np.asarray(window_classes)[order].tolist())
        event_types.extend(np.asarray(window_types, dtype=object)[order].tolist())
        window_ids.extend([window_id] * len(order))

    all_embeddings = np.vstack(embeddings)
    sample_ids = np.arange(len(all_embeddings), dtype=np.int64)
    windows = np.asarray(window_ids, dtype=np.int64)
    timestamps = sample_ids.astype(np.float64)
    replay_known = np.vstack([
        rng.normal(center, 0.35, size=(24, dimensions))
        for center in centers[:2]
    ])
    replay_labels = np.repeat(np.arange(2, dtype=np.int64), 24)
    ood_direction = rng.normal(size=dimensions)
    ood_direction /= np.linalg.norm(ood_direction)
    replay_ood = rng.normal(
        10.0 * ood_direction,
        0.5,
        size=(48, dimensions),
    )
    return ClassIncrementalFixture(
        observable=ObservableStream(
            sample_ids=sample_ids,
            embeddings=all_embeddings,
            timestamps=timestamps,
            window_ids=windows,
        ),
        oracle=DelayedOracle(
            sample_ids=sample_ids.copy(),
            class_ids=np.asarray(class_ids, dtype=np.int64),
            event_types=tuple(event_types),
        ),
        known_centroids=centers[:2].copy(),
        replay=ReplayControls(
            known_embeddings=replay_known,
            known_class_ids=replay_labels,
            ood_embeddings=replay_ood,
        ),
    )


def _buffer_rejections(
    fixture: ClassIncrementalFixture,
    *,
    rejection_threshold: float,
) -> RejectionBuffer:
    observable = fixture.observable
    buffer = RejectionBuffer(
        max_records=len(observable.sample_ids),
        max_embedding_dimensions=observable.embeddings.shape[1],
    )
    for sample_id, embedding, timestamp, window_id in zip(
        observable.sample_ids,
        observable.embeddings,
        observable.timestamps,
        observable.window_ids,
    ):
        distances = np.linalg.norm(fixture.known_centroids - embedding, axis=1)
        nearest = np.argsort(distances, kind="stable")
        novelty = float(distances[nearest[0]])
        if novelty < rejection_threshold:
            continue
        prediction = OpenSetPrediction(
            label=UNKNOWN_LABEL,
            accepted=False,
            candidate_model_signature="stream-known-model-v1",
            candidate_class_id=int(nearest[0]),
            raw_novelty_score=novelty,
            calibrated_novelty_score=novelty,
            threshold=rejection_threshold,
            decision_margin=novelty - rejection_threshold,
            support_profile_version="stream-profile-v1",
            reason_code=OpenSetReason.OUTSIDE_SUPPORT,
        )
        buffer.append_rejection(
            embedding,
            timestamp=float(timestamp),
            window_id=int(window_id),
            prediction=prediction,
            nearest_candidates=tuple(int(value) for value in nearest),
            source_sample_id=int(sample_id),
        )
    return buffer


def _evaluate_strategy(
    fixture: ClassIncrementalFixture,
    records: tuple,
    clusterer,
    policy: ClusterProposalPolicy,
) -> dict:
    oracle_types = np.asarray(fixture.oracle.event_types)
    proposals = []
    reviews = []
    assignment_truth = []
    assignment_predicted = []
    first_window = None
    for window_id in sorted({record.window_id for record in records}):
        prefix = tuple(record for record in records if record.window_id <= window_id)
        if any(
            evaluate_cluster_proposal(cluster, fixture.known_centroids, policy).eligible
            for cluster in clusterer(prefix)
        ):
            first_window = window_id
            break
    for cluster_index, cluster in enumerate(clusterer(records)):
        decision = evaluate_cluster_proposal(cluster, fixture.known_centroids, policy)
        if not decision.eligible and not decision.review_required:
            continue
        sample_ids = np.asarray(
            [record.source_sample_id for record in cluster], dtype=np.int64,
        )
        event_names, event_counts = np.unique(
            oracle_types[sample_ids], return_counts=True,
        )
        oracle_event_counts = {
            str(name): int(count)
            for name, count in zip(event_names, event_counts)
        }
        if decision.review_required:
            reviews.append({
                "review_id": decision.review_id,
                "sample_ids": sample_ids.tolist(),
                "support": decision.support,
                "windows": decision.windows,
                "rms_radius": decision.rms_radius,
                "nearest_known_distance": decision.nearest_known_distance,
                "failed_criteria": list(decision.failed_criteria),
                "oracle_event_counts": oracle_event_counts,
            })
            continue
        proposals.append({
            "temporary_unknown_id": decision.temporary_unknown_id,
            "sample_ids": sample_ids.tolist(),
            "support": decision.support,
            "windows": decision.windows,
            "rms_radius": decision.rms_radius,
            "nearest_known_distance": decision.nearest_known_distance,
            "oracle_event_counts": oracle_event_counts,
        })
        assignment_truth.extend(oracle_types[sample_ids].tolist())
        assignment_predicted.extend([cluster_index] * len(sample_ids))

    proposed_ids = {
        sample_id for proposal in proposals for sample_id in proposal["sample_ids"]
    }
    unseen_ids = set(np.flatnonzero(oracle_types == UNSEEN_CLASS).tolist())
    reviewed_ids = {
        sample_id for review in reviews for sample_id in review["sample_ids"]
    }
    true_positive_ids = proposed_ids & unseen_ids
    purity_parts = []
    for proposal in proposals:
        types = oracle_types[np.asarray(proposal["sample_ids"], dtype=np.int64)]
        counts = np.unique(types, return_counts=True)[1]
        purity_parts.append(int(np.max(counts)))
    return {
        "proposal_count": len(proposals),
        "proposed_sample_count": len(proposed_ids),
        "discovery_precision": (
            len(true_positive_ids) / len(proposed_ids) if proposed_ids else 0.0
        ),
        "discovery_recall": len(true_positive_ids) / len(unseen_ids),
        "cluster_purity": (
            sum(purity_parts) / len(proposed_ids) if proposed_ids else 0.0
        ),
        "adjusted_rand_index": (
            float(adjusted_rand_score(assignment_truth, assignment_predicted))
            if assignment_truth else 0.0
        ),
        "false_proposal_count": sum(
            proposal["oracle_event_counts"].get(UNSEEN_CLASS, 0)
            <= sum(proposal["oracle_event_counts"].values()) / 2
            for proposal in proposals
        ),
        "review_count": len(reviews),
        "reviewed_sample_count": len(reviewed_ids),
        "reviewed_unseen_sample_count": len(reviewed_ids & unseen_ids),
        "unseen_recall_with_review": len(
            (proposed_ids | reviewed_ids) & unseen_ids
        ) / len(unseen_ids),
        "time_to_first_proposal_window": first_window,
        "proposals": proposals,
        "reviews": reviews,
    }


def run_streaming_discovery_study(
    *,
    seed: int = 42,
    rejection_threshold: float = 1.0,
    dbscan_epsilon: float = 0.9,
    dbscan_minimum_samples: int = 4,
    incremental_assignment_radius: float = 0.9,
    policy: ClusterProposalPolicy | None = None,
    stream_family: str = "baseline",
) -> dict:
    fixture = generate_class_incremental_stream(
        seed=seed, stream_family=stream_family,
    )
    buffer = _buffer_rejections(
        fixture, rejection_threshold=rejection_threshold,
    )
    proposal_policy = policy or ClusterProposalPolicy(
        minimum_support=8,
        minimum_windows=2,
        maximum_rms_radius=1.2,
        minimum_known_separation=1.5,
    )
    strategies = {
        "dbscan": partial(
            dbscan_rejections,
            epsilon=dbscan_epsilon,
            minimum_samples=dbscan_minimum_samples,
        ),
        "incremental_centroid": partial(
            incremental_centroid_rejections,
            assignment_radius=incremental_assignment_radius,
        ),
        "no_clustering": no_clustering_rejections,
    }
    return {
        "protocol": {
            "seed": seed,
            "stream_family": stream_family,
            "rejection_threshold": rejection_threshold,
            "oracle_used_for_discovery": False,
            "automatic_class_creation_enabled": False,
            "semantic_proposals_enabled": False,
        },
        "stream": {
            "sample_count": len(fixture.observable.sample_ids),
            "rejected_count": len(buffer),
            "window_count": len(np.unique(fixture.observable.window_ids)),
        },
        "strategies": {
            name: _evaluate_strategy(
                fixture, buffer.snapshot(), clusterer, proposal_policy,
            )
            for name, clusterer in strategies.items()
        },
    }


def run_streaming_discovery_multiseed(
    seeds: tuple[int, ...],
    **study_kwargs,
) -> dict:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique.")
    runs = [
        run_streaming_discovery_study(seed=seed, **study_kwargs)
        for seed in seeds
    ]
    summary = {}
    for strategy_name in runs[0]["strategies"]:
        records = [run["strategies"][strategy_name] for run in runs]
        summary[strategy_name] = {
            "discovery_precision_mean": float(np.mean([
                record["discovery_precision"] for record in records
            ])),
            "discovery_precision_minimum": float(np.min([
                record["discovery_precision"] for record in records
            ])),
            "discovery_recall_mean": float(np.mean([
                record["discovery_recall"] for record in records
            ])),
            "discovery_recall_minimum": float(np.min([
                record["discovery_recall"] for record in records
            ])),
            "false_proposals_total": int(sum(
                record["false_proposal_count"] for record in records
            )),
            "runs_with_proposals": int(sum(
                record["proposal_count"] > 0 for record in records
            )),
            "reviews_total": int(sum(
                record["review_count"] for record in records
            )),
            "reviewed_samples_total": int(sum(
                record["reviewed_sample_count"] for record in records
            )),
            "unseen_recall_with_review_mean": float(np.mean([
                record["unseen_recall_with_review"] for record in records
            ])),
            "unseen_recall_with_review_minimum": float(np.min([
                record["unseen_recall_with_review"] for record in records
            ])),
        }
    return {
        "protocol": {
            "seeds": list(seeds),
            "parameters_selected_before_multiseed_observation": True,
            "oracle_used_for_discovery": False,
            "automatic_class_creation_enabled": False,
        },
        "summary": summary,
        "runs": runs,
    }


def run_streaming_policy_transfer_study(
    *,
    development_seeds: tuple[int, ...],
    holdout_seeds: tuple[int, ...],
    separation_candidates: tuple[float, ...],
    **study_kwargs,
) -> dict:
    if set(development_seeds) & set(holdout_seeds):
        raise ValueError("development and holdout seeds must be disjoint.")
    if not separation_candidates or any(value < 0.0 for value in separation_candidates):
        raise ValueError("separation_candidates must be non-empty and non-negative.")
    development = {}
    for separation in separation_candidates:
        policy = ClusterProposalPolicy(
            minimum_support=8,
            minimum_windows=2,
            maximum_rms_radius=1.2,
            minimum_known_separation=separation,
        )
        development[str(separation)] = run_streaming_discovery_multiseed(
            development_seeds,
            policy=policy,
            **study_kwargs,
        )

    selected = {}
    holdout = {}
    for strategy_name in ("dbscan", "incremental_centroid", "no_clustering"):
        selected_separation = max(
            separation_candidates,
            key=lambda separation: (
                -development[str(separation)]["summary"][strategy_name][
                    "false_proposals_total"
                ],
                development[str(separation)]["summary"][strategy_name][
                    "discovery_precision_mean"
                ],
                development[str(separation)]["summary"][strategy_name][
                    "discovery_recall_mean"
                ],
                -separation,
            ),
        )
        selected[strategy_name] = selected_separation
        policy = ClusterProposalPolicy(
            minimum_support=8,
            minimum_windows=2,
            maximum_rms_radius=1.2,
            minimum_known_separation=selected_separation,
        )
        result = run_streaming_discovery_multiseed(
            holdout_seeds,
            policy=policy,
            **study_kwargs,
        )
        holdout[strategy_name] = {
            "selected_separation": selected_separation,
            "summary": result["summary"][strategy_name],
            "runs": [run["strategies"][strategy_name] for run in result["runs"]],
        }
    return {
        "protocol": {
            "development_seeds": list(development_seeds),
            "holdout_seeds": list(holdout_seeds),
            "separation_candidates": list(separation_candidates),
            "holdout_used_for_selection": False,
            "automatic_class_creation_enabled": False,
        },
        "development": {
            separation: result["summary"]
            for separation, result in development.items()
        },
        "selected_separation": selected,
        "holdout": holdout,
    }


def run_frozen_stream_family_study(
    *,
    holdout_seeds: tuple[int, ...],
    stream_families: tuple[str, ...],
    frozen_separation: float,
    **study_kwargs,
) -> dict:
    if not stream_families or len(set(stream_families)) != len(stream_families):
        raise ValueError("stream_families must be non-empty and unique.")
    policy = ClusterProposalPolicy(
        minimum_support=8,
        minimum_windows=2,
        maximum_rms_radius=1.2,
        minimum_known_separation=frozen_separation,
    )
    results = {
        family: run_streaming_discovery_multiseed(
            holdout_seeds,
            policy=policy,
            stream_family=family,
            **study_kwargs,
        )
        for family in stream_families
    }
    return {
        "protocol": {
            "holdout_seeds": list(holdout_seeds),
            "stream_families": list(stream_families),
            "frozen_separation": frozen_separation,
            "parameters_selected_before_family_observation": True,
            "oracle_used_for_discovery": False,
            "automatic_class_creation_enabled": False,
        },
        "families": {
            family: result["summary"]["dbscan"]
            for family, result in results.items()
        },
        "runs": {
            family: [
                {
                    "seed": run["protocol"]["seed"],
                    **run["strategies"]["dbscan"],
                }
                for run in result["runs"]
            ]
            for family, result in results.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M11 streaming discovery study")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    artifact_path = Path(config.pop("artifact_path"))
    if "stream_families" in config:
        result = run_frozen_stream_family_study(
            holdout_seeds=tuple(config.pop("holdout_seeds")),
            stream_families=tuple(config.pop("stream_families")),
            frozen_separation=config.pop("frozen_separation"),
            **config,
        )
        console_summary = result["families"]
    elif "development_seeds" in config:
        result = run_streaming_policy_transfer_study(
            development_seeds=tuple(config.pop("development_seeds")),
            holdout_seeds=tuple(config.pop("holdout_seeds")),
            separation_candidates=tuple(config.pop("separation_candidates")),
            **config,
        )
        console_summary = result["holdout"]
    else:
        seeds = tuple(config.pop("seeds"))
        result = run_streaming_discovery_multiseed(seeds, **config)
        console_summary = result["summary"]
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(console_summary, indent=2))


if __name__ == "__main__":
    main()