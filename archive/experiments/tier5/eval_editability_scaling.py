import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.inference_engine import InferenceEngine
from src.model_editor import ModelEditor
from src.open_set import RoutingStageCounters
from src.sdf_engine import EllipsoidExpert, Expert


SCALING_AXES = ("class_count", "dimensions", "primitives_per_class")


class _CacheTracker:
    def __init__(self):
        self.entries: dict[int, np.ndarray] = {}
        self.invalidations = 0
        self.invalidation_seconds = 0.0

    def invalidate(self, _models: dict) -> None:
        started = time.perf_counter()
        self.entries.clear()
        self.invalidations += 1
        self.invalidation_seconds += time.perf_counter() - started


def build_scaling_conditions(config: dict) -> list[dict]:
    baseline = {axis: int(config["baseline"][axis]) for axis in SCALING_AXES}
    conditions = [{"name": "baseline", **baseline}]
    for axis in SCALING_AXES:
        for value in config["sweeps"][axis]:
            value = int(value)
            if value == baseline[axis]:
                continue
            condition = baseline.copy()
            condition[axis] = value
            conditions.append({"name": f"{axis}={value}", **condition})
    if any(condition[axis] < 1 for condition in conditions for axis in SCALING_AXES):
        raise ValueError("All scaling dimensions must be positive integers.")
    return conditions


def _class_model(
    class_id: int,
    class_count: int,
    dimensions: int,
    primitives_per_class: int,
    seed: int,
) -> list[Expert]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, class_id]))
    direction = rng.normal(size=dimensions)
    direction /= max(np.linalg.norm(direction), 1e-12)
    anchor = direction * (4.0 + 0.1 * class_id / max(class_count, 1))
    expert = Expert(alpha=2.0)
    for _ in range(primitives_per_class):
        expert.add_ellipsoid(EllipsoidExpert(
            center=anchor + rng.normal(scale=0.25, size=dimensions),
            radii=rng.uniform(0.7, 1.3, size=dimensions),
        ))
    return [expert]


def _models(condition: dict, seed: int) -> dict[int, list[Expert]]:
    return {
        class_id: _class_model(
            class_id,
            condition["class_count"],
            condition["dimensions"],
            condition["primitives_per_class"],
            seed,
        )
        for class_id in range(condition["class_count"])
    }


def _scores(
    models: dict[int, list[Expert]], points: np.ndarray,
) -> tuple[dict[int, np.ndarray], float]:
    started = time.perf_counter()
    scores = {
        class_id: InferenceEngine(experts, alpha=2.0).get_fused_sdf(points)
        for class_id, experts in sorted(models.items())
    }
    return scores, time.perf_counter() - started


def _predictions(scores: dict[int, np.ndarray]) -> np.ndarray:
    class_ids = np.array(sorted(scores))
    matrix = np.column_stack([scores[class_id] for class_id in class_ids])
    return class_ids[np.argmin(matrix, axis=1)]


def _routing_counts(
    models: dict[int, list[Expert]], evaluation_count: int,
) -> RoutingStageCounters:
    class_count = len(models)
    nonempty_classes = sum(bool(experts) for experts in models.values())
    primitive_count = sum(
        len(expert.ellipsoids)
        for experts in models.values()
        for expert in experts
    )
    candidate_pairs = evaluation_count * class_count
    return RoutingStageCounters(
        sample_count=evaluation_count,
        nodes_executed=1,
        compatible_candidate_pairs=candidate_pairs,
        shortlisted_candidate_pairs=candidate_pairs,
        exact_class_sdf_pairs=evaluation_count * nonempty_classes,
        primitive_sdf_pairs=evaluation_count * primitive_count,
        score_values_materialized=candidate_pairs,
    )


def _fit_update(points: np.ndarray, _seed: int) -> EllipsoidExpert:
    return EllipsoidExpert(
        center=np.mean(points, axis=0),
        radii=np.maximum(np.std(points, axis=0), 0.35),
    )


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def run_scaling_condition(
    condition: dict,
    *,
    seed: int,
    repeat_count: int,
    evaluation_count: int,
    update_point_count: int,
    enforce_exit_gate: bool = True,
) -> dict:
    records = []
    for repeat in range(repeat_count):
        repeat_seed = seed + repeat
        rng = np.random.default_rng(repeat_seed)
        models = _models(condition, repeat_seed)
        cache = _CacheTracker()
        editor = ModelEditor(models, invalidate=cache.invalidate)
        baseline_snapshot = editor.snapshot()
        baseline_bytes = len(baseline_snapshot)
        evaluation_points = rng.normal(
            scale=3.0, size=(evaluation_count, condition["dimensions"]),
        )
        baseline_scores, baseline_inference = _scores(models, evaluation_points)
        baseline_routing_counts = _routing_counts(models, len(evaluation_points))
        cache.entries.update(baseline_scores)
        baseline_predictions = _predictions(baseline_scores)
        target_center = models[0][0].ellipsoids[0].center
        update_points = target_center + rng.normal(
            scale=0.45,
            size=(update_point_count, condition["dimensions"]),
        )

        insertion = editor.insert_additive_from_points(
            0, update_points, _fit_update, seed=repeat_seed,
        )
        inserted_snapshot_bytes = len(editor.snapshot())
        inserted_scores, inserted_inference = _scores(models, evaluation_points)
        inserted_routing_counts = _routing_counts(models, len(evaluation_points))
        inserted_predictions = _predictions(inserted_scores)
        insertion_local = all(
            np.array_equal(inserted_scores[class_id], baseline_scores[class_id])
            for class_id in range(1, condition["class_count"])
        )

        deletion = editor.delete_primitive(
            0, insertion["expert_index"], insertion["ellipsoid_index"],
        )
        deleted_scores, _ = _scores(models, evaluation_points)
        deletion_exact = editor.snapshot() == baseline_snapshot
        deletion_restored = np.array_equal(
            _predictions(deleted_scores), baseline_predictions,
        )

        editor.insert_additive_from_points(
            0, update_points, _fit_update, seed=repeat_seed,
        )
        rollback = editor.rollback(baseline_snapshot)
        rollback_exact = editor.snapshot() == baseline_snapshot

        nudge = editor.local_nudge(
            0, update_points, learning_rate=0.1, iterations=2,
        )
        nudged_scores, _ = _scores(models, evaluation_points)
        nudge_local = all(
            np.array_equal(nudged_scores[class_id], baseline_scores[class_id])
            for class_id in range(1, condition["class_count"])
        )

        started = time.perf_counter()
        reconstructed_class = _class_model(
            0,
            condition["class_count"],
            condition["dimensions"],
            condition["primitives_per_class"],
            repeat_seed,
        )
        reconstructed_class[0].add_ellipsoid(_fit_update(update_points, repeat_seed))
        target_reconstruction = time.perf_counter() - started

        started = time.perf_counter()
        reconstructed_models = _models(condition, repeat_seed)
        reconstructed_models[0][0].add_ellipsoid(
            _fit_update(update_points, repeat_seed),
        )
        full_reconstruction = time.perf_counter() - started

        checks = {
            "insertion_untouched_sdf_bitwise_stable": insertion_local,
            "deletion_exact_snapshot": deletion_exact,
            "deletion_restored_predictions": bool(deletion_restored),
            "rollback_exact_snapshot": rollback_exact,
            "nudge_untouched_sdf_bitwise_stable": nudge_local,
        }
        if enforce_exit_gate and not all(checks.values()):
            raise AssertionError(f"M9 locality gate failed: {checks}")
        records.append({
            "repeat": repeat,
            "seed": repeat_seed,
            "serialized_bytes": {
                "baseline": baseline_bytes,
                "after_insertion": inserted_snapshot_bytes,
                "insertion_growth": inserted_snapshot_bytes - baseline_bytes,
            },
            "prediction_changes": {
                "insertion": int(np.count_nonzero(
                    inserted_predictions != baseline_predictions,
                )),
                "nudge": int(np.count_nonzero(
                    _predictions(nudged_scores) != baseline_predictions,
                )),
            },
            "latency_seconds": {
                "baseline_inference": baseline_inference,
                "inserted_inference": inserted_inference,
                "point_fitted_insertion": insertion["edit_seconds"],
                "deletion": deletion["edit_seconds"],
                "rollback": rollback["edit_seconds"],
                "local_nudge": nudge["edit_seconds"],
                "target_class_reconstruction": target_reconstruction,
                "full_model_reconstruction": full_reconstruction,
                "cache_invalidation_total": cache.invalidation_seconds,
            },
            "routing_counts": {
                "baseline": asdict(baseline_routing_counts),
                "after_insertion": asdict(inserted_routing_counts),
            },
            "cache_invalidations": cache.invalidations,
            "checks": checks,
        })

    latency_names = records[0]["latency_seconds"]
    return {
        "condition": condition,
        "total_primitives": (
            condition["class_count"] * condition["primitives_per_class"]
        ),
        "records": records,
        "latency_seconds": {
            name: _summary([record["latency_seconds"][name] for record in records])
            for name in latency_names
        },
        "serialized_bytes": {
            name: _summary([
                record["serialized_bytes"][name] for record in records
            ])
            for name in records[0]["serialized_bytes"]
        },
        "prediction_changes": {
            name: _summary([
                record["prediction_changes"][name] for record in records
            ])
            for name in records[0]["prediction_changes"]
        },
        "routing_counts": {
            stage: {
                name: _summary([
                    record["routing_counts"][stage][name] for record in records
                ])
                for name in records[0]["routing_counts"][stage]
            }
            for stage in records[0]["routing_counts"]
        },
        "cache_invalidations": _summary([
            record["cache_invalidations"] for record in records
        ]),
        "all_exit_gates_passed": all(
            all(record["checks"].values()) for record in records
        ),
    }


def run_benchmark(config: dict) -> dict:
    conditions = build_scaling_conditions(config)
    return {
        "protocol": {
            "design": "one_factor_at_a_time",
            "seed": config["seed"],
            "repeat_count": config["repeat_count"],
            "evaluation_count": config["evaluation_count"],
            "update_point_count": config["update_point_count"],
            "evaluation_used_for_fitting": False,
            "memory_measure": "canonical_serialized_model_bytes",
            "comparators": [
                "target_class_reconstruction",
                "full_model_reconstruction",
            ],
            "comparator_scope_note": (
                "Synthetic deterministic reconstruction, not optimizer retraining."
            ),
        },
        "conditions": [
            run_scaling_condition(
                condition,
                seed=config["seed"],
                repeat_count=config["repeat_count"],
                evaluation_count=config["evaluation_count"],
                update_point_count=config["update_point_count"],
                enforce_exit_gate=config.get("enforce_exit_gate", True),
            )
            for condition in conditions
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GEODE M9 editability scaling study")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = run_benchmark(config)
    output_path = Path(config["artifact_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()