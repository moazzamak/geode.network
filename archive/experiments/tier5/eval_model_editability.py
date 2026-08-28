import argparse
import json
from pathlib import Path

import numpy as np

from src.inference_engine import InferenceEngine
from src.model_editor import ModelEditor
from src.sdf_engine import EllipsoidExpert, Expert


def _base_models() -> dict[int, list[Expert]]:
    models = {}
    for class_id, center in enumerate(([0.0, 0.0], [4.0, 0.0], [0.0, 4.0])):
        expert = Expert(alpha=2.0)
        expert.add_ellipsoid(EllipsoidExpert(
            center=np.asarray(center), radii=np.array([1.0, 1.0]),
        ))
        models[class_id] = [expert]
    return models


def _scores(models: dict[int, list[Expert]], points: np.ndarray) -> dict[int, np.ndarray]:
    return {
        class_id: InferenceEngine(experts, alpha=2.0).get_fused_sdf(points)
        for class_id, experts in sorted(models.items())
    }


def _predictions(scores: dict[int, np.ndarray]) -> np.ndarray:
    class_ids = np.array(sorted(scores))
    return class_ids[np.argmin(np.column_stack([
        scores[class_id] for class_id in class_ids
    ]), axis=1)]


def _fit_ellipsoid(points: np.ndarray, _seed: int) -> EllipsoidExpert:
    return EllipsoidExpert(
        center=np.mean(points, axis=0),
        radii=np.maximum(np.std(points, axis=0), 0.35),
    )


def run_benchmark() -> dict:
    axis = np.linspace(-1.0, 5.0, 49)
    points = np.array(np.meshgrid(axis, axis)).reshape(2, -1).T
    editor = ModelEditor(_base_models())
    baseline_snapshot = editor.snapshot()
    baseline_scores = _scores(editor.models, points)
    baseline_predictions = _predictions(baseline_scores)
    insertion_points = np.array([
        [3.7, 3.7], [4.0, 3.8], [4.2, 4.0], [3.8, 4.2], [4.1, 4.1],
    ])

    insertion = editor.insert_additive_from_points(
        0, insertion_points, _fit_ellipsoid, seed=101,
    )
    inserted_scores = _scores(editor.models, points)
    inserted_predictions = _predictions(inserted_scores)
    insertion_changes = int(np.count_nonzero(
        inserted_predictions != baseline_predictions,
    ))
    insertion_local = all(
        np.array_equal(inserted_scores[class_id], baseline_scores[class_id])
        for class_id in (1, 2)
    )

    deletion = editor.delete_primitive(
        0, insertion["expert_index"], insertion["ellipsoid_index"],
    )
    deleted_scores = _scores(editor.models, points)
    deletion_restored_predictions = bool(np.array_equal(
        _predictions(deleted_scores), baseline_predictions,
    ))
    deletion_local = all(
        np.array_equal(deleted_scores[class_id], baseline_scores[class_id])
        for class_id in (1, 2)
    )

    editor.insert_additive_from_points(0, insertion_points, _fit_ellipsoid, seed=101)
    rollback = editor.rollback(baseline_snapshot)
    rollback_scores = _scores(editor.models, points)
    rollback_exact = editor.snapshot() == baseline_snapshot
    rollback_restored_predictions = bool(np.array_equal(
        _predictions(rollback_scores), baseline_predictions,
    ))

    nudge = editor.local_nudge(
        0, np.array([[0.6, 0.0], [0.8, 0.1], [0.7, -0.1]]),
        learning_rate=0.25,
        iterations=2,
    )
    nudged_scores = _scores(editor.models, points)
    nudge_changes = int(np.count_nonzero(
        _predictions(nudged_scores) != baseline_predictions,
    ))
    nudge_local = all(
        np.array_equal(nudged_scores[class_id], baseline_scores[class_id])
        for class_id in (1, 2)
    )

    checks = {
        "insertion_changed_predictions": insertion_changes > 0,
        "insertion_untouched_class_sdf_bitwise_stable": insertion_local,
        "deletion_restored_predictions": deletion_restored_predictions,
        "deletion_untouched_class_sdf_bitwise_stable": deletion_local,
        "rollback_exact_snapshot": rollback_exact,
        "rollback_restored_predictions": rollback_restored_predictions,
        "nudge_untouched_class_sdf_bitwise_stable": nudge_local,
    }
    if not all(checks.values()):
        raise AssertionError(f"Editability exit gate failed: {checks}")
    return {
        "protocol": {
            "evaluation_count": len(points),
            "target_class": 0,
            "untouched_classes": [1, 2],
            "evaluation_used_for_fitting": False,
        },
        "prediction_changes": {
            "insertion": insertion_changes,
            "deletion_relative_to_inserted": int(np.count_nonzero(
                _predictions(deleted_scores) != inserted_predictions,
            )),
            "nudge": nudge_changes,
        },
        "latency_seconds": {
            "point_fitted_insertion": insertion["edit_seconds"],
            "deletion": deletion["edit_seconds"],
            "rollback": rollback["edit_seconds"],
            "local_nudge": nudge["edit_seconds"],
        },
        "checks": checks,
        "audit_record_count": len(editor.audit_log),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GEODE localized editability benchmark")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/results/tier5_model_editability.json"),
    )
    args = parser.parse_args()
    result = run_benchmark()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()