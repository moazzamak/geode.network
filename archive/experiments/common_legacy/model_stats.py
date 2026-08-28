from collections.abc import Iterable, Mapping

import numpy as np

from src.sdf_engine import Expert


def _class_models(models) -> list[list[Expert]]:
    if isinstance(models, Mapping):
        return [list(experts) for experts in models.values()]
    if isinstance(models, Iterable):
        return [list(models)]
    raise TypeError("models must be an expert iterable or class-to-experts mapping")


def model_structure_stats(models, candidate_evaluations: int = 0) -> dict[str, int]:
    class_models = _class_models(models)
    experts = [expert for class_model in class_models for expert in class_model]
    ellipsoids = [
        ellipsoid for expert in experts for ellipsoid in expert.ellipsoids
    ]
    additive = sum(ellipsoid.polarity > 0 for ellipsoid in ellipsoids)
    subtractive = sum(ellipsoid.polarity < 0 for ellipsoid in ellipsoids)
    fitted_parameters = sum(
        ellipsoid.center.size
        + ellipsoid.radii.size
        + ellipsoid.orientation.size
        for ellipsoid in ellipsoids
    )
    approximate_bytes = sum(
        np.asarray(ellipsoid.center).nbytes
        + np.asarray(ellipsoid.radii).nbytes
        + np.asarray(ellipsoid.orientation).nbytes
        for ellipsoid in ellipsoids
    )
    return {
        "classes": len(class_models),
        "empty_classes": sum(not class_model for class_model in class_models),
        "experts": len(experts),
        "additive_ellipsoids": additive,
        "subtractive_ellipsoids": subtractive,
        "fitted_parameters": fitted_parameters,
        "candidate_evaluations": int(candidate_evaluations),
        "approximate_model_bytes": approximate_bytes,
    }


def performance_stats(
    fit_seconds: float,
    inference_seconds: float,
    inference_samples: int,
    peak_process_bytes: int | None = None,
) -> dict[str, float | int | None]:
    throughput = (
        float(inference_samples) / float(inference_seconds)
        if inference_seconds > 0 else None
    )
    return {
        "fit_seconds": float(fit_seconds),
        "inference_seconds": float(inference_seconds),
        "inference_samples_per_second": throughput,
        "peak_process_bytes": (
            int(peak_process_bytes) if peak_process_bytes is not None else None
        ),
    }