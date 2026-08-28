from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


REGISTERED_COMPONENT_COUNTS = (10, 20, 30, 46, 60, 80, 100, 120)


def marginal_accuracy_per_ten(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, float | int | None]]:
    if [int(row["component_count"]) for row in rows] != list(
        REGISTERED_COMPONENT_COUNTS
    ):
        raise ValueError("Budget rows must use the registered component counts.")
    output: list[dict[str, float | int | None]] = []
    previous: dict[str, Any] | None = None
    for row in rows:
        if previous is None:
            marginal = None
        else:
            count_delta = int(row["component_count"]) - int(
                previous["component_count"]
            )
            marginal = (
                float(row["development_balanced_accuracy"])
                - float(previous["development_balanced_accuracy"])
            ) * 10.0 / count_delta
        output.append(
            {
                "component_count": int(row["component_count"]),
                "marginal_accuracy_per_ten": marginal,
            }
        )
        previous = row
    return output


def classify_capacity_curve(
    rows: Sequence[dict[str, Any]],
    *,
    minimum_high_budget_slope_per_ten: float,
    material_accuracy_reversal: float,
    material_nll_reversal_fraction: float,
) -> dict[str, Any]:
    counts = [int(row["component_count"]) for row in rows]
    if counts != list(REGISTERED_COMPONENT_COUNTS):
        raise ValueError("Capacity classification requires the registered curve.")
    if (
        minimum_high_budget_slope_per_ten <= 0.0
        or material_accuracy_reversal <= 0.0
        or material_nll_reversal_fraction <= 0.0
    ):
        raise ValueError("Capacity classification thresholds must be positive.")
    high = [row for row in rows if int(row["component_count"]) >= 80]
    reversals = []
    for previous, current in zip(high, high[1:]):
        accuracy_change = float(current["development_balanced_accuracy"]) - float(
            previous["development_balanced_accuracy"]
        )
        nll_change_fraction = (
            float(current["development_nll"]) - float(previous["development_nll"])
        ) / float(previous["development_nll"])
        if (
            accuracy_change <= -material_accuracy_reversal
            or nll_change_fraction >= material_nll_reversal_fraction
        ):
            reversals.append(
                {
                    "from_components": int(previous["component_count"]),
                    "to_components": int(current["component_count"]),
                    "accuracy_change": accuracy_change,
                    "nll_change_fraction": nll_change_fraction,
                }
            )
    endpoint_slope = (
        float(high[-1]["development_balanced_accuracy"])
        - float(high[0]["development_balanced_accuracy"])
    ) * 10.0 / (
        int(high[-1]["component_count"]) - int(high[0]["component_count"])
    )
    marginal = marginal_accuracy_per_ten(rows)
    high_marginals = [
        float(item["marginal_accuracy_per_ten"])
        for item in marginal
        if int(item["component_count"]) > 80
        and item["marginal_accuracy_per_ten"] is not None
    ]
    best_gain_above_80 = max(high_marginals)
    if reversals:
        classification = "unstable"
    elif endpoint_slope >= minimum_high_budget_slope_per_ten:
        classification = "budget-limited"
    else:
        classification = "saturated"
    return {
        "classification": classification,
        "endpoint_slope_per_ten": endpoint_slope,
        "best_marginal_gain_above_80_per_ten": best_gain_above_80,
        "material_reversals": reversals,
        "thresholds": {
            "minimum_high_budget_slope_per_ten": minimum_high_budget_slope_per_ten,
            "material_accuracy_reversal": material_accuracy_reversal,
            "material_nll_reversal_fraction": material_nll_reversal_fraction,
        },
    }


def probability_margin_error(
    student_probabilities: np.ndarray,
    teacher_probabilities: np.ndarray,
) -> float:
    student = np.asarray(student_probabilities, dtype=np.float64)
    teacher = np.asarray(teacher_probabilities, dtype=np.float64)
    if (
        student.shape != teacher.shape
        or student.ndim != 2
        or student.shape[1] < 2
        or not np.all(np.isfinite(student))
        or not np.all(np.isfinite(teacher))
    ):
        raise ValueError("Student and teacher probabilities must be matching matrices.")
    student_sorted = np.sort(student, axis=1)
    teacher_sorted = np.sort(teacher, axis=1)
    student_margin = student_sorted[:, -1] - student_sorted[:, -2]
    teacher_margin = teacher_sorted[:, -1] - teacher_sorted[:, -2]
    return float(np.mean(np.abs(student_margin - teacher_margin)))
