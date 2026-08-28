from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import logsumexp, softmax

from experiments.common.v5_artifacts import payload_hash
from experiments.common.v7_protocol import ConfirmationEvent, GraphMigrationSpec
from src.subspace_primitive import SubspacePrimitive, fit_subspace_primitive


@dataclass(frozen=True)
class GaussianClassState:
    label: int
    components: tuple[SubspacePrimitive, ...]

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("Gaussian classes require at least one component.")
        if any(component.class_label != self.label for component in self.components):
            raise ValueError("Component labels must match their Gaussian class.")


@dataclass(frozen=True)
class GaussianBundle:
    classes: tuple[GaussianClassState, ...]
    threshold: float
    parent_hash: str | None = None
    confirmation_id: str | None = None

    @property
    def class_order(self) -> tuple[int, ...]:
        return tuple(state.label for state in self.classes)

    def class_log_likelihoods(self, features: np.ndarray) -> np.ndarray:
        columns = []
        for state in self.classes:
            component_values = np.column_stack(
                [component.log_likelihood(features) for component in state.components]
            )
            columns.append(
                logsumexp(component_values, axis=1) - np.log(len(state.components))
            )
        return np.column_stack(columns)

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        likelihoods = self.class_log_likelihoods(features)
        columns = np.argmax(likelihoods, axis=1)
        return (
            np.asarray(self.class_order, dtype=np.int64)[columns],
            -np.max(likelihoods, axis=1),
        )

    def state_payload(self) -> dict[str, Any]:
        return {
            "class_order": list(self.class_order),
            "threshold": self.threshold,
            "parent_hash": self.parent_hash,
            "confirmation_id": self.confirmation_id,
            "components": [
                [
                    {
                        "center": component.center.tolist(),
                        "basis": component.basis.tolist(),
                        "tangent_variances": component.tangent_variances.tolist(),
                        "residual_variance": component.residual_variance,
                    }
                    for component in state.components
                ]
                for state in self.classes
            ],
        }

    @property
    def bundle_hash(self) -> str:
        return payload_hash(self.state_payload())


class GaussianAdaptationTransaction:
    def __init__(self, parent: GaussianBundle) -> None:
        self.parent = parent
        self.child: GaussianBundle | None = None

    def apply(
        self,
        *,
        confirmation: ConfirmationEvent | None,
        label: int,
        support: np.ndarray,
        rank: int,
        operation: str,
        original_class_support: np.ndarray | None = None,
    ) -> GaussianBundle:
        if confirmation is None:
            raise PermissionError("Adaptation requires a human confirmation event.")
        if confirmation.response not in {"existing_class", "new_class"}:
            raise PermissionError("The confirmation does not authorize adaptation.")
        if confirmation.confirmed_label != str(label):
            raise ValueError("Confirmed label and requested adaptation disagree.")
        support_values = np.asarray(support, dtype=np.float64)
        existing = label in self.parent.class_order
        if existing != (confirmation.response == "existing_class"):
            raise ValueError("Confirmation type does not match bundle class order.")
        if operation not in {
            "native_gaussian",
            "sdf_component",
            "full_class_local_refit",
            "full_model_retrain",
        }:
            raise ValueError("Unsupported adaptation operation.")

        states = list(self.parent.classes)
        if existing:
            index = self.parent.class_order.index(label)
            current = states[index]
            if operation == "sdf_component":
                component = fit_subspace_primitive(
                    support_values,
                    min(rank, support_values.shape[1] - 1, len(support_values) - 2),
                    class_label=label,
                )
                states[index] = GaussianClassState(
                    label=label,
                    components=current.components + (component,),
                )
            else:
                if original_class_support is None:
                    raise ValueError("Class refits require original class support.")
                combined = np.concatenate([original_class_support, support_values])
                component = fit_subspace_primitive(
                    combined,
                    min(rank, combined.shape[1] - 1, len(combined) - 2),
                    class_label=label,
                )
                states[index] = GaussianClassState(label=label, components=(component,))
        else:
            component = fit_subspace_primitive(
                support_values,
                min(rank, support_values.shape[1] - 1, len(support_values) - 2),
                class_label=label,
            )
            states.append(GaussianClassState(label=label, components=(component,)))
        self.child = GaussianBundle(
            classes=tuple(states),
            threshold=self.parent.threshold,
            parent_hash=self.parent.bundle_hash,
            confirmation_id=f"{confirmation.review_id}:{confirmation.confirmed_window}",
        )
        return self.child

    def rollback(self) -> GaussianBundle:
        if self.child is None:
            raise RuntimeError("No adaptation has been applied.")
        self.child = None
        return self.parent


def fit_gaussian_bundle(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    rank: int,
    threshold: float,
) -> GaussianBundle:
    states = []
    for label in np.unique(labels):
        points = features[labels == label]
        states.append(
            GaussianClassState(
                label=int(label),
                components=(
                    fit_subspace_primitive(
                        points,
                        min(rank, points.shape[1] - 1, len(points) - 2),
                        class_label=int(label),
                    ),
                ),
            )
        )
    return GaussianBundle(classes=tuple(states), threshold=threshold)


def bundle_metrics(
    bundle: GaussianBundle,
    known_x: np.ndarray,
    known_y: np.ndarray,
    target_x: np.ndarray,
    target_label: int,
    unknown_x: np.ndarray,
) -> dict[str, float]:
    known_prediction, known_novelty = bundle.predict(known_x)
    target_prediction, target_novelty = bundle.predict(target_x)
    _, unknown_novelty = bundle.predict(unknown_x)
    known_likelihoods = bundle.class_log_likelihoods(known_x)
    probabilities = softmax(known_likelihoods, axis=1)
    columns = {label: index for index, label in enumerate(bundle.class_order)}
    known_target_columns = np.asarray([columns[int(label)] for label in known_y])
    known_nll = -float(
        np.mean(
            np.log(
                np.maximum(
                    probabilities[np.arange(len(known_y)), known_target_columns],
                    np.finfo(np.float64).tiny,
                )
            )
        )
    )
    return {
        "known_balanced_accuracy": float(
            np.mean(known_prediction == known_y)
        ),
        "known_coverage": float(np.mean(known_novelty <= bundle.threshold)),
        "target_success": float(
            np.mean(
                (target_prediction == target_label)
                & (target_novelty <= bundle.threshold)
            )
        ),
        "unknown_recall": float(np.mean(unknown_novelty > bundle.threshold)),
        "known_nll": known_nll,
    }


def new_class_migration(
    parent: GaussianBundle,
    child: GaussianBundle,
    confirmation: ConfirmationEvent,
) -> GraphMigrationSpec:
    return GraphMigrationSpec(
        parent_bundle_hash=parent.bundle_hash,
        parent_class_order=tuple(str(value) for value in parent.class_order),
        child_class_order=tuple(str(value) for value in child.class_order),
        review_id=confirmation.review_id,
        confirmation_id=f"{confirmation.review_id}:{confirmation.confirmed_window}",
        rollback_bundle_hash=parent.bundle_hash,
    )
