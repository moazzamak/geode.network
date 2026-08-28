"""Deterministic affine interface adapter for frozen backbone representations.

Implements identity, linear projection, and low-rank affine maps trained once
under the pre-test development protocol, then frozen forever. The backbone
never receives gradients. Nonlinear or residual adapters are out of scope.

Objective: L = L_CE + lambda_compact * L_within + lambda_margin * L_between + lambda_complexity * Omega
Each term is independent. A temporary linear classifier used by the CE term is
discarded before the interface is serialized/hashed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common.experiment_manifest import canonical_json


# ---------------------------------------------------------------------------
# Objective terms
# ---------------------------------------------------------------------------


def cross_entropy_loss(
    features: np.ndarray,
    labels: np.ndarray,
    classifier_weights: np.ndarray,
    classifier_bias: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Cross-entropy with linear classifier. Returns (loss, dL/d_features, dW, dB)."""
    logits = features @ classifier_weights + classifier_bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    n = len(labels)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), labels] = 1.0
    residual = probs - one_hot

    loss = -np.mean(np.log(np.clip(probs[np.arange(n), labels], 1e-12, 1.0)))
    grad_features = (residual @ classifier_weights.T) / n
    grad_w = (features.T @ residual) / n
    grad_b = residual.mean(axis=0)
    return float(loss), grad_features, grad_w, grad_b


def within_class_compactness(
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Mean squared distance to class centroid. Returns (loss, gradient)."""
    classes = np.unique(labels)
    n = len(features)
    grad = np.zeros_like(features)
    total_loss = 0.0
    for c in classes:
        mask = labels == c
        class_features = features[mask]
        centroid = class_features.mean(axis=0)
        diffs = class_features - centroid
        total_loss += np.sum(diffs ** 2)
        nc = int(mask.sum())
        grad[mask] = 2.0 * diffs * (1.0 - 1.0 / nc)
    total_loss /= n
    grad /= n
    return float(total_loss), grad


def between_class_margin(
    features: np.ndarray,
    labels: np.ndarray,
    target_margin: float = 1.0,
) -> tuple[float, np.ndarray]:
    """Hinge loss on pairwise centroid distances below target margin."""
    classes = np.unique(labels)
    k = len(classes)
    if k < 2:
        return 0.0, np.zeros_like(features)

    centroids = np.array([features[labels == c].mean(axis=0) for c in classes])
    n = len(features)
    grad = np.zeros_like(features)
    total_loss = 0.0
    n_pairs = 0

    for i in range(k):
        for j in range(i + 1, k):
            diff = centroids[i] - centroids[j]
            dist = float(np.sqrt(np.sum(diff ** 2) + 1e-12))
            violation = target_margin - dist
            if violation > 0.0:
                total_loss += violation
                n_pairs += 1
                direction = diff / (dist + 1e-12)
                ni = int((labels == classes[i]).sum())
                nj = int((labels == classes[j]).sum())
                mask_i = labels == classes[i]
                mask_j = labels == classes[j]
                grad[mask_i] -= direction / ni
                grad[mask_j] += direction / nj

    num_total_pairs = k * (k - 1) // 2
    total_loss /= num_total_pairs
    grad /= num_total_pairs
    return float(total_loss), grad


def complexity_penalty(
    weight: np.ndarray,
    bias: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Frobenius norm regularizer on interface parameters."""
    loss = float(np.sum(weight ** 2) + np.sum(bias ** 2))
    grad_w = 2.0 * weight
    grad_b = 2.0 * bias
    return loss, grad_w, grad_b


# ---------------------------------------------------------------------------
# Interface architectures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InterfaceConfig:
    """Configuration for an affine interface."""
    architecture: str  # "identity", "linear", "low_rank"
    input_dim: int
    output_dim: int
    rank: int = 0  # only for low_rank

    def __post_init__(self) -> None:
        if self.architecture not in ("identity", "linear", "low_rank"):
            raise ValueError(f"Unsupported architecture: {self.architecture!r}")
        if self.input_dim < 1 or self.output_dim < 1:
            raise ValueError("Dimensions must be positive.")
        if self.architecture == "identity" and self.input_dim != self.output_dim:
            raise ValueError("Identity interface requires input_dim == output_dim.")
        if self.architecture == "low_rank" and self.rank < 1:
            raise ValueError("Low-rank interface requires rank >= 1.")
        if self.architecture == "low_rank" and self.rank > min(self.input_dim, self.output_dim):
            raise ValueError("Rank exceeds min(input_dim, output_dim).")


class AffineInterface:
    """Trainable affine map: y = x @ W + b (or identity)."""

    def __init__(self, config: InterfaceConfig, seed: int = 11) -> None:
        self.config = config
        self.seed = seed
        rng = np.random.default_rng(seed)

        if config.architecture == "identity":
            self.weight = np.eye(config.input_dim, dtype=np.float64)
            self.bias = np.zeros(config.output_dim, dtype=np.float64)
        elif config.architecture == "linear":
            scale = np.sqrt(2.0 / (config.input_dim + config.output_dim))
            self.weight = rng.normal(0.0, scale, (config.input_dim, config.output_dim)).astype(np.float64)
            self.bias = np.zeros(config.output_dim, dtype=np.float64)
        elif config.architecture == "low_rank":
            scale = np.sqrt(2.0 / (config.input_dim + config.output_dim))
            u = rng.normal(0.0, scale, (config.input_dim, config.rank)).astype(np.float64)
            v = rng.normal(0.0, scale, (config.rank, config.output_dim)).astype(np.float64)
            self.weight = u @ v
            self.bias = np.zeros(config.output_dim, dtype=np.float64)
            self._u = u
            self._v = v
        else:
            raise ValueError(f"Unsupported architecture: {config.architecture!r}")

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Apply the frozen affine map."""
        return features @ self.weight + self.bias

    @property
    def parameter_count(self) -> int:
        if self.config.architecture == "identity":
            return 0
        return int(self.weight.size + self.bias.size)

    @property
    def serialized_bytes(self) -> int:
        payload = self.to_dict()
        return len(canonical_json(payload).encode("utf-8"))

    def weights_digest(self) -> str:
        """SHA-256 of canonical weight serialization."""
        payload = {
            "weight": self.weight.tolist(),
            "bias": self.bias.tolist(),
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Canonical serialization (no temporary classifier)."""
        return {
            "schema_version": 1,
            "architecture": self.config.architecture,
            "input_dim": self.config.input_dim,
            "output_dim": self.config.output_dim,
            "rank": self.config.rank,
            "seed": self.seed,
            "weight": self.weight.tolist(),
            "bias": self.bias.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AffineInterface":
        """Restore from canonical serialization."""
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported interface schema version.")
        config = InterfaceConfig(
            architecture=payload["architecture"],
            input_dim=payload["input_dim"],
            output_dim=payload["output_dim"],
            rank=payload.get("rank", 0),
        )
        instance = cls.__new__(cls)
        instance.config = config
        instance.seed = payload["seed"]
        instance.weight = np.asarray(payload["weight"], dtype=np.float64)
        instance.bias = np.asarray(payload["bias"], dtype=np.float64)
        if instance.weight.shape != (config.input_dim, config.output_dim):
            raise ValueError("Weight shape mismatch.")
        if instance.bias.shape != (config.output_dim,):
            raise ValueError("Bias shape mismatch.")
        return instance

    def save(self, path: str | Path) -> str:
        """Save to JSON. Returns SHA-256 of the file content."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        content = canonical_json(self.to_dict()) + "\n"
        out.write_text(content, encoding="utf-8", newline="\n")
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @classmethod
    def load(cls, path: str | Path) -> "AffineInterface":
        """Load from JSON with integrity check."""
        content = Path(path).read_text(encoding="utf-8")
        payload = json.loads(content)
        return cls.from_dict(payload)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LambdaTuple:
    """Hyperparameter tuple for the interface objective."""
    compact: float
    margin: float
    complexity: float

    def __post_init__(self) -> None:
        for name, val in [("compact", self.compact), ("margin", self.margin),
                          ("complexity", self.complexity)]:
            if not np.isfinite(val) or val < 0.0:
                raise ValueError(f"lambda_{name} must be finite and non-negative.")

    def to_tuple(self) -> tuple[float, float, float]:
        return (self.compact, self.margin, self.complexity)


def train_interface(
    config: InterfaceConfig,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    dev_features: np.ndarray,
    dev_labels: np.ndarray,
    lambdas: LambdaTuple,
    *,
    seed: int = 11,
    learning_rate: float = 0.01,
    max_epochs: int = 200,
    patience: int = 20,
    batch_size: int = 256,
    target_margin: float = 1.0,
) -> tuple[AffineInterface, dict[str, Any]]:
    """Train an affine interface under the development protocol.

    Uses train data for gradient updates and dev data only for early stopping.
    The temporary classifier is discarded. Returns (frozen_interface, training_log).
    """
    if config.architecture == "identity":
        interface = AffineInterface(config, seed=seed)
        log = {
            "architecture": "identity",
            "epochs": 0,
            "final_dev_loss": 0.0,
            "converged": True,
        }
        return interface, log

    interface = AffineInterface(config, seed=seed)
    all_labels = np.concatenate([train_labels, dev_labels])
    n_classes = int(np.max(all_labels) + 1)
    rng = np.random.default_rng(seed + 1000)

    # Temporary classifier (discarded after training)
    cls_w = rng.normal(0.0, 0.01, (config.output_dim, n_classes)).astype(np.float64)
    cls_b = np.zeros(n_classes, dtype=np.float64)

    best_dev_loss = np.inf
    best_weight = interface.weight.copy()
    best_bias = interface.bias.copy()
    epochs_without_improvement = 0
    n_train = len(train_features)

    history: list[dict[str, float]] = []

    for epoch in range(max_epochs):
        # Deterministic epoch order
        indices = rng.permutation(n_train)

        for start in range(0, n_train, batch_size):
            batch_idx = indices[start:start + batch_size]
            x_batch = train_features[batch_idx]
            y_batch = train_labels[batch_idx]

            # Forward
            transformed = x_batch @ interface.weight + interface.bias

            # CE term
            ce_loss, grad_ce_feat, grad_cls_w, grad_cls_b = cross_entropy_loss(
                transformed, y_batch, cls_w, cls_b
            )

            # Within-class compactness
            compact_loss, grad_compact_feat = within_class_compactness(transformed, y_batch)

            # Between-class margin
            margin_loss, grad_margin_feat = between_class_margin(
                transformed, y_batch, target_margin
            )

            # Total feature gradient
            grad_feat = grad_ce_feat
            grad_feat = grad_feat + lambdas.compact * grad_compact_feat
            grad_feat = grad_feat + lambdas.margin * grad_margin_feat

            # Backprop through affine: d/dW = x^T @ grad_feat, d/db = sum(grad_feat)
            grad_w = x_batch.T @ grad_feat / len(batch_idx)
            grad_b_iface = grad_feat.mean(axis=0)

            # Complexity penalty on interface weights
            comp_loss, comp_grad_w, comp_grad_b = complexity_penalty(
                interface.weight, interface.bias
            )
            grad_w = grad_w + lambdas.complexity * comp_grad_w
            grad_b_iface = grad_b_iface + lambdas.complexity * comp_grad_b

            # Update interface
            interface.weight -= learning_rate * grad_w
            interface.bias -= learning_rate * grad_b_iface

            # Update temporary classifier
            cls_w -= learning_rate * grad_cls_w
            cls_b -= learning_rate * grad_cls_b

        # Dev loss for early stopping
        dev_transformed = dev_features @ interface.weight + interface.bias
        dev_ce, _, _, _ = cross_entropy_loss(dev_transformed, dev_labels, cls_w, cls_b)
        dev_compact, _ = within_class_compactness(dev_transformed, dev_labels)
        dev_margin, _ = between_class_margin(dev_transformed, dev_labels, target_margin)
        dev_comp, _, _ = complexity_penalty(interface.weight, interface.bias)
        dev_loss = (dev_ce + lambdas.compact * dev_compact +
                    lambdas.margin * dev_margin + lambdas.complexity * dev_comp)

        history.append({"epoch": epoch, "dev_loss": float(dev_loss), "dev_ce": float(dev_ce)})

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_weight = interface.weight.copy()
            best_bias = interface.bias.copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    # Restore best and discard classifier
    interface.weight = best_weight
    interface.bias = best_bias

    log = {
        "architecture": config.architecture,
        "epochs": epoch + 1,
        "final_dev_loss": float(best_dev_loss),
        "converged": epochs_without_improvement >= patience,
        "lambdas": lambdas.to_tuple(),
    }
    return interface, log


def select_lambda_tuple(
    config: InterfaceConfig,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    dev_features: np.ndarray,
    dev_labels: np.ndarray,
    tuples: list[LambdaTuple],
    *,
    seed: int = 11,
    max_tuples: int = 16,
    **train_kwargs: Any,
) -> tuple[LambdaTuple, list[dict[str, Any]]]:
    """Select the best lambda tuple by dev loss. Cap at max_tuples."""
    if len(tuples) > max_tuples:
        raise ValueError(f"At most {max_tuples} tuples allowed, got {len(tuples)}.")
    if not tuples:
        raise ValueError("At least one lambda tuple required.")

    results: list[dict[str, Any]] = []
    best_loss = np.inf
    best_tuple = tuples[0]

    for lt in tuples:
        interface, log = train_interface(
            config, train_features, train_labels,
            dev_features, dev_labels, lt,
            seed=seed, **train_kwargs,
        )
        results.append({
            "lambdas": lt.to_tuple(),
            "dev_loss": log["final_dev_loss"],
            "epochs": log["epochs"],
        })
        if log["final_dev_loss"] < best_loss:
            best_loss = log["final_dev_loss"]
            best_tuple = lt

    return best_tuple, results
