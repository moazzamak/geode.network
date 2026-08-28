"""Analytic gradient-based parameter refinement for GEODE EllipsoidExpert models.

Mathematical basis
------------------
An EllipsoidExpert stores its geometry as (center μ, radii r, orientation R)
where the SDF is:

    φ(x; μ, R, r) = √( δᵀ P δ ) − 1,   δ = x − μ,   P = R diag(r⁻²) Rᵀ

Because ``compute_sdf`` uses ``local = (x − c) @ R`` (row-vector convention),
the precision matrix reconstructed from stored parameters is:

    P = R diag(r⁻²) Rᵀ     [i.e.  (R * r⁻²) @ R.T  in NumPy]

Gradients w.r.t. the natural parameterisation exist in closed form:

    q        = √( δᵀ P δ )   (= φ + 1, the unnormalised Mahalanobis distance)
    ∂φ/∂μ    = −Pδ / q                 ∈ ℝᵈ
    ∂φ/∂P    = δδᵀ / (2q)             ∈ ℝᵈˣᵈ  (symmetric)

A class model consists of one or more Expert objects, each of which fuses one
or more additive EllipsoidExperts via softmin.  The attribution chain is:

    Class SDF  Φ_k  = softmin_α{ Φ_{k,1}, …, Φ_{k,E} }   over Experts
    Expert SDF Φ    = softmin_α{ φ_1, …, φ_M }            over EllipsoidExperts

Soft attribution weights at each level:

    w_class[i]  = softmax( −α · Φ_{k,i} )          Expert attribution
    w_expert[j] = softmax( −α · φ_j )               EllipsoidExpert attribution

Cross-entropy loss for K-class prediction (argmin over softmin probabilities):

    p_k  = softmax( −α · [Φ_1, …, Φ_K] )_k
    L    = −log p_{y*}

Loss gradient (chain rule through the two softmin levels):

    ∂L/∂Φ_k       = α · ( p_k − 1[k = y*] )
    ∂L/∂Φ_{k,i}   = ∂L/∂Φ_k · w_class[i]
    ∂L/∂φ_{k,i,j} = ∂L/∂Φ_{k,i} · w_expert[j]
    ∂L/∂μ_{k,i,j} = ∂L/∂φ_{k,i,j} · ( −Pδ / q )
    ∂L/∂P_{k,i,j} = ∂L/∂φ_{k,i,j} · ( δδᵀ / 2q )

After accumulating over a mini-batch the precision matrix P is updated and
then projected back to the positive-definite cone via eigendecomposition:

    P_new   = P − lr · ΔP
    P_pd    = U diag(max(λ, λ_min)) Uᵀ
    r_new   = 1 / √(eigenvalues)
    R_new   = U   (orthonormal eigenvectors)

Temporal samplers may provide labeled training pairs to this optimizer, but no
latent-variable posterior or expectation-maximization step is involved.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _precision(e: Any) -> np.ndarray:
    """Reconstruct P = R diag(r⁻²) Rᵀ from stored (radii, orientation).

    Derivation: ``local = (x − c) @ R``, so the SDF squared is
    ``Σ_i local_i² / r_i² = δᵀ (R diag(r⁻²) Rᵀ) δ`` giving
    ``P = R diag(r⁻²) Rᵀ``.
    """
    inv_r2 = 1.0 / (e.radii ** 2)   # (d,)
    R = e.orientation                 # (d, d)
    return (R * inv_r2) @ R.T        # (d, d)  positive definite


def _sdf_and_grads(
    e: Any, delta: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """SDF value and analytic gradients for one EllipsoidExpert.

    Parameters
    ----------
    e     : EllipsoidExpert
    delta : ``x − e.center``  shape (d,)

    Returns
    -------
    phi   : float  — SDF value φ(x) = q − 1
    g_c   : (d,)   — ∂φ/∂μ  = −Pδ / q
    g_P   : (d,d)  — ∂φ/∂P  = δδᵀ / (2q)
    """
    d = len(delta)
    P = _precision(e)
    Pdelta = P @ delta              # (d,)
    q2 = float(delta @ Pdelta)     # δᵀ P δ ≥ 0
    if q2 < 1e-20:
        # Point is at (or numerically at) the ellipsoid centre; gradient
        # is undefined — return the SDF limit φ = -1 with zero gradients.
        return -1.0, np.zeros(d), np.zeros((d, d))
    q   = np.sqrt(q2)
    phi = q - 1.0
    g_c = -Pdelta / q
    g_P = np.outer(delta, delta) / (2.0 * q)
    return phi, g_c, g_P


def _stable_softmin(
    values: np.ndarray, alpha: float
) -> tuple[float, np.ndarray]:
    """Numerically stable softmin value and per-element attribution weights.

    softmin(v, α) = −(1/α) log( (1/M) Σ exp(−α v_i) )
    weights_i     = exp(−α v_i) / Σ exp(−α v_j)     (softmax of −α v)

    The 1/M mixture normalization matches SoftminFusion so that fused SDF
    values are scale-consistent regardless of how many components are present.
    The attribution weights are unaffected by the 1/M factor.

    Returns
    -------
    softmin_val : float  — the fused value (useful for nested softmin)
    weights     : (N,)   — attribution probabilities summing to 1
    """
    v = np.asarray(values, dtype=np.float64)
    M = len(v)
    if alpha == 0.0:
        w = np.ones(M) / M
        return float(v.mean()), w
    neg_av         = -alpha * v
    shifted        = neg_av - neg_av.max()        # for numerical stability
    exp_v          = np.exp(shifted)
    s              = exp_v.sum()
    lse            = np.log(s) + neg_av.max()     # log Σ exp(−α v_i)
    softmin_val    = -(lse - np.log(M)) / alpha   # subtract log(M) for 1/M factor
    return float(softmin_val), exp_v / s


def _stable_softmin_rows(
    values: np.ndarray, alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized softmin values and attribution weights for each row."""
    values = np.asarray(values, dtype=np.float64)
    component_count = values.shape[1]
    if alpha == 0.0:
        weights = np.full_like(values, 1.0 / component_count)
        return values.mean(axis=1), weights
    negative = -alpha * values
    maximum = negative.max(axis=1, keepdims=True)
    exponentials = np.exp(negative - maximum)
    sums = exponentials.sum(axis=1, keepdims=True)
    log_mean_exp = np.log(sums[:, 0]) + maximum[:, 0] - np.log(component_count)
    return -log_mean_exp / alpha, exponentials / sums


# ---------------------------------------------------------------------------
# SDFOptimizer
# ---------------------------------------------------------------------------

@dataclass
class SDFOptimizer:
    """Gradient-based refinement of a ``{class_id: list[Expert]}`` model.

    After a temporal sampler produces labeled ``(context, target)`` pairs, one
    or more calls to :meth:`step` refine the ellipsoid parameters to reduce
    classification cross-entropy.

    Parameters
    ----------
    models :
        ``{class_id: list[Expert]}`` — the fitted model to refine in-place.
    alpha :
        Softmin concentration; must match the value used during inference.
    learning_rate :
        Gradient descent step size (applied per-sample, i.e. lr / N per batch).
    min_eigenvalue :
        Eigenvalue floor applied during the PD projection of the precision
        matrix.  ``1e-6`` corresponds to a max semi-axis radius of 1000.
    score_scales :
        Optional per-class normalisation factors (from ``compute_score_scales``).
        When provided the scaled SDFs are used, matching inference behaviour.
    momentum :
        Exponential decay for first-moment velocity (0 = vanilla SGD,
        0.9 = standard heavy-ball momentum).

    Example
    -------
    >>> opt = SDFOptimizer(models=models, alpha=2.0, learning_rate=0.005)
    >>> for epoch in range(50):
    ...     loss = opt.step(X_train, y_train)
    ...     print(f"epoch {epoch}  loss={loss:.4f}")
    """

    models: dict
    alpha: float = 1.0
    learning_rate: float = 0.01
    min_eigenvalue: float = 1e-6
    score_scales: dict | None = None
    momentum: float = 0.9

    # Velocity buffers keyed by id(EllipsoidExpert) — initialised lazily.
    _vel_c: dict = field(default_factory=dict, init=False, repr=False)
    _vel_P: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        has_subtractive = any(
            ellipsoid.polarity < 0
            for experts in self.models.values()
            for expert in experts
            for ellipsoid in expert.ellipsoids
        )
        if has_subtractive:
            raise ValueError(
                "SDFOptimizer supports additive ellipsoids only; subtractive CSG "
                "gradients are not implemented."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, X: np.ndarray, y: np.ndarray) -> float:
        """Perform one mini-batch gradient descent step.

        Parameters
        ----------
        X : (N, d) float64 — feature vectors (already in the model's input space).
        y : (N,)   int32  — integer class labels.

        Returns
        -------
        float — mean cross-entropy loss over the batch (useful for monitoring).
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)
        class_ids = sorted(self.models.keys())
        class_index = {class_id: index for index, class_id in enumerate(class_ids)}
        class_count = len(class_ids)
        sample_count = len(X)
        if X.ndim != 2:
            raise ValueError(f"Expected X with shape (N, d), got {X.shape}.")
        if y.shape != (sample_count,):
            raise ValueError(
                f"Expected y with shape ({sample_count},), got {y.shape}."
            )
        if sample_count == 0:
            return 0.0

        class_sdfs = np.full((sample_count, class_count), 10.0)
        attribution: list[list] = [[] for _ in class_ids]
        for class_position, class_id in enumerate(class_ids):
            experts = self.models[class_id]
            if not experts:
                continue
            expert_sdfs = np.full((sample_count, len(experts)), 10.0)
            expert_details = []
            for expert_position, expert in enumerate(experts):
                ellipsoids = [item for item in expert.ellipsoids if item.polarity > 0]
                if not ellipsoids:
                    expert_details.append([])
                    continue
                ellipsoid_sdfs = np.empty((sample_count, len(ellipsoids)))
                ellipsoid_details = []
                for ellipsoid_position, ellipsoid in enumerate(ellipsoids):
                    delta = X - ellipsoid.center
                    precision = _precision(ellipsoid)
                    precision_delta = delta @ precision
                    squared_distance = np.einsum("ni,ni->n", delta, precision_delta)
                    distance = np.sqrt(np.maximum(squared_distance, 0.0))
                    ellipsoid_sdfs[:, ellipsoid_position] = distance - 1.0
                    ellipsoid_details.append(
                        (ellipsoid, delta, precision_delta, distance)
                    )
                expert_sdfs[:, expert_position], ellipsoid_weights = (
                    _stable_softmin_rows(ellipsoid_sdfs, self.alpha)
                )
                expert_details.append(list(zip(
                    ellipsoid_weights.T, ellipsoid_details,
                )))

            fused_sdf, expert_weights = _stable_softmin_rows(
                expert_sdfs, self.alpha,
            )
            scale = self.score_scales[class_id] if self.score_scales else 1.0
            class_sdfs[:, class_position] = fused_sdf / scale
            attribution[class_position] = list(zip(
                expert_weights.T, expert_details,
            ))

        _, probabilities = _stable_softmin_rows(class_sdfs, self.alpha)
        target_indices = np.array([
            class_index.get(int(label), -1) for label in y
        ])
        valid_targets = target_indices >= 0
        valid_rows = np.flatnonzero(valid_targets)
        total_loss = -np.log(np.clip(
            probabilities[valid_rows, target_indices[valid_targets]], 1e-12, 1.0,
        )).sum()

        one_hot = np.zeros((sample_count, class_count))
        one_hot[valid_rows, target_indices[valid_targets]] = 1.0
        loss_by_class = self.alpha * (one_hot - probabilities)
        center_gradients: dict[int, np.ndarray] = {}
        precision_gradients: dict[int, np.ndarray] = {}
        for class_position, class_id in enumerate(class_ids):
            scale = self.score_scales[class_id] if self.score_scales else 1.0
            class_gradient = loss_by_class[:, class_position] / scale
            for expert_weights, ellipsoid_attributes in attribution[class_position]:
                expert_gradient = class_gradient * expert_weights
                for ellipsoid_weights, details in ellipsoid_attributes:
                    ellipsoid, delta, precision_delta, distance = details
                    inverse_distance = np.zeros_like(distance)
                    active = distance >= 1e-10
                    inverse_distance[active] = 1.0 / distance[active]
                    coefficient = expert_gradient * ellipsoid_weights
                    key = id(ellipsoid)
                    center_gradients[key] = -(
                        coefficient * inverse_distance
                    ) @ precision_delta
                    precision_gradients[key] = np.einsum(
                        "n,ni,nj->ij",
                        coefficient * inverse_distance * 0.5,
                        delta,
                        delta,
                    )

        self._apply_updates(
            center_gradients, precision_gradients, sample_count,
        )
        return total_loss / sample_count

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Evaluate loss and classification error without updating parameters."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)
        if X.ndim != 2 or y.shape != (len(X),):
            raise ValueError("Expected X=(N,d) and y=(N,).")
        if len(X) == 0:
            return {"loss": 0.0, "error": 0.0}

        class_ids = np.array(sorted(self.models), dtype=np.int32)
        class_sdfs = np.full((len(X), len(class_ids)), 10.0, dtype=np.float64)
        for class_position, class_id in enumerate(class_ids):
            experts = self.models[int(class_id)]
            if not experts:
                continue
            expert_sdfs = np.column_stack([
                expert.compute_sdf(X) for expert in experts
            ])
            fused_sdf, _ = _stable_softmin_rows(expert_sdfs, self.alpha)
            scale = self.score_scales[int(class_id)] if self.score_scales else 1.0
            class_sdfs[:, class_position] = fused_sdf / scale

        _, probabilities = _stable_softmin_rows(class_sdfs, self.alpha)
        lookup = {int(class_id): index for index, class_id in enumerate(class_ids)}
        target_columns = np.array([lookup.get(int(label), -1) for label in y])
        valid = target_columns >= 0
        selected = probabilities[np.flatnonzero(valid), target_columns[valid]]
        loss = -np.log(np.clip(selected, 1e-12, 1.0)).mean() if np.any(valid) else np.inf
        predictions = class_ids[probabilities.argmax(axis=1)]
        return {
            "loss": float(loss),
            "error": float(np.mean(predictions != y)),
        }

    def reset_momentum(self) -> None:
        """Clear velocity buffers.  Call between unrelated training phases."""
        self._vel_c.clear()
        self._vel_P.clear()

    def export_state(self) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """Export geometry and momentum using stable topology-based keys."""
        arrays: dict[str, np.ndarray] = {}
        ellipsoids = []
        for position, (class_id, expert_index, ellipsoid_index, ellipsoid) in enumerate(
            self._ordered_ellipsoids()
        ):
            prefix = f"ellipsoid-{position:06d}"
            key = id(ellipsoid)
            has_center_velocity = key in self._vel_c
            has_precision_velocity = key in self._vel_P
            arrays[f"{prefix}-center"] = np.asarray(ellipsoid.center).copy()
            arrays[f"{prefix}-radii"] = np.asarray(ellipsoid.radii).copy()
            arrays[f"{prefix}-orientation"] = np.asarray(ellipsoid.orientation).copy()
            if has_center_velocity:
                arrays[f"{prefix}-center-velocity"] = self._vel_c[key].copy()
            if has_precision_velocity:
                arrays[f"{prefix}-precision-velocity"] = self._vel_P[key].copy()
            ellipsoids.append({
                "class_id": class_id,
                "expert_index": expert_index,
                "ellipsoid_index": ellipsoid_index,
                "polarity": int(ellipsoid.polarity),
                "has_center_velocity": has_center_velocity,
                "has_precision_velocity": has_precision_velocity,
            })
        state = {
            "schema_version": 1,
            "alpha": self.alpha,
            "learning_rate": self.learning_rate,
            "min_eigenvalue": self.min_eigenvalue,
            "momentum": self.momentum,
            "class_ids": [int(class_id) for class_id in sorted(self.models)],
            "score_scales": (
                None
                if self.score_scales is None
                else [[int(key), float(value)] for key, value in sorted(self.score_scales.items())]
            ),
            "ellipsoids": ellipsoids,
        }
        return state, arrays

    def import_state(
        self,
        state: dict[str, Any],
        arrays: dict[str, np.ndarray],
    ) -> None:
        """Restore a state exported by :meth:`export_state` into this topology."""
        required = {
            "schema_version", "alpha", "learning_rate", "min_eigenvalue",
            "momentum", "class_ids", "score_scales", "ellipsoids",
        }
        if set(state) != required or state["schema_version"] != 1:
            raise ValueError("unsupported optimizer checkpoint state")
        topology = self._ordered_ellipsoids()
        expected_class_ids = [int(class_id) for class_id in sorted(self.models)]
        if state["class_ids"] != expected_class_ids or len(state["ellipsoids"]) != len(topology):
            raise ValueError("optimizer checkpoint topology does not match models")

        self.alpha = float(state["alpha"])
        self.learning_rate = float(state["learning_rate"])
        self.min_eigenvalue = float(state["min_eigenvalue"])
        self.momentum = float(state["momentum"])
        self.score_scales = (
            None
            if state["score_scales"] is None
            else {int(key): float(value) for key, value in state["score_scales"]}
        )
        self._vel_c.clear()
        self._vel_P.clear()
        consumed: set[str] = set()
        for position, (actual, topology_item) in enumerate(zip(state["ellipsoids"], topology)):
            class_id, expert_index, ellipsoid_index, ellipsoid = topology_item
            identity = {
                "class_id": class_id,
                "expert_index": expert_index,
                "ellipsoid_index": ellipsoid_index,
                "polarity": int(ellipsoid.polarity),
            }
            if any(actual.get(name) != value for name, value in identity.items()):
                raise ValueError("optimizer checkpoint topology does not match models")
            prefix = f"ellipsoid-{position:06d}"
            center_name = f"{prefix}-center"
            radii_name = f"{prefix}-radii"
            orientation_name = f"{prefix}-orientation"
            for name in (center_name, radii_name, orientation_name):
                if name not in arrays:
                    raise ValueError(f"optimizer checkpoint missing array {name}")
                consumed.add(name)
            center = np.asarray(arrays[center_name], dtype=np.float64)
            radii = np.asarray(arrays[radii_name], dtype=np.float64)
            orientation = np.asarray(arrays[orientation_name], dtype=np.float64)
            if (
                center.shape != ellipsoid.center.shape
                or radii.shape != ellipsoid.radii.shape
                or orientation.shape != ellipsoid.orientation.shape
            ):
                raise ValueError("optimizer checkpoint array shape does not match models")
            ellipsoid.center = center.copy()
            ellipsoid.radii = radii.copy()
            ellipsoid.orientation = orientation.copy()
            key = id(ellipsoid)
            if actual.get("has_center_velocity"):
                velocity_name = f"{prefix}-center-velocity"
                self._vel_c[key] = np.asarray(arrays[velocity_name], dtype=np.float64).copy()
                consumed.add(velocity_name)
            if actual.get("has_precision_velocity"):
                velocity_name = f"{prefix}-precision-velocity"
                self._vel_P[key] = np.asarray(arrays[velocity_name], dtype=np.float64).copy()
                consumed.add(velocity_name)
            self.models[class_id][expert_index]._bs_cache = None
        if consumed != set(arrays):
            raise ValueError("optimizer checkpoint contains unexpected arrays")

    def _ordered_ellipsoids(self) -> list[tuple[int, int, int, Any]]:
        entries = []
        for class_id in sorted(self.models):
            if not isinstance(class_id, (int, np.integer)):
                raise ValueError("optimizer checkpoint class IDs must be integers")
            for expert_index, expert in enumerate(self.models[class_id]):
                for ellipsoid_index, ellipsoid in enumerate(expert.ellipsoids):
                    entries.append((int(class_id), expert_index, ellipsoid_index, ellipsoid))
        return entries

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _apply_updates(
        self,
        g_c: dict[int, np.ndarray],
        g_P: dict[int, np.ndarray],
        N: int,
    ) -> None:
        """Apply accumulated gradients to every touched EllipsoidExpert."""
        lr   = self.learning_rate / N   # per-sample learning rate
        beta = self.momentum

        for cid, experts in self.models.items():
            for expert in experts:
                additive = [e for e in expert.ellipsoids if e.polarity > 0]
                for e in additive:
                    key = id(e)
                    updated = False

                    # ── center ───────────────────────────────────────────
                    if key in g_c:
                        gc   = g_c[key]
                        v    = self._vel_c.get(key, np.zeros_like(gc))
                        v    = beta * v + (1.0 - beta) * gc
                        self._vel_c[key] = v
                        e.center -= lr * v
                        updated = True

                    # ── precision matrix → (radii, orientation) ──────────
                    if key in g_P:
                        gP   = g_P[key]
                        v    = self._vel_P.get(key, np.zeros_like(gP))
                        v    = beta * v + (1.0 - beta) * gP
                        self._vel_P[key] = v

                        P_new = _precision(e) - lr * v
                        # Enforce symmetry (numerical drift)
                        P_new = (P_new + P_new.T) * 0.5
                        # Project to PD cone
                        vals, vecs = np.linalg.eigh(P_new)
                        vals = np.maximum(vals, self.min_eigenvalue)
                        # Recover stored parameters from eigendecomposition
                        e.radii       = 1.0 / np.sqrt(vals)
                        e.orientation = vecs   # columns = orthonormal eigenvectors
                        updated = True

                    if updated:
                        expert._bs_cache = None   # invalidate bounding sphere
