"""Causal context construction for supervised sequential prediction.

TemporalSampler constructs labelled (context, target) training pairs from a
sequence of feature vectors, enabling GEODE to be trained on sequential data.

Temporal supervised-refinement loop
-----------------------------------
The standard GEODE training pipeline fits geometric experts on a *static*
labelled dataset.  For sequential prediction (e.g. next-step, next-token) the
dataset must first be *created* by rolling out the current model on a sequence:

    Initialise: fit models on any initial labelled data
    Repeat:
        Context sampling (TemporalSampler):
            Roll current model forward on training sequence.
            Emit (context_t, target_{t+lag}) pairs.
        Supervised update (GreedyConstructor or SDFOptimizer):
            Refit / refine experts on those pairs.
    Until convergence

Labels remain observed throughout; this is not expectation-maximization.

Context modes
-------------
``"concat"``
    Concatenate the last *window* raw feature vectors.
    Output dim = ``window × d``.

``"sdf"``
    Use the current model's SDF score vector as the running state.  Output
    dim = ``K`` (number of classes).  This is compact and adaptive — as the
    model improves the representation becomes more informative.

``"concat+sdf"``
    Concatenate both.  Output dim = ``window × d + K``.

Example
-------
>>> sampler = TemporalSampler(lag=1, window=3, context_mode="sdf",
...                           models=models, alpha=2.0)
>>> X_ctx, y_ctx = sampler.fit_transform(X_sequence, y_sequence)
>>> # X_ctx: (T-1, K)   y_ctx: (T-1,)
>>> optimizer = SDFOptimizer(models=models, alpha=2.0)
>>> loss = optimizer.step(X_ctx, y_ctx)
"""

from __future__ import annotations

import numpy as np
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_sdf_state(
    x: np.ndarray,
    models: dict,
    class_ids: list,
    alpha: float,
    score_scales: dict | None,
) -> np.ndarray:
    """Return the K-dimensional SDF score vector for a single sample *x* (d,).

    The i-th element is the fused SDF of class_ids[i]'s experts at x,
    optionally normalised by score_scales.  Inf values are replaced by 10.0
    to keep the state vector finite.
    """
    from src.inference_engine import InferenceEngine  # local import to avoid cycles

    K = len(class_ids)
    state = np.empty(K, dtype=np.float64)
    x1 = x[np.newaxis]  # (1, d)
    for ki, cid in enumerate(class_ids):
        experts = models.get(cid, [])
        if not experts:
            state[ki] = 10.0
            continue
        sdf = InferenceEngine(experts, alpha=alpha).get_fused_sdf(x1)[0]
        scale = score_scales[cid] if score_scales else 1.0
        state[ki] = sdf / scale if not np.isinf(sdf) else 10.0
    return state


class TemporalStateEncoder:
    """Deterministic fixed-width recurrent encoder for arbitrary sequential features.

    The state dimension is independent of observation width and class count, so a
    separate GEODE readout can be fitted without changing input dimensionality when
    labels or upstream feature extractors change.
    """

    def __init__(
        self,
        state_dim: int,
        recurrence: float = 0.8,
        seed: int = 42,
    ) -> None:
        if state_dim < 1:
            raise ValueError("state_dim must be positive.")
        if not 0.0 <= recurrence < 1.0:
            raise ValueError("recurrence must be in [0, 1).")
        self.state_dim = state_dim
        self.recurrence = recurrence
        self.seed = seed
        self.input_dim: int | None = None
        self._input_projection: np.ndarray | None = None
        self._recurrent_projection: np.ndarray | None = None

    def fit(self, input_dim: int) -> "TemporalStateEncoder":
        """Initialize deterministic input and orthogonal recurrent projections."""
        if input_dim < 1:
            raise ValueError("input_dim must be positive.")
        rng = np.random.default_rng(self.seed)
        self.input_dim = input_dim
        self._input_projection = rng.normal(
            scale=1.0 / np.sqrt(input_dim), size=(input_dim, self.state_dim),
        )
        recurrent = rng.normal(size=(self.state_dim, self.state_dim))
        self._recurrent_projection = np.linalg.qr(recurrent)[0]
        return self

    def transform(
        self,
        observations: np.ndarray,
        feedback: np.ndarray | None = None,
    ) -> np.ndarray:
        """Encode a sequence causally, optionally including aligned feedback features."""
        observations = np.asarray(observations, dtype=np.float64)
        if observations.ndim != 2:
            raise ValueError("observations must have shape (time, features).")
        inputs = observations
        if feedback is not None:
            feedback = np.asarray(feedback, dtype=np.float64)
            if feedback.ndim != 2 or len(feedback) != len(observations):
                raise ValueError("feedback must have shape (time, feedback_features).")
            inputs = np.concatenate([observations, feedback], axis=1)

        if self._input_projection is None:
            self.fit(inputs.shape[1])
        if inputs.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} input features, got {inputs.shape[1]}."
            )

        states = np.empty((len(inputs), self.state_dim), dtype=np.float64)
        state = np.zeros(self.state_dim, dtype=np.float64)
        for time_index, features in enumerate(inputs):
            driven = features @ self._input_projection
            recurrent = state @ self._recurrent_projection
            state = np.tanh(driven + self.recurrence * recurrent)
            states[time_index] = state
        return states


def _split_state_width(total_width: int, member_count: int) -> list[int]:
    if total_width < member_count:
        raise ValueError("state_dim must be at least the number of ensemble members.")
    base, remainder = divmod(total_width, member_count)
    return [base + (index < remainder) for index in range(member_count)]


class MultiTimescaleStateEncoder:
    """Concatenate reservoirs with different recurrence timescales at fixed width."""

    def __init__(
        self,
        state_dim: int,
        recurrences: tuple[float, ...] = (0.3, 0.7, 0.95),
        seed: int = 42,
    ) -> None:
        if not recurrences:
            raise ValueError("At least one recurrence is required.")
        widths = _split_state_width(state_dim, len(recurrences))
        self.state_dim = state_dim
        self.encoders = [
            TemporalStateEncoder(width, recurrence=recurrence, seed=seed + index)
            for index, (width, recurrence) in enumerate(zip(widths, recurrences))
        ]

    def transform(self, observations: np.ndarray) -> np.ndarray:
        return np.concatenate([
            encoder.transform(observations) for encoder in self.encoders
        ], axis=1)


class MultiSeedStateEncoder:
    """Concatenate independent reservoirs with one recurrence at fixed width."""

    def __init__(
        self,
        state_dim: int,
        member_count: int = 3,
        recurrence: float = 0.8,
        seed: int = 42,
    ) -> None:
        if member_count < 1:
            raise ValueError("member_count must be positive.")
        widths = _split_state_width(state_dim, member_count)
        self.state_dim = state_dim
        self.encoders = [
            TemporalStateEncoder(width, recurrence=recurrence, seed=seed + 997 * index)
            for index, width in enumerate(widths)
        ]

    def transform(self, observations: np.ndarray) -> np.ndarray:
        return np.concatenate([
            encoder.transform(observations) for encoder in self.encoders
        ], axis=1)


def temporal_state_pairs(
    observations: np.ndarray,
    targets: np.ndarray,
    encoder: TemporalStateEncoder,
    lag: int = 1,
    feedback: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode observations and align each state with a future supervised target."""
    if lag < 1:
        raise ValueError("lag must be positive.")
    targets = np.asarray(targets)
    if len(observations) != len(targets):
        raise ValueError("observations and targets must have equal length.")
    if len(targets) <= lag:
        raise ValueError("sequence must be longer than lag.")
    states = encoder.transform(observations, feedback=feedback)
    return states[:-lag], targets[lag:]


# ---------------------------------------------------------------------------
# TemporalSampler
# ---------------------------------------------------------------------------

class TemporalSampler:
    """Construct (context, target) training pairs from sequential data.

    Parameters
    ----------
    lag :
        Number of steps to look ahead for the target.  ``lag=1`` means
        "predict the next step", ``lag=n`` means "predict n steps ahead".
    window :
        For ``context_mode="concat"`` and ``"concat+sdf"``, the number of
        past raw feature vectors to concatenate into the context.
        Ignored for ``context_mode="sdf"``.
    context_mode :
        One of ``"concat"`` | ``"sdf"`` | ``"concat+sdf"``.
    models :
        ``{class_id: list[Expert]}`` model used for the ``"sdf"`` state.
        Required when *context_mode* contains ``"sdf"``.
    alpha :
        Softmin concentration (must match the model's inference value).
    score_scales :
        Optional per-class normalisation factors.  Pass the same dict used
        by ``predict_labels`` / ``compute_raw_scores`` so the SDF state
        matches what the model sees at inference.
    """

    def __init__(
        self,
        lag: int = 1,
        window: int = 1,
        context_mode: str = "concat",
        models: dict | None = None,
        alpha: float = 1.0,
        score_scales: dict | None = None,
    ) -> None:
        if context_mode not in {"concat", "sdf", "concat+sdf"}:
            raise ValueError(
                f"context_mode must be 'concat', 'sdf', or 'concat+sdf', got {context_mode!r}"
            )
        if context_mode in {"sdf", "concat+sdf"} and models is None:
            raise ValueError("models must be provided for context_mode='sdf' or 'concat+sdf'")

        self.lag          = lag
        self.window       = window
        self.context_mode = context_mode
        self.models       = models
        self.alpha        = alpha
        self.score_scales = score_scales

    # ------------------------------------------------------------------

    def fit_transform(
        self,
        X_seq: np.ndarray,
        y_seq: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build (context, target) pairs from a sequence.

        Parameters
        ----------
        X_seq : (T, d) array — feature vectors in temporal order.
        y_seq : (T,)  array — integer class labels (same length as X_seq).

        Returns
        -------
        X_ctx : (M, context_dim) — context feature matrix.
        y_ctx : (M,)             — target labels at position t + lag.

        ``M = T − max(window − 1, 0) − lag``
        """
        X_seq = np.asarray(X_seq, dtype=np.float64)
        y_seq = np.asarray(y_seq, dtype=np.int32)
        T, d  = X_seq.shape

        class_ids = sorted(self.models.keys()) if self.models else []

        # Pre-compute SDF states for every time-step (expensive but done once)
        if self.context_mode in {"sdf", "concat+sdf"}:
            sdf_states = np.stack([
                _compute_sdf_state(X_seq[t], self.models, class_ids,
                                   self.alpha, self.score_scales)
                for t in range(T)
            ])   # (T, K)
        else:
            sdf_states = None

        # Window start: we need at least `window` previous steps for concat mode
        start = max(self.window - 1, 0)   # first valid context index
        end   = T - self.lag              # last valid context index (exclusive)

        if end <= start:
            raise ValueError(
                f"Sequence too short (T={T}) for window={self.window}, lag={self.lag}. "
                f"Need T > {start + self.lag}."
            )

        contexts = []
        targets  = []

        for t in range(start, end):
            parts = []

            # ── raw-feature concatenation window ─────────────────────────
            if self.context_mode in {"concat", "concat+sdf"}:
                t0 = max(0, t - self.window + 1)
                window_vecs = X_seq[t0 : t + 1]     # (≤window, d)
                # Zero-pad on the left if the window extends before index 0
                if len(window_vecs) < self.window:
                    pad = np.zeros((self.window - len(window_vecs), d), dtype=np.float64)
                    window_vecs = np.concatenate([pad, window_vecs], axis=0)
                parts.append(window_vecs.ravel())    # (window * d,)

            # ── SDF state ─────────────────────────────────────────────────
            if self.context_mode in {"sdf", "concat+sdf"}:
                parts.append(sdf_states[t])          # (K,)

            contexts.append(np.concatenate(parts))
            targets.append(int(y_seq[t + self.lag]))

        X_ctx = np.stack(contexts)                    # (M, context_dim)
        y_ctx = np.array(targets, dtype=np.int32)     # (M,)
        return X_ctx, y_ctx

    # ------------------------------------------------------------------

    @property
    def context_dim(self) -> int:
        """Return the output context dimensionality, or -1 if unknown yet."""
        d = 0
        # Infer from first call if models are set
        if self.context_mode in {"concat", "concat+sdf"}:
            d += -1   # unknown until X_seq is seen
        if self.context_mode in {"sdf", "concat+sdf"} and self.models:
            d += len(self.models)
        return d

    def __repr__(self) -> str:
        return (
            f"TemporalSampler(mode={self.context_mode!r}, "
            f"lag={self.lag}, window={self.window})"
        )
