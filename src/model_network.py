"""Model network: FittedModel container and DAG execution engine.

FittedModel
-----------
A trained GEODE model bundled with its fingerprint and all inference state
(class experts, score scales, Platt calibrator, and the PCA+LDA+Scaler
transform pipeline).  It exposes two public inference methods:

  ``sdf_scores(X)``  → (N, n_classes) normalised SDF matrix
  ``predict(X)``     → (N,) integer class labels

For *source* nodes (raw feature input) X passes through the full
PCA → LDA → StandardScaler transform before reaching the SDF experts.
For *downstream* nodes (SDF-score input) the transform is skipped — the
SDF values from upstream are already geometrically meaningful features.

ModelNetwork
------------
A directed acyclic graph (DAG) of ModelNode entries.  Execution follows
topological order:

  1. Source nodes receive ``X_raw`` (raw features) and produce SDF scores.
  2. Downstream nodes receive the *column-concatenated* SDF score matrices
     from their upstream nodes as their input feature space.

The limited, interpretable SDF range (< 0 = inside a class volume,
≈ 0 = on the surface, > 0 = outside) makes these scores semantically
useful as input features for a downstream geometric classifier.

Example — bird-on-car detection
::

    bird_model = build_fitted_model(X_raw, y_bird, task_name="bird_detector")
    car_model  = build_fitted_model(X_raw, y_car,  task_name="car_detector")

    # compound model trained on [bird_sdf | car_sdf] feature space
    X_scores = np.hstack([bird_model.sdf_scores(X_raw),
                          car_model.sdf_scores(X_raw)])
    bird_on_car = build_fitted_model(
        X_scores, y_compound,
        task_name="bird_on_car",
        input_source="sdf_scores",
        upstream_tasks=("bird_detector", "car_detector"),
    )

    net = ModelNetwork()
    net.add_node("bird",        bird_model)
    net.add_node("car",         car_model)
    net.add_node("bird_on_car", bird_on_car, upstream=["bird", "car"])
    results = net.run(X_test_raw)   # {"bird": ..., "car": ..., "bird_on_car": ...}
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any

from src.model_fingerprint import ModelFingerprint
from src.inference_engine import InferenceEngine
from src.open_set import (
    OpenSetBatchResult,
    RoutingStageCounters,
    SupportProfile,
    apply_rejection_policy,
)


# ---------------------------------------------------------------------------
# FittedModel
# ---------------------------------------------------------------------------


@dataclass
class FittedModel:
    """A trained GEODE classifier bundled with its fingerprint.

    Parameters
    ----------
    fingerprint:
        Encodes task identity and the IO contract (input source, output classes,
        output type).  Two models with identical fingerprints are interchangeable.
    class_models:
        ``{class_id: [Expert, ...]}`` — the SDF expert groups, one entry per
        class.
    score_scales:
        ``{class_id: float}`` — per-class mean(|SDF|) normalisation factor
        computed on training data.
    calibrator:
        Optional sklearn LogisticRegression fitted on the SDF score matrix
        (Platt calibration).  When present, ``predict`` uses calibrator output
        instead of raw argmin.
    pca, lda, scaler:
        Fitted sklearn transformers for source models.  All three are ``None``
        for downstream models that receive SDF scores directly.
    """

    fingerprint: ModelFingerprint
    class_models: dict  # {int: list[Expert]}
    score_scales: dict  # {int: float}
    calibrator: Any = None
    pca: Any = None
    lda: Any = None
    scaler: Any = None
    use_gpu: bool = False
    class_fusion_modes: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        invalid_modes = {
            class_id: mode
            for class_id, mode in self.class_fusion_modes.items()
            if class_id not in self.class_models
            or mode not in {"normalized_softmin", "hard_min"}
        }
        if invalid_modes:
            raise ValueError(f"Invalid class fusion modes: {invalid_modes}.")
        # Runtime-only cache — not part of the serialisable model state.
        # Lazily populated on the first GPU inference call.
        self._gpu_engine: Any = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @property
    def alpha(self) -> float:
        return self.fingerprint.alpha

    @property
    def class_ids(self) -> list[int]:
        return sorted(self.class_models.keys())

    def _to_feature_space(self, X: np.ndarray) -> np.ndarray:
        """Apply the stored transform chain to new data.

        Source models (``pca`` is not None): PCA → LDA → StandardScaler.
        Downstream models: identity — X is already the feature space.
        """
        if self.pca is not None:
            X = self.pca.transform(X)
        if self.lda is not None:
            X = self.lda.transform(X)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return X

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------

    def sdf_scores(self, X: np.ndarray) -> np.ndarray:
        """Return the (N, n_classes) normalised SDF score matrix.

        Negative values indicate the sample is *inside* a class volume
        (high confidence).  The matrix is suitable as input features for
        a downstream node in the network.
        """
        X_feat = self._to_feature_space(np.asarray(X, dtype=np.float64))

        if self.use_gpu and not any(
            mode == "hard_min" for mode in self.class_fusion_modes.values()
        ):
            return self._sdf_scores_gpu(X_feat)

        scores = np.full((len(X_feat), len(self.class_ids)), 10.0, dtype=np.float64)
        for i, cid in enumerate(self.class_ids):
            experts = self.class_models[cid]
            if not experts:
                continue
            fusion_mode = self.class_fusion_modes.get(cid, "normalized_softmin")
            if fusion_mode == "hard_min":
                sdf = np.min(np.asarray([
                    expert.compute_sdf(X_feat) for expert in experts
                ]), axis=0)
            else:
                sdf = InferenceEngine(experts, alpha=self.alpha).get_fused_sdf(X_feat)
            scale = self.score_scales.get(cid, 1.0)
            scores[:, i] = sdf / scale
        return scores

    def _sdf_scores_gpu(self, X_feat: np.ndarray) -> np.ndarray:
        """GPU path for sdf_scores — delegates to a cached GPUInferenceEngine."""
        if self._gpu_engine is None:
            from src.gpu_engine import GPUInferenceEngine
            ordered = [self.class_models[cid] for cid in self.class_ids]
            self._gpu_engine = GPUInferenceEngine(ordered, alpha=self.alpha)
        raw = self._gpu_engine.class_sdfs(X_feat)  # (N, C) float64
        scales = np.array(
            [self.score_scales.get(cid, 1.0) for cid in self.class_ids],
            dtype=np.float64,
        )
        return raw / scales[np.newaxis, :]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for each sample in X.

        Uses Platt calibration when available, otherwise argmin(SDF/scale).
        """
        return self._predict_from_scores(self.sdf_scores(X))

    def predict_open_set(
        self,
        X: np.ndarray,
        support_profile: SupportProfile,
        *,
        feature_transform_fingerprint: str,
        calibrated_novelty_scores: np.ndarray | None = None,
    ) -> OpenSetBatchResult:
        """Predict with an opt-in frozen rejection policy.

        The existing closed-set ``predict`` path is unchanged. Threshold fitting
        and novelty calibration must happen before this method is called.
        """
        scores = self.sdf_scores(X)
        nonempty_classes = sum(bool(self.class_models[class_id]) for class_id in self.class_ids)
        primitive_count = sum(
            len(expert.ellipsoids)
            for experts in self.class_models.values()
            for expert in experts
        )
        return self._predict_open_set_from_scores(
            scores,
            support_profile,
            feature_transform_fingerprint=feature_transform_fingerprint,
            calibrated_novelty_scores=calibrated_novelty_scores,
            counters=RoutingStageCounters(
                sample_count=len(scores),
                nodes_executed=1,
                compatible_candidate_pairs=scores.size,
                shortlisted_candidate_pairs=scores.size,
                exact_class_sdf_pairs=len(scores) * nonempty_classes,
                primitive_sdf_pairs=len(scores) * primitive_count,
                score_values_materialized=scores.size,
            ),
        )

    def _predict_open_set_from_scores(
        self,
        scores: np.ndarray,
        support_profile: SupportProfile,
        *,
        feature_transform_fingerprint: str,
        calibrated_novelty_scores: np.ndarray | None = None,
        counters: RoutingStageCounters | None = None,
    ) -> OpenSetBatchResult:
        """Apply open-set rejection to an existing score matrix."""
        scores = np.asarray(scores, dtype=np.float64)
        labels = self._predict_from_scores(scores)
        probabilities = None
        if support_profile.novelty_score == "maximum_probability":
            if self.calibrator is None or not hasattr(self.calibrator, "predict_proba"):
                raise ValueError(
                    "maximum_probability open-set inference requires a calibrator."
                )
            native_probabilities = self.calibrator.predict_proba(scores)
            native_classes = tuple(self.calibrator.classes_.tolist())
            try:
                columns = [native_classes.index(class_id) for class_id in self.class_ids]
            except ValueError as error:
                raise ValueError(
                    "Calibrator classes do not align with fitted-model classes."
                ) from error
            probabilities = native_probabilities[:, columns]

        result = apply_rejection_policy(
            scores,
            labels,
            tuple(self.class_ids),
            support_profile,
            model_signature=self.fingerprint.signature,
            feature_transform_fingerprint=feature_transform_fingerprint,
            probabilities=probabilities,
            calibrated_novelty_scores=calibrated_novelty_scores,
        )
        return OpenSetBatchResult(
            predictions=result.predictions,
            counters=result.counters if counters is None else counters,
        )

    def _predict_from_scores(self, scores: np.ndarray) -> np.ndarray:
        """Derive labels from a pre-computed score matrix (N, n_classes).

        Separating score computation from label derivation lets callers that
        already hold a score matrix (e.g. :meth:`ModelNetwork.run`) avoid a
        redundant SDF evaluation.
        """
        if self.calibrator is not None:
            return self.calibrator.predict(scores).astype(np.int32)
        return np.array(
            [self.class_ids[i] for i in np.argmin(scores, axis=1)],
            dtype=np.int32,
        )

    # ------------------------------------------------------------------
    # Swappability
    # ------------------------------------------------------------------

    def is_swappable_with(self, other: "FittedModel") -> bool:
        """True when *other* serves the same role and can replace this model."""
        return self.fingerprint.is_swappable_with(other.fingerprint)

    def __repr__(self) -> str:
        return (
            f"FittedModel(task={self.fingerprint.task_name!r}, "
            f"classes={self.class_ids}, "
            f"sig={self.fingerprint.signature})"
        )


# ---------------------------------------------------------------------------
# ModelNode
# ---------------------------------------------------------------------------


@dataclass
class ModelNode:
    """A node in the ModelNetwork DAG.

    The *model* field accepts any object that implements the duck-type
    interface of :class:`FittedModel`: ``sdf_scores()``,
    ``_predict_from_scores()``, ``fingerprint``, and ``is_swappable_with()``.
    :class:`~src.primitive.Primitive` objects satisfy this interface.
    """

    name: str
    model: Any  # FittedModel | Primitive — duck-typed
    upstream: list[str] = field(default_factory=list)

    @property
    def fingerprint(self) -> ModelFingerprint:
        return self.model.fingerprint


# ---------------------------------------------------------------------------
# ModelNetwork
# ---------------------------------------------------------------------------


class ModelNetwork:
    """Directed acyclic graph of FittedModel nodes.

    Data flow:
    - Source nodes (no upstream) receive raw features and produce SDF scores.
    - Downstream nodes receive the column-concatenated SDF score matrices from
      their upstream nodes.  The limited SDF range is preserved across hops, so
      each downstream model still operates in a geometrically interpretable space.

    Nodes are executed in topological order, guaranteeing all upstream outputs
    are ready before a downstream node runs.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, ModelNode] = {}

    # ------------------------------------------------------------------
    # Building the graph
    # ------------------------------------------------------------------

    def add_node(
        self,
        name: str,
        model: Any,  # FittedModel | Primitive
        upstream: list[str] | None = None,
    ) -> None:
        """Register a model or primitive node.

        Parameters
        ----------
        name:
            Unique node identifier used for wiring and result lookup.
        model:
            A trained :class:`FittedModel` **or** a
            :class:`~src.primitive.Primitive`.  Any object that implements
            ``sdf_scores()``, ``_predict_from_scores()``, ``fingerprint``,
            and ``is_swappable_with()`` is accepted.
        upstream:
            Names of nodes whose SDF score matrices are concatenated and fed as
            input to this node.  Leave empty (or ``None``) for source nodes that
            receive raw features.
        """
        if upstream is None:
            upstream = []
        for up in upstream:
            if up not in self._nodes:
                raise ValueError(
                    f"Upstream node {up!r} not registered.  "
                    f"Register upstream nodes before their dependants."
                )
        self._nodes[name] = ModelNode(name=name, model=model, upstream=upstream)

    def swap_node(self, name: str, replacement: Any) -> None:  # FittedModel | Primitive
        """Replace the model at *name* with *replacement*.

        Raises :class:`ValueError` if the replacement fingerprint is not
        swappable with the existing node (different task, IO contract, or
        class set).
        """
        if name not in self._nodes:
            raise KeyError(f"Node {name!r} not in network.")
        existing = self._nodes[name].model
        if not existing.is_swappable_with(replacement):
            raise ValueError(
                f"Cannot swap node {name!r}: fingerprints are incompatible.\n"
                f"  existing:    {existing.fingerprint}\n"
                f"  replacement: {replacement.fingerprint}"
            )
        self._nodes[name].model = replacement

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _topological_order(self) -> list[str]:
        visited: set[str] = set()
        order: list[str] = []

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            for up in self._nodes[name].upstream:
                visit(up)
            order.append(name)

        for name in self._nodes:
            visit(name)
        return order

    def run(self, X_raw: np.ndarray) -> dict[str, np.ndarray]:
        """Execute the network in topological order.

        Parameters
        ----------
        X_raw:
            Raw feature array (N, d) fed to all source nodes.

        Returns
        -------
        dict mapping each node name to its predicted label array (N,).
        The SDF score matrices are available via ``sdf_scores(X_raw)`` on
        individual :class:`FittedModel` instances if needed.
        """
        X_raw = np.asarray(X_raw, dtype=np.float64)
        sdf_cache: dict[str, np.ndarray] = {}
        label_cache: dict[str, np.ndarray] = {}

        for name in self._topological_order():
            node = self._nodes[name]
            if not node.upstream:
                # Source node: receives raw features
                X_in = X_raw
            else:
                # Downstream node: concatenated SDF scores from upstream
                X_in = np.concatenate(
                    [sdf_cache[up] for up in node.upstream], axis=1
                )
            # Compute scores once; derive labels from the same matrix to avoid
            # a redundant GPU kernel dispatch when use_gpu=True.
            sdf_cache[name]   = node.model.sdf_scores(X_in)
            label_cache[name] = node.model._predict_from_scores(sdf_cache[name])

        return label_cache

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Check edge type compatibility.  Returns a list of issue strings.

        An empty list means the network is wired correctly.

        Nodes whose ``input_spec.source`` is ``"primitive"`` are treated as
        flexible transformers and bypass the sdf_scores source constraint —
        primitives can accept any upstream array.
        """
        issues: list[str] = []
        for name, node in self._nodes.items():
            fp = node.model.fingerprint
            is_primitive = fp.input_spec.source == "primitive"
            if isinstance(node.model, FittedModel):
                model_classes = tuple(node.model.class_ids)
                output_classes = tuple(fp.output_spec.classes)
                if model_classes != output_classes:
                    issues.append(
                        f"Node {name!r}: model classes {model_classes!r} do not "
                        f"match output classes {output_classes!r}"
                    )
                missing_scales = [
                    class_id for class_id in model_classes
                    if class_id not in node.model.score_scales
                ]
                if missing_scales:
                    issues.append(
                        f"Node {name!r}: missing score scales for {missing_scales!r}"
                    )
                calibrator_width = getattr(
                    node.model.calibrator, "n_features_in_", None,
                )
                if (
                    calibrator_width is not None
                    and calibrator_width != len(model_classes)
                ):
                    issues.append(
                        f"Node {name!r}: calibrator expects {calibrator_width} "
                        f"scores but model emits {len(model_classes)}"
                    )
            if node.upstream:
                # Downstream node must declare sdf_scores or primitive input
                if not is_primitive and fp.input_spec.source != "sdf_scores":
                    issues.append(
                        f"Node {name!r}: has upstream nodes but "
                        f"input_spec.source={fp.input_spec.source!r} "
                        f"(expected 'sdf_scores' or 'primitive')"
                    )
                if not is_primitive:
                    upstream_width = sum(
                        self._nodes[up_name].fingerprint.output_spec.dim
                        for up_name in node.upstream
                    )
                    if fp.input_spec.dim > 0 and fp.input_spec.dim != upstream_width:
                        issues.append(
                            f"Node {name!r}: input dimension {fp.input_spec.dim} "
                            f"does not match upstream score width {upstream_width}"
                        )
                    for up_name in node.upstream:
                        up_fp = self._nodes[up_name].model.fingerprint
                        if not fp.accepts_from(up_fp):
                            issues.append(
                                f"Edge {up_name!r} \u2192 {name!r}: "
                                f"upstream task {up_fp.task_name!r} not accepted by "
                                f"{name!r} (allowed: {fp.input_spec.upstream_tasks or 'any'})"
                            )
            else:
                # Source node must not declare sdf_scores input
                if fp.input_spec.source == "sdf_scores":
                    issues.append(
                        f"Node {name!r}: no upstream nodes but "
                        f"input_spec.source='sdf_scores'"
                    )
        return issues

    def find_swappable(self, name: str) -> list[str]:
        """Return names of all other nodes whose model is swappable with *name*."""
        target = self._nodes[name].model
        return [
            n for n, node in self._nodes.items()
            if n != name and target.is_swappable_with(node.model)
        ]

    def summary(self) -> str:
        """Human-readable DAG description."""
        lines = ["ModelNetwork:"]
        for name in self._topological_order():
            node = self._nodes[name]
            src = " + ".join(f"[{u}]" for u in node.upstream) or "<raw>"
            lines.append(f"  {src}  →  [{name}]  {node.model.fingerprint}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ModelNetwork(nodes={list(self._nodes)})"
