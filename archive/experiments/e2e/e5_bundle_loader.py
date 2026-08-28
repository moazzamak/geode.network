"""Load the explicit E4 candidate state for routing qualification."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from experiments.e2e.run_tier4_smoke import _deserialize_experts
from src.inference_engine import InferenceEngine
from src.runtime import LocalModelBundleStore, ModelBundleManifest
from src.sdf_engine import Expert


@dataclass(frozen=True)
class ExplicitTransform:
    pca_components: np.ndarray
    pca_mean: np.ndarray
    pca_explained_variance: np.ndarray
    lda_scalings: np.ndarray
    lda_xbar: np.ndarray
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray

    def transform(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        pca_features = (features - self.pca_mean) @ self.pca_components.T
        pca_features /= np.sqrt(self.pca_explained_variance)
        lda_features = (pca_features - self.lda_xbar) @ self.lda_scalings
        lda_features = lda_features[:, :len(self.scaler_mean)]
        return (lda_features - self.scaler_mean) / self.scaler_scale


@dataclass(frozen=True)
class ExplicitReadout:
    classes: np.ndarray
    classifier_classes: np.ndarray
    coefficient: np.ndarray
    intercept: np.ndarray
    input_mean: np.ndarray
    input_scale: np.ndarray

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        standardized = (scores - self.input_mean) / self.input_scale
        logits = standardized @ self.coefficient.T + self.intercept
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        columns = [
            int(np.flatnonzero(self.classifier_classes == class_id)[0])
            for class_id in self.classes
        ]
        return probabilities[:, columns]


@dataclass(frozen=True)
class LoadedE4Candidate:
    manifest: ModelBundleManifest
    transform: ExplicitTransform
    class_models: dict[int, list[Expert]]
    score_scales: dict[int, float]
    readout: ExplicitReadout
    alpha: float

    @property
    def class_ids(self) -> np.ndarray:
        return np.asarray(sorted(self.class_models), dtype=np.int32)

    def raw_scores(self, transformed_features: np.ndarray) -> np.ndarray:
        return np.column_stack([
            InferenceEngine(self.class_models[int(class_id)], alpha=self.alpha)
            .get_fused_sdf(transformed_features) / self.score_scales[int(class_id)]
            for class_id in self.class_ids
        ])

    def predict(self, raw_features: np.ndarray) -> np.ndarray:
        transformed = self.transform.transform(raw_features)
        probabilities = self.readout.predict_proba(self.raw_scores(transformed))
        return self.class_ids[np.argmax(probabilities, axis=1)]


def load_e4_candidate(root: str | Path) -> LoadedE4Candidate:
    store = LocalModelBundleStore(root)
    manifest = store.current()
    if manifest is None:
        raise FileNotFoundError("E4 model registry has no current bundle")
    path = Path(root) / "bundles" / manifest.bundle_id / "components"
    with np.load(path / "transform.npz", allow_pickle=False) as transform_state:
        transform = ExplicitTransform(
            pca_components=transform_state["pca_components"],
            pca_mean=transform_state["pca_mean"],
            pca_explained_variance=transform_state["pca_explained_variance"],
            lda_scalings=transform_state["lda_scalings"],
            lda_xbar=transform_state["lda_xbar"],
            scaler_mean=transform_state["scaler_mean"],
            scaler_scale=transform_state["scaler_scale"],
        )
    model_state = json.loads(
        (path / "model_state.json").read_text(encoding="utf-8")
    )
    class_models = {
        int(class_id): _deserialize_experts(payload)
        for class_id, payload in model_state["class_models"].items()
    }
    score_scales = {
        int(class_id): float(scale)
        for class_id, scale in model_state["score_scales"].items()
    }
    with np.load(path / "readout.npz", allow_pickle=False) as readout_state:
        readout = ExplicitReadout(
            classes=readout_state["classes"],
            classifier_classes=readout_state["classifier_classes"],
            coefficient=readout_state["classifier_coef"],
            intercept=readout_state["classifier_intercept"],
            input_mean=readout_state["classifier_mean"],
            input_scale=readout_state["classifier_scale"],
        )
    node = manifest.nodes[0]
    return LoadedE4Candidate(
        manifest=manifest,
        transform=transform,
        class_models=class_models,
        score_scales=score_scales,
        readout=readout,
        alpha=node.fingerprint.alpha,
    )