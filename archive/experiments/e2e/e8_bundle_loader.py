"""Load E8 point-cloud and text models from explicit bundle state."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from experiments.e2e.e5_bundle_loader import ExplicitReadout, ExplicitTransform
from experiments.e2e.run_tier4_smoke import _deserialize_experts
from src.inference_engine import InferenceEngine
from src.runtime import LocalModelBundleStore, ModelBundleManifest
from src.sdf_engine import Expert


@dataclass(frozen=True)
class LoadedE8Bundle:
    manifest: ModelBundleManifest
    point_experts: list[Expert]
    point_alpha: float
    text_transform: ExplicitTransform
    text_models: dict[int, list[Expert]]
    text_scales: dict[int, float]
    text_readout: ExplicitReadout
    text_alpha: float

    def point_scores(self, points: np.ndarray) -> np.ndarray:
        return InferenceEngine(
            self.point_experts, alpha=self.point_alpha,
        ).get_fused_sdf(points)

    def text_predict(self, contexts: np.ndarray) -> np.ndarray:
        transformed = self.text_transform.transform(contexts)
        class_ids = np.asarray(sorted(self.text_models), dtype=np.int32)
        scores = np.column_stack([
            InferenceEngine(
                self.text_models[int(class_id)], alpha=self.text_alpha,
            ).get_fused_sdf(transformed) / self.text_scales[int(class_id)]
            for class_id in class_ids
        ])
        probabilities = self.text_readout.predict_proba(scores)
        return self.text_readout.classes[np.argmax(probabilities, axis=1)]


def load_e8_bundle(root: str | Path) -> LoadedE8Bundle:
    root = Path(root)
    store = LocalModelBundleStore(root)
    manifest = store.current()
    if manifest is None:
        raise FileNotFoundError("E8 registry has no current bundle")
    components = root / "bundles" / manifest.bundle_id / "components"
    point_state = json.loads(
        (components / "point_model.json").read_text(encoding="utf-8")
    )
    text_state = json.loads(
        (components / "text_model.json").read_text(encoding="utf-8")
    )
    with np.load(components / "text_transform.npz", allow_pickle=False) as state:
        transform = ExplicitTransform(
            pca_components=state["pca_components"],
            pca_mean=state["pca_mean"],
            pca_explained_variance=state["pca_explained_variance"],
            lda_scalings=state["lda_scalings"],
            lda_xbar=state["lda_xbar"],
            scaler_mean=state["scaler_mean"],
            scaler_scale=state["scaler_scale"],
        )
    with np.load(components / "text_readout.npz", allow_pickle=False) as state:
        readout = ExplicitReadout(
            classes=state["classes"],
            classifier_classes=state["classifier_classes"],
            coefficient=state["classifier_coef"],
            intercept=state["classifier_intercept"],
            input_mean=state["classifier_mean"],
            input_scale=state["classifier_scale"],
        )
    return LoadedE8Bundle(
        manifest=manifest,
        point_experts=_deserialize_experts(point_state),
        point_alpha=float(point_state["alpha"]),
        text_transform=transform,
        text_models={
            int(class_id): _deserialize_experts(payload)
            for class_id, payload in text_state["class_models"].items()
        },
        text_scales={
            int(class_id): float(scale)
            for class_id, scale in text_state["score_scales"].items()
        },
        text_readout=readout,
        text_alpha=float(text_state["alpha"]),
    )