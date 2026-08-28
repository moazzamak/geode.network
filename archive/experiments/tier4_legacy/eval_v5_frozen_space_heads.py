"""Evaluate frozen-space heads on M19 bounded S1 study.

Fits six heads from scratch in each frozen representation space:
1. Weighted k-nearest neighbors on normalized frozen features
2. Linear/logistic regression
3. RBF SVM (calibrated)
4. Prototype (nearest centroid with temperature)
5. Gaussian mixture (class-conditional)
6. Current non-discriminative GEODE

For no-interface high-dimensional spaces, no learned dimension reducer is applied.
If real GEODE cannot run under bounded resources, records a transparent blocked result.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common.classification_baselines import (
    ClassConditionalGMMClassifier,
    NearestCentroidClassifier,
    ShrinkageGaussianClassifier,
    WeightedKNNClassifier,
)
from experiments.common.classification_metrics import (
    accuracy,
    balanced_accuracy,
    negative_log_likelihood,
)
from experiments.common.experiment_manifest import canonical_json
from experiments.common.representation_metrics import (
    compute_representation_diagnostics,
)
from experiments.common.v5_frozen_representations import (
    FeatureCacheMetadata,
    RepresentationManifest,
    compute_objective_hash,
    compute_preprocessing_digest,
    compute_split_hash,
    require_cache_binding,
    verify_cache_file_integrity,
)
from src.representation_adapter import (
    AffineInterface,
    InterfaceConfig,
    LambdaTuple,
    select_lambda_tuple,
    train_interface,
    within_class_compactness,
)

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# ---------------------------------------------------------------------------
# Head implementations
# ---------------------------------------------------------------------------


def fit_weighted_knn_head(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_neighbors: int = 20,
    temperature: float = 0.07,
    query_batch_size: int = 1024,
) -> dict[str, Any]:
    """Fit a zero-training weighted kNN probe."""
    model = WeightedKNNClassifier(
        n_neighbors, temperature, query_batch_size
    ).fit(X_train, y_train)
    return {
        "model": model,
        "name": "weighted_knn",
        "classes": model.classes_,
        "configuration": {
            "n_neighbors": n_neighbors,
            "temperature": temperature,
            "query_batch_size": query_batch_size,
            "distance": "cosine",
            "normalization": "l2",
        },
    }


def fit_logistic_head(
    X_train: np.ndarray, y_train: np.ndarray, seed: int = 11
) -> dict[str, Any]:
    """Fit logistic regression head."""
    model = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=seed)
    model.fit(X_train, y_train)
    return {"model": model, "name": "linear_logistic", "classes": model.classes_}


def fit_rbf_head(
    X_train: np.ndarray, y_train: np.ndarray, seed: int = 11
) -> dict[str, Any]:
    """Fit calibrated RBF SVM head."""
    model = CalibratedClassifierCV(
        SVC(C=1.0, kernel="rbf", random_state=seed),
        method="sigmoid", cv=3,
    )
    model.fit(X_train, y_train)
    return {"model": model, "name": "rbf_svm", "classes": np.unique(y_train)}


def fit_prototype_head(
    X_train: np.ndarray, y_train: np.ndarray, seed: int = 11
) -> dict[str, Any]:
    """Fit nearest centroid (prototype) head."""
    model = NearestCentroidClassifier()
    model.fit(X_train, y_train)
    return {"model": model, "name": "prototype", "classes": model.classes_}


def fit_gaussian_mixture_head(
    X_train: np.ndarray, y_train: np.ndarray, seed: int = 11,
    components_per_class: int = 2,
) -> dict[str, Any]:
    """Fit class-conditional Gaussian mixture head."""
    classes = np.unique(y_train)
    components_by_class = {int(c): components_per_class for c in classes}
    model = ClassConditionalGMMClassifier(components_by_class, seed)
    model.fit(X_train, y_train)
    return {"model": model, "name": "gaussian_mixture", "classes": model.classes_}


def fit_geode_head(
    X_train: np.ndarray, y_train: np.ndarray, seed: int = 11,
    max_iterations: int = 200, consensus_threshold: float = 0.05,
    dimension_limit: int = 128,
) -> dict[str, Any]:
    """Fit current non-discriminative GEODE head.

    If the feature dimension exceeds dimension_limit or GEODE fails,
    records a blocked result transparently.
    """
    from src.greedy_constructor import GreedyConstructor
    from src.inference_engine import InferenceEngine

    dim = X_train.shape[1]
    if dim > dimension_limit:
        return {
            "model": None,
            "name": "current_geode",
            "status": "blocked",
            "reason": f"Feature dimension {dim} exceeds limit {dimension_limit} for bounded S1 CPU study.",
            "classes": np.unique(y_train),
        }

    classes = np.unique(y_train)
    class_models: list[Any] = []
    warnings: list[str] = []

    for c in classes:
        class_features = X_train[y_train == c]
        try:
            constructor = GreedyConstructor(
                consensus_threshold=consensus_threshold,
                capture_threshold=0.0,
                task_type="classification",
                max_iterations=max_iterations,
                min_growth_fraction=0.01,
                seed=seed + int(c),
            )
            model = constructor.build_model(class_features)
            class_models.append({"class": int(c), "model": model, "constructor": constructor})
        except Exception as e:
            warnings.append(f"GEODE class {c} fitting failed: {e}")
            class_models.append({"class": int(c), "model": None, "error": str(e)})

    return {
        "model": class_models,
        "name": "current_geode",
        "status": "fitted" if all(m["model"] is not None for m in class_models) else "partial",
        "classes": classes,
        "warnings": warnings,
    }


def predict_geode(
    class_models: list[dict[str, Any]],
    X: np.ndarray,
    classes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict using GEODE class models. Returns (predictions, probabilities)."""
    from src.inference_engine import InferenceEngine

    n = len(X)
    k = len(classes)
    # Use large but finite default for missing classes
    scores = np.full((n, k), 1e6, dtype=np.float64)

    for ci, cm in enumerate(class_models):
        if cm["model"] is None:
            continue
        experts = cm["model"]
        if not experts:
            continue
        engine = InferenceEngine(experts, alpha=1.0)
        for i in range(n):
            sdf_val = engine.get_fused_sdf(X[i:i + 1])
            val = float(sdf_val[0])
            if np.isfinite(val):
                scores[i, ci] = val
            else:
                scores[i, ci] = 1e6

    # Convert SDF scores to probabilities (lower SDF = closer to surface = more likely)
    neg_scores = -scores
    # Clip for numerical stability
    neg_scores = np.clip(neg_scores, -500.0, 500.0)
    shifted = neg_scores - neg_scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / (exp_scores.sum(axis=1, keepdims=True) + 1e-12)
    predictions = classes[probs.argmax(axis=1)]
    return predictions, probs


def compute_geode_component_efficiency(
    head_result: dict[str, Any],
    features: np.ndarray,
    labels: np.ndarray,
    *,
    target_coverage: float = 0.95,
    capture_threshold: float = 0.0,
) -> dict[str, Any]:
    """Count greedy-prefix GEODE primitives needed for per-class coverage."""
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("target_coverage must be in (0, 1].")
    if head_result.get("status") == "blocked":
        return {
            "status": "blocked",
            "reason": head_result["reason"],
            "target_coverage": target_coverage,
        }

    from src.sdf_engine import Expert

    per_class: list[dict[str, Any]] = []
    classes = np.asarray(head_result["classes"])
    for class_id, class_model in zip(
        classes,
        head_result["model"],
        strict=True,
    ):
        class_features = features[labels == class_id]
        experts = class_model["model"]
        if experts is None:
            per_class.append({
                "class": int(class_id),
                "status": "blocked",
                "reason": class_model.get("error", "GEODE fitting failed."),
            })
            continue

        active_experts: list[Expert] = []
        primitive_count = 0
        required_count: int | None = None
        achieved_coverage = 0.0
        for fitted_expert in experts:
            prefix_expert = Expert(alpha=fitted_expert.alpha)
            active_experts.append(prefix_expert)
            for primitive in fitted_expert.ellipsoids:
                prefix_expert.add_ellipsoid(primitive)
                primitive_count += 1
                captured = np.zeros(len(class_features), dtype=bool)
                for active_expert in active_experts:
                    captured |= (
                        active_expert.compute_sdf(class_features)
                        < capture_threshold
                    )
                achieved_coverage = float(np.mean(captured))
                if achieved_coverage >= target_coverage:
                    required_count = primitive_count
                    break
            if required_count is not None:
                break

        per_class.append({
            "class": int(class_id),
            "status": "reached" if required_count is not None else "unmet",
            "components_required": required_count,
            "components_available": sum(
                len(expert.ellipsoids) for expert in experts
            ),
            "achieved_coverage": achieved_coverage,
        })

    reached_counts = [
        record["components_required"]
        for record in per_class
        if record.get("status") == "reached"
    ]
    all_reached = len(reached_counts) == len(per_class)
    return {
        "status": "evaluated" if all_reached else "target_unmet",
        "target_coverage": target_coverage,
        "capture_threshold": capture_threshold,
        "ordering": "constructor_greedy_prefix",
        "classes_reaching_target": len(reached_counts),
        "class_count": len(per_class),
        "mean_components_required": (
            float(np.mean(reached_counts)) if all_reached else None
        ),
        "max_components_required": (
            int(max(reached_counts)) if all_reached else None
        ),
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_head(
    head_result: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Evaluate a fitted head on test data."""
    name = head_result["name"]

    if head_result.get("status") == "blocked":
        return {
            "head": name,
            "status": "blocked",
            "reason": head_result["reason"],
            "accuracy": None,
            "balanced_accuracy": None,
            "nll": None,
        }

    classes = head_result["classes"]
    start_time = time.perf_counter()

    if name == "current_geode":
        class_models = head_result["model"]
        if head_result.get("status") == "partial":
            return {
                "head": name,
                "status": "partial",
                "warnings": head_result.get("warnings", []),
                "accuracy": None,
                "balanced_accuracy": None,
                "nll": None,
            }
        predictions, probs = predict_geode(class_models, X_test, classes)
    else:
        model = head_result["model"]
        predictions = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)
        else:
            probs = None

    elapsed = time.perf_counter() - start_time
    acc = accuracy(y_test, predictions)
    bal_acc = balanced_accuracy(y_test, predictions)

    nll = None
    if probs is not None:
        try:
            nll_val = negative_log_likelihood(y_test, probs, classes)
            if np.isfinite(nll_val):
                nll = nll_val
        except (ValueError, RuntimeWarning):
            nll = None

    result = {
        "head": name,
        "status": "evaluated",
        "accuracy": float(acc) if np.isfinite(acc) else None,
        "balanced_accuracy": float(bal_acc) if np.isfinite(bal_acc) else None,
        "nll": float(nll) if nll is not None else None,
        "inference_wall_seconds": elapsed,
    }
    if "configuration" in head_result:
        result["configuration"] = head_result["configuration"]
    return result


# ---------------------------------------------------------------------------
# Compactness metric
# ---------------------------------------------------------------------------


def compute_compactness(features: np.ndarray, labels: np.ndarray) -> float:
    """Compute within-class compactness (mean squared distance to centroid)."""
    loss, _ = within_class_compactness(features, labels)
    return loss


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------


def run_evaluation(
    config_path: str | Path = "experiments/configs/v5/m19_frozen_space_s1.json",
    cifar_path: str | Path = "data/tier4/cifar10_features.npz",
    feature_dir: str | Path = "data/v5/features/m19_s1",
    output_dir: str | Path = "logs/results/v5/m19_frozen_space",
) -> dict[str, Any]:
    """Run the complete M19 evaluation pipeline."""
    config_path = Path(config_path)
    feature_dir = Path(feature_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    extraction_summary_path = feature_dir / "extraction_summary.json"
    if not extraction_summary_path.exists():
        raise FileNotFoundError(
            f"Frozen feature extraction summary not found: {extraction_summary_path}"
        )
    extraction_summary = json.loads(
        extraction_summary_path.read_text(encoding="utf-8")
    )

    # Load CIFAR-10 for splits
    data = np.load(str(cifar_path))
    all_labels = data["labels"].astype(np.int64)
    # Create splits
    split_cfg = config["split_protocol"]
    from experiments.tier4.prepare_v5_frozen_features import create_stratified_splits
    train_idx, dev_idx, test_idx = create_stratified_splits(
        all_labels,
        train_per_class=split_cfg["total_per_class_train"],
        dev_per_class=split_cfg["total_per_class_dev"],
        test_per_class=split_cfg["total_per_class_test"],
        seed=split_cfg["selection_seed"],
    )

    split_hashes = {
        "train": compute_split_hash(train_idx),
        "dev": compute_split_hash(dev_idx),
        "test": compute_split_hash(test_idx),
    }

    results: dict[str, Any] = {
        "schema_version": 1,
        "milestone": "M19",
        "stage": "S1",
        "seed": config["seed"],
        "split_hashes": split_hashes,
        "representations": {},
        "advancement_gates": {},
        "warnings": [],
        "blocked_arms": [],
    }

    seed = config["seed"]
    knn_cfg = config["weighted_knn"]
    lambda_tuples = [
        LambdaTuple(lt["compact"], lt["margin"], lt["complexity"])
        for lt in config["lambda_tuples"]
    ]

    best_head_accuracy: float = 0.0
    geode_results: list[dict[str, Any]] = []

    for backbone_cfg in config["backbones"]:
        backbone_id = backbone_cfg.get("id", backbone_cfg.get("name", "unknown"))

        if backbone_cfg.get("status") == "blocked":
            results["blocked_arms"].append({
                "backbone": backbone_id,
                "reason": backbone_cfg["reason"],
            })
            continue

        extracted = extraction_summary["representations"].get(backbone_id)
        if extracted is None or extracted.get("status") != "extracted":
            results["blocked_arms"].append({
                "backbone": backbone_id,
                "reason": "No extracted representation in the current feature summary",
            })
            continue

        source_manifest = RepresentationManifest.from_dict(extracted["manifest"])
        preproc_path = PROJECT_ROOT / backbone_cfg["preprocessor_config"]
        preproc_digest = compute_preprocessing_digest(preproc_path)
        expected_source = RepresentationManifest(
            backbone_id=backbone_id,
            upstream_weights_digest=backbone_cfg["weights_sha256"],
            preprocessing_digest=preproc_digest,
            interface_architecture="identity",
            interface_weights_digest="none",
            objective_hash=source_manifest.objective_hash,
            training_split_hash=split_hashes["train"],
            output_dimension=backbone_cfg["output_dimension"],
            checkpoint_source=backbone_cfg["checkpoint_source"],
            checkpoint_license=backbone_cfg["checkpoint_license"],
            token_pooling_policy=backbone_cfg["token_pooling_policy"],
        )
        if source_manifest.representation_hash != expected_source.representation_hash:
            results["blocked_arms"].append({
                "backbone": backbone_id,
                "reason": (
                    "Extracted feature provenance does not match the active "
                    "backbone configuration"
                ),
            })
            continue

        backbone_dir = feature_dir / backbone_id
        cache_metadata = {
            split: FeatureCacheMetadata.from_dict(payload)
            for split, payload in extracted["cache_metadata"].items()
        }
        cache_paths = {
            split: backbone_dir
            / f"features_{split}_{source_manifest.representation_hash[:16]}.npz"
            for split in ("train", "dev", "test")
        }
        for split in ("train", "dev", "test"):
            metadata = cache_metadata[split]
            require_cache_binding(metadata, source_manifest)
            verify_cache_file_integrity(
                cache_paths[split], metadata.feature_file_hash
            )

        train_data = np.load(str(cache_paths["train"]))
        dev_data = np.load(str(cache_paths["dev"]))
        test_data = np.load(str(cache_paths["test"]))

        X_train = train_data["features"].astype(np.float64)
        y_train = train_data["labels"].astype(np.int64)
        X_dev = dev_data["features"].astype(np.float64)
        y_dev = dev_data["labels"].astype(np.int64)
        X_test = test_data["features"].astype(np.float64)
        y_test = test_data["labels"].astype(np.int64)

        output_dim = backbone_cfg["output_dimension"]
        rep_results: dict[str, Any] = {
            "backbone_id": backbone_id,
            "output_dimension": output_dim,
            "interfaces": {},
        }

        # Evaluate each interface type
        for iface_cfg in config["interfaces"]:
            arch = iface_cfg["architecture"]
            iface_out_dim = iface_cfg.get("output_dim") or output_dim

            if arch == "identity":
                iface_config = InterfaceConfig(
                    architecture="identity",
                    input_dim=output_dim,
                    output_dim=output_dim,
                )
                interface = AffineInterface(iface_config, seed=seed)
                interface_log = {"architecture": "identity", "epochs": 0}
                X_train_t = X_train
                X_dev_t = X_dev
                X_test_t = X_test
            else:
                rank = iface_cfg.get("rank", 0)
                iface_config = InterfaceConfig(
                    architecture=arch,
                    input_dim=output_dim,
                    output_dim=iface_out_dim,
                    rank=rank,
                )

                # Select best lambda tuple on dev
                best_lambda, selection_log = select_lambda_tuple(
                    iface_config, X_train, y_train, X_dev, y_dev,
                    lambda_tuples, seed=seed, max_tuples=16,
                    **config["training"],
                )

                # Train with best lambda
                interface, interface_log = train_interface(
                    iface_config, X_train, y_train, X_dev, y_dev,
                    best_lambda, seed=seed, **config["training"],
                )
                interface_log["selected_lambdas"] = best_lambda.to_tuple()
                interface_log["selection_results"] = selection_log

                X_train_t = interface.transform(X_train)
                X_dev_t = interface.transform(X_dev)
                X_test_t = interface.transform(X_test)

            # Compute compactness in transformed space
            train_compactness = compute_compactness(X_train_t, y_train)
            representation_diagnostics = compute_representation_diagnostics(
                X_train_t,
                y_train,
            )

            # Compute interface artifact hash
            iface_weights_digest = interface.weights_digest()

            # Build representation manifest for this interface
            obj_hash = compute_objective_hash(
                interface_log.get("selected_lambdas", (0.0, 0.0, 0.0)), arch
            )
            manifest = RepresentationManifest(
                backbone_id=backbone_id,
                upstream_weights_digest=backbone_cfg["weights_sha256"],
                preprocessing_digest=preproc_digest,
                interface_architecture=arch,
                interface_weights_digest="none" if arch == "identity" else iface_weights_digest,
                objective_hash=obj_hash,
                training_split_hash=split_hashes["train"],
                output_dimension=iface_out_dim if arch != "identity" else output_dim,
                checkpoint_source=backbone_cfg["checkpoint_source"],
                checkpoint_license=backbone_cfg["checkpoint_license"],
                token_pooling_policy=backbone_cfg["token_pooling_policy"],
                parent_artifact=(
                    None if arch == "identity"
                    else source_manifest.representation_hash
                ),
            )
            if (
                arch == "identity"
                and manifest.representation_hash
                != source_manifest.representation_hash
            ):
                raise ValueError(
                    f"Identity manifest drift for backbone {backbone_id}."
                )
            rep_hash = manifest.representation_hash

            # Save interface artifact
            iface_artifact_path = output_dir / "interfaces" / f"{backbone_id}_{arch}_{rep_hash[:12]}.json"
            interface.save(iface_artifact_path)

            # Deterministic replay verification
            replay_interface = AffineInterface.load(iface_artifact_path)
            replay_features = replay_interface.transform(X_train[:10])
            original_features = interface.transform(X_train[:10])
            replay_match = bool(np.array_equal(replay_features, original_features))

            # Fit and evaluate heads
            head_results: dict[str, Any] = {}
            geode_cfg = config["geode_config"]

            heads_to_fit = [
                ("weighted_knn", lambda X, y: fit_weighted_knn_head(
                    X,
                    y,
                    n_neighbors=knn_cfg["n_neighbors"],
                    temperature=knn_cfg["temperature"],
                    query_batch_size=knn_cfg["query_batch_size"],
                )),
                ("linear_logistic", lambda X, y: fit_logistic_head(X, y, seed)),
                ("rbf_svm", lambda X, y: fit_rbf_head(X, y, seed)),
                ("prototype", lambda X, y: fit_prototype_head(X, y, seed)),
                ("gaussian_mixture", lambda X, y: fit_gaussian_mixture_head(X, y, seed)),
                ("current_geode", lambda X, y: fit_geode_head(
                    X, y, seed,
                    max_iterations=geode_cfg["max_iterations"],
                    consensus_threshold=geode_cfg["consensus_threshold"],
                )),
            ]

            for head_name, fit_fn in heads_to_fit:
                print(f"  Fitting {head_name} on {backbone_id}/{arch}...")
                fit_start = time.perf_counter()
                head = fit_fn(X_train_t, y_train)
                fit_time = time.perf_counter() - fit_start

                eval_result = evaluate_head(head, X_test_t, y_test)
                eval_result["fit_wall_seconds"] = fit_time
                if head_name == "current_geode":
                    eval_result["component_efficiency"] = (
                        compute_geode_component_efficiency(
                            head,
                            X_train_t,
                            y_train,
                            target_coverage=geode_cfg[
                                "component_efficiency_target_coverage"
                            ],
                            capture_threshold=geode_cfg["capture_threshold"],
                        )
                    )

                head_results[head_name] = eval_result

                # Track best accuracy for gates
                if eval_result.get("accuracy") is not None:
                    if eval_result["accuracy"] > best_head_accuracy:
                        best_head_accuracy = eval_result["accuracy"]
                    if head_name == "current_geode":
                        geode_results.append({
                            "backbone": backbone_id,
                            "interface": arch,
                            "accuracy": eval_result["accuracy"],
                            "rep_hash": rep_hash,
                        })

            iface_result = {
                "architecture": arch,
                "representation_hash": rep_hash,
                "interface_parameter_count": interface.parameter_count,
                "interface_serialized_bytes": interface.serialized_bytes,
                "training_log": interface_log,
                "compactness": train_compactness,
                "representation_diagnostics": representation_diagnostics,
                "replay_verified": replay_match,
                "heads": head_results,
            }
            rep_results["interfaces"][arch] = iface_result

        results["representations"][backbone_id] = rep_results

    # Advancement gates
    geode_within_threshold = False
    for gr in geode_results:
        gap = best_head_accuracy - gr["accuracy"]
        if gap <= 0.005:  # 0.5 percentage points
            geode_within_threshold = True

    results["advancement_gates"] = {
        "byte_replayable": all(
            r.get("interfaces", {}).get(arch, {}).get("replay_verified", False)
            for r in results["representations"].values()
            if isinstance(r, dict) and "interfaces" in r
            for arch in r.get("interfaces", {})
        ),
        "best_head_accuracy": best_head_accuracy,
        "geode_within_0_5pp": geode_within_threshold,
        "geode_results": geode_results,
        "notes": "S1 bounded feasibility only. S2/S3 confirmation pending.",
    }

    # Save results
    results_path = output_dir / "m19_s1_evidence.json"
    results_path.write_text(
        canonical_json(results) + "\n", encoding="utf-8", newline="\n"
    )

    # Save artifact index
    from experiments.common.v5_artifacts import build_artifact_index
    build_artifact_index(output_dir)

    return results


if __name__ == "__main__":
    run_evaluation()
