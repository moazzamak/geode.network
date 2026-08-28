"""Focused tests for M19 frozen representation infrastructure.

Covers: serialization/replay, each objective term, ablations, tuple cap,
deterministic selection, no test leakage, hash sensitivity, cache/binding
mismatch fail-closed, and extraction preprocessing contracts.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common.experiment_manifest import canonical_json
from experiments.common.v5_frozen_representations import (
    FeatureCacheMetadata,
    RepresentationManifest,
    compute_objective_hash,
    compute_preprocessing_digest,
    compute_split_hash,
    load_manifest,
    require_cache_binding,
    require_feature_dimension,
    save_manifest,
    verify_cache_file_integrity,
)
from src.representation_adapter import (
    AffineInterface,
    InterfaceConfig,
    LambdaTuple,
    between_class_margin,
    complexity_penalty,
    cross_entropy_loss,
    select_lambda_tuple,
    train_interface,
    within_class_compactness,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def simple_data(rng):
    """Simple 2D data with 3 classes."""
    n_per_class = 50
    features = np.vstack([
        rng.normal(loc=[0, 0], scale=0.3, size=(n_per_class, 2)),
        rng.normal(loc=[3, 0], scale=0.3, size=(n_per_class, 2)),
        rng.normal(loc=[1.5, 3], scale=0.3, size=(n_per_class, 2)),
    ])
    labels = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class, dtype=np.int64)
    return features, labels


@pytest.fixture
def identity_manifest():
    """A valid identity representation manifest."""
    return RepresentationManifest(
        backbone_id="test-backbone",
        upstream_weights_digest="a" * 64,
        preprocessing_digest="b" * 64,
        interface_architecture="identity",
        interface_weights_digest="none",
        objective_hash="c" * 64,
        training_split_hash="d" * 64,
        output_dimension=384,
        checkpoint_source="https://example.test/test-backbone",
        checkpoint_license="apache-2.0",
        token_pooling_policy="cls_token",
    )


# ---------------------------------------------------------------------------
# Test: Objective terms
# ---------------------------------------------------------------------------


class TestCrossEntropyLoss:
    def test_returns_positive_loss(self, simple_data):
        features, labels = simple_data
        n_classes = 3
        cls_w = np.zeros((2, n_classes))
        cls_b = np.zeros(n_classes)
        loss, grad_f, grad_w, grad_b = cross_entropy_loss(features, labels, cls_w, cls_b)
        assert loss > 0.0
        assert np.isfinite(loss)

    def test_gradient_shape(self, simple_data):
        features, labels = simple_data
        cls_w = np.zeros((2, 3))
        cls_b = np.zeros(3)
        _, grad_f, grad_w, grad_b = cross_entropy_loss(features, labels, cls_w, cls_b)
        assert grad_f.shape == features.shape
        assert grad_w.shape == cls_w.shape
        assert grad_b.shape == cls_b.shape

    def test_perfect_predictions_low_loss(self):
        """When classifier is perfect, loss should be low."""
        features = np.eye(3, dtype=np.float64)
        labels = np.array([0, 1, 2], dtype=np.int64)
        cls_w = np.eye(3, dtype=np.float64) * 10.0
        cls_b = np.zeros(3)
        loss, _, _, _ = cross_entropy_loss(features, labels, cls_w, cls_b)
        assert loss < 0.1


class TestWithinClassCompactness:
    def test_zero_for_identical_points(self):
        features = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        labels = np.array([0, 0, 0], dtype=np.int64)
        loss, grad = within_class_compactness(features, labels)
        assert loss == 0.0
        assert np.allclose(grad, 0.0)

    def test_positive_for_spread_points(self, simple_data):
        features, labels = simple_data
        loss, grad = within_class_compactness(features, labels)
        assert loss > 0.0
        assert grad.shape == features.shape


class TestBetweenClassMargin:
    def test_no_violation_for_distant_classes(self):
        features = np.array([[0, 0], [0, 0], [10, 0], [10, 0]], dtype=np.float64)
        labels = np.array([0, 0, 1, 1], dtype=np.int64)
        loss, grad = between_class_margin(features, labels, target_margin=1.0)
        assert loss == 0.0

    def test_violation_for_close_classes(self):
        features = np.array([[0, 0], [0, 0], [0.1, 0], [0.1, 0]], dtype=np.float64)
        labels = np.array([0, 0, 1, 1], dtype=np.int64)
        loss, grad = between_class_margin(features, labels, target_margin=1.0)
        assert loss > 0.0

    def test_single_class_returns_zero(self):
        features = np.array([[1, 2], [3, 4]], dtype=np.float64)
        labels = np.array([0, 0], dtype=np.int64)
        loss, grad = between_class_margin(features, labels)
        assert loss == 0.0


class TestComplexityPenalty:
    def test_zero_for_zero_weights(self):
        w = np.zeros((3, 2))
        b = np.zeros(2)
        loss, gw, gb = complexity_penalty(w, b)
        assert loss == 0.0

    def test_positive_for_nonzero_weights(self):
        w = np.ones((3, 2))
        b = np.ones(2)
        loss, gw, gb = complexity_penalty(w, b)
        assert loss == 8.0  # 6 + 2


# ---------------------------------------------------------------------------
# Test: Interface serialization and replay
# ---------------------------------------------------------------------------


class TestInterfaceSerialization:
    def test_identity_roundtrip(self):
        config = InterfaceConfig("identity", 10, 10)
        iface = AffineInterface(config, seed=11)
        payload = iface.to_dict()
        restored = AffineInterface.from_dict(payload)
        assert np.array_equal(iface.weight, restored.weight)
        assert np.array_equal(iface.bias, restored.bias)

    def test_linear_roundtrip(self):
        config = InterfaceConfig("linear", 20, 8)
        iface = AffineInterface(config, seed=42)
        payload = iface.to_dict()
        restored = AffineInterface.from_dict(payload)
        assert np.array_equal(iface.weight, restored.weight)
        assert np.array_equal(iface.bias, restored.bias)

    def test_low_rank_roundtrip(self):
        config = InterfaceConfig("low_rank", 20, 8, rank=4)
        iface = AffineInterface(config, seed=42)
        payload = iface.to_dict()
        restored = AffineInterface.from_dict(payload)
        assert np.array_equal(iface.weight, restored.weight)
        assert np.array_equal(iface.bias, restored.bias)

    def test_file_save_load(self, tmp_path):
        config = InterfaceConfig("linear", 16, 8)
        iface = AffineInterface(config, seed=11)
        path = tmp_path / "test_interface.json"
        file_hash = iface.save(path)
        loaded = AffineInterface.load(path)
        assert np.array_equal(iface.weight, loaded.weight)
        assert np.array_equal(iface.bias, loaded.bias)
        assert len(file_hash) == 64

    def test_replay_deterministic(self, simple_data):
        """Trained interface produces identical outputs on reload."""
        features, labels = simple_data
        n = len(features)
        train_f, dev_f = features[:100], features[100:]
        train_l, dev_l = labels[:100], labels[100:]

        config = InterfaceConfig("linear", 2, 2)
        lt = LambdaTuple(1.0, 1.0, 0.001)
        iface, _ = train_interface(config, train_f, train_l, dev_f, dev_l, lt, seed=11, max_epochs=10)

        original_output = iface.transform(features[:5])
        payload = iface.to_dict()
        restored = AffineInterface.from_dict(payload)
        replay_output = restored.transform(features[:5])
        assert np.array_equal(original_output, replay_output)


# ---------------------------------------------------------------------------
# Test: Ablations and tuple cap
# ---------------------------------------------------------------------------


class TestAblations:
    def test_identity_no_training(self, simple_data):
        """Identity interface requires no training."""
        features, labels = simple_data
        config = InterfaceConfig("identity", 2, 2)
        lt = LambdaTuple(0.0, 0.0, 0.0)
        iface, log = train_interface(config, features[:100], labels[:100],
                                     features[100:], labels[100:], lt)
        assert log["epochs"] == 0
        assert np.array_equal(iface.weight, np.eye(2))

    def test_ce_only_ablation(self, simple_data):
        """CE-only: compact=0, margin=0, complexity=0."""
        features, labels = simple_data
        config = InterfaceConfig("linear", 2, 2)
        lt = LambdaTuple(0.0, 0.0, 0.0)
        iface, log = train_interface(config, features[:100], labels[:100],
                                     features[100:], labels[100:], lt, seed=11, max_epochs=5)
        assert log["epochs"] > 0

    def test_metric_only_ablation(self, simple_data):
        """Metric-only: compact>0, margin>0, complexity=0, CE still runs for classifier."""
        features, labels = simple_data
        config = InterfaceConfig("linear", 2, 2)
        lt = LambdaTuple(1.0, 1.0, 0.0)
        iface, log = train_interface(config, features[:100], labels[:100],
                                     features[100:], labels[100:], lt, seed=11, max_epochs=5)
        assert log["epochs"] > 0


class TestTupleCap:
    def test_max_16_tuples_enforced(self, simple_data):
        """Cannot pass more than 16 tuples."""
        features, labels = simple_data
        config = InterfaceConfig("linear", 2, 2)
        tuples = [LambdaTuple(float(i), 0.0, 0.0) for i in range(17)]
        with pytest.raises(ValueError, match="At most 16"):
            select_lambda_tuple(config, features[:100], labels[:100],
                                features[100:], labels[100:], tuples)

    def test_empty_tuples_rejected(self, simple_data):
        features, labels = simple_data
        config = InterfaceConfig("linear", 2, 2)
        with pytest.raises(ValueError, match="At least one"):
            select_lambda_tuple(config, features[:100], labels[:100],
                                features[100:], labels[100:], [])


class TestDeterministicSelection:
    def test_same_seed_same_result(self, simple_data):
        """Same seed produces identical selection."""
        features, labels = simple_data
        config = InterfaceConfig("linear", 2, 2)
        tuples = [LambdaTuple(0.0, 0.0, 0.0), LambdaTuple(1.0, 1.0, 0.001)]

        best1, _ = select_lambda_tuple(config, features[:100], labels[:100],
                                       features[100:], labels[100:], tuples, seed=11)
        best2, _ = select_lambda_tuple(config, features[:100], labels[:100],
                                       features[100:], labels[100:], tuples, seed=11)
        assert best1.to_tuple() == best2.to_tuple()


# ---------------------------------------------------------------------------
# Test: No test leakage
# ---------------------------------------------------------------------------


class TestNoLeakage:
    def test_stratified_splits_disjoint(self):
        from experiments.tier4.prepare_v5_frozen_features import create_stratified_splits
        labels = np.repeat(np.arange(10), 100).astype(np.int64)
        train, dev, test = create_stratified_splits(labels, 30, 10, 10, seed=11)

        all_idx = np.concatenate([train, dev, test])
        assert len(all_idx) == len(set(all_idx.tolist())), "Splits must be disjoint"

    def test_all_classes_present_in_each_split(self):
        from experiments.tier4.prepare_v5_frozen_features import create_stratified_splits
        labels = np.repeat(np.arange(10), 100).astype(np.int64)
        train, dev, test = create_stratified_splits(labels, 30, 10, 10, seed=11)

        for split_name, idx in [("train", train), ("dev", dev), ("test", test)]:
            split_labels = labels[idx]
            assert len(np.unique(split_labels)) == 10, f"{split_name} missing classes"


# ---------------------------------------------------------------------------
# Test: Hash sensitivity
# ---------------------------------------------------------------------------


class TestHashSensitivity:
    def test_different_backbone_different_hash(self, identity_manifest):
        other = RepresentationManifest(
            backbone_id="other-backbone",
            upstream_weights_digest="a" * 64,
            preprocessing_digest="b" * 64,
            interface_architecture="identity",
            interface_weights_digest="none",
            objective_hash="c" * 64,
            training_split_hash="d" * 64,
            output_dimension=384,
            checkpoint_source="https://example.test/other-backbone",
            checkpoint_license="apache-2.0",
            token_pooling_policy="cls_token",
        )
        assert identity_manifest.representation_hash != other.representation_hash

    def test_different_weights_different_hash(self, identity_manifest):
        other = RepresentationManifest(
            backbone_id="test-backbone",
            upstream_weights_digest="e" * 64,
            preprocessing_digest="b" * 64,
            interface_architecture="identity",
            interface_weights_digest="none",
            objective_hash="c" * 64,
            training_split_hash="d" * 64,
            output_dimension=384,
            checkpoint_source="https://example.test/test-backbone",
            checkpoint_license="apache-2.0",
            token_pooling_policy="cls_token",
        )
        assert identity_manifest.representation_hash != other.representation_hash

    def test_different_preprocessing_different_hash(self, identity_manifest):
        other = RepresentationManifest(
            backbone_id="test-backbone",
            upstream_weights_digest="a" * 64,
            preprocessing_digest="f" * 64,
            interface_architecture="identity",
            interface_weights_digest="none",
            objective_hash="c" * 64,
            training_split_hash="d" * 64,
            output_dimension=384,
            checkpoint_source="https://example.test/test-backbone",
            checkpoint_license="apache-2.0",
            token_pooling_policy="cls_token",
        )
        assert identity_manifest.representation_hash != other.representation_hash

    def test_different_pooling_policy_different_hash(self, identity_manifest):
        payload = identity_manifest.to_dict()
        payload["token_pooling_policy"] = "mean_patch_tokens"
        payload.pop("representation_hash")
        other = RepresentationManifest.from_dict(payload)
        assert identity_manifest.representation_hash != other.representation_hash

    def test_different_interface_different_hash(self):
        base = RepresentationManifest(
            backbone_id="test-backbone",
            upstream_weights_digest="a" * 64,
            preprocessing_digest="b" * 64,
            interface_architecture="linear",
            interface_weights_digest="e" * 64,
            objective_hash="c" * 64,
            training_split_hash="d" * 64,
            output_dimension=64,
            checkpoint_source="https://example.test/test-backbone",
            checkpoint_license="apache-2.0",
            token_pooling_policy="cls_token",
        )
        other = RepresentationManifest(
            backbone_id="test-backbone",
            upstream_weights_digest="a" * 64,
            preprocessing_digest="b" * 64,
            interface_architecture="linear",
            interface_weights_digest="f" * 64,
            objective_hash="c" * 64,
            training_split_hash="d" * 64,
            output_dimension=64,
            checkpoint_source="https://example.test/test-backbone",
            checkpoint_license="apache-2.0",
            token_pooling_policy="cls_token",
        )
        assert base.representation_hash != other.representation_hash

    def test_split_hash_changes(self):
        idx1 = np.array([0, 1, 2, 3], dtype=np.int64)
        idx2 = np.array([0, 1, 2, 4], dtype=np.int64)
        assert compute_split_hash(idx1) != compute_split_hash(idx2)

    def test_objective_hash_changes(self):
        h1 = compute_objective_hash((1.0, 1.0, 0.001), "linear")
        h2 = compute_objective_hash((1.0, 2.0, 0.001), "linear")
        h3 = compute_objective_hash((1.0, 1.0, 0.001), "low_rank")
        assert h1 != h2
        assert h1 != h3


# ---------------------------------------------------------------------------
# Test: Cache/binding mismatch fail-closed
# ---------------------------------------------------------------------------


class TestBindingGuards:
    def test_cache_metadata_roundtrip(self, identity_manifest):
        metadata = FeatureCacheMetadata(
            representation_hash=identity_manifest.representation_hash,
            feature_file_hash="f" * 64,
            n_samples=100,
            feature_dimension=384,
            split_name="train",
        )
        assert FeatureCacheMetadata.from_dict(metadata.to_dict()) == metadata

    def test_matching_binding_passes(self, identity_manifest):
        cache = FeatureCacheMetadata(
            representation_hash=identity_manifest.representation_hash,
            feature_file_hash="f" * 64,
            n_samples=100,
            feature_dimension=384,
            split_name="train",
        )
        # Should not raise
        require_cache_binding(cache, identity_manifest)

    def test_mismatched_binding_fails(self, identity_manifest):
        cache = FeatureCacheMetadata(
            representation_hash="0" * 64,
            feature_file_hash="f" * 64,
            n_samples=100,
            feature_dimension=384,
            split_name="train",
        )
        with pytest.raises(ValueError, match="MISMATCH"):
            require_cache_binding(cache, identity_manifest)

    def test_dimension_mismatch_fails(self):
        cache = FeatureCacheMetadata(
            representation_hash="a" * 64,
            feature_file_hash="f" * 64,
            n_samples=100,
            feature_dimension=256,
            split_name="train",
        )
        with pytest.raises(ValueError, match="dimension"):
            require_feature_dimension(cache, expected_dim=384)

    def test_file_integrity_check(self, tmp_path):
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"hello world")
        correct_hash = hashlib.sha256(b"hello world").hexdigest()

        # Should not raise
        verify_cache_file_integrity(test_file, correct_hash)

        # Wrong hash should raise
        with pytest.raises(ValueError, match="integrity"):
            verify_cache_file_integrity(test_file, "0" * 64)


# ---------------------------------------------------------------------------
# Test: Manifest serialization
# ---------------------------------------------------------------------------


class TestManifestSerialization:
    def test_roundtrip(self, identity_manifest, tmp_path):
        path = tmp_path / "manifest.json"
        save_manifest(identity_manifest, path)
        loaded = load_manifest(path)
        assert loaded.representation_hash == identity_manifest.representation_hash
        assert loaded.backbone_id == identity_manifest.backbone_id

    def test_hash_mismatch_on_load_fails(self, identity_manifest, tmp_path):
        path = tmp_path / "manifest.json"
        payload = identity_manifest.to_dict()
        payload["representation_hash"] = "0" * 64
        path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="hash mismatch"):
            load_manifest(path)

    def test_schema_v1_manifest_remains_loadable(self, identity_manifest, tmp_path):
        payload = identity_manifest.to_dict()
        payload["schema_version"] = 1
        payload.pop("checkpoint_source")
        payload.pop("checkpoint_license")
        payload.pop("token_pooling_policy")
        legacy = RepresentationManifest.from_dict(
            {key: value for key, value in payload.items() if key != "representation_hash"}
        )
        payload["representation_hash"] = legacy.representation_hash
        path = tmp_path / "legacy_manifest.json"
        path.write_text(canonical_json(payload) + "\n", encoding="utf-8")

        loaded = load_manifest(path)

        assert loaded.schema_version == 1
        assert loaded.checkpoint_source is None

    @pytest.mark.parametrize(
        "field_name",
        ["checkpoint_source", "checkpoint_license", "token_pooling_policy"],
    )
    def test_schema_v2_requires_provenance_fields(
        self, identity_manifest, field_name
    ):
        payload = identity_manifest.to_dict()
        payload[field_name] = ""
        payload.pop("representation_hash")
        with pytest.raises(ValueError, match=field_name):
            RepresentationManifest.from_dict(payload)

    def test_invalid_sha256_rejected(self):
        with pytest.raises(ValueError):
            RepresentationManifest(
                backbone_id="test",
                upstream_weights_digest="not-a-hash",
                preprocessing_digest="b" * 64,
                interface_architecture="identity",
                interface_weights_digest="none",
                objective_hash="c" * 64,
                training_split_hash="d" * 64,
                output_dimension=384,
                checkpoint_source="https://example.test/test-backbone",
                checkpoint_license="apache-2.0",
                token_pooling_policy="cls_token",
            )


# ---------------------------------------------------------------------------
# Test: Preprocessing contracts
# ---------------------------------------------------------------------------


class TestPreprocessingContracts:
    def test_dinov2_output_shape(self):
        from experiments.tier4.prepare_v5_frozen_features import preprocess_image_dinov2
        config = {
            "size": {"shortest_edge": 256},
            "crop_size": {"height": 224, "width": 224},
            "rescale_factor": 1.0 / 255.0,
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
        }
        image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        result = preprocess_image_dinov2(image, config)
        assert result.shape == (3, 224, 224)
        assert result.dtype == np.float32

    def test_siglip_output_shape(self):
        from experiments.tier4.prepare_v5_frozen_features import preprocess_image_siglip
        config = {
            "size": {"height": 256, "width": 256},
            "rescale_factor": 1.0 / 255.0,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        }
        image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        result = preprocess_image_siglip(image, config)
        assert result.shape == (3, 256, 256)
        assert result.dtype == np.float32

    def test_ijepa_output_shape_and_range(self):
        from experiments.tier4.prepare_v5_frozen_features import preprocess_image_ijepa
        config = {
            "size": {"height": 448, "width": 448},
            "resample": 2,
            "rescale_factor": 1.0 / 255.0,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        }
        image = np.full((32, 32, 3), 255, dtype=np.uint8)

        result = preprocess_image_ijepa(image, config)

        assert result.shape == (3, 448, 448)
        assert result.dtype == np.float32
        assert np.allclose(result, 1.0)

    def test_declared_pooling_policies(self):
        from experiments.tier4.prepare_v5_frozen_features import _pool_features
        tokens = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        pooled = np.arange(8, dtype=np.float32).reshape(2, 4)

        assert np.array_equal(_pool_features(tokens, "cls_token"), tokens[:, 0, :])
        assert np.array_equal(
            _pool_features(tokens, "mean_patch_tokens"), tokens.mean(axis=1)
        )
        assert np.array_equal(_pool_features(pooled, "pooler_output"), pooled)

    def test_pooling_rejects_shape_mismatch(self):
        from experiments.tier4.prepare_v5_frozen_features import _pool_features
        with pytest.raises(ValueError, match="rank-3"):
            _pool_features(np.ones((2, 4)), "mean_patch_tokens")

    def test_pooling_selects_named_onnx_output(self):
        from experiments.tier4.prepare_v5_frozen_features import _select_model_output
        outputs = {
            "last_hidden_state": np.ones((2, 3, 4)),
            "pooler_output": np.full((2, 4), 2.0),
        }

        assert np.array_equal(
            _select_model_output(outputs, "pooler_output"),
            outputs["pooler_output"],
        )
        assert np.array_equal(
            _select_model_output(outputs, "mean_patch_tokens"),
            outputs["last_hidden_state"],
        )

    def test_pooling_fails_closed_when_named_output_is_absent(self):
        from experiments.tier4.prepare_v5_frozen_features import _select_model_output
        with pytest.raises(ValueError, match="pooler_output"):
            _select_model_output(
                {"last_hidden_state": np.ones((2, 3, 4))},
                "pooler_output",
            )


class TestFlowersCacheResume:
    def test_reuses_only_matching_complete_cache(self, tmp_path):
        from experiments.tier4.prepare_v5_flowers_features import _reuse_cache
        cache_path = tmp_path / "features_train_deadbeef.npz"
        features = np.arange(12, dtype=np.float64).reshape(3, 4)
        labels = np.array([0, 1, 2], dtype=np.int64)
        image_ids = np.array([11, 22, 33], dtype=np.int64)
        np.savez_compressed(
            cache_path,
            features=features,
            labels=labels,
            indices=image_ids,
        )

        metadata = _reuse_cache(
            cache_path,
            labels=labels,
            image_ids=image_ids,
            representation_hash="a" * 64,
            output_dimension=4,
            split="train",
        )

        assert metadata is not None
        assert metadata.n_samples == 3
        assert metadata.feature_dimension == 4

    def test_rejects_cache_with_wrong_source_ids(self, tmp_path):
        from experiments.tier4.prepare_v5_flowers_features import _reuse_cache
        cache_path = tmp_path / "features_train_deadbeef.npz"
        np.savez_compressed(
            cache_path,
            features=np.ones((3, 4)),
            labels=np.array([0, 1, 2]),
            indices=np.array([11, 22, 33]),
        )

        metadata = _reuse_cache(
            cache_path,
            labels=np.array([0, 1, 2]),
            image_ids=np.array([11, 22, 44]),
            representation_hash="a" * 64,
            output_dimension=4,
            split="train",
        )

        assert metadata is None

    def test_preprocessing_deterministic(self):
        from experiments.tier4.prepare_v5_frozen_features import preprocess_image_dinov2
        config = {
            "size": {"shortest_edge": 256},
            "crop_size": {"height": 224, "width": 224},
            "rescale_factor": 1.0 / 255.0,
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
        }
        image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        result1 = preprocess_image_dinov2(image, config)
        result2 = preprocess_image_dinov2(image, config)
        assert np.array_equal(result1, result2)


# ---------------------------------------------------------------------------
# Test: LambdaTuple validation
# ---------------------------------------------------------------------------


class TestLambdaTuple:
    def test_valid_creation(self):
        lt = LambdaTuple(1.0, 2.0, 0.001)
        assert lt.compact == 1.0
        assert lt.margin == 2.0
        assert lt.complexity == 0.001

    def test_negative_rejected(self):
        with pytest.raises(ValueError):
            LambdaTuple(-1.0, 0.0, 0.0)

    def test_nan_rejected(self):
        with pytest.raises(ValueError):
            LambdaTuple(float("nan"), 0.0, 0.0)

    def test_inf_rejected(self):
        with pytest.raises(ValueError):
            LambdaTuple(float("inf"), 0.0, 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
