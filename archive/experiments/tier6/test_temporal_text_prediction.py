import copy
import inspect
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np

from experiments.common.moe_eval import fit_experts
from experiments.tier6.eval_refinement_ablation import (
    _align_probabilities,
    ordered_ablation_split,
)
from experiments.tier6.eval_temporal_text_prediction import (
    CACHE_VERSION,
    VOCAB_FINGERPRINT,
    VOCAB_SIZE,
    class_sample_adequacy,
    fit_adaptive_class_models,
    fit_score_calibrator,
    forward_chaining_splits,
    geometry_calibration_split,
    linear_context_accuracy,
    ngram_accuracy,
    prepare_text_corpus,
    predict_calibrated_labels,
    probability_perplexity,
    run_text_prediction_experiment,
    sample_context_pairs,
    sample_ensemble_state_pairs,
    sample_hybrid_state_pairs,
    sampled_ngram_accuracy,
    sample_temporal_state_pairs,
    supervised_refinement,
)
from src.sdf_engine import EllipsoidExpert, Expert, SoftminFusion
from src.runtime import LocalArtifactStore, LocalCheckpointStore
from src.runtime.refinement_checkpoint import RefinementCheckpointAdapter
from src.sdf_optimizer import SDFOptimizer
from src.temporal_sampler import (
    MultiSeedStateEncoder,
    MultiTimescaleStateEncoder,
    TemporalStateEncoder,
    temporal_state_pairs,
)
from verify_pipeline import _tier6_regression_error


class Tier6DataTests(unittest.TestCase):
    def test_ablation_splits_are_ordered_and_disjoint(self):
        splits = ordered_ablation_split(1_000, gap=5)
        names = ("geometry", "carve", "calibration", "validation")
        for first, second in zip(names, names[1:]):
            self.assertLess(splits[first][-1], splits[second][0])
            self.assertGreaterEqual(splits[second][0] - splits[first][-1], 6)
        combined = np.concatenate([splits[name] for name in names])
        self.assertEqual(len(combined), len(np.unique(combined)))

    def test_probability_alignment_retains_unmodeled_classes(self):
        probabilities = np.array([[0.75, 0.25], [0.10, 0.90]])
        aligned = _align_probabilities(
            probabilities,
            source_classes=np.array([1, 3]),
            evaluation_classes=np.array([1, 2, 3]),
        )
        np.testing.assert_allclose(aligned[:, [0, 2]], probabilities)
        np.testing.assert_array_equal(aligned[:, 1], 0.0)

    def test_fixed_refinement_set_bypasses_temporal_resampling(self):
        ellipsoid = EllipsoidExpert(np.zeros(2), np.ones(2))
        models = {0: [Expert([ellipsoid])], 1: [Expert([copy.deepcopy(ellipsoid)])]}
        refinement_X = np.array([
            [-0.5, 0.0], [-0.2, 0.1], [0.2, -0.1], [0.5, 0.0],
        ])
        refinement_y = np.array([0, 0, 1, 1])
        with patch(
            "experiments.tier6.eval_temporal_text_prediction.sample_context_pairs",
            side_effect=AssertionError("fixed refinement must not resample"),
        ), redirect_stdout(io.StringIO()):
            refined, scales = supervised_refinement(
                models=models,
                train_ids=np.array([], dtype=np.int32),
                pca=None,
                lda=None,
                scaler=None,
                window=1,
                alpha=2.0,
                score_scales={0: 1.0, 1: 1.0},
                n_iters=1,
                n_epochs=1,
                learning_rate=0.001,
                max_samples=4,
                seed=3,
                batch_size=4,
                max_batches_per_epoch=1,
                refinement_X=refinement_X,
                refinement_y=refinement_y,
            )
        self.assertIs(refined, models)
        self.assertEqual(set(scales), {0, 1})

    def test_supervised_refinement_resumes_exact_epoch_state(self):
        def make_models():
            result = {}
            for class_id, center in ((0, -0.6), (1, 0.6)):
                expert = Expert(alpha=2.0)
                expert.add_ellipsoid(EllipsoidExpert([center, 0.0], [0.8, 0.8]))
                result[class_id] = [expert]
            return result

        refinement_X = np.asarray([
            [-0.8, -0.1], [-0.5, 0.1], [-0.3, 0.0],
            [0.3, 0.0], [0.5, -0.1], [0.8, 0.1],
        ])
        refinement_y = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32)
        common = {
            "train_ids": np.array([], dtype=np.int32),
            "pca": None,
            "lda": None,
            "scaler": None,
            "window": 1,
            "alpha": 2.0,
            "score_scales": {0: 1.0, 1: 1.0},
            "n_iters": 1,
            "n_epochs": 4,
            "learning_rate": 0.001,
            "max_samples": 6,
            "seed": 13,
            "batch_size": 2,
            "max_batches_per_epoch": 3,
            "refinement_X": refinement_X,
            "refinement_y": refinement_y,
        }
        uninterrupted_models = make_models()
        uninterrupted_history = []
        with redirect_stdout(io.StringIO()):
            _, uninterrupted_scales = supervised_refinement(
                models=uninterrupted_models,
                epoch_history=uninterrupted_history,
                **common,
            )

        with tempfile.TemporaryDirectory() as directory:
            adapter = RefinementCheckpointAdapter(
                LocalCheckpointStore(LocalArtifactStore(directory))
            )
            with self.assertRaisesRegex(RuntimeError, "epoch 2"), redirect_stdout(io.StringIO()):
                supervised_refinement(
                    models=make_models(),
                    epoch_history=[],
                    checkpoint_adapter=adapter,
                    checkpoint_run_id="run-1",
                    checkpoint_attempt_id="attempt-1",
                    fail_after_global_epoch=2,
                    **common,
                )
            resumed_models = make_models()
            resumed_history = []
            with redirect_stdout(io.StringIO()):
                _, resumed_scales = supervised_refinement(
                    models=resumed_models,
                    epoch_history=resumed_history,
                    checkpoint_adapter=adapter,
                    checkpoint_run_id="run-1",
                    checkpoint_attempt_id="attempt-1",
                    **common,
                )

        self.assertEqual(resumed_history, uninterrupted_history)
        self.assertEqual(resumed_scales, uninterrupted_scales)
        for class_id in uninterrupted_models:
            expected = uninterrupted_models[class_id][0].ellipsoids[0]
            actual = resumed_models[class_id][0].ellipsoids[0]
            np.testing.assert_array_equal(actual.center, expected.center)
            np.testing.assert_array_equal(actual.radii, expected.radii)
            np.testing.assert_array_equal(actual.orientation, expected.orientation)

    def test_context_targets_are_aligned(self):
        ids = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        X, y = sample_context_pairs(ids, window=2, max_samples=None)

        decoded = X.reshape(len(X), 2, VOCAB_SIZE).argmax(axis=2)
        np.testing.assert_array_equal(decoded, [[1, 2], [2, 3], [3, 4]])
        np.testing.assert_array_equal(y, [3, 4, 5])

    def test_out_of_vocabulary_ids_are_rejected(self):
        ids = np.array([1, VOCAB_SIZE, 2], dtype=np.int32)
        with self.assertRaisesRegex(ValueError, "outside the current vocabulary"):
            sample_context_pairs(ids, window=1)

    def test_contiguous_temporal_state_targets_are_aligned(self):
        ids = np.tile(np.array([1, 2, 3], dtype=np.int32), 20)
        states, targets = sample_temporal_state_pairs(
            ids, state_dim=5, max_samples=None, seed=3, warmup=4,
        )
        self.assertEqual(states.shape, (len(ids) - 1, 5))
        np.testing.assert_array_equal(targets, ids[1:])

    def test_temporal_encoder_identity_is_independent_of_sampling_seed(self):
        ids = np.tile(np.array([1, 2, 3], dtype=np.int32), 20)
        states_a, _ = sample_temporal_state_pairs(
            ids, state_dim=5, max_samples=None, seed=1, encoder_seed=9,
        )
        states_b, _ = sample_temporal_state_pairs(
            ids, state_dim=5, max_samples=None, seed=99, encoder_seed=9,
        )
        np.testing.assert_allclose(states_a, states_b)

    def test_hybrid_representation_aligns_exact_context_and_future_target(self):
        ids = np.tile(np.array([1, 2, 3, 4], dtype=np.int32), 15)
        features, targets = sample_hybrid_state_pairs(
            ids, window=2, state_dim=5, max_samples=None, seed=3, warmup=4,
        )
        exact = features[:, :2 * VOCAB_SIZE]
        decoded = exact.reshape(len(exact), 2, VOCAB_SIZE).argmax(axis=2)
        np.testing.assert_array_equal(decoded[0], [1, 2])
        self.assertEqual(targets[0], 3)
        self.assertEqual(features.shape[1], 2 * VOCAB_SIZE + 5)

    def test_hybrid_representation_is_causal(self):
        ids = np.tile(np.array([1, 2, 3, 4], dtype=np.int32), 20)
        changed = ids.copy()
        changed[50:] = 7
        original, _ = sample_hybrid_state_pairs(
            ids, window=2, state_dim=6, max_samples=None, encoder_seed=9,
        )
        modified, _ = sample_hybrid_state_pairs(
            changed, window=2, state_dim=6, max_samples=None, encoder_seed=9,
        )
        np.testing.assert_allclose(original[:48], modified[:48])

    def test_ensemble_pair_sampling_aligns_future_targets(self):
        ids = np.tile(np.array([1, 2, 3], dtype=np.int32), 20)
        for variant in ("multi_timescale", "multi_seed"):
            with self.subTest(variant=variant):
                states, targets = sample_ensemble_state_pairs(
                    ids,
                    state_dim=8,
                    variant=variant,
                    max_samples=None,
                    seed=4,
                    warmup=5,
                )
                self.assertEqual(states.shape, (len(ids) - 1, 8))
                np.testing.assert_array_equal(targets, ids[1:])

    def test_versioned_cache_is_validated(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            path = os.path.join(cache_dir, "wikitext103_full.npz")
            np.savez_compressed(
                path,
                train_ids=np.array([1, 2, 3], dtype=np.int32),
                test_ids=np.array([2, 3], dtype=np.int32),
                cache_version=np.array(CACHE_VERSION, dtype=np.int32),
                vocab_fingerprint=np.array(VOCAB_FINGERPRINT),
            )
            train, test = prepare_text_corpus(cache_dir=cache_dir)
            np.testing.assert_array_equal(train, [1, 2, 3])
            np.testing.assert_array_equal(test, [2, 3])

            np.savez_compressed(
                path,
                train_ids=np.array([1], dtype=np.int32),
                test_ids=np.array([1], dtype=np.int32),
                cache_version=np.array(CACHE_VERSION + 1, dtype=np.int32),
                vocab_fingerprint=np.array(VOCAB_FINGERPRINT),
            )
            with self.assertRaisesRegex(ValueError, "Incompatible Tier 6 cache"):
                prepare_text_corpus(cache_dir=cache_dir)

    def test_synthetic_periodic_corpus_is_offline_and_chronological(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            train, test = prepare_text_corpus(
                dataset="synthetic_periodic",
                max_chars=100,
                cache_dir=cache_dir,
            )
        self.assertEqual(len(train), 80)
        self.assertEqual(len(test), 20)

    def test_variable_order_corpus_is_seeded_and_uses_long_history(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            first_train, first_test = prepare_text_corpus(
                dataset="synthetic_variable_order",
                max_chars=200,
                seed=12,
                cache_dir=cache_dir,
            )
        with tempfile.TemporaryDirectory() as cache_dir:
            second_train, second_test = prepare_text_corpus(
                dataset="synthetic_variable_order",
                max_chars=200,
                seed=12,
                cache_dir=cache_dir,
            )
        np.testing.assert_array_equal(first_train, second_train)
        np.testing.assert_array_equal(first_test, second_test)
        self.assertGreater(len(np.unique(first_train)), 3)
        with tempfile.TemporaryDirectory() as cache_dir:
            seed_a, _ = prepare_text_corpus(
                dataset="synthetic_variable_order",
                max_chars=200,
                seed=12,
                cache_dir=cache_dir,
            )
            seed_b, _ = prepare_text_corpus(
                dataset="synthetic_variable_order",
                max_chars=200,
                seed=13,
                cache_dir=cache_dir,
            )
        self.assertFalse(np.array_equal(seed_a, seed_b))


class Tier6EvaluationTests(unittest.TestCase):
    def test_hard_csg_gradient_matches_finite_differences(self):
        expert = Expert(alpha=2.0)
        expert.add_ellipsoid(EllipsoidExpert([0.0], [2.0]))
        expert.add_ellipsoid(EllipsoidExpert([0.5], [0.5], polarity=-1))

        epsilon = 1e-6
        for point in [-1.0, 0.7]:
            x = np.array([point], dtype=np.float64)
            plus = expert.compute_sdf(x.reshape(1, -1) + epsilon)[0]
            minus = expert.compute_sdf(x.reshape(1, -1) - epsilon)[0]
            numerical = (plus - minus) / (2.0 * epsilon)
            analytic = expert.compute_gradient(x)[0]
            self.assertAlmostEqual(analytic, numerical, places=6)

    def test_pruned_softmin_preserves_total_mixture_normalization(self):
        alpha = 2.0
        experts = []
        for center in [0.0, 100.0, 200.0, 300.0]:
            expert = Expert(alpha=alpha)
            expert.add_ellipsoid(EllipsoidExpert([center], [1.0]))
            experts.append(expert)
        points = np.array([[0.0], [0.5]], dtype=np.float64)

        all_sdfs = np.array([
            expert.compute_sdf(points) for expert in experts
        ])
        logits = -alpha * all_sdfs
        maximum = logits.max(axis=0)
        expected = -(
            maximum
            + np.log(np.exp(logits - maximum).sum(axis=0) / len(experts))
        ) / alpha
        actual = SoftminFusion(alpha=alpha).fuse(experts, points)

        np.testing.assert_allclose(actual, expected, atol=1e-4)

    def test_expert_construction_is_reproducible_for_seed(self):
        points = np.random.default_rng(2).normal(size=(35, 2))
        kwargs = {
            "points": points,
            "consensus_threshold": 0.2,
            "capture_threshold": 0.2,
            "alpha": 2.0,
            "max_iterations": 20,
            "nudge_iterations": 0,
            "nudge_learning_rate": 0.01,
            "seed": 17,
        }
        first = fit_experts(**kwargs)
        second = fit_experts(**kwargs)

        self.assertEqual(len(first), len(second))
        for first_expert, second_expert in zip(first, second):
            self.assertEqual(
                len(first_expert.ellipsoids), len(second_expert.ellipsoids),
            )
            for first_ellipsoid, second_ellipsoid in zip(
                first_expert.ellipsoids, second_expert.ellipsoids,
            ):
                np.testing.assert_allclose(
                    first_ellipsoid.center, second_ellipsoid.center,
                )
                np.testing.assert_allclose(
                    first_ellipsoid.radii, second_ellipsoid.radii,
                )
                np.testing.assert_allclose(
                    first_ellipsoid.orientation, second_ellipsoid.orientation,
                )

    def test_forward_splits_only_train_on_the_past(self):
        for train, validation in forward_chaining_splits(30, 2, gap=3):
            self.assertLess(train[-1] + 3, validation[0])

    def test_temporal_state_is_fixed_width_and_causal(self):
        observations = np.arange(20, dtype=np.float64).reshape(10, 2) / 20.0
        changed_future = observations.copy()
        changed_future[6:] += 100.0
        encoder = TemporalStateEncoder(state_dim=4, seed=5)
        states = encoder.transform(observations)
        changed_states = encoder.transform(changed_future)
        self.assertEqual(states.shape, (10, 4))
        np.testing.assert_allclose(states[:6], changed_states[:6])

    def test_reservoir_ensembles_preserve_total_width_and_causality(self):
        observations = np.arange(30, dtype=np.float64).reshape(10, 3) / 30.0
        changed = observations.copy()
        changed[6:] += 20.0
        for encoder in (
            MultiTimescaleStateEncoder(state_dim=8, seed=3),
            MultiSeedStateEncoder(state_dim=8, member_count=3, seed=3),
        ):
            with self.subTest(encoder=type(encoder).__name__):
                states = encoder.transform(observations)
                changed_states = encoder.transform(changed)
                self.assertEqual(states.shape, (10, 8))
                np.testing.assert_allclose(states[:6], changed_states[:6])

    def test_temporal_state_pairs_align_future_targets(self):
        observations = np.eye(6, 2)
        targets = np.arange(6)
        encoder = TemporalStateEncoder(state_dim=3, seed=8)
        states, future_targets = temporal_state_pairs(
            observations, targets, encoder, lag=2,
        )
        self.assertEqual(states.shape, (4, 3))
        np.testing.assert_array_equal(future_targets, [2, 3, 4, 5])

    def test_geometry_calibration_split_is_purged(self):
        geometry, calibration = geometry_calibration_split(
            np.arange(100), calibration_fraction=0.2, gap=3,
        )
        self.assertEqual(len(calibration), 20)
        self.assertLess(geometry[-1] + 3, calibration[0])

    def test_sample_adequacy_uses_each_class_count(self):
        labels = np.array([0] * 20 + [1] * 3 + [2])
        result = class_sample_adequacy(labels, dimension=2)
        self.assertEqual(result["min_seed"], 5)
        self.assertEqual(result["below_minimum"], 2)
        self.assertEqual(result["median_count"], 3.0)

    def test_adaptive_models_cover_sparse_classes(self):
        rng = np.random.default_rng(11)
        X = np.vstack([
            rng.normal(-1.0, 0.1, size=(4, 3)),
            rng.normal(1.0, 0.1, size=(2, 3)),
            rng.normal(3.0, 0.1, size=(1, 3)),
        ])
        y = np.array([0] * 4 + [1] * 2 + [2], dtype=np.int32)
        models, complexity = fit_adaptive_class_models(
            X=X,
            y=y,
            class_ids=np.unique(y),
            consensus_threshold=0.1,
            capture_threshold=0.1,
            alpha=2.0,
            max_iterations=5,
            nudge_iterations=0,
            nudge_learning_rate=0.01,
        )
        self.assertTrue(all(models[class_id] for class_id in np.unique(y)))
        self.assertEqual(complexity[0], "diagonal")
        self.assertEqual(complexity[1], "spherical")
        self.assertEqual(complexity[2], "spherical")

    def test_ngram_baseline_uses_requested_window(self):
        train = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        test = np.array([1, 2, 3, 1, 2, 3])
        self.assertEqual(ngram_accuracy(train, test, window=2), 1.0)

    def test_matched_ngram_uses_exact_sampled_pairs(self):
        train = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3], dtype=np.int32)
        test = np.array([1, 2, 3, 1, 2, 3], dtype=np.int32)
        X_train, y_train = sample_context_pairs(train, window=2, max_samples=None)
        X_test, y_test = sample_context_pairs(test, window=2, max_samples=None)
        self.assertEqual(
            sampled_ngram_accuracy(X_train, y_train, X_test, y_test, window=2),
            1.0,
        )

    def test_linear_control_learns_periodic_context(self):
        ids = np.tile(np.array([1, 2, 3], dtype=np.int32), 100)
        X, y = sample_context_pairs(ids, window=2, max_samples=None)
        split = 200
        accuracy = linear_context_accuracy(
            X[:split], y[:split], X[split:], y[split:], seed=4,
        )
        self.assertGreater(accuracy, 0.95)

    def test_pipeline_options_are_supported(self):
        parameters = inspect.signature(run_text_prediction_experiment).parameters
        self.assertIn("use_subtractive", parameters)

    def test_scaled_optimizer_step_reduces_toy_loss(self):
        models = {}
        for class_id, center in [(0, -1.0), (1, 1.0)]:
            expert = Expert(alpha=2.0)
            expert.add_ellipsoid(EllipsoidExpert([center], [0.6]))
            models[class_id] = [expert]

        X = np.array([[-0.7], [-0.5], [0.5], [0.7]], dtype=np.float64)
        y = np.array([0, 0, 1, 1], dtype=np.int32)
        optimizer = SDFOptimizer(
            models=models,
            alpha=2.0,
            learning_rate=0.01,
            momentum=0.0,
            score_scales={0: 2.0, 1: 0.5},
        )
        initial_loss = optimizer.step(X, y)
        next_loss = optimizer.step(X, y)
        self.assertLess(next_loss, initial_loss)

    def test_optimizer_is_invariant_to_duplicate_batches(self):
        models = {}
        for class_id, center in [(0, -1.0), (1, 1.0)]:
            expert = Expert(alpha=2.0)
            expert.add_ellipsoid(EllipsoidExpert([center], [0.6]))
            models[class_id] = [expert]
        duplicate_models = copy.deepcopy(models)
        X = np.array([[-0.7], [-0.5], [0.5], [0.7]], dtype=np.float64)
        y = np.array([0, 0, 1, 1], dtype=np.int32)

        optimizer = SDFOptimizer(
            models, alpha=2.0, learning_rate=0.01, momentum=0.0,
        )
        duplicate_optimizer = SDFOptimizer(
            duplicate_models, alpha=2.0, learning_rate=0.01, momentum=0.0,
        )
        loss = optimizer.step(X, y)
        duplicate_loss = duplicate_optimizer.step(
            np.concatenate([X, X]), np.concatenate([y, y]),
        )

        self.assertAlmostEqual(loss, duplicate_loss)
        for class_id in models:
            actual = models[class_id][0].ellipsoids[0]
            duplicate = duplicate_models[class_id][0].ellipsoids[0]
            np.testing.assert_allclose(actual.center, duplicate.center)
            np.testing.assert_allclose(actual.radii, duplicate.radii)
            np.testing.assert_allclose(actual.orientation, duplicate.orientation)

    def test_optimizer_evaluation_does_not_update_geometry(self):
        expert = Expert(alpha=2.0)
        expert.add_ellipsoid(EllipsoidExpert([0.0], [1.0]))
        models = {0: [expert]}
        optimizer = SDFOptimizer(models, alpha=2.0)
        X = np.array([[-0.5], [0.5]], dtype=np.float64)
        y = np.array([0, 0], dtype=np.int32)
        before = copy.deepcopy(expert.ellipsoids[0])

        first = optimizer.evaluate(X, y)
        second = optimizer.evaluate(X, y)

        self.assertEqual(first, second)
        np.testing.assert_array_equal(expert.ellipsoids[0].center, before.center)
        np.testing.assert_array_equal(expert.ellipsoids[0].radii, before.radii)
        np.testing.assert_array_equal(
            expert.ellipsoids[0].orientation, before.orientation,
        )

    def test_optimizer_rejects_subtractive_geometry(self):
        expert = Expert(alpha=2.0)
        expert.add_ellipsoid(EllipsoidExpert([0.0], [1.0], polarity=-1))
        with self.assertRaisesRegex(ValueError, "additive ellipsoids only"):
            SDFOptimizer(models={0: [expert]})

    def test_calibrator_repairs_class_score_offsets(self):
        rng = np.random.default_rng(7)
        y = np.repeat(np.arange(3), 80)
        scores = rng.normal(2.0, 0.2, size=(len(y), 3))
        scores[np.arange(len(y)), y] = rng.normal(0.0, 0.2, size=len(y))
        scores += np.array([0.0, -1.0, -3.0])

        raw_accuracy = np.mean(np.argmin(scores, axis=1) == y)
        calibrator = fit_score_calibrator(scores[:180], y[:180])
        predictions, probabilities = predict_calibrated_labels(calibrator, scores[180:])
        calibrated_accuracy = np.mean(predictions == y[180:])

        self.assertGreater(calibrated_accuracy, raw_accuracy + 0.3)
        self.assertTrue(np.isfinite(
            probability_perplexity(y[180:], probabilities, calibrator.classes_)
        ))

    def test_perplexity_penalizes_unmodeled_targets(self):
        probabilities = np.array([[0.9, 0.1], [0.8, 0.2]])
        perplexity = probability_perplexity(
            np.array([0, 2]), probabilities, np.array([0, 1]),
        )
        self.assertGreater(perplexity, 1e5)

    def test_pipeline_gate_rejects_refinement_regression(self):
        error = _tier6_regression_error({
            "test_acc": 0.3,
            "test_acc_init": 0.3,
            "test_acc_refined": 0.28,
            "unigram_acc": 0.2,
            "ppl_init": 4.0,
            "class_count": 3,
        })
        self.assertIn("Refinement regression", error)

    def test_full_experiment_smoke(self):
        train = np.tile(np.array([1, 2, 3], dtype=np.int32), 1000)
        test = np.tile(np.array([1, 2, 3], dtype=np.int32), 300)
        with patch(
            "experiments.tier6.eval_temporal_text_prediction.prepare_text_corpus",
            return_value=(train, test),
        ), redirect_stdout(io.StringIO()):
            result = run_text_prediction_experiment(
                max_train_samples=600,
                max_test_samples=180,
                window=2,
                pca_components=3,
                n_folds=1,
                max_iterations=10,
                nudge_iterations=0,
                n_refinement_iters=0,
                calibration_fraction=0.2,
                use_subtractive=False,
                seed=7,
            )
        self.assertGreater(result["test_acc_final"], result["unigram_acc"])
        self.assertEqual(result["class_count"], 3)


if __name__ == "__main__":
    unittest.main()