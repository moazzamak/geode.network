import copy
import io
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, log_loss

from experiments.common.classification_metrics import (
    accuracy,
    balanced_accuracy,
    expected_calibration_error,
    multiclass_brier_score,
    negative_log_likelihood,
    paired_bootstrap_interval,
    top_k_accuracy,
)
from experiments.common.classification_baselines import (
    fit_classification_baselines,
)
from experiments.common.experiment_manifest import (
    append_manifest,
    array_fingerprint,
    experiment_id,
    read_manifests,
)
from experiments.common.model_stats import model_structure_stats
from experiments.common.score_readouts import fit_all_readouts, fit_score_readout
from experiments.common.result_records import classification_result_record
from experiments.common.geometry_metrics import (
    sample_fused_surface,
    symmetric_chamfer_distance,
)
from experiments.common.ellipsoid_distance_reference import (
    numerical_closest_point,
    numerical_signed_distance,
)
from experiments.common.ellipsoid_fitters import ELLIPSOID_FITTERS
from experiments.common.ood_metrics import (
    conformal_prediction_sets,
    conformal_probability_threshold,
    conformal_set_metrics,
    ood_detection_metrics,
    ood_operating_point,
    risk_coverage_curve,
    select_ood_threshold,
    select_ood_threshold_at_known_coverage,
)
from experiments.common.ood_scores import (
    fit_class_conditional_ood_scorers,
    fit_feature_ood_scorers,
    maximum_probability_score,
    minimum_sdf_score,
    sdf_energy_score,
)
from experiments.common.robustness_corruptions import (
    apply_covariance_shift,
    class_conditional_label_noise,
    inject_feature_outliers,
    mask_feature_dimensions,
    symmetric_label_noise,
)
from experiments.common.primitive_stability import evaluate_primitive_stability
from src.sdf_engine import EllipsoidExpert, Expert
from src.inference_engine import InferenceEngine
from src.model_editor import ModelEditor
from src.model_fingerprint import InputSpec, ModelFingerprint, OutputSpec
from src.model_network import FittedModel, ModelNetwork
from src.model_migration import dry_run_add_class_migration
from src.orchestrator import DIRECT, ExecutionPlan, Orchestrator, PlanStep
from src.open_set import (
    UNKNOWN_LABEL,
    OpenSetPrediction,
    OpenSetReason,
    SupportProfile,
    apply_rejection_policy,
)
from src.probabilistic_engine import (
    ProbabilisticInferenceEngine,
    gaussian_primitive_nll,
)
from src.rejection_buffer import RejectionBuffer
from src.streaming_discovery import StreamingClusterPolicy, StreamingRejectionMemory
from src.replay_constrained_fitter import fit_replay_constrained_expert
from src.candidate_usefulness import (
    CandidateUsefulnessPolicy,
    evaluate_candidate_usefulness,
)
from src.discovery_policy import ClusterProposalPolicy, evaluate_cluster_proposal
from src.discovery_clustering import (
    estimated_kmeans_rejections,
    finch_rejections,
    hdbscan_rejections,
)
from src.feedback_constraints import (
    PairwiseConstraint,
    build_pairwise_constraints,
    confirm_pairwise_constraints,
    fit_diagonal_constraint_metric,
    refine_rejection_partition,
    select_constraint_queries,
    validate_pairwise_constraints,
)
from src.candidate_routing import (
    CertifiedTopKRouter,
    batched_exact_bound_routing,
    class_major_exact_bound_routing,
    exact_bound_routing,
)
from src.model_compression import compress_primitive_budget
from src.adaptation_policy import (
    AdaptationAction,
    AdaptationCandidateEvidence,
    AdaptationGatePolicy,
    ConfirmationKind,
    select_adaptation_action,
)
from src.greedy_constructor import GreedyConstructor
from src.gpu_engine import (
    GPUInferenceEngine,
    fit_axis_aligned_candidates_gpu,
    select_device,
)
from verify_pipeline import _geometry_normalized_residual_score, _r2_score
from experiments.tier4.eval_complex_classification import (
    evaluate_score_readouts,
    stratified_geometry_calibration_split,
    stratified_geometry_carve_calibration_split,
)
from experiments.tier4.eval_csg_ablation import run_csg_ablation_experiment
from experiments.tier4.eval_probabilistic_field import (
    run_probabilistic_field_ablation,
)
from experiments.tier4.eval_hybrid_field import run_hybrid_field_ablation
from experiments.tier4.summarize_csg_ablation import summarize_runs
from experiments.tier4.summarize_hybrid_field import (
    summarize_runs as summarize_hybrid_runs,
)
from experiments.tier4.summarize_global_temperature import (
    summarize_runs as summarize_global_temperature_runs,
)
from experiments.tier4.summarize_per_class_temperature import (
    summarize_runs as summarize_per_class_temperature_runs,
)
from experiments.tier4.eval_controlled_ood import run_controlled_ood_experiment
from experiments.tier4.eval_real_ood import run_feature_ood_experiment
from experiments.tier4.eval_real_feature_event_review import (
    _fit_review_feedback_metric,
    evaluate_event_review_payload,
    run_event_review_transfer,
)
from experiments.tier1.eval_metric_distance import evaluate_metric_distance
from experiments.tier4.prepare_ood_features import deterministic_sample_indices
from experiments.tier5.eval_fitter_candidate_budgets import (
    run_benchmark as run_fitter_budget_benchmark,
)
from experiments.tier5.eval_high_dimensional_fitters import (
    run_benchmark as run_fitter_benchmark,
)
from experiments.tier5.eval_corruption_robustness import (
    BASELINE_NAMES,
    run_benchmark as run_corruption_benchmark,
)
from experiments.tier5.eval_editability_scaling import (
    build_scaling_conditions,
    run_scaling_condition,
)
from experiments.tier4.eval_leave_class_out import (
    run_leave_class_out_episode,
    run_leave_class_out_study,
)
from experiments.tier4.eval_unknown_streaming import (
    STREAM_EVENT_TYPES,
    generate_class_incremental_stream,
    run_frozen_stream_family_study,
    run_streaming_policy_transfer_study,
    run_streaming_discovery_study,
)
from experiments.tier4.eval_adaptation_actions import run_adaptation_action_study
from experiments.tier4.eval_geode_adaptation_transactions import (
    run_geode_transaction_study,
)
from experiments.tier4.eval_calibrated_graph_migration import (
    run_calibrated_graph_migration,
)
from experiments.tier4.eval_calibrated_mode_update import (
    run_calibrated_mode_update,
)
from experiments.tier4.eval_real_feature_mode_update import (
    run_real_feature_mode_update,
)
from experiments.tier4.eval_real_feature_ood_transfer import (
    run_real_feature_ood_episode,
)
from experiments.tier4.eval_real_feature_support_objective import (
    fit_support_calibrator,
)
from experiments.tier4.eval_ambiguity_resolution import (
    run_ambiguity_resolution,
)


class OpenSetContractTests(unittest.TestCase):
    def setUp(self):
        self.profile = SupportProfile(
            model_signature="model-123",
            feature_transform_fingerprint="transform-456",
            training_dataset_fingerprint="train-789",
            calibration_dataset_fingerprint="calibration-012",
            class_ids=(0, 1),
            score_scales=(1.0, 2.0),
            novelty_score="maximum_probability",
            global_threshold=0.4,
            class_thresholds=(0.3, 0.5),
            routing_centroids=((0.0, 0.0), (2.0, 2.0)),
            routing_radii=(1.0, 1.5),
            version="v1",
            fit_seed=42,
            created_at="2026-07-25T00:00:00Z",
        )

    def test_support_profile_round_trip_and_compatibility(self):
        restored = SupportProfile.from_dict(self.profile.to_dict())
        self.assertEqual(restored, self.profile)
        self.assertEqual(restored.profile_id, self.profile.profile_id)
        self.assertEqual(restored.threshold_for(1), 0.5)
        restored.assert_compatible(
            model_signature="model-123",
            class_ids=(0, 1),
            feature_transform_fingerprint="transform-456",
        )
        with self.assertRaisesRegex(ValueError, "model_signature"):
            restored.assert_compatible(
                model_signature="stale-model",
                class_ids=(0, 1),
                feature_transform_fingerprint="transform-456",
            )

    def test_profile_rejects_misaligned_routing_keys(self):
        payload = self.profile.to_dict()
        payload["routing_radii"] = [1.0]
        with self.assertRaisesRegex(ValueError, "routing_radii"):
            SupportProfile.from_dict(payload)

    def test_prediction_contract_uses_explicit_unknown_label(self):
        accepted = OpenSetPrediction(
            label=1,
            accepted=True,
            candidate_model_signature="model-123",
            candidate_class_id=1,
            raw_novelty_score=0.2,
            calibrated_novelty_score=0.25,
            threshold=0.5,
            decision_margin=-0.25,
            support_profile_version="v1",
            reason_code=OpenSetReason.ACCEPTED,
        )
        rejected = OpenSetPrediction(
            label=UNKNOWN_LABEL,
            accepted=False,
            candidate_model_signature="model-123",
            candidate_class_id=1,
            raw_novelty_score=0.8,
            calibrated_novelty_score=0.75,
            threshold=0.5,
            decision_margin=0.25,
            support_profile_version="v1",
            reason_code=OpenSetReason.OUTSIDE_SUPPORT,
        )
        self.assertEqual(accepted.label, 1)
        self.assertEqual(rejected.label, UNKNOWN_LABEL)

    def test_rejection_policy_uses_frozen_per_class_thresholds(self):
        scores = np.array([[0.2, 0.8], [0.8, 0.4], [0.3, 0.7]])
        result = apply_rejection_policy(
            scores,
            np.array([0, 1, 0]),
            (0, 1),
            self.profile,
            model_signature="model-123",
            feature_transform_fingerprint="transform-456",
            probabilities=np.array([
                [0.9, 0.1], [0.2, 0.8], [0.7, 0.3],
            ]),
        )
        self.assertEqual(
            [prediction.accepted for prediction in result.predictions],
            [True, True, False],
        )
        self.assertEqual(
            [prediction.threshold for prediction in result.predictions],
            [0.3, 0.5, 0.3],
        )
        self.assertEqual(result.predictions[2].label, UNKNOWN_LABEL)
        self.assertEqual(result.candidates_evaluated, 6)
        self.assertEqual(result.exact_sdf_evaluations, 0)

    def test_rejection_policy_rejects_stale_profile(self):
        with self.assertRaisesRegex(ValueError, "model_signature"):
            apply_rejection_policy(
                np.array([[0.2, 0.8]]),
                np.array([0]),
                (0, 1),
                self.profile,
                model_signature="stale-model",
                feature_transform_fingerprint="transform-456",
                probabilities=np.array([[0.9, 0.1]]),
            )

    def test_fitted_model_open_set_is_opt_in_and_counts_exact_scores(self):
        models = {}
        for class_id, center in ((0, 0.0), (1, 5.0)):
            expert = Expert(alpha=2.0)
            expert.add_ellipsoid(EllipsoidExpert(
                center=np.array([center, 0.0]), radii=np.ones(2),
            ))
            models[class_id] = [expert]
        fingerprint = ModelFingerprint(
            task_name="toy",
            input_spec=InputSpec("passthrough", dim=2),
            output_spec=OutputSpec("sdf_scores", (0, 1)),
        )
        model = FittedModel(
            fingerprint=fingerprint,
            class_models=models,
            score_scales={0: 1.0, 1: 1.0},
        )
        profile_payload = self.profile.to_dict()
        profile_payload.update({
            "model_signature": fingerprint.signature,
            "novelty_score": "minimum_sdf",
            "global_threshold": 0.5,
            "class_thresholds": [],
        })
        profile = SupportProfile.from_dict(profile_payload)
        points = np.array([[0.0, 0.0], [20.0, 0.0]])
        closed_predictions = model.predict(points)
        result = model.predict_open_set(
            points,
            profile,
            feature_transform_fingerprint="transform-456",
        )
        np.testing.assert_array_equal(model.predict(points), closed_predictions)
        self.assertTrue(result.predictions[0].accepted)
        self.assertFalse(result.predictions[1].accepted)
        self.assertEqual(result.exact_sdf_evaluations, 4)

    def test_orchestrator_open_set_reuses_plan_scores(self):
        models = {}
        for class_id, center in ((0, 0.0), (1, 5.0)):
            expert = Expert(alpha=2.0)
            expert.add_ellipsoid(EllipsoidExpert(
                center=np.array([center, 0.0]), radii=np.ones(2),
            ))
            models[class_id] = [expert]
        fingerprint = ModelFingerprint(
            task_name="toy",
            input_spec=InputSpec("passthrough", dim=2),
            output_spec=OutputSpec("sdf_scores", (0, 1)),
        )
        model = FittedModel(fingerprint, models, {0: 1.0, 1: 1.0})
        orchestrator = Orchestrator()
        orchestrator._network.add_node("toy", model)
        orchestrator._models["toy"] = model
        plan = ExecutionPlan(
            goal="toy",
            capability_type=DIRECT,
            steps=[PlanStep(node_name="toy")],
            result_node="toy",
        )
        profile_payload = self.profile.to_dict()
        profile_payload.update({
            "model_signature": fingerprint.signature,
            "novelty_score": "minimum_sdf",
            "global_threshold": 0.5,
            "class_thresholds": [],
        })
        result = orchestrator.run_open_set(
            np.array([[0.0, 0.0], [20.0, 0.0]]),
            plan,
            SupportProfile.from_dict(profile_payload),
            feature_transform_fingerprint="transform-456",
        )
        self.assertEqual(result["label"].tolist(), [0, UNKNOWN_LABEL])
        self.assertEqual(result["open_set"].candidates_evaluated, 4)
        self.assertEqual(result["open_set"].exact_sdf_evaluations, 4)
        self.assertEqual(result["open_set"].counters.nodes_executed, 1)
        self.assertEqual(
            result["open_set"].counters.score_values_materialized, 4,
        )

    def test_graph_validation_detects_incomplete_class_width_migration(self):
        source_fingerprint = ModelFingerprint(
            task_name="source",
            input_spec=InputSpec("passthrough", dim=2),
            output_spec=OutputSpec("sdf_scores", (0, 1)),
        )
        source = FittedModel(source_fingerprint, {0: [], 1: []}, {0: 1.0, 1: 1.0})
        downstream_fingerprint = ModelFingerprint(
            task_name="downstream",
            input_spec=InputSpec("sdf_scores", ("source",), dim=2),
            output_spec=OutputSpec("sdf_scores", (0, 1)),
        )
        downstream = FittedModel(
            downstream_fingerprint, {0: [], 1: []}, {0: 1.0, 1: 1.0},
        )
        network = ModelNetwork()
        network.add_node("source", source)
        network.add_node("downstream", downstream, upstream=["source"])
        self.assertEqual(network.validate(), [])

        source.class_models[2] = []
        source.score_scales[2] = 1.0

        issues = network.validate()
        self.assertTrue(any("model classes" in issue for issue in issues))

    def test_new_class_migration_requires_downstream_replacement_and_never_publishes(self):
        source_fingerprint = ModelFingerprint(
            task_name="source",
            input_spec=InputSpec("passthrough", dim=2),
            output_spec=OutputSpec("sdf_scores", (0, 1)),
        )
        source = FittedModel(source_fingerprint, {0: [], 1: []}, {0: 1.0, 1: 1.0})
        downstream_fingerprint = ModelFingerprint(
            task_name="downstream",
            input_spec=InputSpec("sdf_scores", ("source",), dim=2),
            output_spec=OutputSpec("sdf_scores", (0, 1)),
        )
        downstream = FittedModel(
            downstream_fingerprint, {0: [], 1: []}, {0: 1.0, 1: 1.0},
        )
        network = ModelNetwork()
        network.add_node("source", source)
        network.add_node("downstream", downstream, upstream=["source"])
        with self.assertRaises(ValueError):
            dry_run_add_class_migration(
                network,
                source_node="source",
                new_class_id=2,
                new_class_models=[],
                score_scale=1.0,
            )

        migrated_downstream = FittedModel(
            ModelFingerprint(
                task_name="downstream",
                input_spec=InputSpec("sdf_scores", ("source",), dim=3),
                output_spec=OutputSpec("sdf_scores", (0, 1)),
            ),
            {0: [], 1: []},
            {0: 1.0, 1: 1.0},
        )
        dry_run = dry_run_add_class_migration(
            network,
            source_node="source",
            new_class_id=2,
            new_class_models=[],
            score_scale=1.0,
            downstream_replacements={"downstream": migrated_downstream},
        )

        self.assertTrue(dry_run.valid)
        self.assertFalse(dry_run.published)
        self.assertNotEqual(dry_run.old_signature, dry_run.new_signature)
        self.assertEqual(source.class_ids, [0, 1])
        self.assertEqual(
            dry_run.candidate_network._nodes["source"].model.class_ids,
            [0, 1, 2],
        )


class AdaptationPolicyTests(unittest.TestCase):
    def test_candidate_usefulness_requires_gain_headroom_and_coverage(self):
        decision = evaluate_candidate_usefulness(
            baseline_success=0.84,
            geometric_coverage=0.16,
            policy=CandidateUsefulnessPolicy(0.5, 0.5),
        )

        self.assertFalse(decision.eligible)
        self.assertAlmostEqual(decision.maximum_possible_gain, 0.16)
        self.assertEqual(
            decision.failed_criteria,
            ("insufficient_gain_headroom", "insufficient_geometric_coverage"),
        )

    def test_replay_constrained_fit_keeps_exclusions_outside_margin(self):
        positives = np.asarray([[1.0, 0.0], [1.2, 0.1], [1.1, -0.1]])
        exclusions = np.asarray([[1.4, 0.0], [3.0, 3.0]])

        fit = fit_replay_constrained_expert(
            positives, exclusions, exclusion_margin=0.2,
        )
        sdf = fit.expert.ellipsoids[0].compute_sdf(exclusions)

        self.assertTrue(np.all(sdf >= 0.2))
        self.assertEqual(fit.exclusion_violations, 0)
        self.assertLess(fit.radius_scale, 1.0)

    def test_hard_mode_union_preserves_existing_scores(self):
        class_models = {}
        for class_id, center in enumerate(([0.0, 0.0], [3.0, 0.0])):
            expert = Expert(alpha=2.0)
            expert.add_ellipsoid(EllipsoidExpert(
                center=np.asarray(center), radii=np.asarray([1.0, 1.0]),
            ))
            class_models[class_id] = [expert]
        baseline = FittedModel(
            ModelFingerprint(
                task_name="fusion_test",
                input_spec=InputSpec("passthrough", dim=2),
                output_spec=OutputSpec("sdf_scores", (0, 1)),
            ),
            class_models,
            {0: 1.0, 1: 1.0},
        )
        points = np.asarray([[0.0, 0.0], [0.5, 0.0]])
        baseline_scores = baseline.sdf_scores(points)
        candidate = copy.deepcopy(baseline)
        distant = Expert(alpha=2.0)
        distant.add_ellipsoid(EllipsoidExpert(
            center=np.asarray([5.0, 5.0]), radii=np.asarray([0.5, 0.5]),
        ))
        candidate.class_fusion_modes[0] = "hard_min"
        candidate.class_models[0].append(distant)

        candidate_scores = candidate.sdf_scores(points)

        np.testing.assert_allclose(candidate_scores[:, 0], baseline_scores[:, 0])

    def setUp(self):
        self.policy = AdaptationGatePolicy(
            minimum_proposal_gain=0.1,
            maximum_replay_accuracy_drop=0.02,
            maximum_ood_recall_drop=0.03,
        )
        self.update = AdaptationCandidateEvidence(
            action=AdaptationAction.UPDATE_EXISTING,
            target_class_id=1,
            proposal_gain=0.4,
            replay_accuracy_before=0.9,
            replay_accuracy_after=0.89,
            ood_unknown_recall_before=0.8,
            ood_unknown_recall_after=0.79,
            transaction_validated=True,
        )
        self.create = AdaptationCandidateEvidence(
            action=AdaptationAction.CREATE_NEW,
            proposal_gain=0.5,
            replay_accuracy_before=0.9,
            replay_accuracy_after=0.9,
            ood_unknown_recall_before=0.8,
            ood_unknown_recall_after=0.8,
            transaction_validated=True,
        )

    def test_missing_confirmation_always_quarantines(self):
        decision = select_adaptation_action(
            (self.update, self.create), confirmation=None, policy=self.policy,
        )

        self.assertEqual(decision.action, AdaptationAction.QUARANTINE)
        self.assertEqual(decision.failed_gates, ("confirmation_required",))

    def test_confirmation_selects_only_the_matching_passing_action(self):
        decision = select_adaptation_action(
            (self.update, self.create),
            confirmation=ConfirmationKind.EXISTING_CLASS,
            policy=self.policy,
        )

        self.assertEqual(decision.action, AdaptationAction.UPDATE_EXISTING)
        self.assertEqual(decision.target_class_id, 1)
        self.assertEqual(decision.failed_gates, ())

    def test_confirmation_cannot_override_safety_failures(self):
        unsafe = AdaptationCandidateEvidence(
            action=AdaptationAction.CREATE_NEW,
            proposal_gain=0.05,
            replay_accuracy_before=0.9,
            replay_accuracy_after=0.8,
            ood_unknown_recall_before=0.8,
            ood_unknown_recall_after=0.7,
            transaction_validated=False,
        )

        decision = select_adaptation_action(
            (unsafe,),
            confirmation=ConfirmationKind.NEW_CLASS,
            policy=self.policy,
        )

        self.assertEqual(decision.action, AdaptationAction.QUARANTINE)
        self.assertEqual(
            set(decision.failed_gates),
            {
                "insufficient_proposal_gain",
                "replay_regression",
                "ood_regression",
                "transaction_validation_failed",
            },
        )

    def test_action_study_is_deterministic_and_never_publishes_mutations(self):
        first = run_adaptation_action_study(seed=41)
        second = run_adaptation_action_study(seed=41)

        self.assertEqual(first, second)
        self.assertFalse(first["protocol"]["automatic_mutation_enabled"])
        self.assertTrue(first["protocol"]["candidate_models_isolated"])
        self.assertTrue(all(
            not proposal["mutation_published"]
            for proposal in first["proposals"]
        ))

    def test_geode_transaction_study_is_deterministic_and_never_publishes(self):
        first = run_geode_transaction_study(seed=61)
        second = run_geode_transaction_study(seed=61)

        self.assertEqual(first, second)
        self.assertFalse(first["protocol"]["automatic_mutation_enabled"])
        self.assertTrue(all(
            not outcome["mutation_published"] for outcome in first["outcomes"]
        ))

    def test_reviewed_nearby_class_uses_geode_dry_run_without_publishing(self):
        result = run_geode_transaction_study(
            seed=63,
            minimum_known_separation=2.5,
            review_only=True,
        )
        creates = [
            outcome for outcome in result["outcomes"]
            if outcome["expected_action"] == AdaptationAction.CREATE_NEW.value
        ]

        self.assertTrue(result["protocol"]["oracle_used_only_after_review"])
        self.assertGreaterEqual(len(creates), 1)
        self.assertTrue(all(item["review_id"] for item in creates))
        self.assertTrue(all(item["transaction_validated"] for item in creates))
        self.assertTrue(all(not item["mutation_published"] for item in creates))

    def test_reviewed_update_records_replay_constraint_diagnostics(self):
        result = run_geode_transaction_study(
            seed=63,
            minimum_known_separation=2.5,
            review_only=True,
            replay_constrained_updates=True,
        )
        updates = [
            outcome for outcome in result["outcomes"]
            if outcome["expected_action"] == AdaptationAction.UPDATE_EXISTING.value
        ]

        self.assertGreaterEqual(len(updates), 1)
        self.assertTrue(all(
            item["fit_diagnostics"]["exclusion_violations"] == 0
            for item in updates
        ))
        self.assertTrue(all(not item["mutation_published"] for item in updates))

    def test_calibrated_graph_migration_executes_without_publishing(self):
        result = run_calibrated_graph_migration(seed=71)

        self.assertTrue(result["migration"]["valid"])
        self.assertTrue(result["migration"]["candidate_graph_executed"])
        self.assertEqual(result["migration"]["source_calibrator_width"], 3)
        self.assertEqual(result["migration"]["downstream_input_width"], 3)
        self.assertEqual(result["migration"]["downstream_calibrator_width"], 2)
        self.assertTrue(result["migration"]["live_source_unchanged"])
        self.assertTrue(result["migration"]["live_downstream_unchanged"])
        self.assertFalse(result["protocol"]["migration_published"])

    def test_calibrated_mode_update_executes_without_publishing(self):
        result = run_calibrated_mode_update(seed=81)

        self.assertTrue(result["update"]["transaction_accepted"])
        self.assertEqual(result["update"]["class_fusion_mode"], "hard_min")
        self.assertEqual(result["update"]["validation_issues"], [])
        self.assertTrue(result["update"]["candidate_graph_executed"])
        self.assertTrue(result["update"]["live_graph_unchanged"])
        self.assertEqual(result["update"]["source_calibrator_width"], 2)
        self.assertEqual(result["update"]["downstream_input_width"], 2)
        self.assertFalse(result["protocol"]["mutation_published"])

    def test_real_feature_mode_update_keeps_slices_disjoint_and_unpublished(self):
        result = run_real_feature_mode_update(
            dataset_path="data/tier4/cifar10_features.npz",
            seed=91,
            samples_per_slice=8,
            pca_components=4,
        )

        self.assertTrue(
            result["protocol"]["source_images_disjoint_across_slices"]
        )
        self.assertFalse(result["protocol"]["test_used_for_fitting_or_calibration"])
        self.assertTrue(result["update"]["live_model_unchanged"])
        if not result["update"]["usefulness_eligible"]:
            self.assertFalse(result["update"]["transaction_attempted"])
        self.assertFalse(result["protocol"]["mutation_published"])

    def test_real_feature_ood_episode_freezes_final_unknown_holdout(self):
        result = run_real_feature_ood_episode(
            dataset_path="data/tier4/cifar10_features.npz",
            known_classes=(0, 1),
            proxy_unknown_classes=(6,),
            final_unknown_classes=(8,),
            seed=92,
            samples_per_slice=6,
            pca_components=4,
        )

        self.assertTrue(
            result["protocol"]["source_images_disjoint_across_slices"]
        )
        self.assertEqual(result["protocol"]["transform_fit_classes"], [0, 1])
        self.assertTrue(
            result["protocol"]["proxy_unknown_used_for_score_selection"]
        )
        self.assertFalse(
            result["protocol"]["final_unknown_used_for_score_selection"]
        )
        self.assertEqual(set(result["validation"]), set(result["final_test"]))
        self.assertIn("gaussian_mixture_nll", result["final_test"])
        self.assertIn("class_tail_probability", result["final_test"])
        self.assertFalse(result["protocol"]["mutation_published"])

    def test_pooled_rgb_ood_episode_reports_representation_diagnostics(self):
        result = run_real_feature_ood_episode(
            dataset_path="data/tier4/cifar10_features.npz",
            known_classes=(0, 1),
            proxy_unknown_classes=(6,),
            final_unknown_classes=(8,),
            seed=93,
            samples_per_slice=6,
            pca_components=4,
            representation="pooled_rgb_8x8",
        )

        self.assertEqual(result["protocol"]["representation"], "pooled_rgb_8x8")
        self.assertEqual(
            set(result["representation_diagnostics"]),
            {
                "neighborhood_purity",
                "local_intrinsic_dimension",
                "within_class_radius",
                "minimum_centroid_separation",
                "compactness_ratio",
            },
        )
        self.assertTrue(all(
            np.isfinite(value)
            for value in result["representation_diagnostics"].values()
        ))
        self.assertFalse(result["protocol"]["mutation_published"])

    def test_real_feature_ood_score_payload_is_opt_in_and_aligned(self):
        result = run_real_feature_ood_episode(
            dataset_path="data/tier4/cifar10_features.npz",
            known_classes=(0, 1),
            proxy_unknown_classes=(6,),
            final_unknown_classes=(8,),
            seed=94,
            samples_per_slice=6,
            pca_components=4,
            representation="pooled_rgb_8x8",
            include_score_payload=True,
        )

        payload = result["score_payload"]
        self.assertEqual(len(payload["score_names"]), 10)
        self.assertEqual(len(payload["id_validation"]), 12)
        self.assertEqual(len(payload["proxy_unknown"]), 6)
        self.assertEqual(len(payload["id_test"]), 12)
        self.assertEqual(len(payload["final_unknown"]), 6)
        self.assertEqual(
            len(payload["id_validation_predicted_classes"]),
            len(payload["id_validation"]),
        )
        self.assertEqual(
            len(payload["final_unknown_predicted_classes"]),
            len(payload["final_unknown"]),
        )
        for split in ("id_validation", "proxy_unknown", "id_test", "final_unknown"):
            self.assertEqual(len(payload[f"{split}_embeddings"]), len(payload[split]))
            self.assertEqual(
                len(payload[f"{split}_representation_embeddings"]),
                len(payload[split]),
            )
            self.assertEqual(len(payload[f"{split}_labels"]), len(payload[split]))

    def test_support_calibrators_orient_distant_scores_as_unknown(self):
        rng = np.random.default_rng(95)
        id_scores = rng.normal(0.0, 0.2, size=(80, 4))
        proxy_scores = rng.normal(2.0, 0.2, size=(80, 4))
        id_classes = np.repeat([0, 1], 40)
        proxy_classes = np.repeat([0, 1], 40)

        for model_form in ("global", "class_conditional"):
            calibrator = fit_support_calibrator(
                id_scores,
                proxy_scores,
                id_classes,
                proxy_classes,
                model_form=model_form,
                C=0.1,
                seed=95,
            )
            self.assertGreater(
                np.mean(calibrator.score(proxy_scores, proxy_classes)),
                np.mean(calibrator.score(id_scores, id_classes)),
            )
            if model_form == "class_conditional":
                self.assertEqual(set(calibrator.class_models), {0, 1})


class RejectionBufferTests(unittest.TestCase):
    @staticmethod
    def prediction(*, accepted: bool) -> OpenSetPrediction:
        return OpenSetPrediction(
            label=0 if accepted else UNKNOWN_LABEL,
            accepted=accepted,
            candidate_model_signature="model-v1",
            candidate_class_id=0,
            raw_novelty_score=0.4 if accepted else 1.4,
            calibrated_novelty_score=0.4 if accepted else 1.4,
            threshold=1.0,
            decision_margin=-0.6 if accepted else 0.4,
            support_profile_version="profile-v1",
            reason_code=(
                OpenSetReason.ACCEPTED
                if accepted else OpenSetReason.OUTSIDE_SUPPORT
            ),
        )

    def test_accepts_only_rejections_and_evicts_oldest_record(self):
        buffer = RejectionBuffer(max_records=2, max_embedding_dimensions=3)

        with self.assertRaises(ValueError):
            buffer.append_rejection(
                np.zeros(2), timestamp=0.0, window_id=0,
                prediction=self.prediction(accepted=True),
            )
        for window_id in range(3):
            buffer.append_rejection(
                np.full(2, window_id),
                timestamp=float(window_id),
                window_id=window_id,
                prediction=self.prediction(accepted=False),
                nearest_candidates=(0, 1),
            )

        self.assertEqual([record.record_id for record in buffer.snapshot()], [1, 2])
        self.assertEqual(buffer.windows_present, (1, 2))
        self.assertEqual(buffer.evicted_records, 1)
        self.assertEqual(
            [record.record_id for record in buffer.records_in_windows((2,))],
            [2],
        )

    def test_streaming_memory_fades_promotes_and_keeps_stable_ids(self):
        buffer = RejectionBuffer(max_records=8, max_embedding_dimensions=2)
        memory = StreamingRejectionMemory(StreamingClusterPolicy(
            assignment_radius=0.5,
            fading_rate=1.0,
            minimum_weight=0.2,
            promotion_weight=1.4,
            minimum_windows=2,
            minimum_known_separation=2.0,
            max_clusters=4,
            max_records_per_cluster=8,
        ))
        first = buffer.append_rejection(
            np.array([5.0, 5.0]), timestamp=0.0, window_id=0,
            prediction=self.prediction(accepted=False),
        )
        memory.ingest_window((first,), window_id=0)
        initial = memory.snapshots()[0]
        self.assertEqual(initial.state, "emerging")

        second = buffer.append_rejection(
            np.array([5.1, 5.0]), timestamp=1.0, window_id=1,
            prediction=self.prediction(accepted=False),
        )
        memory.ingest_window((second,), window_id=1)
        established = memory.snapshots()[0]
        self.assertEqual(established.cluster_id, initial.cluster_id)
        self.assertEqual(established.state, "established")
        faded = memory.snapshots(window_id=5, include_faded=True)[0]
        self.assertEqual(faded.cluster_id, initial.cluster_id)
        self.assertEqual(faded.state, "faded")

    def test_streaming_lifecycle_is_review_only_and_geometric(self):
        buffer = RejectionBuffer(max_records=8, max_embedding_dimensions=2)
        memory = StreamingRejectionMemory(StreamingClusterPolicy(
            assignment_radius=0.4,
            fading_rate=0.0,
            minimum_weight=0.5,
            promotion_weight=1.5,
            minimum_windows=2,
            minimum_known_separation=2.0,
            max_clusters=4,
            max_records_per_cluster=8,
        ))
        for window_id, embeddings in enumerate((([0.8, 0.0], [5.0, 5.0]), ([0.9, 0.0], [5.1, 5.0]))):
            records = tuple(buffer.append_rejection(
                np.asarray(embedding), timestamp=float(window_id),
                window_id=window_id, prediction=self.prediction(accepted=False),
            ) for embedding in embeddings)
            memory.ingest_window(records, window_id=window_id)

        hypotheses = memory.review_hypotheses(np.array([[0.0, 0.0]]))
        self.assertEqual(
            {hypothesis.relation for hypothesis in hypotheses},
            {"known_extension", "emerging_novel"},
        )
        self.assertTrue(all(hypothesis.review_only for hypothesis in hypotheses))
        self.assertTrue(all(hypothesis.review_id.startswith("review-") for hypothesis in hypotheses))

    def test_stream_fixture_is_deterministic_and_keeps_oracle_separate(self):
        first = generate_class_incremental_stream(seed=17)
        second = generate_class_incremental_stream(seed=17)

        np.testing.assert_array_equal(
            first.observable.embeddings, second.observable.embeddings,
        )
        self.assertFalse(hasattr(first.observable, "class_ids"))
        self.assertFalse(hasattr(first.observable, "event_types"))
        self.assertEqual(set(first.oracle.event_types), set(STREAM_EVENT_TYPES))
        for event_type in STREAM_EVENT_TYPES:
            event_windows = first.observable.window_ids[
                np.asarray(first.oracle.event_types) == event_type
            ]
            self.assertGreaterEqual(len(np.unique(event_windows)), 2)

    def test_cluster_proposal_requires_persistence_and_emits_stable_id(self):
        buffer = RejectionBuffer(max_records=8, max_embedding_dimensions=2)
        for index, embedding in enumerate(([5.0, 5.0], [5.1, 5.0], [4.9, 5.0])):
            buffer.append_rejection(
                np.asarray(embedding),
                timestamp=float(index),
                window_id=min(index, 1),
                prediction=self.prediction(accepted=False),
            )
        policy = ClusterProposalPolicy(
            minimum_support=3,
            minimum_windows=2,
            maximum_rms_radius=0.2,
            minimum_known_separation=2.0,
        )
        records = buffer.snapshot()

        one_window = evaluate_cluster_proposal(
            records[1:], np.asarray([[0.0, 0.0]]), policy,
        )
        first = evaluate_cluster_proposal(
            records, np.asarray([[0.0, 0.0]]), policy,
        )
        second = evaluate_cluster_proposal(
            records, np.asarray([[0.0, 0.0]]), policy,
        )

        self.assertFalse(one_window.eligible)
        self.assertIn("insufficient_persistence", one_window.failed_criteria)
        self.assertTrue(first.eligible)
        self.assertEqual(first.temporary_unknown_id, second.temporary_unknown_id)

    def test_separation_only_failure_emits_stable_review_not_proposal(self):
        buffer = RejectionBuffer(max_records=8, max_embedding_dimensions=2)
        for index, embedding in enumerate(([1.1, 0.0], [1.2, 0.0], [1.3, 0.0])):
            buffer.append_rejection(
                np.asarray(embedding),
                timestamp=float(index),
                window_id=min(index, 1),
                prediction=self.prediction(accepted=False),
            )
        policy = ClusterProposalPolicy(
            minimum_support=3,
            minimum_windows=2,
            maximum_rms_radius=0.2,
            minimum_known_separation=2.5,
        )

        first = evaluate_cluster_proposal(
            buffer.snapshot(), np.asarray([[0.0, 0.0]]), policy,
        )
        second = evaluate_cluster_proposal(
            buffer.snapshot(), np.asarray([[0.0, 0.0]]), policy,
        )

        self.assertFalse(first.eligible)
        self.assertIsNone(first.temporary_unknown_id)
        self.assertTrue(first.review_required)
        self.assertEqual(first.review_id, second.review_id)
        self.assertEqual(first.failed_criteria, ("insufficient_separation",))

    def test_review_only_policy_never_emits_actionable_unknown_id(self):
        buffer = RejectionBuffer(max_records=8, max_embedding_dimensions=2)
        for index, embedding in enumerate(([5.0, 5.0], [5.1, 5.0], [4.9, 5.0])):
            buffer.append_rejection(
                np.asarray(embedding),
                timestamp=float(index),
                window_id=min(index, 1),
                prediction=self.prediction(accepted=False),
            )
        policy = ClusterProposalPolicy(
            minimum_support=3,
            minimum_windows=2,
            maximum_rms_radius=0.2,
            minimum_known_separation=0.0,
            review_only=True,
        )

        first = evaluate_cluster_proposal(
            buffer.snapshot(), np.asarray([[0.0, 0.0]]), policy,
        )
        second = evaluate_cluster_proposal(
            buffer.snapshot(), np.asarray([[0.0, 0.0]]), policy,
        )

        self.assertFalse(first.eligible)
        self.assertIsNone(first.temporary_unknown_id)
        self.assertTrue(first.review_required)
        self.assertEqual(first.review_id, second.review_id)
        self.assertEqual(first.failed_criteria, ())

    def test_hdbscan_recovers_unequal_density_unlabeled_groups(self):
        buffer = RejectionBuffer(max_records=12, max_embedding_dimensions=2)
        embeddings = (
            [0.00, 0.00], [0.03, 0.00], [0.00, 0.03], [0.03, 0.03],
            [4.0, 4.0], [4.3, 4.0], [4.0, 4.3], [4.3, 4.3],
        )
        for index, embedding in enumerate(embeddings):
            buffer.append_rejection(
                np.asarray(embedding),
                timestamp=float(index),
                window_id=index % 2,
                prediction=self.prediction(accepted=False),
            )

        clusters = hdbscan_rejections(
            buffer.snapshot(), minimum_cluster_size=3, minimum_samples=2,
        )

        self.assertEqual(sorted(map(len, clusters)), [4, 4])

    def test_finch_recovers_groups_without_radius_or_group_count(self):
        buffer = RejectionBuffer(max_records=12, max_embedding_dimensions=2)
        embeddings = (
            [0.0, 0.0], [0.1, 0.0], [0.0, 0.1],
            [4.0, 4.0], [4.1, 4.0], [4.0, 4.1],
            [-4.0, 4.0], [-4.1, 4.0], [-4.0, 4.1],
        )
        for index, embedding in enumerate(embeddings):
            buffer.append_rejection(
                np.asarray(embedding),
                timestamp=float(index),
                window_id=index % 2,
                prediction=self.prediction(accepted=False),
            )

        clusters = finch_rejections(buffer.snapshot(), hierarchy_level=0)
        coarser_clusters = finch_rejections(buffer.snapshot(), hierarchy_level=1)

        self.assertEqual(sorted(map(len, clusters)), [3, 3, 3])
        self.assertEqual(sum(map(len, coarser_clusters)), len(embeddings))
        self.assertLess(len(coarser_clusters), len(clusters))

    def test_estimated_kmeans_recovers_group_count_without_labels(self):
        buffer = RejectionBuffer(max_records=12, max_embedding_dimensions=2)
        embeddings = (
            [-4.0, -4.0], [-4.1, -4.0], [-4.0, -4.1],
            [0.0, 4.0], [0.1, 4.0], [0.0, 4.1],
            [4.0, -4.0], [4.1, -4.0], [4.0, -4.1],
        )
        for index, embedding in enumerate(embeddings):
            buffer.append_rejection(
                np.asarray(embedding),
                timestamp=float(index),
                window_id=index % 2,
                prediction=self.prediction(accepted=False),
            )

        groups = estimated_kmeans_rejections(
            buffer.snapshot(), maximum_cluster_count=6,
        )

        self.assertEqual(sorted(map(len, groups)), [3, 3, 3])

    def test_pairwise_feedback_metric_suppresses_within_group_nuisance(self):
        embeddings = {
            0: np.asarray([0.0, 0.0]),
            1: np.asarray([0.1, 4.0]),
            2: np.asarray([5.0, 0.0]),
            3: np.asarray([5.1, 4.0]),
        }
        constraints = build_pairwise_constraints(
            np.arange(4), np.asarray([0, 0, 1, 1]),
        )

        metric = fit_diagonal_constraint_metric(embeddings, constraints)

        self.assertEqual(metric.must_link_count, 2)
        self.assertEqual(metric.cannot_link_count, 4)
        self.assertGreater(metric.feature_weights[0], metric.feature_weights[1])

    def test_pairwise_feedback_splits_and_merges_partition(self):
        buffer = RejectionBuffer(max_records=4, max_embedding_dimensions=2)
        records = []
        for index, embedding in enumerate((
            [0.0, 0.0], [5.0, 5.0], [0.1, 0.0], [5.1, 5.0],
        )):
            records.append(buffer.append_rejection(
                np.asarray(embedding),
                timestamp=0.0,
                window_id=0,
                prediction=self.prediction(accepted=False),
                source_sample_id=index,
            ))
        constraints = (
            PairwiseConstraint(0, 2, "must_link"),
            PairwiseConstraint(1, 3, "must_link"),
            PairwiseConstraint(0, 1, "cannot_link"),
        )

        refined = refine_rejection_partition(
            ((records[0], records[1]), (records[2],), (records[3],)),
            constraints,
        )

        self.assertEqual(
            sorted(sorted(record.source_sample_id for record in group)
                   for group in refined),
            [[0, 2], [1, 3]],
        )

    def test_active_constraint_queries_do_not_inspect_answers(self):
        embeddings = {
            index: np.asarray([float(index), 0.0]) for index in range(4)
        }
        constraints = (
            PairwiseConstraint(0, 1, "must_link"),
            PairwiseConstraint(0, 2, "cannot_link"),
            PairwiseConstraint(1, 3, "cannot_link"),
        )
        flipped = tuple(PairwiseConstraint(
            constraint.left_record_id,
            constraint.right_record_id,
            "cannot_link" if constraint.relation == "must_link" else "must_link",
        ) for constraint in constraints)

        def selected_pairs(candidates):
            return [
                (constraint.left_record_id, constraint.right_record_id)
                for constraint in select_constraint_queries(
                    candidates,
                    embeddings,
                    ((0, 1), (2, 3)),
                    budget=2,
                    strategy="active",
                )
            ]

        self.assertEqual(selected_pairs(constraints), selected_pairs(flipped))

    def test_constraint_consistency_detects_direct_and_transitive_conflicts(self):
        report = validate_pairwise_constraints((
            PairwiseConstraint(0, 1, "must_link"),
            PairwiseConstraint(1, 0, "cannot_link"),
            PairwiseConstraint(2, 3, "must_link"),
            PairwiseConstraint(3, 4, "must_link"),
            PairwiseConstraint(2, 4, "cannot_link"),
        ))

        self.assertFalse(report.is_consistent)
        self.assertEqual(report.direct_conflict_count, 1)
        self.assertEqual(report.transitive_conflict_count, 1)

    def test_pairwise_confirmation_accepts_agreement_and_abstains_on_disagreement(self):
        first = (
            PairwiseConstraint(0, 1, "must_link"),
            PairwiseConstraint(2, 3, "cannot_link"),
        )
        second = (
            PairwiseConstraint(1, 0, "cannot_link"),
            PairwiseConstraint(3, 2, "cannot_link"),
        )

        confirmation = confirm_pairwise_constraints(first, second)

        self.assertEqual(confirmation.accepted, (first[1],))
        self.assertEqual(confirmation.disagreement_count, 1)

    def test_bound_routing_matches_exhaustive_predictions_and_scores(self):
        models = {
            class_id: [Expert(alpha=2.0)] for class_id in range(3)
        }
        for class_id, experts in models.items():
            experts[0].add_ellipsoid(EllipsoidExpert(
                center=np.asarray([class_id * 5.0, 0.0]),
                radii=np.ones(2),
            ))
        points = np.asarray([[0.1, 0.0], [5.2, 0.0], [9.8, 0.0]])
        exhaustive_scores = np.column_stack([
            InferenceEngine(models[class_id], alpha=2.0).get_fused_sdf(points)
            for class_id in sorted(models)
        ])

        result = exact_bound_routing(models, points)
        batched = batched_exact_bound_routing(models, points)
        class_major = class_major_exact_bound_routing(models, points)
        certified = CertifiedTopKRouter(models, candidate_budget=1).route(points)

        self.assertTrue(np.array_equal(
            result.predictions,
            np.argmin(exhaustive_scores, axis=1),
        ))
        self.assertTrue(np.allclose(
            result.winning_scores,
            np.min(exhaustive_scores, axis=1),
        ))
        self.assertTrue(np.all(result.candidate_counts < len(models)))
        self.assertTrue(np.array_equal(batched.predictions, result.predictions))
        self.assertTrue(np.allclose(batched.winning_scores, result.winning_scores))
        self.assertTrue(np.array_equal(
            batched.candidate_counts, result.candidate_counts,
        ))
        self.assertTrue(np.array_equal(
            class_major.predictions, result.predictions,
        ))
        self.assertTrue(np.allclose(
            class_major.winning_scores, result.winning_scores,
        ))
        self.assertTrue(np.array_equal(
            certified.predictions, result.predictions,
        ))
        self.assertTrue(np.allclose(
            certified.winning_scores, result.winning_scores,
        ))

    def test_primitive_compression_respects_budget_and_agreement_gate(self):
        expert = Expert(alpha=2.0)
        for offset in (0.0, 0.01, 4.0):
            expert.add_ellipsoid(EllipsoidExpert(
                center=np.asarray([offset, 0.0]),
                radii=np.ones(2),
            ))
        other = Expert(alpha=2.0)
        other.add_ellipsoid(EllipsoidExpert(
            center=np.asarray([10.0, 0.0]),
            radii=np.ones(2),
        ))
        points = np.asarray([[0.0, 0.0], [0.2, 0.0], [10.0, 0.0]])

        result = compress_primitive_budget(
            {0: [expert], 1: [other]},
            points,
            primitive_budget_per_class=2,
            maximum_score_drift=0.25,
            confirmation_points=points + 0.05,
        )

        self.assertEqual(result.final_primitive_count, 3)
        self.assertEqual(result.prediction_agreement, 1.0)
        self.assertLessEqual(result.maximum_score_drift, 0.25)
        self.assertEqual(result.confirmation_prediction_agreement, 1.0)
        self.assertEqual(len(expert.ellipsoids), 3)

    def test_feedback_metric_ignores_final_labels(self):
        payload = {
            "score_names": ["maximum_probability"],
            "id_validation": [[0.1], [0.1]],
            "proxy_unknown": [[0.9], [0.9], [0.9], [0.9]],
            "id_validation_representation_embeddings": [[1.0, 0.0]] * 2,
            "proxy_unknown_representation_embeddings": [
                [0.0, 1.0], [0.1, 1.0], [1.0, 0.0], [1.0, 0.1],
            ],
            "id_validation_labels": [0, 0],
            "proxy_unknown_labels": [6, 6, 7, 7],
            "id_validation_predicted_classes": [0, 0],
            "proxy_unknown_predicted_classes": [0, 0, 0, 0],
            "final_unknown_labels": [8, 8, 9, 9],
        }
        review = {"reviews": [{"source_sample_ids": [2, 3, 4, 5]}]}

        first = _fit_review_feedback_metric(
            payload,
            review,
            known_split="id_validation",
            unknown_split="proxy_unknown",
        )
        payload["final_unknown_labels"] = [9, 8, 9, 8]
        second = _fit_review_feedback_metric(
            payload,
            review,
            known_split="id_validation",
            unknown_split="proxy_unknown",
        )

        self.assertEqual(first, second)

    def test_event_review_is_label_blind_and_recovers_recurring_events(self):
        known_embeddings = np.asarray([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]])
        unknown_embeddings = np.asarray([
            [5.0, 5.0], [5.1, 5.0], [5.0, 5.1], [5.1, 5.1],
            [-5.0, -5.0], [-5.1, -5.0], [-5.0, -5.1], [-5.1, -5.1],
        ])
        known_representation = np.asarray([
            [1.0, 0.0, 0.0], [1.0, 0.01, 0.0],
            [1.0, 0.0, 0.01], [1.0, 0.01, 0.01],
        ])
        unknown_representation = np.asarray([
            [0.0, 1.0, 0.0], [0.01, 1.0, 0.0],
            [0.0, 1.0, 0.01], [0.01, 1.0, 0.01],
            [0.0, 0.0, 1.0], [0.01, 0.0, 1.0],
            [0.0, 0.01, 1.0], [0.01, 0.01, 1.0],
        ])
        payload = {
            "score_names": ["maximum_probability"],
            "id_validation": [[0.1]] * len(known_embeddings),
            "proxy_unknown": [[0.9]] * len(unknown_embeddings),
            "id_validation_embeddings": known_embeddings.tolist(),
            "proxy_unknown_embeddings": unknown_embeddings.tolist(),
            "id_validation_representation_embeddings": known_representation.tolist(),
            "proxy_unknown_representation_embeddings": unknown_representation.tolist(),
            "id_validation_labels": [0] * len(known_embeddings),
            "proxy_unknown_labels": [8] * 4 + [9] * 4,
            "id_validation_predicted_classes": [0] * len(known_embeddings),
            "proxy_unknown_predicted_classes": [0] * len(unknown_embeddings),
        }

        result = evaluate_event_review_payload(
            payload,
            known_split="id_validation",
            unknown_split="proxy_unknown",
            flag_fraction=1.0,
            dbscan_epsilon=0.3,
            maximum_rms_radius=0.3,
        )
        hierarchical = evaluate_event_review_payload(
            payload,
            known_split="id_validation",
            unknown_split="proxy_unknown",
            flag_fraction=1.0,
            clustering_method="hdbscan",
            embedding_space="representation_l2",
            hdbscan_minimum_cluster_size=3,
            hdbscan_minimum_samples=2,
            maximum_rms_radius=0.3,
        )
        finch = evaluate_event_review_payload(
            payload,
            known_split="id_validation",
            unknown_split="proxy_unknown",
            flag_fraction=1.0,
            clustering_method="finch",
            embedding_space="representation_l2",
            finch_hierarchy_level=0,
            maximum_rms_radius=0.3,
        )
        streaming = evaluate_event_review_payload(
            payload,
            known_split="id_validation",
            unknown_split="proxy_unknown",
            flag_fraction=1.0,
            windows=2,
            clustering_method="streaming",
            streaming_assignment_radius=0.3,
            streaming_fading_rate=0.0,
            streaming_promotion_weight=2.0,
            streaming_minimum_known_separation=0.3,
            minimum_support=2,
            minimum_windows=2,
            maximum_rms_radius=0.3,
        )
        estimated = evaluate_event_review_payload(
            payload,
            known_split="id_validation",
            unknown_split="proxy_unknown",
            flag_fraction=1.0,
            clustering_method="estimated_kmeans",
            embedding_space="representation_l2",
            estimated_kmeans_maximum_cluster_count=5,
            maximum_rms_radius=0.3,
        )
        joint_representation = evaluate_event_review_payload(
            payload,
            known_split="id_validation",
            unknown_split="proxy_unknown",
            flag_fraction=1.0,
            clustering_method="estimated_kmeans",
            embedding_space="joint_pca_l2",
            joint_pca_components=2,
            estimated_kmeans_maximum_cluster_count=5,
            maximum_rms_radius=0.3,
        )

        self.assertFalse(result["protocol"]["oracle_used_for_flagging_or_grouping"])
        self.assertEqual(result["protocol"]["temporary_unknown_ids_emitted"], 0)
        self.assertEqual(result["metrics"]["event_recall"], 1.0)
        self.assertEqual(result["metrics"]["accumulated_event_recall"], 1.0)
        self.assertEqual(result["metrics"]["accumulated_group_count"], 3)
        self.assertEqual(result["metrics"]["accumulated_unknown_group_count"], 2)
        self.assertEqual(result["metrics"]["distinct_group_recall"], 1.0)
        self.assertEqual(result["metrics"]["recovered_unknown_group_count"], 2)
        self.assertEqual(result["metrics"]["unknown_group_ari"], 1.0)
        self.assertGreater(result["metrics"]["useful_review_precision"], 0.0)
        self.assertEqual(hierarchical["metrics"]["distinct_group_recall"], 1.0)
        self.assertEqual(hierarchical["metrics"]["unknown_group_ari"], 1.0)
        self.assertEqual(finch["metrics"]["distinct_group_recall"], 1.0)
        self.assertEqual(finch["metrics"]["unknown_group_ari"], 1.0)
        self.assertTrue(streaming["protocol"]["streaming_memory"])
        self.assertEqual(streaming["metrics"]["distinct_group_recall"], 1.0)
        self.assertEqual(streaming["metrics"]["unknown_group_ari"], 1.0)
        self.assertEqual(streaming["metrics"]["lifecycle_relation_accuracy"], 1.0)
        self.assertEqual(estimated["metrics"]["distinct_group_recall"], 1.0)
        self.assertEqual(estimated["metrics"]["unknown_group_ari"], 1.0)
        self.assertTrue(
            joint_representation["protocol"]["transductive_representation"]
        )
        self.assertEqual(
            joint_representation["protocol"]["feature_model_version"],
            "joint-pca-l2-2-v1",
        )
        self.assertFalse(joint_representation["protocol"]["mutation_published"])

    def test_streaming_transfer_freezes_proxy_selected_parameters(self):
        calls = []

        def fake_episode(**_kwargs):
            return {"score_payload": {}}

        def fake_evaluation(_payload, **kwargs):
            calls.append(kwargs)
            selected = (
                kwargs["clustering_method"] == "streaming"
                and kwargs["streaming_assignment_radius"] == 0.75
                and kwargs["streaming_fading_rate"] == 0.1
            )
            score = 1.0 if selected else 0.0
            return {"metrics": {
                "event_recall": score,
                "distinct_group_recall": score,
                "unknown_group_ari": score,
                "useful_review_precision": score,
                "reviews_per_1000": 1.0,
                "partition_group_count": 2,
            }}

        with patch(
            "experiments.tier4.eval_real_feature_event_review."
            "run_real_feature_ood_episode",
            side_effect=fake_episode,
        ), patch(
            "experiments.tier4.eval_real_feature_event_review."
            "evaluate_event_review_payload",
            side_effect=fake_evaluation,
        ):
            result = run_event_review_transfer(
                dataset_path="unused.npz",
                episodes=[{
                    "known_classes": [0, 1],
                    "proxy_unknown_classes": [6, 7],
                    "final_unknown_classes": [8, 9],
                }],
                seeds=[101],
                flag_fractions=[0.3],
                clustering_methods=["hdbscan", "streaming"],
                hdbscan_minimum_cluster_sizes=[3],
                streaming_assignment_radii=[0.5, 0.75],
                streaming_fading_rates=[0.0, 0.1],
            )

        final_call = calls[-1]
        self.assertEqual(final_call["unknown_split"], "final_unknown")
        self.assertEqual(result["selected_clustering_method"], "streaming")
        self.assertEqual(result["selected_streaming_assignment_radius"], 0.75)
        self.assertEqual(result["selected_streaming_fading_rate"], 0.1)
        self.assertEqual(final_call["streaming_assignment_radius"], 0.75)
        self.assertEqual(final_call["streaming_fading_rate"], 0.1)
        self.assertFalse(result["protocol"]["final_labels_used_for_selection"])
        self.assertTrue(result["protocol"]["review_only"])
        self.assertFalse(result["protocol"]["mutation_published"])

    def test_streaming_clustering_comparison_is_deterministic_and_label_blind(self):
        first = run_streaming_discovery_study(seed=31)
        second = run_streaming_discovery_study(seed=31)

        self.assertEqual(first, second)
        self.assertFalse(first["protocol"]["oracle_used_for_discovery"])
        self.assertEqual(first["strategies"]["no_clustering"]["proposal_count"], 0)
        self.assertGreaterEqual(
            first["strategies"]["dbscan"]["discovery_recall"], 0.5,
        )

    def test_streaming_policy_selection_keeps_holdout_seeds_disjoint(self):
        result = run_streaming_policy_transfer_study(
            development_seeds=(11, 12),
            holdout_seeds=(21, 22),
            separation_candidates=(1.5, 2.0),
        )

        self.assertFalse(result["protocol"]["holdout_used_for_selection"])
        self.assertEqual(set(result["protocol"]["development_seeds"]), {11, 12})
        self.assertEqual(set(result["protocol"]["holdout_seeds"]), {21, 22})
        with self.assertRaises(ValueError):
            run_streaming_policy_transfer_study(
                development_seeds=(11, 12),
                holdout_seeds=(12, 13),
                separation_candidates=(1.5,),
            )

    def test_stream_family_study_keeps_policy_frozen(self):
        result = run_frozen_stream_family_study(
            holdout_seeds=(61, 62),
            stream_families=("heavy_tailed", "intermittent_unseen"),
            frozen_separation=2.5,
            rejection_threshold=1.0,
            dbscan_epsilon=0.9,
            dbscan_minimum_samples=4,
            incremental_assignment_radius=0.9,
        )

        self.assertEqual(result["protocol"]["frozen_separation"], 2.5)
        self.assertTrue(
            result["protocol"]["parameters_selected_before_family_observation"]
        )
        self.assertEqual(
            set(result["families"]), {"heavy_tailed", "intermittent_unseen"}
        )

    def test_nearby_unseen_review_requires_confirmation_and_never_publishes(self):
        result = run_ambiguity_resolution(
            seed=63, stream_family="baseline",
        )

        new_class_reviews = [
            item for item in result["resolutions"]
            if item["oracle_confirmation"] == ConfirmationKind.NEW_CLASS.value
        ]
        self.assertTrue(result["protocol"]["oracle_used_only_after_review"])
        self.assertGreaterEqual(len(new_class_reviews), 1)
        self.assertTrue(all(
            item["review_id"].startswith("review-")
            for item in new_class_reviews
        ))
        self.assertTrue(all(
            not item["mutation_published"] for item in result["resolutions"]
        ))


class LeaveClassOutProtocolTests(unittest.TestCase):
    def test_final_unknown_classes_remain_observational(self):
        from experiments.tier5.eval_corruption_robustness import (
            generate_multiclass_problem,
        )

        problem = generate_multiclass_problem(
            seed=19,
            dimensions=2,
            class_count=5,
            geometry_per_class=18,
            calibration_per_class=10,
            test_per_class=12,
            center_radius=3.0,
            mode_offset=0.5,
            noise_scale=0.5,
        )
        result = run_leave_class_out_episode(
            problem,
            known_classes=(0, 1),
            proxy_unknown_classes=(2,),
            final_unknown_classes=(3, 4),
            seed=19,
            max_iterations=5,
        )
        protocol = result["protocol"]
        self.assertEqual(protocol["geometry_classes"], [0, 1])
        self.assertEqual(protocol["readout_classes"], [0, 1])
        self.assertTrue(protocol["proxy_unknown_used_for_selection"])
        self.assertFalse(protocol["final_unknown_used_for_selection"])
        self.assertFalse(protocol["final_test_used_for_selection"])
        self.assertEqual(
            set(result["validation"]),
            {
                "minimum_sdf",
                "minimum_metric_sdf",
                "sdf_energy",
                "maximum_probability",
                "mahalanobis",
                "gmm_nll",
                "knn_distance",
                "minimum_sdf_per_class",
                "minimum_metric_sdf_per_class",
                "sdf_energy_per_class",
                "maximum_probability_per_class",
                "mahalanobis_per_class",
                "gmm_nll_per_class",
                "knn_distance_per_class",
                "minimum_sdf_coverage90",
                "minimum_metric_sdf_coverage90",
                "sdf_energy_coverage90",
                "maximum_probability_coverage90",
                "mahalanobis_coverage90",
                "gmm_nll_coverage90",
                "knn_distance_coverage90",
                "minimum_sdf_coverage90_per_class",
                "minimum_metric_sdf_coverage90_per_class",
                "sdf_energy_coverage90_per_class",
                "maximum_probability_coverage90_per_class",
                "mahalanobis_coverage90_per_class",
                "gmm_nll_coverage90_per_class",
                "knn_distance_coverage90_per_class",
            },
        )
        self.assertIn(result["selection"]["score"], result["final_test"])

    def test_multi_episode_study_keeps_proxy_and_final_pools_disjoint(self):
        from experiments.tier5.eval_corruption_robustness import (
            generate_multiclass_problem,
        )

        problem = generate_multiclass_problem(
            seed=23,
            dimensions=2,
            class_count=6,
            geometry_per_class=12,
            calibration_per_class=8,
            test_per_class=8,
            center_radius=3.0,
            mode_offset=0.5,
            noise_scale=0.5,
        )
        episodes = [
            {
                "known_classes": [0, 1],
                "proxy_unknown_classes": [2],
                "final_unknown_classes": [4],
            },
            {
                "known_classes": [0, 1],
                "proxy_unknown_classes": [3],
                "final_unknown_classes": [5],
            },
        ]
        result = run_leave_class_out_study(
            problem, episodes, seed=23, max_iterations=3,
        )
        self.assertEqual(result["protocol"]["proxy_unknown_pool"], [2, 3])
        self.assertEqual(result["protocol"]["final_unknown_pool"], [4, 5])
        self.assertTrue(result["protocol"]["proxy_and_final_pools_disjoint"])
        self.assertFalse(result["protocol"]["final_unknown_used_for_selection"])
        self.assertEqual(len(result["episodes"]), 2)
        self.assertEqual(set(result["summary"]), set(result["episodes"][0]["final_test"]))

        invalid = [episodes[0], {
            "known_classes": [0, 1],
            "proxy_unknown_classes": [4],
            "final_unknown_classes": [5],
        }]
        with self.assertRaisesRegex(ValueError, "globally disjoint"):
            run_leave_class_out_study(problem, invalid, seed=23, max_iterations=3)


class ExperimentManifestTests(unittest.TestCase):
    def test_same_config_has_same_id(self):
        first = {"alpha": np.float64(2.0), "seed": 4, "dims": [8, 4]}
        second = {"dims": [8, 4], "seed": 4, "alpha": 2.0}
        self.assertEqual(experiment_id(first), experiment_id(second))

    def test_split_change_changes_fingerprint(self):
        first = np.array([0, 1, 2, 3], dtype=np.int64)
        second = np.array([0, 1, 2, 4], dtype=np.int64)
        self.assertNotEqual(array_fingerprint(first), array_fingerprint(second))

    def test_manifest_jsonl_round_trip_preserves_numeric_types(self):
        manifest = {
            "experiment_id": "toy",
            "seed": np.int64(7),
            "metrics": {"accuracy": np.float64(0.25), "counts": [1, 2]},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.jsonl"
            append_manifest(path, manifest)
            loaded = read_manifests(path)[0]
        self.assertEqual(loaded["seed"], 7)
        self.assertIsInstance(loaded["seed"], int)
        self.assertEqual(loaded["metrics"]["accuracy"], 0.25)
        self.assertIsInstance(loaded["metrics"]["accuracy"], float)


class ClassificationMetricTests(unittest.TestCase):
    def setUp(self):
        self.classes = np.array([0, 1, 2])
        self.truth = np.array([0, 0, 1, 2])
        self.probabilities = np.array([
            [0.8, 0.1, 0.1],
            [0.4, 0.5, 0.1],
            [0.1, 0.7, 0.2],
            [0.1, 0.2, 0.7],
        ])
        self.predictions = self.classes[self.probabilities.argmax(axis=1)]

    def test_point_metrics_match_sklearn(self):
        self.assertEqual(accuracy(self.truth, self.predictions), 0.75)
        self.assertAlmostEqual(
            balanced_accuracy(self.truth, self.predictions),
            balanced_accuracy_score(self.truth, self.predictions),
        )
        self.assertAlmostEqual(
            negative_log_likelihood(
                self.truth, self.probabilities, self.classes,
            ),
            log_loss(self.truth, self.probabilities, labels=self.classes),
        )

    def test_probability_metrics_match_hand_calculation(self):
        targets = np.eye(3)[self.truth]
        expected_brier = np.mean(np.sum(
            (self.probabilities - targets) ** 2, axis=1,
        ))
        self.assertAlmostEqual(
            multiclass_brier_score(
                self.truth, self.probabilities, self.classes,
            ),
            expected_brier,
        )
        self.assertEqual(
            top_k_accuracy(
                self.truth, self.probabilities, self.classes, k=2,
            ),
            1.0,
        )
        self.assertAlmostEqual(
            expected_calibration_error(
                self.truth, self.probabilities, self.classes, n_bins=2,
            ),
            0.075,
        )

    def test_paired_bootstrap_is_seeded_and_directional(self):
        perfect = self.truth.copy()
        baseline = np.zeros_like(self.truth)
        first = paired_bootstrap_interval(
            self.truth, perfect, baseline, n_resamples=100, seed=5,
        )
        second = paired_bootstrap_interval(
            self.truth, perfect, baseline, n_resamples=100, seed=5,
        )
        self.assertEqual(first, second)
        self.assertGreater(first["difference"], 0.0)


class OODMetricTests(unittest.TestCase):
    def test_perfect_ood_scores_have_perfect_detection_metrics(self):
        metrics = ood_detection_metrics(
            np.array([0.0, 0.1, 0.2]), np.array([0.8, 0.9, 1.0]),
        )
        self.assertEqual(metrics["auroc"], 1.0)
        self.assertEqual(metrics["aupr_in"], 1.0)
        self.assertEqual(metrics["aupr_out"], 1.0)
        self.assertEqual(metrics["fpr95"], 0.0)

    def test_risk_falls_when_low_confidence_error_is_rejected(self):
        curve = risk_coverage_curve(
            np.array([0, 1, 1]),
            np.array([0, 1, 0]),
            np.array([0.9, 0.8, 0.1]),
        )
        self.assertEqual(curve["risk"][1], 0.0)
        self.assertAlmostEqual(curve["risk"][-1], 1.0 / 3.0)

    def test_validation_threshold_is_frozen_for_test_operating_point(self):
        threshold = select_ood_threshold(np.arange(20.0), 0.95)
        point = ood_operating_point(
            np.array([-1.0, 0.5, 2.0]), np.array([0.0, 3.0, 4.0]), threshold,
        )
        self.assertEqual(threshold, 0.0)
        self.assertAlmostEqual(point["false_positive_rate"], 2.0 / 3.0)
        self.assertEqual(point["true_positive_rate"], 1.0)

    def test_known_coverage_threshold_meets_finite_sample_target(self):
        scores = np.asarray([0.0, 1.0, 1.0, 2.0, 3.0])
        threshold = select_ood_threshold_at_known_coverage(scores, 0.8)

        self.assertGreaterEqual(np.mean(scores < threshold), 0.8)

    def test_conformal_threshold_uses_calibration_labels(self):
        classes = np.array([0, 1])
        calibration_y = np.array([0, 1, 0, 1])
        calibration_probabilities = np.array([
            [0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6],
        ])
        threshold = conformal_probability_threshold(
            calibration_y, calibration_probabilities, classes, alpha=0.25,
        )
        prediction_sets = conformal_prediction_sets(
            calibration_probabilities, threshold,
        )
        metrics = conformal_set_metrics(
            calibration_y, prediction_sets, classes,
        )
        self.assertGreaterEqual(metrics["coverage"], 0.75)
        self.assertGreaterEqual(metrics["average_set_size"], 1.0)


class OODScoreTests(unittest.TestCase):
    def test_matrix_scores_use_larger_is_more_ood_convention(self):
        class_sdfs = np.array([[-2.0, 1.0], [2.0, 3.0]])
        probabilities = np.array([[0.95, 0.05], [0.55, 0.45]])
        self.assertLess(minimum_sdf_score(class_sdfs)[0], minimum_sdf_score(class_sdfs)[1])
        self.assertLess(sdf_energy_score(class_sdfs)[0], sdf_energy_score(class_sdfs)[1])
        self.assertLess(
            maximum_probability_score(probabilities)[0],
            maximum_probability_score(probabilities)[1],
        )

    def test_feature_scores_increase_for_distant_shift(self):
        rng = np.random.default_rng(17)
        geometry = rng.normal(0.0, 0.4, size=(120, 4))
        in_distribution = rng.normal(0.0, 0.4, size=(40, 4))
        shifted = rng.normal(5.0, 0.4, size=(40, 4))
        scorers = fit_feature_ood_scorers(
            geometry, gmm_components=2, knn_k=5, seed=17,
        )
        in_scores = scorers.score(in_distribution)
        shifted_scores = scorers.score(shifted)
        for name in ("mahalanobis", "gmm_nll", "knn_distance"):
            self.assertGreater(np.mean(shifted_scores[name]), np.mean(in_scores[name]))

    def test_class_conditional_scores_increase_for_distant_shift(self):
        rng = np.random.default_rng(19)
        labels = np.repeat([0, 1], 80)
        geometry = np.vstack([
            rng.normal(-1.0, 0.25, size=(80, 3)),
            rng.normal(1.0, 0.25, size=(80, 3)),
        ])
        in_distribution = rng.normal(-1.0, 0.25, size=(30, 3))
        shifted = rng.normal(6.0, 0.25, size=(30, 3))
        scorers = fit_class_conditional_ood_scorers(geometry, labels)

        in_scores = scorers.score(in_distribution)
        shifted_scores = scorers.score(shifted)

        self.assertEqual(
            set(in_scores),
            {
                "class_mahalanobis",
                "gaussian_mixture_nll",
                "class_tail_probability",
            },
        )
        for name in in_scores:
            self.assertTrue(np.all(np.isfinite(in_scores[name])))
            self.assertGreater(
                np.mean(shifted_scores[name]), np.mean(in_scores[name]),
            )

    def test_controlled_experiment_keeps_final_ood_test_observational(self):
        result = run_controlled_ood_experiment(
            seed=9,
            distances=(6.0,),
            geometry_per_class=20,
            evaluation_per_class=8,
            max_iterations=5,
        )
        self.assertFalse(result["protocol"]["ood_test_used_for_selection"])
        self.assertEqual(
            set(result["ood_test"]["6.0"]),
            {
                "minimum_raw_sdf", "minimum_metric_sdf", "sdf_energy",
                "maximum_probability",
                "mahalanobis", "gmm_nll", "knn_distance",
            },
        )

    def test_fused_metric_correction_matches_single_ellipsoid(self):
        ellipsoid = EllipsoidExpert(
            center=np.zeros(2), radii=np.array([2.0, 0.5]),
        )
        expert = Expert(alpha=2.0)
        expert.add_ellipsoid(ellipsoid)
        points = np.array([[3.0, 0.2], [0.5, 0.6]])
        actual = InferenceEngine([expert], alpha=2.0).get_metric_corrected_sdf(points)
        np.testing.assert_allclose(actual, ellipsoid.compute_metric_sdf(points))

    def test_numerical_ellipsoid_distance_reference_matches_exact_cases(self):
        sphere = EllipsoidExpert(np.zeros(3), np.full(3, 2.0))
        points = np.array([
            [3.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        ])
        np.testing.assert_allclose(
            numerical_signed_distance(sphere, points),
            np.linalg.norm(points, axis=1) - 2.0,
            atol=1e-7,
        )
        ellipsoid = EllipsoidExpert(np.zeros(2), np.array([2.0, 1.0]))
        np.testing.assert_allclose(
            numerical_signed_distance(
                ellipsoid, np.array([[4.0, 0.0], [0.0, 2.0], [0.0, 0.0]]),
            ),
            [2.0, 1.0, -1.0],
            atol=1e-7,
        )
        closest = numerical_closest_point(ellipsoid, np.array([3.0, 2.0]))
        self.assertAlmostEqual(np.sum((closest / ellipsoid.radii) ** 2), 1.0)

    def test_metric_distance_audit_distinguishes_sphere_and_ellipsoid(self):
        result = evaluate_metric_distance(
            dimensions=(2,), eccentricities=(1.0, 4.0), direction_count=8,
        )
        sphere, ellipsoid = result["records"]
        self.assertLess(sphere["max_absolute_error"], 1e-7)
        self.assertGreater(ellipsoid["max_absolute_error"], 0.1)
        self.assertEqual(ellipsoid["unsafe_closest_ray_steps"], 0)

    def test_real_ood_protocol_keeps_original_test_and_ood_test_isolated(self):
        rng = np.random.default_rng(23)
        labels = np.tile(np.arange(3), 30)
        features = rng.normal(size=(90, 6)) + labels[:, None] * 1.5
        source_indices = np.concatenate([np.arange(75), np.arange(100, 115)])
        ood_features = {
            "far": rng.normal(8.0, 1.0, size=(30, 6)),
            "near": rng.normal(4.0, 1.0, size=(30, 6)),
        }
        result = run_feature_ood_experiment(
            features,
            labels,
            source_indices,
            ood_features,
            seed=23,
            original_train_size=100,
            pca_components=5,
            max_iterations=3,
        )
        self.assertEqual(result["protocol"]["id_test_count"], 15)
        self.assertTrue(all(
            count == result["protocol"]["id_test_count"]
            for count in result["protocol"]["ood_test_count"].values()
        ))
        self.assertTrue(all(
            count == result["protocol"]["id_validation_count"]
            for count in result["protocol"]["ood_validation_count"].values()
        ))
        self.assertFalse(result["protocol"]["ood_test_used_for_selection"])
        self.assertEqual(set(result["ood_detection"]), {"far", "near"})
        self.assertIn("minimum_metric_sdf", result["ood_detection"]["far"])

    def test_ood_cache_sampling_is_seeded_and_without_replacement(self):
        first = deterministic_sample_indices(100, 40, seed=7)
        second = deterministic_sample_indices(100, 40, seed=7)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(np.unique(first)), len(first))


class RobustnessCorruptionTests(unittest.TestCase):
    def setUp(self):
        self.features = np.arange(120, dtype=np.float64).reshape(20, 6)
        self.labels = np.tile(np.arange(4), 5)

    def test_label_corruptions_have_exact_audited_counts(self):
        symmetric, audit = symmetric_label_noise(self.labels, 0.2, seed=3)
        self.assertEqual(np.count_nonzero(symmetric != self.labels), 4)
        self.assertEqual(len(audit["indices"]), 4)
        conditional, audit = class_conditional_label_noise(
            self.labels, 0, 1, 0.4, seed=3,
        )
        self.assertEqual(np.count_nonzero(conditional != self.labels), 2)
        self.assertEqual(len(audit["indices"]), 2)

    def test_feature_corruptions_are_seeded_and_non_mutating(self):
        original = self.features.copy()
        outliers, outlier_audit = inject_feature_outliers(
            self.features, 0.25, 5.0, seed=4,
        )
        masked, mask_audit = mask_feature_dimensions(
            self.features, 1.0 / 3.0, seed=4,
        )
        shifted, shift_audit = apply_covariance_shift(
            self.features, 0.5, seed=4,
        )
        np.testing.assert_array_equal(self.features, original)
        self.assertEqual(len(outlier_audit["indices"]), 5)
        self.assertEqual(len(mask_audit["dimensions"]), 2)
        self.assertEqual(np.count_nonzero(masked[:, mask_audit["dimensions"]]), 0)
        self.assertFalse(np.array_equal(outliers, original))
        self.assertFalse(np.array_equal(shifted, original))
        self.assertAlmostEqual(shift_audit["determinant"], 1.0)

    def test_matched_benchmark_uses_clean_test_for_all_conditions(self):
        result = run_corruption_benchmark({
            "artifact_path": "unused.json",
            "seeds": [5],
            "dimensions": 3,
            "class_count": 3,
            "geometry_per_class": 18,
            "calibration_per_class": 8,
            "test_per_class": 12,
            "max_iterations": 3,
            "scenarios": [
                {"name": "clean", "kind": "clean"},
                {
                    "name": "outliers",
                    "kind": "feature_outliers",
                    "rate": 0.1,
                    "distance": 5.0,
                },
            ],
        })
        self.assertEqual(len(result["records"]), 2)
        self.assertEqual(set(result["summary"]), {"clean", "outliers"})
        for record in result["records"]:
            self.assertFalse(record["test_used_for_fitting"])
            self.assertEqual(
                set(record["methods"]),
                {"geode", *BASELINE_NAMES},
            )


class ModelEditorTests(unittest.TestCase):
    def setUp(self):
        self.models = {}
        for class_id, center in ((0, 0.0), (1, 10.0)):
            expert = Expert(alpha=2.0)
            expert.add_ellipsoid(EllipsoidExpert(
                center=np.array([center, 0.0]), radii=np.ones(2),
            ))
            self.models[class_id] = [expert]
        self.invalidations = 0
        self.editor = ModelEditor(
            self.models,
            invalidate=lambda _models: setattr(
                self, "invalidations", self.invalidations + 1,
            ),
        )
        self.points = np.array([[0.0, 0.0], [3.0, 0.0], [10.0, 0.0]])

    def _class_scores(self, class_id: int) -> np.ndarray:
        return InferenceEngine(self.models[class_id], 2.0).get_fused_sdf(self.points)

    def test_insert_delete_and_rollback_are_localized_and_audited(self):
        snapshot = self.editor.snapshot()
        original_class_zero = self._class_scores(0)
        original_class_one = self._class_scores(1)
        insertion = self.editor.insert_additive(
            0, EllipsoidExpert(center=np.array([3.0, 0.0]), radii=np.ones(2)),
        )
        self.assertFalse(np.array_equal(self._class_scores(0), original_class_zero))
        np.testing.assert_array_equal(self._class_scores(1), original_class_one)

        self.editor.delete_primitive(
            0, insertion["expert_index"], insertion["ellipsoid_index"],
        )
        np.testing.assert_array_equal(self._class_scores(0), original_class_zero)
        np.testing.assert_array_equal(self._class_scores(1), original_class_one)

        self.editor.insert_additive(
            0, EllipsoidExpert(center=np.array([4.0, 0.0]), radii=np.ones(2)),
        )
        self.editor.rollback(snapshot)
        np.testing.assert_array_equal(self._class_scores(0), original_class_zero)
        np.testing.assert_array_equal(self._class_scores(1), original_class_one)
        self.assertEqual(
            [record["operation"] for record in self.editor.audit_log],
            ["insert_additive", "delete_primitive", "insert_additive", "rollback"],
        )
        self.assertEqual(self.invalidations, 4)

    def test_generic_transaction_restores_exact_snapshot_when_rejected(self):
        original = self.editor.snapshot()

        record = self.editor.apply_transaction(
            lambda: self.models[0][0].add_ellipsoid(EllipsoidExpert(
                center=np.array([2.0, 0.0]), radii=np.ones(2),
            )),
            lambda _models: False,
            operation_name="confirmed_existing_class_update",
            class_id=0,
        )

        self.assertFalse(record["accepted"])
        self.assertEqual(self.editor.snapshot(), original)
        self.assertEqual(record["before_snapshot_id"], record["after_snapshot_id"])
        self.assertEqual(self.invalidations, 2)

    def test_point_insertion_and_local_nudge_leave_other_class_bitwise_stable(self):
        original_class_one = self._class_scores(1)
        points = np.array([[2.8, 0.0], [3.0, 0.1], [3.2, -0.1]])
        insertion = self.editor.insert_additive_from_points(
            0,
            points,
            fitter=lambda values, _seed: EllipsoidExpert(
                center=values.mean(axis=0), radii=np.full(2, 0.5),
            ),
        )
        self.assertEqual(insertion["point_count"], 3)
        np.testing.assert_array_equal(self._class_scores(1), original_class_one)
        nudge = self.editor.local_nudge(
            0, points + np.array([0.5, 0.0]), learning_rate=0.5,
        )
        self.assertTrue(nudge["accepted"])
        np.testing.assert_array_equal(self._class_scores(1), original_class_one)
        self.assertGreater(insertion["edit_seconds"], 0.0)
        self.assertGreater(nudge["edit_seconds"], 0.0)

    def test_rejected_subtractive_insert_rolls_back(self):
        original = self.editor.snapshot()
        record = self.editor.insert_validated_subtractive(
            0,
            0,
            EllipsoidExpert(center=np.zeros(2), radii=np.full(2, 0.5)),
            validator=lambda _models: False,
        )
        self.assertFalse(record["accepted"])
        self.assertEqual(self.editor.snapshot(), original)


class PrimitiveStabilityTests(unittest.TestCase):
    def test_permuted_identical_primitives_have_zero_drift(self):
        def make_models(reverse: bool) -> dict:
            expert = Expert(alpha=2.0)
            ellipsoids = [
                EllipsoidExpert(np.array([0.0, 0.0]), np.array([1.0, 2.0])),
                EllipsoidExpert(np.array([3.0, 0.0]), np.array([0.5, 1.0])),
            ]
            for ellipsoid in reversed(ellipsoids) if reverse else ellipsoids:
                expert.add_ellipsoid(ellipsoid)
            return {0: [expert]}

        result = evaluate_primitive_stability(
            {1: make_models(False), 2: make_models(True)},
            np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]]),
        )
        self.assertEqual(result["matched_center_drift_mean"], 0.0)
        self.assertEqual(result["matched_precision_drift_mean"], 0.0)
        self.assertEqual(result["prediction_agreement_mean"], 1.0)


class EditabilityScalingTests(unittest.TestCase):
    def test_conditions_vary_exactly_one_axis_from_baseline(self):
        config = {
            "baseline": {
                "class_count": 4,
                "dimensions": 3,
                "primitives_per_class": 2,
            },
            "sweeps": {
                "class_count": [2, 4, 8],
                "dimensions": [2, 3, 6],
                "primitives_per_class": [1, 2, 4],
            },
        }
        conditions = build_scaling_conditions(config)
        self.assertEqual(len(conditions), 7)
        baseline = conditions[0]
        for condition in conditions[1:]:
            changed = sum(
                condition[name] != baseline[name]
                for name in (
                    "class_count", "dimensions", "primitives_per_class",
                )
            )
            self.assertEqual(changed, 1)

    def test_toy_condition_preserves_locality_and_reports_scaling_metrics(self):
        result = run_scaling_condition(
            {
                "name": "toy",
                "class_count": 3,
                "dimensions": 2,
                "primitives_per_class": 2,
            },
            seed=7,
            repeat_count=1,
            evaluation_count=24,
            update_point_count=6,
        )
        self.assertTrue(result["all_exit_gates_passed"])
        self.assertEqual(result["total_primitives"], 6)
        self.assertGreater(
            result["serialized_bytes"]["insertion_growth"]["median"], 0,
        )
        self.assertEqual(result["cache_invalidations"]["median"], 5)
        self.assertEqual(
            result["routing_counts"]["baseline"][
                "shortlisted_candidate_pairs"
            ]["median"],
            72,
        )
        self.assertEqual(
            result["routing_counts"]["baseline"]["primitive_sdf_pairs"]["median"],
            144,
        )
        self.assertEqual(
            result["routing_counts"]["after_insertion"][
                "primitive_sdf_pairs"
            ]["median"],
            168,
        )
        self.assertIn("full_model_reconstruction", result["latency_seconds"])


class ClassificationBaselineTests(unittest.TestCase):
    def test_all_matched_baselines_return_aligned_probabilities(self):
        rng = np.random.default_rng(14)
        labels = np.repeat(np.arange(3), 30)
        features = np.vstack([
            rng.normal([class_id * 3.0, 0.0], 0.35, size=(30, 2))
            for class_id in range(3)
        ])
        baselines = fit_classification_baselines(
            features,
            labels,
            components_by_class={0: 2, 1: 2, 2: 2},
            seed=14,
            rbf_sample_limit=100,
        )
        self.assertEqual(
            set(baselines),
            {
                "logistic_regression", "nearest_centroid", "shrinkage_gaussian",
                "matched_gmm", "knn", "linear_svm", "rbf_svm",
                "histogram_gradient_boosting",
            },
        )
        for baseline in baselines.values():
            probabilities = baseline.predict_proba(features)
            self.assertEqual(probabilities.shape, (len(features), 3))
            np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
            self.assertGreater(np.mean(baseline.predict(features) == labels), 0.9)


class CSGAblationTests(unittest.TestCase):
    def test_toy_ablation_produces_comparable_variant_records(self):
        rng = np.random.default_rng(19)
        labels = np.repeat(np.arange(3), 40)
        features = np.vstack([
            rng.normal(class_id * 2.0, 0.5, size=(40, 4))
            for class_id in range(3)
        ])
        with redirect_stdout(io.StringIO()):
            result = run_csg_ablation_experiment(
                features,
                labels,
                seed=19,
                pca_components=3,
                max_iterations=2,
                nudge_iterations=0,
                bootstrap_resamples=10,
                use_gpu=False,
            )
        self.assertEqual(len(result["records"]), 21)
        self.assertEqual(
            {record["geometry_variant"] for record in result["records"]},
            {"none", "A0", "A1", "A2"},
        )
        self.assertEqual(
            {
                record["method"] for record in result["records"]
                if record["geometry_variant"] == "none"
            },
            {
                "logistic_regression", "nearest_centroid",
                "shrinkage_gaussian", "matched_gmm", "knn", "linear_svm",
                "rbf_svm", "histogram_gradient_boosting",
            },
        )
        self.assertEqual(
            len({record["split_hash"] for record in result["records"]}), 1,
        )
        self.assertEqual(sum(result["split_counts"].values()), len(labels))
        self.assertEqual(set(result["carve_audits"]), {"A0", "A1", "A2"})

    def test_sphere_csg_records_report_primitive_and_gpu_backend(self):
        try:
            select_device()
        except Exception as error:
            self.skipTest(f"OpenCL GPU unavailable: {error}")
        rng = np.random.default_rng(29)
        labels = np.repeat(np.arange(3), 40)
        features = np.vstack([
            rng.normal(class_id * 1.5, 0.6, size=(40, 4))
            for class_id in range(3)
        ])
        output = io.StringIO()
        with redirect_stdout(output):
            result = run_csg_ablation_experiment(
                features,
                labels,
                seed=29,
                pca_components=3,
                max_iterations=2,
                nudge_iterations=0,
                bootstrap_resamples=10,
                baseline_rbf_sample_limit=100,
                use_gpu=True,
                fitter="spherical_covariance",
            )

        self.assertEqual(result["primitive_family"], "sphere")
        self.assertTrue(result["gpu_candidate_fitting"])
        self.assertIn("Primitive family: sphere (OpenCL)", output.getvalue())
        self.assertEqual(
            {
                record["geometry_variant"] for record in result["records"]
                if record["geometry_variant"] != "none"
            },
            {
                "spherical_covariance_A0",
                "spherical_covariance_A1",
                "spherical_covariance_A2",
            },
        )

    def test_summary_includes_absolute_baseline_and_geode_metrics(self):
        records = []
        for readout in ("raw", "temperature", "diagonal", "multinomial"):
            for variant, accuracy in (("A0", 0.5), ("A1", 0.75), ("A2", 1.0)):
                records.append({
                    "method": "geode",
                    "geometry_variant": variant,
                    "readout": readout,
                    "targets": [0, 1, 1, 0],
                    "predictions": [0, 1, 1, 0] if accuracy == 1.0 else [0, 0, 1, 0],
                    "metrics": {"accuracy": accuracy},
                    "performance": {"inference_seconds": 0.1},
                    "model_stats": {"experts": 2},
                })
        records.append({
            "method": "logistic_regression",
            "geometry_variant": "none",
            "readout": "native_probability",
            "targets": [0, 1, 1, 0],
            "predictions": [0, 1, 1, 0],
            "metrics": {"accuracy": 1.0},
            "performance": {"fit_seconds": 0.2},
            "model_stats": {"training_samples": 4},
        })
        summary = summarize_runs([{
            "seed": 7,
            "metrics": {
                "records": records,
                "carve_audits": {"A1": [], "A2": []},
            },
        }], bootstrap_resamples=10)
        baseline = next(
            result for result in summary["absolute_results"]
            if result["geometry_variant"] == "none"
        )
        self.assertEqual(baseline["metrics_mean"]["accuracy"], 1.0)
        self.assertEqual(baseline["performance_mean"]["fit_seconds"], 0.2)
        self.assertEqual(baseline["model_stats_mean"]["training_samples"], 4.0)


class EllipsoidFitterTests(unittest.TestCase):
    def test_sphere_is_the_default_primitive_family(self):
        constructor = GreedyConstructor(seed=7)
        self.assertEqual(constructor.primitive_family, "sphere")
        self.assertEqual(constructor._minimal_seed_size(19), 21)

    def test_sphere_uses_direct_d_plus_two_seed(self):
        dimension = 19
        constructor = GreedyConstructor(primitive_family="sphere", seed=7)
        self.assertEqual(constructor._minimal_seed_size(dimension), dimension + 2)

        points = np.random.default_rng(7).normal(
            size=(dimension + 2, dimension)
        )
        candidate = constructor._generate_candidate(points)

        expected_radius = np.sqrt(
            np.sum(np.var(points, axis=0, ddof=1))
        )
        np.testing.assert_allclose(candidate.center, np.mean(points, axis=0))
        np.testing.assert_allclose(candidate.radii, expected_radius)
        np.testing.assert_array_equal(
            candidate.orientation,
            np.eye(dimension),
        )

    def test_sphere_custom_fitter_also_uses_d_plus_two_seed(self):
        constructor = GreedyConstructor(
            candidate_fitter=ELLIPSOID_FITTERS["spherical_covariance"],
            primitive_family="sphere",
            seed=7,
        )

        self.assertEqual(constructor._minimal_seed_size(19), 21)

    def test_sphere_rejects_fewer_than_d_plus_two_points(self):
        constructor = GreedyConstructor(primitive_family="sphere", seed=7)
        with self.assertRaisesRegex(ValueError, "At least 6 points"):
            constructor._generate_candidate(np.ones((5, 4)))

    def test_family_projection_restores_sphere_after_nudge(self):
        candidate = EllipsoidExpert(
            center=np.zeros(3),
            radii=np.asarray([1.0, 2.0, 3.0]),
            orientation=np.asarray([
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]),
        )
        GreedyConstructor(primitive_family="sphere")._project_primitive_family(candidate)

        np.testing.assert_allclose(candidate.radii, np.sqrt(14.0 / 3.0))
        np.testing.assert_array_equal(candidate.orientation, np.eye(3))

    def test_gpu_sphere_constructor_preserves_constraint_and_labels(self):
        try:
            select_device()
        except Exception as error:
            self.skipTest(f"OpenCL GPU unavailable: {error}")
        points = np.random.default_rng(7).normal(size=(120, 8))
        constructor = GreedyConstructor(
            consensus_threshold=0.05,
            capture_threshold=0.2,
            max_iterations=4,
            use_gpu=True,
            seed=7,
            candidate_fitter=ELLIPSOID_FITTERS["spherical_covariance"],
            primitive_family="sphere",
            gpu_candidate_fitting=True,
        )
        output = io.StringIO()
        with redirect_stdout(output):
            models = constructor.build_model(points)
        primitives = [item for expert in models for item in expert.ellipsoids]

        self.assertTrue(primitives)
        self.assertTrue(all(np.allclose(item.radii, item.radii[0]) for item in primitives))
        self.assertIn("Expert sphere #", output.getvalue())
        self.assertIn(" spheres,", output.getvalue())
        self.assertNotIn("ellipsoid(s)", output.getvalue())

    def test_gpu_axis_aligned_fitters_match_numpy(self):
        try:
            select_device()
        except Exception as error:
            self.skipTest(f"OpenCL GPU unavailable: {error}")
        seeds = np.random.default_rng(19).normal(size=(5, 24, 4))
        expected_centers = np.mean(seeds, axis=1)
        variances = np.var(seeds, axis=1, ddof=1)

        diagonal_centers, diagonal_radii = fit_axis_aligned_candidates_gpu(
            seeds, "diagonal_ellipsoid",
        )
        sphere_centers, sphere_radii = fit_axis_aligned_candidates_gpu(
            seeds, "sphere",
        )

        np.testing.assert_allclose(diagonal_centers, expected_centers, atol=2e-6)
        np.testing.assert_allclose(
            diagonal_radii, np.sqrt(variances * seeds.shape[2]), atol=2e-6,
        )
        np.testing.assert_allclose(sphere_centers, expected_centers, atol=2e-6)
        np.testing.assert_allclose(
            sphere_radii,
            np.repeat(np.sqrt(np.sum(variances, axis=1))[:, None], seeds.shape[2], axis=1),
            atol=2e-6,
        )

    def test_spherical_fitter_preserves_trace_with_equal_radii(self):
        points = np.asarray([
            [-3.0, -1.0, -0.5],
            [-1.0, -0.5, 0.0],
            [1.0, 0.5, 0.0],
            [3.0, 1.0, 0.5],
        ])
        model = ELLIPSOID_FITTERS["spherical_covariance"](points, 17)
        expected_radius = np.sqrt(np.sum(np.var(points, axis=0, ddof=1)))

        np.testing.assert_allclose(model.radii, expected_radius)
        np.testing.assert_array_equal(model.orientation, np.eye(points.shape[1]))

    def test_all_fitters_return_finite_three_dimensional_models(self):
        rng = np.random.default_rng(17)
        directions = rng.normal(size=(120, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        points = directions * np.array([3.0, 2.0, 1.0])
        for name, fitter in ELLIPSOID_FITTERS.items():
            with self.subTest(fitter=name):
                model = fitter(points, 17)
                self.assertEqual(model.center.shape, (3,))
                self.assertTrue(np.all(np.isfinite(model.radii)))
                self.assertTrue(np.all(model.radii > 0.0))
                self.assertTrue(np.all(np.isfinite(model.compute_sdf(points))))

    def test_low_support_failure_does_not_abort_other_fitters(self):
        result = run_fitter_benchmark({
            "anisotropy": 4.0,
            "dimensions": [19],
            "fitters": ["quadric_svd", "shrinkage_covariance"],
            "scenarios": [{"name": "low_support", "fit_samples": 40}],
            "seeds": [3],
            "test_samples": 20,
        })
        records = {record["fitter"]: record for record in result["records"]}
        self.assertFalse(records["quadric_svd"]["success"])
        self.assertTrue(records["shrinkage_covariance"]["success"])

    def test_candidate_budget_keeps_test_data_observational(self):
        result = run_fitter_budget_benchmark({
            "anisotropy": 3.0,
            "candidate_counts": [2],
            "dimensions": [3],
            "fitters": ["full_covariance"],
            "scenarios": [{"name": "toy", "fit_samples": 30}],
            "seeds": [5],
            "test_samples": 20,
            "wall_clock_seconds": [],
        })
        self.assertEqual(result["records"][0]["attempts"], 2)
        self.assertFalse(result["records"][0]["test_used_for_selection"])

    def test_custom_fitter_uses_linear_default_seed_size(self):
        fitter = ELLIPSOID_FITTERS["shrinkage_covariance"]
        constructor = GreedyConstructor(candidate_fitter=fitter, seed=7)
        self.assertEqual(constructor._minimal_seed_size(19), 39)
        points = np.random.default_rng(7).normal(size=(39, 19))
        candidate = constructor._generate_candidate(points)
        self.assertEqual(candidate.center.shape, (19,))

    def test_knn_seed_accepts_pool_exactly_equal_to_seed_size(self):
        points = np.arange(24, dtype=np.float64).reshape(6, 4)
        sampled = GreedyConstructor._knn_seed(
            points, 6, np.random.default_rng(2),
        )
        np.testing.assert_array_equal(sampled, points)


class ProbabilisticInferenceTests(unittest.TestCase):
    def test_per_class_temperature_summary_stops_without_improvement(self):
        runs = []
        for seed in (3, 5):
            records = []
            for name, accuracy_value, nll_value in (
                ("probabilistic_global_temperature", 0.80, 0.49),
                ("probabilistic_per_class_temperature", 0.80, 0.491),
                ("hybrid_global_temperature", 0.81, 0.47),
                ("hybrid_per_class_temperature", 0.81, 0.471),
            ):
                records.append({
                    "split": "test",
                    "readout": name,
                    "converged": True,
                    "metrics": {
                        "accuracy": accuracy_value,
                        "negative_log_likelihood": nll_value,
                        "brier_score": 0.2,
                        "expected_calibration_error": 0.03,
                    },
                })
            runs.append({
                "experiment_id": f"per-class-{seed}",
                "config": {"seed": seed},
                "metrics": {
                    "records": records,
                    "selected_score_input": "hybrid_global_temperature",
                    "per_class_likelihood_optimization": {
                        "converged": True,
                        "covariance_temperatures": [1.8, 2.2],
                        "baseline_calibration_nll": 0.5,
                        "fitted_calibration_nll": 0.49,
                    },
                },
            })

        summary = summarize_per_class_temperature_runs(runs)

        self.assertFalse(summary["advancement_gate"]["passed"])
        self.assertFalse(summary["advancement_gate"]["accuracy_improved"])
        self.assertFalse(
            summary["advancement_gate"]["negative_log_likelihood_improved"]
        )
        self.assertLess(summary["calibration_nll_improvement"]["mean"], 0.0)

    def test_global_temperature_summary_applies_advancement_gate(self):
        runs = []
        for seed in (3, 5):
            records = []
            for name, accuracy_value, nll_value in (
                ("probabilistic", 0.80, 0.50),
                ("probabilistic_global_temperature", 0.80, 0.49),
                ("hybrid", 0.81, 0.48),
                ("hybrid_global_temperature", 0.811, 0.47),
            ):
                records.append({
                    "split": "test",
                    "readout": name,
                    "converged": True,
                    "metrics": {
                        "accuracy": accuracy_value,
                        "negative_log_likelihood": nll_value,
                        "brier_score": 0.2,
                        "expected_calibration_error": 0.03,
                    },
                })
            runs.append({
                "experiment_id": f"temperature-{seed}",
                "config": {"seed": seed},
                "metrics": {
                    "records": records,
                    "selected_score_input": "hybrid_global_temperature",
                    "likelihood_optimization": {
                        "converged": True,
                        "covariance_temperature": 2.0 + seed / 10.0,
                    },
                },
            })

        summary = summarize_global_temperature_runs(runs)

        self.assertTrue(summary["advancement_gate"]["passed"])
        self.assertTrue(summary["advancement_gate"]["accuracy_improved"])
        self.assertTrue(
            summary["advancement_gate"]["negative_log_likelihood_improved"]
        )
        self.assertEqual(summary["selection_winners"], {
            "hybrid_global_temperature": 2,
        })

    def test_hybrid_summary_applies_predeclared_gate(self):
        runs = []
        for seed in (3, 5):
            records = []
            for name, accuracy_value, nll_value in (
                ("geometric", 0.80, 0.50),
                ("probabilistic", 0.801, 0.49),
                ("hybrid", 0.799, 0.48),
                ("feature_control", 0.81, 0.47),
            ):
                records.append({
                    "split": "test",
                    "readout": name,
                    "converged": True,
                    "metrics": {
                        "accuracy": accuracy_value,
                        "negative_log_likelihood": nll_value,
                        "brier_score": 0.2,
                        "expected_calibration_error": 0.03,
                    },
                })
            runs.append({
                "experiment_id": f"run-{seed}",
                "config": {"seed": seed},
                "metrics": {
                    "records": records,
                    "selected_score_input": "hybrid",
                },
            })

        summary = summarize_hybrid_runs(runs)

        self.assertTrue(summary["advancement_gate"]["passed"])
        self.assertTrue(summary["advancement_gate"]["nll_improves_both"])
        self.assertTrue(summary["advancement_gate"]["accuracy_within_tolerance"])
        self.assertEqual(summary["selection_winners"], {"hybrid": 2})
        self.assertEqual(summary["unique_experiments"], 2)

    def test_hybrid_field_ablation_reuses_one_model_and_audits_inputs(self):
        rng = np.random.default_rng(47)
        labels = np.repeat(np.arange(3), 40)
        features = np.vstack([
            rng.normal(class_id * 1.5, 0.6, size=(40, 5))
            for class_id in range(3)
        ])
        with redirect_stdout(io.StringIO()):
            result = run_hybrid_field_ablation(
                features,
                labels,
                fitter="spherical_covariance",
                seed=47,
                pca_components=4,
                consensus_threshold=0.1,
                capture_threshold=0.5,
                max_iterations=10,
                nudge_iterations=0,
                bootstrap_resamples=10,
                use_gpu=False,
            )

        self.assertEqual(result["model_fit_count"], 1)
        self.assertEqual(result["readout_fit_count"], 4)
        self.assertEqual(len(result["records"]), 8)
        self.assertFalse(result["test_used_for_selection"])
        self.assertTrue(result["selection_used_for_model_choice"])
        self.assertIn(
            result["selected_score_input"],
            {"geometric", "probabilistic", "hybrid"},
        )
        self.assertEqual(
            {record["readout"] for record in result["records"]},
            {"geometric", "probabilistic", "hybrid", "feature_control"},
        )
        widths = {
            record["readout"]: record["performance"]["readout_input_width"]
            for record in result["records"]
            if record["split"] == "selection"
        }
        self.assertEqual(widths["geometric"], 3)
        self.assertEqual(widths["probabilistic"], 3)
        self.assertEqual(widths["hybrid"], 6)
        self.assertEqual(widths["feature_control"], 2)
        self.assertTrue(all(record["converged"] for record in result["records"]))
        self.assertEqual(
            len({record["split_hash"] for record in result["records"]}), 2,
        )

    def test_global_temperature_uses_calibration_and_frozen_geometry(self):
        rng = np.random.default_rng(49)
        labels = np.repeat(np.arange(3), 40)
        features = np.vstack([
            rng.normal(class_id * 1.5, 0.6, size=(40, 5))
            for class_id in range(3)
        ])
        with redirect_stdout(io.StringIO()):
            result = run_hybrid_field_ablation(
                features,
                labels,
                fitter="spherical_covariance",
                seed=49,
                pca_components=4,
                consensus_threshold=0.1,
                capture_threshold=0.5,
                max_iterations=10,
                nudge_iterations=0,
                bootstrap_resamples=10,
                use_gpu=False,
                optimize_global_temperature=True,
            )

        optimization = result["likelihood_optimization"]
        self.assertEqual(result["model_fit_count"], 1)
        self.assertEqual(result["readout_fit_count"], 6)
        self.assertEqual(len(result["records"]), 12)
        self.assertTrue(optimization["converged"])
        self.assertGreater(optimization["covariance_temperature"], 0.0)
        self.assertLessEqual(
            optimization["fitted_calibration_nll"],
            optimization["baseline_calibration_nll"] + 1e-10,
        )
        self.assertFalse(result["test_used_for_selection"])
        self.assertEqual(
            {record["readout"] for record in result["records"]},
            {
                "geometric",
                "probabilistic",
                "hybrid",
                "feature_control",
                "probabilistic_global_temperature",
                "hybrid_global_temperature",
            },
        )

    def test_per_class_temperature_extends_global_with_frozen_geometry(self):
        rng = np.random.default_rng(51)
        labels = np.repeat(np.arange(3), 40)
        features = np.vstack([
            rng.normal(class_id * 1.5, 0.6 + class_id * 0.15, size=(40, 5))
            for class_id in range(3)
        ])
        with redirect_stdout(io.StringIO()):
            result = run_hybrid_field_ablation(
                features,
                labels,
                fitter="spherical_covariance",
                seed=51,
                pca_components=4,
                consensus_threshold=0.1,
                capture_threshold=0.5,
                max_iterations=10,
                nudge_iterations=0,
                bootstrap_resamples=10,
                use_gpu=False,
                optimize_global_temperature=True,
                optimize_per_class_temperature=True,
            )

        optimization = result["per_class_likelihood_optimization"]
        temperatures = optimization["covariance_temperatures"]
        self.assertEqual(result["model_fit_count"], 1)
        self.assertEqual(result["readout_fit_count"], 8)
        self.assertEqual(len(result["records"]), 16)
        self.assertTrue(optimization["converged"])
        self.assertEqual(len(temperatures), 3)
        self.assertTrue(all(temperature > 0.0 for temperature in temperatures))
        self.assertLessEqual(
            optimization["fitted_calibration_nll"],
            optimization["baseline_calibration_nll"] + 1e-10,
        )
        self.assertFalse(result["test_used_for_selection"])
        self.assertIn(
            "hybrid_per_class_temperature",
            {record["readout"] for record in result["records"]},
        )

    def test_gpu_class_nlls_match_cpu_hierarchical_mixture(self):
        try:
            select_device()
        except Exception as error:
            self.skipTest(f"OpenCL GPU unavailable: {error}")
        first = Expert()
        first.add_ellipsoid(EllipsoidExpert([-1.0, 0.0], [1.2, 0.8]))
        first.add_ellipsoid(EllipsoidExpert([-0.5, 0.3], [0.9, 1.1]))
        second = Expert()
        second.add_ellipsoid(EllipsoidExpert([1.0, 0.0], [1.0, 1.3]))
        models = {0: [first], 1: [second]}
        points = np.random.default_rng(31).normal(size=(25, 2))

        temperatures = np.asarray([0.8, 2.3])
        cpu = ProbabilisticInferenceEngine(models).class_nlls(
            points, covariance_temperature=temperatures,
        )
        gpu = GPUInferenceEngine([models[0], models[1]]).class_nlls(
            points, covariance_temperature=temperatures,
        )
        np.testing.assert_allclose(gpu, cpu, rtol=2e-5, atol=2e-5)

    def test_covariance_temperature_matches_scaled_gaussian(self):
        primitive = EllipsoidExpert(np.zeros(2), np.sqrt([2.0, 8.0]))
        points = np.asarray([[0.0, 0.0], [1.0, 2.0]])
        temperature = 2.5
        covariance = np.asarray([1.0, 4.0]) * temperature
        expected = 0.5 * (
            np.sum(np.square(points) / covariance, axis=1)
            + np.sum(np.log(covariance))
            + 2 * np.log(2.0 * np.pi)
        )

        np.testing.assert_allclose(
            gaussian_primitive_nll(
                primitive,
                points,
                covariance_temperature=temperature,
            ),
            expected,
            atol=1e-12,
        )
        with self.assertRaisesRegex(ValueError, "covariance_temperature"):
            gaussian_primitive_nll(
                primitive, points, covariance_temperature=0.0,
            )

    def test_primitive_nll_matches_covariance_density(self):
        center = np.asarray([0.5, -1.0])
        covariance = np.asarray([4.0, 0.25])
        primitive = EllipsoidExpert(
            center=center,
            radii=np.sqrt(len(center) * covariance),
        )
        points = np.asarray([[0.5, -1.0], [2.5, -0.5], [-1.5, -1.5]])
        delta = points - center
        expected = 0.5 * (
            np.sum(np.square(delta) / covariance, axis=1)
            + np.sum(np.log(covariance))
            + len(center) * np.log(2.0 * np.pi)
        )

        np.testing.assert_allclose(
            gaussian_primitive_nll(primitive, points), expected, atol=1e-12,
        )

    def test_duplicate_components_preserve_mixture_nll(self):
        primitive = EllipsoidExpert(np.zeros(2), np.ones(2) * np.sqrt(2.0))
        points = np.asarray([[0.0, 0.0], [1.0, -1.0]])
        single = Expert()
        single.add_ellipsoid(copy.deepcopy(primitive))
        duplicate = Expert()
        duplicate.add_ellipsoid(copy.deepcopy(primitive))
        duplicate.add_ellipsoid(copy.deepcopy(primitive))

        single_scores = ProbabilisticInferenceEngine({0: [single]}).class_nlls(points)
        duplicate_scores = ProbabilisticInferenceEngine({0: [duplicate]}).class_nlls(points)
        np.testing.assert_allclose(duplicate_scores, single_scores, atol=1e-12)

    def test_subtractive_probability_fails_closed(self):
        expert = Expert()
        expert.add_ellipsoid(EllipsoidExpert(np.zeros(2), np.ones(2)))
        expert.add_ellipsoid(
            EllipsoidExpert(np.zeros(2), np.ones(2), polarity=-1)
        )
        with self.assertRaisesRegex(ValueError, "subtractive"):
            ProbabilisticInferenceEngine({0: [expert]})

    def test_empty_class_probability_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "no probability model"):
            ProbabilisticInferenceEngine({0: []})
        with self.assertRaisesRegex(ValueError, "no probability model"):
            ProbabilisticInferenceEngine({0: [Expert()]})

    def test_matched_field_ablation_reuses_models_and_splits(self):
        rng = np.random.default_rng(43)
        labels = np.repeat(np.arange(3), 40)
        features = np.vstack([
            rng.normal(class_id * 1.5, 0.6, size=(40, 5))
            for class_id in range(3)
        ])
        with redirect_stdout(io.StringIO()):
            result = run_probabilistic_field_ablation(
                features,
                labels,
                fitters=["spherical_covariance"],
                seed=43,
                pca_components=4,
                consensus_threshold=0.1,
                capture_threshold=0.5,
                max_iterations=10,
                nudge_iterations=0,
                bootstrap_resamples=10,
                use_gpu=False,
            )

        self.assertEqual(len(result["records"]), 20)
        self.assertFalse(result["selection_used_for_model_choice"])
        self.assertFalse(result["test_used_for_selection"])
        self.assertFalse(result["probabilistic_fitting_used"])
        self.assertEqual(
            {record["score_semantics"] for record in result["records"]},
            {"geometric", "probabilistic"},
        )
        self.assertEqual(
            len({record["split_hash"] for record in result["records"]}), 2,
        )
        self.assertTrue(all(
            diagnostic["finite"]
            for diagnostic in result["score_diagnostics"].values()
        ))


class ScoreReadoutTests(unittest.TestCase):
    def test_evaluation_records_include_readout_diagnostics(self):
        rng = np.random.default_rng(25)
        labels = np.repeat(np.arange(3), 20)
        features = rng.normal(size=(len(labels), 4))
        scores = rng.normal(size=(len(labels), 3))
        scores[np.arange(len(labels)), labels] -= 2.0

        records = evaluate_score_readouts(
            calibration_scores=scores,
            calibration_labels=labels,
            calibration_features=features,
            evaluation_scores=scores,
            evaluation_labels=labels,
            evaluation_features=features,
            class_ids=np.arange(3),
            dataset="toy",
            split="validation",
            representation="identity",
            geometry_variant="fixed",
            model_stats={"experts": 3},
            geometry_sample_count=60,
            geometry_fit_seconds=0.0,
            seed=25,
            evaluation_indices=np.arange(len(labels)),
            bootstrap_resamples=10,
        )

        self.assertFalse(records["raw"]["performance"]["readout_input_standardized"])
        self.assertTrue(
            records["multinomial"]["performance"]["readout_input_standardized"]
        )
        self.assertTrue(
            records["feature_logistic"]["performance"]["readout_input_standardized"]
        )
        self.assertTrue(records["multinomial"]["converged"])
        self.assertEqual(records["multinomial"]["warnings"], [])
        self.assertGreater(
            records["multinomial"]["performance"]["readout_fit_iterations"], 0,
        )
        self.assertEqual(
            records["multinomial"]["performance"]["readout_iteration_limit"],
            1000,
        )

    def test_classifier_scaling_uses_calibration_statistics(self):
        rng = np.random.default_rng(18)
        labels = np.repeat(np.arange(3), 30)
        scores = rng.normal(size=(len(labels), 3)) * np.array([1e-3, 1e3, 2.0])
        readout = fit_score_readout(
            "multinomial", scores, labels, np.arange(3), seed=18,
        )

        np.testing.assert_allclose(readout.classifier_mean, np.mean(scores, axis=0))
        np.testing.assert_allclose(readout.classifier_scale, np.std(scores, axis=0))
        self.assertTrue(readout.converged)
        self.assertLess(readout.fit_iterations, readout.iteration_limit)
        probabilities = readout.predict_proba(scores + np.array([0.0, 1e6, 0.0]))
        self.assertEqual(probabilities.shape, scores.shape)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_iteration_limit_is_reported(self):
        rng = np.random.default_rng(21)
        labels = np.repeat(np.arange(3), 30)
        scores = rng.normal(size=(len(labels), 3))
        readout = fit_score_readout(
            "multinomial",
            scores,
            labels,
            np.arange(3),
            seed=21,
            logistic_max_iter=1,
        )

        self.assertFalse(readout.converged)
        self.assertEqual(readout.fit_iterations, 1)
        self.assertEqual(readout.iteration_limit, 1)
        self.assertTrue(readout.fit_warnings)

    def test_three_way_ablation_split_is_disjoint_and_stratified(self):
        labels = np.repeat(np.arange(3), 20)
        geometry, carve, calibration = stratified_geometry_carve_calibration_split(
            np.arange(len(labels)), labels, seed=6,
        )
        self.assertFalse(set(geometry) & set(carve))
        self.assertFalse(set(geometry) & set(calibration))
        self.assertFalse(set(carve) & set(calibration))
        self.assertEqual(
            set(np.concatenate([geometry, carve, calibration])),
            set(range(len(labels))),
        )
        for subset in (geometry, carve, calibration):
            np.testing.assert_array_equal(np.unique(labels[subset]), np.arange(3))

    def test_geometry_calibration_split_is_stratified_and_reproducible(self):
        labels = np.repeat(np.arange(3), 10)
        indices = np.arange(len(labels))
        first = stratified_geometry_calibration_split(
            indices, labels, calibration_fraction=0.2, seed=4,
        )
        second = stratified_geometry_calibration_split(
            indices, labels, calibration_fraction=0.2, seed=4,
        )
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(set(first[0]).intersection(first[1]), set())
        np.testing.assert_array_equal(np.unique(labels[first[0]]), np.arange(3))
        np.testing.assert_array_equal(np.unique(labels[first[1]]), np.arange(3))

    def test_all_modes_share_fixed_geometry_and_return_probabilities(self):
        rng = np.random.default_rng(8)
        labels = np.repeat(np.arange(3), 30)
        features = rng.normal(size=(len(labels), 4))
        scores = rng.normal(2.0, 0.2, size=(len(labels), 3))
        scores[np.arange(len(labels)), labels] = rng.normal(
            -1.0, 0.2, size=len(labels),
        )
        readouts = fit_all_readouts(
            scores, labels, np.arange(3), features, seed=8,
        )

        self.assertEqual(
            set(readouts),
            {"raw", "temperature", "diagonal", "multinomial", "feature_logistic"},
        )
        for mode, readout in readouts.items():
            probabilities = readout.predict_proba(
                scores, features if mode == "feature_logistic" else None,
            )
            self.assertEqual(probabilities.shape, scores.shape)
            np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        np.testing.assert_array_equal(
            readouts["raw"].predict(scores), np.argmin(scores, axis=1),
        )

    def test_long_form_record_contains_comparable_fields(self):
        truth = np.array([0, 1, 1, 0])
        probabilities = np.array([
            [0.8, 0.2], [0.1, 0.9], [0.4, 0.6], [0.7, 0.3],
        ])
        record = classification_result_record(
            dataset="toy",
            split="test",
            seed=3,
            method="geode",
            representation="identity",
            geometry_variant="additive",
            readout="raw",
            y_true=truth,
            probabilities=probabilities,
            classes=np.array([0, 1]),
            model_stats={"experts": 2},
            performance={"fit_seconds": 0.1},
            adequacy={"minimum_class_count": 2},
            bootstrap_resamples=50,
            split_hash="split-123",
            feature_hash="feature-456",
        )
        self.assertEqual(record["sample_count"], 4)
        self.assertEqual(record["metrics"]["accuracy"], 1.0)
        self.assertIn("accuracy", record["confidence_intervals"])
        self.assertEqual(record["readout"], "raw")
        self.assertEqual(record["split_hash"], "split-123")
        self.assertEqual(record["feature_hash"], "feature-456")


class ModelStatsTests(unittest.TestCase):
    def test_counts_hand_built_models(self):
        first = Expert(alpha=2.0)
        first.add_ellipsoid(EllipsoidExpert([0.0, 0.0], [1.0, 2.0]))
        first.add_ellipsoid(
            EllipsoidExpert([0.0, 0.0], [0.5, 0.5], polarity=-1),
        )
        second = Expert(alpha=2.0)
        second.add_ellipsoid(EllipsoidExpert([2.0, 0.0], [1.0, 1.0]))

        stats = model_structure_stats({0: [first, second], 1: []}, 123)

        self.assertEqual(stats["classes"], 2)
        self.assertEqual(stats["empty_classes"], 1)
        self.assertEqual(stats["experts"], 2)
        self.assertEqual(stats["additive_ellipsoids"], 2)
        self.assertEqual(stats["subtractive_ellipsoids"], 1)
        self.assertEqual(stats["fitted_parameters"], 24)
        self.assertEqual(stats["approximate_model_bytes"], 24 * 8)
        self.assertEqual(stats["candidate_evaluations"], 123)


class GeometryMetricTests(unittest.TestCase):
    def test_carve_acceptance_rewards_recovery_and_rejects_damage(self):
        constructor = GreedyConstructor(capture_threshold=0.0, seed=3)
        positive = np.array([[-0.8], [-0.6], [0.6], [0.8]])
        negative = np.array([[-0.1], [0.0], [0.1]])

        helpful_expert = Expert(alpha=2.0)
        helpful_expert.add_ellipsoid(EllipsoidExpert([0.0], [1.0]))
        helpful = EllipsoidExpert([0.0], [0.25], polarity=-1)
        helpful_expert.add_ellipsoid(helpful)
        helpful_decision = constructor._carve_acceptance_decision(
            helpful_expert, helpful, positive, negative, 0.0, 0.0,
        )
        self.assertTrue(helpful_decision["accepted"])
        self.assertEqual(helpful_decision["recovered_false_positives"], 3)
        self.assertEqual(helpful_decision["damaged_true_positives"], 0)

        damaging_expert = Expert(alpha=2.0)
        damaging_expert.add_ellipsoid(EllipsoidExpert([0.0], [1.0]))
        damaging = EllipsoidExpert([0.7], [0.3], polarity=-1)
        damaging_expert.add_ellipsoid(damaging)
        damaging_decision = constructor._carve_acceptance_decision(
            damaging_expert, damaging, positive, negative, 0.0, 0.0,
        )
        self.assertFalse(damaging_decision["accepted"])
        self.assertGreater(damaging_decision["damaged_true_positives"], 0)

    def test_geometry_score_retains_historical_alias(self):
        expert = Expert(alpha=1.0)
        expert.add_ellipsoid(EllipsoidExpert([0.0, 0.0], [1.0, 1.0]))
        points = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        corrected = _geometry_normalized_residual_score([expert], points)
        self.assertAlmostEqual(corrected, 1.0)
        self.assertEqual(_r2_score([expert], points), corrected)

    def test_sampled_sphere_has_near_zero_self_chamfer(self):
        expert = Expert(alpha=2.0)
        expert.add_ellipsoid(
            EllipsoidExpert([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        )
        reference = sample_fused_surface(
            [expert], samples_per_ellipsoid=128, seed=9,
        )
        distance = symmetric_chamfer_distance(
            [expert], reference, samples_per_ellipsoid=128, seed=9,
        )
        self.assertLess(distance, 1e-20)


if __name__ == "__main__":
    unittest.main()