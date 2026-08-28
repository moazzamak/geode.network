import unittest
import tempfile
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler

from src.model_fingerprint import InputSpec, ModelFingerprint, OutputSpec
from src.open_set import SupportProfile
from src.runtime.artifact_store import LocalArtifactStore
from src.runtime.adaptation_transaction import (
    AdaptationPublicationEvidence,
    AdaptationPublicationPolicy,
    ReviewConfirmation,
    publish_confirmed_adaptation,
    rollback_adaptation,
)
from src.adaptation_policy import ConfirmationKind
from src.runtime.checkpoint import LocalCheckpointStore
from src.runtime.domainnet_manifest import (
    DOMAINNET_DOMAINS,
    DomainNetFile,
    DomainNetManifest,
)
from src.runtime.distributed_qualification import DistributedQualificationEvidence
from src.runtime.modelnet_manifest import ModelNet40Manifest, ModelNetFile
from src.runtime.episode_partitions import (
    build_stratified_episode_partitions,
    validate_episode_partitions,
)
from src.runtime.geometry_feasibility import evaluate_geometry_feasibility
from src.runtime.history_export import export_metric_history
from src.runtime.local_executor import (
    LocalExecutor,
    StageExecutionStatus,
    StageSpec,
)
from src.runtime.metrics import MetricLedger
from src.runtime.model_bundle import (
    BundleNode,
    BundleProvenance,
    LocalModelBundleStore,
    assert_node_replacement,
)
from src.runtime.ray_executor import RayExecutor, RayUnavailableError
from src.runtime.production_service import (
    ProductionPromotionCoordinator,
    ReplicatedBundleService,
)
from src.runtime.refinement_checkpoint import RefinementCheckpointAdapter
from src.runtime.representation_checkpoint import RepresentationCheckpointAdapter
from src.runtime.representation_training import train_representation
from src.runtime.schemas import (
    CheckpointMetadata,
    DatasetEpisodeContract,
    GeometryCapacityContract,
    LifecycleState,
    MetricEvent,
    ModelSelectionContract,
    PretrainingLane,
    PretrainingProvenance,
    ReproducibilityContract,
    ReproducibilityLevel,
    RunContract,
    StageManifest,
)
from src.sdf_engine import EllipsoidExpert, Expert
from src.sdf_optimizer import SDFOptimizer


class RuntimeSchemaTests(unittest.TestCase):
    def _run_contract(self) -> RunContract:
        return RunContract(
            run_id="run-123",
            config_hash="config-456",
            episode=DatasetEpisodeContract(
                geometry_split="geometry-v1",
                readout_calibration_split="readout-v1",
                risk_control_split="risk-v1",
                validation_split="validation-v1",
                final_test_split="final-v1",
            ),
            pretraining=PretrainingProvenance(
                lane=PretrainingLane.CONTROLLED,
                source_datasets=("cifar100-train",),
                objective="supervised classification",
                checkpoint_hash="checkpoint-789",
                license="dataset license recorded separately",
                access_date="2026-07-25",
                overlap_risk="none by construction",
            ),
            geometry=GeometryCapacityContract(
                allowed_families=("sphere", "axis_aligned", "shrinkage"),
                max_condition_number=1e6,
                max_parameter_sample_ratio=0.25,
                min_effective_rank=2.0,
            ),
            model_selection=ModelSelectionContract(
                validation_domains=("proxy-domain",),
                final_domains=("final-domain",),
                selection_rule="maximize validation balanced accuracy",
                primary_metric="balanced_accuracy",
            ),
            reproducibility=ReproducibilityContract(
                level=ReproducibilityLevel.REPLAY_IDENTITY,
                environment_fingerprint="environment-123",
                absolute_tolerance=0.0,
                relative_tolerance=0.0,
            ),
        )

    def test_run_contract_round_trip_includes_preflight_contracts(self):
        contract = self._run_contract()
        self.assertEqual(RunContract.from_dict(contract.to_dict()), contract)

    def test_run_contract_rejects_unknown_and_future_fields(self):
        payload = self._run_contract().to_dict()
        payload["unreviewed_policy"] = True
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            RunContract.from_dict(payload)

        payload = self._run_contract().to_dict()
        payload["schema_version"] = 2
        with self.assertRaisesRegex(ValueError, "unsupported schema_version"):
            RunContract.from_dict(payload)

    def test_calibration_splits_and_selection_domains_must_be_disjoint(self):
        with self.assertRaisesRegex(ValueError, "pairwise distinct"):
            DatasetEpisodeContract(
                geometry_split="geometry",
                readout_calibration_split="shared-calibration",
                risk_control_split="shared-calibration",
                validation_split="validation",
                final_test_split="final",
            )
        with self.assertRaisesRegex(ValueError, "overlap"):
            ModelSelectionContract(
                validation_domains=("domain-a",),
                final_domains=("domain-a",),
                selection_rule="best validation loss",
                primary_metric="loss",
            )

    def test_geometry_and_reproducibility_contracts_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "unsupported geometry"):
            GeometryCapacityContract(
                allowed_families=("unbounded_quadric",),
                max_condition_number=1e6,
                max_parameter_sample_ratio=0.25,
                min_effective_rank=2.0,
            )
        with self.assertRaisesRegex(ValueError, "zero numeric tolerances"):
            ReproducibilityContract(
                level=ReproducibilityLevel.REPLAY_IDENTITY,
                environment_fingerprint="environment-123",
                absolute_tolerance=1e-8,
                relative_tolerance=0.0,
            )
        with self.assertRaisesRegex(ValueError, "max_condition_number must be positive"):
            GeometryCapacityContract(
                allowed_families=("sphere",),
                max_condition_number=float("nan"),
                max_parameter_sample_ratio=0.25,
                min_effective_rank=2.0,
            )

    def test_stage_checkpoint_and_metric_round_trip(self):
        stage = StageManifest.from_dict({
            "schema_version": 1,
            "run_id": "run-123",
            "attempt_id": "attempt-1",
            "stage_name": "features",
            "state": LifecycleState.FEATURES_READY.value,
            "created_at": "2026-07-25T00:00:00Z",
            "input_hashes": {"split": "split-hash"},
            "output_hashes": {"features": "feature-hash"},
        })
        checkpoint = CheckpointMetadata.from_dict({
            "schema_version": 1,
            "run_id": "run-123",
            "attempt_id": "attempt-1",
            "stage_name": "representation",
            "epoch": 2,
            "global_step": 100,
            "created_at": "2026-07-25T00:00:00Z",
            "artifact_hashes": {"weights": "weights-hash"},
        })
        metric = MetricEvent.from_dict({
            "schema_version": 1,
            "event_id": "event-1",
            "run_id": "run-123",
            "attempt_id": "attempt-1",
            "stage_name": "representation",
            "split": "validation",
            "metric_name": "loss",
            "value": 0.5,
            "sample_count": 100,
            "created_at": "2026-07-25T00:00:00Z",
            "epoch": 2,
            "global_step": 100,
            "namespace": "selection",
        })

        self.assertEqual(StageManifest.from_dict(stage.to_dict()), stage)
        self.assertEqual(CheckpointMetadata.from_dict(checkpoint.to_dict()), checkpoint)
        self.assertEqual(MetricEvent.from_dict(metric.to_dict()), metric)

        with self.assertRaisesRegex(ValueError, "value must be finite"):
            MetricEvent(
                event_id="invalid-event",
                run_id="run-123",
                attempt_id="attempt-1",
                stage_name="representation",
                split="validation",
                metric_name="loss",
                value=float("nan"),
                sample_count=100,
                created_at="2026-07-25T00:00:00Z",
            )


class EpisodePartitionTests(unittest.TestCase):
    def test_official_final_test_is_preserved_and_audit_is_deterministic(self):
        labels = np.repeat(np.arange(3), 20)
        development = np.concatenate([
            np.arange(class_id * 20, class_id * 20 + 15)
            for class_id in range(3)
        ])
        final_test = np.concatenate([
            np.arange(class_id * 20 + 15, class_id * 20 + 20)
            for class_id in range(3)
        ])

        first_partitions, first_audit = build_stratified_episode_partitions(
            labels,
            development_indices=development,
            final_test_indices=final_test,
            seed=42,
        )
        second_partitions, second_audit = build_stratified_episode_partitions(
            labels,
            development_indices=development,
            final_test_indices=final_test,
            seed=42,
        )

        np.testing.assert_array_equal(first_partitions["final_test"], final_test)
        self.assertEqual(first_audit, second_audit)
        self.assertTrue(first_audit.complete_coverage)
        self.assertTrue(first_audit.pairwise_disjoint)
        for name in first_partitions:
            np.testing.assert_array_equal(first_partitions[name], second_partitions[name])

    def test_partition_validation_rejects_final_test_leakage(self):
        labels = np.repeat(np.arange(2), 20)
        partitions, _ = build_stratified_episode_partitions(
            labels,
            development_indices=np.concatenate((np.arange(15), np.arange(20, 35))),
            final_test_indices=np.concatenate((np.arange(15, 20), np.arange(35, 40))),
            seed=7,
        )
        leaked = {name: indices.copy() for name, indices in partitions.items()}
        leaked["geometry"] = np.append(
            leaked["geometry"], leaked["final_test"][0],
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_episode_partitions(leaked, dataset_size=len(labels))


class GeometryFeasibilityTests(unittest.TestCase):
    def setUp(self):
        self.contract = GeometryCapacityContract(
            allowed_families=("sphere", "axis_aligned", "shrinkage", "full"),
            max_condition_number=1e4,
            max_parameter_sample_ratio=0.6,
            min_effective_rank=2.0,
        )

    def test_probe_rejects_full_geometry_when_class_support_is_too_small(self):
        rng = np.random.default_rng(17)
        class_zero = rng.normal(size=(100, 4))
        class_one = rng.normal(loc=2.0, size=(10, 4))
        report = evaluate_geometry_feasibility(
            np.vstack([class_zero, class_one]),
            np.concatenate([np.zeros(100, dtype=int), np.ones(10, dtype=int)]),
            self.contract,
        )

        self.assertTrue(report.supportable)
        sparse_class = report.classes[1]
        self.assertIn("sphere", sparse_class.eligible_families)
        self.assertNotIn("full", sparse_class.eligible_families)
        full = next(item for item in sparse_class.families if item.family == "full")
        self.assertIn("parameter_sample_ratio_exceeded", full.rejection_reasons)

    def test_probe_fails_closed_for_collapsed_representation(self):
        features = np.zeros((20, 4), dtype=np.float64)
        labels = np.repeat([0, 1], 10)
        report = evaluate_geometry_feasibility(features, labels, self.contract)

        self.assertFalse(report.supportable)
        self.assertEqual(report.unsupported_classes, (0, 1))
        with self.assertRaisesRegex(ValueError, "no authorized geometry family"):
            report.require_supportable()

    def test_probe_rejects_nonfinite_features(self):
        features = np.asarray([[0.0, 1.0], [np.nan, 2.0]])
        with self.assertRaisesRegex(ValueError, "finite"):
            evaluate_geometry_feasibility(features, np.asarray([0, 0]), self.contract)


class LocalArtifactStoreTests(unittest.TestCase):
    def test_stage_commit_is_verified_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(directory)
            calls = []

            def write_outputs(path: Path) -> None:
                calls.append(path)
                (path / "features.bin").write_bytes(b"immutable features")

            first = store.commit_stage(
                "run-1",
                "attempt-1",
                "features",
                write_outputs,
                state=LifecycleState.FEATURES_READY,
                input_hashes={"split": "split-hash"},
            )
            second = store.commit_stage(
                "run-1",
                "attempt-1",
                "features",
                write_outputs,
                state=LifecycleState.FEATURES_READY,
                input_hashes={"split": "split-hash"},
            )

            self.assertEqual(first, second)
            self.assertEqual(len(calls), 1)
            self.assertTrue(store.is_committed("run-1", "attempt-1", "features"))
            self.assertEqual(
                store.read_stage("run-1", "attempt-1", "features"), first,
            )

    def test_writer_failure_leaves_no_readable_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(directory)

            def fail_after_output(path: Path) -> None:
                (path / "partial.bin").write_bytes(b"incomplete")
                raise RuntimeError("injected failure")

            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                store.commit_stage(
                    "run-1",
                    "attempt-1",
                    "features",
                    fail_after_output,
                    state=LifecycleState.FEATURES_READY,
                )
            self.assertFalse(store.is_committed("run-1", "attempt-1", "features"))
            with self.assertRaises(FileNotFoundError):
                store.read_stage("run-1", "attempt-1", "features")

    def test_corruption_and_mismatched_retry_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(directory)
            store.commit_stage(
                "run-1",
                "attempt-1",
                "features",
                lambda path: (path / "features.bin").write_bytes(b"original"),
                state=LifecycleState.FEATURES_READY,
                input_hashes={"split": "split-hash"},
            )
            with self.assertRaisesRegex(ValueError, "retry inputs"):
                store.commit_stage(
                    "run-1",
                    "attempt-1",
                    "features",
                    lambda path: None,
                    state=LifecycleState.FEATURES_READY,
                    input_hashes={"split": "different-hash"},
                )

            stage_path = store.stage_path("run-1", "attempt-1", "features")
            (stage_path / "features.bin").write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "output hashes"):
                store.read_stage("run-1", "attempt-1", "features")


class LocalExecutorTests(unittest.TestCase):
    def test_failure_status_resume_and_corruption_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(directory)
            executor = LocalExecutor(store)
            calls = []

            def write_features(path: Path) -> None:
                calls.append("features")
                (path / "features.bin").write_bytes(b"features")

            def fail_geometry(path: Path) -> None:
                calls.append("geometry-failed")
                (path / "geometry.bin").write_bytes(b"partial")
                raise RuntimeError("injected geometry failure")

            stages = (
                StageSpec(
                    "features",
                    LifecycleState.FEATURES_READY,
                    write_features,
                    {"dataset": "dataset-hash"},
                ),
                StageSpec(
                    "geometry",
                    LifecycleState.GEOMETRY_READY,
                    fail_geometry,
                    {"features": "feature-hash"},
                ),
            )
            with self.assertRaisesRegex(RuntimeError, "injected geometry failure"):
                executor.run("run-1", "attempt-1", stages)

            interrupted = executor.status(
                "run-1", "attempt-1", ("features", "geometry", "assembly")
            )
            self.assertEqual(
                [stage.status for stage in interrupted.stages],
                [
                    StageExecutionStatus.COMMITTED,
                    StageExecutionStatus.PARTIAL,
                    StageExecutionStatus.PENDING,
                ],
            )

            resumed_stages = (
                stages[0],
                StageSpec(
                    "geometry",
                    LifecycleState.GEOMETRY_READY,
                    lambda path: (path / "geometry.bin").write_bytes(b"geometry"),
                    {"features": "feature-hash"},
                ),
            )
            executions = executor.resume("run-1", "attempt-1", resumed_stages)
            self.assertTrue(executions[0].reused)
            self.assertFalse(executions[1].reused)
            self.assertEqual(calls.count("features"), 1)
            self.assertTrue(executor.status(
                "run-1", "attempt-1", ("features", "geometry")
            ).complete)

            geometry_path = store.stage_path("run-1", "attempt-1", "geometry")
            (geometry_path / "geometry.bin").write_bytes(b"corrupt")
            corrupt = executor.status("run-1", "attempt-1", ("geometry",))
            self.assertEqual(
                corrupt.stages[0].status,
                StageExecutionStatus.CORRUPT,
            )


class MetricLedgerTests(unittest.TestCase):
    @staticmethod
    def _event(event_id: str = "event-1", value: float = 0.5) -> MetricEvent:
        return MetricEvent(
            event_id=event_id,
            run_id="run-1",
            attempt_id="attempt-1",
            stage_name="representation",
            split="validation",
            metric_name="loss",
            value=value,
            sample_count=100,
            created_at="2026-07-25T00:00:00Z",
            epoch=1,
            global_step=10,
            namespace="selection",
        )

    def test_metric_append_round_trip_and_idempotent_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = MetricLedger(Path(directory) / "metrics.jsonl")
            event = self._event()

            self.assertTrue(ledger.append(event))
            self.assertFalse(ledger.append(event))
            self.assertEqual(ledger.read_events(), [event])

    def test_metric_ledger_rejects_conflicts_and_corrupt_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            ledger = MetricLedger(path)
            ledger.append(self._event())
            with self.assertRaisesRegex(ValueError, "conflicting content"):
                ledger.append(self._event(value=0.6))
            with self.assertRaisesRegex(ValueError, "logical key"):
                ledger.append(self._event(event_id="event-2"))

            with path.open("a", encoding="utf-8") as stream:
                stream.write('{"partial":')
            with self.assertRaisesRegex(ValueError, "invalid metric event"):
                ledger.read_events()

    def test_history_export_is_deterministic_and_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = MetricLedger(root / "metrics.jsonl")
            ledger.append(self._event("event-1", 0.5))
            ledger.append(MetricEvent(
                event_id="event-2",
                run_id="run-1",
                attempt_id="attempt-2",
                stage_name="representation",
                split="train",
                metric_name="loss",
                value=0.4,
                sample_count=100,
                created_at="2026-07-25T00:01:00Z",
                epoch=2,
                global_step=20,
                namespace="exploratory",
            ))

            first_hashes = export_metric_history(ledger, root / "export-1")
            second_hashes = export_metric_history(ledger, root / "export-2")
            summary = json.loads(
                (root / "export-1" / "history.json").read_text(encoding="utf-8")
            )

            self.assertEqual(first_hashes, second_hashes)
            self.assertEqual(set(first_hashes), {
                "dashboard.html", "events.csv", "history.json",
            })
            self.assertEqual(summary["event_count"], 2)
            self.assertEqual(summary["attempt_ids"], ["attempt-1", "attempt-2"])
            self.assertEqual(summary["namespace_counts"], {
                "exploratory": 1, "final": 0, "selection": 1,
            })
            self.assertEqual(len(summary["series"]), 2)


class ModelBundleTests(unittest.TestCase):
    @staticmethod
    def _provenance():
        return BundleProvenance(
            routing_mode="exhaustive",
            semantic_router_cache_version="cache-v1",
            training_manifest_hash="1" * 64,
            evaluation_manifest_hash="2" * 64,
            metric_summary_hash="3" * 64,
            software_compatibility=">=1.0,<2.0",
            environment_fingerprint="environment-v1",
            created_at="2026-07-26T00:00:00Z",
            created_by="e3-qualification",
        )

    def _node(
        self,
        *,
        name="source",
        classes=(0, 1),
        transform="transform-v1",
        task="source-task",
        upstream=(),
        input_dim=2,
    ):
        fingerprint = ModelFingerprint(
            task_name=task,
            input_spec=InputSpec(
                "passthrough" if not upstream else "sdf_scores",
                ("source-task",) if upstream else (),
                input_dim,
            ),
            output_spec=OutputSpec("sdf_scores", classes),
        )
        profile = SupportProfile(
            model_signature=fingerprint.signature,
            feature_transform_fingerprint=transform,
            training_dataset_fingerprint="train-v1",
            calibration_dataset_fingerprint="calibration-v1",
            class_ids=classes,
            score_scales=tuple(1.0 for _ in classes),
            novelty_score="minimum_sdf",
            global_threshold=0.5,
            version="support-v1",
            fit_seed=7,
            created_at="2026-07-26T00:00:00Z",
        )
        return BundleNode(
            name=name,
            artifact_path=f"{name}.bin",
            fingerprint=fingerprint,
            class_order=classes,
            feature_transform_fingerprint=transform,
            upstream=upstream,
            support_profile=profile,
        )

    def test_fingerprint_round_trip_preserves_role_and_class_order(self):
        fingerprint = self._node().fingerprint
        restored = ModelFingerprint.from_dict(fingerprint.to_dict())
        self.assertEqual(restored, fingerprint)
        self.assertEqual(restored.output_spec.classes, (0, 1))

    def test_bundle_publish_swap_and_rollback_are_hash_verified(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalModelBundleStore(temporary_directory)
            source = self._node()
            first = store.publish(
                {"source.bin": b"model-v1"}, [source], provenance=self._provenance(),
            )
            store.activate(first.bundle_id)

            replacement = self._node()
            assert_node_replacement(source, replacement)
            second = store.publish(
                {"source.bin": b"model-v2"},
                [replacement],
                provenance=self._provenance(),
                parent_bundle_id=first.bundle_id,
            )
            store.activate(second.bundle_id)

            self.assertNotEqual(first.bundle_id, second.bundle_id)
            self.assertEqual(store.current(), second)
            self.assertEqual(store.rollback(), first)
            self.assertEqual(store.current(), first)

    def test_replacement_rejects_role_and_transform_mismatch(self):
        existing = self._node()
        with self.assertRaisesRegex(ValueError, "role-compatible"):
            assert_node_replacement(existing, self._node(task="other-task"))
        with self.assertRaisesRegex(ValueError, "transform fingerprint"):
            assert_node_replacement(existing, self._node(transform="transform-v2"))

    def test_class_expansion_requires_coordinated_downstream_width(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalModelBundleStore(temporary_directory)
            source = self._node(classes=(0, 1, 2))
            stale_downstream = self._node(
                name="downstream", task="downstream-task", upstream=("source",),
                input_dim=2,
            )
            components = {"source.bin": b"source", "downstream.bin": b"downstream"}
            with self.assertRaisesRegex(ValueError, "input dimension"):
                store.publish(
                    components, [source, stale_downstream], provenance=self._provenance(),
                )

            migrated_downstream = self._node(
                name="downstream", task="downstream-task", upstream=("source",),
                input_dim=3,
            )
            manifest = store.publish(
                components, [source, migrated_downstream], provenance=self._provenance(),
            )
            self.assertEqual(len(manifest.nodes), 2)

    def test_corruption_refuses_activation_without_pointer_change(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalModelBundleStore(temporary_directory)
            first = store.publish(
                {"source.bin": b"model-v1"}, [self._node()],
                provenance=self._provenance(),
            )
            store.activate(first.bundle_id)
            second = store.publish(
                {"source.bin": b"model-v2"}, [self._node()],
                provenance=self._provenance(),
                parent_bundle_id=first.bundle_id,
            )
            component = (
                Path(temporary_directory) / "bundles" / second.bundle_id
                / "components" / "source.bin"
            )
            component.write_bytes(b"corrupted")

            with self.assertRaisesRegex(ValueError, "verification failed"):
                store.activate(second.bundle_id)
            self.assertEqual(store.current(), first)

    def test_confirmed_adaptation_gates_publication_and_exact_rollback(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalModelBundleStore(temporary_directory)
            parent = store.publish(
                {"source.bin": b"model-v1"}, [self._node()],
                provenance=self._provenance(),
            )
            store.activate(parent.bundle_id)
            confirmation = ReviewConfirmation(
                review_id="review-123456789abc",
                kind=ConfirmationKind.EXISTING_CLASS,
                confirmed_label="class-0",
                confirmed_at="2026-07-26T01:00:00Z",
            )
            policy = AdaptationPublicationPolicy(0.02)
            rejected = publish_confirmed_adaptation(
                store,
                review_id=confirmation.review_id,
                confirmation=confirmation,
                evidence=AdaptationPublicationEvidence(
                    True, 0.20, 0.25, (), True,
                ),
                policy=policy,
                components={"source.bin": b"model-v2"},
                nodes=[self._node()],
                provenance=self._provenance(),
                publish=True,
            )
            self.assertEqual(rejected.failed_gates, ("calibration_gate_failed",))
            self.assertEqual(store.current(), parent)
            dry_run = publish_confirmed_adaptation(
                store,
                review_id=confirmation.review_id,
                confirmation=confirmation,
                evidence=AdaptationPublicationEvidence(
                    True, 0.20, 0.21, (), True,
                ),
                policy=policy,
                components={"source.bin": b"model-v2"},
                nodes=[self._node()],
                provenance=self._provenance(),
                publish=False,
            )
            self.assertEqual(dry_run.status, "validated_dry_run")
            self.assertEqual(store.current(), parent)
            published = publish_confirmed_adaptation(
                store,
                review_id=confirmation.review_id,
                confirmation=confirmation,
                evidence=AdaptationPublicationEvidence(
                    True, 0.20, 0.21, (), True,
                ),
                policy=policy,
                components={"source.bin": b"model-v2"},
                nodes=[self._node()],
                provenance=self._provenance(),
                publish=True,
            )
            self.assertEqual(store.current().parent_bundle_id, parent.bundle_id)
            rolled_back = rollback_adaptation(store, published)
            self.assertEqual(rolled_back.rollback_bundle_id, parent.bundle_id)
            self.assertEqual(store.current(), parent)

    def test_replicated_service_recovers_bad_canary_and_coordinator_loss(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalModelBundleStore(temporary_directory)
            parent = store.publish(
                {"source.bin": b"1"}, [self._node()], provenance=self._provenance(),
            )
            store.activate(parent.bundle_id)
            child = store.publish(
                {"source.bin": b"-1"}, [self._node()],
                provenance=self._provenance(), parent_bundle_id=parent.bundle_id,
            )

            def loader(bundle_id):
                component = (
                    Path(temporary_directory) / "bundles" / bundle_id
                    / "components" / "source.bin"
                )
                direction = int(component.read_text(encoding="ascii"))
                return lambda values: (direction * values[:, 0] >= 0).astype(np.int32)

            service = ReplicatedBundleService(store, loader, replica_count=2)
            values = np.asarray([[-1.0, 0.0], [1.0, 0.0]])
            self.assertEqual(service.serving_bundle_ids, (parent.bundle_id,) * 2)
            shadow = service.shadow(child.bundle_id, values)
            self.assertEqual(shadow.agreement, 0.0)
            self.assertFalse(shadow.candidate_controls_outputs)
            coordinator = ProductionPromotionCoordinator(store)
            coordinator.promote_or_rollback(
                child.bundle_id, canary_gate_passed=False,
            )
            self.assertEqual(store.current(), parent)

            coordinator.begin_promotion(child.bundle_id)
            restarted = ProductionPromotionCoordinator(store)
            recovered = restarted.recover()
            self.assertEqual(recovered["production_bundle_id"], parent.bundle_id)
            self.assertEqual(store.current(), parent)
            self.assertFalse(any(store.root.glob("*.partial")))


class LocalCheckpointStoreTests(unittest.TestCase):
    def test_checkpoint_round_trip_and_latest_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            store = LocalCheckpointStore(LocalArtifactStore(directory))
            first = store.save(
                "run-1",
                "attempt-1",
                "representation",
                1,
                10,
                state={"learning_rate": 0.01, "rng": [1, 2, 3]},
                arrays={"weights": np.arange(6).reshape(2, 3)},
                input_hashes={"features": "features-hash"},
            )
            second = store.save(
                "run-1",
                "attempt-1",
                "representation",
                2,
                20,
                state={"learning_rate": 0.005, "rng": [4, 5, 6]},
                arrays={"weights": np.arange(6, 12).reshape(2, 3)},
                input_hashes={"features": "features-hash"},
            )

            latest = store.latest("run-1", "attempt-1", "representation")
            self.assertIsNotNone(latest)
            self.assertEqual(first.epoch, 1)
            self.assertEqual(second.epoch, 2)
            self.assertEqual(latest.metadata, second)
            self.assertEqual(latest.state["rng"], [4, 5, 6])
            np.testing.assert_array_equal(
                latest.arrays["weights"], np.arange(6, 12).reshape(2, 3),
            )

    def test_checkpoint_rejects_non_json_state_and_object_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            store = LocalCheckpointStore(LocalArtifactStore(directory))
            with self.assertRaises(TypeError):
                store.save(
                    "run-1", "attempt-1", "representation", 1, 10,
                    state={"unsupported": {1, 2}}, arrays={},
                )
            with self.assertRaisesRegex(ValueError, "object dtype"):
                store.save(
                    "run-1", "attempt-1", "representation", 1, 10,
                    state={}, arrays={"objects": np.asarray([object()])},
                )

    def test_checkpoint_retry_rejects_different_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            store = LocalCheckpointStore(LocalArtifactStore(directory))
            store.save(
                "run-1", "attempt-1", "representation", 1, 10,
                state={"epoch_loss": 0.5}, arrays={"weights": np.asarray([1.0])},
            )
            with self.assertRaisesRegex(ValueError, "retry inputs"):
                store.save(
                    "run-1", "attempt-1", "representation", 1, 10,
                    state={"epoch_loss": 0.4}, arrays={"weights": np.asarray([2.0])},
                )
            with self.assertRaisesRegex(ValueError, "stage_name"):
                store.latest("run-1", "attempt-1", "*")


class RefinementCheckpointAdapterTests(unittest.TestCase):
    @staticmethod
    def _models() -> dict[int, list[Expert]]:
        models = {}
        for class_id, center in ((0, -1.0), (1, 1.0)):
            expert = Expert(alpha=2.0)
            expert.add_ellipsoid(EllipsoidExpert([center], [0.7]))
            models[class_id] = [expert]
        return models

    @staticmethod
    def _train_epochs(
        optimizer: SDFOptimizer,
        rng: np.random.Generator,
        X: np.ndarray,
        y: np.ndarray,
        start_epoch: int,
        end_epoch: int,
        history: list[dict],
        global_step: int,
    ) -> int:
        for epoch in range(start_epoch, end_epoch):
            losses = []
            order = rng.permutation(len(X))
            for start in range(0, len(X), 4):
                indices = order[start:start + 4]
                losses.append(optimizer.step(X[indices], y[indices]))
                global_step += 1
            metrics = optimizer.evaluate(X, y)
            history.append({
                "epoch": epoch + 1,
                "batch_training_loss": float(np.mean(losses)),
                "train_loss": metrics["loss"],
                "train_error": metrics["error"],
            })
        return global_step

    def test_interrupted_refinement_restores_exact_training_state(self):
        X = np.asarray([
            [-1.4], [-1.1], [-0.8], [-0.5],
            [0.5], [0.8], [1.1], [1.4],
        ])
        y = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
        initial_models = self._models()

        uninterrupted_optimizer = SDFOptimizer(
            copy.deepcopy(initial_models), alpha=2.0, learning_rate=0.01,
            momentum=0.9, score_scales={0: 1.0, 1: 1.0},
        )
        uninterrupted_rng = np.random.default_rng(17)
        uninterrupted_history: list[dict] = []
        uninterrupted_steps = self._train_epochs(
            uninterrupted_optimizer, uninterrupted_rng, X, y,
            0, 4, uninterrupted_history, 0,
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_store = LocalCheckpointStore(LocalArtifactStore(directory))
            adapter = RefinementCheckpointAdapter(checkpoint_store)
            interrupted_optimizer = SDFOptimizer(
                copy.deepcopy(initial_models), alpha=2.0, learning_rate=0.01,
                momentum=0.9, score_scales={0: 1.0, 1: 1.0},
            )
            interrupted_rng = np.random.default_rng(17)
            interrupted_history: list[dict] = []
            interrupted_steps = self._train_epochs(
                interrupted_optimizer, interrupted_rng, X, y,
                0, 2, interrupted_history, 0,
            )
            adapter.save(
                "run-1", "attempt-1", "refinement", 2, interrupted_steps,
                optimizer=interrupted_optimizer,
                rng=interrupted_rng,
                epoch_history=interrupted_history,
                sampler_state={"epoch": 2},
                input_hashes={"features": "features-hash"},
            )

            resumed_optimizer = SDFOptimizer(
                copy.deepcopy(initial_models), alpha=1.0, learning_rate=1.0,
                momentum=0.0, score_scales=None,
            )
            resumed_rng = np.random.default_rng(999)
            restored = adapter.restore_latest(
                "run-1", "attempt-1", "refinement",
                optimizer=resumed_optimizer,
                rng=resumed_rng,
            )
            self.assertIsNotNone(restored)
            resumed_history = list(restored.epoch_history)
            resumed_steps = self._train_epochs(
                resumed_optimizer, resumed_rng, X, y,
                restored.metadata.epoch, 4, resumed_history,
                restored.metadata.global_step,
            )

        self.assertEqual(resumed_steps, uninterrupted_steps)
        self.assertEqual(resumed_history, uninterrupted_history)
        expected_state, expected_arrays = uninterrupted_optimizer.export_state()
        actual_state, actual_arrays = resumed_optimizer.export_state()
        self.assertEqual(actual_state, expected_state)
        self.assertEqual(set(actual_arrays), set(expected_arrays))
        for name in expected_arrays:
            np.testing.assert_array_equal(actual_arrays[name], expected_arrays[name])

    def test_restore_rejects_topology_and_rng_mismatches(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = RefinementCheckpointAdapter(
                LocalCheckpointStore(LocalArtifactStore(directory))
            )
            optimizer = SDFOptimizer(self._models(), alpha=2.0)
            adapter.save(
                "run-1", "attempt-1", "refinement", 0, 0,
                optimizer=optimizer,
                rng=np.random.default_rng(17),
                epoch_history=[],
                sampler_state={"epoch": 0},
            )

            mismatched_models = self._models()
            del mismatched_models[1]
            with self.assertRaisesRegex(ValueError, "topology"):
                adapter.restore_latest(
                    "run-1", "attempt-1", "refinement",
                    optimizer=SDFOptimizer(mismatched_models),
                    rng=np.random.default_rng(17),
                )

            with self.assertRaisesRegex(ValueError, "bit generator"):
                adapter.restore_latest(
                    "run-1", "attempt-1", "refinement",
                    optimizer=SDFOptimizer(self._models()),
                    rng=np.random.Generator(np.random.MT19937(17)),
                )


class RepresentationCheckpointAdapterTests(unittest.TestCase):
    @staticmethod
    def _train_epochs(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        rng: np.random.Generator,
        X: torch.Tensor,
        y: torch.Tensor,
        start_epoch: int,
        end_epoch: int,
        history: list[dict],
        global_step: int,
    ) -> int:
        for epoch in range(start_epoch, end_epoch):
            losses = []
            for start in range(0, len(X), 3):
                indices = torch.from_numpy(rng.permutation(len(X))[start:start + 3])
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(X[indices]), y[indices])
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
                global_step += 1
            scheduler.step()
            history.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})
        return global_step

    @staticmethod
    def _objects(seed: int):
        torch.manual_seed(seed)
        model = torch.nn.Sequential(
            torch.nn.Linear(3, 5),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(5, 2),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
        return model, optimizer, scheduler

    def test_interrupted_representation_restores_exact_training_state(self):
        X = torch.tensor([
            [-1.0, -0.5, 0.2], [-0.8, -0.2, 0.1], [-0.4, -0.6, 0.3],
            [0.4, 0.6, -0.3], [0.8, 0.2, -0.1], [1.0, 0.5, -0.2],
        ])
        y = torch.tensor([0, 0, 0, 1, 1, 1])
        full_model, full_optimizer, full_scheduler = self._objects(23)
        full_rng = np.random.default_rng(41)
        full_history: list[dict] = []
        full_steps = self._train_epochs(
            full_model, full_optimizer, full_scheduler, full_rng,
            X, y, 0, 4, full_history, 0,
        )

        with tempfile.TemporaryDirectory() as directory:
            adapter = RepresentationCheckpointAdapter(
                LocalCheckpointStore(LocalArtifactStore(directory))
            )
            model, optimizer, scheduler = self._objects(23)
            rng = np.random.default_rng(41)
            history: list[dict] = []
            steps = self._train_epochs(
                model, optimizer, scheduler, rng, X, y, 0, 2, history, 0,
            )
            adapter.save(
                "run-1", "attempt-1", "representation", 2, steps,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                rng=rng,
                epoch_history=history,
                sampler_state={"epoch": 2},
                input_hashes={"training_data": "fixed-data-v1"},
            )

            resumed_model, resumed_optimizer, resumed_scheduler = self._objects(999)
            resumed_rng = np.random.default_rng(999)
            restored = adapter.restore_latest(
                "run-1", "attempt-1", "representation",
                model=resumed_model,
                optimizer=resumed_optimizer,
                scheduler=resumed_scheduler,
                rng=resumed_rng,
            )
            self.assertIsNotNone(restored)
            resumed_history = list(restored.epoch_history)
            resumed_steps = self._train_epochs(
                resumed_model, resumed_optimizer, resumed_scheduler, resumed_rng,
                X, y, restored.metadata.epoch, 4, resumed_history,
                restored.metadata.global_step,
            )

        self.assertEqual(resumed_steps, full_steps)
        self.assertEqual(resumed_history, full_history)
        self.assertEqual(resumed_scheduler.state_dict(), full_scheduler.state_dict())
        for name, expected in full_model.state_dict().items():
            torch.testing.assert_close(resumed_model.state_dict()[name], expected, rtol=0, atol=0)
        expected_optimizer = full_optimizer.state_dict()
        actual_optimizer = resumed_optimizer.state_dict()
        self.assertEqual(actual_optimizer["param_groups"], expected_optimizer["param_groups"])
        for parameter_id, expected_state in expected_optimizer["state"].items():
            for name, expected in expected_state.items():
                torch.testing.assert_close(
                    actual_optimizer["state"][parameter_id][name], expected,
                    rtol=0, atol=0,
                )

    def test_production_training_loop_resumes_exactly_after_checkpoint(self):
        X = torch.tensor([
            [-1.0, -0.5, 0.2], [-0.8, -0.2, 0.1], [-0.4, -0.6, 0.3],
            [0.4, 0.6, -0.3], [0.8, 0.2, -0.1], [1.0, 0.5, -0.2],
        ])
        y = torch.tensor([0, 0, 0, 1, 1, 1])
        full_model, full_optimizer, full_scheduler = self._objects(29)
        full_result = train_representation(
            full_model,
            full_optimizer,
            X,
            y,
            epochs=4,
            batch_size=3,
            rng=np.random.default_rng(43),
            scheduler=full_scheduler,
        )

        with tempfile.TemporaryDirectory() as directory:
            adapter = RepresentationCheckpointAdapter(
                LocalCheckpointStore(LocalArtifactStore(directory))
            )
            interrupted_model, interrupted_optimizer, interrupted_scheduler = (
                self._objects(29)
            )
            with self.assertRaisesRegex(RuntimeError, "epoch 2"):
                train_representation(
                    interrupted_model,
                    interrupted_optimizer,
                    X,
                    y,
                    epochs=4,
                    batch_size=3,
                    rng=np.random.default_rng(43),
                    scheduler=interrupted_scheduler,
                    checkpoint_adapter=adapter,
                    checkpoint_input_hashes={"training_data": "fixed-data-v1"},
                    fail_after_epoch=2,
                )

            resumed_model, resumed_optimizer, resumed_scheduler = self._objects(999)
            resumed_result = train_representation(
                resumed_model,
                resumed_optimizer,
                X,
                y,
                epochs=4,
                batch_size=3,
                rng=np.random.default_rng(999),
                scheduler=resumed_scheduler,
                checkpoint_adapter=adapter,
                checkpoint_input_hashes={"training_data": "fixed-data-v1"},
            )
            checkpoints = adapter.store.list_checkpoints(
                "representation-run", "attempt-1", "representation",
            )

        self.assertEqual(resumed_result.resumed_from_epoch, 2)
        self.assertEqual(resumed_result.global_step, full_result.global_step)
        self.assertEqual(resumed_result.epoch_history, full_result.epoch_history)
        self.assertEqual(len(checkpoints), 4)
        self.assertEqual(resumed_scheduler.state_dict(), full_scheduler.state_dict())
        for name, expected in full_model.state_dict().items():
            torch.testing.assert_close(
                resumed_model.state_dict()[name], expected, rtol=0, atol=0,
            )
        expected_optimizer = full_optimizer.state_dict()
        actual_optimizer = resumed_optimizer.state_dict()
        self.assertEqual(actual_optimizer["param_groups"], expected_optimizer["param_groups"])
        for parameter_id, expected_state in expected_optimizer["state"].items():
            for name, expected in expected_state.items():
                torch.testing.assert_close(
                    actual_optimizer["state"][parameter_id][name], expected,
                    rtol=0, atol=0,
                )

    def test_distributed_sampler_resume_preserves_rank_shard_exactly(self):
        X = torch.tensor([
            [-1.0, -0.5, 0.2], [-0.8, -0.2, 0.1], [-0.4, -0.6, 0.3],
            [-0.2, -0.1, 0.4], [0.2, 0.1, -0.4], [0.4, 0.6, -0.3],
            [0.8, 0.2, -0.1], [1.0, 0.5, -0.2],
        ])
        y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

        def sampler(rank: int) -> DistributedSampler:
            return DistributedSampler(
                X,
                num_replicas=2,
                rank=rank,
                shuffle=True,
                seed=47,
                drop_last=False,
            )

        full_model, full_optimizer, full_scheduler = self._objects(31)
        full_result = train_representation(
            full_model,
            full_optimizer,
            X,
            y,
            epochs=4,
            batch_size=2,
            rng=np.random.default_rng(53),
            scheduler=full_scheduler,
            sampler=sampler(1),
        )

        with tempfile.TemporaryDirectory() as directory:
            adapter = RepresentationCheckpointAdapter(
                LocalCheckpointStore(LocalArtifactStore(directory))
            )
            model, optimizer, scheduler = self._objects(31)
            with self.assertRaisesRegex(RuntimeError, "epoch 2"):
                train_representation(
                    model,
                    optimizer,
                    X,
                    y,
                    epochs=4,
                    batch_size=2,
                    rng=np.random.default_rng(53),
                    scheduler=scheduler,
                    sampler=sampler(1),
                    checkpoint_adapter=adapter,
                    fail_after_epoch=2,
                )

            mismatched_model, mismatched_optimizer, mismatched_scheduler = (
                self._objects(999)
            )
            with self.assertRaisesRegex(ValueError, "sampler state"):
                train_representation(
                    mismatched_model,
                    mismatched_optimizer,
                    X,
                    y,
                    epochs=4,
                    batch_size=2,
                    rng=np.random.default_rng(999),
                    scheduler=mismatched_scheduler,
                    sampler=sampler(0),
                    checkpoint_adapter=adapter,
                )

            resumed_model, resumed_optimizer, resumed_scheduler = self._objects(999)
            resumed_result = train_representation(
                resumed_model,
                resumed_optimizer,
                X,
                y,
                epochs=4,
                batch_size=2,
                rng=np.random.default_rng(999),
                scheduler=resumed_scheduler,
                sampler=sampler(1),
                checkpoint_adapter=adapter,
            )

        self.assertEqual(resumed_result.epoch_history, full_result.epoch_history)
        self.assertEqual(resumed_result.global_step, full_result.global_step)
        self.assertEqual(resumed_scheduler.state_dict(), full_scheduler.state_dict())
        for name, expected in full_model.state_dict().items():
            torch.testing.assert_close(
                resumed_model.state_dict()[name], expected, rtol=0, atol=0,
            )

    def test_restore_rejects_representation_topology_and_rng_mismatches(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = RepresentationCheckpointAdapter(
                LocalCheckpointStore(LocalArtifactStore(directory))
            )
            model, optimizer, scheduler = self._objects(23)
            adapter.save(
                "run-1", "attempt-1", "representation", 0, 0,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                rng=np.random.default_rng(41),
                epoch_history=[],
                sampler_state={"epoch": 0},
            )

            mismatched_model = torch.nn.Linear(3, 2)
            with self.assertRaisesRegex(ValueError, "model topology"):
                adapter.restore_latest(
                    "run-1", "attempt-1", "representation",
                    model=mismatched_model,
                    optimizer=torch.optim.Adam(mismatched_model.parameters()),
                    scheduler=torch.optim.lr_scheduler.StepLR(
                        torch.optim.Adam(mismatched_model.parameters()), step_size=1,
                    ),
                    rng=np.random.default_rng(41),
                )

            resumed_model, resumed_optimizer, resumed_scheduler = self._objects(23)
            with self.assertRaisesRegex(ValueError, "bit generator"):
                adapter.restore_latest(
                    "run-1", "attempt-1", "representation",
                    model=resumed_model,
                    optimizer=resumed_optimizer,
                    scheduler=resumed_scheduler,
                    rng=np.random.Generator(np.random.MT19937(41)),
                )


class DomainNetRuntimeTests(unittest.TestCase):
    def _manifest(self, root: Path) -> Path:
        shards = []
        for domain in DOMAINNET_DOMAINS:
            for split in ("train", "test"):
                relative = f"{domain}/{split}.npz"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = f"{domain}:{split}".encode("ascii")
                path.write_bytes(payload)
                shards.append({
                    "domain": domain,
                    "split": split,
                    "path": relative,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "samples": 1,
                })
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({
            "schema_version": 1,
            "version": "test-v1",
            "class_count": 345,
            "shards": shards,
        }), encoding="utf-8")
        return manifest

    def test_domainnet_manifest_verifies_all_domains_and_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = DomainNetManifest.load(self._manifest(root)).verify(root)
            self.assertEqual(report["class_count"], 345)
            self.assertEqual(len(report["verified_shards"]), 12)

    def test_domainnet_manifest_rejects_tampered_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = DomainNetManifest.load(self._manifest(root))
            (root / "real" / "test.npz").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                manifest.verify(root)

    def test_domainnet_hub_manifest_round_trips_mixed_parquet_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b"mixed-domain-parquet"
            relative = "data/train.parquet"
            path = root / relative
            path.parent.mkdir(parents=True)
            path.write_bytes(payload)
            manifest = DomainNetManifest(
                files=(DomainNetFile(
                    relative, hashlib.sha256(payload).hexdigest(), len(payload),
                ),),
                class_count=345,
                version="hub-test-v1",
                source_repository="owner/domainnet",
                source_revision="abc123",
                split_samples=(("train", 10), ("test", 5)),
            )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
            loaded = DomainNetManifest.load(manifest_path)
            self.assertEqual(loaded, manifest)
            self.assertEqual(loaded.verify(root)["total_verified_bytes"], len(payload))

    @unittest.skipIf(importlib.util.find_spec("ray") is not None, "Ray is installed")
    def test_ray_executor_fails_explicitly_when_dependency_is_missing(self):
        with self.assertRaisesRegex(RayUnavailableError, "Python <3.14"):
            RayExecutor()

    def test_local_cluster_evidence_cannot_pass_multihost_gate(self):
        local = DistributedQualificationEvidence(
            scope="local_simulation",
            logical_nodes=3,
            executing_nodes=3,
            physical_hosts=1,
            task_retry_passed=True,
            worker_process_loss_recovered=True,
            worker_node_loss_recovered=False,
            complete_histories=True,
            artifact_identity_verified=True,
        ).evaluate()
        self.assertTrue(local["local_simulation_gate_passed"])
        self.assertFalse(local["multihost_gate_passed"])
        self.assertFalse(local["e7_gate_passed"])

        multihost = DistributedQualificationEvidence(
            scope="multihost",
            logical_nodes=3,
            executing_nodes=3,
            physical_hosts=3,
            task_retry_passed=True,
            worker_process_loss_recovered=True,
            worker_node_loss_recovered=True,
            complete_histories=True,
            artifact_identity_verified=True,
        ).evaluate()
        self.assertTrue(multihost["e7_gate_passed"])


class ModelNetRuntimeTests(unittest.TestCase):
    def test_modelnet40_manifest_verifies_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_payload = b"source-parquet"
            artifact_payload = b"derived-npz"
            (root / "repository").mkdir()
            (root / "repository" / "train.parquet").write_bytes(source_payload)
            (root / "modelnet40_2048.npz").write_bytes(artifact_payload)
            manifest = ModelNet40Manifest(
                source_repository="owner/modelnet40",
                source_revision="abc123",
                source_files=(ModelNetFile(
                    "repository/train.parquet",
                    hashlib.sha256(source_payload).hexdigest(),
                    len(source_payload),
                ),),
                artifact=ModelNetFile(
                    "modelnet40_2048.npz",
                    hashlib.sha256(artifact_payload).hexdigest(),
                    len(artifact_payload),
                ),
                split_samples=(("train", 10), ("test", 5)),
            )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
            loaded = ModelNet40Manifest.load(manifest_path)
            self.assertEqual(loaded.verify(root)["total_samples"], 15)
            (root / "modelnet40_2048.npz").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                loaded.verify(root)


if __name__ == "__main__":
    unittest.main()