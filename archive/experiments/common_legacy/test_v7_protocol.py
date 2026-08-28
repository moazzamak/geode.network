from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.common.v5_artifacts import sha256_file
from experiments.common.v7_protocol import (
    AcceptanceHeadSpec,
    ConfirmationEvent,
    EmpiricalRoutingProfile,
    GraphMigrationSpec,
    ReviewEvent,
    schedule_locks,
    synthetic_contract_fixture,
    validate_parent_locks,
    validate_v7_m38_config,
)
from experiments.tier1.eval_v7_m38_lock import DEFAULT_CONFIG, verify_m38


class V7ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))

    def test_registered_config_is_valid_and_sealed(self):
        validate_v7_m38_config(self.config)
        changed = copy.deepcopy(self.config)
        changed["final_labels_opened"] = True
        with self.assertRaises(PermissionError):
            validate_v7_m38_config(changed)

    def test_stale_routing_policy_fails_closed(self):
        changed = copy.deepcopy(self.config)
        changed["routing_policy"]["stale_profile_action"] = "continue"
        with self.assertRaises(ValueError):
            validate_v7_m38_config(changed)

    def test_parent_hash_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "parent.json"
            parent.write_text("{}\n", encoding="utf-8")
            locks = [{"id": "parent", "path": parent.name, "sha256": sha256_file(parent)}]
            self.assertEqual(len(validate_parent_locks(locks, root)), 1)
            parent.write_text('{"drift":true}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_parent_locks(locks, root)

    def test_acceptance_head_rejects_duplicate_class_order(self):
        digest = hashlib.sha256(b"representation").hexdigest()
        with self.assertRaises(ValueError):
            AcceptanceHeadSpec(
                "knn",
                digest,
                ("a", "a"),
                "higher_is_novel",
                "fixed",
                "unsupported",
            )

    def test_routing_profile_rejects_bad_lineage(self):
        digest = hashlib.sha256(b"lineage").hexdigest()
        with self.assertRaises(ValueError):
            EmpiricalRoutingProfile(
                "model",
                "bad",
                ("a",),
                "centroid",
                digest,
                digest,
                "higher_is_match",
                0.0,
                2,
            )

    def test_review_and_confirmation_require_semantic_boundary(self):
        review = ReviewEvent("review-a", ("sample-a",), 1, "review_requested")
        self.assertEqual(review.state, "review_requested")
        with self.assertRaises(ValueError):
            ConfirmationEvent(review.review_id, "new_class", None, 2)

    def test_graph_migration_requires_exact_rollback_parent(self):
        parent = hashlib.sha256(b"parent").hexdigest()
        other = hashlib.sha256(b"other").hexdigest()
        with self.assertRaises(ValueError):
            GraphMigrationSpec(
                parent,
                ("a",),
                ("a", "b"),
                "review-a",
                "confirmation-a",
                other,
            )

    def test_schedule_locks_are_ordered_and_deterministic(self):
        first = schedule_locks(self.config["schedules"])
        second = schedule_locks(self.config["schedules"])
        self.assertEqual(first, second)
        self.assertEqual([item["id"] for item in first], [
            "synthetic_causal",
            "cifar100_class_stream",
            "domainnet_bundle_routing",
        ])

    def test_synthetic_contract_fixture_is_deterministic(self):
        self.assertEqual(synthetic_contract_fixture(), synthetic_contract_fixture())

    def test_full_m38_replay_is_byte_identical_and_data_sealed(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = verify_m38(DEFAULT_CONFIG, Path(directory) / "output")
            self.assertTrue(summary["byte_identical_replay"])
            self.assertFalse(summary["training_data_loaded"])
            self.assertFalse(summary["final_labels_opened"])
            self.assertFalse(summary["outcome_e_triggered"])


if __name__ == "__main__":
    unittest.main()
