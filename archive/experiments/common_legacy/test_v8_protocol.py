from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.common.v8_protocol import (
    assert_review_budget,
    assert_rollback_parent,
    assert_threshold_lineage,
    build_episode_contracts,
    endpoint_from_config,
    require_confirmation,
    synthetic_episode_replay,
    validate_m45_config,
)
from experiments.tier1.eval_v8_m45_lock import DEFAULT_CONFIG, verify_m45
from src.runtime.schemas import EpisodeReplayContract, InterfaceContractAudit


class V8ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        self.digest = hashlib.sha256(b"fixture").hexdigest()

    def test_registered_config_and_endpoint_are_valid(self):
        validate_m45_config(self.config)
        endpoint = endpoint_from_config(self.config)
        self.assertEqual(endpoint.review_budget, 50)
        self.assertAlmostEqual(endpoint.utility(0.65, 0.72), 0.07)

    def test_partition_leakage_fails_closed(self):
        contract = build_episode_contracts(self.config, self.digest)[0]
        partitions = list(contract.partition_hashes)
        partitions[1] = (partitions[1][0], partitions[0][1])
        with self.assertRaises(ValueError):
            EpisodeReplayContract(
                **{
                    **contract.__dict__,
                    "partition_hashes": tuple(partitions),
                }
            )

    def test_stale_class_order_fails_closed(self):
        contract = build_episode_contracts(self.config, self.digest)[0]
        with self.assertRaises(ValueError):
            EpisodeReplayContract(
                **{
                    **contract.__dict__,
                    "child_class_order": (*contract.parent_class_order, "wrong"),
                }
            )

    def test_threshold_lineage_mismatch_fails_closed(self):
        with self.assertRaises(ValueError):
            assert_threshold_lineage("threshold-v1", "threshold-v2")

    def test_review_budget_overflow_fails_closed(self):
        with self.assertRaises(ValueError):
            assert_review_budget(tuple(f"sample-{index}" for index in range(51)), 50)

    def test_missing_confirmation_fails_closed(self):
        with self.assertRaises(PermissionError):
            require_confirmation(None)

    def test_rollback_parent_drift_fails_closed(self):
        with self.assertRaises(ValueError):
            assert_rollback_parent("bundle-parent", "bundle-other")

    def test_interface_audit_exposes_missing_statistics(self):
        audit = InterfaceContractAudit(
            interface_name="clusterer_to_review",
            producer_schema="clusterer_v1",
            consumer_schema="review_v1",
            producer_artifact_hash=self.digest,
            required_statistics=("core_member_ids", "boundary_member_ids"),
            supplied_statistics=("core_member_ids",),
            unsupported_diagnostics=("boundary_member_ids",),
            class_order_version="v8-order",
            calibration_version="v8-calibration",
        )
        self.assertFalse(audit.complete)
        self.assertEqual(audit.missing_statistics, ("boundary_member_ids",))

    def test_sealed_config_fails_closed(self):
        changed = copy.deepcopy(self.config)
        changed["sealed_data"]["final_labels_opened"] = True
        with self.assertRaises(PermissionError):
            validate_m45_config(changed)

    def test_synthetic_replay_is_deterministic(self):
        self.assertEqual(
            synthetic_episode_replay(self.config),
            synthetic_episode_replay(self.config),
        )

    def test_full_m45_replay_is_byte_identical_and_data_sealed(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = verify_m45(DEFAULT_CONFIG, Path(directory) / "output")
        self.assertTrue(summary["byte_identical_replay"])
        self.assertTrue(summary["all_interfaces_complete"])
        self.assertEqual(summary["registered_failure_cases"], 6)
        self.assertFalse(summary["training_data_loaded"])
        self.assertFalse(summary["final_labels_opened"])


if __name__ == "__main__":
    unittest.main()
