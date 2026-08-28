"""Unit tests for M205: the DNN-component admission validator."""
from __future__ import annotations

import unittest

from geode.core.dnn_admission import (
    AdmissionRegistry,
    DNNSubmission,
    validate_submission,
)


def _good_submission() -> DNNSubmission:
    return DNNSubmission(
        architecture_hash="a" * 64,
        seed_hash="b" * 64,
        data_digest="c" * 64,
        software_hash="d" * 64,
        weights_hash="e" * 64,
        training_log_digest="f" * 64,
        eval_report={"split": "test", "n_test": 2000,
                     "accuracy": 0.31},
    )


class TestValidate(unittest.TestCase):

    def test_complete_submission_admitted_with_replay_hash(self):
        sub = _good_submission()
        result = validate_submission(sub)
        self.assertTrue(result.admitted, result.reasons)
        self.assertEqual(result.replay_hash, sub.replay_hash())
        self.assertEqual(len(result.replay_hash), 64)

    def test_every_hash_field_is_checked(self):
        for field_name in ("architecture_hash", "seed_hash",
                           "data_digest", "software_hash", "weights_hash",
                           "training_log_digest"):
            sub = _good_submission()
            setattr(sub, field_name, "not-a-hash")
            result = validate_submission(sub)
            self.assertFalse(result.admitted)
            self.assertTrue(any(field_name in r for r in result.reasons))

    def test_short_hash_rejected(self):
        sub = _good_submission()
        sub.architecture_hash = "ab12"
        result = validate_submission(sub)
        self.assertFalse(result.admitted)

    def test_eval_report_rules(self):
        sub = _good_submission()
        sub.eval_report = {"split": "train", "n_test": 2000,
                           "accuracy": 0.5}
        self.assertFalse(validate_submission(sub).admitted)
        sub.eval_report = {"split": "test", "n_test": 10,
                           "accuracy": 0.5}
        self.assertFalse(validate_submission(sub).admitted)
        sub.eval_report = {"split": "test", "n_test": 2000,
                           "accuracy": 1.7}
        self.assertFalse(validate_submission(sub).admitted)
        sub.eval_report = "garbage"
        self.assertFalse(validate_submission(sub).admitted)

    def test_chance_floor_rejection(self):
        # a chance-level artifact carries no signal (registered rule)
        sub = _good_submission()
        sub.eval_report = {"split": "test", "n_test": 2000,
                           "accuracy": 1.0 / 345}
        result = validate_submission(sub)
        self.assertFalse(result.admitted)
        self.assertTrue(any("chance floor" in r for r in result.reasons))


class TestAdmissionRegistry(unittest.TestCase):

    def test_first_admission_accepted_then_duplicate_rejected(self):
        registry = AdmissionRegistry()
        sub = _good_submission()
        first = registry.admit(sub)
        self.assertTrue(first.admitted, first.reasons)
        self.assertEqual(registry.admitted_count(), 1)
        second = registry.admit(_good_submission())
        self.assertFalse(second.admitted)
        self.assertTrue(second.duplicate)
        self.assertIn("duplicate", second.reasons[0])
        self.assertEqual(registry.admitted_count(), 1)

    def test_distinct_data_digest_is_a_new_component(self):
        registry = AdmissionRegistry()
        registry.admit(_good_submission())
        other = _good_submission()
        other.data_digest = "9" * 64
        result = registry.admit(other)
        self.assertTrue(result.admitted, result.reasons)
        self.assertEqual(registry.admitted_count(), 2)

    def test_replay_hash_of_returns_none_for_unknown(self):
        registry = AdmissionRegistry()
        sub = _good_submission()
        self.assertIsNone(registry.replay_hash_of(sub))
        registry.admit(sub)
        self.assertEqual(registry.replay_hash_of(sub), sub.replay_hash())


if __name__ == "__main__":
    unittest.main()
