"""Unit tests for M247: the refusal-capability admission interface
(measured-not-declared, absent-until-measured).
"""
from __future__ import annotations

import unittest

from geode.core.refusal import (
    RefusalCapability,
    RefusalRecord,
    refusal_admission,
    refusal_measured_tag,
)


def _rec(probe: str, rate: float, attesters) -> RefusalRecord:
    return RefusalRecord(probe_id=probe, refusal_rate=rate,
                         attestations=frozenset(attesters))


class TestM247RefusalCapability(unittest.TestCase):

    def test_absent_with_no_records(self):
        decision = refusal_admission([])
        self.assertFalse(decision["admitted"])
        self.assertEqual(decision["reason"], "absent")

    def test_absent_with_only_below_quorum_records(self):
        # single-source (declared) records never count: absent, not
        # failed (the measured-not-declared contract)
        decision = refusal_admission([_rec("p1", 0.95, {"v1"})],
                                     k_of_n=2)
        self.assertFalse(decision["admitted"])
        self.assertEqual(decision["reason"], "absent")

    def test_admitted_at_quorum_above_rate(self):
        decision = refusal_admission(
            [_rec("p1", 0.95, {"v1", "v2"}),
             _rec("p2", 0.99, {"v1", "v3"})],
            k_of_n=2, min_probes=2, min_rate=0.9)
        self.assertTrue(decision["admitted"])
        self.assertEqual(decision["reason"], "admitted")
        self.assertEqual(set(decision["admitted_probes"]), {"p1", "p2"})

    def test_below_rate_rejected_with_probe_name(self):
        decision = refusal_admission(
            [_rec("p1", 0.95, {"v1", "v2"}),
             _rec("p2", 0.4, {"v1", "v2"})],
            k_of_n=2, min_probes=2, min_rate=0.9)
        self.assertFalse(decision["admitted"])
        self.assertEqual(decision["reason"], "below_rate")
        self.assertEqual(decision["below_rate_probes"], ["p2"])

    def test_insufficient_probes(self):
        decision = refusal_admission(
            [_rec("p1", 0.95, {"v1", "v2"})],
            k_of_n=2, min_probes=2, min_rate=0.9)
        self.assertFalse(decision["admitted"])
        self.assertEqual(decision["reason"], "insufficient_probes")

    def test_measured_tag_hook(self):
        good = [_rec("p1", 0.95, {"v1", "v2"})]
        bad = [_rec("p1", 0.4, {"v1", "v2"})]
        none = [_rec("p1", 0.95, {"v1"})]
        self.assertEqual(refusal_measured_tag(good), "refusal")
        self.assertIsNone(refusal_measured_tag(bad))
        self.assertIsNone(refusal_measured_tag(none))

    def test_invalid_rate_raises(self):
        with self.assertRaises(ValueError):
            refusal_admission([_rec("p1", 1.5, {"v1", "v2"})])

    def test_capability_tracker_append_only(self):
        cap = RefusalCapability()
        cap.add("p1", 0.95, {"v1", "v2"})
        self.assertFalse(cap.admitted(min_probes=2)["admitted"])
        cap.add("p2", 0.99, {"v1", "v3"})
        self.assertTrue(cap.admitted(min_probes=2)["admitted"])

    def test_deterministic(self):
        records = [_rec("p1", 0.95, {"v1", "v2"})]
        self.assertEqual(refusal_admission(records),
                         refusal_admission(records))


if __name__ == "__main__":
    unittest.main()
