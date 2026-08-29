"""M355 / M352 - declared-label-set scoring and the privacy tier.

The regression this file exists for is G20: the retired rule
``metric x coverage`` ranked the sealed 0.901 scoped Open Images
arm at 0.0441, below a 0.1643 full-coverage arm. That inversion is
reproduced here from the sealed numbers before the replacement is
shown to fix it, so the test fails if the replacement is quietly
reverted.
"""
from __future__ import annotations

import unittest

from geode.core.coverage_adjusted import AxisScore
from geode.core.declared_label_set import (
    Capability,
    CrossTierRoute,
    Declaration,
    NotQualified,
    PrivacyTier,
    declared_score,
    qualifies,
    rank,
)

SERVED = [f"c{i}" for i in range(129)]
UNSERVED = [f"c{i}" for i in range(129, 601)]

# The sealed M355 measurement: on rows whose true class is in the
# 129 served classes, the scoped arm reads 0.9378 (argmax over its
# own 129) and the full-coverage arm reads 0.9010 (argmax over all
# 601). See analysis/m355_declared_label_set.json.
SCOPED = Capability(
    name="scoped",
    per_label_metric={label: 0.9378 for label in SERVED},
)
FULL = Capability(
    name="full_coverage",
    per_label_metric={label: 0.9010 for label in SERVED}
    | {label: 0.0 for label in UNSERVED},
)


class TestRetiredRuleInverts(unittest.TestCase):
    """Reproduce the failure before showing the repair."""

    def test_coverage_multiplication_ranks_the_worse_arm_first(
            self) -> None:
        scoped = AxisScore(accuracy=0.901, coverage=0.049)
        full = AxisScore(accuracy=0.1643, coverage=1.0)
        self.assertAlmostEqual(scoped.coverage_adjusted, 0.0441,
                               places=4)
        self.assertAlmostEqual(full.coverage_adjusted, 0.1643,
                               places=4)
        self.assertLess(scoped.coverage_adjusted,
                        full.coverage_adjusted)


class TestDeclaredLabelSet(unittest.TestCase):

    def test_scoped_arm_outranks_for_the_129_declaration(
            self) -> None:
        declaration = Declaration(labels=frozenset(SERVED))
        ranked = rank([FULL, SCOPED], declaration)
        self.assertEqual([name for name, _ in ranked],
                         ["scoped", "full_coverage"])
        self.assertAlmostEqual(ranked[0][1], 0.9378, places=4)
        self.assertAlmostEqual(ranked[1][1], 0.9010, places=4)

    def test_scoped_arm_is_unqualified_for_the_601_declaration(
            self) -> None:
        declaration = Declaration(
            labels=frozenset(SERVED) | frozenset(UNSERVED))
        self.assertFalse(qualifies(SCOPED, declaration))
        self.assertTrue(qualifies(FULL, declaration))
        ranked = rank([FULL, SCOPED], declaration)
        self.assertEqual([name for name, _ in ranked],
                         ["full_coverage"])

    def test_unqualified_raises_rather_than_scoring_zero(
            self) -> None:
        declaration = Declaration(
            labels=frozenset(SERVED) | frozenset(UNSERVED))
        with self.assertRaises(NotQualified):
            declared_score(SCOPED, declaration)

    def test_score_is_row_weighted_over_declared_labels_only(
            self) -> None:
        capability = Capability(
            name="uneven",
            per_label_metric={"a": 1.0, "b": 0.0, "c": 0.5},
            label_rows={"a": 30, "b": 10, "c": 1000},
        )
        # 'c' is not declared and must not enter the score.
        declaration = Declaration(labels=frozenset({"a", "b"}))
        self.assertAlmostEqual(declared_score(capability, declaration),
                               30.0 / 40.0)

    def test_empty_declaration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Declaration(labels=frozenset())

    def test_metric_outside_unit_interval_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Capability(name="bad", per_label_metric={"a": 1.5})


class TestPrivacyTier(unittest.TestCase):

    def test_cross_tier_route_is_refused_not_ranked(self) -> None:
        private = Capability(
            name="private",
            per_label_metric={"a": 0.99},
            privacy_tier=PrivacyTier.PRIVATE,
        )
        declaration = Declaration(labels=frozenset({"a"}),
                                  privacy_tier=PrivacyTier.PLAINTEXT)
        self.assertFalse(qualifies(private, declaration))
        with self.assertRaises(CrossTierRoute):
            declared_score(private, declaration)

    def test_a_better_arm_in_another_tier_never_wins(self) -> None:
        plaintext = Capability(name="plain",
                               per_label_metric={"a": 0.60})
        private = Capability(name="private",
                             per_label_metric={"a": 0.99},
                             privacy_tier=PrivacyTier.PRIVATE)
        declaration = Declaration(labels=frozenset({"a"}),
                                  privacy_tier=PrivacyTier.PLAINTEXT)
        self.assertEqual(rank([private, plaintext], declaration),
                         [("plain", 0.60)])

    def test_no_tier_qualifies_when_none_matches(self) -> None:
        plaintext = Capability(name="plain",
                               per_label_metric={"a": 0.60})
        declaration = Declaration(
            labels=frozenset({"a"}),
            privacy_tier=PrivacyTier.DEVICE_ENCODER)
        self.assertEqual(rank([plaintext], declaration), [])


if __name__ == "__main__":
    unittest.main()
