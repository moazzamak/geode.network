"""Tests for M90.1's repaired negative control.

The defect M90.1 exists to fix was invisible because nobody checked what the
control partitioned -- it balanced the domain mixture, which was the thing M85's
docstring asserted, while leaving class membership disjoint. So these tests do
not merely check that the new split runs: they reproduce the old defect, then
pin every property the repaired split is supposed to have, including the one the
old split satisfied.
"""

from __future__ import annotations

import numpy as np

from experiments.tier4.eval_v14_m90_1_control_repair import (
    _h90_reading,
    controls_with_repair,
    replication_report,
    split_profile,
    stratified_halves,
)


def _class_sorted(class_count: int, per_class: int, domain_count: int):
    """Rows ordered by class, each class carrying the same domain mixture."""
    labels = np.repeat(np.arange(class_count), per_class).astype(np.int64)
    domains = np.tile(
        np.arange(domain_count).repeat(per_class // domain_count), class_count
    ).astype(np.int64)
    return labels, domains


def _positional_halves(count: int) -> tuple[np.ndarray, np.ndarray]:
    midpoint = count // 2
    return np.arange(midpoint), np.arange(midpoint, count)


def test_the_old_split_separates_the_classes_it_claimed_to_balance() -> None:
    """The M90 defect, reproduced so the repair is measured against it."""
    labels, domains = _class_sorted(32, 12, 6)
    left, right = _positional_halves(len(labels))
    shared = np.intersect1d(labels[left], labels[right])
    assert len(shared) <= 1
    left_mix = split_profile(labels, domains, left, domain_count=6)["domain_mixture"]
    right_mix = split_profile(labels, domains, right, domain_count=6)["domain_mixture"]
    assert left_mix == right_mix  # balanced on domain, disjoint on class


def test_the_repaired_split_shares_every_class() -> None:
    labels, domains = _class_sorted(32, 12, 6)
    left, right = stratified_halves(labels, domains, seed=7)
    assert len(np.unique(labels[left])) == 32
    assert len(np.unique(labels[right])) == 32


def test_the_repaired_split_keeps_the_domain_mixture() -> None:
    labels, domains = _class_sorted(32, 12, 6)
    left, right = stratified_halves(labels, domains, seed=7)
    left_mix = split_profile(labels, domains, left, domain_count=6)["domain_mixture"]
    right_mix = split_profile(labels, domains, right, domain_count=6)["domain_mixture"]
    assert left_mix == right_mix


def test_the_repaired_split_halves_every_class_and_domain_cell() -> None:
    labels, domains = _class_sorted(16, 12, 6)
    left, right = stratified_halves(labels, domains, seed=11)
    for c in range(16):
        for d in range(6):
            cell = (labels == c) & (domains == d)
            in_left = int(np.isin(np.flatnonzero(cell), left).sum())
            in_right = int(np.isin(np.flatnonzero(cell), right).sum())
            assert abs(in_left - in_right) <= 1


def test_the_repaired_split_is_a_partition() -> None:
    labels, domains = _class_sorted(8, 15, 3)
    left, right = stratified_halves(labels, domains, seed=3)
    assert len(np.intersect1d(left, right)) == 0
    assert np.array_equal(np.union1d(left, right), np.arange(len(labels)))


def test_the_repaired_split_is_deterministic_for_a_seed() -> None:
    labels, domains = _class_sorted(8, 12, 4)
    first = stratified_halves(labels, domains, seed=5)
    second = stratified_halves(labels, domains, seed=5)
    other = stratified_halves(labels, domains, seed=6)
    assert np.array_equal(first[0], second[0])
    assert not np.array_equal(first[0], other[0])


def _controls(known, far, strat_split, block_split):
    return controls_with_repair(
        known,
        far,
        floor=0.99,
        tolerance=0.02,
        class_block=block_split,
        stratified=strat_split,
    )


def test_validity_is_decided_on_the_repaired_control() -> None:
    """An arm the old split failed must not be failed by the old split again."""
    rng = np.random.default_rng(0)
    known = np.concatenate([rng.normal(0.0, 1.0, 200), rng.normal(3.0, 1.0, 200)])
    far = rng.normal(50.0, 1.0, 200)
    block = (np.arange(200), np.arange(200, 400))  # the shifted halves
    strat = (np.arange(0, 400, 2), np.arange(1, 400, 2))  # interleaved, exchangeable
    controls = _controls(known, far, strat, block)
    assert controls["negative_control_class_block_passes"] is False
    assert controls["negative_passes"] is True
    assert controls["valid"] is True
    assert controls["decided_on"] == "negative_control_stratified"


def test_both_controls_stay_on_the_record() -> None:
    rng = np.random.default_rng(1)
    known = rng.normal(0.0, 1.0, 400)
    far = rng.normal(50.0, 1.0, 200)
    controls = _controls(
        known,
        far,
        (np.arange(0, 400, 2), np.arange(1, 400, 2)),
        (np.arange(200), np.arange(200, 400)),
    )
    assert "negative_control_class_block" in controls
    assert "negative_control_stratified" in controls


def test_a_failed_positive_control_still_voids_the_arm() -> None:
    rng = np.random.default_rng(2)
    known = rng.normal(0.0, 1.0, 400)
    far = rng.normal(0.0, 1.0, 200)  # indistinguishable from known
    controls = _controls(
        known,
        far,
        (np.arange(0, 400, 2), np.arange(1, 400, 2)),
        (np.arange(200), np.arange(200, 400)),
    )
    assert controls["positive_passes"] is False
    assert controls["valid"] is False


def _arm(mult: float, recall: float, auroc: float, accuracy: float = 0.5) -> dict:
    return {
        "acceptance_multiplicity": {"mean": mult},
        "rejection_recall": {"rejection_recall": recall},
        "auroc": {"auroc": auroc},
        "known_accuracy": accuracy,
        "controls": {
            "positive_control": 1.0,
            "negative_control": 0.48,
            "negative_control_class_block": 0.48,
            "valid": True,
        },
    }


def test_replication_passes_when_only_the_control_changed() -> None:
    arms = {"baseline": _arm(78.74, 0.11875, 0.5851)}
    reference = {"evidence_hash": "abc", "arms": {"baseline": _arm(78.74, 0.11875, 0.5851)}}
    report = replication_report(arms, reference, tolerance=1e-12)
    assert report["operands_reproduce"] is True
    assert report["largest_absolute_delta"] == 0.0


def test_replication_flags_an_operand_that_moved() -> None:
    """Only the control changed, so a moved operand is a defect, not a result."""
    arms = {"baseline": _arm(78.74, 0.13000, 0.5851)}
    reference = {"evidence_hash": "abc", "arms": {"baseline": _arm(78.74, 0.11875, 0.5851)}}
    report = replication_report(arms, reference, tolerance=1e-12)
    assert report["operands_reproduce"] is False
    assert report["largest_absolute_delta"] > 0.01
    assert "defect" in report["reading"]


def test_h90_stays_undetermined_when_the_mixture_still_fails() -> None:
    arm = _arm(75.87, 0.08715, 0.5890)
    arm["controls"]["valid"] = False
    reading = _h90_reading({"domain_mixture": arm})
    assert reading["outcome"] == "mixture_unmeasurable"
    assert "not converted into a negative" in reading["detail"]


def test_h90_is_read_when_the_mixture_becomes_measurable() -> None:
    reading = _h90_reading({"domain_mixture": _arm(75.87, 0.08715, 0.5890)})
    assert reading["outcome"] == "mixture_measurable"
