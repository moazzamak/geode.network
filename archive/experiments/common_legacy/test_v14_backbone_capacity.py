"""Tests for V14-M91, backbone capacity.

The tests that matter here are the ones about *not* earning a verdict. M91 has
four ways to fail to refute H92 and only one way to refute it, and the whole
point of N91.5 is that a backbone which never demonstrated capacity cannot close
a hypothesis about capacity.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.tier4.eval_v14_m91_backbone_capacity import (
    _identity,
    _verdict,
    capacity_check,
    rank_variance_retained,
)
from experiments.tier4.eval_v14_m90_representation_remedies import class_assignment
from experiments.tier4.prepare_v14_m91_backbones import (
    manifest_identity,
    quantisation_divergence,
)

FIELDS = [
    "source_file",
    "source_row",
    "class_label",
    "domain",
    "image_path",
    "native_width",
    "native_height",
]

IDENTITY = {
    "m84_zero_rung_recall": 0.11875,
    "m85a_geometry_auroc": 0.585085105895996,
    "tolerance": 1e-9,
}

GATE = {
    "multiplicity_bar": 40.0,
    "v13_rejection_recall": 0.11875,
    "v13_geometry_auroc": 0.585085105895996,
    "auroc_margin": 0.02,
}

PROBE = {"probe_positive_floor": 0.5}


def _entry(**overrides: object) -> dict[str, object]:
    base = {
        "source_file": "train-00000-of-00003.parquet",
        "source_row": 294,
        "class_label": 0,
        "domain": 3,
        "image_path": "quickdraw/aircraft_carrier/4509494895706112.png",
        "native_width": 300,
        "native_height": 300,
    }
    base.update(overrides)
    return base


def _arm(
    *,
    accuracy: float,
    clears: bool = False,
    valid: bool = True,
    converged: bool = True,
    probe: float = 0.89,
) -> dict[str, object]:
    return {
        "known_accuracy": accuracy,
        "controls": {"valid": valid},
        "gate": {"clears": clears},
        "probe": {"converged": converged, "balanced_accuracy": probe},
    }


def _with_capacity(
    arms: dict[str, dict[str, object]], reference: str
) -> dict[str, dict[str, object]]:
    for arm in arms.values():
        arm["capacity"] = capacity_check(
            arm, arms[reference], config={"operand": "known_accuracy"}
        )
    return arms


# ---------------------------------------------------------------------------
# N91.2 -- identical rows is a check
# ---------------------------------------------------------------------------


def test_the_manifest_check_passes_when_the_selection_is_reproduced() -> None:
    produced = [_entry(), _entry(source_row=295, class_label=1)]
    report = manifest_identity(
        produced, list(produced), fields=FIELDS, maximum_differing=0
    )
    assert report["passes"]
    assert report["differing_entries"] == 0


def test_the_manifest_check_names_the_field_that_moved() -> None:
    produced = [_entry(source_row=999)]
    reference = [_entry()]
    report = manifest_identity(
        produced, reference, fields=FIELDS, maximum_differing=0
    )
    assert not report["passes"]
    assert report["first_differences"][0]["fields"] == ["source_row"]
    assert report["first_differences"][0]["produced"]["source_row"] == 999


def test_the_manifest_check_fails_on_a_row_count_mismatch() -> None:
    """A short extraction that agrees on every shared row is still not v13's."""
    report = manifest_identity(
        [_entry()], [_entry(), _entry(source_row=295)], fields=FIELDS, maximum_differing=0
    )
    assert not report["row_counts_agree"]
    assert not report["passes"]


def test_the_manifest_check_ignores_fields_it_was_not_asked_about() -> None:
    """Only the registered fields identify a row; extraction timings do not."""
    produced = [_entry(extraction_seconds=1.0)]
    reference = [_entry(extraction_seconds=2.0)]
    report = manifest_identity(
        produced, reference, fields=FIELDS, maximum_differing=0
    )
    assert report["passes"]


# ---------------------------------------------------------------------------
# N91.6 -- quantisation divergence
# ---------------------------------------------------------------------------


def test_a_graph_compared_against_itself_shows_no_divergence() -> None:
    """The divergence instrument reads zero when there is nothing to read.

    Run against the sealed dinov2-small INT8 graph twice, so a non-zero result
    would mean the measurement itself is noisy rather than the quantisation.
    """
    graph = Path("data/v5/backbones/dinov2-small/onnx/model_int8.onnx")
    if not graph.is_file():
        pytest.skip("the sealed dinov2-small graph is not present")
    preprocessing = json.loads(
        Path("data/v5/backbones/dinov2-small/preprocessor_config.json").read_text(
            encoding="utf-8"
        )
    )
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 256, size=(300, 300, 3), dtype=np.uint8) for _ in range(2)]
    report = quantisation_divergence(
        images,
        backbone={"id": "dinov2-small", "token_pooling_policy": "cls_token"},
        preprocessing=preprocessing,
        int8_path=graph,
        float_path=graph,
    )
    assert report["mean_relative_divergence"] == 0.0
    assert report["gated"] is False


# ---------------------------------------------------------------------------
# N91.5 -- capacity is a precondition, not a partial pass
# ---------------------------------------------------------------------------


def test_capacity_needs_a_strict_improvement_on_known_accuracy() -> None:
    reference = {"known_accuracy": 0.5076}
    better = capacity_check(
        {"known_accuracy": 0.5312}, reference, config={"operand": "known_accuracy"}
    )
    assert better["capacity_demonstrated"]
    assert better["delta"] == pytest.approx(0.0236)


def test_matching_the_reference_is_not_demonstrating_capacity() -> None:
    reference = {"known_accuracy": 0.5076}
    tied = capacity_check(
        {"known_accuracy": 0.5076}, reference, config={"operand": "known_accuracy"}
    )
    assert not tied["capacity_demonstrated"]
    assert tied["verdict"] == "capacity_not_demonstrated"


# ---------------------------------------------------------------------------
# N91.8 -- the reference arm pins the code path
# ---------------------------------------------------------------------------


def test_the_reference_arm_reproduces_v13_within_tolerance() -> None:
    arm = {
        "rejection_recall": {"rejection_recall": 0.11875},
        "auroc": {"auroc": 0.585085105895996},
    }
    assert _identity(arm, IDENTITY)["is_v13_geometry"]


def test_a_reference_arm_that_misses_either_operand_is_not_v13() -> None:
    arm = {
        "rejection_recall": {"rejection_recall": 0.11875},
        "auroc": {"auroc": 0.5851},
    }
    report = _identity(arm, IDENTITY)
    assert report["reproduces_m84"]
    assert not report["reproduces_m85a"]
    assert not report["is_v13_geometry"]


# ---------------------------------------------------------------------------
# The verdict, and its four ways of not being a refutation
# ---------------------------------------------------------------------------


def test_nothing_is_read_when_the_reference_arm_does_not_reproduce() -> None:
    arms = _with_capacity(
        {
            "dinov2_small": _arm(accuracy=0.4000),
            "dinov2_base": _arm(accuracy=0.6000, clears=True),
        },
        "dinov2_small",
    )
    arms["dinov2_small"]["identity"] = _identity(
        {
            "rejection_recall": {"rejection_recall": 0.09},
            "auroc": {"auroc": 0.55},
        },
        IDENTITY,
    )
    verdict = _verdict(
        arms, reference_arm="dinov2_small", gate=GATE, identity=IDENTITY, probe_config=PROBE
    )
    assert verdict["h92_capacity"] == "not_v13_geometry"
    assert "arms_clearing_all_bars" not in verdict


def _reference_identity() -> dict[str, object]:
    return _identity(
        {
            "rejection_recall": {"rejection_recall": 0.11875},
            "auroc": {"auroc": 0.585085105895996},
        },
        IDENTITY,
    )


def test_h92_is_untestable_when_no_arm_demonstrates_capacity() -> None:
    """The registered expectation is that accuracy improves. If it does not,
    the milestone has not tested capacity at all."""
    arms = _with_capacity(
        {
            "dinov2_small": _arm(accuracy=0.5076),
            "dinov2_base": _arm(accuracy=0.4800),
        },
        "dinov2_small",
    )
    arms["dinov2_small"]["identity"] = _reference_identity()
    verdict = _verdict(
        arms, reference_arm="dinov2_small", gate=GATE, identity=IDENTITY, probe_config=PROBE
    )
    assert verdict["h92_capacity"] == "untestable"
    assert verdict["arms_demonstrating_capacity"] == []


def test_h92_is_refuted_only_when_a_capable_arm_fails_the_bars() -> None:
    arms = _with_capacity(
        {
            "dinov2_small": _arm(accuracy=0.5076),
            "dinov2_base": _arm(accuracy=0.5500, clears=False),
        },
        "dinov2_small",
    )
    arms["dinov2_small"]["identity"] = _reference_identity()
    verdict = _verdict(
        arms, reference_arm="dinov2_small", gate=GATE, identity=IDENTITY, probe_config=PROBE
    )
    assert verdict["h92_capacity"] == "refuted"
    assert verdict["arms_demonstrating_capacity"] == ["dinov2_base"]


def test_h92_survives_when_a_capable_arm_clears_the_bars() -> None:
    arms = _with_capacity(
        {
            "dinov2_small": _arm(accuracy=0.5076),
            "dinov2_base": _arm(accuracy=0.5500, clears=True),
        },
        "dinov2_small",
    )
    arms["dinov2_small"]["identity"] = _reference_identity()
    verdict = _verdict(
        arms, reference_arm="dinov2_small", gate=GATE, identity=IDENTITY, probe_config=PROBE
    )
    assert verdict["h92_capacity"] == "survives"
    assert verdict["arms_clearing_all_bars"] == ["dinov2_base"]


def test_clearing_the_bars_without_capacity_does_not_rescue_h92() -> None:
    """N91.5 has to bite in both directions or it is not a rule.

    An arm that never beat the reference on accuracy has not demonstrated
    capacity, so its clearing the rejection bars says nothing about H92 either.
    """
    arms = _with_capacity(
        {
            "dinov2_small": _arm(accuracy=0.5076),
            "dinov2_base": _arm(accuracy=0.4900, clears=True),
        },
        "dinov2_small",
    )
    arms["dinov2_small"]["identity"] = _reference_identity()
    verdict = _verdict(
        arms, reference_arm="dinov2_small", gate=GATE, identity=IDENTITY, probe_config=PROBE
    )
    assert verdict["h92_capacity"] == "untestable"
    assert verdict["arms_clearing_all_bars"] == []


def test_an_arm_failing_its_control_is_void_rather_than_negative() -> None:
    arms = _with_capacity(
        {
            "dinov2_small": _arm(accuracy=0.5076),
            "dinov2_base": _arm(accuracy=0.5500, valid=False),
        },
        "dinov2_small",
    )
    arms["dinov2_small"]["identity"] = _reference_identity()
    verdict = _verdict(
        arms, reference_arm="dinov2_small", gate=GATE, identity=IDENTITY, probe_config=PROBE
    )
    assert verdict["h92_capacity"] == "void_instrument"


def test_an_unconverged_probe_invalidates_the_probe_not_the_milestone() -> None:
    """M90.2's N90.2.12 lesson: an unconverged probe understates what it
    measures. It is a diagnostic here, so it flags itself without touching H92."""
    arms = _with_capacity(
        {
            "dinov2_small": _arm(accuracy=0.5076),
            "dinov2_base": _arm(accuracy=0.5500, converged=False),
        },
        "dinov2_small",
    )
    arms["dinov2_small"]["identity"] = _reference_identity()
    verdict = _verdict(
        arms, reference_arm="dinov2_small", gate=GATE, identity=IDENTITY, probe_config=PROBE
    )
    assert not verdict["probe_instrument_valid"]
    assert verdict["h92_capacity"] == "refuted"


# ---------------------------------------------------------------------------
# N91.4 -- the rank is derived, not inherited
# ---------------------------------------------------------------------------


def test_the_registered_rank_is_what_the_floor_actually_permits() -> None:
    """The floor binds on fit rows per class, not on the ambient dimension, so
    the same rank at 384, 768 and 1024 dimensions is a derivation rather than a
    carry-over."""
    config = json.loads(
        Path("experiments/configs/v14/m91_backbone_capacity.json").read_text(
            encoding="utf-8"
        )
    )
    fit_per_class = int(config["partition"]["fit_per_class"])
    floor = int(config["separation"]["floor_samples_per_dimension"])
    assert int(config["rank"]) == fit_per_class // floor
    assert int(config["rank"]) <= 53
    assert {int(arm["output_dimension"]) for arm in config["arms"]} == {384, 768, 1024}


def test_every_backbone_shares_the_registered_preprocessing_hash() -> None:
    """N91.1. If the preprocessing differed the comparison would not be
    single-factor, and this is the cheapest place to notice."""
    config = json.loads(
        Path("experiments/configs/v14/m91_backbones.json").read_text(encoding="utf-8")
    )
    hashes = {backbone["preprocessor_sha256"] for backbone in config["backbones"]}
    assert hashes == {
        "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828"
    }
    assert {backbone["token_pooling_policy"] for backbone in config["backbones"]} == {
        "cls_token"
    }


def test_a_rank_covering_every_active_direction_retains_all_variance() -> None:
    generator = np.random.default_rng(4242)
    latent = generator.normal(size=(64, 3))
    basis = generator.normal(size=(3, 32))
    fit = (latent @ basis).astype(np.float32)
    assignment = class_assignment(np.zeros(64, dtype=np.int64), 1)

    retained = rank_variance_retained(fit, assignment, rank=10)

    assert retained["mean"] == pytest.approx(1.0, abs=1e-9)
    assert retained["classes_measured"] == 1
    assert retained["gated"] is False


def test_a_wider_representation_retains_less_at_the_same_rank() -> None:
    generator = np.random.default_rng(99)
    labels = np.repeat(np.arange(4), 64)
    assignment = class_assignment(labels, 4)
    narrow = generator.normal(size=(256, 40)).astype(np.float32)
    wide = generator.normal(size=(256, 160)).astype(np.float32)

    narrow_retained = rank_variance_retained(narrow, assignment, rank=20)
    wide_retained = rank_variance_retained(wide, assignment, rank=20)

    assert narrow_retained["mean"] > wide_retained["mean"]
    assert narrow_retained["minimum"] <= narrow_retained["mean"]
    assert wide_retained["maximum"] >= wide_retained["mean"]
