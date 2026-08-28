"""Focused tests for the v13 M77 probe-degeneracy forensics."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.common.v12_metric_fields import (
    _torch_probes,
    _torch_scores,
    initialize_metric_fields,
)
from experiments.common.v13_probe_forensics import (
    SCALE_RELATIVE_FAMILIES,
    _probe_source_classes,
    probe_scale_invariance,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = REPO_ROOT / "logs" / "results" / "v13" / "m77_probe_degeneracy" / "evidence.json"
FAMILIES = ("axis_tangent", "normal", "masking", "random_direction")


def _fixture_fields(rank: int = 3, dimension: int = 8, classes: int = 4):
    generator = np.random.default_rng(7701)
    features = []
    labels = []
    for class_label in range(classes):
        offset = generator.normal(scale=3.0, size=dimension)
        block = generator.normal(size=(60, dimension)) + offset
        features.append(block)
        labels.append(np.full(60, class_label))
    return initialize_metric_fields(
        np.vstack(features), np.concatenate(labels), rank=rank
    )


class ProbeSourceOrderingTests(unittest.TestCase):
    def test_source_ordering_matches_probe_generator_length(self) -> None:
        fields = _fixture_fields()
        sources, labels = _probe_source_classes(
            len(fields.classes), fields.rank, FAMILIES
        )
        probes = _torch_probes(
            torch.as_tensor(fields.centers),
            torch.as_tensor(fields.bases),
            torch.as_tensor(fields.tangent_scales),
            torch.as_tensor(fields.residual_scales),
            families=FAMILIES,
            seed=11,
        )
        self.assertEqual(len(sources), len(probes))
        self.assertEqual(len(labels), len(probes))

    def test_axis_probes_recover_their_source_class(self) -> None:
        fields = _fixture_fields()
        sources, labels = _probe_source_classes(
            len(fields.classes), fields.rank, ("axis_tangent",)
        )
        # Two signed probes per rank direction, per class.
        self.assertEqual(len(sources), len(fields.classes) * fields.rank * 2)
        self.assertTrue(np.all(labels == "axis_tangent"))
        self.assertEqual(sources[0], 0)
        self.assertEqual(sources[-1], len(fields.classes) - 1)


class ProbeDegeneracyTests(unittest.TestCase):
    def test_scale_relative_probe_scores_are_exactly_four(self) -> None:
        fields = _fixture_fields()
        sources, labels = _probe_source_classes(
            len(fields.classes), fields.rank, FAMILIES
        )
        probes = _torch_probes(
            torch.as_tensor(fields.centers),
            torch.as_tensor(fields.bases),
            torch.as_tensor(fields.tangent_scales),
            torch.as_tensor(fields.residual_scales),
            families=FAMILIES,
            seed=11,
        )
        scores, _ = _torch_scores(
            probes,
            torch.as_tensor(fields.centers),
            torch.as_tensor(fields.bases),
            torch.as_tensor(fields.tangent_scales),
            torch.as_tensor(fields.residual_scales),
        )
        own = scores[torch.arange(len(sources)), torch.as_tensor(sources)]
        for family in SCALE_RELATIVE_FAMILIES:
            values = own.detach().numpy()[labels == family]
            self.assertTrue(
                np.allclose(values, 4.0, rtol=0.0, atol=1e-9),
                msg=f"{family} own-class scores were not exactly 4.0",
            )

    def test_probe_term_has_no_gradient_to_the_fitted_extents(self) -> None:
        fields = _fixture_fields()
        log_tangent = torch.nn.Parameter(
            torch.log(torch.as_tensor(fields.tangent_scales))
        )
        log_residual = torch.nn.Parameter(
            torch.log(torch.as_tensor(fields.residual_scales))
        )
        centers = torch.nn.Parameter(torch.as_tensor(fields.centers))
        bases = torch.as_tensor(fields.bases)
        tangent = torch.exp(log_tangent).clamp_min(1e-6)
        residual = torch.exp(log_residual).clamp_min(1e-6)
        probes = _torch_probes(
            centers, bases, tangent, residual, families=SCALE_RELATIVE_FAMILIES, seed=11
        )
        scores, _ = _torch_scores(probes, centers, bases, tangent, residual)
        # The v12 target is detached, so it contributes no gradient.
        probe_target = torch.tensor(12.0, dtype=scores.dtype)
        loss = torch.mean(
            torch.relu(probe_target - torch.min(scores, dim=1).values)
        )
        grads = torch.autograd.grad(
            loss, [log_tangent, log_residual, centers], allow_unused=True
        )
        for grad in grads:
            norm = 0.0 if grad is None else float(torch.linalg.vector_norm(grad))
            self.assertLess(norm, 1e-12)

    def test_invariance_helper_flags_every_trained_family(self) -> None:
        fields = _fixture_fields()
        report = probe_scale_invariance(
            fields, families=FAMILIES, scale_factors=(0.1, 1.0, 100.0), seed=11
        )
        self.assertEqual(report["scale_coupled_families"], [])
        self.assertEqual(sorted(report["invariant_families"]), sorted(FAMILIES))


@unittest.skipUnless(EVIDENCE.exists(), "M77 evidence has not been generated")
class M77EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    def test_instrumentation_reproduced_v12_exactly(self) -> None:
        gate = self.evidence["gate"]
        self.assertEqual(gate["history_reproduction_delta"], 0.0)
        self.assertTrue(gate["trained_state_hash_match"])
        self.assertTrue(gate["instrumentation_faithful"])

    def test_registered_decision_rule_fired(self) -> None:
        gate = self.evidence["gate"]
        self.assertTrue(gate["invariance_confirmed"])
        self.assertTrue(gate["probe_gradient_degenerate"])
        self.assertGreaterEqual(gate["loss_drop_explained_by_target_fraction"], 1.0)
        self.assertTrue(gate["h77_confirmed"])
        self.assertFalse(gate["final_labels_opened"])

    def test_own_class_probe_scores_never_moved(self) -> None:
        first = self.evidence["probe_diagnostics"][0]
        last = self.evidence["probe_diagnostics"][-1]
        for family in SCALE_RELATIVE_FAMILIES:
            key = f"own_score__{family}"
            self.assertAlmostEqual(first[key], 4.0, places=9)
            self.assertAlmostEqual(last[key], 4.0, places=9)


if __name__ == "__main__":
    unittest.main()
