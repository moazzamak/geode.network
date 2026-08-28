"""M316 unit tests - chains as first-class artifacts and the
marginal-contribution split."""
from __future__ import annotations

import unittest

from geode.core.chains import (
    ChainArtifact,
    ChainStage,
    ContractMismatchError,
    attribution_shares,
    measured_gap,
    shapley_split,
)


def _asr_stage(artifact: str, otype: str) -> ChainStage:
    return ChainStage(artifact_id=artifact,
                      contract_in=frozenset({"audio"}),
                      contract_out=frozenset({otype}))


def _two_stage_chain() -> ChainArtifact:
    asr = ChainStage("asr", frozenset({"audio"}), frozenset({"text"}))
    intent = ChainStage("intent", frozenset({"text"}),
                        frozenset({"intent_label"}))
    return ChainArtifact(chain_id="transcribe-intent",
                         stages=[asr, intent], axis="audio->intent",
                         end_to_end_score=0.81,
                         stage_scores={"asr": 0.90, "intent": 0.90})


class TestAdmissibility(unittest.TestCase):
    def test_type_level_chain_admits(self):
        chain = _two_stage_chain()
        self.assertTrue(chain.admissible)

    def test_contract_mismatch_refuses_to_assemble(self):
        with self.assertRaises(ContractMismatchError):
            ChainArtifact(
                chain_id="bad",
                stages=[
                    ChainStage("a", frozenset({"audio"}),
                               frozenset({"text"})),
                    ChainStage("b", frozenset({"image"}),
                               frozenset({"label"})),
                ],
                axis="audio->label")

    def test_one_fingerprint(self):
        chain = _two_stage_chain()
        self.assertEqual(chain.fingerprint,
                         ("audio->intent", "asr", "intent"))


class TestSplit(unittest.TestCase):
    def test_shapley_additive_returns_marginals(self):
        split = shapley_split({"asr": 0.9, "intent": 0.9},
                              chain_score=0.81, identity_score=0.0)
        self.assertAlmostEqual(split["asr"], 0.9)
        self.assertAlmostEqual(split["intent"], 0.9)

    def test_identity_stage_earns_zero(self):
        shares = attribution_shares(
            {"asr": 0.9, "identity": 0.0, "intent": 0.9},
            chain_score=0.81, identity_score=0.0)
        self.assertAlmostEqual(shares.get("identity", 0.0), 0.0)
        self.assertAlmostEqual(sum(shares.values()), 1.0)

    def test_harmful_stage_earns_zero(self):
        shares = attribution_shares(
            {"asr": 0.9, "harmful": 0.3, "intent": 0.9},
            chain_score=0.81, identity_score=0.5)
        self.assertAlmostEqual(shares.get("harmful", 0.0), 0.0)
        self.assertAlmostEqual(sum(shares.values()), 1.0)

    def test_split_sums_to_one(self):
        shares = attribution_shares(
            {"asr": 0.9, "intent": 0.9, "intent2": 0.6},
            chain_score=0.81, identity_score=0.0)
        self.assertAlmostEqual(sum(shares.values()), 1.0)

    def test_empty_pool_returns_nothing(self):
        self.assertEqual(attribution_shares(
            {"asr": 0.0}, chain_score=0.0, identity_score=0.0), {})

    def test_measured_gap_is_the_composition_reading(self):
        gap = measured_gap({"asr": 0.9, "intent": 0.9},
                           chain_score=0.81)
        self.assertAlmostEqual(gap, 0.81 - 0.81)
        self.assertAlmostEqual(
            measured_gap({"asr": 0.9, "intent": 0.9},
                         chain_score=0.85), 0.04)


if __name__ == "__main__":
    unittest.main()
