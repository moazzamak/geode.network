"""M375 - the measured chain, the single attribution rule, and the
chain-length cap.

Registered in ``analysis/WHITEPAPER_REVIEW_2026-08-28_R2.md`` under
G11 (composition asserted from fusion evidence) and G12 (two
attribution rules). The measured evidence is
``analysis/m375_measured_chain.json``.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from geode.core.chains import (
    ChainArtifact,
    ChainStage,
    ChainTooLongError,
    ContractMismatchError,
    MAX_CHAIN_STAGES,
    shapley_split,
)
from geode.core.representation import (
    RepresentationArtifact,
    attribution_share,
)
from tools.m375_measured_chain import ContractError, check_contract

EVIDENCE = (Path(__file__).resolve().parents[2]
            / "analysis" / "m375_measured_chain.json")


def _stage(name: str) -> ChainStage:
    return ChainStage(name, frozenset({"f"}), frozenset({"f"}))


class TestTwoRulesDiverge(unittest.TestCase):
    """The failure G12 names, reproduced FIRST: on one set of
    measured coalition values the two rules pay different amounts."""

    def test_shapley_and_loo_disagree_on_the_measured_chain(self) -> None:
        v_empty, v_a, v_b, v_ab = 0.002899, 0.005768, 0.245014, 0.274058
        phi_a = 0.5 * ((v_a - v_empty) + (v_ab - v_b))
        phi_b = 0.5 * ((v_b - v_empty) + (v_ab - v_a))
        loo_a, loo_b = v_ab - v_b, v_ab - v_a

        shapley_share = phi_a / (phi_a + phi_b)
        loo_share = loo_a / (loo_a + loo_b)
        self.assertAlmostEqual(shapley_share, 0.0588, places=4)
        self.assertAlmostEqual(loo_share, 0.0977, places=4)
        self.assertGreater(loo_share / shapley_share, 1.6)

        # both rules are efficient here; they differ in the SPLIT,
        # which is why naming both is a real ambiguity and not a
        # notational one
        self.assertAlmostEqual(phi_a + phi_b, v_ab - v_empty, places=12)


class TestSingleAttributionRule(unittest.TestCase):

    def _artifact(self) -> RepresentationArtifact:
        return RepresentationArtifact(
            input_contract=("trunk",), output_name="adapter",
            output_width=64, utility=0.05, ubar=1.5,
            price_per_unit=2.0, weights_digest="a" * 64)

    def test_measured_coalitions_give_the_shapley_split(self) -> None:
        values = {
            frozenset(): 0.002899,
            frozenset({"adapter"}): 0.005768,
            frozenset({"trunk"}): 0.245014,
            frozenset({"adapter", "trunk"}): 0.274058,
        }
        shares = attribution_share(0.274058, 0.245014, self._artifact(),
                                   ["trunk"], coalition_values=values)
        self.assertAlmostEqual(shares["adapter"], 0.015957, places=6)
        self.assertAlmostEqual(shares["trunk"], 0.2552025, places=7)
        # efficiency: the split exhausts v(AB) - v(empty)
        self.assertAlmostEqual(sum(shares.values()),
                               0.274058 - 0.002899, places=12)

    def test_shapley_path_refuses_incomplete_coalitions(self) -> None:
        with self.assertRaises(ValueError):
            attribution_share(
                0.274058, 0.245014, self._artifact(), ["trunk"],
                coalition_values={frozenset({"adapter", "trunk"}): 0.27})

    def test_stand_in_still_available_without_coalitions(self) -> None:
        shares = attribution_share(0.60, 0.50, self._artifact(),
                                   ["trunk"])
        self.assertAlmostEqual(shares["adapter"], 0.10, places=9)


class TestChainLengthCap(unittest.TestCase):

    def test_cap_is_four(self) -> None:
        self.assertEqual(MAX_CHAIN_STAGES, 4)

    def test_four_stages_admitted(self) -> None:
        chain = ChainArtifact("c", [_stage(f"s{i}") for i in range(4)],
                              axis="vision")
        self.assertTrue(chain.admissible)

    def test_five_stages_refused(self) -> None:
        with self.assertRaises(ChainTooLongError):
            ChainArtifact("c", [_stage(f"s{i}") for i in range(5)],
                          axis="vision")

    def test_cap_keeps_the_coalition_count_replayable(self) -> None:
        self.assertEqual(2 ** MAX_CHAIN_STAGES, 16)

    def test_contract_mismatch_still_refused(self) -> None:
        stages = [ChainStage("a", frozenset({"x"}), frozenset({"y"})),
                  ChainStage("b", frozenset({"z"}), frozenset({"w"}))]
        with self.assertRaises(ContractMismatchError):
            ChainArtifact("c", stages, axis="vision")


class TestMeasuredChainEvidence(unittest.TestCase):
    """The sealed M375 numbers, pinned so a later edit to the paper
    cannot drift away from the measurement."""

    def setUp(self) -> None:
        if not EVIDENCE.exists():
            self.skipTest("M375 evidence not present")
        self.ev = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    def test_anchor_reproduced_before_anything_was_read(self) -> None:
        anchor = self.ev["anchor_reproduction"]
        self.assertTrue(anchor["reproduced"])
        self.assertLessEqual(
            abs(anchor["measured"] - anchor["registered_m144_r56_read"]),
            anchor["tolerance"])

    def test_chain_beats_the_strongest_single_stage(self) -> None:
        c = self.ev["coalitions"]
        self.assertGreater(c["v_AB (the chain)"],
                           c["v_B (null router + monolithic head)"])
        self.assertGreater(c["v_AB (the chain)"],
                           c["v_A (router + null head)"])

    def test_shapley_split_is_efficient(self) -> None:
        self.assertTrue(self.ev["shapley"]["sums_to_v_AB_minus_v_empty"])

    def test_contract_refuses_a_mismatched_pairing(self) -> None:
        check = self.ev["contract_check"]
        self.assertTrue(check["matched_pairing_admitted"])
        self.assertTrue(check["mismatched_pairing_refused"])

    def test_oracle_diagnostic_separates_the_two_failure_modes(self
                                                               ) -> None:
        d = self.ev["diagnostics"]
        # routing error costs less than specialisation is worth, which
        # is why the routed chain wins and not only the oracle one
        self.assertGreater(d["specialisation_gain_at_oracle"],
                           d["routing_loss_vs_oracle"])


class TestContractCheckHelper(unittest.TestCase):

    def test_matched_kinds_admitted(self) -> None:
        check_contract("domain_label", 6, "domain_label", 6)

    def test_mismatched_cardinality_refused(self) -> None:
        with self.assertRaises(ContractError):
            check_contract("class_label", 345, "domain_label", 6)

    def test_mismatched_kind_refused(self) -> None:
        with self.assertRaises(ContractError):
            check_contract("audio_embedding", 6, "domain_label", 6)


if __name__ == "__main__":
    unittest.main()
