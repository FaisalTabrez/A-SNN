from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.evidence_synthesis import EVIDENCE_FILENAMES, synthesize_gen5_evidence


class EvidenceSynthesisContractTest(unittest.TestCase):
    def test_required_evidence_chain_is_complete(self) -> None:
        self.assertEqual(tuple(EVIDENCE_FILENAMES), (
            "phase44", "phase45", "phase46", "phase47", "phase48", "phase49",
            "milestone_a", "gen6", "gen7", "gen8", "gen9", "gen10", "gen11",
        ))
        self.assertEqual(len(set(EVIDENCE_FILENAMES.values())), 13)

    def test_milestone_a_terminal_decision_is_in_final_ledger(self) -> None:
        result = synthesize_gen5_evidence(ROOT / "outputs")
        self.assertAlmostEqual(
            result.metrics["milestone_a_tcn_confirmation_accuracy"],
            0.5916985575507802,
        )
        self.assertEqual(result.metrics["milestone_a_qualified_arm_count"], 0.0)
        claims = {row["claim"]: row for row in result.claims}
        readiness = claims[
            "The current Gen-5 architecture qualifies for hardware optimization"
        ]
        self.assertEqual(readiness["status"], "rejected")
        self.assertFalse(readiness["gate_passed"])

    def test_gen6_terminal_decision_is_in_final_ledger(self) -> None:
        result = synthesize_gen5_evidence(ROOT / "outputs")
        self.assertAlmostEqual(result.metrics["gen6_tcn_accuracy"], 0.5908154253753312)
        self.assertAlmostEqual(result.metrics["gen6_lif_accuracy"], 0.5901612533935171)
        self.assertAlmostEqual(
            result.metrics["gen6_lif_state_specificity_vs_shuffled"],
            -0.006574428417230882,
        )
        self.assertEqual(result.metrics["gen6_qualified_arm_count"], 0.0)
        claims = {row["claim"]: row for row in result.claims}
        self.assertEqual(
            claims[
                "The Gen-6 shared residual LIF preserves TCN predictive accuracy"
            ]["status"],
            "supported",
        )
        self.assertEqual(
            claims[
                "The Gen-6 LIF correction is beneficially sample-specific"
            ]["status"],
            "rejected",
        )
        self.assertEqual(
            claims[
                "The Gen-6 successor qualifies for hardware optimization"
            ]["status"],
            "rejected",
        )

    def test_gen7_terminal_decision_is_in_final_ledger(self) -> None:
        result = synthesize_gen5_evidence(ROOT / "outputs")
        self.assertAlmostEqual(
            result.metrics["gen7_lif_gain_vs_tcn"],
            0.004170346384064438,
        )
        self.assertAlmostEqual(
            result.metrics["gen7_lif_future_alignment_margin"],
            0.2927859868889986,
        )
        self.assertAlmostEqual(
            result.metrics["gen7_lif_state_specificity_vs_shuffled"],
            -0.01022143721584401,
        )
        self.assertEqual(result.metrics["gen7_qualified_arm_count"], 0.0)
        claims = {row["claim"]: row for row in result.claims}
        self.assertEqual(
            claims["Gen-7 paired future prediction improves state alignment"][
                "status"
            ],
            "supported",
        )
        self.assertEqual(
            claims[
                "Gen-7 uses predictive state beneficially by sample identity and temporal order"
            ]["status"],
            "rejected",
        )
        self.assertEqual(
            claims["The Gen-7 successor qualifies for hardware optimization"][
                "status"
            ],
            "rejected",
        )

    def test_gen8_terminal_decision_is_in_final_ledger(self) -> None:
        result = synthesize_gen5_evidence(ROOT / "outputs")
        self.assertAlmostEqual(
            result.metrics["gen8_pooled_lif_gain_vs_tcn"],
            0.015356687273084146,
        )
        self.assertAlmostEqual(
            result.metrics["gen8_analog_temporal_order_vs_reversed"],
            0.0056095247440552205,
        )
        self.assertAlmostEqual(
            result.metrics["gen8_analog_state_specificity_vs_shuffled"],
            0.0011775095672652187,
        )
        self.assertAlmostEqual(
            result.metrics["gen8_candidate_screen_validation_accuracy"],
            0.07266666666666667,
        )
        self.assertAlmostEqual(
            result.metrics["gen8_candidate_screen_spike_rate"],
            0.5065631499290466,
        )
        self.assertEqual(result.metrics["gen8_qualified_arm_count"], 0.0)
        claims = {row["claim"]: row for row in result.claims}
        self.assertEqual(
            claims[
                "Gen-8 time-local analog binding introduces temporal-order sensitivity"
            ]["status"],
            "supported",
        )
        self.assertEqual(
            claims[
                "Gen-8 time-local analog binding uses the correct sample identity"
            ]["status"],
            "rejected",
        )
        self.assertEqual(
            claims[
                "The Gen-8 paired time-local LIF candidate is stable enough for confirmation"
            ]["status"],
            "rejected",
        )
        self.assertEqual(
            claims["The Gen-8 successor qualifies for hardware optimization"][
                "status"
            ],
            "rejected",
        )

    def test_gen9_terminal_decision_is_in_final_ledger(self) -> None:
        result = synthesize_gen5_evidence(ROOT / "outputs")
        self.assertAlmostEqual(
            result.metrics["gen9_predictive_lif_screen_gap_vs_tcn"],
            -0.06466666666666665,
        )
        self.assertAlmostEqual(
            result.metrics["gen9_tcn_static_shift_drop"],
            0.09364471919667683,
        )
        self.assertAlmostEqual(
            result.metrics["gen9_tcn_readout_adaptation_gain"],
            0.05601347594282539,
        )
        self.assertAlmostEqual(
            result.metrics["gen9_tcn_full_adaptation_gain"],
            0.08461714584764339,
        )
        self.assertEqual(result.metrics["gen9_qualified_arm_count"], 0.0)
        claims = {row["claim"]: row for row in result.claims}
        self.assertEqual(
            claims["Gen-9 sensor damage creates a non-trivial distribution shift"]["status"],
            "supported",
        )
        self.assertEqual(
            claims["The Gen-9 predictive LIF representation is source-competent"]["status"],
            "rejected",
        )
        self.assertEqual(
            claims["A frozen TCN representation adapts through a trainable readout"]["status"],
            "supported",
        )
        self.assertEqual(
            claims["Gen-9 qualifies for STW/LTW memory experiments"]["status"],
            "rejected",
        )

    def test_gen10_terminal_decision_is_in_final_ledger(self) -> None:
        result = synthesize_gen5_evidence(ROOT / "outputs")
        self.assertAlmostEqual(
            result.metrics["gen10_dropout_clean_gain_vs_tcn"],
            0.026837405553920113,
        )
        self.assertAlmostEqual(
            result.metrics["gen10_dropout_damaged_gain_vs_tcn"],
            0.0876263369639878,
        )
        self.assertAlmostEqual(
            result.metrics["gen10_lif_screen_clean_gap"],
            -0.09199999999999997,
        )
        self.assertAlmostEqual(
            result.metrics["gen10_lif_screen_spike_rate"],
            0.11538150972127914,
        )
        self.assertEqual(result.metrics["gen10_qualified_arm_count"], 0.0)
        claims = {row["claim"]: row for row in result.claims}
        self.assertEqual(
            claims["Sensor dropout improves conventional robustness in Gen-10"]["status"],
            "supported",
        )
        self.assertEqual(
            claims["The Gen-10 masked residual LIF representation is source-competent"]["status"],
            "rejected",
        )
        self.assertEqual(
            claims["Gen-10 qualifies a spiking representation for adaptation"]["status"],
            "rejected",
        )

    def test_gen11_terminal_decision_is_in_final_ledger(self) -> None:
        result = synthesize_gen5_evidence(ROOT / "outputs")
        self.assertAlmostEqual(
            result.metrics["gen11_readout_adaptation_gain"],
            0.02330487685212442,
        )
        self.assertAlmostEqual(
            result.metrics["gen11_full_adaptation_gain"],
            0.03295391358388119,
        )
        self.assertAlmostEqual(
            result.metrics["gen11_lif_adaptation_gain"],
            0.00783370948222284,
        )
        self.assertAlmostEqual(
            result.metrics["gen11_lif_state_specificity"],
            0.00011448009681743383,
        )
        self.assertEqual(result.metrics["gen11_qualified_arm_count"], 0.0)
        claims = {row["claim"]: row for row in result.claims}
        self.assertEqual(
            claims["Gen-11 state adapters improve damaged-task accuracy by the preregistered margin"]["status"],
            "rejected",
        )
        self.assertEqual(
            claims["Gen-11 LIF adaptation depends on sample-specific spiking state"]["status"],
            "rejected",
        )
        self.assertEqual(
            claims["Gen-11 qualifies for synaptic STW/LTW consolidation"]["status"],
            "rejected",
        )


if __name__ == "__main__":
    unittest.main()
