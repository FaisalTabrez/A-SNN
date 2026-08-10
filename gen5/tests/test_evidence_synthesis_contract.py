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
            "milestone_a", "gen6",
        ))
        self.assertEqual(len(set(EVIDENCE_FILENAMES.values())), 8)

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


if __name__ == "__main__":
    unittest.main()
