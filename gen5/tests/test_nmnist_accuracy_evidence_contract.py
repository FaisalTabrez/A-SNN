from __future__ import annotations

import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "outputs"
    / "nmnist_accuracy_benchmark_log_recovery_2026-08-11"
    / "nmnist_accuracy_benchmark.json"
)


class NmnistAccuracyEvidenceContractTest(unittest.TestCase):
    def test_recovered_terminal_decision_and_confirmation_are_frozen(self) -> None:
        payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        decision = payload["decision"]
        self.assertEqual(payload["evidence_status"], "log_recovery")
        self.assertEqual(decision["status"], "pass")
        self.assertTrue(decision["practical_gate_99_0"])
        self.assertTrue(decision["stretch_gate_99_4"])
        self.assertFalse(decision["spiking_confirmed"])
        self.assertEqual(decision["next_milestone"], "return_to_gen20")

        summary = {row["arm"]: row for row in payload["summary"]}
        self.assertAlmostEqual(
            summary["spatiotemporal_cnn"]["mean_test_accuracy"],
            0.9947666666666667,
        )
        self.assertAlmostEqual(
            summary["frame_cnn"]["mean_test_accuracy"],
            0.9912333333333333,
        )
        self.assertEqual(len(payload["confirmation_records"]), 6)

    def test_spiking_arm_was_not_promoted(self) -> None:
        payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        screen = {row["arm"]: row for row in payload["screen_records"]}
        self.assertAlmostEqual(
            screen["spatiotemporal_cnn"]["best_validation_accuracy"], 0.987
        )
        self.assertAlmostEqual(screen["conv_plif"]["best_validation_accuracy"], 0.9307)
        self.assertNotIn("conv_plif", payload["promoted_arms"])


if __name__ == "__main__":
    unittest.main()
