from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.shd_delay_interaction import (
    DELAY_PATTERNS,
    default_shd_delay_interaction_arms,
    summarize_shd_delay_interaction,
)


class SHDDelayInteractionContractTest(unittest.TestCase):
    def test_default_matrix_is_full_capacity_by_pattern_factorial(self) -> None:
        arms = default_shd_delay_interaction_arms()
        self.assertEqual(len(arms), 8)
        for hidden in (256, 512):
            patterns = {
                arm.delay_pattern for arm in arms if arm.hidden_neurons == hidden
            }
            self.assertEqual(patterns, set(DELAY_PATTERNS))

    def test_summary_keeps_heterogeneous_pass_evidence(self) -> None:
        arm = default_shd_delay_interaction_arms()[2]
        base = {
            "arm": arm.name,
            "test_accuracy": 0.60,
            "gain_vs_same_width_no_delay": 0.03,
            "final_hidden_event_rate": 0.2,
            "event_rate_vs_same_width_no_delay": 1.0,
            "active_edges": 100,
            "delayed_edges": 60,
            "effective_trainable_parameters": 1000,
            "mean_absolute_ltw_change": 0.01,
            "upper_ltw_saturation_rate": 0.0,
            "train_seconds": 1.0,
            "test_examples_per_second": 4000.0,
        }
        summary = summarize_shd_delay_interaction(
            [base, {**base, "gain_vs_same_width_no_delay": 0.015}], arms=(arm,)
        )
        self.assertEqual(summary[0]["improved_seed_count"], 2)
        self.assertEqual(summary[0]["one_point_seed_count"], 2)
        self.assertEqual(summary[0]["two_point_seed_count"], 1)


if __name__ == "__main__":
    unittest.main()
