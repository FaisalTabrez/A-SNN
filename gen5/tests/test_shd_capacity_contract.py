from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.shd_capacity import (
    _next_power_of_two,
    default_shd_capacity_arms,
    summarize_shd_capacity,
)


class SHDCapacityContractTest(unittest.TestCase):
    def test_default_matrix_contains_no_delay_curve_and_one_delay_anchor(self) -> None:
        arms = default_shd_capacity_arms()
        self.assertEqual(
            tuple(arm.name for arm in arms),
            (
                "event_count_mlp",
                "sparse128_mlp_no_delay",
                "sparse192_mlp_no_delay",
                "sparse256_mlp_no_delay",
                "sparse384_mlp_no_delay",
                "sparse512_mlp_no_delay",
                "sparse256_mlp_distance012",
            ),
        )
        self.assertEqual(sum(arm.delay_pattern != "none" for arm in arms), 1)

    def test_summary_reports_capacity_gain_and_efficiency(self) -> None:
        arm = default_shd_capacity_arms()[3]
        base = {
            "arm": arm.name,
            "test_accuracy": 0.55,
            "gain_vs_128_no_delay": 0.10,
            "gain_vs_256_no_delay": 0.0,
            "gain_vs_same_scale_no_delay": 0.0,
            "final_hidden_event_rate": 0.2,
            "active_edges": 1700,
            "effective_trainable_parameters": 70_000,
            "mean_absolute_ltw_change": 0.02,
            "upper_ltw_saturation_rate": 0.0,
            "train_seconds": 1.0,
            "test_examples_per_second": 5000.0,
        }
        summary = summarize_shd_capacity([base], arms=(arm,))
        self.assertAlmostEqual(summary[0]["mean_gain_vs_128_no_delay"], 0.10)
        self.assertAlmostEqual(
            summary[0]["accuracy_per_1k_effective_parameters"], 0.55 / 70.0
        )

    def test_capacity_edge_pool_rounds_up(self) -> None:
        self.assertEqual(_next_power_of_two(2748), 4096)


if __name__ == "__main__":
    unittest.main()
