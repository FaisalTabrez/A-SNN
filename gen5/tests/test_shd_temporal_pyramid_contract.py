from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.shd_temporal_pyramid import (
    default_shd_temporal_pyramid_arms,
    parameter_matched_bottleneck,
    summarize_shd_temporal_pyramid,
)


class SHDTemporalPyramidContractTest(unittest.TestCase):
    def test_default_matrix_contains_width_and_shuffle_controls(self) -> None:
        arms = default_shd_temporal_pyramid_arms()
        self.assertEqual(len(arms), 5)
        self.assertEqual(sum(arm.readout_mode == "global" for arm in arms), 2)
        shuffled = [arm for arm in arms if arm.temporal_order == "fixed_shuffle"]
        self.assertEqual(len(shuffled), 1)
        self.assertEqual(shuffled[0].hidden_neurons, 512)

    def test_pyramid_readout_is_parameter_matched(self) -> None:
        for hidden in (256, 512):
            bottleneck, actual, baseline = parameter_matched_bottleneck(
                hidden_neurons=hidden,
                classes=20,
                projection_dim=32,
                temporal_levels=(1, 2, 4, 8),
                baseline_hidden_units=128,
            )
            self.assertGreater(bottleneck, 0)
            self.assertLessEqual(actual, baseline)
            self.assertGreaterEqual(actual / baseline, 0.90)

    def test_summary_keeps_global_and_shuffle_gates(self) -> None:
        arm = default_shd_temporal_pyramid_arms()[3]
        base = {
            "arm": arm.name,
            "test_accuracy": 0.65,
            "gain_vs_same_width_global": 0.04,
            "gain_vs_same_arch_shuffled": 0.03,
            "final_hidden_event_rate": 0.14,
            "event_rate_vs_same_width_global": 1.0,
            "active_edges": 2748,
            "feature_dim": 992,
            "readout_bottleneck_units": 115,
            "effective_trainable_parameters": 135_000,
            "parameter_ratio_vs_same_width_global": 0.995,
            "mean_absolute_ltw_change": 0.02,
            "upper_ltw_saturation_rate": 0.0,
            "train_seconds": 1.0,
            "test_examples_per_second": 4000.0,
        }
        summary = summarize_shd_temporal_pyramid(
            [base, {**base, "gain_vs_same_width_global": 0.025}], arms=(arm,)
        )
        self.assertEqual(summary[0]["two_point_seed_count_vs_global"], 2)
        self.assertEqual(summary[0]["three_point_seed_count_vs_global"], 1)
        self.assertEqual(summary[0]["two_point_seed_count_vs_shuffled"], 2)


if __name__ == "__main__":
    unittest.main()
