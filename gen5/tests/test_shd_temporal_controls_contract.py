from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.shd_temporal_controls import (
    SHD_TEMPORAL_CONTROL_ARMS,
    available_shd_temporal_control_arms,
    budget_matched_bottleneck,
    summarize_shd_temporal_controls,
)


class SHDTemporalControlsContractTest(unittest.TestCase):
    def test_arm_matrix_separates_readout_feedforward_and_recurrence(self) -> None:
        self.assertEqual(len(SHD_TEMPORAL_CONTROL_ARMS), 5)
        self.assertEqual(
            set(available_shd_temporal_control_arms()),
            {
                "event_count_mlp",
                "raw_temporal_pyramid",
                "sparse512_global",
                "sparse512_feedforward_pyramid",
                "sparse512_recurrent_pyramid",
            },
        )

    def test_raw_temporal_readout_matches_recurrent_budget(self) -> None:
        bottleneck, actual = budget_matched_bottleneck(
            trace_dim=700,
            final_dim=700,
            classes=20,
            projection_dim=32,
            temporal_levels=(1, 2, 4, 8),
            target_parameters=133_780,
        )
        self.assertEqual(bottleneck, 92)
        self.assertLessEqual(actual, 133_780)
        self.assertGreaterEqual(actual / 133_780, 0.99)

    def test_summary_retains_recurrence_and_raw_control_gates(self) -> None:
        base = {
            "arm": "sparse512_recurrent_pyramid",
            "test_accuracy": 0.82,
            "gain_vs_event_count": 0.30,
            "gain_vs_raw_temporal": 0.04,
            "gain_vs_sparse_global": 0.20,
            "recurrence_gain_vs_feedforward": 0.05,
            "topology": "recurrent_pyramid",
            "active_edges": 2748,
            "active_recurrent_edges": 2048,
            "feature_dim": 992,
            "effective_trainable_parameters": 135_679,
            "parameter_ratio_vs_recurrent_pyramid": 1.0,
            "final_event_rate": 0.13,
            "event_rate_kind": "hidden",
            "mean_absolute_ltw_change": 0.01,
            "upper_ltw_saturation_rate": 0.0,
            "train_seconds": 1.0,
            "test_examples_per_second": 5000.0,
        }
        summary = summarize_shd_temporal_controls(
            [base, {**base, "gain_vs_raw_temporal": 0.015}]
        )
        row = next(
            item for item in summary if item["arm"] == "sparse512_recurrent_pyramid"
        )
        self.assertEqual(row["one_point_seed_count_vs_raw"], 2)
        self.assertEqual(row["two_point_seed_count_recurrence"], 2)


if __name__ == "__main__":
    unittest.main()
