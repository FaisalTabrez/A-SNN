from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.shd_matched_baselines import (
    SHD_MATCHED_BASELINE_ARMS,
    available_shd_matched_baseline_arms,
    gru_parameter_count,
    matched_gru_hidden_units,
    summarize_shd_matched_baselines,
)


class SHDMatchedBaselinesContractTest(unittest.TestCase):
    def test_matrix_contains_standard_and_sparse_temporal_models(self) -> None:
        self.assertEqual(len(SHD_MATCHED_BASELINE_ARMS), 6)
        self.assertEqual(
            set(available_shd_matched_baseline_arms()),
            {
                "event_count_mlp",
                "raw_temporal_pyramid",
                "dense_lif_recurrent",
                "gru_temporal",
                "sparse512_feedforward_pyramid",
                "sparse512_recurrent_pyramid",
            },
        )

    def test_gru_width_is_parameter_matched(self) -> None:
        hidden, actual = matched_gru_hidden_units(700, 20, 135_679)
        self.assertEqual(hidden, 58)
        self.assertEqual(actual, gru_parameter_count(700, hidden, 20))
        self.assertLessEqual(actual, 135_679)
        self.assertGreaterEqual(actual / 135_679, 0.90)

    def test_summary_keeps_dense_lif_gate(self) -> None:
        base = {
            "arm": "sparse512_recurrent_pyramid",
            "test_accuracy": 0.82,
            "gain_vs_raw_temporal": 0.04,
            "gain_vs_dense_lif": 0.03,
            "gain_vs_gru": -0.02,
            "gain_vs_sparse_feedforward": 0.01,
            "topology": "sparse_recurrent_lif",
            "hidden_units": 512,
            "active_edges": 2748,
            "active_recurrent_edges": 2048,
            "effective_trainable_parameters": 135_679,
            "parameter_ratio_vs_target": 1.0,
            "final_activity": 0.13,
            "activity_kind": "hidden_spike_rate",
            "mean_absolute_ltw_change": 0.01,
            "upper_ltw_saturation_rate": 0.0,
            "train_seconds": 1.0,
            "test_examples_per_second": 5000.0,
        }
        summary = summarize_shd_matched_baselines(
            [base, {**base, "gain_vs_dense_lif": 0.015}]
        )
        row = next(
            item for item in summary if item["arm"] == "sparse512_recurrent_pyramid"
        )
        self.assertEqual(row["one_point_seed_count_vs_dense_lif"], 2)


if __name__ == "__main__":
    unittest.main()
