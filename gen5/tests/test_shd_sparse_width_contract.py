from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.shd_sparse_width import default_shd_sparse_width_arms, summarize_shd_sparse_width


class SHDSparseWidthContractTest(unittest.TestCase):
    def test_default_matrix(self) -> None:
        arms = default_shd_sparse_width_arms()
        self.assertEqual([arm.hidden_neurons for arm in arms], [0, 128, 256, 512, 1024])

    def test_summary_keeps_width_and_occupancy_metrics(self) -> None:
        base = {
            "arm": "sparse_analog_leaky_512", "hidden_neurons": 512,
            "test_accuracy": 0.82, "gain_vs_raw_temporal": 0.04,
            "gain_vs_smallest_width": 0.03, "gain_vs_previous_width": 0.015,
            "active_edges": 700, "connected_hidden_neurons": 380,
            "hidden_occupancy_rate": 380 / 512, "mean_sensor_fanin_connected": 700 / 380,
            "effective_model_parameters": 133631, "parameter_ratio_vs_target": 1.0,
            "final_activity": 0.33, "train_seconds": 1.0,
            "test_examples_per_second": 8000.0,
        }
        summary = summarize_shd_sparse_width([base, dict(base)])
        self.assertEqual(summary[0]["previous_width_positive_seed_count"], 2)
        self.assertAlmostEqual(summary[0]["mean_connected_hidden_neurons"], 380.0)


if __name__ == "__main__":
    unittest.main()
