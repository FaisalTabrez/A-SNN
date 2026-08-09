from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.shd_initialization_robustness import SHD_INITIALIZATION_ARMS, available_shd_initialization_arms, summarize_shd_initialization_robustness


class SHDInitializationRobustnessContractTest(unittest.TestCase):
    def test_registered_arms(self) -> None:
        self.assertEqual(len(SHD_INITIALIZATION_ARMS), 3)
        self.assertEqual(set(available_shd_initialization_arms()), {"raw_temporal_pyramid", "sparse_analog_leaky_512", "sparse_analog_leaky_1024"})

    def test_summary_reports_variance_components(self) -> None:
        base = {
            "arm": "sparse_analog_leaky_512", "test_accuracy": 0.80,
            "gain_vs_raw_temporal": 0.02, "gain_vs_sparse_512": 0.0,
            "topology_seed": 42, "readout_seed": 142,
            "connected_hidden_neurons": 390, "effective_model_parameters": 133631,
            "parameter_ratio_vs_target": 1.0, "final_activity": 0.33,
            "train_seconds": 1.0, "test_examples_per_second": 8000.0,
        }
        rows = [base, {**base, "readout_seed": 143, "test_accuracy": 0.82, "gain_vs_raw_temporal": 0.03}]
        summary = summarize_shd_initialization_robustness(rows)
        self.assertGreater(summary[0]["mean_within_topology_readout_std"], 0.0)
        self.assertEqual(summary[0]["positive_pair_count_vs_raw"], 2)


if __name__ == "__main__":
    unittest.main()
