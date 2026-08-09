from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.shd_analog_topology import SHD_ANALOG_TOPOLOGY_ARMS, available_shd_analog_topology_arms, summarize_shd_analog_topology


class SHDAnalogTopologyContractTest(unittest.TestCase):
    def test_registered_matrix(self) -> None:
        self.assertEqual(len(SHD_ANALOG_TOPOLOGY_ARMS), 6)
        self.assertIn("dense_analog_feedforward", available_shd_analog_topology_arms())
        self.assertIn("sparse_analog_instant", available_shd_analog_topology_arms())
        self.assertIn("sparse_analog_leaky", available_shd_analog_topology_arms())

    def test_summary_retains_topology_and_leak_gates(self) -> None:
        base = {
            "arm": "sparse_analog_leaky", "test_accuracy": 0.82,
            "gain_vs_raw_temporal": 0.04, "gain_vs_dense_lif": 0.05,
            "gain_vs_dense_analog_feedforward": 0.03, "gain_vs_sparse_instant": 0.015,
            "hidden_neurons": 512, "active_edges": 700, "dynamics_parameters": 700,
            "effective_model_parameters": 133631, "trainable_parameters": 132931,
            "parameter_ratio_vs_target": 1.0, "final_activity": 0.33,
            "activity_kind": "analog_activation", "train_seconds": 1.0,
            "test_examples_per_second": 9000.0,
        }
        summary = summarize_shd_analog_topology([base, dict(base)])
        row = summary[0]
        self.assertEqual(row["sparse_one_point_seed_count"], 2)
        self.assertEqual(row["leak_positive_seed_count"], 2)


if __name__ == "__main__":
    unittest.main()
