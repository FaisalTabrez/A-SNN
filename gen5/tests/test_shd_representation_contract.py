from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.shd_representation import (
    SHD_REPRESENTATION_ARMS,
    _next_power_of_two,
    summarize_shd_representation,
)


class SHDRepresentationContractTest(unittest.TestCase):
    def test_registered_matrix_is_factorized(self) -> None:
        self.assertEqual(
            tuple(arm.name for arm in SHD_REPRESENTATION_ARMS),
            (
                "event_count_linear",
                "event_count_mlp",
                "sparse128_linear_no_delay",
                "sparse128_mlp_no_delay",
                "sparse128_mlp_distance012",
                "sparse256_mlp_distance012",
                "sparse128_mlp_distance012_threshold1p5",
            ),
        )

    def test_summary_preserves_relevant_control_gate(self) -> None:
        arm = SHD_REPRESENTATION_ARMS[3]
        base = {
            "arm": arm.name,
            "hidden_neurons": 128,
            "reservoir_threshold": 1.0,
            "test_accuracy": 0.40,
            "gain_vs_sparse_linear": 0.04,
            "gain_vs_mlp_no_delay": 0.0,
            "gain_vs_base_distance": 0.0,
            "gain_vs_relevant_control": 0.04,
            "final_hidden_event_rate": 0.2,
            "event_rate_ratio": 1.0,
            "active_edges": 100,
            "effective_trainable_parameters": 500,
            "mean_absolute_ltw_change": 0.01,
            "upper_ltw_saturation_rate": 0.0,
            "train_seconds": 1.0,
        }
        summary = summarize_shd_representation(
            [base, {**base, "gain_vs_relevant_control": 0.02}], arms=(arm,)
        )
        self.assertEqual(summary[0]["improved_seed_count_vs_relevant_control"], 2)
        self.assertEqual(summary[0]["practical_seed_count_vs_relevant_control"], 1)

    def test_capacity_rounds_up_to_stable_edge_pool(self) -> None:
        self.assertEqual(_next_power_of_two(1724), 2048)
        self.assertEqual(_next_power_of_two(2049), 4096)


if __name__ == "__main__":
    unittest.main()
