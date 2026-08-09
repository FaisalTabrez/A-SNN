from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.shd_sparse_mechanisms import (
    SHD_SPARSE_MECHANISM_ARMS,
    available_shd_sparse_mechanism_arms,
    summarize_shd_sparse_mechanisms,
)


class SHDSparseMechanismsContractTest(unittest.TestCase):
    def test_matrix_is_paired_across_dynamics_and_ltw(self) -> None:
        self.assertEqual(len(SHD_SPARSE_MECHANISM_ARMS), 5)
        self.assertEqual(
            set(available_shd_sparse_mechanism_arms()),
            {
                "raw_temporal_pyramid",
                "sparse_lif_frozen_ltw",
                "sparse_lif_trainable_ltw",
                "sparse_analog_frozen_ltw",
                "sparse_analog_trainable_ltw",
            },
        )

    def test_summary_retains_spiking_and_ltw_gates(self) -> None:
        base = {
            "arm": "sparse_lif_trainable_ltw",
            "test_accuracy": 0.81,
            "gain_vs_raw_temporal": 0.03,
            "gain_vs_matched_analog": 0.025,
            "gain_vs_frozen_ltw": 0.015,
            "active_edges": 700,
            "effective_trainable_parameters": 133_631,
            "parameter_ratio_vs_target": 1.0,
            "final_activity": 0.08,
            "activity_kind": "hidden_spike_rate",
            "mean_absolute_ltw_change": 0.01,
            "train_seconds": 1.0,
            "test_examples_per_second": 7000.0,
        }
        summary = summarize_shd_sparse_mechanisms(
            [base, {**base, "gain_vs_matched_analog": 0.012}]
        )
        row = next(item for item in summary if item["arm"] == base["arm"])
        self.assertEqual(row["spiking_one_point_seed_count"], 2)
        self.assertEqual(row["ltw_one_point_seed_count"], 2)


if __name__ == "__main__":
    unittest.main()
