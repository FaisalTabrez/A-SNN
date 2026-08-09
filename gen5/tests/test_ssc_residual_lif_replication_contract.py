from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.ssc_benchmark import SSC_FILES
from ammc_gen5.ssc_residual_lif_replication import summarize_ssc_residual_lif_replication


class SSCResidualLIFReplicationContractTest(unittest.TestCase):
    def test_official_ssc_split_files(self) -> None:
        self.assertEqual(
            SSC_FILES,
            ("ssc_train.h5.gz", "ssc_valid.h5.gz", "ssc_test.h5.gz"),
        )

    def test_summary_reports_replication_gates(self) -> None:
        rows = []
        for seed in (142, 143, 144):
            rows.append(
                {
                    "seed": seed,
                    "conv_reference_accuracy": 0.70,
                    "full_accuracy": 0.72,
                    "direct_only_accuracy": 0.69,
                    "state_only_accuracy": 0.20,
                    "shuffled_state_accuracy": 0.68,
                    "full_gain_vs_conv": 0.02,
                    "state_contribution_vs_direct_only": 0.03,
                    "state_specificity_vs_shuffled": 0.04,
                    "direct_contribution_vs_state_only": 0.52,
                    "full_activity": 0.25,
                    "effective_trainable_parameters": 132000,
                    "parameter_ratio_vs_target": 0.988,
                    "train_samples": 75466,
                    "validation_samples": 9981,
                    "test_samples": 20382,
                    "conv_test_examples_per_second": 50000.0,
                    "full_test_examples_per_second": 19000.0,
                    "train_seconds": 100.0,
                }
            )
        summary = summarize_ssc_residual_lif_replication(rows)
        self.assertEqual(summary["within_two_points_seed_count_vs_conv"], 3)
        self.assertEqual(summary["one_point_seed_count_state_contribution"], 3)
        self.assertEqual(summary["one_point_seed_count_state_specificity"], 3)


if __name__ == "__main__":
    unittest.main()
