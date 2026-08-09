from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.shd_validation_checkpoint import SHD_VALIDATION_CHECKPOINT_ARMS, available_shd_validation_checkpoint_arms, summarize_shd_validation_checkpoint


class SHDValidationCheckpointContractTest(unittest.TestCase):
    def test_final_audit_has_raw_and_sparse(self) -> None:
        self.assertEqual(len(SHD_VALIDATION_CHECKPOINT_ARMS), 2)
        self.assertEqual(set(available_shd_validation_checkpoint_arms()), {"raw_temporal_pyramid", "sparse_analog_leaky_512"})

    def test_summary_reports_checkpoint_effect(self) -> None:
        base = {
            "arm": "sparse_analog_leaky_512", "final_test_accuracy": 0.78,
            "checkpoint_test_accuracy": 0.81, "checkpoint_gain_vs_final": 0.03,
            "checkpoint_gain_vs_raw": 0.025, "best_epoch": 9,
            "best_validation_accuracy": 0.80, "effective_model_parameters": 133631,
            "parameter_ratio_vs_target": 1.0, "train_seconds": 1.0,
        }
        summary = summarize_shd_validation_checkpoint([base, dict(base)])
        self.assertEqual(summary[0]["two_point_pair_count_vs_raw"], 2)
        self.assertAlmostEqual(summary[0]["mean_checkpoint_gain_vs_final"], 0.03)


if __name__ == "__main__":
    unittest.main()
