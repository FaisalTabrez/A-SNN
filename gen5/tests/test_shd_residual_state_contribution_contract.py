from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.shd_residual_state_contribution import (
    RESIDUAL_ABLATION_MODES,
    SHD_RESIDUAL_STATE_CONTRIBUTION_ARMS,
    available_shd_residual_state_contribution_arms,
    summarize_shd_residual_state_contribution,
)


class SHDResidualStateContributionContractTest(unittest.TestCase):
    def test_registered_dynamics_and_ablation_modes(self) -> None:
        self.assertEqual(len(SHD_RESIDUAL_STATE_CONTRIBUTION_ARMS), 2)
        self.assertEqual(
            available_shd_residual_state_contribution_arms(),
            ("residual_analog", "residual_lif"),
        )
        self.assertEqual(
            RESIDUAL_ABLATION_MODES,
            ("full", "direct_only", "state_only", "shuffled_state"),
        )

    def test_summary_reports_causal_contribution_gates(self) -> None:
        records = []
        for arm, dynamics in (("residual_analog", "analog"), ("residual_lif", "lif")):
            for seed in (1, 2, 3):
                records.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "dynamics": dynamics,
                        "conv_reference_accuracy": 0.82,
                        "full_accuracy": 0.84,
                        "direct_only_accuracy": 0.82,
                        "state_only_accuracy": 0.74,
                        "shuffled_state_accuracy": 0.81,
                        "full_gain_vs_conv": 0.02,
                        "state_contribution_vs_direct_only": 0.02,
                        "state_specificity_vs_shuffled": 0.03,
                        "direct_contribution_vs_state_only": 0.10,
                        "full_activity": 0.25,
                        "effective_trainable_parameters": 131956,
                        "parameter_ratio_vs_target": 0.987,
                        "train_seconds": 1.0,
                        "full_test_examples_per_second": 1000.0,
                    }
                )
        summary = summarize_shd_residual_state_contribution(records)
        self.assertEqual(len(summary), 2)
        self.assertEqual(summary[0]["one_point_seed_count_state_contribution"], 3)
        self.assertEqual(summary[1]["one_point_seed_count_state_specificity"], 3)


if __name__ == "__main__":
    unittest.main()
