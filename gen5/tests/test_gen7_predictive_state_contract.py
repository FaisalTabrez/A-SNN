from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.event_mnist import torch
from ammc_gen5.gen7_predictive_state import (
    GEN7_PREDICTIVE_STATE_ARMS,
    PredictiveStateTCNClassifier,
    available_gen7_predictive_state_arms,
    decide_gen7_predictive_state,
    matched_predictive_state_channels,
    predictive_state_parameter_count,
    select_gen7_promoted_arms,
)
from ammc_gen5.shd_benchmark import SHDConfig
from ammc_gen5.ssc_efficiency_baselines import (
    TemporalDilatedTCNClassifier,
    matched_temporal_tcn_channels,
)


class Gen7PredictiveStateContractTest(unittest.TestCase):
    def test_registered_matrix_has_paired_and_shuffled_objective_controls(self) -> None:
        self.assertEqual(len(GEN7_PREDICTIVE_STATE_ARMS), 5)
        self.assertEqual(
            available_gen7_predictive_state_arms(),
            (
                "dilated_tcn",
                "lif_no_predictive",
                "analog_paired_predictive",
                "lif_shuffled_predictive",
                "lif_paired_predictive",
            ),
        )
        lookup = {arm.name: arm for arm in GEN7_PREDICTIVE_STATE_ARMS}
        self.assertEqual(
            lookup["lif_paired_predictive"].predictive_weight,
            lookup["lif_shuffled_predictive"].predictive_weight,
        )
        self.assertFalse(lookup["lif_paired_predictive"].shuffled_future_targets)
        self.assertTrue(lookup["lif_shuffled_predictive"].shuffled_future_targets)

    def test_predictive_successor_preserves_tcn_width_and_budget(self) -> None:
        levels = (1, 2, 4, 8)
        base_channels, _ = matched_temporal_tcn_channels(
            700,
            35,
            133631,
            input_kernel_size=5,
            hidden_kernel_size=3,
            temporal_levels=levels,
        )
        channels, actual = matched_predictive_state_channels(
            700,
            35,
            133631,
            input_kernel_size=5,
            hidden_kernel_size=3,
            temporal_levels=levels,
        )
        self.assertEqual(channels, base_channels)
        self.assertEqual(
            actual,
            predictive_state_parameter_count(
                700,
                channels,
                35,
                input_kernel_size=5,
                hidden_kernel_size=3,
                temporal_levels=levels,
                spiking=True,
            ),
        )
        self.assertGreaterEqual(actual / 133631, 0.95)
        self.assertLessEqual(actual / 133631, 1.05)

    def test_candidate_promotion_forces_required_lif_controls(self) -> None:
        rows = []
        validations = {
            "dilated_tcn": 0.60,
            "lif_no_predictive": 0.50,
            "analog_paired_predictive": 0.50,
            "lif_shuffled_predictive": 0.50,
            "lif_paired_predictive": 0.595,
        }
        for arm in GEN7_PREDICTIVE_STATE_ARMS:
            rows.append(
                {
                    "arm": arm.name,
                    "best_validation_accuracy": validations[arm.name],
                    "parameter_ratio_vs_target": 1.0,
                    "checkpoint_activity": 0.05,
                }
            )
        promoted = select_gen7_promoted_arms(
            rows,
            promotion_margin=0.01,
            minimum_parameter_ratio=0.95,
            maximum_parameter_ratio=1.05,
            minimum_spike_rate=0.01,
            maximum_spike_rate=0.30,
        )
        self.assertEqual(
            promoted,
            (
                "dilated_tcn",
                "lif_no_predictive",
                "lif_shuffled_predictive",
                "lif_paired_predictive",
            ),
        )

    def test_terminal_gate_requires_pair_specific_prediction_and_state(self) -> None:
        baseline = {
            "arm": "dilated_tcn",
            "runs": 3,
            "mean_full_accuracy": 0.60,
            "mean_gain_vs_tcn": 0.0,
            "mean_state_contribution_vs_direct_only": 0.0,
            "half_point_seed_count_state_contribution": 0,
            "mean_state_specificity_vs_shuffled": 0.0,
            "half_point_seed_count_state_specificity": 0,
            "mean_state_temporal_order_vs_reversed": 0.0,
            "half_point_seed_count_temporal_order": 0,
            "mean_future_alignment_margin": 0.0,
            "alignment_seed_count": 0,
            "mean_activity": 0.4,
            "mean_absolute_gate": 0.0,
        }
        shuffled = dict(
            baseline,
            arm="lif_shuffled_predictive",
            mean_full_accuracy=0.59,
            mean_gain_vs_tcn=-0.01,
            mean_future_alignment_margin=0.01,
            mean_activity=0.05,
            mean_absolute_gate=0.03,
        )
        candidate = dict(
            shuffled,
            arm="lif_paired_predictive",
            mean_full_accuracy=0.595,
            mean_gain_vs_tcn=-0.005,
            mean_state_contribution_vs_direct_only=0.01,
            half_point_seed_count_state_contribution=3,
            mean_state_specificity_vs_shuffled=0.008,
            half_point_seed_count_state_specificity=2,
            mean_state_temporal_order_vs_reversed=0.007,
            half_point_seed_count_temporal_order=2,
            mean_future_alignment_margin=0.03,
            alignment_seed_count=3,
        )
        passed = decide_gen7_predictive_state(
            [baseline, shuffled, candidate],
            accuracy_margin=0.01,
            causal_margin=0.005,
            alignment_margin=0.02,
            alignment_control_margin=0.01,
            minimum_spike_rate=0.01,
            maximum_spike_rate=0.30,
            minimum_gate=0.01,
        )
        self.assertEqual(passed["status"], "pass")
        candidate["mean_state_specificity_vs_shuffled"] = -0.001
        stopped = decide_gen7_predictive_state(
            [baseline, shuffled, candidate],
            accuracy_margin=0.01,
            causal_margin=0.005,
            alignment_margin=0.02,
            alignment_control_margin=0.01,
            minimum_spike_rate=0.01,
            maximum_spike_rate=0.30,
            minimum_gate=0.01,
        )
        self.assertEqual(stopped["status"], "stop")

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_zero_conditional_gate_exactly_preserves_tcn_logits(self) -> None:
        config = SHDConfig(
            input_neurons=8,
            classes=4,
            timesteps=8,
            epochs=1,
            batch_size=2,
        )
        levels = (1, 2)
        base = TemporalDilatedTCNClassifier(
            config,
            channels=3,
            input_kernel_size=3,
            hidden_kernel_size=3,
            dilation=2,
            temporal_levels=levels,
        )
        successor = PredictiveStateTCNClassifier(
            config,
            channels=3,
            input_kernel_size=3,
            hidden_kernel_size=3,
            dilation=2,
            temporal_levels=levels,
            dynamics="lif",
            surrogate_slope=10.0,
            future_horizon=2,
        )
        successor.input_conv.load_state_dict(base.input_conv.state_dict())
        successor.hidden_conv.load_state_dict(base.hidden_conv.state_dict())
        successor.classifier.load_state_dict(base.classifier.state_dict())
        events = torch.randint(0, 2, (2, 8, 8), dtype=torch.uint8)
        with torch.no_grad():
            self.assertTrue(torch.equal(base(events), successor(events)))


if __name__ == "__main__":
    unittest.main()
