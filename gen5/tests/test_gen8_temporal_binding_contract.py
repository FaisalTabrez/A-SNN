from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.event_mnist import torch
from ammc_gen5.gen8_temporal_binding import (
    GEN8_TEMPORAL_BINDING_ARMS,
    TimeLocalBindingTCNClassifier,
    available_gen8_temporal_binding_arms,
    decide_gen8_temporal_binding,
    matched_temporal_binding_channels,
    select_gen8_promoted_arms,
)
from ammc_gen5.shd_benchmark import SHDConfig
from ammc_gen5.ssc_efficiency_baselines import (
    TemporalDilatedTCNClassifier,
    matched_temporal_tcn_channels,
)


class Gen8TemporalBindingContractTest(unittest.TestCase):
    def test_registered_matrix_is_frozen(self) -> None:
        self.assertEqual(
            available_gen8_temporal_binding_arms(),
            (
                "dilated_tcn",
                "lif_pooled_predictive",
                "analog_time_local_binding",
                "lif_shuffled_time_local",
                "lif_time_local_binding",
            ),
        )
        lookup = {arm.name: arm for arm in GEN8_TEMPORAL_BINDING_ARMS}
        self.assertEqual(
            lookup["lif_time_local_binding"].predictive_weight,
            lookup["lif_shuffled_time_local"].predictive_weight,
        )
        self.assertFalse(lookup["lif_time_local_binding"].shuffled_future_targets)
        self.assertTrue(lookup["lif_shuffled_time_local"].shuffled_future_targets)

    def test_binding_model_preserves_tcn_width_and_budget(self) -> None:
        levels = (1, 2, 4, 8)
        base_channels, _ = matched_temporal_tcn_channels(
            700,
            35,
            133631,
            input_kernel_size=5,
            hidden_kernel_size=3,
            temporal_levels=levels,
        )
        channels, actual = matched_temporal_binding_channels(
            700,
            35,
            133631,
            input_kernel_size=5,
            hidden_kernel_size=3,
            temporal_levels=levels,
        )
        self.assertEqual(channels, base_channels)
        self.assertGreaterEqual(actual / 133631, 0.95)
        self.assertLessEqual(actual / 133631, 1.05)

    def test_candidate_promotion_forces_all_mechanistic_controls(self) -> None:
        records = []
        for arm in GEN8_TEMPORAL_BINDING_ARMS:
            records.append(
                {
                    "arm": arm.name,
                    "best_validation_accuracy": (
                        0.60 if arm.name == "dilated_tcn" else 0.595
                    ),
                    "parameter_ratio_vs_target": 1.0,
                    "checkpoint_activity": 0.05,
                }
            )
        promoted = select_gen8_promoted_arms(
            records,
            promotion_margin=0.01,
            minimum_parameter_ratio=0.95,
            maximum_parameter_ratio=1.05,
            minimum_spike_rate=0.01,
            maximum_spike_rate=0.30,
        )
        self.assertEqual(promoted, available_gen8_temporal_binding_arms())

    def test_terminal_gate_requires_binding_gain_over_pooled_reference(self) -> None:
        common = {
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
            "mean_activity": 0.05,
            "mean_absolute_gate": 0.0,
        }
        baseline = dict(common, arm="dilated_tcn")
        pooled = dict(
            common,
            arm="lif_pooled_predictive",
            mean_state_specificity_vs_shuffled=0.001,
            mean_state_temporal_order_vs_reversed=0.001,
        )
        shuffled = dict(common, arm="lif_shuffled_time_local", mean_future_alignment_margin=0.01)
        candidate = dict(
            common,
            arm="lif_time_local_binding",
            mean_gain_vs_tcn=-0.005,
            mean_state_contribution_vs_direct_only=0.01,
            half_point_seed_count_state_contribution=3,
            mean_state_specificity_vs_shuffled=0.008,
            half_point_seed_count_state_specificity=2,
            mean_state_temporal_order_vs_reversed=0.008,
            half_point_seed_count_temporal_order=2,
            mean_future_alignment_margin=0.03,
            alignment_seed_count=3,
            mean_absolute_gate=0.03,
        )
        decision = decide_gen8_temporal_binding(
            [baseline, pooled, shuffled, candidate],
            accuracy_margin=0.01,
            causal_margin=0.005,
            alignment_margin=0.02,
            alignment_control_margin=0.01,
            binding_gain_margin=0.005,
            minimum_spike_rate=0.01,
            maximum_spike_rate=0.30,
            minimum_gate=0.01,
        )
        self.assertEqual(decision["status"], "pass")
        pooled["mean_state_specificity_vs_shuffled"] = 0.006
        stopped = decide_gen8_temporal_binding(
            [baseline, pooled, shuffled, candidate],
            accuracy_margin=0.01,
            causal_margin=0.005,
            alignment_margin=0.02,
            alignment_control_margin=0.01,
            binding_gain_margin=0.005,
            minimum_spike_rate=0.01,
            maximum_spike_rate=0.30,
            minimum_gate=0.01,
        )
        self.assertEqual(stopped["status"], "stop")

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_zero_binding_projection_exactly_preserves_tcn_logits(self) -> None:
        config = SHDConfig(input_neurons=8, classes=4, timesteps=8, epochs=1, batch_size=2)
        levels = (1, 2)
        base = TemporalDilatedTCNClassifier(
            config,
            channels=3,
            input_kernel_size=3,
            hidden_kernel_size=3,
            dilation=2,
            temporal_levels=levels,
        )
        successor = TimeLocalBindingTCNClassifier(
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
