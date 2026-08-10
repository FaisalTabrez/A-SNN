from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.event_mnist import torch
from ammc_gen5.gen6_successor import (
    GEN6_SUCCESSOR_ARMS,
    SharedResidualStateTCNClassifier,
    available_gen6_successor_arms,
    decide_gen6_successor,
    matched_shared_residual_channels,
    select_gen6_promoted_arms,
    shared_residual_parameter_count,
)
from ammc_gen5.shd_benchmark import SHDConfig
from ammc_gen5.ssc_efficiency_baselines import (
    TemporalDilatedTCNClassifier,
    matched_temporal_tcn_channels,
)


class Gen6SuccessorContractTest(unittest.TestCase):
    def test_registered_successor_matrix(self) -> None:
        self.assertEqual(len(GEN6_SUCCESSOR_ARMS), 3)
        self.assertEqual(
            available_gen6_successor_arms(),
            (
                "dilated_tcn",
                "shared_residual_analog",
                "shared_residual_lif",
            ),
        )

    def test_shared_successor_retains_tcn_width_and_budget(self) -> None:
        levels = (1, 2, 4, 8)
        base_channels, base_parameters = matched_temporal_tcn_channels(
            700,
            35,
            133631,
            input_kernel_size=5,
            hidden_kernel_size=3,
            temporal_levels=levels,
        )
        channels, actual = matched_shared_residual_channels(
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
            shared_residual_parameter_count(
                700,
                channels,
                35,
                input_kernel_size=5,
                hidden_kernel_size=3,
                temporal_levels=levels,
                spiking=True,
            ),
        )
        self.assertEqual(actual - base_parameters, 2 * channels + 35)
        self.assertLessEqual(actual, 133631)
        self.assertGreaterEqual(actual / 133631, 0.95)

    def test_promotion_is_validation_activity_and_budget_gated(self) -> None:
        common = {
            "parameter_ratio_vs_target": 0.99,
            "checkpoint_activity": 0.05,
        }
        rows = [
            dict(
                common,
                arm="dilated_tcn",
                best_validation_accuracy=0.60,
            ),
            dict(
                common,
                arm="shared_residual_analog",
                best_validation_accuracy=0.595,
            ),
            dict(
                common,
                arm="shared_residual_lif",
                best_validation_accuracy=0.592,
            ),
        ]
        promoted = select_gen6_promoted_arms(
            rows,
            promotion_margin=0.01,
            minimum_parameter_ratio=0.95,
            maximum_parameter_ratio=1.05,
            minimum_spike_rate=0.01,
            maximum_spike_rate=0.30,
        )
        self.assertEqual(
            promoted,
            ("dilated_tcn", "shared_residual_analog", "shared_residual_lif"),
        )
        rows[-1]["checkpoint_activity"] = 0.0
        self.assertNotIn(
            "shared_residual_lif",
            select_gen6_promoted_arms(
                rows,
                promotion_margin=0.01,
                minimum_parameter_ratio=0.95,
                maximum_parameter_ratio=1.05,
                minimum_spike_rate=0.01,
                maximum_spike_rate=0.30,
            ),
        )

    def test_terminal_decision_requires_all_lif_gates(self) -> None:
        baseline = {
            "arm": "dilated_tcn",
            "runs": 3,
            "mean_full_accuracy": 0.60,
            "mean_gain_vs_tcn": 0.0,
            "mean_state_contribution_vs_direct_only": 0.0,
            "half_point_seed_count_state_contribution": 0,
            "mean_state_specificity_vs_shuffled": 0.0,
            "half_point_seed_count_state_specificity": 0,
            "mean_activity": 0.4,
            "mean_absolute_gate": 0.0,
        }
        successor = dict(
            baseline,
            arm="shared_residual_lif",
            mean_full_accuracy=0.595,
            mean_gain_vs_tcn=-0.005,
            mean_state_contribution_vs_direct_only=0.01,
            half_point_seed_count_state_contribution=3,
            mean_state_specificity_vs_shuffled=0.008,
            half_point_seed_count_state_specificity=2,
            mean_activity=0.05,
            mean_absolute_gate=0.03,
        )
        decision = decide_gen6_successor(
            [baseline, successor],
            accuracy_margin=0.01,
            causal_margin=0.005,
            minimum_spike_rate=0.01,
            maximum_spike_rate=0.30,
            minimum_gate=0.01,
        )
        self.assertEqual(decision["status"], "pass")
        self.assertEqual(decision["qualified_arms"], ["shared_residual_lif"])
        successor["mean_absolute_gate"] = 0.0
        stopped = decide_gen6_successor(
            [baseline, successor],
            accuracy_margin=0.01,
            causal_margin=0.005,
            minimum_spike_rate=0.01,
            maximum_spike_rate=0.30,
            minimum_gate=0.01,
        )
        self.assertEqual(stopped["status"], "stop")

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_zero_gate_exactly_preserves_tcn_logits(self) -> None:
        config = SHDConfig(
            input_neurons=8,
            classes=4,
            timesteps=6,
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
        successor = SharedResidualStateTCNClassifier(
            config,
            channels=3,
            input_kernel_size=3,
            hidden_kernel_size=3,
            dilation=2,
            temporal_levels=levels,
            dynamics="lif",
            surrogate_slope=10.0,
        )
        successor.input_conv.load_state_dict(base.input_conv.state_dict())
        successor.hidden_conv.load_state_dict(base.hidden_conv.state_dict())
        successor.classifier.load_state_dict(base.classifier.state_dict())
        events = torch.randint(0, 2, (2, 6, 8), dtype=torch.uint8)
        with torch.no_grad():
            self.assertTrue(torch.equal(base(events), successor(events)))


if __name__ == "__main__":
    unittest.main()
