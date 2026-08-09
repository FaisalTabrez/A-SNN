from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.event_mnist import EventMNISTConfig, torch
from ammc_gen5.trainable_delays_mnist import (
    TRAINABLE_DELAY_ARMS,
    TrainableDelaySequentialClassifier,
    summarize_trainable_delays_mnist,
)


class TrainableDelaysMNISTContractTest(unittest.TestCase):
    def test_arm_set_keeps_fixed_winner_and_learned_controls(self) -> None:
        self.assertEqual(
            tuple(arm.name for arm in TRAINABLE_DELAY_ARMS),
            (
                "raw",
                "lif_no_delay_warm_all",
                "fixed_distance012_warm_all",
                "learned_soft_distance_init",
                "learned_st_distance_init",
                "learned_soft_flat_init",
            ),
        )

    def test_summary_keeps_paired_fixed_delay_gate(self) -> None:
        arm = TRAINABLE_DELAY_ARMS[3]
        base = {
            "arm": arm.name,
            "classifier": "linear",
            "test_accuracy": 0.65,
            "accuracy_gain_vs_fixed_distance": 0.01,
            "effective_trainable_parameters": 100,
            "changed_delay_assignments": 8,
            "mean_recurrent_delay": 1.0,
            "final_delay_entropy": 0.4,
            "event_rate_ratio": 1.0,
            "mean_absolute_ltw_change": 0.01,
            "lower_ltw_saturation_rate": 0.0,
            "upper_ltw_saturation_rate": 0.0,
            "train_seconds": 1.0,
        }
        summary = summarize_trainable_delays_mnist(
            [base, {**base, "accuracy_gain_vs_fixed_distance": -0.002}],
            arms=(arm,),
        )
        self.assertAlmostEqual(
            summary[0]["mean_accuracy_gain_vs_fixed_distance"], 0.004
        )
        self.assertEqual(summary[0]["improved_seed_count"], 1)
        self.assertEqual(summary[0]["practical_gain_seed_count"], 1)

    @unittest.skipIf(torch is None, "PyTorch is unavailable")
    def test_delay_logits_are_recurrent_only_trainable_state(self) -> None:
        config = EventMNISTConfig(
            image_size=2,
            hidden_neurons=4,
            sensor_fanout=1,
            recurrent_fanout=1,
            max_edges=8,
            timesteps=2,
        )
        model = TrainableDelaySequentialClassifier(
            config,
            seed=42,
            classifier="linear",
            surrogate_slope=10.0,
            gate_mode="straight_through",
            delay_initialization="distance",
            train_delays=True,
            device="cpu",
        )
        self.assertTrue(model.delay_logits.requires_grad)
        self.assertEqual(tuple(model.delay_logits.shape), (8, 3))
        gates = model.delay_gates()
        self.assertTrue(
            bool((gates.sum(dim=1) - 1.0).abs().max().item() < 1e-6)
        )
        sensor = model.graph.active_mask & (model.graph.sources < model.input_neurons)
        self.assertTrue(bool((gates[sensor, 0] == 1.0).all().item()))
        logits = model(torch.ones((3, 4)))
        logits.sum().backward()
        self.assertIsNotNone(model.delay_logits.grad)


if __name__ == "__main__":
    unittest.main()
