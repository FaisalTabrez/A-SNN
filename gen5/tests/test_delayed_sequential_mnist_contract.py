from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.delayed_sequential_mnist import (
    DELAYED_SEQUENTIAL_ARMS,
    DelayedSequentialClassifier,
    assign_fixed_delays,
    delayed_sparse_current,
    summarize_delayed_sequential_mnist,
)
from ammc_gen5.event_mnist import EventMNISTConfig, torch


class DelayedSequentialMNISTContractTest(unittest.TestCase):
    def test_arm_set_keeps_no_delay_controls_and_fixed_delay_patterns(self) -> None:
        self.assertEqual(
            tuple(arm.name for arm in DELAYED_SEQUENTIAL_ARMS),
            (
                "raw",
                "lif_no_delay_frozen",
                "lif_no_delay_warm_all",
                "recurrent_delay1_frozen",
                "recurrent_delay1_warm_all",
                "recurrent_hash012_warm_all",
                "recurrent_distance012_warm_all",
            ),
        )

    def test_summary_retains_paired_no_delay_gate(self) -> None:
        arm = DELAYED_SEQUENTIAL_ARMS[4]
        base = {
            "arm": arm.name,
            "classifier": "linear",
            "test_accuracy": 0.60,
            "accuracy_gain_vs_no_delay_control": 0.01,
            "active_edges": 6,
            "delayed_edges": 4,
            "mean_recurrent_delay": 1.0,
            "effective_trainable_parameters": 96,
            "event_rate_ratio": 1.0,
            "event_rate_vs_no_delay_control": 0.9,
            "mean_absolute_ltw_change": 0.01,
            "lower_ltw_saturation_rate": 0.0,
            "upper_ltw_saturation_rate": 0.0,
            "train_seconds": 1.0,
        }
        summary = summarize_delayed_sequential_mnist(
            [
                base,
                {
                    **base,
                    "test_accuracy": 0.58,
                    "accuracy_gain_vs_no_delay_control": -0.002,
                },
            ],
            arms=(arm,),
        )
        self.assertAlmostEqual(
            summary[0]["mean_accuracy_gain_vs_no_delay_control"], 0.004
        )
        self.assertEqual(summary[0]["improved_seed_count"], 1)
        self.assertEqual(summary[0]["practical_gain_seed_count"], 1)

    @unittest.skipIf(torch is None, "PyTorch is unavailable")
    def test_delay_assignment_preserves_sensor_edges(self) -> None:
        config = EventMNISTConfig(
            image_size=2,
            hidden_neurons=4,
            sensor_fanout=1,
            recurrent_fanout=1,
            max_edges=8,
            timesteps=2,
        )
        model = DelayedSequentialClassifier(
            config,
            seed=42,
            classifier="linear",
            train_ltw=True,
            surrogate_slope=10.0,
            delay_pattern="uniform_1",
            max_delay_steps=1,
            device="cpu",
        )
        sensor = model.graph.active_mask & (model.graph.sources < model.input_neurons)
        recurrent = model.graph.active_mask & ~sensor
        self.assertTrue(bool((model.graph.delay_steps[sensor] == 0).all().item()))
        self.assertTrue(bool((model.graph.delay_steps[recurrent] == 1).all().item()))
        self.assertEqual(int(recurrent.sum().item()), 4)

    @unittest.skipIf(torch is None, "PyTorch is unavailable")
    def test_delay_bucket_defers_edge_current(self) -> None:
        config = EventMNISTConfig(
            image_size=2,
            hidden_neurons=4,
            sensor_fanout=1,
            recurrent_fanout=1,
            max_edges=8,
            timesteps=2,
        )
        model = DelayedSequentialClassifier(
            config,
            seed=42,
            classifier="linear",
            train_ltw=False,
            surrogate_slope=10.0,
            delay_pattern="uniform_1",
            max_delay_steps=1,
            device="cpu",
        )
        with torch.no_grad():
            model.graph.active_mask.zero_()
            model.graph.active_mask[0] = True
            model.graph.sources[0] = model.input_neurons
            model.graph.targets[0] = model.input_neurons + 1
            model.graph.long_term_weight.zero_()
            model.graph.long_term_weight[0] = 1.0
            model.graph.signs[0] = 1.0
            model.graph.delay_steps[0] = 1
        old = torch.zeros((1, model.neuron_count))
        old[0, model.input_neurons] = 2.0
        current = torch.zeros_like(old)
        output = delayed_sparse_current(
            model.graph,
            [current, old],
            zero_state=torch.zeros_like(old),
            max_delay_steps=1,
        )
        self.assertEqual(float(output[0, model.input_neurons + 1].item()), 2.0)


if __name__ == "__main__":
    unittest.main()
