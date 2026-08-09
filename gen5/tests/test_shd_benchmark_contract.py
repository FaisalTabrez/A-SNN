from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.shd_benchmark import (
    SHD_ARMS,
    SHDConfig,
    SHDSparseClassifier,
    bin_shd_events,
    summarize_shd_benchmark,
)
from ammc_gen5.event_mnist import torch


class SHDBenchmarkContractTest(unittest.TestCase):
    def test_registered_arms_include_timing_ablation_and_paired_delay(self) -> None:
        self.assertEqual(
            tuple(arm.name for arm in SHD_ARMS),
            (
                "event_count_linear",
                "event_count_mlp",
                "sparse_no_delay_warm_all",
                "sparse_distance012_warm_all",
            ),
        )

    def test_summary_retains_one_point_transfer_gate(self) -> None:
        arm = SHD_ARMS[-1]
        base = {
            "arm": arm.name,
            "test_accuracy": 0.55,
            "accuracy_gain_vs_no_delay": 0.015,
            "active_edges": 12,
            "delayed_edges": 4,
            "mean_recurrent_delay": 1.0,
            "effective_trainable_parameters": 100,
            "allocated_trainable_parameters": 120,
            "event_rate_ratio": 1.0,
            "event_rate_vs_no_delay": 0.95,
            "mean_absolute_ltw_change": 0.01,
            "lower_ltw_saturation_rate": 0.0,
            "upper_ltw_saturation_rate": 0.0,
            "train_seconds": 1.0,
        }
        summary = summarize_shd_benchmark(
            [base, {**base, "accuracy_gain_vs_no_delay": -0.002}],
            arms=(arm,),
        )
        self.assertAlmostEqual(summary[0]["mean_accuracy_gain_vs_no_delay"], 0.0065)
        self.assertEqual(summary[0]["improved_seed_count"], 1)
        self.assertEqual(summary[0]["practical_gain_seed_count"], 1)

    @unittest.skipIf(torch is None, "PyTorch is unavailable")
    def test_event_binning_preserves_channel_and_time(self) -> None:
        events = bin_shd_events(
            [0.0, 0.25, 0.99, 1.5],
            [0, 3, 6, 1],
            timesteps=4,
            input_neurons=7,
            duration_seconds=1.0,
        )
        self.assertEqual(tuple(events.shape), (4, 7))
        self.assertEqual(int(events.sum().item()), 3)
        self.assertEqual(int(events[0, 0].item()), 1)
        self.assertEqual(int(events[1, 3].item()), 1)
        self.assertEqual(int(events[3, 6].item()), 1)

    @unittest.skipIf(torch is None, "PyTorch is unavailable")
    def test_distance_delays_leave_sensors_at_zero(self) -> None:
        config = SHDConfig(
            input_neurons=700,
            classes=20,
            timesteps=4,
            hidden_neurons=4,
            sensor_fanout=1,
            recurrent_fanout=1,
            max_edges=704,
            epochs=1,
            warmup_epochs=0,
        )
        model = SHDSparseClassifier(
            config,
            seed=42,
            delay_pattern="distance_0_2",
            max_delay_steps=2,
            surrogate_slope=10.0,
            device="cpu",
        )
        sensor = model.graph.active_mask & (model.graph.sources < config.input_neurons)
        recurrent = model.graph.active_mask & ~sensor
        self.assertTrue(bool((model.graph.delay_steps[sensor] == 0).all().item()))
        self.assertTrue(bool((model.graph.delay_steps[recurrent] <= 2).all().item()))
        output = model(torch.zeros((2, 4, 700), dtype=torch.uint8))
        self.assertEqual(tuple(output.shape), (2, 20))


if __name__ == "__main__":
    unittest.main()
