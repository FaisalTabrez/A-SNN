from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.adaptive_sequential_mnist import (
    ADAPTIVE_SEQUENTIAL_ARMS,
    AdaptiveSequentialClassifier,
    summarize_adaptive_sequential_mnist,
)
from ammc_gen5.event_mnist import EventMNISTConfig, torch


class AdaptiveSequentialMNISTContractTest(unittest.TestCase):
    def test_arm_set_keeps_paired_lif_controls_and_adaptive_dose(self) -> None:
        self.assertEqual(
            tuple(arm.name for arm in ADAPTIVE_SEQUENTIAL_ARMS),
            (
                "raw",
                "lif_frozen",
                "lif_warm_all",
                "alif50_frozen",
                "alif25_warm_all",
                "alif50_warm_all",
                "alif100_warm_all",
            ),
        )
        self.assertEqual(
            tuple(arm.adaptive_fraction for arm in ADAPTIVE_SEQUENTIAL_ARMS[-3:]),
            (0.25, 0.5, 1.0),
        )

    def test_summary_retains_paired_lif_gate(self) -> None:
        arm = ADAPTIVE_SEQUENTIAL_ARMS[5]
        base = {
            "arm": arm.name,
            "classifier": "linear",
            "test_accuracy": 0.60,
            "accuracy_gain_vs_lif_control": 0.01,
            "active_edges": 6,
            "adaptive_neurons": 2,
            "effective_trainable_parameters": 96,
            "event_rate_ratio": 1.0,
            "event_rate_vs_lif_control": 0.9,
            "final_mean_adaptation": 0.4,
            "final_mean_adaptive_threshold": 1.2,
            "mean_absolute_ltw_change": 0.01,
            "lower_ltw_saturation_rate": 0.0,
            "upper_ltw_saturation_rate": 0.0,
            "train_seconds": 1.0,
        }
        summary = summarize_adaptive_sequential_mnist(
            [base, {**base, "test_accuracy": 0.58, "accuracy_gain_vs_lif_control": -0.002}],
            arms=(arm,),
        )
        self.assertAlmostEqual(summary[0]["mean_accuracy_gain_vs_lif_control"], 0.004)
        self.assertEqual(summary[0]["improved_seed_count"], 1)
        self.assertEqual(summary[0]["practical_gain_seed_count"], 1)

    @unittest.skipIf(torch is None, "PyTorch is unavailable")
    def test_adaptation_changes_threshold_without_adding_parameters(self) -> None:
        config = EventMNISTConfig(
            image_size=2,
            hidden_neurons=4,
            sensor_fanout=1,
            recurrent_fanout=1,
            max_edges=8,
            timesteps=2,
            input_gain=4.0,
        )
        model = AdaptiveSequentialClassifier(
            config,
            seed=42,
            classifier="linear",
            train_ltw=True,
            surrogate_slope=10.0,
            adaptive_fraction=0.5,
            adaptation_decay=0.95,
            adaptation_strength=0.5,
            device="cpu",
        )
        self.assertEqual(model.active_edge_count, 6)
        self.assertEqual(model.adaptive_neuron_count, 2)
        self.assertFalse(model.adaptive_mask.requires_grad)
        logits, event_rate, adaptation, threshold = model(
            torch.ones((3, 4)), return_diagnostics=True
        )
        self.assertEqual(tuple(logits.shape), (3, 10))
        self.assertEqual(event_rate.ndim, 0)
        self.assertGreaterEqual(float(adaptation.item()), 0.0)
        self.assertGreaterEqual(float(threshold.item()), 1.0)


if __name__ == "__main__":
    unittest.main()
