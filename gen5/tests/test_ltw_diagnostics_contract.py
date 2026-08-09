from __future__ import annotations

import unittest

from ammc_gen5.event_mnist import EventMNISTConfig, torch
from ammc_gen5.ltw_diagnostics import (
    LTW_DIAGNOSTIC_ARMS,
    available_ltw_diagnostic_arms,
    ltw_scope_mask,
    summarize_ltw_diagnostic,
)
from ammc_gen5.trainable_temporal_mnist import SparseTemporalClassifier


class LTWDiagnosticsContractTest(unittest.TestCase):
    def test_default_sweep_preserves_frozen_and_phase21_controls(self) -> None:
        names = available_ltw_diagnostic_arms()
        self.assertEqual(names[0], "frozen")
        self.assertIn("joint_all_1em3_s10", names)
        self.assertIn("warm_sensor_3em4_s10", names)
        self.assertIn("warm_recurrent_3em4_s10", names)

    def test_summary_retains_paired_gain_and_stability_counts(self) -> None:
        arm = LTW_DIAGNOSTIC_ARMS[1]
        base = {
            "arm": arm.name,
            "classifier": "linear",
            "test_accuracy": 0.91,
            "accuracy_gain_vs_frozen": 0.01,
            "active_edges": 8,
            "scope_trainable_edges": 8,
            "effective_trainable_parameters": 108,
            "event_rate_ratio": 1.1,
            "mean_absolute_ltw_change": 0.01,
            "mean_sensor_ltw_change": 0.01,
            "mean_recurrent_ltw_change": 0.01,
            "lower_ltw_saturation_rate": 0.0,
            "upper_ltw_saturation_rate": 0.0,
            "train_seconds": 1.0,
            "end_to_end_examples_per_second": 1000.0,
        }
        summary = summarize_ltw_diagnostic(
            [base, {**base, "test_accuracy": 0.90, "accuracy_gain_vs_frozen": -0.001}],
            arms=(arm,),
        )
        self.assertAlmostEqual(summary[0]["mean_accuracy_gain_vs_frozen"], 0.0045)
        self.assertEqual(summary[0]["improved_seed_count"], 1)
        self.assertEqual(summary[0]["practical_gain_seed_count"], 1)

    @unittest.skipIf(torch is None, "PyTorch is unavailable")
    def test_scope_masks_partition_active_edges(self) -> None:
        config = EventMNISTConfig(
            image_size=2,
            hidden_neurons=4,
            sensor_fanout=1,
            recurrent_fanout=1,
            max_edges=8,
            timesteps=4,
        )
        model = SparseTemporalClassifier(
            config,
            seed=42,
            classifier="linear",
            hidden_units=1,
            train_ltw=True,
            surrogate_slope=10.0,
            device="cpu",
        )
        sensor = ltw_scope_mask(model, "sensor")
        recurrent = ltw_scope_mask(model, "recurrent")
        self.assertEqual(int(sensor.sum().item()), 4)
        self.assertEqual(int(recurrent.sum().item()), 4)
        self.assertTrue(torch.equal(sensor | recurrent, model.graph.active_mask))


if __name__ == "__main__":
    unittest.main()
