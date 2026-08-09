from __future__ import annotations

import unittest

from ammc_gen5.event_mnist import EventMNISTConfig, FrozenEventReservoir, torch
from ammc_gen5.recurrence_ablation import (
    RECURRENCE_ABLATION_FEATURES,
    disable_recurrent_edges,
    summarize_recurrence_ablation,
)


class RecurrenceAblationContractTest(unittest.TestCase):
    def test_feature_set_contains_feedforward_and_recurrent_pairs(self) -> None:
        self.assertIn("hidden_feedforward_temporal", RECURRENCE_ABLATION_FEATURES)
        self.assertIn("hidden_recurrent_temporal", RECURRENCE_ABLATION_FEATURES)
        self.assertIn("full_feedforward_temporal", RECURRENCE_ABLATION_FEATURES)
        self.assertIn("full_recurrent_temporal", RECURRENCE_ABLATION_FEATURES)

    def test_summary_preserves_causal_deltas(self) -> None:
        base = {
            "feature": "full_recurrent_temporal",
            "classifier": "linear",
            "topology": "recurrent",
            "test_accuracy": 0.91,
            "accuracy_gain_vs_sensor": 0.02,
            "recurrence_gain": 0.01,
            "frozen_active_edges": 8,
            "feature_dim": 32,
            "trainable_parameters": 330,
            "mean_hidden_spike_rate": 0.02,
            "feature_seconds": 1.0,
            "feature_examples_per_second": 1000.0,
            "train_seconds": 1.0,
        }
        summary = summarize_recurrence_ablation(
            [base, {**base, "test_accuracy": 0.90, "recurrence_gain": -0.002}]
        )
        self.assertAlmostEqual(summary[0]["mean_recurrence_gain"], 0.004)
        self.assertEqual(summary[0]["recurrence_improved_seed_count"], 1)

    @unittest.skipIf(torch is None, "PyTorch is unavailable")
    def test_disabling_recurrence_preserves_sensor_edges(self) -> None:
        config = EventMNISTConfig(
            image_size=2,
            hidden_neurons=4,
            sensor_fanout=1,
            recurrent_fanout=1,
            max_edges=8,
            timesteps=4,
        )
        reservoir = FrozenEventReservoir(config, seed=42, device="cpu")
        self.assertEqual(reservoir.active_edge_count, 8)
        self.assertEqual(disable_recurrent_edges(reservoir), 4)
        self.assertEqual(reservoir.active_edge_count, 4)
        active_sources = reservoir.graph.sources[reservoir.graph.active_mask]
        self.assertTrue(bool((active_sources < config.sensor_neurons).all().item()))


if __name__ == "__main__":
    unittest.main()
