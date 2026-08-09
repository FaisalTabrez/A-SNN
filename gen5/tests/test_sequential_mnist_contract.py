from __future__ import annotations

import unittest

from ammc_gen5.event_mnist import EventMNISTConfig, torch
from ammc_gen5.sequential_mnist import (
    SEQUENTIAL_MNIST_FEATURES,
    StreamingMNISTReservoir,
    summarize_sequential_mnist,
)


class SequentialMNISTContractTest(unittest.TestCase):
    def test_feature_set_contains_memory_controls_and_causal_pair(self) -> None:
        self.assertEqual(
            SEQUENTIAL_MNIST_FEATURES,
            (
                "raw_flattened",
                "last_row",
                "integrated_rows",
                "hidden_feedforward_final",
                "hidden_recurrent_final",
            ),
        )

    def test_summary_preserves_recurrence_evidence(self) -> None:
        base = {
            "feature": "hidden_recurrent_final",
            "classifier": "linear",
            "topology": "recurrent",
            "test_accuracy": 0.81,
            "accuracy_gain_vs_last_row": 0.31,
            "accuracy_gain_vs_integrated_rows": 0.11,
            "recurrence_gain": 0.02,
            "frozen_active_edges": 6,
            "feature_dim": 8,
            "trainable_parameters": 90,
            "mean_hidden_spike_rate": 0.03,
            "feature_seconds": 1.0,
            "feature_examples_per_second": 100.0,
            "train_seconds": 1.0,
        }
        summary = summarize_sequential_mnist(
            [base, {**base, "test_accuracy": 0.80, "recurrence_gain": -0.004}]
        )
        self.assertAlmostEqual(summary[0]["mean_recurrence_gain"], 0.008)
        self.assertEqual(summary[0]["recurrence_improved_seed_count"], 1)

    @unittest.skipIf(torch is None, "PyTorch is unavailable")
    def test_feedforward_and_recurrent_edge_contract(self) -> None:
        config = EventMNISTConfig(
            image_size=2,
            hidden_neurons=4,
            sensor_fanout=1,
            recurrent_fanout=1,
            max_edges=8,
            timesteps=2,
        )
        feedforward = StreamingMNISTReservoir(config, seed=42, recurrent=False, device="cpu")
        recurrent = StreamingMNISTReservoir(config, seed=42, recurrent=True, device="cpu")
        self.assertEqual(feedforward.active_edge_count, 2)
        self.assertEqual(recurrent.active_edge_count, 6)
        features, rate = recurrent(torch.ones((3, 4)))
        self.assertEqual(tuple(features.shape), (3, 8))
        self.assertEqual(rate.ndim, 0)


if __name__ == "__main__":
    unittest.main()
