from __future__ import annotations

import unittest

from ammc_gen5.event_mnist import EventMNISTConfig, torch
from ammc_gen5.trainable_temporal_mnist import (
    TRAINABLE_TEMPORAL_GROUPS,
    SparseTemporalClassifier,
    summarize_trainable_temporal_mnist,
)


class TrainableTemporalMNISTContractTest(unittest.TestCase):
    def test_group_set_keeps_raw_and_frozen_controls(self) -> None:
        self.assertEqual(
            TRAINABLE_TEMPORAL_GROUPS,
            (
                "raw_linear",
                "raw_mlp",
                "frozen_temporal_linear",
                "frozen_temporal_mlp",
                "trained_ltw_temporal_linear",
                "trained_ltw_temporal_mlp",
            ),
        )

    def test_summary_aggregates_records(self) -> None:
        base = {
            "group": "trained_ltw_temporal_linear",
            "train_accuracy": 0.9,
            "active_edges": 8,
            "readout_parameters": 100,
            "optimizer_parameters": 108,
            "effective_trainable_parameters": 108,
            "mean_hidden_event_rate": 0.1,
            "mean_ltw": 0.4,
            "mean_absolute_ltw_change": 0.02,
            "train_seconds": 1.0,
            "end_to_end_examples_per_second": 1000.0,
        }
        summary = summarize_trainable_temporal_mnist(
            [{**base, "test_accuracy": 0.8}, {**base, "test_accuracy": 0.9}]
        )
        self.assertEqual(summary[0]["seeds"], 2)
        self.assertAlmostEqual(summary[0]["mean_test_accuracy"], 0.85)

    @unittest.skipIf(torch is None, "PyTorch is unavailable")
    def test_surrogate_gradient_reaches_active_ltws(self) -> None:
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
        logits = model(torch.ones((3, 4)))
        self.assertEqual(tuple(logits.shape), (3, 10))
        self.assertEqual(model.active_edge_count, 8)
        logits.sum().backward()
        self.assertIsNotNone(model.graph.long_term_weight.grad)


if __name__ == "__main__":
    unittest.main()
