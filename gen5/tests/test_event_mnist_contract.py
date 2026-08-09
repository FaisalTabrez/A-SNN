from __future__ import annotations

import unittest

from ammc_gen5.event_mnist import (
    EventMNISTConfig,
    build_event_reservoir_edges,
    latency_encode,
    summarize_event_mnist_records,
    torch,
)


class EventMNISTContractTest(unittest.TestCase):
    def test_default_topology_is_sparse_reproducible_and_in_bounds(self) -> None:
        config = EventMNISTConfig()
        first = build_event_reservoir_edges(
            config.sensor_neurons,
            config.hidden_neurons,
            sensor_fanout=config.sensor_fanout,
            recurrent_fanout=config.recurrent_fanout,
            seed=42,
        )
        second = build_event_reservoir_edges(
            config.sensor_neurons,
            config.hidden_neurons,
            sensor_fanout=config.sensor_fanout,
            recurrent_fanout=config.recurrent_fanout,
            seed=42,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 384)
        self.assertLessEqual(len(first), config.max_edges)
        self.assertTrue(all(0 <= edge.source < config.neuron_count for edge in first))
        self.assertTrue(all(config.sensor_neurons <= edge.target < config.neuron_count for edge in first))
        self.assertEqual(len({(edge.source, edge.target) for edge in first}), len(first))

    def test_summary_aggregates_seed_accuracy(self) -> None:
        records = [
            {
                "model": "raw_pixel_linear",
                "test_accuracy": 0.8,
                "train_accuracy": 0.9,
                "feature_dim": 64,
                "trainable_parameters": 650,
                "classifier_hidden_units": 0,
                "frozen_active_edges": 0,
                "mean_hidden_spike_rate": 0.0,
                "feature_seconds": 0.0,
                "train_seconds": 2.0,
                "inference_examples_per_second": 1000.0,
            },
            {
                "model": "raw_pixel_linear",
                "test_accuracy": 0.9,
                "train_accuracy": 1.0,
                "feature_dim": 64,
                "trainable_parameters": 650,
                "classifier_hidden_units": 0,
                "frozen_active_edges": 0,
                "mean_hidden_spike_rate": 0.0,
                "feature_seconds": 0.0,
                "train_seconds": 4.0,
                "inference_examples_per_second": 2000.0,
            },
        ]
        summary = summarize_event_mnist_records(records)
        self.assertEqual(summary[0]["seeds"], 2)
        self.assertAlmostEqual(summary[0]["mean_test_accuracy"], 0.85)
        self.assertAlmostEqual(summary[0]["std_test_accuracy"], 0.05)
        self.assertAlmostEqual(summary[0]["mean_train_seconds"], 3.0)

    @unittest.skipIf(torch is None, "PyTorch is unavailable")
    def test_latency_encoder_fires_bright_pixels_earlier(self) -> None:
        pixels = torch.tensor([[1.0, 0.5, 0.0]])
        encoded = latency_encode(pixels, timesteps=5, event_threshold=0.05)
        self.assertEqual(tuple(encoded.shape), (5, 1, 3))
        self.assertEqual(int(encoded[:, 0, 0].argmax().item()), 0)
        self.assertEqual(int(encoded[:, 0, 1].argmax().item()), 2)
        self.assertEqual(float(encoded[:, 0, 2].sum().item()), 0.0)


if __name__ == "__main__":
    unittest.main()
