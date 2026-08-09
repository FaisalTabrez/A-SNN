from __future__ import annotations

import unittest

from ammc_gen5.event_mnist import EventMNISTConfig, FrozenEventReservoir, torch
from ammc_gen5.temporal_mnist import TEMPORAL_FEATURES, summarize_temporal_mnist


class TemporalMNISTContractTest(unittest.TestCase):
    def test_temporal_feature_set_preserves_controls_and_residual(self) -> None:
        self.assertEqual(
            TEMPORAL_FEATURES,
            (
                "raw_intensity",
                "flattened_latency",
                "full_summary",
                "sensor_temporal",
                "hidden_temporal",
                "full_temporal",
                "raw_plus_hidden_temporal",
            ),
        )

    def test_summary_aggregates_temporal_records(self) -> None:
        base = {
            "feature": "raw_intensity",
            "classifier": "linear",
            "train_accuracy": 0.9,
            "feature_dim": 64,
            "trainable_parameters": 650,
            "classifier_hidden_units": 0,
            "frozen_active_edges": 0,
            "mean_hidden_spike_rate": 0.0,
            "feature_seconds": 0.0,
            "feature_examples_per_second": 0.0,
            "train_seconds": 1.0,
        }
        summary = summarize_temporal_mnist(
            [{**base, "test_accuracy": 0.8}, {**base, "test_accuracy": 0.9}]
        )
        self.assertEqual(summary[0]["feature"], "raw_intensity")
        self.assertEqual(summary[0]["classifier"], "linear")
        self.assertAlmostEqual(summary[0]["mean_test_accuracy"], 0.85)

    @unittest.skipIf(torch is None, "PyTorch is unavailable")
    def test_temporal_components_preserve_each_timestep(self) -> None:
        config = EventMNISTConfig(
            image_size=2,
            hidden_neurons=4,
            sensor_fanout=1,
            recurrent_fanout=1,
            max_edges=8,
            timesteps=4,
        )
        reservoir = FrozenEventReservoir(config, seed=42, device="cpu")
        components = reservoir.temporal_components(torch.ones((3, 4)))
        self.assertEqual(tuple(components["sensor_temporal"].shape), (3, 16))
        self.assertEqual(tuple(components["hidden_temporal"].shape), (3, 16))
        self.assertEqual(tuple(components["full_temporal"].shape), (3, 32))


if __name__ == "__main__":
    unittest.main()
