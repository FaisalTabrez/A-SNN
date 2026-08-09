from __future__ import annotations

import unittest

from ammc_gen5.event_mnist import EventMNISTConfig, torch
from ammc_gen5.structural_sequential_mnist import (
    STRUCTURAL_SEQUENTIAL_ARMS,
    sprout_targeted_edges,
    summarize_structural_sequential_mnist,
)
from ammc_gen5.trainable_sequential_mnist import TrainableSequentialClassifier


class StructuralSequentialMNISTContractTest(unittest.TestCase):
    def test_arm_set_preserves_controls_and_localization_arms(self) -> None:
        names = tuple(arm.name for arm in STRUCTURAL_SEQUENTIAL_ARMS)
        self.assertEqual(
            names,
            (
                "raw",
                "frozen_recurrent",
                "fixed_warm_all",
                "sensor_sprout_16",
                "sensor_sprout_48",
                "recurrent_sprout_64",
            ),
        )

    def test_summary_retains_fixed_topology_comparison(self) -> None:
        arm = STRUCTURAL_SEQUENTIAL_ARMS[3]
        base = {
            "arm": arm.name,
            "classifier": "linear",
            "test_accuracy": 0.62,
            "accuracy_gain_vs_fixed_warm_all": 0.02,
            "accuracy_gain_vs_frozen": 0.04,
            "initial_active_edges": 6,
            "final_active_edges": 8,
            "sprouted_edges": 2,
            "effective_trainable_parameters": 98,
            "event_rate_ratio": 1.1,
            "mean_core_ltw_change": 0.01,
            "mean_sprouted_ltw_change_from_birth": 0.02,
            "lower_ltw_saturation_rate": 0.0,
            "upper_ltw_saturation_rate": 0.0,
            "train_seconds": 1.0,
            "end_to_end_examples_per_second": 100.0,
        }
        summary = summarize_structural_sequential_mnist(
            [
                base,
                {
                    **base,
                    "test_accuracy": 0.60,
                    "accuracy_gain_vs_fixed_warm_all": -0.002,
                },
            ],
            arms=(arm,),
        )
        self.assertAlmostEqual(
            summary[0]["mean_accuracy_gain_vs_fixed_warm_all"], 0.009
        )
        self.assertEqual(summary[0]["fixed_improved_seed_count"], 1)
        self.assertEqual(summary[0]["fixed_practical_gain_seed_count"], 1)

    @unittest.skipIf(torch is None, "PyTorch is unavailable")
    def test_targeted_sprouting_adds_unique_edges_and_preserves_core(self) -> None:
        config = EventMNISTConfig(
            image_size=2,
            hidden_neurons=4,
            sensor_fanout=1,
            recurrent_fanout=1,
            max_edges=16,
            timesteps=2,
        )
        model = TrainableSequentialClassifier(
            config,
            seed=42,
            classifier="linear",
            train_ltw=True,
            surrogate_slope=10.0,
            device="cpu",
        )
        original = set(
            (edge.source, edge.target) for edge in model.graph.edge_records()
        )
        slots = sprout_targeted_edges(
            model, "sensor", 2, seed=42, birth_weight=0.1
        )
        final = [(edge.source, edge.target) for edge in model.graph.edge_records()]
        self.assertEqual(len(slots), 2)
        self.assertEqual(model.active_edge_count, 8)
        self.assertTrue(original.issubset(set(final)))
        self.assertEqual(len(final), len(set(final)))

        larger = TrainableSequentialClassifier(
            config,
            seed=42,
            classifier="linear",
            train_ltw=True,
            surrogate_slope=10.0,
            device="cpu",
        )
        sprout_targeted_edges(larger, "sensor", 4, seed=42, birth_weight=0.1)
        larger_edges = set(
            (edge.source, edge.target) for edge in larger.graph.edge_records()
        )
        self.assertTrue((set(final) - original).issubset(larger_edges - original))


if __name__ == "__main__":
    unittest.main()
