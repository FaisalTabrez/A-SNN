from __future__ import annotations

import unittest

from ammc_gen5.event_mnist import EventMNISTConfig, torch
from ammc_gen5.trainable_sequential_mnist import TrainableSequentialClassifier
from ammc_gen5.utility_gated_structural_mnist import (
    UTILITY_GATED_ARMS,
    build_sensor_candidate_pool,
    prune_weak_sprouted_edges,
    summarize_utility_gated_structural_mnist,
)


class UtilityGatedStructuralMNISTContractTest(unittest.TestCase):
    def test_arm_set_has_paired_random_control_and_peripheral_pruning(self) -> None:
        self.assertEqual(
            tuple(arm.name for arm in UTILITY_GATED_ARMS),
            (
                "raw",
                "frozen_recurrent",
                "fixed_warm_all",
                "random_sensor_48",
                "gradient_sensor_16",
                "gradient_sensor_48",
                "gradient_sensor_48_prune",
            ),
        )
        prune_arm = UTILITY_GATED_ARMS[-1]
        self.assertEqual(prune_arm.selection, "gradient")
        self.assertEqual(prune_arm.prune_fraction, 0.5)

    def test_candidate_pool_is_unique_deterministic_and_inactive(self) -> None:
        existing = {(0, 3), (1, 4)}
        first = build_sensor_candidate_pool(2, 4, existing, 5, seed=42)
        second = build_sensor_candidate_pool(2, 4, existing, 5, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))
        self.assertTrue(set(first).isdisjoint(existing))
        self.assertTrue(all(source in {0, 1} for source, _ in first))
        self.assertTrue(all(2 <= target < 6 for _, target in first))

    def test_summary_keeps_paired_random_growth_gate(self) -> None:
        arm = UTILITY_GATED_ARMS[4]
        base = {
            "arm": arm.name,
            "classifier": "linear",
            "test_accuracy": 0.62,
            "accuracy_gain_vs_fixed_warm_all": 0.03,
            "accuracy_gain_vs_random_sensor_48": 0.01,
            "initial_active_edges": 6,
            "final_active_edges": 8,
            "sprouted_edges": 2,
            "pruned_sprouted_edges": 0,
            "retained_sprouted_edges": 2,
            "mean_selected_gradient_score": 0.4,
            "event_rate_ratio": 1.1,
            "mean_core_ltw_change": 0.01,
            "mean_retained_sprouted_ltw_change_from_birth": 0.02,
            "mean_final_retained_sprouted_ltw": 0.12,
            "lower_ltw_saturation_rate": 0.0,
            "upper_ltw_saturation_rate": 0.0,
            "train_seconds": 1.0,
        }
        summary = summarize_utility_gated_structural_mnist(
            [
                base,
                {
                    **base,
                    "test_accuracy": 0.60,
                    "accuracy_gain_vs_random_sensor_48": -0.002,
                },
            ],
            arms=(arm,),
        )
        self.assertAlmostEqual(
            summary[0]["mean_accuracy_gain_vs_random_sensor_48"], 0.004
        )
        self.assertEqual(summary[0]["random_improved_seed_count"], 1)
        self.assertEqual(summary[0]["random_practical_gain_seed_count"], 1)

    @unittest.skipIf(torch is None, "PyTorch is unavailable")
    def test_pruning_cannot_touch_core_edges(self) -> None:
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
        core = set(model.graph.active_mask.nonzero(as_tuple=False).flatten().tolist())
        slots = [
            model.graph.sprout(0, 3, long_term_weight=0.08),
            model.graph.sprout(1, 4, long_term_weight=0.12),
        ]
        pruned, weights = prune_weak_sprouted_edges(
            model,
            slots,
            birth_weight=0.1,
            threshold_ratio=0.95,
            maximum_fraction=0.5,
        )
        self.assertEqual(pruned, [slots[0]])
        self.assertEqual(len(weights), 1)
        self.assertAlmostEqual(weights[0], 0.08, places=6)
        self.assertTrue(all(bool(model.graph.active_mask[slot].item()) for slot in core))
        self.assertFalse(bool(model.graph.active_mask[slots[0]].item()))
        self.assertTrue(bool(model.graph.active_mask[slots[1]].item()))


if __name__ == "__main__":
    unittest.main()
