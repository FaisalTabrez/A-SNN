from __future__ import annotations

import unittest

from ammc_gen5.event_mnist import EventMNISTConfig, torch
from ammc_gen5.trainable_sequential_mnist import (
    SEQUENTIAL_LTW_ARMS,
    TrainableSequentialClassifier,
    sequential_ltw_scope_mask,
    summarize_trainable_sequential_mnist,
)


class TrainableSequentialMNISTContractTest(unittest.TestCase):
    def test_arm_set_preserves_raw_frozen_and_scoped_training_controls(self) -> None:
        names = tuple(arm.name for arm in SEQUENTIAL_LTW_ARMS)
        self.assertEqual(
            names,
            (
                "raw",
                "frozen_recurrent",
                "warm_all_3em4",
                "warm_recurrent_3em4",
            ),
        )

    def test_summary_retains_paired_gain_counts(self) -> None:
        arm = SEQUENTIAL_LTW_ARMS[2]
        base = {
            "arm": arm.name,
            "classifier": "linear",
            "test_accuracy": 0.60,
            "accuracy_gain_vs_frozen": 0.02,
            "active_edges": 6,
            "scope_trainable_edges": 6,
            "effective_trainable_parameters": 96,
            "event_rate_ratio": 1.1,
            "mean_absolute_ltw_change": 0.01,
            "mean_sensor_ltw_change": 0.02,
            "mean_recurrent_ltw_change": 0.005,
            "lower_ltw_saturation_rate": 0.0,
            "upper_ltw_saturation_rate": 0.0,
            "train_seconds": 1.0,
            "end_to_end_examples_per_second": 100.0,
        }
        summary = summarize_trainable_sequential_mnist(
            [base, {**base, "test_accuracy": 0.58, "accuracy_gain_vs_frozen": -0.002}],
            arms=(arm,),
        )
        self.assertAlmostEqual(summary[0]["mean_accuracy_gain_vs_frozen"], 0.009)
        self.assertEqual(summary[0]["improved_seed_count"], 1)
        self.assertEqual(summary[0]["practical_gain_seed_count"], 1)

    @unittest.skipIf(torch is None, "PyTorch is unavailable")
    def test_final_state_model_and_scope_masks(self) -> None:
        config = EventMNISTConfig(
            image_size=2,
            hidden_neurons=4,
            sensor_fanout=1,
            recurrent_fanout=1,
            max_edges=8,
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
        self.assertEqual(model.active_edge_count, 6)
        self.assertEqual(int(sequential_ltw_scope_mask(model, "all").sum().item()), 6)
        self.assertEqual(
            int(sequential_ltw_scope_mask(model, "recurrent").sum().item()), 4
        )
        logits, event_rate = model(torch.ones((3, 4)), return_event_rate=True)
        self.assertEqual(tuple(logits.shape), (3, 10))
        self.assertEqual(event_rate.ndim, 0)


if __name__ == "__main__":
    unittest.main()
