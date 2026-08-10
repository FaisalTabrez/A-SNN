from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.gen13_local_plasticity import (
    GEN13_PLASTICITY_STRATEGIES,
    available_gen13_plasticity_strategies,
    decide_gen13_local_plasticity,
    summarize_gen13_local_plasticity,
    three_factor_update,
)
from ammc_gen5.event_mnist import torch


class Gen13LocalPlasticityContractTest(unittest.TestCase):
    def test_registered_strategy_matrix_is_frozen(self) -> None:
        self.assertEqual(
            available_gen13_plasticity_strategies(),
            (
                "dropout_tcn_static",
                "dropout_tcn_readout",
                "dropout_tcn_full_finetune",
                "analog_three_factor_readout",
                "spiking_three_factor_readout",
            ),
        )
        self.assertEqual(len(GEN13_PLASTICITY_STRATEGIES), 5)

    @unittest.skipIf(torch is None, "PyTorch unavailable")
    def test_three_factor_rule_strengthens_target_synapses(self) -> None:
        weights = torch.zeros((2, 2))
        trace = torch.tensor([[1.0, 0.0]])
        logits = torch.zeros((1, 2))
        labels = torch.tensor([1])
        three_factor_update(
            weights, trace, logits, labels, learning_rate=1.0
        )
        self.assertGreater(float(weights[1, 0]), 0.0)
        self.assertLess(float(weights[0, 0]), 0.0)
        self.assertEqual(float(weights[:, 1].abs().sum()), 0.0)

    def test_summary_computes_fast_weight_causal_metrics(self) -> None:
        summary = summarize_gen13_local_plasticity(_records(), budgets=(0, 50, 100))
        local = next(row for row in summary if row["strategy"] == "spiking_three_factor_readout")
        self.assertAlmostEqual(local["mean_adaptation_gain"], 0.10)
        self.assertAlmostEqual(local["mean_adaptation_auc"], 0.5125)
        self.assertEqual(local["fast_weight_contribution_seed_count"], 3)
        self.assertEqual(local["class_specificity_seed_count"], 3)

    def test_terminal_gate_requires_gain_retention_and_class_specificity(self) -> None:
        summary = summarize_gen13_local_plasticity(_records(), budgets=(0, 50, 100))
        decision = decide_gen13_local_plasticity(
            summary,
            minimum_shift_drop=0.02,
            minimum_adaptation_gain=0.02,
            auc_margin=0.01,
            final_accuracy_margin=0.01,
            forgetting_margin=0.005,
            causal_margin=0.005,
            minimum_spike_density=0.05,
            maximum_spike_density=0.35,
        )
        self.assertEqual(decision["status"], "pass")
        local = next(row for row in summary if row["strategy"] == "spiking_three_factor_readout")
        local["mean_class_specificity"] = 0.0
        self.assertEqual(
            decide_gen13_local_plasticity(
                summary,
                minimum_shift_drop=0.02,
                minimum_adaptation_gain=0.02,
                auc_margin=0.01,
                final_accuracy_margin=0.01,
                forgetting_margin=0.005,
                causal_margin=0.005,
                minimum_spike_density=0.05,
                maximum_spike_density=0.35,
            )["status"],
            "stop",
        )


def _records():
    rows = []
    for strategy in GEN13_PLASTICITY_STRATEGIES:
        for seed in (1, 2, 3):
            for budget, shifted in ((0, 0.45), (50, 0.525), (100, 0.55)):
                if strategy == "dropout_tcn_static":
                    shifted = 0.45
                local = strategy.endswith("three_factor_readout") and budget == 100
                rows.append({
                    "seed": seed,
                    "strategy": strategy,
                    "adaptation_samples": budget,
                    "source_accuracy": 0.60,
                    "shifted_accuracy": shifted,
                    "activity": 0.20,
                    "fast_weight_contribution": 0.01 if local else None,
                    "class_specificity": 0.01 if local else None,
                    "test_examples_per_second": 1000.0,
                    "cumulative_adaptation_seconds": 1.0,
                    "active_fast_synapses": 100 if local else 0,
                    "mean_absolute_fast_weight": 0.01 if local else 0.0,
                    "adaptation_trainable_parameters": 0,
                })
    return rows


if __name__ == "__main__":
    unittest.main()
