from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.gen12_associative_memory import (
    GEN12_MEMORY_STRATEGIES,
    available_gen12_memory_strategies,
    decide_gen12_associative_memory,
    summarize_gen12_associative_memory,
    top_fraction_spike_code,
)
from ammc_gen5.event_mnist import torch


class Gen12AssociativeMemoryContractTest(unittest.TestCase):
    def test_registered_strategy_matrix_is_frozen(self) -> None:
        self.assertEqual(
            available_gen12_memory_strategies(),
            (
                "dropout_tcn_static",
                "dropout_tcn_readout",
                "dropout_tcn_full_finetune",
                "dense_prototype_memory",
                "spiking_prototype_memory",
            ),
        )
        self.assertEqual(len(GEN12_MEMORY_STRATEGIES), 5)

    @unittest.skipIf(torch is None, "PyTorch unavailable")
    def test_rank_order_code_has_fixed_sparse_density(self) -> None:
        features = torch.arange(20, dtype=torch.float32).reshape(2, 10)
        code = top_fraction_spike_code(features, 0.20)
        self.assertTrue(torch.equal(code.sum(dim=1), torch.tensor([2.0, 2.0])))
        self.assertAlmostEqual(float(code.mean().item()), 0.20)

    def test_summary_computes_memory_causal_metrics(self) -> None:
        summary = summarize_gen12_associative_memory(_records(), budgets=(0, 50, 100))
        memory = next(row for row in summary if row["strategy"] == "spiking_prototype_memory")
        self.assertAlmostEqual(memory["mean_adaptation_gain"], 0.10)
        self.assertAlmostEqual(memory["mean_adaptation_auc"], 0.5125)
        self.assertEqual(memory["memory_contribution_seed_count"], 3)
        self.assertEqual(memory["association_specificity_seed_count"], 3)

    def test_terminal_gate_requires_gain_retention_and_causal_associations(self) -> None:
        summary = summarize_gen12_associative_memory(_records(), budgets=(0, 50, 100))
        decision = decide_gen12_associative_memory(
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
        memory = next(row for row in summary if row["strategy"] == "spiking_prototype_memory")
        memory["mean_association_specificity"] = 0.0
        self.assertEqual(
            decide_gen12_associative_memory(
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
    for strategy in GEN12_MEMORY_STRATEGIES:
        for seed in (1, 2, 3):
            for budget, shifted in ((0, 0.45), (50, 0.525), (100, 0.55)):
                if strategy == "dropout_tcn_static":
                    shifted = 0.45
                memory = strategy.endswith("prototype_memory") and budget == 100
                rows.append({
                    "seed": seed,
                    "strategy": strategy,
                    "adaptation_samples": budget,
                    "source_accuracy": 0.60,
                    "shifted_accuracy": shifted,
                    "activity": 0.20,
                    "memory_contribution": 0.01 if memory else None,
                    "association_specificity": 0.01 if memory else None,
                    "test_examples_per_second": 1000.0,
                    "cumulative_adaptation_seconds": 1.0,
                    "active_memory_cells": 100 if memory else 0,
                    "adaptation_trainable_parameters": 0,
                })
    return rows


if __name__ == "__main__":
    unittest.main()
