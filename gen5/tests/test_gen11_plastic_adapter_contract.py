from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.gen11_plastic_adapter import (
    GEN11_ADAPTATION_STRATEGIES,
    available_gen11_adaptation_strategies,
    decide_gen11_plastic_adapter,
    summarize_gen11_adaptation,
)


class Gen11PlasticAdapterContractTest(unittest.TestCase):
    def test_registered_strategy_matrix_is_frozen(self) -> None:
        self.assertEqual(
            available_gen11_adaptation_strategies(),
            (
                "dropout_tcn_static",
                "dropout_tcn_readout",
                "dropout_tcn_full_finetune",
                "analog_state_adapter",
                "lif_state_adapter",
            ),
        )
        self.assertEqual(len(GEN11_ADAPTATION_STRATEGIES), 5)

    def test_summary_computes_adapter_causal_metrics(self) -> None:
        rows = _records()
        summary = summarize_gen11_adaptation(rows, budgets=(0, 50, 100))
        lif = next(row for row in summary if row["strategy"] == "lif_state_adapter")
        self.assertAlmostEqual(lif["mean_adaptation_auc"], 0.5125)
        self.assertAlmostEqual(lif["mean_adaptation_gain"], 0.10)
        self.assertEqual(lif["state_contribution_seed_count"], 3)
        self.assertEqual(lif["state_specificity_seed_count"], 3)

    def test_terminal_gate_requires_adaptation_retention_and_causal_state(self) -> None:
        summary = summarize_gen11_adaptation(_records(), budgets=(0, 50, 100))
        decision = decide_gen11_plastic_adapter(
            summary,
            minimum_shift_drop=0.02,
            minimum_adaptation_gain=0.02,
            auc_margin=0.01,
            final_accuracy_margin=0.01,
            forgetting_margin=0.005,
            causal_margin=0.005,
            minimum_spike_rate=0.01,
            maximum_spike_rate=0.30,
        )
        self.assertEqual(decision["status"], "pass")
        lif = next(row for row in summary if row["strategy"] == "lif_state_adapter")
        lif["mean_state_specificity"] = 0.0
        self.assertEqual(
            decide_gen11_plastic_adapter(
                summary,
                minimum_shift_drop=0.02,
                minimum_adaptation_gain=0.02,
                auc_margin=0.01,
                final_accuracy_margin=0.01,
                forgetting_margin=0.005,
                causal_margin=0.005,
                minimum_spike_rate=0.01,
                maximum_spike_rate=0.30,
            )["status"],
            "stop",
        )


def _records():
    rows = []
    for strategy in GEN11_ADAPTATION_STRATEGIES:
        for seed in (1, 2, 3):
            for budget, shifted in ((0, 0.45), (50, 0.525), (100, 0.55)):
                if strategy == "dropout_tcn_static":
                    shifted = 0.45
                rows.append({
                    "seed": seed,
                    "strategy": strategy,
                    "adaptation_samples": budget,
                    "source_accuracy": 0.60,
                    "shifted_accuracy": shifted,
                    "activity": 0.08 if strategy == "lif_state_adapter" else 0.20,
                    "state_contribution": 0.01 if strategy.endswith("state_adapter") and budget == 100 else None,
                    "state_specificity": 0.01 if strategy.endswith("state_adapter") and budget == 100 else None,
                    "mean_absolute_gate": 0.10 if strategy.endswith("state_adapter") else 0.0,
                    "test_examples_per_second": 1000.0,
                    "cumulative_adaptation_seconds": 1.0,
                    "adaptation_trainable_parameters": 100,
                })
    return rows


if __name__ == "__main__":
    unittest.main()
