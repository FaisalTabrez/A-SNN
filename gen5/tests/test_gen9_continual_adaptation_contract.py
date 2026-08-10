from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.event_mnist import torch
from ammc_gen5.gen9_continual_adaptation import (
    GEN9_ADAPTATION_STRATEGIES,
    GEN9_SOURCE_ARMS,
    apply_sensor_damage,
    available_gen9_adaptation_strategies,
    available_gen9_source_arms,
    decide_gen9_continual_adaptation,
    select_gen9_promoted_source_arms,
    sensor_damage_indices,
    summarize_gen9_adaptation,
    _save_progress,
)
from ammc_gen5.milestone_a_architecture import _load_progress


class Gen9ContinualAdaptationContractTest(unittest.TestCase):
    def test_registered_matrix_is_frozen(self) -> None:
        self.assertEqual(
            available_gen9_source_arms(), ("dilated_tcn", "predictive_lif")
        )
        self.assertEqual(
            available_gen9_adaptation_strategies(),
            (
                "tcn_static",
                "tcn_readout",
                "tcn_full_finetune",
                "predictive_lif_static",
                "predictive_lif_readout",
            ),
        )
        self.assertEqual(len(GEN9_SOURCE_ARMS), 2)
        self.assertEqual(len(GEN9_ADAPTATION_STRATEGIES), 5)

    def test_lif_promotion_requires_accuracy_budget_and_activity(self) -> None:
        rows = [
            {
                "arm": "dilated_tcn",
                "best_validation_accuracy": 0.60,
                "parameter_ratio_vs_target": 0.99,
                "checkpoint_activity": 0.40,
            },
            {
                "arm": "predictive_lif",
                "best_validation_accuracy": 0.595,
                "parameter_ratio_vs_target": 1.00,
                "checkpoint_activity": 0.08,
            },
        ]
        self.assertEqual(
            select_gen9_promoted_source_arms(
                rows,
                promotion_margin=0.01,
                minimum_parameter_ratio=0.95,
                maximum_parameter_ratio=1.05,
                minimum_spike_rate=0.01,
                maximum_spike_rate=0.30,
            ),
            ("dilated_tcn", "predictive_lif"),
        )
        rows[1]["checkpoint_activity"] = 0.40
        self.assertEqual(
            select_gen9_promoted_source_arms(
                rows,
                promotion_margin=0.01,
                minimum_parameter_ratio=0.95,
                maximum_parameter_ratio=1.05,
                minimum_spike_rate=0.01,
                maximum_spike_rate=0.30,
            ),
            ("dilated_tcn",),
        )

    def test_terminal_gate_requires_auc_replication_and_retention(self) -> None:
        common = {
            "runs": 3,
            "mean_source_initial_accuracy": 0.60,
            "mean_shifted_initial_accuracy": 0.50,
            "mean_shift_drop": 0.10,
            "mean_source_final_accuracy": 0.59,
            "mean_shifted_final_accuracy": 0.55,
            "mean_adaptation_gain": 0.05,
            "two_point_gain_seed_count": 3,
            "mean_forgetting": 0.01,
            "mean_adaptation_auc": 0.53,
            "mean_activity": 0.08,
            "one_point_auc_advantage_seed_count_vs_tcn_readout": 0,
        }
        tcn_static = dict(common, strategy="tcn_static", mean_shifted_final_accuracy=0.50, mean_adaptation_auc=0.50)
        tcn_readout = dict(common, strategy="tcn_readout", mean_adaptation_auc=0.52)
        lif_static = dict(common, strategy="predictive_lif_static", mean_shifted_final_accuracy=0.50, mean_adaptation_auc=0.50)
        lif_readout = dict(
            common,
            strategy="predictive_lif_readout",
            mean_adaptation_auc=0.535,
            one_point_auc_advantage_seed_count_vs_tcn_readout=2,
        )
        decision = decide_gen9_continual_adaptation(
            [tcn_static, tcn_readout, lif_static, lif_readout],
            minimum_shift_drop=0.05,
            minimum_adaptation_gain=0.02,
            minimum_auc_advantage=0.01,
            accuracy_margin=0.01,
            forgetting_margin=0.005,
            minimum_spike_rate=0.01,
            maximum_spike_rate=0.30,
        )
        self.assertEqual(decision["status"], "pass")
        lif_readout["mean_adaptation_auc"] = 0.525
        stopped = decide_gen9_continual_adaptation(
            [tcn_static, tcn_readout, lif_static, lif_readout],
            minimum_shift_drop=0.05,
            minimum_adaptation_gain=0.02,
            minimum_auc_advantage=0.01,
            accuracy_margin=0.01,
            forgetting_margin=0.005,
            minimum_spike_rate=0.01,
            maximum_spike_rate=0.30,
        )
        self.assertEqual(stopped["status"], "stop")

    def test_summary_computes_linear_budget_auc(self) -> None:
        rows = []
        for strategy in ("tcn_static", "tcn_readout"):
            for seed in (1, 2, 3):
                for budget, accuracy in ((0, 0.40), (50, 0.50), (100, 0.60)):
                    rows.append(
                        {
                            "seed": seed,
                            "strategy": strategy,
                            "source_model": "dilated_tcn",
                            "adaptation_kind": "static" if strategy.endswith("static") else "readout",
                            "adaptation_samples": budget,
                            "source_accuracy": 0.70,
                            "shifted_accuracy": accuracy,
                            "activity": 0.10,
                            "test_examples_per_second": 1000.0,
                            "cumulative_adaptation_seconds": 1.0,
                            "adaptation_trainable_parameters": 10,
                        }
                    )
        summary = summarize_gen9_adaptation(rows, budgets=(0, 50, 100))
        lookup = {row["strategy"]: row for row in summary}
        self.assertAlmostEqual(lookup["tcn_readout"]["mean_adaptation_auc"], 0.50)

    def test_progress_round_trip_uses_gen9_schema(self) -> None:
        signature = {"version": 1, "experiment": "gen9"}
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "progress.json"
            _save_progress(
                path,
                signature,
                stage="adaptation",
                screen_records=[{"arm": "dilated_tcn"}],
                promoted_source_arms=("dilated_tcn", "predictive_lif"),
                adaptation_records=[{"strategy": "tcn_static"}],
            )
            payload = _load_progress(path, signature)
        self.assertEqual(payload["stage"], "adaptation")
        self.assertEqual(
            payload["promoted_source_arms"], ["dilated_tcn", "predictive_lif"]
        )
        self.assertEqual(payload["adaptation_records"][0]["strategy"], "tcn_static")

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_sensor_damage_is_deterministic_and_non_mutating(self) -> None:
        first = sensor_damage_indices(10, 0.30, seed=909)
        second = sensor_damage_indices(10, 0.30, seed=909)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        events = torch.ones((2, 4, 10), dtype=torch.uint8)
        damaged = apply_sensor_damage(events, first)
        self.assertTrue(torch.equal(events, torch.ones_like(events)))
        self.assertEqual(int(damaged[:, :, list(first)].sum().item()), 0)


if __name__ == "__main__":
    unittest.main()
