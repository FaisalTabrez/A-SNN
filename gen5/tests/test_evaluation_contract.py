from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from ammc_gen5 import (
    PlasticityAblationConfig,
    PlasticityAblationRunner,
    RetentionAblationConfig,
    RetentionAblationRunner,
    TrialRunner,
    TrialRunnerConfig,
)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class EvaluationContractTest(unittest.TestCase):
    def test_trial_runner_emits_records_and_aggregate_outputs(self) -> None:
        runner = TrialRunner(
            TrialRunnerConfig(
                seeds=(42, 43),
                generations=3,
                epoch_steps=2,
                population_size=6,
                food_count=2,
                toxin_count=2,
                max_edges=8,
                device="cpu",
            )
        )
        result = runner.run()

        self.assertEqual(len(result.trial_records), 6)
        self.assertEqual(len(result.aggregate_records), 3)
        self.assertEqual(result.aggregate_records[-1].generation, 3)

        with tempfile.TemporaryDirectory() as tmp:
            paths = runner.save_outputs(result, tmp, plot=False)
            self.assertTrue(pathlib.Path(paths["json"]).exists())
            self.assertTrue(pathlib.Path(paths["trial_csv"]).exists())
            self.assertTrue(pathlib.Path(paths["aggregate_csv"]).exists())

    def test_plasticity_ablation_emits_group_summaries(self) -> None:
        runner = PlasticityAblationRunner(
            PlasticityAblationConfig(
                seeds=(42,),
                generations=2,
                epoch_steps=2,
                population_size=6,
                food_count=2,
                toxin_count=2,
                max_edges=12,
                device="cpu",
            )
        )
        result = runner.run()

        groups = {row.group for row in result.summary}
        self.assertEqual(groups, {"static_snn", "full_plasticity_infant", "gated_plasticity_adult"})
        self.assertEqual(len(result.records), 6)
        self.assertEqual(len(result.summary), 3)

    def test_retention_ablation_emits_three_phase_records(self) -> None:
        runner = RetentionAblationRunner(
            RetentionAblationConfig(
                seeds=(42,),
                original_generations=1,
                perturbation_generations=1,
                recovery_generations=1,
                epoch_steps=2,
                population_size=6,
                food_count=2,
                toxin_count=2,
                max_edges=12,
                device="cpu",
            )
        )
        result = runner.run()

        phases = {row.phase for row in result.records}
        groups = {row.group for row in result.summary}
        self.assertEqual(phases, {"original", "perturbed", "recovery"})
        self.assertEqual(groups, {"static_snn", "full_plasticity_infant", "gated_plasticity_adult"})
        self.assertEqual(len(result.records), 9)
        self.assertEqual(len(result.summary), 3)


if __name__ == "__main__":
    unittest.main()
