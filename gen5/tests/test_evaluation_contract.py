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
    NeuronScalePoint,
    NeuronScalingConfig,
    NeuronScalingGenerationRecord,
    NeuronScalingRunner,
    PlasticityAblationConfig,
    PlasticityAblationRunner,
    RetentionAblationConfig,
    RetentionAblationRunner,
    SparseEfficiencyGenerationRecord,
    TrialRunner,
    TrialRunnerConfig,
    hidden_neuron_count,
    summarize_neuron_scaling_records,
    summarize_sparse_efficiency_records,
)


class NeuronScalingSummaryTest(unittest.TestCase):
    def test_neuron_scaling_summary_math_without_torch(self) -> None:
        records = [
            NeuronScalingGenerationRecord(16, 4, 128, 42, 1, 12.0, 12.0, 0.1, 8.0, 0.0625, 1, 0, 8),
            NeuronScalingGenerationRecord(16, 4, 128, 42, 2, 26.0, 26.0, 0.2, 9.0, 0.0703125, 1, 0, 8),
            NeuronScalingGenerationRecord(16, 4, 128, 43, 1, 10.0, 10.0, 0.0, 7.0, 0.0546875, 1, 0, 8),
            NeuronScalingGenerationRecord(16, 4, 128, 43, 2, 24.0, 24.0, 0.1, 8.0, 0.0625, 1, 0, 8),
        ]
        summary = summarize_neuron_scaling_records(records, threshold=25.0)

        self.assertEqual(hidden_neuron_count(16), 4)
        self.assertEqual(len(summary), 1)
        row = summary[0]
        self.assertEqual(row.seeds, 2)
        self.assertEqual(row.hidden_neurons, 4)
        self.assertAlmostEqual(row.final_mean_best_fitness, 25.0)
        self.assertAlmostEqual(row.final_mean_active_synapses, 8.5)
        self.assertAlmostEqual(row.threshold_success_rate, 0.5)
        self.assertAlmostEqual(row.mean_generation_to_threshold or 0.0, 2.0)


class SparseEfficiencySummaryTest(unittest.TestCase):
    def test_sparse_efficiency_summary_math_without_torch(self) -> None:
        records = [
            SparseEfficiencyGenerationRecord(
                "edge_penalty", 16, 4, 128, 42, 1, 20.0, 20.0, 18.0, 0.1, -0.2,
                80.0, 0.625, 0.25, 20.0, 0.25, 0.1, 1, 2, 1, 80,
            ),
            SparseEfficiencyGenerationRecord(
                "edge_penalty", 16, 4, 128, 42, 2, 26.0, 26.0, 24.5, 0.2, -0.1,
                70.0, 0.546875, 0.3714285714, 25.0, 0.35, 0.1, 1, 2, 1, 70,
            ),
            SparseEfficiencyGenerationRecord(
                "edge_penalty", 16, 4, 128, 43, 1, 24.0, 24.0, 22.0, 0.1, -0.2,
                60.0, 0.46875, 0.4, 18.0, 0.3, 0.1, 1, 2, 1, 60,
            ),
        ]
        summary = summarize_sparse_efficiency_records(records, threshold=25.0)

        self.assertEqual(len(summary), 1)
        row = summary[0]
        self.assertEqual(row.group, "edge_penalty")
        self.assertEqual(row.seeds, 2)
        self.assertAlmostEqual(row.final_mean_best_fitness, 25.0)
        self.assertAlmostEqual(row.final_mean_active_synapses, 65.0)
        self.assertAlmostEqual(row.threshold_success_rate, 0.5)
        self.assertAlmostEqual(row.mean_generation_to_threshold or 0.0, 2.0)


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

    def test_neuron_scaling_emits_scale_summaries(self) -> None:
        runner = NeuronScalingRunner(
            NeuronScalingConfig(
                seeds=(42,),
                generations=2,
                epoch_steps=2,
                population_size=6,
                food_count=2,
                toxin_count=2,
                scale_points=(
                    NeuronScalePoint(neuron_count=16, max_edges=8),
                    NeuronScalePoint(neuron_count=24, max_edges=12),
                ),
                device="cpu",
            )
        )
        result = runner.run()

        self.assertEqual(len(result.records), 4)
        self.assertEqual(len(result.summary), 2)
        self.assertEqual({row.neuron_count for row in result.summary}, {16, 24})
        self.assertEqual({row.hidden_neurons for row in result.summary}, {4, 12})

        with tempfile.TemporaryDirectory() as tmp:
            paths = runner.save_outputs(result, tmp, plot=False)
            self.assertTrue(pathlib.Path(paths["json"]).exists())
            self.assertTrue(pathlib.Path(paths["records_csv"]).exists())
            self.assertTrue(pathlib.Path(paths["summary_csv"]).exists())


if __name__ == "__main__":
    unittest.main()
