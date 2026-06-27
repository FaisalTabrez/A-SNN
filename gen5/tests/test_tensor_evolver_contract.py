from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from ammc_gen5 import TensorEvolver, TensorEvolverConfig
from ammc_gen5.dynamic_sparse import EdgeRecord


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TensorEvolverContractTest(unittest.TestCase):
    def test_culling_broadcasts_top_half_to_bottom_half(self) -> None:
        evolver = TensorEvolver(
            TensorEvolverConfig(
                population_size=4,
                neuron_count=6,
                max_edges=3,
                sprout_probability=0.0,
                prune_probability=0.0,
                ltw_noise_std=0.0,
            )
        )
        evolver.seed_from_edges([EdgeRecord(0, 1, long_term_weight=0.2)])
        with torch.no_grad():
            evolver.long_term_weight[:, 0] = torch.tensor([0.1, 0.9, 0.2, 0.8])
            before = evolver.long_term_weight.clone()

        fitness = torch.tensor([1.0, 10.0, 2.0, 8.0])
        report = evolver.cull_and_broadcast(fitness)

        self.assertEqual(report["survivor_indices"].tolist(), [1, 3])
        self.assertEqual(report["culled_indices"].tolist(), [2, 0])
        self.assertTrue(torch.equal(evolver.long_term_weight[2], before[1]))
        self.assertTrue(torch.equal(evolver.long_term_weight[0], before[3]))

    def test_mutation_sprouts_inactive_slots(self) -> None:
        evolver = TensorEvolver(
            TensorEvolverConfig(
                population_size=4,
                neuron_count=6,
                max_edges=4,
                sprout_probability=1.0,
                prune_probability=0.0,
                ltw_noise_std=0.0,
                initial_sprout_ltw=0.05,
            )
        )
        evolver.seed_from_edges([EdgeRecord(0, 1, long_term_weight=0.2)])
        child_indices = torch.tensor([2, 3])
        report = evolver.mutate_children(child_indices)

        self.assertEqual(report["sprout_count"], 6)
        self.assertTrue(torch.all(evolver.active_edge_counts()[child_indices] == 4))
        self.assertTrue(torch.all(evolver.sources[child_indices] != evolver.targets[child_indices]))

    def test_batched_sparse_forward_shape(self) -> None:
        evolver = TensorEvolver(TensorEvolverConfig(population_size=8, neuron_count=5, max_edges=4))
        evolver.seed_from_edges(
            [
                EdgeRecord(0, 1, long_term_weight=0.5),
                EdgeRecord(2, 3, short_term_weight=0.1, long_term_weight=0.2),
            ]
        )
        state = torch.rand((8, 5))
        current = evolver(state)
        self.assertEqual(tuple(current.shape), (8, 5))

    def test_positive_fitness_gate_controls_pruning(self) -> None:
        gated = TensorEvolver(
            TensorEvolverConfig(
                population_size=4,
                neuron_count=6,
                max_edges=2,
                sprout_probability=0.0,
                prune_probability=1.0,
                ltw_noise_std=0.0,
                gate_pruning_by_positive_fitness=True,
                plasticity_reward_threshold=0.0,
            )
        )
        gated.seed_from_edges([EdgeRecord(0, 1, long_term_weight=0.2)])

        no_reward = gated.evolve(torch.tensor([0.0, 0.0, -1.0, -2.0]))
        self.assertEqual(no_reward["prune_count"], 0)

        rewarded = TensorEvolver(
            TensorEvolverConfig(
                population_size=4,
                neuron_count=6,
                max_edges=2,
                sprout_probability=0.0,
                prune_probability=1.0,
                ltw_noise_std=0.0,
                gate_pruning_by_positive_fitness=True,
                plasticity_reward_threshold=0.0,
            )
        )
        rewarded.seed_from_edges([EdgeRecord(0, 1, long_term_weight=0.2)])
        reward_report = rewarded.evolve(torch.tensor([3.0, 2.0, -1.0, -2.0]))
        self.assertEqual(reward_report["prune_count"], 2)

    def test_protected_core_survives_pruning_pressure(self) -> None:
        evolver = TensorEvolver(
            TensorEvolverConfig(
                population_size=2,
                neuron_count=16,
                max_edges=3,
                sprout_probability=0.0,
                prune_probability=1.0,
                ltw_noise_std=0.0,
                low_ltw_prune_threshold=0.5,
                low_ltw_prune_probability=1.0,
                protected_edge_count=1,
                protect_core_topology=True,
            )
        )
        evolver.seed_from_edges(
            [
                EdgeRecord(0, 8, long_term_weight=0.2),
                EdgeRecord(1, 9, long_term_weight=0.2),
            ]
        )

        report = evolver.mutate_children(torch.tensor([0, 1]))

        self.assertTrue(torch.all(evolver.active_mask[:, 0]))
        self.assertTrue(torch.all(~evolver.active_mask[:, 1]))
        self.assertEqual(report["prune_count"], 2)
        self.assertEqual(report["low_ltw_prune_count"], 2)

    def test_edge_usage_stats_reports_hidden_routing(self) -> None:
        evolver = TensorEvolver(
            TensorEvolverConfig(
                population_size=2,
                neuron_count=16,
                max_edges=4,
                sprout_probability=0.0,
                prune_probability=0.0,
                ltw_noise_std=0.0,
            )
        )
        evolver.seed_from_edges(
            [
                EdgeRecord(0, 8, long_term_weight=0.2),
                EdgeRecord(0, 12, long_term_weight=0.2),
                EdgeRecord(12, 9, long_term_weight=0.2),
                EdgeRecord(13, 14, long_term_weight=0.2),
            ]
        )

        stats = evolver.edge_usage_stats()

        self.assertAlmostEqual(stats["mean_active_synapses"], 4.0)
        self.assertAlmostEqual(stats["mean_hidden_edges"], 3.0)
        self.assertAlmostEqual(stats["mean_hidden_edge_fraction"], 0.75)
        self.assertAlmostEqual(stats["mean_direct_sensor_motor_fraction"], 0.25)


if __name__ == "__main__":
    unittest.main()
