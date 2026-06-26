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

from ammc_gen5 import (
    EvolvingHeadlessAMMCLoop,
    EvolvingLoopConfig,
    TensorEnvironment2D,
    TensorEnvironmentConfig,
    TensorEvolver,
    TensorEvolverConfig,
    VectorizedTransducer,
)
from ammc_gen5.dynamic_sparse import EdgeRecord


@unittest.skipIf(torch is None, "PyTorch is not installed")
class EvolvingLoopContractTest(unittest.TestCase):
    def make_loop(self, *, epoch_steps: int = 3):
        environment = TensorEnvironment2D(TensorEnvironmentConfig(agent_count=16, food_count=4, toxin_count=4))
        evolver = TensorEvolver(
            TensorEvolverConfig(
                population_size=16,
                neuron_count=16,
                max_edges=16,
                sprout_probability=0.0,
                prune_probability=0.0,
                ltw_noise_std=0.0,
            )
        )
        evolver.seed_from_edges(
            [
                EdgeRecord(0, 8, long_term_weight=0.5),
                EdgeRecord(1, 9, long_term_weight=0.5),
                EdgeRecord(2, 10, long_term_weight=0.5),
                EdgeRecord(3, 11, long_term_weight=0.5),
            ]
        )
        return EvolvingHeadlessAMMCLoop(
            environment,
            evolver,
            VectorizedTransducer(),
            EvolvingLoopConfig(epoch_steps=epoch_steps),
        )

    def test_step_contract(self) -> None:
        loop = self.make_loop(epoch_steps=5)
        telemetry = loop.step()

        self.assertEqual(tuple(telemetry["sensory"].shape), (16, 8))
        self.assertEqual(tuple(telemetry["spikes"].shape), (16, 16))
        self.assertEqual(tuple(telemetry["action"].shape), (16, 2))
        self.assertEqual(telemetry["epoch_step"], 1)
        self.assertEqual(telemetry["generation"], 1)
        self.assertIsNone(telemetry["epoch_report"])

    def test_epoch_triggers_evolution_and_reset(self) -> None:
        loop = self.make_loop(epoch_steps=2)
        first_positions = loop.environment.agent_pos.clone()
        loop.step()
        telemetry = loop.step()

        self.assertEqual(telemetry["epoch_step"], 0)
        self.assertEqual(telemetry["generation"], 2)
        self.assertIsNotNone(telemetry["epoch_report"])
        self.assertEqual(int(loop.evolver.epoch.item()), 1)
        self.assertTrue(torch.all(loop.environment.fitness == 0))
        self.assertFalse(torch.equal(first_positions, loop.environment.agent_pos))
        self.assertTrue(torch.all(loop.membrane == 0))


if __name__ == "__main__":
    unittest.main()

