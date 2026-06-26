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
    DynamicSparseLinear,
    HeadlessAMMCLoop,
    TensorEnvironment2D,
    TensorEnvironmentConfig,
    VectorizedTransducer,
)
from ammc_gen5.dynamic_sparse import EdgeRecord


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TensorEnvironmentContractTest(unittest.TestCase):
    def test_environment_is_batched(self) -> None:
        env = TensorEnvironment2D(TensorEnvironmentConfig(agent_count=128, food_count=16, toxin_count=16))
        sensory = env.sensory_tensor()
        self.assertEqual(tuple(sensory.shape), (128, 8))

        action = torch.zeros((128, 2))
        telemetry = env.step(action)
        self.assertEqual(tuple(telemetry["fitness"].shape), (128,))
        self.assertEqual(tuple(telemetry["nearest_food_vec"].shape), (128, 2))
        self.assertEqual(tuple(telemetry["nearest_toxin_vec"].shape), (128, 2))

    def test_step_can_skip_diagnostic_telemetry(self) -> None:
        env = TensorEnvironment2D(TensorEnvironmentConfig(agent_count=128, food_count=16, toxin_count=16))
        action = torch.zeros((128, 2))

        telemetry = env.step(action, collect_telemetry=False)

        self.assertIsNone(telemetry)
        self.assertEqual(tuple(env.fitness.shape), (128,))

    def test_transducer_loop_shapes(self) -> None:
        env = TensorEnvironment2D(TensorEnvironmentConfig(agent_count=32, food_count=8, toxin_count=8))
        brain = DynamicSparseLinear(16, 16, max_edges=8)
        brain.load_edges(
            [
                EdgeRecord(0, 8, long_term_weight=0.5),
                EdgeRecord(1, 9, long_term_weight=0.5),
                EdgeRecord(2, 10, long_term_weight=0.5),
                EdgeRecord(3, 11, long_term_weight=0.5),
            ]
        )
        loop = HeadlessAMMCLoop(env, brain, VectorizedTransducer())
        telemetry = loop.step()

        self.assertEqual(tuple(telemetry["sensory"].shape), (32, 8))
        self.assertEqual(tuple(telemetry["spikes"].shape), (32, 16))
        self.assertEqual(tuple(telemetry["action"].shape), (32, 2))


if __name__ == "__main__":
    unittest.main()
