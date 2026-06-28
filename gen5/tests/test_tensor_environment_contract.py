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
    world_preset_config,
    world_preset_names,
)
from ammc_gen5.dynamic_sparse import EdgeRecord


class WorldPresetContractTest(unittest.TestCase):
    def test_world_presets_are_discoverable_without_torch(self) -> None:
        names = world_preset_names()

        self.assertIn("simple", names)
        self.assertIn("moving_toxins", names)
        self.assertIn("delayed_reward", names)
        self.assertIn("gauntlet", names)

    def test_world_preset_config_applies_overrides(self) -> None:
        config = world_preset_config("gauntlet", agent_count=12, sensor_radius=None)

        self.assertEqual(config.agent_count, 12)
        self.assertEqual(config.world_size, 2.0)
        self.assertEqual(config.reward_delay_steps, 12)
        self.assertGreater(config.moving_toxin_speed, 0.0)
        self.assertEqual(config.sensor_radius, 0.25)

    def test_explicit_override_beats_preset(self) -> None:
        config = world_preset_config("sparse_cues", sensor_radius=0.5)

        self.assertEqual(config.sensor_radius, 0.5)


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

    def test_moving_toxin_updates_position(self) -> None:
        env = TensorEnvironment2D(
            TensorEnvironmentConfig(agent_count=1, food_count=1, toxin_count=1, moving_toxin_speed=0.5)
        )
        with torch.no_grad():
            env.toxin_pos.fill_(0.5)
            env.toxin_vel.zero_()
            env.toxin_vel[:, 0] = 0.5

        before = env.toxin_pos.clone()
        env.step(torch.zeros((1, 2)), collect_telemetry=False)

        self.assertGreater(float((env.toxin_pos - before).abs().sum().item()), 0.0)

    def test_reward_delay_defers_fitness_credit(self) -> None:
        env = TensorEnvironment2D(
            TensorEnvironmentConfig(
                agent_count=1,
                food_count=1,
                toxin_count=1,
                collision_radius=0.05,
                reward_delay_steps=1,
            )
        )
        with torch.no_grad():
            env.agent_pos.fill_(0.5)
            env.agent_vel.zero_()
            env.food_pos.fill_(0.5)
            env.toxin_pos.fill_(0.95)

        action = torch.zeros((1, 2))
        first = env.step(action)
        second = env.step(action)

        self.assertEqual(float(first["fitness"][0].item()), 0.0)
        self.assertGreaterEqual(float(second["fitness"][0].item()), 1.0)


if __name__ == "__main__":
    unittest.main()
