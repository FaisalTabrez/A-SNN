from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "sprint16_frozen_embodied_adapter.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("sprint16_frozen_embodied_adapter", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Sprint16EmbodiedAdapterContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = _load_runner()

    def test_policy_ablation_has_three_expected_arms(self) -> None:
        self.assertEqual(
            self.runner.POLICIES,
            ("fixed_motor_argmax", "base_adapter", "augmented_adapter"),
        )

    def test_summary_groups_world_policy_and_noise(self) -> None:
        rows = [
            {
                "world": "simple",
                "policy": "base_adapter",
                "sensor_noise_std": 0.05,
                "mean_fitness": 1.0,
                "mean_food_hits": 2.0,
                "mean_toxin_hits": 1.0,
                "survival_rate": 0.75,
                "cue_action_coverage": 0.5,
                "oracle_action_agreement": 0.8,
                "mean_action_magnitude": 0.5,
            },
            {
                "world": "simple",
                "policy": "base_adapter",
                "sensor_noise_std": 0.05,
                "mean_fitness": 3.0,
                "mean_food_hits": 4.0,
                "mean_toxin_hits": 1.0,
                "survival_rate": 1.0,
                "cue_action_coverage": 1.0,
                "oracle_action_agreement": 1.0,
                "mean_action_magnitude": 1.0,
            },
        ]

        summary = self.runner._summarize(rows)

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["seeds"], 2)
        self.assertAlmostEqual(summary[0]["mean_fitness"], 2.0)
        self.assertAlmostEqual(summary[0]["std_mean_fitness"], 1.0)


if __name__ == "__main__":
    unittest.main()
