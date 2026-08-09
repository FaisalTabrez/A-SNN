from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
SCRIPT = EXAMPLES / "sprint17_embodied_action_controls.py"


def _load_runner():
    sys.path.insert(0, str(EXAMPLES))
    spec = importlib.util.spec_from_file_location("sprint17_embodied_action_controls", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Sprint17ActionControlsContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = _load_runner()

    def test_policy_set_contains_movement_and_oracle_controls(self) -> None:
        policies = self.runner.POLICIES
        self.assertIn("random_cardinal", policies)
        self.assertIn("fixed_analog_cardinal", policies)
        self.assertIn("direct_sensor_oracle", policies)
        self.assertIn("augmented_adapter", policies)

    def test_summary_preserves_action_efficiency(self) -> None:
        base = {
            "world": "simple",
            "policy": "random_cardinal",
            "sensor_noise_std": 0.0,
            "mean_food_hits": 1.0,
            "mean_toxin_hits": 1.0,
            "survival_rate": 0.5,
            "cue_action_coverage": 1.0,
            "oracle_action_agreement": 0.25,
            "mean_action_magnitude": 1.0,
        }
        rows = [
            {**base, "mean_fitness": -1.0, "fitness_per_unit_action": -1.0},
            {**base, "mean_fitness": 1.0, "fitness_per_unit_action": 1.0},
        ]

        summary = self.runner._summarize(rows)

        self.assertEqual(summary[0]["seeds"], 2)
        self.assertAlmostEqual(summary[0]["mean_fitness"], 0.0)
        self.assertAlmostEqual(summary[0]["std_mean_fitness"], 1.0)
        self.assertAlmostEqual(summary[0]["mean_fitness_per_unit_action"], 0.0)


if __name__ == "__main__":
    unittest.main()
