from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.gen14_reward_eligibility import (
    GEN14_STRATEGIES,
    Gen14Config,
    available_gen14_strategies,
    decide_gen14,
    reward_modulated_update,
    summarize_gen14,
    torch,
)


class Gen14RewardEligibilityContractTest(unittest.TestCase):
    def test_registered_strategy_matrix_is_frozen(self) -> None:
        self.assertEqual(available_gen14_strategies(), GEN14_STRATEGIES)
        self.assertEqual(len(GEN14_STRATEGIES), 5)
        self.assertIn("spiking_shuffled_reward", GEN14_STRATEGIES)

    @unittest.skipIf(torch is None, "PyTorch unavailable")
    def test_positive_reward_strengthens_chosen_local_path(self) -> None:
        weight, eligibility, baseline = reward_modulated_update(
            torch.zeros((1, 4, 2)),
            torch.zeros((1, 4, 2)),
            torch.tensor([[1.0, 0.0]]),
            torch.full((1, 4), 0.25),
            torch.tensor([1]),
            torch.tensor([1.0]),
            torch.tensor([0.0]),
            eligibility_decay=0.9,
            reward_baseline_decay=0.9,
            learning_rate=1.0,
            weight_decay=0.0,
            maximum_weight=2.0,
        )
        self.assertGreater(float(weight[0, 1, 0]), 0.0)
        self.assertLess(float(weight[0, 0, 0]), 0.0)
        self.assertGreater(float(eligibility.abs().sum()), 0.0)
        self.assertGreater(float(baseline[0]), 0.0)

    def test_terminal_gate_requires_learning_and_reward_specificity(self) -> None:
        records = _records(spiking_final=2.0, shuffled_final=0.5)
        summary = summarize_gen14(records)
        decision = decide_gen14(summary, Gen14Config())
        self.assertEqual(decision["status"], "pass")
        decision = decide_gen14(
            summarize_gen14(_records(spiking_final=0.55, shuffled_final=0.5)),
            Gen14Config(),
        )
        self.assertEqual(decision["status"], "stop")


def _records(*, spiking_final: float, shuffled_final: float):
    final = {
        "static_random": 0.5,
        "oracle_food_reflex": 3.0,
        "analog_reward_eligibility": 1.5,
        "spiking_reward_eligibility": spiking_final,
        "spiking_shuffled_reward": shuffled_final,
    }
    rows = []
    for seed in (1, 2, 3):
        for strategy in GEN14_STRATEGIES:
            for phase, fitness in (("baseline", 0.5), ("evaluation", final[strategy])):
                rows.append({
                    "seed": seed,
                    "strategy": strategy,
                    "phase": phase,
                    "steps": 100,
                    "mean_net_fitness_per_1000_steps": fitness,
                    "mean_shaped_reward_per_1000_steps": fitness,
                    "mean_spike_density": 0.20,
                    "mean_absolute_fast_weight": 0.01,
                    "fast_weight_saturation": 0.0,
                })
    return rows


if __name__ == "__main__":
    unittest.main()
