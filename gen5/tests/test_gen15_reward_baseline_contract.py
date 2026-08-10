from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.gen15_reward_baseline import (
    GEN15_STRATEGIES,
    Gen15Config,
    available_gen15_strategies,
    decide_gen15,
    discounted_returns,
    summarize_gen15,
    torch,
)


class Gen15RewardBaselineContractTest(unittest.TestCase):
    def test_registered_strategies_are_frozen(self) -> None:
        self.assertEqual(available_gen15_strategies(), GEN15_STRATEGIES)
        self.assertEqual(len(GEN15_STRATEGIES), 4)

    @unittest.skipIf(torch is None, "PyTorch unavailable")
    def test_discounted_returns_credit_earlier_actions(self) -> None:
        reward = torch.tensor([[0.0], [0.0], [1.0]])
        returns = discounted_returns(reward, 0.5)
        self.assertTrue(torch.allclose(returns[:, 0], torch.tensor([0.25, 0.5, 1.0])))

    def test_decision_requires_reset_and_reward_identity(self) -> None:
        decision = decide_gen15(summarize_gen15(_records(reinforce=2.0, shuffled=0.5)), Gen15Config())
        self.assertEqual(decision["status"], "pass")
        decision = decide_gen15(summarize_gen15(_records(reinforce=0.4, shuffled=0.5)), Gen15Config())
        self.assertEqual(decision["status"], "stop")


def _records(*, reinforce, shuffled):
    final = {
        "static_random": 0.5,
        "oracle_food_reflex": 3.0,
        "reinforce_shared_policy": reinforce,
        "reinforce_shuffled_reward": shuffled,
    }
    rows = []
    for seed in (1, 2, 3):
        for strategy in GEN15_STRATEGIES:
            baseline = 0.5
            for phase, value in (("baseline", baseline), ("evaluation", final[strategy])):
                rows.append({
                    "seed": seed, "strategy": strategy, "phase": phase,
                    "steps": 100, "mean_net_fitness_per_1000_steps": value,
                    "mean_shaped_reward_per_1000_steps": value,
                    "mean_policy_entropy": 1.0, "mean_training_loss": 0.0,
                    "mean_training_reward_per_1000_steps": 0.0,
                    "training_updates": 1,
                })
    return rows


if __name__ == "__main__":
    unittest.main()
