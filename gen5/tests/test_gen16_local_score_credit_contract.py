from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.gen16_local_score_credit import (
    GEN16_STRATEGIES,
    Gen16Config,
    LinearRewardPolicy,
    available_gen16_strategies,
    decide_gen16,
    manual_score_gradients,
    score_gradient_parity,
    summarize_gen16,
    torch,
)


class Gen16LocalScoreCreditContractTest(unittest.TestCase):
    def test_registered_strategies_are_frozen(self) -> None:
        self.assertEqual(available_gen16_strategies(), GEN16_STRATEGIES)
        self.assertEqual(len(GEN16_STRATEGIES), 5)

    @unittest.skipIf(torch is None, "PyTorch unavailable")
    def test_manual_score_gradient_matches_autograd(self) -> None:
        torch.manual_seed(7)
        policy = LinearRewardPolicy()
        sensory = torch.randn(3, 5, 8)
        action = torch.randint(0, 4, (3, 5))
        returns = torch.randn(3, 5)
        error = score_gradient_parity(policy, sensory, action, returns)
        self.assertLessEqual(error, 1e-6)
        weight, bias = manual_score_gradients(policy, sensory, action, returns)
        self.assertEqual(tuple(weight.shape), (4, 8))
        self.assertEqual(tuple(bias.shape), (4,))

    def test_decision_requires_local_equivalence_and_reward_identity(self) -> None:
        records = _records(local=2.0, autograd=2.0, shuffled=0.5, gradient_error=1e-7)
        decision = decide_gen16(records, summarize_gen16(records), Gen16Config())
        self.assertEqual(decision["status"], "pass")
        records = _records(local=0.4, autograd=2.0, shuffled=0.5, gradient_error=1e-7)
        decision = decide_gen16(records, summarize_gen16(records), Gen16Config())
        self.assertEqual(decision["status"], "stop")


def _records(*, local, autograd, shuffled, gradient_error):
    final = {
        "static_linear_policy": 0.5,
        "oracle_food_reflex": 3.0,
        "autograd_score_policy": autograd,
        "manual_local_score_policy": local,
        "manual_local_shuffled_reward": shuffled,
    }
    rows = []
    for seed in (1, 2, 3):
        for strategy in GEN16_STRATEGIES:
            for phase, value in (("baseline", 0.5), ("evaluation", final[strategy])):
                rows.append({
                    "seed": seed,
                    "strategy": strategy,
                    "phase": phase,
                    "steps": 100,
                    "mean_net_fitness_per_1000_steps": value,
                    "mean_shaped_reward_per_1000_steps": value,
                    "mean_policy_entropy": 1.0,
                    "mean_training_loss": 0.0,
                    "mean_training_reward_per_1000_steps": 0.0,
                    "training_updates": 1,
                    "maximum_score_gradient_error": (
                        gradient_error if strategy == "manual_local_score_policy" else 0.0
                    ),
                    "policy_weight_delta_norm": 1.0,
                })
    return rows


if __name__ == "__main__":
    unittest.main()
