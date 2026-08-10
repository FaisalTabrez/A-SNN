from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.gen18_local_credit_replication import (
    GEN18_STRATEGIES,
    Gen18Config,
    available_gen18_strategies,
    confidence_lower_bound,
    decide_gen18,
    summarize_gen18,
)


class Gen18LocalCreditReplicationContractTest(unittest.TestCase):
    def test_registered_strategies_are_frozen(self) -> None:
        self.assertEqual(available_gen18_strategies(), GEN18_STRATEGIES)
        self.assertEqual(len(GEN18_STRATEGIES), 4)

    def test_confidence_bound_penalizes_variance(self) -> None:
        self.assertAlmostEqual(confidence_lower_bound([2.0, 2.0, 2.0]), 2.0)
        self.assertLess(confidence_lower_bound([-1.0, 2.0, 2.0]), 1.0)

    def test_decision_requires_replicated_gain_and_reward_identity(self) -> None:
        config = Gen18Config(
            seeds=(1, 2, 3),
            minimum_qualified_seed_count=2,
            confidence_z=1.0,
        )
        records = _records(local=2.0, shuffled=0.5)
        decision = decide_gen18(records, summarize_gen18(records, 1.0), config)
        self.assertEqual(decision["status"], "pass")

        records = _records(local=2.0, shuffled=2.1)
        decision = decide_gen18(records, summarize_gen18(records, 1.0), config)
        self.assertEqual(decision["status"], "stop")
        self.assertFalse(decision["replicated_reward_identity_gate"])


def _records(*, local: float, shuffled: float) -> list[dict]:
    final = {
        "static_linear_policy": 0.5,
        "oracle_food_reflex": 3.0,
        "manual_local_score_policy": local,
        "manual_local_shuffled_reward": shuffled,
    }
    rows = []
    for seed in (1, 2, 3):
        for strategy in GEN18_STRATEGIES:
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
                    "maximum_score_gradient_error": 1e-8,
                    "policy_weight_delta_norm": 1.0,
                })
    return rows


if __name__ == "__main__":
    unittest.main()
