from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.gen17_sparse_spiking_credit import (
    GEN17_STRATEGIES,
    Gen17Config,
    available_gen17_strategies,
    bernoulli_spike_code,
    decide_gen17,
    summarize_gen17,
    torch,
)


class Gen17SparseSpikingCreditContractTest(unittest.TestCase):
    def test_registered_strategies_are_frozen(self) -> None:
        self.assertEqual(available_gen17_strategies(), GEN17_STRATEGIES)
        self.assertEqual(len(GEN17_STRATEGIES), 5)

    @unittest.skipIf(torch is None, "PyTorch unavailable")
    def test_bernoulli_encoder_is_binary_and_seeded(self) -> None:
        sensory = torch.full((256, 8), 0.25)
        first_generator = torch.Generator().manual_seed(9)
        second_generator = torch.Generator().manual_seed(9)
        first = bernoulli_spike_code(sensory, first_generator)
        second = bernoulli_spike_code(sensory, second_generator)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(bool(((first == 0.0) | (first == 1.0)).all()))
        self.assertAlmostEqual(float(first.mean()), 0.25, delta=0.04)

    def test_decision_requires_spiking_gain_activity_and_identity(self) -> None:
        records = _records(spiking=2.0, analog=2.05, shuffled=0.5, density=0.20)
        decision = decide_gen17(records, summarize_gen17(records), Gen17Config())
        self.assertEqual(decision["status"], "pass")
        records = _records(spiking=2.0, analog=2.05, shuffled=0.5, density=0.0)
        decision = decide_gen17(records, summarize_gen17(records), Gen17Config())
        self.assertEqual(decision["status"], "stop")


def _records(*, spiking, analog, shuffled, density):
    final = {
        "static_spiking_policy": 0.5,
        "oracle_food_reflex": 3.0,
        "manual_analog_score_policy": analog,
        "manual_spiking_score_policy": spiking,
        "manual_spiking_shuffled_reward": shuffled,
    }
    rows = []
    for seed in (1, 2, 3):
        for strategy in GEN17_STRATEGIES:
            is_spiking = strategy in (
                "static_spiking_policy",
                "manual_spiking_score_policy",
                "manual_spiking_shuffled_reward",
            )
            for phase, value in (("baseline", 0.5), ("evaluation", final[strategy])):
                rows.append({
                    "seed": seed,
                    "strategy": strategy,
                    "phase": phase,
                    "steps": 100,
                    "mean_net_fitness_per_1000_steps": value,
                    "mean_shaped_reward_per_1000_steps": value,
                    "mean_policy_entropy": 1.0,
                    "mean_evaluation_spike_density": density if is_spiking else 0.0,
                    "mean_training_loss": 0.0,
                    "mean_training_reward_per_1000_steps": 0.0,
                    "training_updates": 1,
                    "maximum_score_gradient_error": 1e-8,
                    "policy_weight_delta_norm": 1.0,
                    "mean_training_spike_density": density if is_spiking else 0.0,
                })
    return rows


if __name__ == "__main__":
    unittest.main()
