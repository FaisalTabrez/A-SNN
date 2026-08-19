from __future__ import annotations

import unittest

from ammc_gen5 import Gen26Config, decide_gen26, sparse_temporal_currents_fp64
from ammc_gen5.event_mnist import nn, torch


@unittest.skipIf(torch is None, "PyTorch unavailable")
class Gen26ContractTest(unittest.TestCase):
    def test_fp64_sparse_operator_matches_dense_nonbinary_values(self):
        torch.manual_seed(5)
        temporal = nn.Conv1d(4, 3, 3, padding=1)
        events = torch.rand((2, 5, 4)) * (torch.rand((2, 5, 4)) > 0.5)
        dense = temporal(events.transpose(1, 2)).transpose(1, 2)
        sparse = sparse_temporal_currents_fp64(events, temporal)
        self.assertTrue(torch.allclose(dense, sparse, atol=1e-6, rtol=1e-5))

    def test_decision_selects_count_repair(self):
        config = Gen26Config(seeds=(1, 2, 3), batch_sizes=(1,))
        records = []
        for seed in config.seeds:
            for variant in ("coo_fp32_counts", "coo_fp64_counts", "coo_fp32_binary"):
                records.append({
                    "seed": seed,
                    "variant": variant,
                    "maximum_current_difference": 1e-6,
                    "mean_current_difference": 1e-7,
                    "maximum_logit_difference": 1e-6,
                    "mean_logit_difference": 1e-7,
                    "prediction_agreement": 1.0,
                    "binary_vs_count_dense_prediction_agreement": 0.5,
                    "state_amplification_ratio": 1.0,
                    "source_nonbinary_fraction": 0.1,
                })
        decision = decide_gen26(records, config)
        self.assertEqual(decision["status"], "pass")
        self.assertEqual(decision["selected_operator"], "coo_fp64_counts")
        self.assertFalse(decision["hardware_energy_claim_authorized"])

    def test_binary_exact_without_semantic_stability_does_not_pass(self):
        config = Gen26Config(seeds=(1, 2, 3), batch_sizes=(1,))
        records = []
        for seed in config.seeds:
            for variant in ("coo_fp32_counts", "coo_fp64_counts", "coo_fp32_binary"):
                is_binary = variant == "coo_fp32_binary"
                records.append({
                    "seed": seed,
                    "variant": variant,
                    "maximum_current_difference": 0.0 if is_binary else 1e-2,
                    "mean_current_difference": 0.0,
                    "maximum_logit_difference": 0.0 if is_binary else 1e-2,
                    "mean_logit_difference": 0.0,
                    "prediction_agreement": 1.0,
                    "binary_vs_count_dense_prediction_agreement": 0.8,
                    "state_amplification_ratio": 1.0,
                    "source_nonbinary_fraction": 0.1,
                })
        decision = decide_gen26(records, config)
        self.assertEqual(decision["status"], "stop")
        self.assertEqual(decision["next_milestone"], "train_and_validate_binary_event_encoding")


if __name__ == "__main__":
    unittest.main()
