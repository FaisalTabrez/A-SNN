from __future__ import annotations

import pathlib
import sys
import unittest

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / "ammc_gen5"
sys.path.insert(0, str(PACKAGE))

from gen30_dendritic_predictive_credit import (  # noqa: E402
    GEN30_ARMS,
    DendriticCreditNetwork,
    Gen30Config,
    decide_gen30,
    generate_contextual_binding,
    summarize_gen30,
    torch,
)


class Gen30DendriticPredictiveCreditContractTest(unittest.TestCase):
    def test_frozen_arm_set_contains_upper_and_causal_controls(self):
        self.assertEqual(GEN30_ARMS, (
            "static",
            "bptt",
            "eprop_broadcast",
            "dendritic_predictive_credit",
            "dpc_shuffled_apical",
            "dpc_no_eligibility",
            "dpc_shuffled_modulator",
        ))

    def test_context_mapping_conflicts_and_delay_is_nonzero(self):
        if torch is None:
            self.skipTest("PyTorch unavailable")
        config = Gen30Config()
        events_a, labels_a = generate_contextual_binding(config, samples=64, context=0, seed=7)
        events_b, labels_b = generate_contextual_binding(config, samples=64, context=1, seed=7)
        cues = events_a[:, 0, :4].argmax(1)
        self.assertTrue(torch.equal(labels_a, cues))
        self.assertTrue(torch.equal(labels_b, torch.tensor((1, 0, 3, 2))[cues]))
        self.assertTrue(torch.equal(events_a[:, 0, :4], events_b[:, 0, :4]))
        self.assertGreater(config.query_time - config.context_time, 10)

    def test_fixed_resource_budget_and_symmetric_feedback(self):
        config = Gen30Config()
        self.assertEqual(config.train_samples_per_context, 2048)
        self.assertEqual(config.test_samples_per_context, 1024)
        self.assertEqual(config.batch_size, 256)
        self.assertEqual((config.phase_a_epochs, config.phase_b_epochs), (10, 10))
        if torch is None:
            self.skipTest("PyTorch unavailable")
        model = DendriticCreditNetwork(config)
        self.assertTrue(torch.equal(model.feedback, model.decoder))
        self.assertEqual(model.w_in.numel() + model.w_rec.numel(), 4800)

    def test_decision_requires_replication_and_all_causal_controls(self):
        config = Gen30Config()
        records = []
        joint = {
            "static": 0.25,
            "bptt": 0.94,
            "eprop_broadcast": 0.91,
            "dendritic_predictive_credit": 0.89,
            "dpc_shuffled_apical": 0.70,
            "dpc_no_eligibility": 0.68,
            "dpc_shuffled_modulator": 0.69,
        }
        for seed in config.seeds:
            for arm in GEN30_ARMS:
                dpc = arm == "dendritic_predictive_credit"
                after_a = 0.90 if arm != "static" else 0.25
                retained = 0.88 if dpc else joint[arm]
                learned = 0.90 if dpc else joint[arm]
                records.append({
                    "seed": seed,
                    "arm": arm,
                    "after_a_a_accuracy": after_a,
                    "after_a_b_accuracy": 0.25,
                    "after_b_a_accuracy": retained,
                    "after_b_b_accuracy": learned,
                    "a_retention_drop": after_a - retained,
                    "joint_after_b_accuracy": 0.5 * (retained + learned),
                    "mean_spike_activity": 0.10,
                    "trainable_synapses": 4800,
                    "seconds": 1.0,
                })
        summary = summarize_gen30(records)
        decision = decide_gen30(records, summary, config)
        self.assertEqual(decision["status"], "pass")
        self.assertEqual(decision["qualified_seed_count"], 10)
        self.assertTrue(all(decision["gates"].values()))
        self.assertFalse(decision["structural_plasticity_claim_authorized"])

    def test_missing_arm_stops(self):
        decision = decide_gen30([], [], Gen30Config())
        self.assertEqual(decision["status"], "stop")
        self.assertEqual(decision["next_milestone"], "complete_gen30")


if __name__ == "__main__":
    unittest.main()
