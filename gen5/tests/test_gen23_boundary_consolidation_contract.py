from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.gen21_matched_causal_mechanisms import Gen21Config, Gen21MechanismReadout
from ammc_gen5.gen23_boundary_consolidation import GEN23_ARMS, Gen23Config, _consolidate_boundary, available_gen23_arms, decide_gen23, torch
from ammc_gen5.shd_benchmark import SHDConfig
from ammc_gen5.shd_state_placement_diagnostic import ResidualTemporalConvStateClassifier


class Gen23ContractTest(unittest.TestCase):
    def test_frozen_protocol(self) -> None:
        config = Gen23Config()
        self.assertEqual(available_gen23_arms(), GEN23_ARMS)
        self.assertEqual(config.protected_fraction, 0.50)
        self.assertEqual(config.stw_decay, 0.995)
        self.assertEqual(config.minimum_qualifying_seed_count, 3)

    @unittest.skipIf(torch is None, "PyTorch unavailable")
    def test_boundary_transfer_preserves_immediate_function(self) -> None:
        base = SHDConfig(input_neurons=4, classes=3, timesteps=4, hidden_neurons=2, max_edges=16)
        backbone = ResidualTemporalConvStateClassifier(base, channels=3, kernel_size=3, temporal_levels=(1, 2), dynamics="lif", surrogate_slope=10.0)
        config = Gen21Config(input_neurons=4, classes=3, timesteps=4, temporal_levels=(1, 2), delay_slots=3)
        model = Gen21MechanismReadout(backbone, "dual_memory_only", config, seed=7)
        model.register_buffer("protected_mask", torch.zeros_like(model.active_mask))
        with torch.no_grad(): model.delta.normal_(0.0, 0.1)
        events = torch.rand((8, 4, 4)); before = model(events).detach()
        _consolidate_boundary(model, 0.5, seed=7, shuffled=False)
        self.assertTrue(torch.allclose(before, model(events).detach(), atol=1e-6))
        self.assertGreater(int(model.protected_mask.sum().item()), 0)

    def test_decision_requires_every_causal_gate(self) -> None:
        rows = []
        for seed in Gen23Config().seeds:
            for arm in GEN23_ARMS:
                rows.append({
                    "seed": seed, "arm": arm,
                    "after_b_a_accuracy": 0.52 if arm == "boundary_selective_dual_memory" else 0.50,
                    "after_b_b_accuracy": 0.51, "clean_retention_drop": 0.01,
                    "stability_plasticity_score": 0.515 if arm == "boundary_selective_dual_memory" else 0.50,
                    "a_retention_gain_vs_single": 0.02 if arm == "boundary_selective_dual_memory" else 0.0,
                    "b_accuracy_gain_vs_single": 0.0,
                    "ltw_causal_margin": 0.02 if arm == "boundary_selective_dual_memory" else 0.0,
                    "selection_identity_margin": 0.015 if arm == "boundary_selective_dual_memory" else 0.0,
                    "boundary_accuracy_change": 0.0, "protected_slots": 10,
                    "active_slots": 20, "allocated_slots": 60, "adapter_memory_bytes": 960,
                })
        self.assertEqual(decide_gen23(rows, Gen23Config())["status"], "pass")
        dual = [row for row in rows if row["arm"] == "boundary_selective_dual_memory"]
        for row in dual[:3]: row["selection_identity_margin"] = 0.0
        self.assertEqual(decide_gen23(rows, Gen23Config())["status"], "stop")


if __name__ == "__main__":
    unittest.main()
