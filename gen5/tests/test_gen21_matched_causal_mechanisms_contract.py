from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.gen21_matched_causal_mechanisms import (
    GEN21_ARMS,
    GEN21_PRIMARY_ARMS,
    Gen21Config,
    Gen21MechanismReadout,
    Gen21Result,
    available_gen21_arms,
    bundle_gen21_artifacts,
    decide_gen21,
    _load_progress,
    _save_progress,
    select_gen21_promoted_arms,
    torch,
)
from ammc_gen5.shd_benchmark import SHDConfig
from ammc_gen5.shd_state_placement_diagnostic import ResidualTemporalConvStateClassifier


class Gen21ContractTest(unittest.TestCase):
    def test_frozen_protocol(self) -> None:
        config = Gen21Config()
        self.assertEqual(available_gen21_arms(), GEN21_ARMS)
        self.assertEqual(config.screen_seed, 321)
        self.assertEqual(config.confirmation_seeds, (322, 323, 324))
        self.assertEqual(config.sensor_damage_fraction, 0.35)
        self.assertEqual(config.minimum_adaptation_gain, 0.01)
        self.assertEqual(config.minimum_causal_margin, 0.005)

    @unittest.skipIf(torch is None, "PyTorch unavailable")
    def test_backbone_extraction_preserves_forward_and_arm_budgets(self) -> None:
        backbone_config = SHDConfig(
            input_neurons=4, classes=3, timesteps=4, hidden_neurons=2,
            max_edges=16, sensor_fanout=1, recurrent_fanout=1,
        )
        backbone = ResidualTemporalConvStateClassifier(
            backbone_config, channels=3, kernel_size=3,
            temporal_levels=(1, 2), dynamics="lif", surrogate_slope=10.0,
        )
        events = torch.rand((5, 4, 4))
        direct, state, final, activity = backbone.encode_trace(events)
        features, encoded_activity = backbone.encode_features(events)
        self.assertEqual(tuple(direct.shape), (5, 4, 3))
        self.assertEqual(tuple(state.shape), (5, 4, 3))
        self.assertEqual(tuple(final.shape), (5, 3))
        self.assertTrue(torch.allclose(activity, encoded_activity))
        self.assertTrue(torch.allclose(backbone(events), backbone.classifier(features)))

        config = Gen21Config(
            input_neurons=4, classes=3, timesteps=4, temporal_levels=(1, 2),
            delay_slots=3, active_slot_fraction=0.5,
        )
        budgets = []
        for arm in GEN21_ARMS:
            model = Gen21MechanismReadout(backbone, arm, config, seed=1)
            logits, rate = model(events, return_event_rate=True)
            self.assertEqual(tuple(logits.shape), (5, 3))
            self.assertEqual(rate.ndim, 0)
            budgets.append((model.allocated_slots, model.active_slots))
        self.assertEqual(len(set(budgets)), 1)

    def test_screen_promotion_requires_all_three_gates(self) -> None:
        config = Gen21Config()
        rows = [
            {"arm": arm, "adaptation_gain": 0.02, "retention_drop": 0.01, "causal_margin": 0.01}
            for arm in GEN21_PRIMARY_ARMS
        ]
        self.assertEqual(select_gen21_promoted_arms(rows, config), list(GEN21_PRIMARY_ARMS))
        rows[1]["causal_margin"] = 0.0
        self.assertNotIn("dual_memory_only", select_gen21_promoted_arms(rows, config))

    def test_combined_arm_requires_every_mechanism_to_confirm(self) -> None:
        config = Gen21Config()
        summary = [
            {
                "arm": arm,
                "positive_adaptation_seed_count": 2,
                "positive_causal_seed_count": 2,
                "mean_retention_drop": 0.01,
            }
            for arm in GEN21_PRIMARY_ARMS
        ]
        decision = decide_gen21(summary, GEN21_PRIMARY_ARMS, config)
        self.assertEqual(decision["status"], "pass")
        self.assertTrue(decision["combined_arm_authorized"])
        summary[-1]["positive_causal_seed_count"] = 1
        decision = decide_gen21(summary, GEN21_PRIMARY_ARMS, config)
        self.assertFalse(decision["combined_arm_authorized"])
        self.assertFalse(decision["hardware_energy_claim_authorized"])

    def test_bundle_contains_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory)
            result_path = output / "gen21_matched_causal_mechanisms.json"
            result_path.write_text("{}\n", encoding="utf-8")
            paths = bundle_gen21_artifacts({"json": str(result_path)}, output)
            with zipfile.ZipFile(paths["bundle"]) as archive:
                self.assertEqual(
                    set(archive.namelist()),
                    {result_path.name, "gen21_matched_causal_mechanisms_manifest.json"},
                )

    def test_progress_round_trip_normalizes_tuple_fields(self) -> None:
        config = Gen21Config()
        rows = [{"stage": "screen", "seed": 321, "arm": "static_backbone"}]
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "progress.json"
            _save_progress(path, config, rows, "screen")
            self.assertEqual(_load_progress(path, config), rows)


if __name__ == "__main__":
    unittest.main()
