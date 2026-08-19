from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
import zipfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.gen20_spiking_spatiotemporal import (
    GEN20_ARMS,
    Gen20Config,
    Gen20Result,
    MultiTimescaleResidualPLIF,
    available_gen20_arms,
    build_gen20_model,
    bundle_gen20_artifacts,
    decide_gen20,
    estimate_gen20_operations,
    gen20_plot_series,
    select_gen20_promoted_arms,
    torch,
)


class Gen20ContractTest(unittest.TestCase):
    def test_frozen_protocol(self) -> None:
        config = Gen20Config()
        self.assertEqual(available_gen20_arms(), GEN20_ARMS)
        self.assertEqual(config.screen_seed, 220)
        self.assertEqual(config.confirmation_seeds, (221, 222, 223))
        self.assertEqual(config.screen_epochs, 6)
        self.assertEqual(config.confirmation_epochs, 12)
        self.assertEqual(config.minimum_screen_accuracy, 0.975)
        self.assertEqual(config.minimum_ops_reduction, 5.0)

    @unittest.skipIf(torch is None, "PyTorch unavailable")
    def test_models_preserve_shape_and_spiking_state_control(self) -> None:
        events = torch.zeros((2, 4, 2, 34, 34))
        events[:, :, :, 8:20, 8:20] = 1
        for arm in GEN20_ARMS:
            model = build_gen20_model(arm)
            model.eval()
            with torch.no_grad():
                logits, activity = model(events)
            self.assertEqual(tuple(logits.shape), (2, 10))
            self.assertEqual(activity.ndim, 0)
            dense, analog = estimate_gen20_operations(model, 4)
            self.assertGreater(dense, 0)
            self.assertGreater(analog, 0)
            self.assertLessEqual(analog, dense)
            if isinstance(model, MultiTimescaleResidualPLIF):
                with torch.no_grad():
                    removed, _ = model(events, state_mode="removed")
                self.assertEqual(tuple(removed.shape), (2, 10))
                self.assertLess(analog, dense)

    def test_promotion_applies_accuracy_activity_and_arm_scope(self) -> None:
        rows = [
            {"arm": "spatiotemporal_cnn", "best_validation_accuracy": 0.995, "validation_activity": 0.0},
            {"arm": "conv_plif", "best_validation_accuracy": 0.980, "validation_activity": 0.10},
            {"arm": "multiscale_residual_plif", "best_validation_accuracy": 0.981, "validation_activity": 0.12},
            {"arm": "distilled_multiscale_plif", "best_validation_accuracy": 0.984, "validation_activity": 0.08},
        ]
        self.assertEqual(
            select_gen20_promoted_arms(rows, Gen20Config()),
            ["distilled_multiscale_plif", "multiscale_residual_plif"],
        )
        rows[-1]["validation_activity"] = 0.0
        self.assertEqual(
            select_gen20_promoted_arms(rows, Gen20Config()),
            ["multiscale_residual_plif"],
        )

    def test_decision_requires_every_frozen_gate(self) -> None:
        teacher = {
            "arm": "spatiotemporal_cnn",
            "mean_test_accuracy": 0.995,
        }
        candidate = {
            "arm": "distilled_multiscale_plif",
            "mean_test_accuracy": 0.991,
            "minimum_test_accuracy": 0.989,
            "mean_test_activity": 0.12,
            "mean_state_contribution": 0.02,
            "state_contribution_seed_count": 3,
            "mean_temporal_order_contribution": 0.018,
            "temporal_order_seed_count": 2,
            "ops_reduction_vs_dense_teacher": 7.0,
        }
        decision = decide_gen20(
            [teacher, candidate], ["distilled_multiscale_plif"], Gen20Config()
        )
        self.assertEqual(decision["status"], "pass")
        self.assertEqual(decision["qualified_arms"], ["distilled_multiscale_plif"])
        self.assertFalse(decision["energy_claim_authorized"])
        candidate["mean_temporal_order_contribution"] = 0.0
        self.assertEqual(
            decide_gen20(
                [teacher, candidate], ["distilled_multiscale_plif"], Gen20Config()
            )["status"],
            "stop",
        )

    def test_empty_promotion_is_terminal_stop(self) -> None:
        decision = decide_gen20([], [], Gen20Config())
        self.assertEqual(decision["status"], "stop")
        self.assertEqual(decision["next_milestone"], "evidence_synthesis")

    def test_early_stop_plot_uses_screen_records(self) -> None:
        config = Gen20Config()
        result = Gen20Result(
            config=config.__dict__,
            device="cpu",
            dataset={},
            screen_records=[
                {
                    "arm": "spatiotemporal_cnn",
                    "best_validation_accuracy": 0.991,
                    "validation_activity": 0.0,
                    "dense_macs_per_sample": 1000,
                    "analog_dense_macs_per_sample": 1000,
                },
                {
                    "arm": "multiscale_residual_plif",
                    "best_validation_accuracy": 0.963,
                    "validation_activity": 0.10,
                    "dense_macs_per_sample": 200,
                    "analog_dense_macs_per_sample": 50,
                },
            ],
            promoted_arms=[],
            confirmation_records=[],
            summary=[],
            decision={"status": "stop"},
        )
        series = gen20_plot_series(result)
        self.assertEqual(series["stage"], "screen")
        self.assertEqual(series["accuracy"], [99.1, 96.3])
        self.assertEqual(series["activity"], [0.0, 10.0])
        self.assertAlmostEqual(series["reduction"][1], 1000 / 65)
        self.assertEqual(series["accuracy_gate"], 97.5)

    def test_bundle_contains_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory)
            result = output / "gen20_spiking_spatiotemporal.json"
            result.write_text('{"decision": "stop"}\n', encoding="utf-8")
            paths = bundle_gen20_artifacts({"json": str(result)}, output)
            with zipfile.ZipFile(paths["bundle"]) as archive:
                self.assertEqual(
                    set(archive.namelist()),
                    {result.name, "gen20_spiking_spatiotemporal_manifest.json"},
                )


if __name__ == "__main__":
    unittest.main()
