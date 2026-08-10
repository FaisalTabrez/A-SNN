from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.nmnist_accuracy_benchmark import (
    NMNIST_ACCURACY_ARMS,
    NMNISTAccuracyConfig,
    available_nmnist_accuracy_arms,
    build_nmnist_accuracy_model,
    bundle_nmnist_accuracy_artifacts,
    decide_nmnist_accuracy,
    encode_nmnist_full_resolution,
    estimate_nmnist_dense_macs,
    np,
    select_nmnist_accuracy_promoted_arms,
    torch,
)


class NMNISTAccuracyBenchmarkContractTest(unittest.TestCase):
    def test_frozen_arms(self) -> None:
        self.assertEqual(
            available_nmnist_accuracy_arms(),
            ("frame_cnn", "spatiotemporal_cnn", "conv_plif"),
        )
        self.assertEqual(available_nmnist_accuracy_arms(), NMNIST_ACCURACY_ARMS)

    @unittest.skipIf(np is None, "NumPy unavailable")
    def test_full_resolution_encoder_preserves_time_polarity_and_location(self) -> None:
        events = np.array(
            [(0, 0, 0, 0), (33, 33, 299999, 1), (17, 12, 150000, 1)],
            dtype=[("x", "i2"), ("y", "i2"), ("t", "i8"), ("p", "i1")],
        )
        encoded = encode_nmnist_full_resolution(
            events, timesteps=10, duration_us=300000
        )
        self.assertEqual(encoded.shape, (10, 2, 34, 34))
        self.assertEqual(int(encoded.sum()), 3)
        self.assertEqual(int(encoded[0, 0, 0, 0]), 1)
        self.assertEqual(int(encoded[9, 1, 33, 33]), 1)

    @unittest.skipIf(torch is None, "PyTorch unavailable")
    def test_each_arm_has_valid_logits_activity_and_operation_count(self) -> None:
        events = torch.zeros((2, 4, 2, 34, 34))
        events[:, :, :, 12:18, 12:18] = 1
        for arm in NMNIST_ACCURACY_ARMS:
            model = build_nmnist_accuracy_model(arm)
            model.eval()
            with torch.no_grad():
                logits, activity = model(events)
            self.assertEqual(tuple(logits.shape), (2, 10))
            self.assertEqual(activity.ndim, 0)
            self.assertGreater(estimate_nmnist_dense_macs(model, 4), 0)
            self.assertGreater(sum(parameter.numel() for parameter in model.parameters()), 0)

    def test_promotion_is_validation_only_bounded_and_deterministic(self) -> None:
        rows = [
            {"arm": "frame_cnn", "best_validation_accuracy": 0.980},
            {"arm": "spatiotemporal_cnn", "best_validation_accuracy": 0.985},
            {"arm": "conv_plif", "best_validation_accuracy": 0.978},
        ]
        promoted = select_nmnist_accuracy_promoted_arms(rows, gap=0.01, maximum=2)
        self.assertEqual(promoted, ["spatiotemporal_cnn", "frame_cnn"])

    def test_decision_returns_to_gen20_after_bounded_track(self) -> None:
        config = NMNISTAccuracyConfig()
        result = decide_nmnist_accuracy(
            [{"arm": "conv_plif", "mean_test_accuracy": 0.991}], config
        )
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["practical_accuracy_gate"])
        self.assertFalse(result["stretch_accuracy_gate"])
        self.assertEqual(result["next_milestone"], "return_to_gen20")

    def test_bundle_contains_manifest_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory)
            result = output / "nmnist_accuracy_benchmark.json"
            progress = output / "nmnist_accuracy_benchmark_progress.json"
            result.write_text('{"decision": "stop"}\n', encoding="utf-8")
            progress.write_text('{"screen_records": []}\n', encoding="utf-8")
            paths = bundle_nmnist_accuracy_artifacts(
                {"json": str(result), "progress": str(progress)}, output
            )
            with zipfile.ZipFile(paths["bundle"]) as archive:
                self.assertEqual(
                    set(archive.namelist()),
                    {
                        result.name,
                        progress.name,
                        "nmnist_accuracy_benchmark_manifest.json",
                    },
                )


if __name__ == "__main__":
    unittest.main()
