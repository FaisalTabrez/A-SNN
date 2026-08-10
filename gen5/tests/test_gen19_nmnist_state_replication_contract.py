from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.gen19_nmnist_state_replication import (
    GEN19_ARMS,
    Gen19Config,
    available_gen19_arms,
    decide_gen19,
    encode_nmnist_events,
    np,
    run_gen19_nmnist_state_replication,
    summarize_gen19,
    torch,
)


class Gen19NMNISTStateReplicationContractTest(unittest.TestCase):
    def test_registered_arms_are_frozen(self) -> None:
        self.assertEqual(available_gen19_arms(), GEN19_ARMS)
        self.assertEqual(GEN19_ARMS, ("temporal_conv1d", "residual_lif"))

    @unittest.skipIf(np is None, "NumPy unavailable")
    def test_encoder_preserves_time_polarity_and_spatial_cells(self) -> None:
        events = np.array(
            [(0, 0, 0, 0), (33, 33, 299999, 1), (17, 17, 150000, 1)],
            dtype=[("x", "i2"), ("y", "i2"), ("t", "i8"), ("p", "i1")],
        )
        frame = encode_nmnist_events(
            events, timesteps=30, spatial_bins=8, duration_us=300000
        )
        self.assertEqual(frame.shape, (30, 128))
        self.assertEqual(int(frame.sum()), 3)
        self.assertEqual(int(frame[0, 0]), 1)
        self.assertEqual(int(frame[29, 127]), 1)

    def test_decision_requires_accuracy_causal_identity_and_activity(self) -> None:
        config = Gen19Config()
        records = _records(contribution=0.02, specificity=0.02, activity=0.10)
        decision = decide_gen19(
            summarize_gen19(records, minimum_state_effect=config.minimum_state_effect),
            config,
        )
        self.assertEqual(decision["status"], "pass")

        records = _records(contribution=0.02, specificity=-0.01, activity=0.10)
        decision = decide_gen19(
            summarize_gen19(records, minimum_state_effect=config.minimum_state_effect),
            config,
        )
        self.assertEqual(decision["status"], "stop")
        self.assertFalse(decision["state_identity_gate"])

    @unittest.skipIf(torch is None, "PyTorch unavailable")
    def test_cached_tiny_run_completes_without_tonic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = pathlib.Path(directory) / "ammc_cache"
            cache.mkdir()
            generator = torch.Generator().manual_seed(19)
            train_events = torch.randint(0, 2, (40, 4, 8), generator=generator, dtype=torch.uint8)
            train_labels = torch.arange(10).repeat_interleave(4)
            test_events = torch.randint(0, 2, (20, 4, 8), generator=generator, dtype=torch.uint8)
            test_labels = torch.arange(10).repeat_interleave(2)
            torch.save(
                {"events": train_events, "labels": train_labels},
                cache / "train_t4_s2_d100_nall_seed99.pt",
            )
            torch.save(
                {"events": test_events, "labels": test_labels},
                cache / "test_t4_s2_d100_nall_seed99.pt",
            )
            config = Gen19Config(
                timesteps=4,
                spatial_bins=2,
                duration_us=100,
                epochs=1,
                batch_size=10,
                data_seed=99,
                data_root=directory,
                download=False,
                validation_fraction=0.25,
                target_parameters=500,
                temporal_levels=(1,),
                temporal_conv_kernel_size=3,
            )
            progress_path = pathlib.Path(directory) / "progress.json"
            result = run_gen19_nmnist_state_replication(
                config, device="cpu", progress_path=progress_path
            )
            self.assertEqual(len(result.records), 3)
            self.assertEqual(result.dataset["test_samples"], 20)
            self.assertIn(result.decision["status"], {"pass", "stop"})
            resumed = run_gen19_nmnist_state_replication(
                config, device="cpu", progress_path=progress_path
            )
            self.assertEqual(resumed.records, result.records)


def _records(*, contribution: float, specificity: float, activity: float) -> list[dict]:
    rows = []
    for seed in (190, 191, 192):
        full = 0.95
        rows.append({
            "seed": seed,
            "conv_test_accuracy": 0.95,
            "full_accuracy": full,
            "direct_only_accuracy": full - contribution,
            "state_only_accuracy": 0.40,
            "shuffled_state_accuracy": full - specificity,
            "full_gain_vs_conv": 0.0,
            "state_contribution_vs_direct_only": contribution,
            "state_specificity_vs_shuffled": specificity,
            "direct_contribution_vs_state_only": 0.55,
            "full_spike_activity": activity,
            "conv_test_examples_per_second": 1000.0,
            "residual_test_examples_per_second": 500.0,
        })
    return rows


if __name__ == "__main__":
    unittest.main()
