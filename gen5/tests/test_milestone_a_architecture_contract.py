from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.milestone_a_architecture import (
    MILESTONE_A_ARMS,
    _load_progress,
    _save_progress,
    available_milestone_a_arms,
    decide_milestone_a,
    hierarchical_residual_parameter_count,
    matched_hierarchical_residual_channels,
    select_promoted_arms,
)


class MilestoneAArchitectureContractTest(unittest.TestCase):
    def test_unified_arm_matrix_has_controls_and_causal_candidates(self) -> None:
        self.assertEqual(len(MILESTONE_A_ARMS), 5)
        self.assertEqual(
            available_milestone_a_arms(),
            (
                "temporal_conv1d",
                "dilated_tcn",
                "residual_lif",
                "hierarchical_residual_analog",
                "hierarchical_residual_lif",
            ),
        )
        self.assertEqual(sum(arm.conventional for arm in MILESTONE_A_ARMS), 2)
        self.assertEqual(sum(arm.causal_state for arm in MILESTONE_A_ARMS), 3)

    def test_hierarchical_width_is_parameter_matched(self) -> None:
        channels, actual = matched_hierarchical_residual_channels(
            700,
            35,
            133631,
            input_kernel_size=5,
            hidden_kernel_size=3,
            temporal_levels=(1, 2, 4, 8),
        )
        self.assertEqual(
            actual,
            hierarchical_residual_parameter_count(
                700,
                channels,
                35,
                input_kernel_size=5,
                hidden_kernel_size=3,
                temporal_levels=(1, 2, 4, 8),
                spiking=True,
            ),
        )
        self.assertLessEqual(actual, 133631)
        self.assertGreaterEqual(actual / 133631, 0.95)

    def test_promotion_uses_validation_budget_and_activity_gates(self) -> None:
        common = {
            "parameter_ratio_vs_target": 1.0,
            "checkpoint_activity": 0.05,
        }
        rows = [
            dict(
                common,
                arm="temporal_conv1d",
                model_kind="conv1d",
                conventional=True,
                best_validation_accuracy=0.55,
            ),
            dict(
                common,
                arm="dilated_tcn",
                model_kind="tcn",
                conventional=True,
                best_validation_accuracy=0.60,
            ),
            dict(
                common,
                arm="residual_lif",
                model_kind="residual_lif",
                conventional=False,
                best_validation_accuracy=0.59,
            ),
            dict(
                common,
                arm="hierarchical_residual_analog",
                model_kind="hierarchical_analog",
                conventional=False,
                best_validation_accuracy=0.56,
            ),
            dict(
                common,
                arm="hierarchical_residual_lif",
                model_kind="hierarchical_lif",
                conventional=False,
                best_validation_accuracy=0.595,
                checkpoint_activity=0.0,
            ),
        ]
        promoted = select_promoted_arms(
            rows,
            promotion_margin=0.02,
            minimum_parameter_ratio=0.95,
            maximum_parameter_ratio=1.05,
            minimum_spike_rate=0.01,
            maximum_spike_rate=0.30,
        )
        self.assertEqual(promoted, ("dilated_tcn", "residual_lif"))

    def test_decision_requires_competitive_sample_specific_state(self) -> None:
        conventional = {
            "arm": "dilated_tcn",
            "model_kind": "tcn",
            "causal_state": False,
            "runs": 3,
            "mean_full_accuracy": 0.60,
            "mean_gain_vs_best_conventional": 0.0,
            "mean_state_contribution_vs_direct_only": 0.0,
            "one_point_seed_count_state_contribution": 0,
            "mean_state_specificity_vs_shuffled": 0.0,
            "one_point_seed_count_state_specificity": 0,
            "mean_activity": 0.2,
            "best_conventional_arm": "dilated_tcn",
        }
        causal = dict(
            conventional,
            arm="hierarchical_residual_lif",
            model_kind="hierarchical_lif",
            causal_state=True,
            mean_full_accuracy=0.59,
            mean_gain_vs_best_conventional=-0.01,
            mean_state_contribution_vs_direct_only=0.03,
            one_point_seed_count_state_contribution=3,
            mean_state_specificity_vs_shuffled=0.02,
            one_point_seed_count_state_specificity=2,
            mean_activity=0.05,
        )
        decision = decide_milestone_a(
            [conventional, causal],
            causal_margin=0.01,
            minimum_spike_rate=0.01,
            maximum_spike_rate=0.30,
        )
        self.assertEqual(decision["status"], "pass")
        self.assertEqual(decision["next_milestone"], "hardware_efficiency")
        self.assertEqual(decision["qualified_arms"], ["hierarchical_residual_lif"])

        analog_only = dict(
            causal,
            arm="hierarchical_residual_analog",
            model_kind="hierarchical_analog",
        )
        stopped = decide_milestone_a(
            [conventional, analog_only],
            causal_margin=0.01,
            minimum_spike_rate=0.01,
            maximum_spike_rate=0.30,
        )
        self.assertEqual(stopped["status"], "stop")

    def test_progress_checkpoint_is_atomic_and_signature_guarded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "progress.json"
            signature = {"version": 1, "seed": 142}
            _save_progress(
                path,
                signature,
                stage="screen",
                screen_records=[{"seed": 142, "arm": "dilated_tcn"}],
                promoted_arms=(),
                confirmation_records=(),
            )
            payload = _load_progress(path, signature)
            self.assertEqual(payload["stage"], "screen")
            self.assertEqual(payload["screen_records"][0]["arm"], "dilated_tcn")
            self.assertFalse(path.with_suffix(".json.part").exists())
            with self.assertRaises(ValueError):
                _load_progress(path, {"version": 1, "seed": 143})


if __name__ == "__main__":
    unittest.main()
