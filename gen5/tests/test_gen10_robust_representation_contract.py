from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.gen10_robust_representation import (
    GEN10_REPRESENTATION_ARMS,
    available_gen10_representation_arms,
    decide_gen10_robust_representation,
    select_gen10_promoted_arms,
    summarize_gen10_confirmation,
)


class Gen10RobustRepresentationContractTest(unittest.TestCase):
    def test_registered_matrix_is_frozen(self) -> None:
        self.assertEqual(
            available_gen10_representation_arms(),
            (
                "dilated_tcn",
                "dropout_tcn",
                "masked_residual_analog",
                "masked_residual_lif",
            ),
        )
        self.assertEqual(len(GEN10_REPRESENTATION_ARMS), 4)
        self.assertEqual(sum(arm.conventional for arm in GEN10_REPRESENTATION_ARMS), 2)

    def test_promotion_requires_clean_damaged_budget_and_activity(self) -> None:
        rows = [
            _screen("dilated_tcn", True, 0.60, 0.50, 0.40),
            _screen("dropout_tcn", True, 0.59, 0.54, 0.40),
            _screen("masked_residual_analog", False, 0.595, 0.525, 0.20),
            _screen("masked_residual_lif", False, 0.595, 0.525, 0.08),
        ]
        promoted = select_gen10_promoted_arms(
            rows,
            promotion_margin=0.01,
            damaged_promotion_margin=0.02,
            minimum_parameter_ratio=0.95,
            maximum_parameter_ratio=1.05,
            minimum_spike_rate=0.01,
            maximum_spike_rate=0.30,
        )
        self.assertEqual(
            promoted,
            ("dilated_tcn", "dropout_tcn", "masked_residual_analog", "masked_residual_lif"),
        )
        rows[-1]["checkpoint_activity"] = 0.40
        self.assertNotIn(
            "masked_residual_lif",
            select_gen10_promoted_arms(
                rows,
                promotion_margin=0.01,
                damaged_promotion_margin=0.02,
                minimum_parameter_ratio=0.95,
                maximum_parameter_ratio=1.05,
                minimum_spike_rate=0.01,
                maximum_spike_rate=0.30,
            ),
        )

    def test_summary_and_terminal_gate_require_causal_robust_lif(self) -> None:
        records = []
        for seed in (1, 2, 3):
            records.extend(
                (
                    _confirmation(seed, "dilated_tcn", True, 0.60, 0.50),
                    _confirmation(seed, "dropout_tcn", True, 0.59, 0.56),
                    _confirmation(seed, "masked_residual_lif", False, 0.595, 0.555, causal=True),
                )
            )
        summary = summarize_gen10_confirmation(records)
        decision = decide_gen10_robust_representation(
            summary,
            accuracy_margin=0.01,
            causal_margin=0.005,
            robustness_margin=0.05,
            minimum_spike_rate=0.01,
            maximum_spike_rate=0.30,
        )
        self.assertEqual(decision["status"], "pass")
        candidate = next(row for row in summary if row["arm"] == "masked_residual_lif")
        self.assertEqual(candidate["causal_seed_count"], 3)
        self.assertEqual(candidate["specificity_seed_count"], 3)
        candidate["mean_damaged_state_specificity"] = 0.0
        self.assertEqual(
            decide_gen10_robust_representation(
                summary,
                accuracy_margin=0.01,
                causal_margin=0.005,
                robustness_margin=0.05,
                minimum_spike_rate=0.01,
                maximum_spike_rate=0.30,
            )["status"],
            "stop",
        )


def _screen(arm, conventional, clean, damaged, activity):
    return {
        "arm": arm,
        "conventional": conventional,
        "best_validation_accuracy": clean,
        "damaged_validation_accuracy": damaged,
        "parameter_ratio_vs_target": 1.0,
        "checkpoint_activity": activity,
    }


def _confirmation(seed, arm, conventional, clean, damaged, *, causal=False):
    return {
        "seed": seed,
        "arm": arm,
        "model_kind": "tcn" if conventional else "shared_residual",
        "conventional": conventional,
        "causal_state": causal,
        "clean_accuracy": clean,
        "damaged_accuracy": damaged,
        "damaged_state_contribution": 0.01 if causal else None,
        "damaged_state_specificity": 0.01 if causal else None,
        "checkpoint_activity": 0.08 if causal else 0.30,
        "activity_kind": "spike_rate" if causal else "relu_activation",
        "mean_absolute_gate": 0.10 if causal else 0.0,
        "effective_trainable_parameters": 100,
        "test_examples_per_second": 1000.0,
        "train_seconds": 1.0,
    }


if __name__ == "__main__":
    unittest.main()
