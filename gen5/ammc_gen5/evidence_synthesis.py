"""Reproducible synthesis and architecture closeout of Gen-5 evidence."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import pathlib


EVIDENCE_FILENAMES = {
    "phase44": "shd_calibrated_baselines.json",
    "phase45": "shd_spiking_temporal_conv.json",
    "phase46": "shd_state_placement_diagnostic.json",
    "phase47": "shd_residual_state_contribution.json",
    "phase48": "ssc_residual_lif_replication.json",
    "phase49": "ssc_efficiency_baselines.json",
    "milestone_a": "milestone_a_architecture.json",
    "gen6": "gen6_successor.json",
    "gen7": "gen7_predictive_state.json",
    "gen8": "gen8_temporal_binding.json",
    "gen9": "gen9_continual_adaptation.json",
    "gen10": "gen10_robust_representation.json",
    "gen11": "gen11_plastic_adapter.json",
    "gen12": "gen12_associative_memory.json",
    "gen13": "gen13_local_plasticity.json",
    "gen14": "gen14_reward_eligibility.json",
    "gen15": "gen15_reward_baseline.json",
    "gen16": "gen16_local_score_credit.json",
    "gen17": "gen17_sparse_spiking_credit.json",
    "gen18": "gen18_local_credit_replication.json",
    "gen19": "gen19_nmnist_state_replication.json",
    "gen20": "gen20_spiking_spatiotemporal.json",
}


@dataclass
class Gen5EvidenceSynthesisResult:
    sources: dict[str, str]
    metrics: dict[str, float]
    claims: list[dict]
    roadmap: list[dict]

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "gen5_evidence_synthesis.json"
        claims_path = output / "gen5_evidence_claims.csv"
        report_path = output / "gen5_evidence_report.md"
        payload = {
            "sources": self.sources,
            "metrics": self.metrics,
            "claims": self.claims,
            "roadmap": self.roadmap,
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        _write_csv(claims_path, self.claims)
        report_path.write_text(_render_report(self), encoding="utf-8")
        paths = {
            "json": str(json_path),
            "claims_csv": str(claims_path),
            "report": str(report_path),
        }
        if plot:
            plot_path = output / "gen5_evidence_summary.png"
            plot_gen5_evidence_synthesis(self, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def synthesize_gen5_evidence(
    evidence_root: str | pathlib.Path,
) -> Gen5EvidenceSynthesisResult:
    root = pathlib.Path(evidence_root)
    sources: dict[str, str] = {}
    evidence: dict[str, dict] = {}
    for phase, filename in EVIDENCE_FILENAMES.items():
        matches = sorted(root.rglob(filename))
        if not matches:
            raise FileNotFoundError(f"missing {phase} evidence: {filename}")
        selected = matches[-1]
        sources[phase] = str(selected)
        evidence[phase] = json.loads(selected.read_text(encoding="utf-8"))

    phase44_conv = _summary_row(evidence["phase44"], "temporal_conv1d")
    phase45_lif = _summary_row(evidence["phase45"], "temporal_conv_leaky_lif")
    phase45_conv = _summary_row(evidence["phase45"], "temporal_conv1d")
    phase46_lif = _summary_row(evidence["phase46"], "leaky_lif_residual")
    phase46_conv = _summary_row(evidence["phase46"], "temporal_conv1d")
    phase47_lif = _summary_row(evidence["phase47"], "residual_lif")
    phase48 = evidence["phase48"]["summary"]
    phase49_conv = _summary_row(evidence["phase49"], "temporal_conv1d")
    phase49_lif = _summary_row(evidence["phase49"], "residual_lif")
    phase49_tcn = _summary_row(evidence["phase49"], "dilated_tcn")
    milestone_a = evidence["milestone_a"]
    gen6 = evidence["gen6"]
    gen7 = evidence["gen7"]
    gen8 = evidence["gen8"]
    gen9 = evidence["gen9"]
    gen10 = evidence["gen10"]
    gen11 = evidence["gen11"]
    gen12 = evidence["gen12"]
    gen13 = evidence["gen13"]
    gen14 = evidence["gen14"]
    gen15 = evidence["gen15"]
    gen16 = evidence["gen16"]
    gen17 = evidence["gen17"]
    gen18 = evidence["gen18"]
    gen19 = evidence["gen19"]
    gen20 = evidence["gen20"]
    milestone_screen = {
        row["arm"]: row for row in milestone_a["screen_records"]
    }
    milestone_tcn = _summary_row(milestone_a, "dilated_tcn", key="confirmation_summary")
    milestone_tcn_validation = float(
        milestone_screen["dilated_tcn"]["best_validation_accuracy"]
    )
    milestone_hierarchical_gaps = (
        milestone_tcn_validation
        - float(
            milestone_screen["hierarchical_residual_analog"][
                "best_validation_accuracy"
            ]
        ),
        milestone_tcn_validation
        - float(
            milestone_screen["hierarchical_residual_lif"][
                "best_validation_accuracy"
            ]
        ),
    )
    milestone_scaling_passed = min(milestone_hierarchical_gaps) <= 0.02
    milestone_qualified_count = len(milestone_a["decision"]["qualified_arms"])
    gen6_tcn = _summary_row(gen6, "dilated_tcn", key="confirmation_summary")
    gen6_lif = _summary_row(
        gen6, "shared_residual_lif", key="confirmation_summary"
    )
    gen6_qualified_count = len(gen6["decision"]["qualified_arms"])
    gen7_tcn = _summary_row(gen7, "dilated_tcn", key="confirmation_summary")
    gen7_lif = _summary_row(
        gen7, "lif_paired_predictive", key="confirmation_summary"
    )
    gen7_shuffled = _summary_row(
        gen7, "lif_shuffled_predictive", key="confirmation_summary"
    )
    gen7_qualified_count = len(gen7["decision"]["qualified_arms"])
    gen8_screen = {row["arm"]: row for row in gen8["screen_records"]}
    gen8_tcn = _summary_row(gen8, "dilated_tcn", key="confirmation_summary")
    gen8_pooled = _summary_row(
        gen8, "lif_pooled_predictive", key="confirmation_summary"
    )
    gen8_analog = _summary_row(
        gen8, "analog_time_local_binding", key="confirmation_summary"
    )
    gen8_qualified_count = len(gen8["decision"]["qualified_arms"])
    gen9_screen = {row["arm"]: row for row in gen9["screen_records"]}
    gen9_static = _strategy_row(gen9, "tcn_static")
    gen9_readout = _strategy_row(gen9, "tcn_readout")
    gen9_full = _strategy_row(gen9, "tcn_full_finetune")
    gen9_qualified_count = len(gen9["decision"]["qualified_arms"])
    gen10_screen = {row["arm"]: row for row in gen10["screen_records"]}
    gen10_tcn = _summary_row(gen10, "dilated_tcn", key="confirmation_summary")
    gen10_dropout = _summary_row(gen10, "dropout_tcn", key="confirmation_summary")
    gen10_qualified_count = len(gen10["decision"]["qualified_arms"])
    gen11_static = _strategy_row(gen11, "dropout_tcn_static", key="summary")
    gen11_readout = _strategy_row(gen11, "dropout_tcn_readout", key="summary")
    gen11_full = _strategy_row(gen11, "dropout_tcn_full_finetune", key="summary")
    gen11_analog = _strategy_row(gen11, "analog_state_adapter", key="summary")
    gen11_lif = _strategy_row(gen11, "lif_state_adapter", key="summary")
    gen11_qualified_count = len(gen11["decision"]["qualified_arms"])
    gen12_static = _strategy_row(gen12, "dropout_tcn_static", key="summary")
    gen12_readout = _strategy_row(gen12, "dropout_tcn_readout", key="summary")
    gen12_full = _strategy_row(gen12, "dropout_tcn_full_finetune", key="summary")
    gen12_dense = _strategy_row(gen12, "dense_prototype_memory", key="summary")
    gen12_spiking = _strategy_row(gen12, "spiking_prototype_memory", key="summary")
    gen12_qualified_count = len(gen12["decision"]["qualified_arms"])
    gen13_static = _strategy_row(gen13, "dropout_tcn_static", key="summary")
    gen13_readout = _strategy_row(gen13, "dropout_tcn_readout", key="summary")
    gen13_full = _strategy_row(gen13, "dropout_tcn_full_finetune", key="summary")
    gen13_analog = _strategy_row(gen13, "analog_three_factor_readout", key="summary")
    gen13_spiking = _strategy_row(gen13, "spiking_three_factor_readout", key="summary")
    gen13_qualified_count = len(gen13["decision"]["qualified_arms"])
    gen14_static = _strategy_row(gen14, "static_random", key="summary")
    gen14_oracle = _strategy_row(gen14, "oracle_food_reflex", key="summary")
    gen14_analog = _strategy_row(gen14, "analog_reward_eligibility", key="summary")
    gen14_spiking = _strategy_row(gen14, "spiking_reward_eligibility", key="summary")
    gen14_shuffled = _strategy_row(gen14, "spiking_shuffled_reward", key="summary")
    gen15_static = _strategy_row(gen15, "static_random", key="summary")
    gen15_oracle = _strategy_row(gen15, "oracle_food_reflex", key="summary")
    gen15_reinforce = _strategy_row(gen15, "reinforce_shared_policy", key="summary")
    gen15_shuffled = _strategy_row(gen15, "reinforce_shuffled_reward", key="summary")
    gen16_static = _strategy_row(gen16, "static_linear_policy", key="summary")
    gen16_oracle = _strategy_row(gen16, "oracle_food_reflex", key="summary")
    gen16_autograd = _strategy_row(gen16, "autograd_score_policy", key="summary")
    gen16_local = _strategy_row(gen16, "manual_local_score_policy", key="summary")
    gen16_shuffled = _strategy_row(gen16, "manual_local_shuffled_reward", key="summary")
    gen17_static = _strategy_row(gen17, "static_spiking_policy", key="summary")
    gen17_oracle = _strategy_row(gen17, "oracle_food_reflex", key="summary")
    gen17_analog = _strategy_row(gen17, "manual_analog_score_policy", key="summary")
    gen17_spiking = _strategy_row(gen17, "manual_spiking_score_policy", key="summary")
    gen17_shuffled = _strategy_row(gen17, "manual_spiking_shuffled_reward", key="summary")
    gen18_static = _strategy_row(gen18, "static_linear_policy", key="summary")
    gen18_oracle = _strategy_row(gen18, "oracle_food_reflex", key="summary")
    gen18_local = _strategy_row(gen18, "manual_local_score_policy", key="summary")
    gen18_shuffled = _strategy_row(gen18, "manual_local_shuffled_reward", key="summary")
    gen20_screen = {row["arm"]: row for row in gen20["screen_records"]}
    gen20_teacher = gen20_screen["spatiotemporal_cnn"]
    gen20_conv = gen20_screen["conv_plif"]
    gen20_multiscale = gen20_screen["multiscale_residual_plif"]
    gen20_distilled = gen20_screen["distilled_multiscale_plif"]
    gen20_teacher_ops = float(gen20_teacher["dense_macs_per_sample"])
    gen20_multiscale_proxy = _screen_ops_proxy(gen20_multiscale)
    gen20_distilled_proxy = _screen_ops_proxy(gen20_distilled)

    shd_state_only_gap = (
        float(phase45_lif["mean_checkpoint_test_accuracy"])
        - float(phase45_conv["mean_checkpoint_test_accuracy"])
    )
    shd_residual_gain = (
        float(phase46_lif["mean_checkpoint_test_accuracy"])
        - float(phase46_conv["mean_checkpoint_test_accuracy"])
    )
    ssc_tcn_gain = (
        float(phase49_tcn["mean_test_accuracy"])
        - float(phase49_lif["mean_test_accuracy"])
    )
    throughput_ratio = (
        float(phase49_lif["mean_test_examples_per_second"])
        / float(phase49_tcn["mean_test_examples_per_second"])
    )
    mac_reduction = 1.0 - (
        float(phase49_lif["dense_macs_per_sample"])
        / float(phase49_tcn["dense_macs_per_sample"])
    )
    metrics = {
        "phase44_shd_conv1d_accuracy": float(
            phase44_conv["mean_checkpoint_test_accuracy"]
        ),
        "shd_state_only_lif_accuracy": float(
            phase45_lif["mean_checkpoint_test_accuracy"]
        ),
        "shd_state_only_lif_gap_vs_conv": shd_state_only_gap,
        "shd_residual_lif_accuracy": float(
            phase46_lif["mean_checkpoint_test_accuracy"]
        ),
        "shd_residual_lif_gain_vs_conv": shd_residual_gain,
        "shd_state_contribution_vs_direct_only": float(
            phase47_lif["mean_state_contribution_vs_direct_only"]
        ),
        "shd_state_specificity_vs_shuffled": float(
            phase47_lif["mean_state_specificity_vs_shuffled"]
        ),
        "ssc_residual_lif_accuracy": float(phase48["mean_full_accuracy"]),
        "ssc_residual_lif_gain_vs_conv": float(phase48["mean_full_gain_vs_conv"]),
        "ssc_state_contribution_vs_direct_only": float(
            phase48["mean_state_contribution_vs_direct_only"]
        ),
        "ssc_state_specificity_vs_shuffled": float(
            phase48["mean_state_specificity_vs_shuffled"]
        ),
        "ssc_tcn_accuracy": float(phase49_tcn["mean_test_accuracy"]),
        "ssc_conv1d_final_accuracy": float(phase49_conv["mean_test_accuracy"]),
        "ssc_residual_lif_final_accuracy": float(phase49_lif["mean_test_accuracy"]),
        "ssc_tcn_gain_vs_residual_lif": ssc_tcn_gain,
        "ssc_residual_lif_throughput_ratio_vs_tcn": throughput_ratio,
        "ssc_tcn_test_examples_per_second": float(
            phase49_tcn["mean_test_examples_per_second"]
        ),
        "ssc_residual_lif_test_examples_per_second": float(
            phase49_lif["mean_test_examples_per_second"]
        ),
        "ssc_residual_lif_dense_mac_reduction_vs_tcn": mac_reduction,
        "ssc_residual_lif_spike_rate": float(phase49_lif["mean_activity"]),
        "ssc_residual_lif_accuracy_std": float(phase49_lif["std_test_accuracy"]),
        "milestone_a_tcn_screen_validation_accuracy": milestone_tcn_validation,
        "milestone_a_residual_lif_screen_validation_accuracy": float(
            milestone_screen["residual_lif"]["best_validation_accuracy"]
        ),
        "milestone_a_hierarchical_analog_screen_validation_accuracy": float(
            milestone_screen["hierarchical_residual_analog"][
                "best_validation_accuracy"
            ]
        ),
        "milestone_a_hierarchical_lif_screen_validation_accuracy": float(
            milestone_screen["hierarchical_residual_lif"][
                "best_validation_accuracy"
            ]
        ),
        "milestone_a_tcn_confirmation_accuracy": float(
            milestone_tcn["mean_full_accuracy"]
        ),
        "milestone_a_tcn_confirmation_accuracy_std": float(
            milestone_tcn["std_full_accuracy"]
        ),
        "milestone_a_qualified_arm_count": float(
            milestone_qualified_count
        ),
        "gen6_tcn_accuracy": float(gen6_tcn["mean_full_accuracy"]),
        "gen6_lif_accuracy": float(gen6_lif["mean_full_accuracy"]),
        "gen6_lif_gap_vs_tcn": float(gen6_lif["mean_gain_vs_tcn"]),
        "gen6_lif_state_contribution_vs_direct_only": float(
            gen6_lif["mean_state_contribution_vs_direct_only"]
        ),
        "gen6_lif_state_contribution_replication_count": float(
            gen6_lif["half_point_seed_count_state_contribution"]
        ),
        "gen6_lif_state_specificity_vs_shuffled": float(
            gen6_lif["mean_state_specificity_vs_shuffled"]
        ),
        "gen6_lif_state_specificity_replication_count": float(
            gen6_lif["half_point_seed_count_state_specificity"]
        ),
        "gen6_lif_spike_rate": float(gen6_lif["mean_activity"]),
        "gen6_lif_mean_absolute_gate": float(
            gen6_lif["mean_absolute_gate"]
        ),
        "gen6_lif_throughput_ratio_vs_tcn": (
            float(gen6_lif["mean_test_examples_per_second"])
            / float(gen6_tcn["mean_test_examples_per_second"])
        ),
        "gen6_qualified_arm_count": float(gen6_qualified_count),
        "gen7_tcn_accuracy": float(gen7_tcn["mean_full_accuracy"]),
        "gen7_lif_accuracy": float(gen7_lif["mean_full_accuracy"]),
        "gen7_lif_gain_vs_tcn": float(gen7_lif["mean_gain_vs_tcn"]),
        "gen7_lif_state_contribution_vs_direct_only": float(
            gen7_lif["mean_state_contribution_vs_direct_only"]
        ),
        "gen7_lif_state_specificity_vs_shuffled": float(
            gen7_lif["mean_state_specificity_vs_shuffled"]
        ),
        "gen7_lif_temporal_order_vs_reversed": float(
            gen7_lif["mean_state_temporal_order_vs_reversed"]
        ),
        "gen7_lif_future_alignment_margin": float(
            gen7_lif["mean_future_alignment_margin"]
        ),
        "gen7_alignment_gain_vs_shuffled_training": float(
            gen7_lif["mean_future_alignment_margin"]
            - gen7_shuffled["mean_future_alignment_margin"]
        ),
        "gen7_lif_spike_rate": float(gen7_lif["mean_activity"]),
        "gen7_lif_mean_absolute_gate": float(gen7_lif["mean_absolute_gate"]),
        "gen7_lif_throughput_ratio_vs_tcn": (
            float(gen7_lif["mean_test_examples_per_second"])
            / float(gen7_tcn["mean_test_examples_per_second"])
        ),
        "gen7_qualified_arm_count": float(gen7_qualified_count),
        "gen8_tcn_accuracy": float(gen8_tcn["mean_full_accuracy"]),
        "gen8_pooled_lif_accuracy": float(gen8_pooled["mean_full_accuracy"]),
        "gen8_pooled_lif_gain_vs_tcn": float(gen8_pooled["mean_gain_vs_tcn"]),
        "gen8_pooled_lif_state_specificity_vs_shuffled": float(
            gen8_pooled["mean_state_specificity_vs_shuffled"]
        ),
        "gen8_analog_accuracy": float(gen8_analog["mean_full_accuracy"]),
        "gen8_analog_gain_vs_tcn": float(gen8_analog["mean_gain_vs_tcn"]),
        "gen8_analog_state_contribution_vs_direct_only": float(
            gen8_analog["mean_state_contribution_vs_direct_only"]
        ),
        "gen8_analog_state_specificity_vs_shuffled": float(
            gen8_analog["mean_state_specificity_vs_shuffled"]
        ),
        "gen8_analog_temporal_order_vs_reversed": float(
            gen8_analog["mean_state_temporal_order_vs_reversed"]
        ),
        "gen8_analog_temporal_order_replication_count": float(
            gen8_analog["half_point_seed_count_temporal_order"]
        ),
        "gen8_analog_identity_replication_count": float(
            gen8_analog["half_point_seed_count_state_specificity"]
        ),
        "gen8_candidate_screen_validation_accuracy": float(
            gen8_screen["lif_time_local_binding"]["best_validation_accuracy"]
        ),
        "gen8_candidate_screen_spike_rate": float(
            gen8_screen["lif_time_local_binding"]["checkpoint_activity"]
        ),
        "gen8_shuffled_screen_validation_accuracy": float(
            gen8_screen["lif_shuffled_time_local"]["best_validation_accuracy"]
        ),
        "gen8_shuffled_screen_spike_rate": float(
            gen8_screen["lif_shuffled_time_local"]["checkpoint_activity"]
        ),
        "gen8_qualified_arm_count": float(gen8_qualified_count),
        "gen9_tcn_screen_validation_accuracy": float(
            gen9_screen["dilated_tcn"]["best_validation_accuracy"]
        ),
        "gen9_predictive_lif_screen_validation_accuracy": float(
            gen9_screen["predictive_lif"]["best_validation_accuracy"]
        ),
        "gen9_predictive_lif_screen_gap_vs_tcn": float(
            gen9_screen["predictive_lif"]["best_validation_accuracy"]
            - gen9_screen["dilated_tcn"]["best_validation_accuracy"]
        ),
        "gen9_predictive_lif_screen_spike_rate": float(
            gen9_screen["predictive_lif"]["checkpoint_activity"]
        ),
        "gen9_tcn_static_shift_drop": float(gen9_static["mean_shift_drop"]),
        "gen9_tcn_static_shifted_accuracy": float(
            gen9_static["mean_shifted_final_accuracy"]
        ),
        "gen9_tcn_readout_final_shifted_accuracy": float(
            gen9_readout["mean_shifted_final_accuracy"]
        ),
        "gen9_tcn_readout_adaptation_gain": float(
            gen9_readout["mean_adaptation_gain"]
        ),
        "gen9_tcn_readout_adaptation_auc": float(
            gen9_readout["mean_adaptation_auc"]
        ),
        "gen9_tcn_readout_forgetting": float(gen9_readout["mean_forgetting"]),
        "gen9_tcn_full_final_shifted_accuracy": float(
            gen9_full["mean_shifted_final_accuracy"]
        ),
        "gen9_tcn_full_adaptation_gain": float(gen9_full["mean_adaptation_gain"]),
        "gen9_tcn_full_adaptation_auc": float(gen9_full["mean_adaptation_auc"]),
        "gen9_tcn_full_forgetting": float(gen9_full["mean_forgetting"]),
        "gen9_qualified_arm_count": float(gen9_qualified_count),
        "gen10_tcn_clean_accuracy": float(gen10_tcn["mean_clean_accuracy"]),
        "gen10_tcn_damaged_accuracy": float(gen10_tcn["mean_damaged_accuracy"]),
        "gen10_dropout_clean_accuracy": float(gen10_dropout["mean_clean_accuracy"]),
        "gen10_dropout_damaged_accuracy": float(gen10_dropout["mean_damaged_accuracy"]),
        "gen10_dropout_clean_gain_vs_tcn": float(gen10_dropout["mean_clean_accuracy"] - gen10_tcn["mean_clean_accuracy"]),
        "gen10_dropout_damaged_gain_vs_tcn": float(gen10_dropout["mean_damaged_accuracy"] - gen10_tcn["mean_damaged_accuracy"]),
        "gen10_dropout_damage_drop_improvement": float(gen10_tcn["mean_damage_drop"] - gen10_dropout["mean_damage_drop"]),
        "gen10_analog_screen_clean_gap": float(gen10_screen["masked_residual_analog"]["best_validation_accuracy"] - gen10_screen["dropout_tcn"]["best_validation_accuracy"]),
        "gen10_analog_screen_damaged_gap": float(gen10_screen["masked_residual_analog"]["damaged_validation_accuracy"] - gen10_screen["dropout_tcn"]["damaged_validation_accuracy"]),
        "gen10_lif_screen_clean_gap": float(gen10_screen["masked_residual_lif"]["best_validation_accuracy"] - gen10_screen["dropout_tcn"]["best_validation_accuracy"]),
        "gen10_lif_screen_damaged_gap": float(gen10_screen["masked_residual_lif"]["damaged_validation_accuracy"] - gen10_screen["dropout_tcn"]["damaged_validation_accuracy"]),
        "gen10_lif_screen_damaged_accuracy": float(gen10_screen["masked_residual_lif"]["damaged_validation_accuracy"]),
        "gen10_lif_screen_spike_rate": float(gen10_screen["masked_residual_lif"]["checkpoint_activity"]),
        "gen10_qualified_arm_count": float(gen10_qualified_count),
        "gen11_static_shift_drop": float(gen11_static["mean_shift_drop"]),
        "gen11_readout_adaptation_gain": float(gen11_readout["mean_adaptation_gain"]),
        "gen11_readout_final_shifted_accuracy": float(gen11_readout["mean_shifted_final_accuracy"]),
        "gen11_readout_forgetting": float(gen11_readout["mean_forgetting"]),
        "gen11_full_adaptation_gain": float(gen11_full["mean_adaptation_gain"]),
        "gen11_full_final_shifted_accuracy": float(gen11_full["mean_shifted_final_accuracy"]),
        "gen11_analog_adaptation_gain": float(gen11_analog["mean_adaptation_gain"]),
        "gen11_analog_state_contribution": float(gen11_analog["mean_state_contribution"]),
        "gen11_analog_state_specificity": float(gen11_analog["mean_state_specificity"]),
        "gen11_lif_adaptation_gain": float(gen11_lif["mean_adaptation_gain"]),
        "gen11_lif_final_shifted_accuracy": float(gen11_lif["mean_shifted_final_accuracy"]),
        "gen11_lif_forgetting": float(gen11_lif["mean_forgetting"]),
        "gen11_lif_state_contribution": float(gen11_lif["mean_state_contribution"]),
        "gen11_lif_state_specificity": float(gen11_lif["mean_state_specificity"]),
        "gen11_lif_spike_rate": float(gen11_lif["mean_activity"]),
        "gen11_qualified_arm_count": float(gen11_qualified_count),
        "gen12_static_shift_drop": float(gen12_static["mean_shift_drop"]),
        "gen12_readout_adaptation_gain": float(gen12_readout["mean_adaptation_gain"]),
        "gen12_readout_final_shifted_accuracy": float(gen12_readout["mean_shifted_final_accuracy"]),
        "gen12_readout_forgetting": float(gen12_readout["mean_forgetting"]),
        "gen12_full_adaptation_gain": float(gen12_full["mean_adaptation_gain"]),
        "gen12_full_final_shifted_accuracy": float(gen12_full["mean_shifted_final_accuracy"]),
        "gen12_dense_adaptation_gain": float(gen12_dense["mean_adaptation_gain"]),
        "gen12_dense_association_specificity": float(gen12_dense["mean_association_specificity"]),
        "gen12_spiking_adaptation_gain": float(gen12_spiking["mean_adaptation_gain"]),
        "gen12_spiking_final_shifted_accuracy": float(gen12_spiking["mean_shifted_final_accuracy"]),
        "gen12_spiking_memory_contribution": float(gen12_spiking["mean_memory_contribution"]),
        "gen12_spiking_association_specificity": float(gen12_spiking["mean_association_specificity"]),
        "gen12_spiking_activity": float(gen12_spiking["mean_activity"]),
        "gen12_spiking_active_memory_cells": float(gen12_spiking["mean_active_memory_cells"]),
        "gen12_qualified_arm_count": float(gen12_qualified_count),
        "gen13_static_shift_drop": float(gen13_static["mean_shift_drop"]),
        "gen13_readout_adaptation_gain": float(gen13_readout["mean_adaptation_gain"]),
        "gen13_readout_final_shifted_accuracy": float(gen13_readout["mean_shifted_final_accuracy"]),
        "gen13_readout_forgetting": float(gen13_readout["mean_forgetting"]),
        "gen13_full_adaptation_gain": float(gen13_full["mean_adaptation_gain"]),
        "gen13_full_final_shifted_accuracy": float(gen13_full["mean_shifted_final_accuracy"]),
        "gen13_analog_adaptation_gain": float(gen13_analog["mean_adaptation_gain"]),
        "gen13_analog_fast_weight_contribution": float(gen13_analog["mean_fast_weight_contribution"]),
        "gen13_analog_class_specificity": float(gen13_analog["mean_class_specificity"]),
        "gen13_spiking_adaptation_gain": float(gen13_spiking["mean_adaptation_gain"]),
        "gen13_spiking_final_shifted_accuracy": float(gen13_spiking["mean_shifted_final_accuracy"]),
        "gen13_spiking_forgetting": float(gen13_spiking["mean_forgetting"]),
        "gen13_spiking_fast_weight_contribution": float(gen13_spiking["mean_fast_weight_contribution"]),
        "gen13_spiking_class_specificity": float(gen13_spiking["mean_class_specificity"]),
        "gen13_spiking_activity": float(gen13_spiking["mean_activity"]),
        "gen13_spiking_active_fast_synapses": float(gen13_spiking["mean_active_fast_synapses"]),
        "gen13_spiking_mean_absolute_fast_weight": float(gen13_spiking["mean_absolute_fast_weight"]),
        "gen13_qualified_arm_count": float(gen13_qualified_count),
        "gen14_static_baseline_fitness": float(gen14_static["mean_baseline_net_fitness_per_1000_steps"]),
        "gen14_static_final_fitness": float(gen14_static["mean_final_net_fitness_per_1000_steps"]),
        "gen14_static_phase_gain": float(gen14_static["mean_fitness_gain_per_1000_steps"]),
        "gen14_oracle_final_fitness": float(gen14_oracle["mean_final_net_fitness_per_1000_steps"]),
        "gen14_analog_final_fitness": float(gen14_analog["mean_final_net_fitness_per_1000_steps"]),
        "gen14_analog_phase_gain": float(gen14_analog["mean_fitness_gain_per_1000_steps"]),
        "gen14_spiking_final_fitness": float(gen14_spiking["mean_final_net_fitness_per_1000_steps"]),
        "gen14_spiking_phase_gain": float(gen14_spiking["mean_fitness_gain_per_1000_steps"]),
        "gen14_spiking_activity": float(gen14_spiking["mean_spike_density"]),
        "gen14_spiking_mean_absolute_fast_weight": float(gen14_spiking["mean_absolute_fast_weight"]),
        "gen14_spiking_fast_weight_saturation": float(gen14_spiking["mean_fast_weight_saturation"]),
        "gen14_shuffled_final_fitness": float(gen14_shuffled["mean_final_net_fitness_per_1000_steps"]),
        "gen14_shuffled_phase_gain": float(gen14_shuffled["mean_fitness_gain_per_1000_steps"]),
        "gen14_spiking_margin_vs_static": float(gen14["decision"]["spiking_margin_vs_static_per_1000_steps"]),
        "gen14_spiking_margin_vs_shuffled": float(gen14["decision"]["spiking_margin_vs_shuffled_per_1000_steps"]),
        "gen14_passed": 1.0 if gen14["decision"]["status"] == "pass" else 0.0,
        "gen15_static_final_fitness": float(gen15_static["mean_final_fitness_per_1000_steps"]),
        "gen15_oracle_final_fitness": float(gen15_oracle["mean_final_fitness_per_1000_steps"]),
        "gen15_reinforce_final_fitness": float(gen15_reinforce["mean_final_fitness_per_1000_steps"]),
        "gen15_reinforce_gain": float(gen15_reinforce["mean_fitness_gain_per_1000_steps"]),
        "gen15_reinforce_positive_gain_seed_count": float(gen15_reinforce["positive_gain_seed_count"]),
        "gen15_shuffled_final_fitness": float(gen15_shuffled["mean_final_fitness_per_1000_steps"]),
        "gen15_reinforce_margin_vs_static": float(gen15["decision"]["reinforce_margin_vs_static_per_1000_steps"]),
        "gen15_reinforce_margin_vs_shuffled": float(gen15["decision"]["reinforce_margin_vs_shuffled_per_1000_steps"]),
        "gen15_passed": 1.0 if gen15["decision"]["status"] == "pass" else 0.0,
        "gen16_static_final_fitness": float(gen16_static["mean_final_fitness_per_1000_steps"]),
        "gen16_oracle_final_fitness": float(gen16_oracle["mean_final_fitness_per_1000_steps"]),
        "gen16_autograd_final_fitness": float(gen16_autograd["mean_final_fitness_per_1000_steps"]),
        "gen16_local_final_fitness": float(gen16_local["mean_final_fitness_per_1000_steps"]),
        "gen16_local_gain": float(gen16_local["mean_fitness_gain_per_1000_steps"]),
        "gen16_local_positive_gain_seed_count": float(gen16_local["positive_gain_seed_count"]),
        "gen16_shuffled_final_fitness": float(gen16_shuffled["mean_final_fitness_per_1000_steps"]),
        "gen16_local_autograd_gap": float(gen16["decision"]["local_autograd_final_gap_per_1000_steps"]),
        "gen16_maximum_gradient_error": float(gen16["decision"]["maximum_manual_gradient_error"]),
        "gen16_local_margin_vs_static": float(gen16["decision"]["local_margin_vs_static_per_1000_steps"]),
        "gen16_local_margin_vs_shuffled": float(gen16["decision"]["local_margin_vs_shuffled_per_1000_steps"]),
        "gen16_reward_identity_seed_count": float(gen16["decision"]["reward_identity_seed_count"]),
        "gen16_passed": 1.0 if gen16["decision"]["status"] == "pass" else 0.0,
        "gen17_static_final_fitness": float(gen17_static["mean_final_fitness_per_1000_steps"]),
        "gen17_oracle_final_fitness": float(gen17_oracle["mean_final_fitness_per_1000_steps"]),
        "gen17_analog_final_fitness": float(gen17_analog["mean_final_fitness_per_1000_steps"]),
        "gen17_analog_gain": float(gen17_analog["mean_fitness_gain_per_1000_steps"]),
        "gen17_spiking_final_fitness": float(gen17_spiking["mean_final_fitness_per_1000_steps"]),
        "gen17_spiking_gain": float(gen17_spiking["mean_fitness_gain_per_1000_steps"]),
        "gen17_shuffled_final_fitness": float(gen17_shuffled["mean_final_fitness_per_1000_steps"]),
        "gen17_training_spike_density": float(gen17_spiking["mean_training_spike_density"]),
        "gen17_evaluation_spike_density": float(gen17_spiking["mean_evaluation_spike_density"]),
        "gen17_maximum_gradient_error": float(gen17["decision"]["maximum_manual_gradient_error"]),
        "gen17_analog_minus_spiking_gain": float(gen17["decision"]["analog_minus_spiking_gain_per_1000_steps"]),
        "gen17_spiking_margin_vs_static": float(gen17["decision"]["spiking_margin_vs_static_per_1000_steps"]),
        "gen17_spiking_margin_vs_shuffled": float(gen17["decision"]["spiking_margin_vs_shuffled_per_1000_steps"]),
        "gen17_reward_identity_seed_count": float(gen17["decision"]["reward_identity_seed_count"]),
        "gen17_passed": 1.0 if gen17["decision"]["status"] == "pass" else 0.0,
        "gen18_static_final_fitness": float(gen18_static["mean_final_fitness_per_1000_steps"]),
        "gen18_oracle_final_fitness": float(gen18_oracle["mean_final_fitness_per_1000_steps"]),
        "gen18_local_final_fitness": float(gen18_local["mean_final_fitness_per_1000_steps"]),
        "gen18_local_gain": float(gen18["decision"]["local_gain_mean_per_1000_steps"]),
        "gen18_local_gain_ci95_lower": float(gen18["decision"]["local_gain_ci95_lower_per_1000_steps"]),
        "gen18_shuffled_final_fitness": float(gen18_shuffled["mean_final_fitness_per_1000_steps"]),
        "gen18_local_margin_vs_static": float(gen18["decision"]["local_margin_vs_static_mean_per_1000_steps"]),
        "gen18_local_margin_vs_static_ci95_lower": float(gen18["decision"]["local_margin_vs_static_ci95_lower_per_1000_steps"]),
        "gen18_local_margin_vs_shuffled": float(gen18["decision"]["local_margin_vs_shuffled_mean_per_1000_steps"]),
        "gen18_local_margin_vs_shuffled_ci95_lower": float(gen18["decision"]["local_margin_vs_shuffled_ci95_lower_per_1000_steps"]),
        "gen18_qualified_gain_seed_count": float(gen18["decision"]["qualified_gain_seed_count"]),
        "gen18_reward_identity_seed_count": float(gen18["decision"]["qualified_reward_identity_seed_count"]),
        "gen18_maximum_gradient_error": float(gen18["decision"]["maximum_manual_gradient_error"]),
        "gen18_passed": 1.0 if gen18["decision"]["status"] == "pass" else 0.0,
        "gen19_conv_accuracy": float(gen19["summary"]["mean_conv_accuracy"]),
        "gen19_residual_lif_accuracy": float(gen19["summary"]["mean_full_accuracy"]),
        "gen19_residual_lif_gain_vs_conv": float(gen19["summary"]["mean_gain_vs_conv"]),
        "gen19_state_contribution_vs_direct_only": float(
            gen19["summary"]["mean_state_contribution_vs_direct_only"]
        ),
        "gen19_state_contribution_seed_count": float(
            gen19["summary"]["state_contribution_seed_count"]
        ),
        "gen19_state_specificity_vs_shuffled": float(
            gen19["summary"]["mean_state_specificity_vs_shuffled"]
        ),
        "gen19_state_specificity_seed_count": float(
            gen19["summary"]["state_specificity_seed_count"]
        ),
        "gen19_spike_activity": float(gen19["summary"]["mean_spike_activity"]),
        "gen19_residual_throughput_ratio_vs_conv": float(
            gen19["summary"]["mean_residual_test_examples_per_second"]
            / gen19["summary"]["mean_conv_test_examples_per_second"]
        ),
        "gen19_passed": 1.0 if gen19["decision"]["status"] == "pass" else 0.0,
        "gen20_teacher_screen_accuracy": float(
            gen20_teacher["best_validation_accuracy"]
        ),
        "gen20_conv_plif_screen_accuracy": float(
            gen20_conv["best_validation_accuracy"]
        ),
        "gen20_multiscale_screen_accuracy": float(
            gen20_multiscale["best_validation_accuracy"]
        ),
        "gen20_distilled_screen_accuracy": float(
            gen20_distilled["best_validation_accuracy"]
        ),
        "gen20_screen_accuracy_gate": float(
            gen20["config"]["minimum_screen_accuracy"]
        ),
        "gen20_multiscale_gap_to_gate": float(
            gen20_multiscale["best_validation_accuracy"]
            - gen20["config"]["minimum_screen_accuracy"]
        ),
        "gen20_multiscale_gain_vs_conv_plif": float(
            gen20_multiscale["best_validation_accuracy"]
            - gen20_conv["best_validation_accuracy"]
        ),
        "gen20_distillation_gain_vs_multiscale": float(
            gen20_distilled["best_validation_accuracy"]
            - gen20_multiscale["best_validation_accuracy"]
        ),
        "gen20_multiscale_activity": float(
            gen20_multiscale["validation_activity"]
        ),
        "gen20_distilled_activity": float(
            gen20_distilled["validation_activity"]
        ),
        "gen20_multiscale_ops_reduction_vs_teacher": (
            gen20_teacher_ops / gen20_multiscale_proxy
        ),
        "gen20_distilled_ops_reduction_vs_teacher": (
            gen20_teacher_ops / gen20_distilled_proxy
        ),
        "gen20_promoted_arm_count": float(len(gen20["promoted_arms"])),
        "gen20_passed": 1.0 if gen20["decision"]["status"] == "pass" else 0.0,
    }
    claims = [
        _claim(
            "Standalone state-only LIF is competitive on SHD",
            False,
            f"State-only LIF trails Conv1D by {-100.0 * shd_state_only_gap:.3f} points.",
            "rejected",
        ),
        _claim(
            "Residual LIF is viable on SHD",
            shd_residual_gain >= -0.02,
            f"Residual LIF changes accuracy by {100.0 * shd_residual_gain:+.3f} points versus Conv1D.",
            "supported" if shd_residual_gain >= -0.02 else "rejected",
        ),
        _claim(
            "Sample-specific LIF state contributes on SHD",
            (
                metrics["shd_state_contribution_vs_direct_only"] >= 0.01
                and metrics["shd_state_specificity_vs_shuffled"] >= 0.01
            ),
            "Removing state costs "
            f"{100.0 * metrics['shd_state_contribution_vs_direct_only']:.3f} points; "
            "shuffling state costs "
            f"{100.0 * metrics['shd_state_specificity_vs_shuffled']:.3f} points.",
            "supported",
        ),
        _claim(
            "The residual-state contribution replicates on SSC",
            (
                metrics["ssc_state_contribution_vs_direct_only"] >= 0.01
                and metrics["ssc_state_specificity_vs_shuffled"] >= 0.01
            ),
            "Removing state costs "
            f"{100.0 * metrics['ssc_state_contribution_vs_direct_only']:.3f} points; "
            "shuffling state costs "
            f"{100.0 * metrics['ssc_state_specificity_vs_shuffled']:.3f} points.",
            "supported",
        ),
        _claim(
            "Residual LIF matches the stronger SSC temporal baseline",
            ssc_tcn_gain <= 0.02,
            f"Matched dilated TCN leads by {100.0 * ssc_tcn_gain:.3f} points.",
            "supported" if ssc_tcn_gain <= 0.02 else "rejected",
        ),
        _claim(
            "Residual LIF is faster in the current T4 implementation",
            throughput_ratio >= 1.0,
            f"Residual LIF delivers {throughput_ratio:.3f}x TCN throughput.",
            "supported" if throughput_ratio >= 1.0 else "rejected",
        ),
        _claim(
            "Residual LIF has a lower dense-operation proxy",
            mac_reduction > 0.0,
            f"Dense MAC proxy is {100.0 * mac_reduction:.3f}% lower than TCN.",
            "proxy_only",
        ),
        _claim(
            "The current results establish hardware energy efficiency",
            False,
            "No direct power or energy measurement was performed; dense PyTorch is not event-driven.",
            "not_tested",
        ),
        _claim(
            "Hierarchical residual scaling closes the SSC accuracy gap",
            milestone_scaling_passed,
            "Hierarchical analog and LIF trail TCN validation by "
            f"{100.0 * milestone_hierarchical_gaps[0]:.3f} "
            "and "
            f"{100.0 * milestone_hierarchical_gaps[1]:.3f} points.",
            "supported" if milestone_scaling_passed else "rejected",
        ),
        _claim(
            "The current Gen-5 architecture qualifies for hardware optimization",
            milestone_a["decision"]["status"] == "pass",
            "Milestone A promoted "
            f"{', '.join(milestone_a['promoted_arms'])} and returned "
            f"status={milestone_a['decision']['status']} with "
            f"{milestone_qualified_count} qualified causal arms.",
            (
                "supported"
                if milestone_a["decision"]["status"] == "pass"
                else "rejected"
            ),
        ),
        _claim(
            "The Gen-6 shared residual LIF preserves TCN predictive accuracy",
            float(gen6_lif["mean_gain_vs_tcn"]) >= -0.01,
            "Shared residual LIF changes SSC accuracy by "
            f"{100.0 * float(gen6_lif['mean_gain_vs_tcn']):+.3f} points "
            "versus the matched TCN.",
            (
                "supported"
                if float(gen6_lif["mean_gain_vs_tcn"]) >= -0.01
                else "rejected"
            ),
        ),
        _claim(
            "The Gen-6 LIF correction is beneficially sample-specific",
            (
                float(gen6_lif["mean_state_contribution_vs_direct_only"])
                >= 0.005
                and float(gen6_lif["mean_state_specificity_vs_shuffled"])
                >= 0.005
                and int(gen6_lif["half_point_seed_count_state_contribution"])
                >= 2
                and int(gen6_lif["half_point_seed_count_state_specificity"])
                >= 2
            ),
            "Removing state costs "
            f"{100.0 * float(gen6_lif['mean_state_contribution_vs_direct_only']):.3f} points "
            f"({int(gen6_lif['half_point_seed_count_state_contribution'])}/3 seeds pass), "
            "while shuffling state changes accuracy by "
            f"{-100.0 * float(gen6_lif['mean_state_specificity_vs_shuffled']):+.3f} points "
            f"in the shuffled model's favor ({int(gen6_lif['half_point_seed_count_state_specificity'])}/3 seeds pass).",
            "rejected",
        ),
        _claim(
            "The Gen-6 successor qualifies for hardware optimization",
            gen6["decision"]["status"] == "pass",
            f"The terminal decision is status={gen6['decision']['status']} with "
            f"{gen6_qualified_count} qualified arms.",
            "supported" if gen6["decision"]["status"] == "pass" else "rejected",
        ),
        _claim(
            "Gen-7 paired future prediction improves state alignment",
            (
                float(gen7_lif["mean_future_alignment_margin"]) >= 0.02
                and int(gen7_lif["alignment_seed_count"]) >= 2
                and float(gen7_lif["mean_future_alignment_margin"])
                - float(gen7_shuffled["mean_future_alignment_margin"])
                >= 0.01
            ),
            "Paired LIF future alignment is "
            f"{float(gen7_lif['mean_future_alignment_margin']):.4f} versus "
            f"{float(gen7_shuffled['mean_future_alignment_margin']):.4f} "
            "under shuffled-target training.",
            "supported",
        ),
        _claim(
            "Gen-7 paired predictive LIF matches TCN accuracy",
            float(gen7_lif["mean_gain_vs_tcn"]) >= -0.01,
            "Paired predictive LIF changes accuracy by "
            f"{100.0 * float(gen7_lif['mean_gain_vs_tcn']):+.3f} points "
            "versus TCN.",
            (
                "supported"
                if float(gen7_lif["mean_gain_vs_tcn"]) >= -0.01
                else "rejected"
            ),
        ),
        _claim(
            "Gen-7 uses predictive state beneficially by sample identity and temporal order",
            (
                float(gen7_lif["mean_state_specificity_vs_shuffled"]) >= 0.005
                and int(gen7_lif["half_point_seed_count_state_specificity"]) >= 2
                and float(gen7_lif["mean_state_temporal_order_vs_reversed"]) >= 0.005
                and int(gen7_lif["half_point_seed_count_temporal_order"]) >= 2
            ),
            "Shuffling state changes accuracy by "
            f"{-100.0 * float(gen7_lif['mean_state_specificity_vs_shuffled']):+.3f} "
            "points in the shuffled model's favor; reversing state costs only "
            f"{100.0 * float(gen7_lif['mean_state_temporal_order_vs_reversed']):.3f} points.",
            "rejected",
        ),
        _claim(
            "The Gen-7 successor qualifies for hardware optimization",
            gen7["decision"]["status"] == "pass",
            f"The terminal decision is status={gen7['decision']['status']} with "
            f"{gen7_qualified_count} qualified arms.",
            "supported" if gen7["decision"]["status"] == "pass" else "rejected",
        ),
        _claim(
            "Gen-8 time-local analog binding introduces temporal-order sensitivity",
            (
                float(gen8_analog["mean_state_temporal_order_vs_reversed"])
                >= 0.005
                and int(gen8_analog["half_point_seed_count_temporal_order"])
                >= 2
            ),
            "Reversing analog state costs "
            f"{100.0 * float(gen8_analog['mean_state_temporal_order_vs_reversed']):.3f} points "
            f"with {int(gen8_analog['half_point_seed_count_temporal_order'])}/3 seeds passing.",
            "supported",
        ),
        _claim(
            "Gen-8 time-local analog binding uses the correct sample identity",
            (
                float(gen8_analog["mean_state_specificity_vs_shuffled"]) >= 0.005
                and int(gen8_analog["half_point_seed_count_state_specificity"])
                >= 2
            ),
            "Shuffling analog state costs only "
            f"{100.0 * float(gen8_analog['mean_state_specificity_vs_shuffled']):.3f} points "
            f"with {int(gen8_analog['half_point_seed_count_state_specificity'])}/3 seeds passing.",
            "rejected",
        ),
        _claim(
            "The Gen-8 paired time-local LIF candidate is stable enough for confirmation",
            "lif_time_local_binding" in gen8["promoted_arms"],
            "The candidate screened at "
            f"{100.0 * float(gen8_screen['lif_time_local_binding']['best_validation_accuracy']):.3f}% "
            "validation accuracy with a "
            f"{100.0 * float(gen8_screen['lif_time_local_binding']['checkpoint_activity']):.3f}% spike rate.",
            "rejected",
        ),
        _claim(
            "The Gen-8 successor qualifies for hardware optimization",
            gen8["decision"]["status"] == "pass",
            f"The terminal decision is status={gen8['decision']['status']} with "
            f"{gen8_qualified_count} qualified arms.",
            "supported" if gen8["decision"]["status"] == "pass" else "rejected",
        ),
        _claim(
            "Gen-9 sensor damage creates a non-trivial distribution shift",
            float(gen9_static["mean_shift_drop"]) >= 0.05,
            "Static TCN accuracy falls by "
            f"{100.0 * float(gen9_static['mean_shift_drop']):.3f} points across confirmation seeds.",
            "supported" if float(gen9_static["mean_shift_drop"]) >= 0.05 else "rejected",
        ),
        _claim(
            "The Gen-9 predictive LIF representation is source-competent",
            "predictive_lif" in gen9["promoted_source_arms"],
            "Predictive LIF trails TCN screening validation by "
            f"{-100.0 * metrics['gen9_predictive_lif_screen_gap_vs_tcn']:.3f} points "
            f"with {100.0 * metrics['gen9_predictive_lif_screen_spike_rate']:.3f}% spike activity.",
            "supported" if "predictive_lif" in gen9["promoted_source_arms"] else "rejected",
        ),
        _claim(
            "A frozen TCN representation adapts through a trainable readout",
            (
                float(gen9_readout["mean_adaptation_gain"]) >= 0.02
                and int(gen9_readout["two_point_gain_seed_count"]) >= 2
            ),
            "Readout adaptation gains "
            f"{100.0 * float(gen9_readout['mean_adaptation_gain']):.3f} points "
            f"with {int(gen9_readout['two_point_gain_seed_count'])}/3 seeds passing, "
            f"at {100.0 * float(gen9_readout['mean_forgetting']):.3f} points forgetting.",
            "supported",
        ),
        _claim(
            "Gen-9 qualifies for STW/LTW memory experiments",
            gen9["decision"]["status"] == "pass",
            f"The terminal decision is status={gen9['decision']['status']} with "
            f"{gen9_qualified_count} qualified arms; only {', '.join(gen9['promoted_source_arms'])} passed source screening.",
            "supported" if gen9["decision"]["status"] == "pass" else "rejected",
        ),
        _claim(
            "Sensor dropout improves conventional robustness in Gen-10",
            metrics["gen10_dropout_damaged_gain_vs_tcn"] >= 0.02,
            "Dropout TCN changes clean accuracy by "
            f"{100.0 * metrics['gen10_dropout_clean_gain_vs_tcn']:+.3f} points and damaged accuracy by "
            f"{100.0 * metrics['gen10_dropout_damaged_gain_vs_tcn']:+.3f} points.",
            "supported",
        ),
        _claim(
            "The Gen-10 masked residual analog representation is source-competent",
            "masked_residual_analog" in gen10["promoted_arms"],
            "Residual analog trails dropout TCN screening by "
            f"{-100.0 * metrics['gen10_analog_screen_clean_gap']:.3f} clean and "
            f"{-100.0 * metrics['gen10_analog_screen_damaged_gap']:.3f} damaged points.",
            "supported" if "masked_residual_analog" in gen10["promoted_arms"] else "rejected",
        ),
        _claim(
            "The Gen-10 masked residual LIF representation is source-competent",
            "masked_residual_lif" in gen10["promoted_arms"],
            "Residual LIF trails dropout TCN screening by "
            f"{-100.0 * metrics['gen10_lif_screen_clean_gap']:.3f} clean and "
            f"{-100.0 * metrics['gen10_lif_screen_damaged_gap']:.3f} damaged points with "
            f"{100.0 * metrics['gen10_lif_screen_spike_rate']:.3f}% spikes.",
            "supported" if "masked_residual_lif" in gen10["promoted_arms"] else "rejected",
        ),
        _claim(
            "Gen-10 qualifies a spiking representation for adaptation",
            gen10["decision"]["status"] == "pass",
            f"The terminal decision is status={gen10['decision']['status']} with "
            f"{gen10_qualified_count} qualified arms.",
            "supported" if gen10["decision"]["status"] == "pass" else "rejected",
        ),
        _claim(
            "Gen-11 state adapters improve damaged-task accuracy by the preregistered margin",
            (
                float(gen11_lif["mean_adaptation_gain"]) >= 0.02
                and int(gen11_lif["two_point_gain_seed_count"]) >= 2
            ),
            "The analog and LIF adapters gain "
            f"{100.0 * float(gen11_analog['mean_adaptation_gain']):.3f} and "
            f"{100.0 * float(gen11_lif['mean_adaptation_gain']):.3f} points, versus "
            f"{100.0 * float(gen11_readout['mean_adaptation_gain']):.3f} for readout adaptation.",
            "supported" if float(gen11_lif["mean_adaptation_gain"]) >= 0.02 else "rejected",
        ),
        _claim(
            "Gen-11 LIF adaptation depends on sample-specific spiking state",
            (
                float(gen11_lif["mean_state_contribution"]) >= 0.005
                and int(gen11_lif["state_contribution_seed_count"]) >= 2
                and float(gen11_lif["mean_state_specificity"]) >= 0.005
                and int(gen11_lif["state_specificity_seed_count"]) >= 2
            ),
            "Removing LIF state costs "
            f"{100.0 * float(gen11_lif['mean_state_contribution']):.3f} points, but shuffling sample identity costs only "
            f"{100.0 * float(gen11_lif['mean_state_specificity']):.3f} points.",
            "supported" if float(gen11_lif["mean_state_specificity"]) >= 0.005 else "rejected",
        ),
        _claim(
            "Gen-11 qualifies for synaptic STW/LTW consolidation",
            gen11["decision"]["status"] == "pass",
            f"The terminal decision is status={gen11['decision']['status']} with "
            f"{gen11_qualified_count} qualified arms.",
            "supported" if gen11["decision"]["status"] == "pass" else "rejected",
        ),
        _claim(
            "Gen-12 prototype memory provides useful fast adaptation",
            (
                float(gen12_spiking["mean_adaptation_gain"]) >= 0.02
                and int(gen12_spiking["two_point_gain_seed_count"]) >= 2
            ),
            "Dense and spiking prototypes gain "
            f"{100.0 * float(gen12_dense['mean_adaptation_gain']):.3f} and "
            f"{100.0 * float(gen12_spiking['mean_adaptation_gain']):.3f} points, versus "
            f"{100.0 * float(gen12_readout['mean_adaptation_gain']):.3f} for readout adaptation.",
            "supported" if float(gen12_spiking["mean_adaptation_gain"]) >= 0.02 else "rejected",
        ),
        _claim(
            "Gen-12 spiking memory depends on correct class associations",
            (
                float(gen12_spiking["mean_memory_contribution"]) >= 0.005
                and int(gen12_spiking["memory_contribution_seed_count"]) >= 2
                and float(gen12_spiking["mean_association_specificity"]) >= 0.005
                and int(gen12_spiking["association_specificity_seed_count"]) >= 2
            ),
            "Removing memory costs "
            f"{100.0 * float(gen12_spiking['mean_memory_contribution']):.3f} points and shuffling class associations costs "
            f"{100.0 * float(gen12_spiking['mean_association_specificity']):.3f} points, with "
            f"{100.0 * float(gen12_spiking['mean_activity']):.3f}% event density.",
            "supported" if float(gen12_spiking["mean_association_specificity"]) >= 0.005 else "rejected",
        ),
        _claim(
            "Gen-12 qualifies for context-free consolidation",
            gen12["decision"]["status"] == "pass",
            f"The terminal decision is status={gen12['decision']['status']} with "
            f"{gen12_qualified_count} qualified arms.",
            "supported" if gen12["decision"]["status"] == "pass" else "rejected",
        ),
        _claim(
            "Gen-13 local output plasticity provides useful adaptation",
            (
                float(gen13_spiking["mean_adaptation_gain"]) >= 0.02
                and int(gen13_spiking["two_point_gain_seed_count"]) >= 2
            ),
            "Analog and spiking local rules gain "
            f"{100.0 * float(gen13_analog['mean_adaptation_gain']):.3f} and "
            f"{100.0 * float(gen13_spiking['mean_adaptation_gain']):.3f} points, versus "
            f"{100.0 * float(gen13_readout['mean_adaptation_gain']):.3f} for autograd readout adaptation.",
            "supported" if float(gen13_spiking["mean_adaptation_gain"]) >= 0.02 else "rejected",
        ),
        _claim(
            "Gen-13 spiking fast weights are causally class-specific",
            (
                float(gen13_spiking["mean_fast_weight_contribution"]) >= 0.005
                and int(gen13_spiking["fast_weight_contribution_seed_count"]) >= 2
                and float(gen13_spiking["mean_class_specificity"]) >= 0.005
                and int(gen13_spiking["class_specificity_seed_count"]) >= 2
            ),
            "Removing spiking fast weights costs "
            f"{100.0 * float(gen13_spiking['mean_fast_weight_contribution']):.3f} points and shuffling output classes costs "
            f"{100.0 * float(gen13_spiking['mean_class_specificity']):.3f} points, with "
            f"{100.0 * float(gen13_spiking['mean_activity']):.3f}% trace density.",
            "supported" if (
                float(gen13_spiking["mean_fast_weight_contribution"]) >= 0.005
                and float(gen13_spiking["mean_class_specificity"]) >= 0.005
            ) else "rejected",
        ),
        _claim(
            "Gen-13 qualifies for STW/LTW consolidation",
            gen13["decision"]["status"] == "pass",
            f"The terminal decision is status={gen13['decision']['status']} with "
            f"{gen13_qualified_count} qualified arms.",
            "supported" if gen13["decision"]["status"] == "pass" else "rejected",
        ),
        _claim(
            "Gen-14 embodied sensor-to-action mapping is solvable",
            float(gen14_oracle["mean_final_net_fitness_per_1000_steps"])
            > float(gen14_static["mean_final_net_fitness_per_1000_steps"]),
            "The oracle reaches "
            f"{float(gen14_oracle['mean_final_net_fitness_per_1000_steps']):.3f} versus "
            f"{float(gen14_static['mean_final_net_fitness_per_1000_steps']):.3f} static net fitness per 1,000 steps.",
            "supported",
        ),
        _claim(
            "Gen-14 baseline-to-evaluation improvement identifies local learning",
            False,
            "Spiking eligibility rises by "
            f"{float(gen14_spiking['mean_fitness_gain_per_1000_steps']):.3f}, but the unchanged static arm rises by "
            f"{float(gen14_static['mean_fitness_gain_per_1000_steps']):.3f}; the phase comparison is non-stationary.",
            "rejected",
        ),
        _claim(
            "Gen-14 spiking eligibility depends on correctly assigned reward",
            bool(gen14["decision"]["spiking_specificity_gate"]),
            "Correctly rewarded spiking eligibility finishes "
            f"{float(gen14['decision']['spiking_margin_vs_static_per_1000_steps']):+.3f} versus static and "
            f"{float(gen14['decision']['spiking_margin_vs_shuffled_per_1000_steps']):+.3f} versus shuffled reward.",
            "supported" if bool(gen14["decision"]["spiking_specificity_gate"]) else "rejected",
        ),
        _claim(
            "Gen-14 qualifies for reward-eligibility confirmation",
            gen14["decision"]["status"] == "pass",
            f"The terminal decision is status={gen14['decision']['status']}; "
            f"next_milestone={gen14['decision']['next_milestone']}.",
            "supported" if gen14["decision"]["status"] == "pass" else "rejected",
        ),
        _claim(
            "Gen-15 identical-reset evaluation removes phase non-stationarity",
            bool(gen15["decision"]["identical_reset_gate"]),
            "The unchanged static policy has exactly "
            f"{float(gen15_static['mean_fitness_gain_per_1000_steps']):+.3f} fitness gain under replayed seeded evaluation.",
            "supported" if bool(gen15["decision"]["identical_reset_gate"]) else "rejected",
        ),
        _claim(
            "Gen-15 delayed scalar reward supports conventional learning",
            bool(gen15["decision"]["reinforce_gain_gate"]),
            "Correct-reward REINFORCE gains "
            f"{float(gen15_reinforce['mean_fitness_gain_per_1000_steps']):+.3f} fitness per 1,000 steps on "
            f"{int(gen15_reinforce['positive_gain_seed_count'])}/3 positive-gain seeds.",
            "supported" if bool(gen15["decision"]["reinforce_gain_gate"]) else "rejected",
        ),
        _claim(
            "Gen-15 conventional learning depends on agent-specific reward",
            bool(gen15["decision"]["reward_identity_gate"]),
            "Correct reward finishes "
            f"{float(gen15['decision']['reinforce_margin_vs_static_per_1000_steps']):+.3f} versus static and "
            f"{float(gen15['decision']['reinforce_margin_vs_shuffled_per_1000_steps']):+.3f} versus shuffled reward.",
            "supported" if bool(gen15["decision"]["reward_identity_gate"]) else "rejected",
        ),
        _claim(
            "Gen-15 validates an AMMC local-learning mechanism",
            False,
            "Gen-15 tests only a conventional autograd REINFORCE baseline; final mean fitness remains "
            f"{float(gen15_reinforce['mean_final_fitness_per_1000_steps']):+.3f} and no local AMMC rule is present.",
            "not tested",
        ),
        _claim(
            "Gen-16 manual score-function gradient matches autograd",
            bool(gen16["decision"]["manual_gradient_parity_gate"]),
            "Maximum analytic gradient error is "
            f"{float(gen16['decision']['maximum_manual_gradient_error']):.3e}.",
            "supported" if bool(gen16["decision"]["manual_gradient_parity_gate"]) else "rejected",
        ),
        _claim(
            "Gen-16 local reward credit is behaviorally equivalent to autograd",
            bool(gen16["decision"]["autograd_equivalence_gate"]),
            "Manual and autograd policies finish with a fitness gap of "
            f"{float(gen16['decision']['local_autograd_final_gap_per_1000_steps']):.3f} per 1,000 steps.",
            "supported" if bool(gen16["decision"]["autograd_equivalence_gate"]) else "rejected",
        ),
        _claim(
            "Gen-16 local learning depends on agent-specific reward",
            bool(gen16["decision"]["reward_identity_gate"]),
            "The local rule finishes "
            f"{float(gen16['decision']['local_margin_vs_static_per_1000_steps']):+.3f} versus static and "
            f"{float(gen16['decision']['local_margin_vs_shuffled_per_1000_steps']):+.3f} versus shuffled reward on "
            f"{int(gen16['decision']['reward_identity_seed_count'])}/3 identity-qualified seeds.",
            "supported" if bool(gen16["decision"]["reward_identity_gate"]) else "rejected",
        ),
        _claim(
            "Gen-16 establishes sparse-spiking or structural continuous learning",
            False,
            "Gen-16 uses a dense linear analog policy; spikes, STW/LTW, replay, and topology changes are absent.",
            "not tested",
        ),
        _claim(
            "Gen-17 sparse event generation and local gradient are operational",
            bool(gen17["decision"]["spike_activity_gate"])
            and bool(gen17["decision"]["manual_gradient_parity_gate"]),
            "Training/evaluation spike density is "
            f"{100.0 * float(gen17_spiking['mean_training_spike_density']):.3f}%/"
            f"{100.0 * float(gen17_spiking['mean_evaluation_spike_density']):.3f}% and maximum gradient error is "
            f"{float(gen17['decision']['maximum_manual_gradient_error']):.3e}.",
            "supported" if (
                bool(gen17["decision"]["spike_activity_gate"])
                and bool(gen17["decision"]["manual_gradient_parity_gate"])
            ) else "rejected",
        ),
        _claim(
            "Gen-16 analog local-credit gain replicates on Gen-17 seeds",
            bool(gen17["decision"]["analog_reference_gate"]),
            "The analog reference gains "
            f"{float(gen17_analog['mean_fitness_gain_per_1000_steps']):+.3f} with "
            f"{int(gen17['decision']['analog_qualified_gain_seed_count'])}/3 qualified seeds.",
            "supported" if bool(gen17["decision"]["analog_reference_gate"]) else "rejected",
        ),
        _claim(
            "Gen-17 Bernoulli sparse translation preserves local learning",
            bool(gen17["decision"]["spiking_translation_gate"]),
            "The correct-reward spiking policy gains "
            f"{float(gen17_spiking['mean_fitness_gain_per_1000_steps']):+.3f} and trails the analog gain by "
            f"{float(gen17['decision']['analog_minus_spiking_gain_per_1000_steps']):.3f} fitness per 1,000 steps.",
            "supported" if bool(gen17["decision"]["spiking_translation_gate"]) else "rejected",
        ),
        _claim(
            "Gen-17 sparse local learning depends on correctly assigned reward",
            bool(gen17["decision"]["reward_identity_gate"]),
            "Correct reward finishes "
            f"{float(gen17['decision']['spiking_margin_vs_static_per_1000_steps']):+.3f} versus static and "
            f"{float(gen17['decision']['spiking_margin_vs_shuffled_per_1000_steps']):+.3f} versus shuffled reward.",
            "supported" if bool(gen17["decision"]["reward_identity_gate"]) else "rejected",
        ),
        _claim(
            "Gen-18 stationary controls and manual-gradient implementation remain valid",
            bool(gen18["decision"]["identical_reset_gate"])
            and bool(gen18["decision"]["oracle_positive_control"])
            and bool(gen18["decision"]["manual_gradient_parity_gate"]),
            "Static reset is exact, oracle fitness is "
            f"{float(gen18_oracle['mean_final_fitness_per_1000_steps']):+.3f}, and maximum gradient error is "
            f"{float(gen18['decision']['maximum_manual_gradient_error']):.3e}.",
            "supported" if (
                bool(gen18["decision"]["identical_reset_gate"])
                and bool(gen18["decision"]["oracle_positive_control"])
                and bool(gen18["decision"]["manual_gradient_parity_gate"])
            ) else "rejected",
        ),
        _claim(
            "Gen-16 analog local-credit behavior replicates across ten held-out seeds",
            bool(gen18["decision"]["replicated_local_gain_gate"]),
            "Mean gain is "
            f"{float(gen18['decision']['local_gain_mean_per_1000_steps']):+.3f} with lower 95% bound "
            f"{float(gen18['decision']['local_gain_ci95_lower_per_1000_steps']):+.3f} and "
            f"{int(gen18['decision']['qualified_gain_seed_count'])}/10 qualified seeds.",
            "supported" if bool(gen18["decision"]["replicated_local_gain_gate"]) else "rejected",
        ),
        _claim(
            "Gen-18 local behavior depends reliably on correctly assigned reward",
            bool(gen18["decision"]["replicated_reward_identity_gate"]),
            "Correct minus shuffled reward is "
            f"{float(gen18['decision']['local_margin_vs_shuffled_mean_per_1000_steps']):+.3f} with lower 95% bound "
            f"{float(gen18['decision']['local_margin_vs_shuffled_ci95_lower_per_1000_steps']):+.3f} and "
            f"{int(gen18['decision']['qualified_reward_identity_seed_count'])}/10 qualified seeds.",
            "supported" if bool(gen18["decision"]["replicated_reward_identity_gate"]) else "rejected",
        ),
        _claim(
            "The tested local reward-credit program qualifies for further mechanism expansion",
            gen18["decision"]["status"] == "pass",
            f"Gen-18 returned status={gen18['decision']['status']} and next_milestone="
            f"{gen18['decision']['next_milestone']}.",
            "supported" if gen18["decision"]["status"] == "pass" else "rejected",
        ),
        _claim(
            "Gen-19 establishes a learnable parameter-matched N-MNIST benchmark",
            bool(gen19["decision"]["dataset_learnability_gate"])
            and bool(gen19["decision"]["matched_accuracy_gate"]),
            "Conv1D reaches "
            f"{100.0 * float(gen19['summary']['mean_conv_accuracy']):.3f}% and residual LIF reaches "
            f"{100.0 * float(gen19['summary']['mean_full_accuracy']):.3f}%.",
            "supported" if (
                bool(gen19["decision"]["dataset_learnability_gate"])
                and bool(gen19["decision"]["matched_accuracy_gate"])
            ) else "rejected",
        ),
        _claim(
            "Residual LIF state is causally used on N-MNIST",
            bool(gen19["decision"]["state_contribution_gate"]),
            "Removing state costs "
            f"{100.0 * float(gen19['summary']['mean_state_contribution_vs_direct_only']):.3f} points on average with "
            f"{int(gen19['summary']['state_contribution_seed_count'])}/3 qualifying seeds.",
            "supported" if bool(gen19["decision"]["state_contribution_gate"]) else "rejected",
        ),
        _claim(
            "Residual LIF state is beneficially sample-specific on N-MNIST",
            bool(gen19["decision"]["state_identity_gate"]),
            "Full minus shuffled-state accuracy is "
            f"{100.0 * float(gen19['summary']['mean_state_specificity_vs_shuffled']):+.3f} points with "
            f"{int(gen19['summary']['state_specificity_seed_count'])}/3 qualifying seeds.",
            "supported" if bool(gen19["decision"]["state_identity_gate"]) else "rejected",
        ),
        _claim(
            "The event-audio residual-state result generalizes to event vision",
            gen19["decision"]["status"] == "pass",
            f"Gen-19 returned status={gen19['decision']['status']} and next_milestone="
            f"{gen19['decision']['next_milestone']}.",
            "supported" if gen19["decision"]["status"] == "pass" else "rejected",
        ),
        _claim(
            "Gen-20 retains the strong dense N-MNIST spatial-temporal representation",
            float(gen20_teacher["best_validation_accuracy"])
            >= float(gen20["config"]["minimum_screen_accuracy"]),
            "The dense teacher reaches "
            f"{100.0 * float(gen20_teacher['best_validation_accuracy']):.3f}% validation accuracy against the "
            f"{100.0 * float(gen20['config']['minimum_screen_accuracy']):.1f}% screen gate.",
            "supported",
        ),
        _claim(
            "Gen-20 multiscale residual PLIF closes the N-MNIST representation gap",
            float(gen20_multiscale["best_validation_accuracy"])
            >= float(gen20["config"]["minimum_screen_accuracy"]),
            "The best new spiking arm reaches "
            f"{100.0 * float(gen20_multiscale['best_validation_accuracy']):.3f}%, missing promotion by "
            f"{100.0 * (float(gen20['config']['minimum_screen_accuracy']) - float(gen20_multiscale['best_validation_accuracy'])):.3f} points.",
            "rejected",
        ),
        _claim(
            "Gen-20 teacher distillation improves the multiscale spiking translation",
            float(gen20_distilled["best_validation_accuracy"])
            > float(gen20_multiscale["best_validation_accuracy"]),
            "Distillation changes validation accuracy by "
            f"{100.0 * (float(gen20_distilled['best_validation_accuracy']) - float(gen20_multiscale['best_validation_accuracy'])):+.3f} points.",
            "rejected",
        ),
        _claim(
            "Gen-20 proposed arms maintain sparse activity and a low operation proxy",
            (
                float(gen20["config"]["minimum_activity"])
                <= float(gen20_multiscale["validation_activity"])
                <= float(gen20["config"]["maximum_activity"])
                and gen20_teacher_ops / gen20_multiscale_proxy
                >= float(gen20["config"]["minimum_ops_reduction"])
            ),
            "The best arm has "
            f"{100.0 * float(gen20_multiscale['validation_activity']):.3f}% activity and a "
            f"{gen20_teacher_ops / gen20_multiscale_proxy:.2f}x activity-scaled operation reduction versus the teacher.",
            "supported",
        ),
        _claim(
            "Gen-20 establishes causal temporal state use on N-MNIST",
            False,
            "No new arm passed the screen, so confirmation, state removal, and temporal-order controls did not run.",
            "not tested",
        ),
        _claim(
            "Gen-20 qualifies the program for an automatic Gen-21 architecture phase",
            gen20["decision"]["status"] == "pass",
            f"Gen-20 returned status={gen20['decision']['status']}, reason="
            f"{gen20['decision']['reason']}, and next_milestone="
            f"{gen20['decision']['next_milestone']}.",
            "rejected",
        ),
    ]
    roadmap = [
        {
            "priority": 1,
            "workstream": "publication_evidence_closeout",
            "objective": "Package the supported event-audio mechanism and the Gen-19/20 event-vision boundary conditions.",
            "success_measure": "A reproducible 22-source ledger reports positive, negative, and untested gates without post-hoc rescue.",
        },
        {
            "priority": 2,
            "workstream": "matched_causal_mechanism_benchmark",
            "objective": "Test one supported event-audio residual-state backbone with factorial adaptive-mechanism ablations.",
            "success_measure": "Dynamic topology, dual memory, learned delays, and local reward credit are each compared under matched parameters, active operations, seeds, and optimization budgets.",
        },
        {
            "priority": 3,
            "workstream": "event_vision_theory_reset",
            "objective": "Require a genuinely new representation hypothesis before reopening N-MNIST state identity.",
            "success_measure": "No Gen-19 or Gen-20 rescue sweep is labeled confirmatory evidence.",
        },
        {
            "priority": 4,
            "workstream": "complex_plasticity_remains_gated",
            "objective": "Keep strong continuous-learning and hardware-energy claims closed until factorial causal gates pass.",
            "success_measure": "Each mechanism must add replicated task, adaptation, or retention value beyond matched static controls; energy requires direct measurement.",
        },
    ]
    return Gen5EvidenceSynthesisResult(
        sources=sources,
        metrics=metrics,
        claims=claims,
        roadmap=roadmap,
    )


def plot_gen5_evidence_synthesis(
    result: Gen5EvidenceSynthesisResult, path: str | pathlib.Path
) -> None:
    import matplotlib.pyplot as plt

    metrics = result.metrics
    labels = ("SHD Conv1D", "SHD state-only LIF", "SHD residual LIF")
    shd_values = (
        metrics["phase44_shd_conv1d_accuracy"],
        metrics["shd_state_only_lif_accuracy"],
        metrics["shd_residual_lif_accuracy"],
    )
    ssc_labels = ("SSC Conv1D", "SSC residual LIF", "SSC dilated TCN")
    ssc_values = (
        metrics["ssc_conv1d_final_accuracy"],
        metrics["ssc_residual_lif_final_accuracy"],
        metrics["ssc_tcn_accuracy"],
    )
    figure, axes = plt.subplots(18, 1, figsize=(13, 75), constrained_layout=True)
    axes[0].bar(labels, [100.0 * value for value in shd_values], color=("#167d55", "#bd3d3a", "#35b4f2"))
    axes[0].set_ylabel("SHD test accuracy (%)")
    axes[0].set_title("AMMC Gen-5 final evidence synthesis")
    axes[1].bar(ssc_labels, [100.0 * value for value in ssc_values], color=("#167d55", "#35b4f2", "#8b6fd6"))
    axes[1].set_ylabel("SSC test accuracy (%)")
    milestone_labels = (
        "TCN",
        "Residual LIF",
        "Hierarchical analog",
        "Hierarchical LIF",
    )
    milestone_values = (
        metrics["milestone_a_tcn_screen_validation_accuracy"],
        metrics["milestone_a_residual_lif_screen_validation_accuracy"],
        metrics["milestone_a_hierarchical_analog_screen_validation_accuracy"],
        metrics["milestone_a_hierarchical_lif_screen_validation_accuracy"],
    )
    axes[2].bar(
        milestone_labels,
        [100.0 * value for value in milestone_values],
        color=("#8b6fd6", "#35b4f2", "#d88935", "#bd3d3a"),
    )
    axes[2].set_ylabel("Milestone A validation accuracy (%)")
    axes[2].set_title("Terminal architecture screen")
    axes[3].bar(
        ("TCN", "Shared residual LIF"),
        [
            100.0 * metrics["gen6_tcn_accuracy"],
            100.0 * metrics["gen6_lif_accuracy"],
        ],
        color=("#8b6fd6", "#35b4f2"),
    )
    axes[3].set_ylabel("Gen-6 SSC test accuracy (%)")
    axes[3].set_title("Gen-6 parity without beneficial state specificity")
    axes[4].bar(
        ("TCN", "Gen-7 paired LIF"),
        [
            100.0 * metrics["gen7_tcn_accuracy"],
            100.0 * metrics["gen7_lif_accuracy"],
        ],
        color=("#8b6fd6", "#35b4f2"),
    )
    axes[4].set_ylabel("Gen-7 SSC test accuracy (%)")
    axes[4].set_title("Gen-7 alignment succeeds; causal identity gate fails")
    axes[5].bar(
        ("TCN", "Pooled LIF", "Analog local", "Paired local LIF"),
        [
            100.0 * metrics["gen8_tcn_accuracy"],
            100.0 * metrics["gen8_pooled_lif_accuracy"],
            100.0 * metrics["gen8_analog_accuracy"],
            100.0 * metrics["gen8_candidate_screen_validation_accuracy"],
        ],
        color=("#8b6fd6", "#35b4f2", "#d88935", "#bd3d3a"),
    )
    axes[5].set_ylabel("Accuracy (confirmation; candidate screen)")
    axes[5].set_title("Gen-8 local LIF fails screening; analog order effect is partial")
    axes[6].bar(
        ("Static", "Readout", "Full fine-tune", "Predictive LIF screen"),
        [
            100.0 * metrics["gen9_tcn_static_shifted_accuracy"],
            100.0 * metrics["gen9_tcn_readout_final_shifted_accuracy"],
            100.0 * metrics["gen9_tcn_full_final_shifted_accuracy"],
            100.0 * metrics["gen9_predictive_lif_screen_validation_accuracy"],
        ],
        color=("#8b6fd6", "#35b4f2", "#167d55", "#bd3d3a"),
    )
    axes[6].set_ylabel("Accuracy (%)")
    axes[6].set_title("Gen-9 controls adapt; predictive LIF fails source screening")
    axes[7].bar(
        ("TCN clean", "TCN damaged", "Dropout clean", "Dropout damaged", "LIF screen damaged"),
        [
            100.0 * metrics["gen10_tcn_clean_accuracy"],
            100.0 * metrics["gen10_tcn_damaged_accuracy"],
            100.0 * metrics["gen10_dropout_clean_accuracy"],
            100.0 * metrics["gen10_dropout_damaged_accuracy"],
            100.0 * metrics["gen10_lif_screen_damaged_accuracy"],
        ],
        color=("#8b6fd6", "#bd3d3a", "#167d55", "#35b4f2", "#d88935"),
    )
    axes[7].set_ylabel("Accuracy (%)")
    axes[7].set_title("Gen-10 dropout helps; residual LIF fails screening")
    axes[8].bar(
        ("Static", "Readout", "Full", "Analog adapter", "LIF adapter"),
        [
            100.0 * (metrics["gen11_readout_final_shifted_accuracy"] - metrics["gen11_readout_adaptation_gain"]),
            100.0 * metrics["gen11_readout_final_shifted_accuracy"],
            100.0 * metrics["gen11_full_final_shifted_accuracy"],
            100.0 * (
                metrics["gen11_readout_final_shifted_accuracy"]
                - metrics["gen11_readout_adaptation_gain"]
                + metrics["gen11_analog_adaptation_gain"]
            ),
            100.0 * metrics["gen11_lif_final_shifted_accuracy"],
        ],
        color=("#8b6fd6", "#35b4f2", "#167d55", "#d88935", "#bd3d3a"),
    )
    axes[8].set_ylabel("Damaged accuracy (%)")
    axes[8].set_title("Gen-11 conventional adaptation wins; state identity gate fails")
    axes[9].bar(
        ("Static", "Readout", "Full", "Dense memory", "Spiking memory"),
        [
            100.0 * (metrics["gen12_readout_final_shifted_accuracy"] - metrics["gen12_readout_adaptation_gain"]),
            100.0 * metrics["gen12_readout_final_shifted_accuracy"],
            100.0 * metrics["gen12_full_final_shifted_accuracy"],
            100.0 * (
                metrics["gen12_readout_final_shifted_accuracy"]
                - metrics["gen12_readout_adaptation_gain"]
                + metrics["gen12_dense_adaptation_gain"]
            ),
            100.0 * metrics["gen12_spiking_final_shifted_accuracy"],
        ],
        color=("#8b6fd6", "#35b4f2", "#167d55", "#d88935", "#bd3d3a"),
    )
    axes[9].set_ylabel("Damaged accuracy (%)")
    axes[9].set_title("Gen-12 conventional adaptation wins; prototype memory fails")
    axes[10].bar(
        ("Static", "Readout", "Full", "Analog local", "Spiking local"),
        [
            100.0 * (metrics["gen13_readout_final_shifted_accuracy"] - metrics["gen13_readout_adaptation_gain"]),
            100.0 * metrics["gen13_readout_final_shifted_accuracy"],
            100.0 * metrics["gen13_full_final_shifted_accuracy"],
            100.0 * (
                metrics["gen13_readout_final_shifted_accuracy"]
                - metrics["gen13_readout_adaptation_gain"]
                + metrics["gen13_analog_adaptation_gain"]
            ),
            100.0 * metrics["gen13_spiking_final_shifted_accuracy"],
        ],
        color=("#8b6fd6", "#35b4f2", "#167d55", "#d88935", "#bd3d3a"),
    )
    axes[10].set_ylabel("Damaged accuracy (%)")
    axes[10].set_title("Gen-13 conventional adaptation wins; local plasticity fails")
    axes[11].bar(
        ("Static", "Oracle", "Analog eligibility", "Spiking eligibility", "Shuffled reward"),
        [
            metrics["gen14_static_final_fitness"],
            metrics["gen14_oracle_final_fitness"],
            metrics["gen14_analog_final_fitness"],
            metrics["gen14_spiking_final_fitness"],
            metrics["gen14_shuffled_final_fitness"],
        ],
        color=("#8b6fd6", "#167d55", "#d88935", "#bd3d3a", "#35b4f2"),
    )
    axes[11].set_ylabel("Net fitness / 1,000 steps")
    axes[11].set_title("Gen-14 oracle succeeds; reward-specific eligibility fails")
    axes[12].bar(
        ("Static", "Oracle", "REINFORCE", "Shuffled reward"),
        [
            metrics["gen15_static_final_fitness"],
            metrics["gen15_oracle_final_fitness"],
            metrics["gen15_reinforce_final_fitness"],
            metrics["gen15_shuffled_final_fitness"],
        ],
        color=("#8b6fd6", "#167d55", "#d88935", "#35b4f2"),
    )
    axes[12].set_ylabel("Net fitness / 1,000 steps")
    axes[12].set_title("Gen-15 stationary reward protocol supports conventional learning")
    axes[13].bar(
        ("Static", "Oracle", "Autograd", "Manual local", "Shuffled reward"),
        [
            metrics["gen16_static_final_fitness"],
            metrics["gen16_oracle_final_fitness"],
            metrics["gen16_autograd_final_fitness"],
            metrics["gen16_local_final_fitness"],
            metrics["gen16_shuffled_final_fitness"],
        ],
        color=("#8b6fd6", "#167d55", "#35b4f2", "#d88935", "#bd3d3a"),
    )
    axes[13].set_ylabel("Net fitness / 1,000 steps")
    axes[13].set_title("Gen-16 exact local score credit matches autograd")
    axes[14].bar(
        ("Static spikes", "Oracle", "Analog local", "Spiking local", "Shuffled spikes"),
        [
            metrics["gen17_static_final_fitness"],
            metrics["gen17_oracle_final_fitness"],
            metrics["gen17_analog_final_fitness"],
            metrics["gen17_spiking_final_fitness"],
            metrics["gen17_shuffled_final_fitness"],
        ],
        color=("#8b6fd6", "#167d55", "#d88935", "#bd3d3a", "#35b4f2"),
    )
    axes[14].set_ylabel("Net fitness / 1,000 steps")
    axes[14].set_title("Gen-17 active Bernoulli spikes fail gain and reward identity")
    axes[15].bar(
        ("Static", "Oracle", "Correct local", "Shuffled local"),
        [
            metrics["gen18_static_final_fitness"],
            metrics["gen18_oracle_final_fitness"],
            metrics["gen18_local_final_fitness"],
            metrics["gen18_shuffled_final_fitness"],
        ],
        color=("#8b6fd6", "#167d55", "#d88935", "#35b4f2"),
    )
    axes[15].set_ylabel("Net fitness / 1,000 steps")
    axes[15].set_title("Gen-18 positive mean local credit fails held-out confidence gates")
    axes[16].bar(
        ("Conv1D", "Residual full", "Direct only", "Shuffled state"),
        [
            100.0 * metrics["gen19_conv_accuracy"],
            100.0 * metrics["gen19_residual_lif_accuracy"],
            100.0 * (
                metrics["gen19_residual_lif_accuracy"]
                - metrics["gen19_state_contribution_vs_direct_only"]
            ),
            100.0 * (
                metrics["gen19_residual_lif_accuracy"]
                - metrics["gen19_state_specificity_vs_shuffled"]
            ),
        ],
        color=("#167d55", "#35b4f2", "#d88935", "#bd3d3a"),
    )
    axes[16].set_ylabel("N-MNIST accuracy (%)")
    axes[16].set_title("Gen-19 state removal hurts, but state shuffling improves event vision")
    axes[17].bar(
        ("Dense teacher", "ConvPLIF", "Multiscale PLIF", "Distilled PLIF"),
        [
            100.0 * metrics["gen20_teacher_screen_accuracy"],
            100.0 * metrics["gen20_conv_plif_screen_accuracy"],
            100.0 * metrics["gen20_multiscale_screen_accuracy"],
            100.0 * metrics["gen20_distilled_screen_accuracy"],
        ],
        color=("#167d55", "#8b6fd6", "#35b4f2", "#d88935"),
    )
    axes[17].axhline(
        100.0 * metrics["gen20_screen_accuracy_gate"],
        color="#bd3d3a",
        linestyle="--",
        label="promotion gate",
    )
    axes[17].set_ylabel("Validation accuracy (%)")
    axes[17].set_title("Gen-20 sparse arms remain below the frozen promotion gate")
    axes[17].legend()
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _summary_row(payload: dict, arm: str, *, key: str = "summary") -> dict:
    for row in payload[key]:
        if row["arm"] == arm:
            return row
    raise KeyError(f"missing summary arm: {arm}")


def _strategy_row(payload: dict, strategy: str, *, key: str = "adaptation_summary") -> dict:
    for row in payload[key]:
        if row["strategy"] == strategy:
            return row
    raise KeyError(f"missing adaptation strategy: {strategy}")


def _screen_ops_proxy(row: dict) -> float:
    dense = float(row["dense_macs_per_sample"])
    analog = float(row["analog_dense_macs_per_sample"])
    activity = float(row["validation_activity"])
    return analog + (dense - analog) * activity


def _claim(name: str, passed: bool, evidence: str, status: str) -> dict:
    return {
        "claim": name,
        "status": status,
        "gate_passed": bool(passed),
        "evidence": evidence,
    }


def _render_report(result: Gen5EvidenceSynthesisResult) -> str:
    metrics = result.metrics
    lines = [
        "# AMMC Gen-5 through Gen-20 evidence report",
        "",
        "## Executive conclusion",
        "",
        "Across SHD and SSC, a residual LIF state carries sample-specific information that complements direct temporal convolution features. The mechanism is causal under feature-removal and shuffled-state tests, but it is neither standalone nor computationally competitive in the current dense PyTorch implementation.",
        "",
        "The strongest matched SSC baseline, a dilated TCN, exceeds residual LIF by "
        f"{100.0 * metrics['ssc_tcn_gain_vs_residual_lif']:.3f} accuracy points and runs at "
        f"{1.0 / metrics['ssc_residual_lif_throughput_ratio_vs_tcn']:.2f}x its throughput. "
        "The residual model uses a lower dense-MAC proxy, but no hardware-energy claim is justified.",
        "",
        "Milestone A then tested whether residual and hierarchical state models could clear a preregistered validation screen. Only the dilated TCN was promoted. Gen-6 subsequently preserved that predictor and added a zero-initialized, weight-shared LIF correction. It matched TCN accuracy within "
        f"{abs(100.0 * metrics['gen6_lif_gap_vs_tcn']):.3f} points and learned a non-zero gate with healthy spiking, but removing state cost only "
        f"{100.0 * metrics['gen6_lif_state_contribution_vs_direct_only']:.3f} points and shuffled state improved accuracy by "
        f"{-100.0 * metrics['gen6_lif_state_specificity_vs_shuffled']:.3f} points. Gen-6 therefore returned `stop` with no qualified causal arms.",
        "",
        "The accuracy-preservation hypothesis is supported; the beneficial sample-specific-correction hypothesis is rejected. Hardware optimization remains closed under the preregistered rule.",
        "",
        "Gen-7 then assigned state a paired future-prediction objective and a sample-conditioned gate. Paired LIF leads TCN by "
        f"{100.0 * metrics['gen7_lif_gain_vs_tcn']:+.3f} points and its future-alignment margin reaches "
        f"{metrics['gen7_lif_future_alignment_margin']:.4f}, but shuffled state improves accuracy by "
        f"{-100.0 * metrics['gen7_lif_state_specificity_vs_shuffled']:.3f} points and time reversal costs only "
        f"{100.0 * metrics['gen7_lif_temporal_order_vs_reversed']:.3f} points. Representation learning succeeded; beneficial identity/order-specific use did not. Gen-7 returned `stop`.",
        "",
        "Gen-8 moved prediction and fusion to aligned timesteps. Its paired local LIF candidate screened at "
        f"{100.0 * metrics['gen8_candidate_screen_validation_accuracy']:.3f}% with a "
        f"{100.0 * metrics['gen8_candidate_screen_spike_rate']:.3f}% spike rate and was not confirmed. "
        "The analog local binder remained within "
        f"{abs(100.0 * metrics['gen8_analog_gain_vs_tcn']):.3f} TCN points and reversal cost "
        f"{100.0 * metrics['gen8_analog_temporal_order_vs_reversed']:.3f} points, but state shuffling cost only "
        f"{100.0 * metrics['gen8_analog_state_specificity_vs_shuffled']:.3f} points. Local fusion introduces partial order sensitivity without beneficial identity-specific spiking use. Gen-8 returned `stop`.",
        "",
        "Gen-9 then tested adaptation after a fixed 35% sensor-bank failure. The confirmed TCN shift was "
        f"{100.0 * metrics['gen9_tcn_static_shift_drop']:.3f} points. A frozen TCN readout recovered "
        f"{100.0 * metrics['gen9_tcn_readout_adaptation_gain']:.3f} points, while full fine-tuning recovered "
        f"{100.0 * metrics['gen9_tcn_full_adaptation_gain']:.3f} points and retained the source task better. "
        "However, predictive LIF trailed the TCN screen by "
        f"{-100.0 * metrics['gen9_predictive_lif_screen_gap_vs_tcn']:.3f} points and was not promoted. "
        "Gen-9 therefore returned `stop`; STW/LTW, replay, modulation, and structural plasticity remain closed.",
        "",
        "Gen-10 tested masked-sensor residual state. Sensor dropout improved conventional clean and damaged accuracy by "
        f"{100.0 * metrics['gen10_dropout_clean_gain_vs_tcn']:.3f} and "
        f"{100.0 * metrics['gen10_dropout_damaged_gain_vs_tcn']:.3f} points. "
        "Residual analog missed the dropout-TCN clean/damaged screen by "
        f"{-100.0 * metrics['gen10_analog_screen_clean_gap']:.3f}/{-100.0 * metrics['gen10_analog_screen_damaged_gap']:.3f} points; "
        "residual LIF missed by "
        f"{-100.0 * metrics['gen10_lif_screen_clean_gap']:.3f}/{-100.0 * metrics['gen10_lif_screen_damaged_gap']:.3f} points despite healthy spiking. Gen-10 returned `stop`.",
        "",
        "Gen-11 froze that robust dropout-TCN backbone and adapted bounded downstream state. Full fine-tuning, readout adaptation, analog state, and LIF state recovered "
        f"{100.0 * metrics['gen11_full_adaptation_gain']:.3f}, "
        f"{100.0 * metrics['gen11_readout_adaptation_gain']:.3f}, "
        f"{100.0 * metrics['gen11_analog_adaptation_gain']:.3f}, and "
        f"{100.0 * metrics['gen11_lif_adaptation_gain']:.3f} points. Removing LIF state erased "
        f"{100.0 * metrics['gen11_lif_state_contribution']:.3f} points, but shuffling sample identity cost only "
        f"{100.0 * metrics['gen11_lif_state_specificity']:.3f} points. Gen-11 returned `stop`; synaptic STW/LTW remains closed.",
        "",
        "Gen-12 replaced the adapter with context-gated associative prototypes. Full fine-tuning and readout adaptation recovered "
        f"{100.0 * metrics['gen12_full_adaptation_gain']:.3f} and "
        f"{100.0 * metrics['gen12_readout_adaptation_gain']:.3f} points, while dense and spiking memories recovered only "
        f"{100.0 * metrics['gen12_dense_adaptation_gain']:.3f} and "
        f"{100.0 * metrics['gen12_spiking_adaptation_gain']:.3f}. Removing spiking memory cost "
        f"{100.0 * metrics['gen12_spiking_memory_contribution']:.3f} points and shuffling its class associations cost "
        f"{100.0 * metrics['gen12_spiking_association_specificity']:.3f}, despite the registered "
        f"{100.0 * metrics['gen12_spiking_activity']:.1f}% event density. Gen-12 returned `stop`.",
        "",
        "Gen-13 then localized supervised class-error credit to manual analog and spiking output-synapse updates. Full fine-tuning and autograd readout adaptation recovered "
        f"{100.0 * metrics['gen13_full_adaptation_gain']:.3f} and "
        f"{100.0 * metrics['gen13_readout_adaptation_gain']:.3f} points. Analog and spiking local rules recovered only "
        f"{100.0 * metrics['gen13_analog_adaptation_gain']:.3f} and "
        f"{100.0 * metrics['gen13_spiking_adaptation_gain']:.3f}. Removing spiking fast weights cost "
        f"{100.0 * metrics['gen13_spiking_fast_weight_contribution']:.3f} points and class shuffling cost "
        f"{100.0 * metrics['gen13_spiking_class_specificity']:.3f}, despite exactly "
        f"{100.0 * metrics['gen13_spiking_activity']:.1f}% trace density and zero source forgetting. Gen-13 returned `stop`.",
        "",
        "Gen-14 moved to delayed scalar reward in the embodied tensor world. The oracle reached "
        f"{metrics['gen14_oracle_final_fitness']:.3f} net fitness per 1,000 steps versus "
        f"{metrics['gen14_static_final_fitness']:.3f} for static behavior, confirming that the sensor-action task is solvable. "
        "Spiking eligibility improved relative to its own cold-start phase, but static behavior improved more over the same phase transition. "
        "The learned spiking policy finished "
        f"{metrics['gen14_spiking_margin_vs_static']:+.3f} versus static and "
        f"{metrics['gen14_spiking_margin_vs_shuffled']:+.3f} versus shuffled reward. "
        f"Activity remained healthy at {100.0 * metrics['gen14_spiking_activity']:.1f}% and weights did not saturate. "
        "Gen-14 returned `stop`: reward-specific local learning is rejected, and the baseline-to-evaluation rise is treated as phase non-stationarity.",
        "",
        "Gen-15 rebuilt each baseline and final evaluation from identical seeded state. Static behavior reproduced exactly, while conventional correct-reward REINFORCE gained "
        f"{metrics['gen15_reinforce_gain']:+.3f} fitness per 1,000 steps and finished "
        f"{metrics['gen15_reinforce_margin_vs_shuffled']:+.3f} above agent-shuffled reward. "
        f"The final mean remained {metrics['gen15_reinforce_final_fitness']:+.3f} and the improvement was seed-sensitive. "
        "Gen-15 validates the delayed reward and identity protocol, not Gen-14 or an AMMC local-learning mechanism.",
        "",
        "Gen-16 derived the exact score-function update on a matched linear policy. The manual gradient matched autograd within "
        f"{metrics['gen16_maximum_gradient_error']:.3e}, and both policies finished at "
        f"{metrics['gen16_local_final_fitness']:+.3f} with zero behavioral gap. The local rule gained "
        f"{metrics['gen16_local_gain']:+.3f}, finished "
        f"{metrics['gen16_local_margin_vs_shuffled']:+.3f} above shuffled reward, and passed identity on "
        f"{int(metrics['gen16_reward_identity_seed_count'])}/3 seeds. This validates analog linear local credit, not sparse spiking or memory.",
        "",
        "Gen-17 translated that rule to one Bernoulli sensory event per channel and decision step. Event activity remained healthy at "
        f"{100.0 * metrics['gen17_training_spike_density']:.3f}% during training and "
        f"{100.0 * metrics['gen17_evaluation_spike_density']:.3f}% during evaluation, while the manual gradient error remained "
        f"{metrics['gen17_maximum_gradient_error']:.3e}. Nevertheless, correct-reward spiking credit changed fitness by "
        f"{metrics['gen17_spiking_gain']:+.3f} and finished "
        f"{metrics['gen17_spiking_margin_vs_shuffled']:+.3f} relative to shuffled reward. The analog reference itself gained only "
        f"{metrics['gen17_analog_gain']:+.3f} on the fresh seeds. Gen-17 therefore rejects this sparse translation and reopens analog-credit replication.",
        "",
        "Gen-18 then held the analog rule fixed across ten untouched seeds. Correct reward improved mean fitness by "
        f"{metrics['gen18_local_gain']:+.3f} and finished "
        f"{metrics['gen18_local_margin_vs_shuffled']:+.3f} above shuffled reward. However, only "
        f"{int(metrics['gen18_qualified_gain_seed_count'])}/10 seeds met the gain gate and "
        f"{int(metrics['gen18_reward_identity_seed_count'])}/10 met reward identity; the lower 95% bounds were "
        f"{metrics['gen18_local_gain_ci95_lower']:+.3f} and "
        f"{metrics['gen18_local_margin_vs_shuffled_ci95_lower']:+.3f}. The local reward-credit program is therefore closed despite its positive mean.",
        "",
        "Gen-19 transferred the frozen residual-state test to N-MNIST event vision. Conv1D reached "
        f"{100.0 * metrics['gen19_conv_accuracy']:.3f}% and residual LIF reached "
        f"{100.0 * metrics['gen19_residual_lif_accuracy']:.3f}%, while removing state cost "
        f"{100.0 * metrics['gen19_state_contribution_vs_direct_only']:.3f} points. However, shuffling state between samples improved accuracy by "
        f"{-100.0 * metrics['gen19_state_specificity_vs_shuffled']:.3f} points and zero of three seeds passed identity. "
        "The external replication therefore stopped: sample-specific residual-state benefit is supported on SHD/SSC event audio, not N-MNIST event vision.",
        "",
        "Gen-20 then attempted to translate the successful dense N-MNIST spatial-temporal representation into multiscale residual PLIF state. The dense teacher screened at "
        f"{100.0 * metrics['gen20_teacher_screen_accuracy']:.3f}%, while ConvPLIF, multiscale PLIF, and distilled multiscale PLIF reached "
        f"{100.0 * metrics['gen20_conv_plif_screen_accuracy']:.3f}%, "
        f"{100.0 * metrics['gen20_multiscale_screen_accuracy']:.3f}%, and "
        f"{100.0 * metrics['gen20_distilled_screen_accuracy']:.3f}%. The best new arm missed the frozen promotion gate by "
        f"{-100.0 * metrics['gen20_multiscale_gap_to_gate']:.3f} points; distillation changed accuracy by "
        f"{100.0 * metrics['gen20_distillation_gain_vs_multiscale']:+.3f} points. Its "
        f"{100.0 * metrics['gen20_multiscale_activity']:.3f}% activity and "
        f"{metrics['gen20_multiscale_ops_reduction_vs_teacher']:.2f}x operation proxy are operational strengths, not substitutes for the failed accuracy gate. No arm was promoted, so causal state and time-order controls were not tested.",
        "",
        "## Claim ledger",
        "",
        "| Claim | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for claim in result.claims:
        lines.append(
            f"| {claim['claim']} | {claim['status']} | {claim['evidence']} |"
        )
    lines.extend(
        [
            "",
            "## Defensible contribution",
            "",
            "The supported contribution is a residual temporal mechanism in which direct convolutional features and LIF state are jointly necessary and beneficially sample-specific on two event-audio datasets. Gen-19 and Gen-20 define the event-vision boundary: state shuffling improves the former, while the latter's more ambitious sparse translation fails its promotion gate despite healthy activity and a low operation proxy. Later generations establish predictive alignment, partial analog order sensitivity, a valid damage-adaptation task, strong sensor-dropout robustness, conventional few-shot adaptation, a solvable embodied sensor-action control, a stationary delayed-reward protocol, and an exact manual score-function gradient. Frozen causal gates reject reliable behavioral replication of that local-credit rule, end-to-end spiking state, bounded state adapters, associative class prototypes, supervised three-factor output plasticity, the earlier reward-modulated eligibility rule, and the Gen-17 one-sample Bernoulli translation. Local continual learning, structural plasticity, dual memory, learned-delay benefit, sparse-spiking credit, and hardware energy remain unproven. These are qualified mechanism, protocol, boundary-condition, and negative-selection results—not a best-SNN, Transformer-replacement, continuous-learning, synaptic-memory, or hardware-efficiency result.",
            "",
            "## Next-generation roadmap",
            "",
            "| Priority | Workstream | Objective | Success measure |",
            "| ---: | --- | --- | --- |",
        ]
    )
    for item in result.roadmap:
        lines.append(
            f"| {item['priority']} | {item['workstream']} | {item['objective']} | {item['success_measure']} |"
        )
    lines.extend(["", "## Evidence sources", ""])
    for phase, source in result.sources.items():
        lines.append(f"- {phase}: `{source}`")
    return "\n".join(lines) + "\n"


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
