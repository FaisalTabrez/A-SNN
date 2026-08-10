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
    ]
    roadmap = [
        {
            "priority": 1,
            "workstream": "gen12_terminal_closeout",
            "objective": "Package conventional adaptation gains and the failed associative-prototype gate.",
            "success_measure": "A reproducible 14-source ledger whose claims match the Gen-12 stop decision.",
        },
        {
            "priority": 2,
            "workstream": "publication_package",
            "objective": "Report the supported mechanism chain through Gen-12 without architecture-superiority claims.",
            "success_measure": "Exact protocols, seeds, checkpoints, causal controls, and negative gates are publication-ready.",
        },
        {
            "priority": 3,
            "workstream": "hardware_work_deferred",
            "objective": "Keep synaptic STW/LTW, replay, structural plasticity, and event-driven kernel optimization closed after the Gen-12 terminal failure.",
            "success_measure": "No hardware-efficiency claim is pursued for an architecture with no qualified causal arm.",
        },
        {
            "priority": 4,
            "workstream": "local_credit_assignment_hypothesis",
            "objective": "Test explicit three-factor local output-synapse updates rather than tune failed prototype retrieval.",
            "success_measure": "Sparse local plasticity matches autograd readout adaptation and passes weight-removal and class-shuffle controls.",
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
    figure, axes = plt.subplots(10, 1, figsize=(13, 43), constrained_layout=True)
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
        "# AMMC Gen-5 through Gen-12 evidence report",
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
            "The supported contribution is a residual temporal mechanism in which direct convolutional features and LIF state are jointly necessary on two event-audio datasets. Later generations establish predictive alignment, partial analog order sensitivity, a valid damage-adaptation task, strong sensor-dropout robustness, and conventional few-shot adaptation, while rejecting end-to-end spiking state, bounded state adapters, and associative class prototypes at frozen causal gates. Gen-12 localizes the remaining positive signal to task-specific output-synapse credit assignment. These are qualified mechanism and negative-selection results—not a best-SNN, Transformer-replacement, continuous-learning, synaptic-memory, or hardware-efficiency result.",
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
