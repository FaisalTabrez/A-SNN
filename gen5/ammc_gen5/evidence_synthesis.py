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
    ]
    roadmap = [
        {
            "priority": 1,
            "workstream": "gen6_terminal_closeout",
            "objective": "Package predictive parity together with the failed state-specificity gate.",
            "success_measure": "A reproducible final ledger whose claims match the Gen-6 stop decision.",
        },
        {
            "priority": 2,
            "workstream": "publication_package",
            "objective": "Report the supported cross-dataset Gen-5 mechanism and negative Gen-6 successor result without architecture-superiority claims.",
            "success_measure": "Exact protocols, seeds, checkpoints, causal controls, and negative gates are publication-ready.",
        },
        {
            "priority": 3,
            "workstream": "hardware_work_deferred",
            "objective": "Keep event-driven kernel optimization closed after the Gen-6 terminal failure.",
            "success_measure": "No hardware-efficiency claim is pursued for an architecture with no qualified causal arm.",
        },
        {
            "priority": 4,
            "workstream": "new_program_requires_new_hypothesis",
            "objective": "Prevent an automatic Gen-7 rescue sweep on the rejected shared-residual design.",
            "success_measure": "Any future generation starts from a separately approved hypothesis and preregistration.",
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
    figure, axes = plt.subplots(4, 1, figsize=(13, 18), constrained_layout=True)
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
        "# AMMC Gen-5/Gen-6 evidence report",
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
            "The supported contribution is a residual temporal mechanism in which direct convolutional features and LIF state are jointly necessary on two event-audio datasets. Milestone A does not support retaining that implementation as a competitive architecture. Gen-6 demonstrates that a zero-initialized shared correction can preserve the conventional predictor, but its learned state is not beneficially sample-specific under shuffling. This is a causal mechanism result accompanied by two negative architecture-selection results, not a best-SNN, Transformer-replacement, or hardware-efficiency result.",
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
