"""Phase 48 cross-dataset replication of residual LIF on SSC."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import pathlib
import statistics
from typing import Iterable

from .event_mnist import torch
from .runtime import device_kind, resolve_device, seed_everything
from .shd_benchmark import SHDConfig, _measure
from .shd_calibrated_baselines import TemporalConvClassifier, matched_temporal_conv_channels
from .shd_residual_state_contribution import RESIDUAL_ABLATION_MODES
from .shd_state_placement_diagnostic import (
    ResidualTemporalConvStateClassifier,
    matched_temporal_conv_residual_channels,
)
from .shd_temporal_pyramid import DEFAULT_TEMPORAL_LEVELS
from .shd_validation_checkpoint import _train_validation_selected
from .ssc_benchmark import load_ssc_tensors


@dataclass
class SSCResidualLIFReplicationResult:
    config: SHDConfig
    device: str
    readout_seeds: tuple[int, ...]
    target_parameters: int
    temporal_levels: tuple[int, ...]
    validation_samples: int
    direct_channels: int
    residual_channels: int
    records: list[dict]
    summary: dict

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "ssc_residual_lif_replication.json"
        records_path = output / "ssc_residual_lif_replication_records.csv"
        summary_path = output / "ssc_residual_lif_replication_summary.csv"
        payload = {
            "config": asdict(self.config),
            "device": self.device,
            "readout_seeds": list(self.readout_seeds),
            "target_parameters": self.target_parameters,
            "temporal_levels": list(self.temporal_levels),
            "validation_samples": self.validation_samples,
            "direct_channels": self.direct_channels,
            "residual_channels": self.residual_channels,
            "records": self.records,
            "summary": self.summary,
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        _write_csv(records_path, self.records)
        _write_csv(summary_path, [self.summary])
        paths = {
            "json": str(json_path),
            "records_csv": str(records_path),
            "summary_csv": str(summary_path),
        }
        if plot:
            plot_path = output / "ssc_residual_lif_replication_summary.png"
            plot_ssc_residual_lif_replication(self.summary, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def run_ssc_residual_lif_replication(
    config: SHDConfig,
    *,
    readout_seeds: Iterable[int] = (142, 143, 144),
    validation_samples: int = 0,
    target_parameters: int = 133_631,
    device="auto",
    temporal_levels: Iterable[int] = DEFAULT_TEMPORAL_LEVELS,
    temporal_conv_kernel_size: int = 5,
    surrogate_slope: float = 10.0,
) -> SSCResidualLIFReplicationResult:
    if torch is None:
        raise ImportError("Phase 48 SSC replication requires PyTorch")
    seeds = tuple(int(seed) for seed in readout_seeds)
    levels = tuple(int(level) for level in temporal_levels)
    direct_channels, _ = matched_temporal_conv_channels(
        config.input_neurons,
        config.classes,
        target_parameters,
        kernel_size=temporal_conv_kernel_size,
        temporal_levels=levels,
    )
    residual_channels, _ = matched_temporal_conv_residual_channels(
        config.input_neurons,
        config.classes,
        target_parameters,
        kernel_size=temporal_conv_kernel_size,
        temporal_levels=levels,
    )
    resolved = resolve_device(device)
    (
        train_events,
        train_labels,
        validation_events,
        validation_labels,
        test_events,
        test_labels,
    ) = load_ssc_tensors(config, validation_samples=validation_samples)
    records: list[dict] = []
    for seed in seeds:
        seed_everything(seed, device=resolved)
        direct_model = TemporalConvClassifier(
            config,
            channels=direct_channels,
            kernel_size=temporal_conv_kernel_size,
            temporal_levels=levels,
        ).to(resolved)
        direct_training = _train_validation_selected(
            direct_model,
            train_events,
            train_labels,
            validation_events,
            validation_labels,
            config,
            seed=seed,
            device=resolved,
        )
        direct_model.load_state_dict(direct_training["best_state"])
        direct_accuracy, direct_seconds, _ = _measure(
            direct_model, test_events, test_labels, config.batch_size, resolved
        )

        seed_everything(seed, device=resolved)
        model = ResidualTemporalConvStateClassifier(
            config,
            channels=residual_channels,
            kernel_size=temporal_conv_kernel_size,
            temporal_levels=levels,
            dynamics="lif",
            surrogate_slope=surrogate_slope,
        ).to(resolved)
        training = _train_validation_selected(
            model,
            train_events,
            train_labels,
            validation_events,
            validation_labels,
            config,
            seed=seed,
            device=resolved,
        )
        model.load_state_dict(training["best_state"])
        measurements = {}
        for mode in RESIDUAL_ABLATION_MODES:
            model.set_ablation_mode(mode)
            accuracy, seconds, activity = _measure(
                model, test_events, test_labels, config.batch_size, resolved
            )
            measurements[mode] = {
                "accuracy": float(accuracy),
                "seconds": float(seconds),
                "activity": float(activity),
            }
        model.set_ablation_mode("full")
        full = measurements["full"]["accuracy"]
        direct_only = measurements["direct_only"]["accuracy"]
        state_only = measurements["state_only"]["accuracy"]
        shuffled = measurements["shuffled_state"]["accuracy"]
        parameters = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        records.append(
            {
                "seed": int(seed),
                "conv_reference_accuracy": float(direct_accuracy),
                "full_accuracy": float(full),
                "direct_only_accuracy": float(direct_only),
                "state_only_accuracy": float(state_only),
                "shuffled_state_accuracy": float(shuffled),
                "full_gain_vs_conv": float(full - direct_accuracy),
                "state_contribution_vs_direct_only": float(full - direct_only),
                "state_specificity_vs_shuffled": float(full - shuffled),
                "direct_contribution_vs_state_only": float(full - state_only),
                "best_epoch": int(training["best_epoch"]),
                "best_validation_accuracy": float(training["best_validation_accuracy"]),
                "full_activity": float(measurements["full"]["activity"]),
                "effective_trainable_parameters": int(parameters),
                "parameter_ratio_vs_target": float(parameters / target_parameters),
                "train_samples": int(train_events.shape[0]),
                "validation_samples": int(validation_events.shape[0]),
                "test_samples": int(test_events.shape[0]),
                "conv_test_examples_per_second": float(
                    test_events.shape[0] / max(direct_seconds, 1e-12)
                ),
                "full_test_examples_per_second": float(
                    test_events.shape[0] / max(measurements["full"]["seconds"], 1e-12)
                ),
                "train_seconds": float(training["train_seconds"]),
            }
        )
    return SSCResidualLIFReplicationResult(
        config=config,
        device=device_kind(resolved),
        readout_seeds=seeds,
        target_parameters=int(target_parameters),
        temporal_levels=levels,
        validation_samples=int(validation_samples),
        direct_channels=int(direct_channels),
        residual_channels=int(residual_channels),
        records=records,
        summary=summarize_ssc_residual_lif_replication(records),
    )


def summarize_ssc_residual_lif_replication(records: Iterable[dict]) -> dict:
    rows = list(records)
    if not rows:
        return {}
    contribution = [float(row["state_contribution_vs_direct_only"]) for row in rows]
    specificity = [float(row["state_specificity_vs_shuffled"]) for row in rows]
    conv_gains = [float(row["full_gain_vs_conv"]) for row in rows]
    return {
        "runs": len(rows),
        "mean_conv_reference_accuracy": statistics.fmean(
            float(row["conv_reference_accuracy"]) for row in rows
        ),
        "mean_full_accuracy": statistics.fmean(float(row["full_accuracy"]) for row in rows),
        "std_full_accuracy": statistics.pstdev(float(row["full_accuracy"]) for row in rows),
        "mean_direct_only_accuracy": statistics.fmean(
            float(row["direct_only_accuracy"]) for row in rows
        ),
        "mean_state_only_accuracy": statistics.fmean(
            float(row["state_only_accuracy"]) for row in rows
        ),
        "mean_shuffled_state_accuracy": statistics.fmean(
            float(row["shuffled_state_accuracy"]) for row in rows
        ),
        "mean_full_gain_vs_conv": statistics.fmean(conv_gains),
        "within_two_points_seed_count_vs_conv": sum(gain >= -0.02 for gain in conv_gains),
        "mean_state_contribution_vs_direct_only": statistics.fmean(contribution),
        "one_point_seed_count_state_contribution": sum(gain >= 0.01 for gain in contribution),
        "mean_state_specificity_vs_shuffled": statistics.fmean(specificity),
        "one_point_seed_count_state_specificity": sum(gain >= 0.01 for gain in specificity),
        "mean_direct_contribution_vs_state_only": statistics.fmean(
            float(row["direct_contribution_vs_state_only"]) for row in rows
        ),
        "mean_full_activity": statistics.fmean(float(row["full_activity"]) for row in rows),
        "effective_trainable_parameters": int(rows[0]["effective_trainable_parameters"]),
        "parameter_ratio_vs_target": statistics.fmean(
            float(row["parameter_ratio_vs_target"]) for row in rows
        ),
        "train_samples": int(rows[0]["train_samples"]),
        "validation_samples": int(rows[0]["validation_samples"]),
        "test_samples": int(rows[0]["test_samples"]),
        "mean_conv_test_examples_per_second": statistics.fmean(
            float(row["conv_test_examples_per_second"]) for row in rows
        ),
        "mean_full_test_examples_per_second": statistics.fmean(
            float(row["full_test_examples_per_second"]) for row in rows
        ),
        "mean_train_seconds": statistics.fmean(float(row["train_seconds"]) for row in rows),
    }


def plot_ssc_residual_lif_replication(summary: dict, path: str | pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    labels = ("Conv1D", "Full residual LIF", "Direct only", "State only", "Shuffled state")
    values = (
        summary["mean_conv_reference_accuracy"],
        summary["mean_full_accuracy"],
        summary["mean_direct_only_accuracy"],
        summary["mean_state_only_accuracy"],
        summary["mean_shuffled_state_accuracy"],
    )
    figure, axes = plt.subplots(2, 1, figsize=(13, 10), constrained_layout=True)
    axes[0].bar(labels, [100.0 * value for value in values], color=("#167d55", "#35b4f2", "#ffb31a", "#8b6fd6", "#bd3d3a"))
    axes[0].set_ylabel("SSC test accuracy (%)")
    axes[0].set_title("AMMC Gen-5 Phase 48: SSC residual-LIF replication")
    contribution = 100.0 * summary["mean_state_contribution_vs_direct_only"]
    specificity = 100.0 * summary["mean_state_specificity_vs_shuffled"]
    axes[1].bar(("Full - direct only", "Full - shuffled state"), (contribution, specificity), color=("#35b4f2", "#167d55"))
    axes[1].axhline(1.0, color="#bd3d3a", linestyle="--", label="+1 point replication gate")
    axes[1].set_ylabel("State contribution (points)")
    axes[1].legend()
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
