"""Phase 47 causal contribution ablation for residual temporal state on SHD."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import pathlib
import statistics
from typing import Iterable

from .event_mnist import torch
from .runtime import device_kind, resolve_device, seed_everything
from .shd_benchmark import SHDConfig, _measure, load_shd_tensors
from .shd_calibrated_baselines import TemporalConvClassifier, matched_temporal_conv_channels
from .shd_state_placement_diagnostic import (
    ResidualTemporalConvStateClassifier,
    matched_temporal_conv_residual_channels,
)
from .shd_temporal_pyramid import DEFAULT_TEMPORAL_LEVELS
from .shd_validation_checkpoint import _stratified_split, _train_validation_selected


@dataclass(frozen=True)
class SHDResidualStateContributionArm:
    name: str
    dynamics: str


SHD_RESIDUAL_STATE_CONTRIBUTION_ARMS = (
    SHDResidualStateContributionArm("residual_analog", "analog"),
    SHDResidualStateContributionArm("residual_lif", "lif"),
)

RESIDUAL_ABLATION_MODES = ("full", "direct_only", "state_only", "shuffled_state")


def available_shd_residual_state_contribution_arms() -> tuple[str, ...]:
    return tuple(arm.name for arm in SHD_RESIDUAL_STATE_CONTRIBUTION_ARMS)


@dataclass
class SHDResidualStateContributionResult:
    config: SHDConfig
    device: str
    readout_seeds: tuple[int, ...]
    validation_fraction: float
    target_parameters: int
    temporal_levels: tuple[int, ...]
    direct_channels: int
    residual_channels: int
    arms: list[dict]
    records: list[dict]
    summary: list[dict]

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "shd_residual_state_contribution.json"
        records_path = output / "shd_residual_state_contribution_records.csv"
        summary_path = output / "shd_residual_state_contribution_summary.csv"
        payload = {
            "config": asdict(self.config),
            "device": self.device,
            "readout_seeds": list(self.readout_seeds),
            "validation_fraction": self.validation_fraction,
            "target_parameters": self.target_parameters,
            "temporal_levels": list(self.temporal_levels),
            "direct_channels": self.direct_channels,
            "residual_channels": self.residual_channels,
            "arms": self.arms,
            "records": self.records,
            "summary": self.summary,
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        _write_csv(records_path, self.records)
        _write_csv(summary_path, self.summary)
        paths = {
            "json": str(json_path),
            "records_csv": str(records_path),
            "summary_csv": str(summary_path),
        }
        if plot:
            plot_path = output / "shd_residual_state_contribution_summary.png"
            plot_shd_residual_state_contribution(self.summary, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def run_shd_residual_state_contribution(
    config: SHDConfig,
    *,
    readout_seeds: Iterable[int] = (142, 143, 144),
    validation_fraction: float = 0.10,
    target_parameters: int = 133_631,
    device="auto",
    temporal_levels: Iterable[int] = DEFAULT_TEMPORAL_LEVELS,
    temporal_conv_kernel_size: int = 5,
    surrogate_slope: float = 10.0,
) -> SHDResidualStateContributionResult:
    if torch is None:
        raise ImportError("Phase 47 residual-state contribution requires PyTorch")
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
    all_train_events, all_train_labels, test_events, test_labels = load_shd_tensors(config)
    train_events, train_labels, validation_events, validation_labels = _stratified_split(
        all_train_events,
        all_train_labels,
        fraction=validation_fraction,
        seed=config.data_seed + 43_000,
    )
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
        direct_accuracy, _, _ = _measure(
            direct_model, test_events, test_labels, config.batch_size, resolved
        )
        for arm in SHD_RESIDUAL_STATE_CONTRIBUTION_ARMS:
            seed_everything(seed, device=resolved)
            model = ResidualTemporalConvStateClassifier(
                config,
                channels=residual_channels,
                kernel_size=temporal_conv_kernel_size,
                temporal_levels=levels,
                dynamics=arm.dynamics,
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
            parameters = sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            )
            full = measurements["full"]["accuracy"]
            direct_only = measurements["direct_only"]["accuracy"]
            state_only = measurements["state_only"]["accuracy"]
            shuffled = measurements["shuffled_state"]["accuracy"]
            records.append(
                {
                    "seed": int(seed),
                    "arm": arm.name,
                    "dynamics": arm.dynamics,
                    "best_epoch": int(training["best_epoch"]),
                    "best_validation_accuracy": float(training["best_validation_accuracy"]),
                    "conv_reference_accuracy": float(direct_accuracy),
                    "full_accuracy": float(full),
                    "direct_only_accuracy": float(direct_only),
                    "state_only_accuracy": float(state_only),
                    "shuffled_state_accuracy": float(shuffled),
                    "full_gain_vs_conv": float(full - direct_accuracy),
                    "state_contribution_vs_direct_only": float(full - direct_only),
                    "state_specificity_vs_shuffled": float(full - shuffled),
                    "direct_contribution_vs_state_only": float(full - state_only),
                    "effective_trainable_parameters": int(parameters),
                    "parameter_ratio_vs_target": float(parameters / target_parameters),
                    "full_activity": float(measurements["full"]["activity"]),
                    "full_test_examples_per_second": float(
                        test_events.shape[0] / max(measurements["full"]["seconds"], 1e-12)
                    ),
                    "train_seconds": float(training["train_seconds"]),
                }
            )
    return SHDResidualStateContributionResult(
        config=config,
        device=device_kind(resolved),
        readout_seeds=seeds,
        validation_fraction=float(validation_fraction),
        target_parameters=int(target_parameters),
        temporal_levels=levels,
        direct_channels=int(direct_channels),
        residual_channels=int(residual_channels),
        arms=[asdict(arm) for arm in SHD_RESIDUAL_STATE_CONTRIBUTION_ARMS],
        records=records,
        summary=summarize_shd_residual_state_contribution(records),
    )


def summarize_shd_residual_state_contribution(records: Iterable[dict]) -> list[dict]:
    rows = list(records)
    summary: list[dict] = []
    for arm in SHD_RESIDUAL_STATE_CONTRIBUTION_ARMS:
        group = [row for row in rows if row["arm"] == arm.name]
        if not group:
            continue
        state_contribution = [
            float(row["state_contribution_vs_direct_only"]) for row in group
        ]
        state_specificity = [
            float(row["state_specificity_vs_shuffled"]) for row in group
        ]
        summary.append(
            {
                "arm": arm.name,
                "dynamics": arm.dynamics,
                "runs": len(group),
                "mean_conv_reference_accuracy": statistics.fmean(
                    float(row["conv_reference_accuracy"]) for row in group
                ),
                "mean_full_accuracy": statistics.fmean(
                    float(row["full_accuracy"]) for row in group
                ),
                "std_full_accuracy": statistics.pstdev(
                    float(row["full_accuracy"]) for row in group
                ),
                "mean_direct_only_accuracy": statistics.fmean(
                    float(row["direct_only_accuracy"]) for row in group
                ),
                "mean_state_only_accuracy": statistics.fmean(
                    float(row["state_only_accuracy"]) for row in group
                ),
                "mean_shuffled_state_accuracy": statistics.fmean(
                    float(row["shuffled_state_accuracy"]) for row in group
                ),
                "mean_full_gain_vs_conv": statistics.fmean(
                    float(row["full_gain_vs_conv"]) for row in group
                ),
                "mean_state_contribution_vs_direct_only": statistics.fmean(
                    state_contribution
                ),
                "one_point_seed_count_state_contribution": sum(
                    gain >= 0.01 for gain in state_contribution
                ),
                "mean_state_specificity_vs_shuffled": statistics.fmean(
                    state_specificity
                ),
                "one_point_seed_count_state_specificity": sum(
                    gain >= 0.01 for gain in state_specificity
                ),
                "mean_direct_contribution_vs_state_only": statistics.fmean(
                    float(row["direct_contribution_vs_state_only"]) for row in group
                ),
                "mean_full_activity": statistics.fmean(
                    float(row["full_activity"]) for row in group
                ),
                "effective_trainable_parameters": int(group[0]["effective_trainable_parameters"]),
                "parameter_ratio_vs_target": statistics.fmean(
                    float(row["parameter_ratio_vs_target"]) for row in group
                ),
                "mean_train_seconds": statistics.fmean(
                    float(row["train_seconds"]) for row in group
                ),
                "mean_full_test_examples_per_second": statistics.fmean(
                    float(row["full_test_examples_per_second"]) for row in group
                ),
            }
        )
    return summary


def plot_shd_residual_state_contribution(
    summary: list[dict], path: str | pathlib.Path
) -> None:
    import matplotlib.pyplot as plt

    labels = [row["arm"].replace("_", "\n") for row in summary]
    x = list(range(len(summary)))
    series = (
        ("Full", "mean_full_accuracy", "#167d55"),
        ("Direct only", "mean_direct_only_accuracy", "#ffb31a"),
        ("State only", "mean_state_only_accuracy", "#8b6fd6"),
        ("Shuffled state", "mean_shuffled_state_accuracy", "#bd3d3a"),
    )
    width = 0.18
    figure, axes = plt.subplots(2, 1, figsize=(13, 10), constrained_layout=True)
    for index, (label, key, color) in enumerate(series):
        offset = (index - 1.5) * width
        axes[0].bar(
            [value + offset for value in x],
            [100.0 * float(row[key]) for row in summary],
            width,
            label=label,
            color=color,
        )
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_title("AMMC Gen-5 Phase 47: residual-state contribution")
    axes[0].legend()
    contribution = [
        100.0 * float(row["mean_state_contribution_vs_direct_only"])
        for row in summary
    ]
    specificity = [
        100.0 * float(row["mean_state_specificity_vs_shuffled"])
        for row in summary
    ]
    axes[1].bar([value - width / 2 for value in x], contribution, width, label="Full - direct only", color="#35b4f2")
    axes[1].bar([value + width / 2 for value in x], specificity, width, label="Full - shuffled state", color="#167d55")
    axes[1].axhline(1.0, color="#bd3d3a", linestyle="--", label="+1 point contribution gate")
    axes[1].set_ylabel("State contribution (points)")
    axes[1].legend()
    for axis in axes:
        axis.set_xticks(x, labels)
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
