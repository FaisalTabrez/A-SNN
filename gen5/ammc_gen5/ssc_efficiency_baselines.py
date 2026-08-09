"""Phase 49 matched SSC temporal baselines and efficiency proxies."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import pathlib
import statistics
from typing import Iterable

from .event_mnist import nn, torch
from .runtime import device_kind, resolve_device, seed_everything
from .shd_benchmark import SHDConfig, _measure
from .shd_calibrated_baselines import TemporalConvClassifier, matched_temporal_conv_channels
from .shd_state_placement_diagnostic import (
    ResidualTemporalConvStateClassifier,
    matched_temporal_conv_residual_channels,
)
from .shd_temporal_pyramid import DEFAULT_TEMPORAL_LEVELS
from .shd_validation_checkpoint import _train_validation_selected
from .ssc_benchmark import load_ssc_tensors


@dataclass(frozen=True)
class SSCEfficiencyBaselineArm:
    name: str
    model_kind: str


SSC_EFFICIENCY_BASELINE_ARMS = (
    SSCEfficiencyBaselineArm("temporal_conv1d", "conv1d"),
    SSCEfficiencyBaselineArm("dilated_tcn", "tcn"),
    SSCEfficiencyBaselineArm("residual_lif", "residual_lif"),
)


def available_ssc_efficiency_baseline_arms() -> tuple[str, ...]:
    return tuple(arm.name for arm in SSC_EFFICIENCY_BASELINE_ARMS)


def temporal_tcn_parameter_count(
    input_neurons: int,
    channels: int,
    classes: int,
    *,
    input_kernel_size: int,
    hidden_kernel_size: int,
    temporal_levels: Iterable[int],
) -> int:
    return int(
        channels * input_neurons * input_kernel_size
        + channels
        + channels * channels * hidden_kernel_size
        + channels
        + channels * sum(int(level) for level in temporal_levels) * classes
        + classes
    )


def matched_temporal_tcn_channels(
    input_neurons: int,
    classes: int,
    target_parameters: int,
    *,
    input_kernel_size: int,
    hidden_kernel_size: int,
    temporal_levels: Iterable[int],
) -> tuple[int, int]:
    channels = 1
    while temporal_tcn_parameter_count(
        input_neurons,
        channels + 1,
        classes,
        input_kernel_size=input_kernel_size,
        hidden_kernel_size=hidden_kernel_size,
        temporal_levels=temporal_levels,
    ) <= target_parameters:
        channels += 1
    return channels, temporal_tcn_parameter_count(
        input_neurons,
        channels,
        classes,
        input_kernel_size=input_kernel_size,
        hidden_kernel_size=hidden_kernel_size,
        temporal_levels=temporal_levels,
    )


class TemporalDilatedTCNClassifier(nn.Module):
    """Two-layer residual temporal convolution with dilation two."""

    def __init__(
        self,
        config: SHDConfig,
        *,
        channels: int,
        input_kernel_size: int,
        hidden_kernel_size: int,
        dilation: int,
        temporal_levels: Iterable[int],
    ) -> None:
        if torch is None:
            raise ImportError("Phase 49 SSC baselines require PyTorch")
        super().__init__()
        self.config = config
        self.channels = int(channels)
        self.temporal_levels = tuple(int(level) for level in temporal_levels)
        self.input_conv = nn.Conv1d(
            config.input_neurons,
            channels,
            input_kernel_size,
            padding=input_kernel_size // 2,
        )
        self.hidden_conv = nn.Conv1d(
            channels,
            channels,
            hidden_kernel_size,
            padding=dilation * (hidden_kernel_size // 2),
            dilation=dilation,
        )
        self.classifier = nn.Linear(
            channels * sum(self.temporal_levels), config.classes
        )

    def forward(self, events, *, return_event_rate: bool = False):  # type: ignore[override]
        if events.ndim != 3 or events.shape[2] != self.config.input_neurons:
            raise ValueError("events must have shape [batch, time, input_neurons]")
        direct = torch.relu(
            self.input_conv(events.to(torch.float32).transpose(1, 2))
        )
        trace = torch.relu(self.hidden_conv(direct) + direct).transpose(1, 2)
        features = _multiscale_features(trace, self.temporal_levels)
        logits = self.classifier(torch.cat(features, dim=1))
        if return_event_rate:
            return logits, trace.mean()
        return logits


@dataclass
class SSCEfficiencyBaselinesResult:
    config: SHDConfig
    device: str
    readout_seeds: tuple[int, ...]
    target_parameters: int
    temporal_levels: tuple[int, ...]
    validation_samples: int
    arms: list[dict]
    records: list[dict]
    summary: list[dict]

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "ssc_efficiency_baselines.json"
        records_path = output / "ssc_efficiency_baselines_records.csv"
        summary_path = output / "ssc_efficiency_baselines_summary.csv"
        payload = {
            "config": asdict(self.config),
            "device": self.device,
            "readout_seeds": list(self.readout_seeds),
            "target_parameters": self.target_parameters,
            "temporal_levels": list(self.temporal_levels),
            "validation_samples": self.validation_samples,
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
            plot_path = output / "ssc_efficiency_baselines_summary.png"
            plot_ssc_efficiency_baselines(self.summary, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def run_ssc_efficiency_baselines(
    config: SHDConfig,
    *,
    readout_seeds: Iterable[int] = (142, 143, 144),
    validation_samples: int = 0,
    target_parameters: int = 133_631,
    device="auto",
    temporal_levels: Iterable[int] = DEFAULT_TEMPORAL_LEVELS,
    input_kernel_size: int = 5,
    hidden_kernel_size: int = 3,
    tcn_dilation: int = 2,
    surrogate_slope: float = 10.0,
) -> SSCEfficiencyBaselinesResult:
    if torch is None:
        raise ImportError("Phase 49 SSC baselines require PyTorch")
    seeds = tuple(int(seed) for seed in readout_seeds)
    levels = tuple(int(level) for level in temporal_levels)
    conv_channels, _ = matched_temporal_conv_channels(
        config.input_neurons,
        config.classes,
        target_parameters,
        kernel_size=input_kernel_size,
        temporal_levels=levels,
    )
    tcn_channels, _ = matched_temporal_tcn_channels(
        config.input_neurons,
        config.classes,
        target_parameters,
        input_kernel_size=input_kernel_size,
        hidden_kernel_size=hidden_kernel_size,
        temporal_levels=levels,
    )
    lif_channels, _ = matched_temporal_conv_residual_channels(
        config.input_neurons,
        config.classes,
        target_parameters,
        kernel_size=input_kernel_size,
        temporal_levels=levels,
    )
    channel_lookup = {
        "temporal_conv1d": conv_channels,
        "dilated_tcn": tcn_channels,
        "residual_lif": lif_channels,
    }
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
        for arm in SSC_EFFICIENCY_BASELINE_ARMS:
            seed_everything(seed, device=resolved)
            channels = channel_lookup[arm.name]
            if arm.model_kind == "conv1d":
                model = TemporalConvClassifier(
                    config,
                    channels=channels,
                    kernel_size=input_kernel_size,
                    temporal_levels=levels,
                ).to(resolved)
                activity_kind = "relu_activation"
            elif arm.model_kind == "tcn":
                model = TemporalDilatedTCNClassifier(
                    config,
                    channels=channels,
                    input_kernel_size=input_kernel_size,
                    hidden_kernel_size=hidden_kernel_size,
                    dilation=tcn_dilation,
                    temporal_levels=levels,
                ).to(resolved)
                activity_kind = "relu_activation"
            else:
                model = ResidualTemporalConvStateClassifier(
                    config,
                    channels=channels,
                    kernel_size=input_kernel_size,
                    temporal_levels=levels,
                    dynamics="lif",
                    surrogate_slope=surrogate_slope,
                ).to(resolved)
                activity_kind = "spike_rate"
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
            accuracy, seconds, activity = _measure(
                model, test_events, test_labels, config.batch_size, resolved
            )
            parameters = sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            )
            dense_macs = _dense_macs_per_sample(
                arm.model_kind,
                timesteps=config.timesteps,
                input_neurons=config.input_neurons,
                channels=channels,
                classes=config.classes,
                input_kernel_size=input_kernel_size,
                hidden_kernel_size=hidden_kernel_size,
                temporal_levels=levels,
            )
            records.append(
                {
                    "seed": int(seed),
                    "arm": arm.name,
                    "model_kind": arm.model_kind,
                    "channels": int(channels),
                    "best_epoch": int(training["best_epoch"]),
                    "best_validation_accuracy": float(training["best_validation_accuracy"]),
                    "test_accuracy": float(accuracy),
                    "activity": float(activity),
                    "activity_kind": activity_kind,
                    "effective_trainable_parameters": int(parameters),
                    "parameter_ratio_vs_target": float(parameters / target_parameters),
                    "dense_macs_per_sample": int(dense_macs),
                    "state_updates_per_sample": int(
                        config.timesteps * channels if arm.model_kind == "residual_lif" else 0
                    ),
                    "estimated_spike_events_per_sample": float(
                        activity * config.timesteps * channels
                        if arm.model_kind == "residual_lif"
                        else 0.0
                    ),
                    "test_examples_per_second": float(
                        test_events.shape[0] / max(seconds, 1e-12)
                    ),
                    "train_seconds": float(training["train_seconds"]),
                    "train_samples": int(train_events.shape[0]),
                    "validation_samples": int(validation_events.shape[0]),
                    "test_samples": int(test_events.shape[0]),
                }
            )
    _attach_residual_comparisons(records)
    return SSCEfficiencyBaselinesResult(
        config=config,
        device=device_kind(resolved),
        readout_seeds=seeds,
        target_parameters=int(target_parameters),
        temporal_levels=levels,
        validation_samples=int(validation_samples),
        arms=[asdict(arm) for arm in SSC_EFFICIENCY_BASELINE_ARMS],
        records=records,
        summary=summarize_ssc_efficiency_baselines(records),
    )


def summarize_ssc_efficiency_baselines(records: Iterable[dict]) -> list[dict]:
    rows = list(records)
    summary: list[dict] = []
    for arm in SSC_EFFICIENCY_BASELINE_ARMS:
        group = [row for row in rows if row["arm"] == arm.name]
        if not group:
            continue
        gains = [float(row["accuracy_gain_vs_residual_lif"]) for row in group]
        summary.append(
            {
                "arm": arm.name,
                "model_kind": arm.model_kind,
                "runs": len(group),
                "mean_test_accuracy": statistics.fmean(
                    float(row["test_accuracy"]) for row in group
                ),
                "std_test_accuracy": statistics.pstdev(
                    float(row["test_accuracy"]) for row in group
                ),
                "mean_accuracy_gain_vs_residual_lif": statistics.fmean(gains),
                "two_point_seed_count_over_residual_lif": sum(gain >= 0.02 for gain in gains),
                "mean_activity": statistics.fmean(float(row["activity"]) for row in group),
                "activity_kind": group[0]["activity_kind"],
                "channels": int(group[0]["channels"]),
                "effective_trainable_parameters": int(group[0]["effective_trainable_parameters"]),
                "parameter_ratio_vs_target": statistics.fmean(
                    float(row["parameter_ratio_vs_target"]) for row in group
                ),
                "dense_macs_per_sample": int(group[0]["dense_macs_per_sample"]),
                "state_updates_per_sample": int(group[0]["state_updates_per_sample"]),
                "mean_estimated_spike_events_per_sample": statistics.fmean(
                    float(row["estimated_spike_events_per_sample"]) for row in group
                ),
                "mean_test_examples_per_second": statistics.fmean(
                    float(row["test_examples_per_second"]) for row in group
                ),
                "mean_train_seconds": statistics.fmean(
                    float(row["train_seconds"]) for row in group
                ),
            }
        )
    return summary


def plot_ssc_efficiency_baselines(summary: list[dict], path: str | pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    labels = [row["arm"].replace("_", "\n") for row in summary]
    x = list(range(len(summary)))
    colors = ("#167d55", "#8b6fd6", "#35b4f2")
    figure, axes = plt.subplots(3, 1, figsize=(13, 12), constrained_layout=True)
    axes[0].bar(
        x,
        [100.0 * float(row["mean_test_accuracy"]) for row in summary],
        yerr=[100.0 * float(row["std_test_accuracy"]) for row in summary],
        capsize=5,
        color=colors,
    )
    axes[0].set_ylabel("SSC test accuracy (%)")
    axes[0].set_title("AMMC Gen-5 Phase 49: SSC accuracy and efficiency audit")
    axes[1].bar(
        x,
        [float(row["mean_test_examples_per_second"]) for row in summary],
        color=colors,
    )
    axes[1].set_ylabel("Test examples / second")
    axes[2].bar(
        x,
        [float(row["dense_macs_per_sample"]) / 1_000_000 for row in summary],
        color=colors,
    )
    axes[2].set_ylabel("Dense MAC proxy / sample (millions)")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25)
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _dense_macs_per_sample(
    model_kind: str,
    *,
    timesteps: int,
    input_neurons: int,
    channels: int,
    classes: int,
    input_kernel_size: int,
    hidden_kernel_size: int,
    temporal_levels: Iterable[int],
) -> int:
    first_conv = timesteps * input_neurons * channels * input_kernel_size
    if model_kind == "tcn":
        hidden_conv = timesteps * channels * channels * hidden_kernel_size
        readout = channels * sum(int(level) for level in temporal_levels) * classes
        return int(first_conv + hidden_conv + readout)
    if model_kind == "residual_lif":
        readout_features = 2 * sum(int(level) for level in temporal_levels) + 1
        readout = channels * readout_features * classes
        return int(first_conv + readout)
    readout = channels * sum(int(level) for level in temporal_levels) * classes
    return int(first_conv + readout)


def _attach_residual_comparisons(records: list[dict]) -> None:
    residual = {
        int(row["seed"]): row for row in records if row["arm"] == "residual_lif"
    }
    for row in records:
        reference = residual[int(row["seed"])]
        row["accuracy_gain_vs_residual_lif"] = float(row["test_accuracy"]) - float(
            reference["test_accuracy"]
        )


def _multiscale_features(trace, levels: Iterable[int]) -> list:
    timesteps = int(trace.shape[1])
    features = []
    for level in levels:
        for window in range(int(level)):
            start = window * timesteps // int(level)
            stop = (window + 1) * timesteps // int(level)
            features.append(trace[:, start:stop].mean(dim=1))
    return features


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
