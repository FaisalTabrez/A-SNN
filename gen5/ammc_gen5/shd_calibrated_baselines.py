"""Phase 44 validation-calibrated, parameter-matched SHD temporal baselines."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import pathlib
import statistics
from typing import Iterable

from .event_mnist import nn, torch
from .runtime import device_kind, resolve_device, seed_everything
from .shd_benchmark import SHDConfig, _measure, load_shd_tensors
from .shd_matched_baselines import DenseLIFTemporalClassifier, GRUTemporalClassifier, matched_gru_hidden_units
from .shd_temporal_controls import SHDRawTemporalPyramidClassifier
from .shd_temporal_pyramid import DEFAULT_TEMPORAL_LEVELS
from .shd_validation_checkpoint import _stratified_split, _train_validation_selected


@dataclass(frozen=True)
class SHDCalibratedBaselineArm:
    name: str
    model_kind: str


SHD_CALIBRATED_BASELINE_ARMS = (
    SHDCalibratedBaselineArm("raw_temporal_pyramid", "raw_temporal"),
    SHDCalibratedBaselineArm("temporal_conv1d", "temporal_conv1d"),
    SHDCalibratedBaselineArm("gru_temporal", "gru"),
    SHDCalibratedBaselineArm("dense_lif_recurrent", "dense_lif"),
)


def available_shd_calibrated_baseline_arms() -> tuple[str, ...]:
    return tuple(arm.name for arm in SHD_CALIBRATED_BASELINE_ARMS)


def temporal_conv_parameter_count(input_neurons: int, channels: int, classes: int, *, kernel_size: int, temporal_levels: Iterable[int]) -> int:
    levels = tuple(int(level) for level in temporal_levels)
    return int(
        channels * input_neurons * kernel_size + channels
        + channels * sum(levels) * classes + classes
    )


def matched_temporal_conv_channels(input_neurons: int, classes: int, target_parameters: int, *, kernel_size: int, temporal_levels: Iterable[int]) -> tuple[int, int]:
    channels = 1
    while temporal_conv_parameter_count(input_neurons, channels + 1, classes, kernel_size=kernel_size, temporal_levels=temporal_levels) <= target_parameters:
        channels += 1
    return channels, temporal_conv_parameter_count(input_neurons, channels, classes, kernel_size=kernel_size, temporal_levels=temporal_levels)


class TemporalConvClassifier(nn.Module):
    """One-layer temporal CNN with multi-scale pooling and matched parameters."""

    def __init__(self, config: SHDConfig, *, channels: int, kernel_size: int, temporal_levels: Iterable[int]) -> None:
        if torch is None:
            raise ImportError("Phase 44 calibrated baselines require PyTorch")
        super().__init__()
        self.config = config
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.temporal_levels = tuple(int(level) for level in temporal_levels)
        self.temporal = nn.Conv1d(
            config.input_neurons, channels, kernel_size,
            padding=kernel_size // 2,
        )
        self.classifier = nn.Linear(channels * sum(self.temporal_levels), config.classes)

    def forward(self, events, *, return_event_rate: bool = False):  # type: ignore[override]
        if events.ndim != 3 or events.shape[2] != self.config.input_neurons:
            raise ValueError("events must have shape [batch, time, input_neurons]")
        trace = torch.relu(self.temporal(events.to(torch.float32).transpose(1, 2))).transpose(1, 2)
        timesteps = int(trace.shape[1])
        features = []
        for level in self.temporal_levels:
            for window in range(level):
                start = window * timesteps // level
                stop = (window + 1) * timesteps // level
                features.append(trace[:, start:stop].mean(dim=1))
        logits = self.classifier(torch.cat(features, dim=1))
        if return_event_rate:
            return logits, trace.mean()
        return logits


@dataclass
class SHDCalibratedBaselinesResult:
    config: SHDConfig
    device: str
    readout_seeds: tuple[int, ...]
    validation_fraction: float
    target_parameters: int
    temporal_levels: tuple[int, ...]
    temporal_conv_channels: int
    gru_hidden_units: int
    arms: list[dict]
    records: list[dict]
    summary: list[dict]

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "shd_calibrated_baselines.json"
        records_path = output / "shd_calibrated_baselines_records.csv"
        summary_path = output / "shd_calibrated_baselines_summary.csv"
        payload = {
            "config": asdict(self.config), "device": self.device,
            "readout_seeds": list(self.readout_seeds), "validation_fraction": self.validation_fraction,
            "target_parameters": self.target_parameters, "temporal_levels": list(self.temporal_levels),
            "temporal_conv_channels": self.temporal_conv_channels,
            "gru_hidden_units": self.gru_hidden_units, "arms": self.arms,
            "records": self.records, "summary": self.summary,
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        _write_csv(records_path, self.records)
        _write_csv(summary_path, self.summary)
        paths = {"json": str(json_path), "records_csv": str(records_path), "summary_csv": str(summary_path)}
        if plot:
            plot_path = output / "shd_calibrated_baselines_summary.png"
            plot_shd_calibrated_baselines(self.summary, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def run_shd_calibrated_baselines(
    config: SHDConfig,
    *,
    readout_seeds: Iterable[int] = (142, 143, 144),
    validation_fraction: float = 0.10,
    target_parameters: int = 133_631,
    device="auto",
    temporal_levels: Iterable[int] = DEFAULT_TEMPORAL_LEVELS,
    raw_projection_dim: int = 32,
    dense_lif_hidden_neurons: int = 128,
    dense_lif_projection_dim: int = 16,
    temporal_conv_kernel_size: int = 5,
    surrogate_slope: float = 10.0,
) -> SHDCalibratedBaselinesResult:
    if torch is None:
        raise ImportError("Phase 44 calibrated baselines require PyTorch")
    seeds = tuple(int(seed) for seed in readout_seeds)
    levels = tuple(int(level) for level in temporal_levels)
    if not seeds or not 0.0 < validation_fraction < 0.5:
        raise ValueError("invalid readout seeds or validation fraction")
    conv_channels, _ = matched_temporal_conv_channels(
        config.input_neurons, config.classes, target_parameters,
        kernel_size=temporal_conv_kernel_size, temporal_levels=levels,
    )
    gru_hidden, _ = matched_gru_hidden_units(config.input_neurons, config.classes, target_parameters)
    resolved = resolve_device(device)
    all_train_events, all_train_labels, test_events, test_labels = load_shd_tensors(config)
    train_events, train_labels, validation_events, validation_labels = _stratified_split(
        all_train_events, all_train_labels,
        fraction=validation_fraction, seed=config.data_seed + 43_000,
    )
    records: list[dict] = []
    for seed in seeds:
        for arm in SHD_CALIBRATED_BASELINE_ARMS:
            seed_everything(seed, device=resolved)
            if arm.model_kind == "raw_temporal":
                model = SHDRawTemporalPyramidClassifier(
                    config, projection_dim=raw_projection_dim,
                    temporal_levels=levels, target_parameters=target_parameters,
                ).to(resolved)
                hidden_units = model.readout.bottleneck_units
                activity_kind = "input_event_rate"
            elif arm.model_kind == "temporal_conv1d":
                model = TemporalConvClassifier(
                    config, channels=conv_channels,
                    kernel_size=temporal_conv_kernel_size, temporal_levels=levels,
                ).to(resolved)
                hidden_units = conv_channels
                activity_kind = "relu_activation"
            elif arm.model_kind == "gru":
                model = GRUTemporalClassifier(config, hidden_units=gru_hidden).to(resolved)
                hidden_units = gru_hidden
                activity_kind = "analog_activation"
            else:
                model = DenseLIFTemporalClassifier(
                    config, hidden_neurons=dense_lif_hidden_neurons,
                    temporal_levels=levels, projection_dim=dense_lif_projection_dim,
                    target_parameters=target_parameters, surrogate_slope=surrogate_slope,
                ).to(resolved)
                hidden_units = dense_lif_hidden_neurons
                activity_kind = "hidden_spike_rate"
            training = _train_validation_selected(
                model, train_events, train_labels, validation_events, validation_labels,
                config, seed=seed, device=resolved,
            )
            final_accuracy, final_seconds, final_activity = _measure(
                model, test_events, test_labels, config.batch_size, resolved
            )
            model.load_state_dict(training["best_state"])
            checkpoint_accuracy, checkpoint_seconds, checkpoint_activity = _measure(
                model, test_events, test_labels, config.batch_size, resolved
            )
            parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            records.append(
                {
                    "seed": int(seed), "arm": arm.name, "model_kind": arm.model_kind,
                    "hidden_units": int(hidden_units), "train_samples": int(train_events.shape[0]),
                    "validation_samples": int(validation_events.shape[0]),
                    "best_epoch": int(training["best_epoch"]),
                    "best_validation_accuracy": float(training["best_validation_accuracy"]),
                    "final_validation_accuracy": float(training["final_validation_accuracy"]),
                    "final_test_accuracy": float(final_accuracy),
                    "checkpoint_test_accuracy": float(checkpoint_accuracy),
                    "checkpoint_gain_vs_final": float(checkpoint_accuracy - final_accuracy),
                    "effective_trainable_parameters": int(parameters),
                    "parameter_ratio_vs_target": float(parameters / target_parameters),
                    "final_activity": float(final_activity), "checkpoint_activity": float(checkpoint_activity),
                    "activity_kind": activity_kind, "train_seconds": float(training["train_seconds"]),
                    "final_test_examples_per_second": float(test_events.shape[0] / max(final_seconds, 1e-12)),
                    "checkpoint_test_examples_per_second": float(test_events.shape[0] / max(checkpoint_seconds, 1e-12)),
                }
            )
    _attach_raw_comparisons(records)
    return SHDCalibratedBaselinesResult(
        config=config, device=device_kind(resolved), readout_seeds=seeds,
        validation_fraction=float(validation_fraction), target_parameters=int(target_parameters),
        temporal_levels=levels, temporal_conv_channels=int(conv_channels),
        gru_hidden_units=int(gru_hidden), arms=[asdict(arm) for arm in SHD_CALIBRATED_BASELINE_ARMS],
        records=records, summary=summarize_shd_calibrated_baselines(records),
    )


def summarize_shd_calibrated_baselines(records: Iterable[dict]) -> list[dict]:
    rows = list(records)
    summary: list[dict] = []
    for arm in SHD_CALIBRATED_BASELINE_ARMS:
        group = [row for row in rows if row["arm"] == arm.name]
        if not group:
            continue
        gains = [float(row["checkpoint_gain_vs_raw"]) for row in group]
        summary.append(
            {
                "arm": arm.name, "model_kind": arm.model_kind, "runs": len(group),
                "mean_final_test_accuracy": statistics.fmean(float(row["final_test_accuracy"]) for row in group),
                "std_final_test_accuracy": statistics.pstdev(float(row["final_test_accuracy"]) for row in group),
                "mean_checkpoint_test_accuracy": statistics.fmean(float(row["checkpoint_test_accuracy"]) for row in group),
                "std_checkpoint_test_accuracy": statistics.pstdev(float(row["checkpoint_test_accuracy"]) for row in group),
                "mean_checkpoint_gain_vs_raw": statistics.fmean(gains),
                "positive_seed_count_vs_raw": sum(gain > 0.0 for gain in gains),
                "two_point_seed_count_vs_raw": sum(gain >= 0.02 for gain in gains),
                "mean_checkpoint_gain_vs_final": statistics.fmean(float(row["checkpoint_gain_vs_final"]) for row in group),
                "mean_best_epoch": statistics.fmean(int(row["best_epoch"]) for row in group),
                "mean_best_validation_accuracy": statistics.fmean(float(row["best_validation_accuracy"]) for row in group),
                "hidden_units": int(group[0]["hidden_units"]),
                "effective_trainable_parameters": int(group[0]["effective_trainable_parameters"]),
                "parameter_ratio_vs_target": statistics.fmean(float(row["parameter_ratio_vs_target"]) for row in group),
                "mean_checkpoint_activity": statistics.fmean(float(row["checkpoint_activity"]) for row in group),
                "activity_kind": group[0]["activity_kind"],
                "mean_train_seconds": statistics.fmean(float(row["train_seconds"]) for row in group),
                "mean_checkpoint_test_examples_per_second": statistics.fmean(float(row["checkpoint_test_examples_per_second"]) for row in group),
            }
        )
    return summary


def plot_shd_calibrated_baselines(summary: list[dict], path: str | pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    labels = [row["arm"].replace("_", "\n") for row in summary]
    x = list(range(len(summary)))
    accuracy = [100.0 * float(row["mean_checkpoint_test_accuracy"]) for row in summary]
    errors = [100.0 * float(row["std_checkpoint_test_accuracy"]) for row in summary]
    gains = [100.0 * float(row["mean_checkpoint_gain_vs_raw"]) for row in summary]
    throughput = [float(row["mean_checkpoint_test_examples_per_second"]) for row in summary]
    colors = ("#ffb31a", "#167d55", "#666666", "#bd3d3a")
    figure, axes = plt.subplots(3, 1, figsize=(15, 13), constrained_layout=True)
    axes[0].bar(x, accuracy, yerr=errors, capsize=5, color=colors)
    axes[0].set_ylabel("Best-validation test accuracy (%)")
    axes[0].set_title("AMMC Gen-5 Phase 44: calibrated SHD temporal baselines")
    axes[1].bar(x, gains, color=colors)
    axes[1].axhline(2.0, color="#bd3d3a", linestyle="--", label="+2 points vs raw")
    axes[1].set_ylabel("Gain vs paired raw (points)")
    axes[1].legend()
    axes[2].bar(x, throughput, color=colors)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Test examples / second (log)")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25)
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _attach_raw_comparisons(records: list[dict]) -> None:
    raw = {int(row["seed"]): row for row in records if row["arm"] == "raw_temporal_pyramid"}
    for row in records:
        row["checkpoint_gain_vs_raw"] = float(row["checkpoint_test_accuracy"]) - float(raw[int(row["seed"])]["checkpoint_test_accuracy"])


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
