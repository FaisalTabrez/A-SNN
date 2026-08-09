"""Phase 45 learned temporal convolution with analog and spiking state."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import math
import pathlib
import statistics
from typing import Iterable

from .event_mnist import nn, torch
from .runtime import device_kind, resolve_device, seed_everything
from .shd_benchmark import SHDConfig, _measure, load_shd_tensors
from .shd_calibrated_baselines import TemporalConvClassifier, matched_temporal_conv_channels
from .shd_matched_baselines import DenseLIFTemporalClassifier
from .shd_temporal_controls import SHDRawTemporalPyramidClassifier
from .shd_temporal_pyramid import DEFAULT_TEMPORAL_LEVELS
from .shd_validation_checkpoint import _stratified_split, _train_validation_selected
from .trainable_temporal_mnist import SurrogateSpike


@dataclass(frozen=True)
class SHDSpikingTemporalConvArm:
    name: str
    model_kind: str


SHD_SPIKING_TEMPORAL_CONV_ARMS = (
    SHDSpikingTemporalConvArm("raw_temporal_pyramid", "raw"),
    SHDSpikingTemporalConvArm("temporal_conv1d", "conv_ann"),
    SHDSpikingTemporalConvArm("temporal_conv_leaky_analog", "conv_analog"),
    SHDSpikingTemporalConvArm("temporal_conv_leaky_lif", "conv_lif"),
    SHDSpikingTemporalConvArm("dense_lif_recurrent", "dense_lif"),
)


def available_shd_spiking_temporal_conv_arms() -> tuple[str, ...]:
    return tuple(arm.name for arm in SHD_SPIKING_TEMPORAL_CONV_ARMS)


def temporal_conv_state_parameter_count(input_neurons: int, channels: int, classes: int, *, kernel_size: int, temporal_levels: Iterable[int], spiking: bool) -> int:
    state_parameters = channels * (2 if spiking else 1)
    return int(
        channels * input_neurons * kernel_size + channels
        + channels * (sum(int(level) for level in temporal_levels) + 1) * classes
        + classes + state_parameters
    )


def matched_temporal_conv_state_channels(input_neurons: int, classes: int, target_parameters: int, *, kernel_size: int, temporal_levels: Iterable[int]) -> tuple[int, int]:
    channels = 1
    while temporal_conv_state_parameter_count(input_neurons, channels + 1, classes, kernel_size=kernel_size, temporal_levels=temporal_levels, spiking=True) <= target_parameters:
        channels += 1
    return channels, temporal_conv_state_parameter_count(input_neurons, channels, classes, kernel_size=kernel_size, temporal_levels=temporal_levels, spiking=True)


class TemporalConvStateClassifier(nn.Module):
    """Trainable temporal filters followed by leaky analog or LIF state."""

    def __init__(
        self,
        config: SHDConfig,
        *,
        channels: int,
        kernel_size: int,
        temporal_levels: Iterable[int],
        dynamics: str,
        surrogate_slope: float,
        initial_leak: float = 0.90,
        initial_threshold: float = 1.0,
    ) -> None:
        if torch is None:
            raise ImportError("Phase 45 spiking temporal convolution requires PyTorch")
        if dynamics not in {"analog", "lif"}:
            raise ValueError("dynamics must be analog or lif")
        super().__init__()
        self.config = config
        self.channels = int(channels)
        self.temporal_levels = tuple(int(level) for level in temporal_levels)
        self.dynamics = dynamics
        self.surrogate_slope = float(surrogate_slope)
        self.temporal = nn.Conv1d(
            config.input_neurons, channels, kernel_size,
            padding=kernel_size // 2,
        )
        leak_logit = math.log(initial_leak / (1.0 - initial_leak))
        self.leak_logit = nn.Parameter(torch.full((channels,), leak_logit))
        if dynamics == "lif":
            threshold_raw = math.log(math.expm1(initial_threshold))
            self.threshold_raw = nn.Parameter(torch.full((channels,), threshold_raw))
        else:
            self.register_parameter("threshold_raw", None)
        feature_dim = channels * (sum(self.temporal_levels) + 1)
        self.classifier = nn.Linear(feature_dim, config.classes)

    def forward(self, events, *, return_event_rate: bool = False):  # type: ignore[override]
        if events.ndim != 3 or events.shape[2] != self.config.input_neurons:
            raise ValueError("events must have shape [batch, time, input_neurons]")
        currents = self.temporal(events.to(torch.float32).transpose(1, 2)).transpose(1, 2)
        leak = torch.sigmoid(self.leak_logit)
        membrane = currents.new_zeros((events.shape[0], self.channels))
        trace = []
        activity_sum = currents.new_zeros(())
        threshold = None
        if self.dynamics == "lif":
            threshold = torch.nn.functional.softplus(self.threshold_raw).clamp_min(1e-3)
        for step in range(currents.shape[1]):
            pre_reset = leak * membrane + currents[:, step]
            if self.dynamics == "lif":
                activation = SurrogateSpike.apply(
                    pre_reset - threshold, self.surrogate_slope
                )
                membrane = pre_reset - activation * threshold
                activity_sum = activity_sum + activation.mean()
            else:
                membrane = pre_reset
                activation = torch.tanh(membrane)
                activity_sum = activity_sum + activation.abs().mean()
            trace.append(activation)
        stacked = torch.stack(trace, dim=1)
        features = []
        timesteps = int(stacked.shape[1])
        for level in self.temporal_levels:
            for window in range(level):
                start = window * timesteps // level
                stop = (window + 1) * timesteps // level
                features.append(stacked[:, start:stop].mean(dim=1))
        final_state = torch.tanh(membrane) if self.dynamics == "analog" else membrane / threshold
        features.append(final_state)
        logits = self.classifier(torch.cat(features, dim=1))
        if return_event_rate:
            return logits, activity_sum / timesteps
        return logits


@dataclass
class SHDSpikingTemporalConvResult:
    config: SHDConfig
    device: str
    readout_seeds: tuple[int, ...]
    validation_fraction: float
    target_parameters: int
    temporal_levels: tuple[int, ...]
    ann_channels: int
    state_channels: int
    arms: list[dict]
    records: list[dict]
    summary: list[dict]

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "shd_spiking_temporal_conv.json"
        records_path = output / "shd_spiking_temporal_conv_records.csv"
        summary_path = output / "shd_spiking_temporal_conv_summary.csv"
        payload = {
            "config": asdict(self.config), "device": self.device,
            "readout_seeds": list(self.readout_seeds), "validation_fraction": self.validation_fraction,
            "target_parameters": self.target_parameters, "temporal_levels": list(self.temporal_levels),
            "ann_channels": self.ann_channels, "state_channels": self.state_channels,
            "arms": self.arms, "records": self.records, "summary": self.summary,
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        _write_csv(records_path, self.records)
        _write_csv(summary_path, self.summary)
        paths = {"json": str(json_path), "records_csv": str(records_path), "summary_csv": str(summary_path)}
        if plot:
            plot_path = output / "shd_spiking_temporal_conv_summary.png"
            plot_shd_spiking_temporal_conv(self.summary, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def run_shd_spiking_temporal_conv(
    config: SHDConfig,
    *,
    readout_seeds: Iterable[int] = (142, 143, 144),
    validation_fraction: float = 0.10,
    target_parameters: int = 133_631,
    device="auto",
    temporal_levels: Iterable[int] = DEFAULT_TEMPORAL_LEVELS,
    projection_dim: int = 32,
    dense_lif_hidden_neurons: int = 128,
    dense_lif_projection_dim: int = 16,
    temporal_conv_kernel_size: int = 5,
    surrogate_slope: float = 10.0,
) -> SHDSpikingTemporalConvResult:
    if torch is None:
        raise ImportError("Phase 45 spiking temporal convolution requires PyTorch")
    seeds = tuple(int(seed) for seed in readout_seeds)
    levels = tuple(int(level) for level in temporal_levels)
    ann_channels, _ = matched_temporal_conv_channels(
        config.input_neurons, config.classes, target_parameters,
        kernel_size=temporal_conv_kernel_size, temporal_levels=levels,
    )
    state_channels, _ = matched_temporal_conv_state_channels(
        config.input_neurons, config.classes, target_parameters,
        kernel_size=temporal_conv_kernel_size, temporal_levels=levels,
    )
    resolved = resolve_device(device)
    all_train_events, all_train_labels, test_events, test_labels = load_shd_tensors(config)
    train_events, train_labels, validation_events, validation_labels = _stratified_split(
        all_train_events, all_train_labels,
        fraction=validation_fraction, seed=config.data_seed + 43_000,
    )
    records: list[dict] = []
    for seed in seeds:
        for arm in SHD_SPIKING_TEMPORAL_CONV_ARMS:
            seed_everything(seed, device=resolved)
            if arm.model_kind == "raw":
                model = SHDRawTemporalPyramidClassifier(
                    config, projection_dim=projection_dim,
                    temporal_levels=levels, target_parameters=target_parameters,
                ).to(resolved)
                channels = model.readout.bottleneck_units
                activity_kind = "input_event_rate"
            elif arm.model_kind == "conv_ann":
                model = TemporalConvClassifier(
                    config, channels=ann_channels,
                    kernel_size=temporal_conv_kernel_size, temporal_levels=levels,
                ).to(resolved)
                channels = ann_channels
                activity_kind = "relu_activation"
            elif arm.model_kind in {"conv_analog", "conv_lif"}:
                dynamics = "analog" if arm.model_kind == "conv_analog" else "lif"
                model = TemporalConvStateClassifier(
                    config, channels=state_channels,
                    kernel_size=temporal_conv_kernel_size,
                    temporal_levels=levels, dynamics=dynamics,
                    surrogate_slope=surrogate_slope,
                ).to(resolved)
                channels = state_channels
                activity_kind = "analog_activation" if dynamics == "analog" else "spike_rate"
            else:
                model = DenseLIFTemporalClassifier(
                    config, hidden_neurons=dense_lif_hidden_neurons,
                    temporal_levels=levels, projection_dim=dense_lif_projection_dim,
                    target_parameters=target_parameters, surrogate_slope=surrogate_slope,
                ).to(resolved)
                channels = dense_lif_hidden_neurons
                activity_kind = "hidden_spike_rate"
            training = _train_validation_selected(
                model, train_events, train_labels, validation_events, validation_labels,
                config, seed=seed, device=resolved,
            )
            final_accuracy, _, final_activity = _measure(
                model, test_events, test_labels, config.batch_size, resolved
            )
            model.load_state_dict(training["best_state"])
            checkpoint_accuracy, checkpoint_seconds, checkpoint_activity = _measure(
                model, test_events, test_labels, config.batch_size, resolved
            )
            parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            mean_leak = mean_threshold = 0.0
            if isinstance(model, TemporalConvStateClassifier):
                mean_leak = float(torch.sigmoid(model.leak_logit).mean().item())
                if model.threshold_raw is not None:
                    mean_threshold = float(torch.nn.functional.softplus(model.threshold_raw).mean().item())
            records.append(
                {
                    "seed": int(seed), "arm": arm.name, "model_kind": arm.model_kind,
                    "channels": int(channels), "best_epoch": int(training["best_epoch"]),
                    "best_validation_accuracy": float(training["best_validation_accuracy"]),
                    "final_test_accuracy": float(final_accuracy),
                    "checkpoint_test_accuracy": float(checkpoint_accuracy),
                    "checkpoint_gain_vs_final": float(checkpoint_accuracy - final_accuracy),
                    "effective_trainable_parameters": int(parameters),
                    "parameter_ratio_vs_target": float(parameters / target_parameters),
                    "final_activity": float(final_activity), "checkpoint_activity": float(checkpoint_activity),
                    "activity_kind": activity_kind, "mean_leak": mean_leak,
                    "mean_threshold": mean_threshold,
                    "train_seconds": float(training["train_seconds"]),
                    "checkpoint_test_examples_per_second": float(test_events.shape[0] / max(checkpoint_seconds, 1e-12)),
                }
            )
    _attach_comparisons(records)
    return SHDSpikingTemporalConvResult(
        config=config, device=device_kind(resolved), readout_seeds=seeds,
        validation_fraction=float(validation_fraction), target_parameters=int(target_parameters),
        temporal_levels=levels, ann_channels=int(ann_channels), state_channels=int(state_channels),
        arms=[asdict(arm) for arm in SHD_SPIKING_TEMPORAL_CONV_ARMS], records=records,
        summary=summarize_shd_spiking_temporal_conv(records),
    )


def summarize_shd_spiking_temporal_conv(records: Iterable[dict]) -> list[dict]:
    rows = list(records)
    summary: list[dict] = []
    for arm in SHD_SPIKING_TEMPORAL_CONV_ARMS:
        group = [row for row in rows if row["arm"] == arm.name]
        if not group:
            continue
        ann_gains = [float(row["checkpoint_gain_vs_conv_ann"]) for row in group]
        analog_gains = [float(row["checkpoint_gain_vs_conv_analog"]) for row in group]
        dense_lif_gains = [float(row["checkpoint_gain_vs_dense_lif"]) for row in group]
        summary.append(
            {
                "arm": arm.name, "model_kind": arm.model_kind, "runs": len(group),
                "mean_checkpoint_test_accuracy": statistics.fmean(float(row["checkpoint_test_accuracy"]) for row in group),
                "std_checkpoint_test_accuracy": statistics.pstdev(float(row["checkpoint_test_accuracy"]) for row in group),
                "mean_gain_vs_conv_ann": statistics.fmean(ann_gains),
                "within_two_points_seed_count_vs_ann": sum(gain >= -0.02 for gain in ann_gains),
                "mean_gain_vs_conv_analog": statistics.fmean(analog_gains),
                "within_two_points_seed_count_vs_analog": sum(gain >= -0.02 for gain in analog_gains),
                "mean_gain_vs_dense_lif": statistics.fmean(dense_lif_gains),
                "three_point_seed_count_vs_dense_lif": sum(gain >= 0.03 for gain in dense_lif_gains),
                "mean_checkpoint_gain_vs_final": statistics.fmean(float(row["checkpoint_gain_vs_final"]) for row in group),
                "mean_best_epoch": statistics.fmean(int(row["best_epoch"]) for row in group),
                "mean_best_validation_accuracy": statistics.fmean(float(row["best_validation_accuracy"]) for row in group),
                "channels": int(group[0]["channels"]),
                "effective_trainable_parameters": int(group[0]["effective_trainable_parameters"]),
                "parameter_ratio_vs_target": statistics.fmean(float(row["parameter_ratio_vs_target"]) for row in group),
                "mean_checkpoint_activity": statistics.fmean(float(row["checkpoint_activity"]) for row in group),
                "activity_kind": group[0]["activity_kind"],
                "mean_leak": statistics.fmean(float(row["mean_leak"]) for row in group),
                "mean_threshold": statistics.fmean(float(row["mean_threshold"]) for row in group),
                "mean_train_seconds": statistics.fmean(float(row["train_seconds"]) for row in group),
                "mean_checkpoint_test_examples_per_second": statistics.fmean(float(row["checkpoint_test_examples_per_second"]) for row in group),
            }
        )
    return summary


def plot_shd_spiking_temporal_conv(summary: list[dict], path: str | pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    labels = [row["arm"].replace("_", "\n") for row in summary]
    x = list(range(len(summary)))
    accuracy = [100.0 * float(row["mean_checkpoint_test_accuracy"]) for row in summary]
    errors = [100.0 * float(row["std_checkpoint_test_accuracy"]) for row in summary]
    ann_gain = [100.0 * float(row["mean_gain_vs_conv_ann"]) for row in summary]
    activity = [100.0 * float(row["mean_checkpoint_activity"]) for row in summary]
    colors = ("#ffb31a", "#167d55", "#8b6fd6", "#35b4f2", "#bd3d3a")
    figure, axes = plt.subplots(3, 1, figsize=(16, 13), constrained_layout=True)
    axes[0].bar(x, accuracy, yerr=errors, capsize=5, color=colors)
    axes[0].set_ylabel("Best-validation test accuracy (%)")
    axes[0].set_title("AMMC Gen-5 Phase 45: learned spiking temporal convolution")
    axes[1].bar(x, ann_gain, color=colors)
    axes[1].axhline(-2.0, color="#bd3d3a", linestyle="--", label="-2 point viability floor")
    axes[1].set_ylabel("Gain vs temporal Conv1D (points)")
    axes[1].legend()
    axes[2].bar(x, activity, color=colors)
    axes[2].set_ylabel("Mean activity (%)")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25)
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _attach_comparisons(records: list[dict]) -> None:
    lookup = {(int(row["seed"]), row["arm"]): row for row in records}
    for row in records:
        seed = int(row["seed"])
        value = float(row["checkpoint_test_accuracy"])
        row["checkpoint_gain_vs_conv_ann"] = value - float(lookup[(seed, "temporal_conv1d")]["checkpoint_test_accuracy"])
        row["checkpoint_gain_vs_conv_analog"] = value - float(lookup[(seed, "temporal_conv_leaky_analog")]["checkpoint_test_accuracy"])
        row["checkpoint_gain_vs_dense_lif"] = value - float(lookup[(seed, "dense_lif_recurrent")]["checkpoint_test_accuracy"])


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
