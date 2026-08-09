"""Phase 46 diagnostic for temporal-state placement on SHD."""

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
from .shd_spiking_temporal_conv import (
    TemporalConvStateClassifier,
    matched_temporal_conv_state_channels,
)
from .shd_temporal_pyramid import DEFAULT_TEMPORAL_LEVELS
from .shd_validation_checkpoint import _stratified_split, _train_validation_selected
from .trainable_temporal_mnist import SurrogateSpike


@dataclass(frozen=True)
class SHDStatePlacementArm:
    name: str
    dynamics: str
    feature_source: str


SHD_STATE_PLACEMENT_ARMS = (
    SHDStatePlacementArm("temporal_conv1d", "none", "direct"),
    SHDStatePlacementArm("leaky_analog_state_only", "analog", "state_only"),
    SHDStatePlacementArm("leaky_lif_state_only", "lif", "state_only"),
    SHDStatePlacementArm("leaky_analog_residual", "analog", "direct_plus_state"),
    SHDStatePlacementArm("leaky_lif_residual", "lif", "direct_plus_state"),
)


def available_shd_state_placement_arms() -> tuple[str, ...]:
    return tuple(arm.name for arm in SHD_STATE_PLACEMENT_ARMS)


def temporal_conv_residual_parameter_count(
    input_neurons: int,
    channels: int,
    classes: int,
    *,
    kernel_size: int,
    temporal_levels: Iterable[int],
    spiking: bool,
) -> int:
    pooled_features = 2 * sum(int(level) for level in temporal_levels) + 1
    state_parameters = channels * (2 if spiking else 1)
    return int(
        channels * input_neurons * kernel_size
        + channels
        + channels * pooled_features * classes
        + classes
        + state_parameters
    )


def matched_temporal_conv_residual_channels(
    input_neurons: int,
    classes: int,
    target_parameters: int,
    *,
    kernel_size: int,
    temporal_levels: Iterable[int],
) -> tuple[int, int]:
    channels = 1
    while temporal_conv_residual_parameter_count(
        input_neurons,
        channels + 1,
        classes,
        kernel_size=kernel_size,
        temporal_levels=temporal_levels,
        spiking=True,
    ) <= target_parameters:
        channels += 1
    return channels, temporal_conv_residual_parameter_count(
        input_neurons,
        channels,
        classes,
        kernel_size=kernel_size,
        temporal_levels=temporal_levels,
        spiking=True,
    )


class ResidualTemporalConvStateClassifier(nn.Module):
    """Preserve direct Conv1D features beside an analog or spiking state."""

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
            raise ImportError("Phase 46 state-placement diagnostic requires PyTorch")
        if dynamics not in {"analog", "lif"}:
            raise ValueError("dynamics must be analog or lif")
        super().__init__()
        self.config = config
        self.channels = int(channels)
        self.temporal_levels = tuple(int(level) for level in temporal_levels)
        self.dynamics = dynamics
        self.surrogate_slope = float(surrogate_slope)
        self.ablation_mode = "full"
        self.temporal = nn.Conv1d(
            config.input_neurons,
            channels,
            kernel_size,
            padding=kernel_size // 2,
        )
        leak_logit = math.log(initial_leak / (1.0 - initial_leak))
        self.leak_logit = nn.Parameter(torch.full((channels,), leak_logit))
        if dynamics == "lif":
            threshold_raw = math.log(math.expm1(initial_threshold))
            self.threshold_raw = nn.Parameter(torch.full((channels,), threshold_raw))
        else:
            self.register_parameter("threshold_raw", None)
        pooled_features = 2 * sum(self.temporal_levels) + 1
        self.classifier = nn.Linear(channels * pooled_features, config.classes)

    def forward(self, events, *, return_event_rate: bool = False):  # type: ignore[override]
        if events.ndim != 3 or events.shape[2] != self.config.input_neurons:
            raise ValueError("events must have shape [batch, time, input_neurons]")
        currents = self.temporal(events.to(torch.float32).transpose(1, 2)).transpose(1, 2)
        direct_trace = torch.relu(currents)
        leak = torch.sigmoid(self.leak_logit)
        membrane = currents.new_zeros((events.shape[0], self.channels))
        state_trace = []
        activity_sum = currents.new_zeros(())
        threshold = None
        if self.dynamics == "lif":
            threshold = torch.nn.functional.softplus(self.threshold_raw).clamp_min(1e-3)
        for step in range(currents.shape[1]):
            pre_reset = leak * membrane + currents[:, step]
            if self.dynamics == "lif":
                state = SurrogateSpike.apply(pre_reset - threshold, self.surrogate_slope)
                membrane = pre_reset - state * threshold
                activity_sum = activity_sum + state.mean()
            else:
                membrane = pre_reset
                state = torch.tanh(membrane)
                activity_sum = activity_sum + state.abs().mean()
            state_trace.append(state)
        stacked_state = torch.stack(state_trace, dim=1)
        final_state = torch.tanh(membrane) if self.dynamics == "analog" else membrane / threshold
        if self.ablation_mode == "direct_only":
            stacked_state = torch.zeros_like(stacked_state)
            final_state = torch.zeros_like(final_state)
        elif self.ablation_mode == "state_only":
            direct_trace = torch.zeros_like(direct_trace)
        elif self.ablation_mode == "shuffled_state":
            stacked_state = torch.roll(stacked_state, shifts=1, dims=0)
            final_state = torch.roll(final_state, shifts=1, dims=0)
        features = _multiscale_features(direct_trace, self.temporal_levels)
        features.extend(_multiscale_features(stacked_state, self.temporal_levels))
        features.append(final_state)
        logits = self.classifier(torch.cat(features, dim=1))
        if return_event_rate:
            return logits, activity_sum / int(currents.shape[1])
        return logits

    def set_ablation_mode(self, mode: str) -> None:
        if mode not in {"full", "direct_only", "state_only", "shuffled_state"}:
            raise ValueError("unsupported residual-state ablation mode")
        self.ablation_mode = mode


@dataclass
class SHDStatePlacementResult:
    config: SHDConfig
    device: str
    readout_seeds: tuple[int, ...]
    validation_fraction: float
    target_parameters: int
    temporal_levels: tuple[int, ...]
    direct_channels: int
    state_only_channels: int
    residual_channels: int
    arms: list[dict]
    records: list[dict]
    summary: list[dict]

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "shd_state_placement_diagnostic.json"
        records_path = output / "shd_state_placement_diagnostic_records.csv"
        summary_path = output / "shd_state_placement_diagnostic_summary.csv"
        payload = {
            "config": asdict(self.config),
            "device": self.device,
            "readout_seeds": list(self.readout_seeds),
            "validation_fraction": self.validation_fraction,
            "target_parameters": self.target_parameters,
            "temporal_levels": list(self.temporal_levels),
            "direct_channels": self.direct_channels,
            "state_only_channels": self.state_only_channels,
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
            plot_path = output / "shd_state_placement_diagnostic_summary.png"
            plot_shd_state_placement_diagnostic(self.summary, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def run_shd_state_placement_diagnostic(
    config: SHDConfig,
    *,
    readout_seeds: Iterable[int] = (142, 143, 144),
    validation_fraction: float = 0.10,
    target_parameters: int = 133_631,
    device="auto",
    temporal_levels: Iterable[int] = DEFAULT_TEMPORAL_LEVELS,
    temporal_conv_kernel_size: int = 5,
    surrogate_slope: float = 10.0,
) -> SHDStatePlacementResult:
    if torch is None:
        raise ImportError("Phase 46 state-placement diagnostic requires PyTorch")
    seeds = tuple(int(seed) for seed in readout_seeds)
    levels = tuple(int(level) for level in temporal_levels)
    direct_channels, _ = matched_temporal_conv_channels(
        config.input_neurons,
        config.classes,
        target_parameters,
        kernel_size=temporal_conv_kernel_size,
        temporal_levels=levels,
    )
    state_channels, _ = matched_temporal_conv_state_channels(
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
        for arm in SHD_STATE_PLACEMENT_ARMS:
            seed_everything(seed, device=resolved)
            if arm.feature_source == "direct":
                model = TemporalConvClassifier(
                    config,
                    channels=direct_channels,
                    kernel_size=temporal_conv_kernel_size,
                    temporal_levels=levels,
                ).to(resolved)
                channels = direct_channels
                activity_kind = "relu_activation"
            elif arm.feature_source == "state_only":
                model = TemporalConvStateClassifier(
                    config,
                    channels=state_channels,
                    kernel_size=temporal_conv_kernel_size,
                    temporal_levels=levels,
                    dynamics=arm.dynamics,
                    surrogate_slope=surrogate_slope,
                ).to(resolved)
                channels = state_channels
                activity_kind = "spike_rate" if arm.dynamics == "lif" else "analog_activation"
            else:
                model = ResidualTemporalConvStateClassifier(
                    config,
                    channels=residual_channels,
                    kernel_size=temporal_conv_kernel_size,
                    temporal_levels=levels,
                    dynamics=arm.dynamics,
                    surrogate_slope=surrogate_slope,
                ).to(resolved)
                channels = residual_channels
                activity_kind = "spike_rate" if arm.dynamics == "lif" else "analog_activation"
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
            final_accuracy, _, final_activity = _measure(
                model, test_events, test_labels, config.batch_size, resolved
            )
            model.load_state_dict(training["best_state"])
            checkpoint_accuracy, checkpoint_seconds, checkpoint_activity = _measure(
                model, test_events, test_labels, config.batch_size, resolved
            )
            parameters = sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            )
            mean_leak = mean_threshold = 0.0
            if hasattr(model, "leak_logit"):
                mean_leak = float(torch.sigmoid(model.leak_logit).mean().item())
            if getattr(model, "threshold_raw", None) is not None:
                mean_threshold = float(
                    torch.nn.functional.softplus(model.threshold_raw).mean().item()
                )
            records.append(
                {
                    "seed": int(seed),
                    "arm": arm.name,
                    "dynamics": arm.dynamics,
                    "feature_source": arm.feature_source,
                    "channels": int(channels),
                    "best_epoch": int(training["best_epoch"]),
                    "best_validation_accuracy": float(training["best_validation_accuracy"]),
                    "final_test_accuracy": float(final_accuracy),
                    "checkpoint_test_accuracy": float(checkpoint_accuracy),
                    "checkpoint_gain_vs_final": float(checkpoint_accuracy - final_accuracy),
                    "effective_trainable_parameters": int(parameters),
                    "parameter_ratio_vs_target": float(parameters / target_parameters),
                    "final_activity": float(final_activity),
                    "checkpoint_activity": float(checkpoint_activity),
                    "activity_kind": activity_kind,
                    "mean_leak": mean_leak,
                    "mean_threshold": mean_threshold,
                    "train_seconds": float(training["train_seconds"]),
                    "checkpoint_test_examples_per_second": float(
                        test_events.shape[0] / max(checkpoint_seconds, 1e-12)
                    ),
                }
            )
    _attach_comparisons(records)
    return SHDStatePlacementResult(
        config=config,
        device=device_kind(resolved),
        readout_seeds=seeds,
        validation_fraction=float(validation_fraction),
        target_parameters=int(target_parameters),
        temporal_levels=levels,
        direct_channels=int(direct_channels),
        state_only_channels=int(state_channels),
        residual_channels=int(residual_channels),
        arms=[asdict(arm) for arm in SHD_STATE_PLACEMENT_ARMS],
        records=records,
        summary=summarize_shd_state_placement_diagnostic(records),
    )


def summarize_shd_state_placement_diagnostic(records: Iterable[dict]) -> list[dict]:
    rows = list(records)
    summary: list[dict] = []
    for arm in SHD_STATE_PLACEMENT_ARMS:
        group = [row for row in rows if row["arm"] == arm.name]
        if not group:
            continue
        conv_gains = [float(row["checkpoint_gain_vs_conv"]) for row in group]
        state_gains = [float(row["checkpoint_gain_vs_state_only"]) for row in group]
        summary.append(
            {
                "arm": arm.name,
                "dynamics": arm.dynamics,
                "feature_source": arm.feature_source,
                "runs": len(group),
                "mean_checkpoint_test_accuracy": statistics.fmean(
                    float(row["checkpoint_test_accuracy"]) for row in group
                ),
                "std_checkpoint_test_accuracy": statistics.pstdev(
                    float(row["checkpoint_test_accuracy"]) for row in group
                ),
                "mean_gain_vs_conv": statistics.fmean(conv_gains),
                "within_two_points_seed_count_vs_conv": sum(gain >= -0.02 for gain in conv_gains),
                "mean_gain_vs_state_only": statistics.fmean(state_gains),
                "four_point_seed_count_vs_state_only": sum(gain >= 0.04 for gain in state_gains),
                "mean_checkpoint_gain_vs_final": statistics.fmean(
                    float(row["checkpoint_gain_vs_final"]) for row in group
                ),
                "mean_best_validation_accuracy": statistics.fmean(
                    float(row["best_validation_accuracy"]) for row in group
                ),
                "channels": int(group[0]["channels"]),
                "effective_trainable_parameters": int(group[0]["effective_trainable_parameters"]),
                "parameter_ratio_vs_target": statistics.fmean(
                    float(row["parameter_ratio_vs_target"]) for row in group
                ),
                "mean_checkpoint_activity": statistics.fmean(
                    float(row["checkpoint_activity"]) for row in group
                ),
                "activity_kind": group[0]["activity_kind"],
                "mean_leak": statistics.fmean(float(row["mean_leak"]) for row in group),
                "mean_threshold": statistics.fmean(
                    float(row["mean_threshold"]) for row in group
                ),
                "mean_train_seconds": statistics.fmean(
                    float(row["train_seconds"]) for row in group
                ),
                "mean_checkpoint_test_examples_per_second": statistics.fmean(
                    float(row["checkpoint_test_examples_per_second"]) for row in group
                ),
            }
        )
    return summary


def plot_shd_state_placement_diagnostic(
    summary: list[dict], path: str | pathlib.Path
) -> None:
    import matplotlib.pyplot as plt

    labels = [row["arm"].replace("_", "\n") for row in summary]
    x = list(range(len(summary)))
    accuracy = [100.0 * float(row["mean_checkpoint_test_accuracy"]) for row in summary]
    errors = [100.0 * float(row["std_checkpoint_test_accuracy"]) for row in summary]
    conv_gain = [100.0 * float(row["mean_gain_vs_conv"]) for row in summary]
    state_gain = [100.0 * float(row["mean_gain_vs_state_only"]) for row in summary]
    colors = ("#167d55", "#8b6fd6", "#35b4f2", "#ffb31a", "#bd3d3a")
    figure, axes = plt.subplots(3, 1, figsize=(16, 13), constrained_layout=True)
    axes[0].bar(x, accuracy, yerr=errors, capsize=5, color=colors)
    axes[0].set_ylabel("Best-validation test accuracy (%)")
    axes[0].set_title("AMMC Gen-5 Phase 46: SHD temporal-state placement")
    axes[1].bar(x, conv_gain, color=colors)
    axes[1].axhline(-2.0, color="#bd3d3a", linestyle="--", label="-2 point viability floor")
    axes[1].set_ylabel("Gain vs Conv1D (points)")
    axes[1].legend()
    axes[2].bar(x, state_gain, color=colors)
    axes[2].axhline(4.0, color="#167d55", linestyle="--", label="+4 point recovery gate")
    axes[2].set_ylabel("Gain vs matching state-only arm (points)")
    axes[2].legend()
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25)
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _multiscale_features(trace, levels: Iterable[int]) -> list:
    timesteps = int(trace.shape[1])
    features = []
    for level in levels:
        for window in range(int(level)):
            start = window * timesteps // int(level)
            stop = (window + 1) * timesteps // int(level)
            features.append(trace[:, start:stop].mean(dim=1))
    return features


def _attach_comparisons(records: list[dict]) -> None:
    lookup = {(int(row["seed"]), row["arm"]): row for row in records}
    for row in records:
        seed = int(row["seed"])
        value = float(row["checkpoint_test_accuracy"])
        conv = float(lookup[(seed, "temporal_conv1d")]["checkpoint_test_accuracy"])
        row["checkpoint_gain_vs_conv"] = value - conv
        if row["dynamics"] == "analog":
            state_name = "leaky_analog_state_only"
        elif row["dynamics"] == "lif":
            state_name = "leaky_lif_state_only"
        else:
            state_name = row["arm"]
        state_value = float(lookup[(seed, state_name)]["checkpoint_test_accuracy"])
        row["checkpoint_gain_vs_state_only"] = value - state_value


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
