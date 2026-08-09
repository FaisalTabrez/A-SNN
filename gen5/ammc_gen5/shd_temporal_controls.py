"""Phase 37 causal controls for the Phase 36 SHD temporal-pyramid result.

This phase separates the contribution of a time-aware decoder, sparse sensor
expansion, and recurrent AMMC dynamics. The matched raw-event control and AMMC
pyramid use approximately the same trainable readout budget.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
import json
import pathlib
import statistics
from typing import Iterable

from .event_mnist import nn, torch
from .runtime import device_kind, resolve_device, seed_everything
from .shd_benchmark import (
    SHDConfig,
    SHDEventCountClassifier,
    SHDSparseClassifier,
    _measure,
    _train_model,
    load_shd_tensors,
)
from .shd_temporal_pyramid import (
    DEFAULT_TEMPORAL_LEVELS,
    SHDTemporalPyramidClassifier,
)


@dataclass(frozen=True)
class SHDTemporalControlArm:
    name: str
    model_kind: str


SHD_TEMPORAL_CONTROL_ARMS = (
    SHDTemporalControlArm("event_count_mlp", "event_count"),
    SHDTemporalControlArm("raw_temporal_pyramid", "raw_temporal"),
    SHDTemporalControlArm("sparse512_global", "sparse_global"),
    SHDTemporalControlArm(
        "sparse512_feedforward_pyramid", "sparse_feedforward_pyramid"
    ),
    SHDTemporalControlArm(
        "sparse512_recurrent_pyramid", "sparse_recurrent_pyramid"
    ),
)


def available_shd_temporal_control_arms() -> tuple[str, ...]:
    return tuple(arm.name for arm in SHD_TEMPORAL_CONTROL_ARMS)


def budget_matched_bottleneck(
    *,
    trace_dim: int,
    final_dim: int,
    classes: int,
    projection_dim: int,
    temporal_levels: Iterable[int],
    target_parameters: int,
) -> tuple[int, int]:
    """Return bottleneck width and actual parameters under a fixed budget."""

    levels = tuple(int(level) for level in temporal_levels)
    if trace_dim <= 0 or final_dim < 0 or classes <= 1 or projection_dim <= 0:
        raise ValueError("invalid temporal readout dimensions")
    if not levels or any(level <= 0 for level in levels):
        raise ValueError("temporal levels must be positive")
    projection_parameters = trace_dim * projection_dim + projection_dim
    feature_dim = final_dim + projection_dim * sum(levels)
    per_bottleneck = feature_dim + classes + 1
    available = target_parameters - projection_parameters - classes
    if available < per_bottleneck:
        raise ValueError("target parameter budget is too small")
    bottleneck = available // per_bottleneck
    actual = projection_parameters + bottleneck * per_bottleneck + classes
    return int(bottleneck), int(actual)


class BudgetMatchedTemporalReadout(nn.Module):
    """Shared temporal projection and MLP constrained to a target budget."""

    def __init__(
        self,
        *,
        trace_dim: int,
        final_dim: int,
        classes: int,
        projection_dim: int,
        temporal_levels: Iterable[int],
        target_parameters: int,
    ) -> None:
        if torch is None:
            raise ImportError("Phase 37 temporal controls require PyTorch")
        super().__init__()
        self.temporal_levels = tuple(int(level) for level in temporal_levels)
        bottleneck, actual = budget_matched_bottleneck(
            trace_dim=trace_dim,
            final_dim=final_dim,
            classes=classes,
            projection_dim=projection_dim,
            temporal_levels=self.temporal_levels,
            target_parameters=target_parameters,
        )
        self.projection = nn.Sequential(
            nn.Linear(trace_dim, projection_dim),
            nn.ReLU(),
        )
        self.feature_dim = final_dim + projection_dim * sum(self.temporal_levels)
        self.decoder = nn.Sequential(
            nn.Linear(self.feature_dim, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, classes),
        )
        self.bottleneck_units = int(bottleneck)
        self.actual_parameter_count = int(actual)
        self.target_parameter_count = int(target_parameters)

    def forward(self, trace, final_state):  # type: ignore[override]
        if trace.ndim != 3:
            raise ValueError("trace must have shape [batch, time, channels]")
        timesteps = int(trace.shape[1])
        features = []
        for level in self.temporal_levels:
            if level > timesteps:
                raise ValueError("temporal level cannot exceed timestep count")
            for window in range(level):
                start = window * timesteps // level
                stop = (window + 1) * timesteps // level
                features.append(self.projection(trace[:, start:stop].mean(dim=1)))
        features.append(final_state)
        return self.decoder(torch.cat(features, dim=1))


class SHDRawTemporalPyramidClassifier(nn.Module):
    """Time-aware non-spiking control over the original binned SHD events."""

    def __init__(
        self,
        config: SHDConfig,
        *,
        projection_dim: int,
        temporal_levels: Iterable[int],
        target_parameters: int,
    ) -> None:
        if torch is None:
            raise ImportError("Phase 37 temporal controls require PyTorch")
        super().__init__()
        self.config = config
        self.readout = BudgetMatchedTemporalReadout(
            trace_dim=config.input_neurons,
            final_dim=config.input_neurons,
            classes=config.classes,
            projection_dim=projection_dim,
            temporal_levels=temporal_levels,
            target_parameters=target_parameters,
        )

    def forward(self, events, *, return_event_rate: bool = False):  # type: ignore[override]
        if events.ndim != 3 or events.shape[2] != self.config.input_neurons:
            raise ValueError("events must have shape [batch, time, input_neurons]")
        trace = events.to(torch.float32)
        final_state = torch.zeros_like(trace[:, 0])
        for step in range(trace.shape[1]):
            final_state = final_state * self.config.reservoir_leak + trace[:, step]
        final_state = final_state / final_state.amax(dim=1, keepdim=True).clamp_min(1.0)
        logits = self.readout(trace, final_state)
        if return_event_rate:
            return logits, trace.mean()
        return logits


@dataclass
class SHDTemporalControlsResult:
    config: SHDConfig
    device: str
    hidden_neurons: int
    temporal_levels: tuple[int, ...]
    projection_dim: int
    target_readout_parameters: int
    arms: list[dict]
    records: list[dict]
    summary: list[dict]

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "shd_temporal_controls.json"
        records_path = output / "shd_temporal_controls_records.csv"
        summary_path = output / "shd_temporal_controls_summary.csv"
        payload = {
            "config": asdict(self.config),
            "device": self.device,
            "hidden_neurons": self.hidden_neurons,
            "temporal_levels": list(self.temporal_levels),
            "projection_dim": self.projection_dim,
            "target_readout_parameters": self.target_readout_parameters,
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
            plot_path = output / "shd_temporal_controls_summary.png"
            plot_shd_temporal_controls(self.summary, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def run_shd_temporal_controls(
    config: SHDConfig,
    *,
    hidden_neurons: int = 512,
    device="auto",
    surrogate_slope: float = 10.0,
    projection_dim: int = 32,
    temporal_levels: Iterable[int] = DEFAULT_TEMPORAL_LEVELS,
    readout_hidden_units: int = 128,
    ltw_minimum: float = 0.0,
    ltw_maximum: float = 1.0,
) -> SHDTemporalControlsResult:
    if torch is None:
        raise ImportError("Phase 37 temporal controls require PyTorch")
    if hidden_neurons <= 1 or readout_hidden_units <= 0:
        raise ValueError("invalid hidden/readout dimensions")
    levels = tuple(int(level) for level in temporal_levels)
    if not levels or any(level <= 0 or level > config.timesteps for level in levels):
        raise ValueError("temporal levels must be between one and timesteps")
    target_readout_parameters = (
        2 * hidden_neurons * readout_hidden_units
        + readout_hidden_units
        + readout_hidden_units * config.classes
        + config.classes
    )
    required_edges = (
        config.input_neurons * config.sensor_fanout
        + hidden_neurons * config.recurrent_fanout
    )
    arm_config = replace(
        config,
        hidden_neurons=hidden_neurons,
        max_edges=max(config.max_edges, _next_power_of_two(required_edges)),
    )
    resolved = resolve_device(device)
    train_events, train_labels, test_events, test_labels = load_shd_tensors(config)
    records: list[dict] = []
    for seed in config.seeds:
        for arm in SHD_TEMPORAL_CONTROL_ARMS:
            seed_everything(seed, device=resolved)
            if arm.model_kind == "event_count":
                model = SHDEventCountClassifier(arm_config, kind="mlp").to(resolved)
                feature_dim = config.input_neurons
                bottleneck_units = config.count_hidden_units
                topology = "raw_count"
            elif arm.model_kind == "raw_temporal":
                model = SHDRawTemporalPyramidClassifier(
                    arm_config,
                    projection_dim=projection_dim,
                    temporal_levels=levels,
                    target_parameters=target_readout_parameters,
                ).to(resolved)
                feature_dim = model.readout.feature_dim
                bottleneck_units = model.readout.bottleneck_units
                topology = "raw_temporal"
            elif arm.model_kind == "sparse_global":
                model = SHDSparseClassifier(
                    arm_config,
                    seed=seed,
                    delay_pattern="none",
                    max_delay_steps=0,
                    surrogate_slope=surrogate_slope,
                    readout_kind="mlp",
                    readout_hidden_units=readout_hidden_units,
                    device=resolved,
                ).to(resolved)
                feature_dim = 2 * hidden_neurons
                bottleneck_units = readout_hidden_units
                topology = "recurrent_global"
            else:
                model = SHDTemporalPyramidClassifier(
                    arm_config,
                    seed=seed,
                    surrogate_slope=surrogate_slope,
                    projection_dim=projection_dim,
                    temporal_levels=levels,
                    baseline_hidden_units=readout_hidden_units,
                    temporal_order="ordered",
                    device=resolved,
                ).to(resolved)
                feature_dim = model.readout.feature_dim
                bottleneck_units = model.readout.bottleneck_units
                if arm.model_kind == "sparse_feedforward_pyramid":
                    disable_shd_recurrent_edges(model)
                    topology = "feedforward_pyramid"
                else:
                    topology = "recurrent_pyramid"
            initial_ltw = None
            if isinstance(model, SHDSparseClassifier):
                initial_ltw = model.graph.long_term_weight.detach().clone()
            _, _, initial_event_rate = _measure(
                model, test_events, test_labels, config.batch_size, resolved
            )
            train_seconds = _train_model(
                model,
                train_events,
                train_labels,
                arm_config,
                seed=seed,
                device=resolved,
                ltw_minimum=ltw_minimum,
                ltw_maximum=ltw_maximum,
            )
            train_accuracy, _, _ = _measure(
                model, train_events, train_labels, config.batch_size, resolved
            )
            test_accuracy, inference_seconds, final_event_rate = _measure(
                model, test_events, test_labels, config.batch_size, resolved
            )
            active_edges = recurrent_edges = 0
            mean_ltw_change = lower_saturation = upper_saturation = 0.0
            if isinstance(model, SHDSparseClassifier):
                active = model.graph.active_mask
                recurrent = active & (model.graph.sources >= config.input_neurons)
                active_edges = int(active.sum().item())
                recurrent_edges = int(recurrent.sum().item())
                final_ltw = model.graph.long_term_weight.detach()
                mean_ltw_change = float(
                    (final_ltw[active] - initial_ltw[active]).abs().mean().item()
                )
                lower_saturation = float(
                    (final_ltw[active] <= ltw_minimum + 1e-6)
                    .to(torch.float32).mean().item()
                )
                upper_saturation = float(
                    (final_ltw[active] >= ltw_maximum - 1e-6)
                    .to(torch.float32).mean().item()
                )
                trainable_head_parameters = sum(
                    parameter.numel() for parameter in model.readout.parameters()
                )
                effective_parameters = trainable_head_parameters + active_edges
            else:
                trainable_head_parameters = sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                )
                effective_parameters = trainable_head_parameters
            allocated_parameters = sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            records.append(
                {
                    "seed": int(seed),
                    "arm": arm.name,
                    "model_kind": arm.model_kind,
                    "topology": topology,
                    "hidden_neurons": int(
                        hidden_neurons if isinstance(model, SHDSparseClassifier) else 0
                    ),
                    "feature_dim": int(feature_dim),
                    "readout_bottleneck_units": int(bottleneck_units),
                    "train_accuracy": float(train_accuracy),
                    "test_accuracy": float(test_accuracy),
                    "active_edges": int(active_edges),
                    "active_recurrent_edges": int(recurrent_edges),
                    "trainable_head_parameters": int(trainable_head_parameters),
                    "effective_trainable_parameters": int(effective_parameters),
                    "allocated_trainable_parameters": int(allocated_parameters),
                    "initial_event_rate": float(initial_event_rate),
                    "final_event_rate": float(final_event_rate),
                    "event_rate_kind": (
                        "hidden" if isinstance(model, SHDSparseClassifier) else "input"
                    ),
                    "mean_absolute_ltw_change": float(mean_ltw_change),
                    "lower_ltw_saturation_rate": float(lower_saturation),
                    "upper_ltw_saturation_rate": float(upper_saturation),
                    "train_seconds": float(train_seconds),
                    "inference_seconds": float(inference_seconds),
                    "test_examples_per_second": float(
                        test_events.shape[0] / max(inference_seconds, 1e-12)
                    ),
                }
            )
    _attach_control_comparisons(records)
    return SHDTemporalControlsResult(
        config=config,
        device=device_kind(resolved),
        hidden_neurons=int(hidden_neurons),
        temporal_levels=levels,
        projection_dim=int(projection_dim),
        target_readout_parameters=int(target_readout_parameters),
        arms=[asdict(arm) for arm in SHD_TEMPORAL_CONTROL_ARMS],
        records=records,
        summary=summarize_shd_temporal_controls(records),
    )


def disable_shd_recurrent_edges(model: SHDSparseClassifier) -> int:
    """Deactivate hidden-source edges while preserving sensor projections."""

    recurrent = model.graph.active_mask & (
        model.graph.sources >= model.config.input_neurons
    )
    count = int(recurrent.sum().item())
    with torch.no_grad():
        model.graph.active_mask[recurrent] = False
        model.graph.short_term_weight[recurrent] = 0.0
        model.graph.long_term_weight[recurrent] = 0.0
    return count


def summarize_shd_temporal_controls(records: Iterable[dict]) -> list[dict]:
    rows = list(records)
    summary: list[dict] = []
    for arm in SHD_TEMPORAL_CONTROL_ARMS:
        group = [row for row in rows if row["arm"] == arm.name]
        if not group:
            continue
        recurrence_gains = [
            float(row["recurrence_gain_vs_feedforward"]) for row in group
        ]
        raw_gains = [float(row["gain_vs_raw_temporal"]) for row in group]
        summary.append(
            {
                "arm": arm.name,
                "model_kind": arm.model_kind,
                "topology": group[0]["topology"],
                "seeds": len(group),
                "mean_test_accuracy": statistics.fmean(
                    float(row["test_accuracy"]) for row in group
                ),
                "std_test_accuracy": statistics.pstdev(
                    float(row["test_accuracy"]) for row in group
                ),
                "mean_gain_vs_event_count": statistics.fmean(
                    float(row["gain_vs_event_count"]) for row in group
                ),
                "mean_gain_vs_raw_temporal": statistics.fmean(raw_gains),
                "improved_seed_count_vs_raw": sum(gain > 0 for gain in raw_gains),
                "one_point_seed_count_vs_raw": sum(gain >= 0.01 for gain in raw_gains),
                "mean_gain_vs_sparse_global": statistics.fmean(
                    float(row["gain_vs_sparse_global"]) for row in group
                ),
                "mean_recurrence_gain_vs_feedforward": statistics.fmean(
                    recurrence_gains
                ),
                "improved_seed_count_recurrence": sum(
                    gain > 0 for gain in recurrence_gains
                ),
                "two_point_seed_count_recurrence": sum(
                    gain >= 0.02 for gain in recurrence_gains
                ),
                "active_edges": int(group[0]["active_edges"]),
                "active_recurrent_edges": int(group[0]["active_recurrent_edges"]),
                "feature_dim": int(group[0]["feature_dim"]),
                "effective_trainable_parameters": int(
                    group[0]["effective_trainable_parameters"]
                ),
                "parameter_ratio_vs_recurrent_pyramid": statistics.fmean(
                    float(row["parameter_ratio_vs_recurrent_pyramid"])
                    for row in group
                ),
                "mean_final_event_rate": statistics.fmean(
                    float(row["final_event_rate"]) for row in group
                ),
                "event_rate_kind": group[0]["event_rate_kind"],
                "mean_absolute_ltw_change": statistics.fmean(
                    float(row["mean_absolute_ltw_change"]) for row in group
                ),
                "mean_upper_ltw_saturation_rate": statistics.fmean(
                    float(row["upper_ltw_saturation_rate"]) for row in group
                ),
                "mean_train_seconds": statistics.fmean(
                    float(row["train_seconds"]) for row in group
                ),
                "mean_test_examples_per_second": statistics.fmean(
                    float(row["test_examples_per_second"]) for row in group
                ),
            }
        )
    return summary


def plot_shd_temporal_controls(summary: list[dict], path: str | pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    labels = [row["arm"].replace("_", "\n") for row in summary]
    accuracy = [100.0 * float(row["mean_test_accuracy"]) for row in summary]
    errors = [100.0 * float(row["std_test_accuracy"]) for row in summary]
    raw_gains = [100.0 * float(row["mean_gain_vs_raw_temporal"]) for row in summary]
    parameter_ratio = [
        100.0 * float(row["parameter_ratio_vs_recurrent_pyramid"])
        for row in summary
    ]
    colors = ("#8b6fd6", "#ffb31a", "#35b4f2", "#48c78e", "#167d55")
    x = list(range(len(summary)))
    figure, axes = plt.subplots(3, 1, figsize=(15, 13), constrained_layout=True)
    axes[0].bar(x, accuracy, yerr=errors, capsize=5, color=colors)
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_title("AMMC Gen-5 Phase 37: SHD temporal-control decomposition")
    axes[1].bar(x, raw_gains, color=colors)
    axes[1].axhline(2.0, color="#bd3d3a", linestyle="--", label="+2 point reservoir gate")
    axes[1].set_ylabel("Gain vs raw temporal (points)")
    axes[1].legend()
    axes[2].bar(x, parameter_ratio, color=colors)
    axes[2].axhline(100.0, color="#222222", linestyle="--")
    axes[2].set_ylabel("Parameters vs recurrent pyramid (%)")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25)
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _attach_control_comparisons(records: list[dict]) -> None:
    by_seed_arm = {(int(row["seed"]), row["arm"]): row for row in records}
    for row in records:
        seed = int(row["seed"])
        count = by_seed_arm[(seed, "event_count_mlp")]
        raw = by_seed_arm[(seed, "raw_temporal_pyramid")]
        global_control = by_seed_arm[(seed, "sparse512_global")]
        feedforward = by_seed_arm[(seed, "sparse512_feedforward_pyramid")]
        recurrent = by_seed_arm[(seed, "sparse512_recurrent_pyramid")]
        row["gain_vs_event_count"] = float(row["test_accuracy"]) - float(
            count["test_accuracy"]
        )
        row["gain_vs_raw_temporal"] = float(row["test_accuracy"]) - float(
            raw["test_accuracy"]
        )
        row["gain_vs_sparse_global"] = float(row["test_accuracy"]) - float(
            global_control["test_accuracy"]
        )
        if row["arm"] == "sparse512_recurrent_pyramid":
            row["recurrence_gain_vs_feedforward"] = float(
                row["test_accuracy"]
            ) - float(feedforward["test_accuracy"])
        else:
            row["recurrence_gain_vs_feedforward"] = 0.0
        row["parameter_ratio_vs_recurrent_pyramid"] = float(
            row["effective_trainable_parameters"]
        ) / max(float(recurrent["effective_trainable_parameters"]), 1.0)


def _next_power_of_two(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
