"""Phase 36 parameter-matched temporal-pyramid readouts on SHD.

The established SHD classifier collapses all hidden spikes into one global
mean before decoding. This experiment preserves coarse temporal location with
a multi-scale 1/2/4/8-window pyramid and includes a fixed time-shuffle control
to test whether any gain depends on natural event order.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
import json
import pathlib
import statistics
from typing import Iterable

from .delayed_sequential_mnist import delayed_sparse_current
from .event_mnist import nn, torch
from .runtime import device_kind, resolve_device, seed_everything
from .shd_benchmark import (
    SHDConfig,
    SHDSparseClassifier,
    _measure,
    _train_model,
    load_shd_tensors,
)
from .trainable_temporal_mnist import SurrogateSpike


DEFAULT_TEMPORAL_LEVELS = (1, 2, 4, 8)


@dataclass(frozen=True)
class SHDTemporalPyramidArm:
    name: str
    hidden_neurons: int
    readout_mode: str
    temporal_order: str = "ordered"


def default_shd_temporal_pyramid_arms() -> tuple[SHDTemporalPyramidArm, ...]:
    """Return the registered parameter-matched Phase 36 comparison matrix."""

    return (
        SHDTemporalPyramidArm("sparse256_global", 256, "global"),
        SHDTemporalPyramidArm(
            "sparse256_pyramid1248_ordered", 256, "pyramid", "ordered"
        ),
        SHDTemporalPyramidArm("sparse512_global", 512, "global"),
        SHDTemporalPyramidArm(
            "sparse512_pyramid1248_ordered", 512, "pyramid", "ordered"
        ),
        SHDTemporalPyramidArm(
            "sparse512_pyramid1248_shuffled", 512, "pyramid", "fixed_shuffle"
        ),
    )


def parameter_matched_bottleneck(
    *,
    hidden_neurons: int,
    classes: int,
    projection_dim: int,
    temporal_levels: Iterable[int],
    baseline_hidden_units: int,
) -> tuple[int, int, int]:
    """Choose a pyramid bottleneck that does not exceed the global MLP budget.

    Returns ``(bottleneck, actual_parameters, baseline_parameters)``. The graph
    edge weights are excluded because they are identical within each width.
    """

    levels = tuple(int(level) for level in temporal_levels)
    if hidden_neurons <= 0 or classes <= 1 or projection_dim <= 0:
        raise ValueError("invalid readout dimensions")
    if not levels or any(level <= 0 for level in levels):
        raise ValueError("temporal levels must be positive")
    if baseline_hidden_units <= 0:
        raise ValueError("baseline hidden units must be positive")
    baseline_input = 2 * hidden_neurons
    baseline_parameters = (
        baseline_input * baseline_hidden_units
        + baseline_hidden_units
        + baseline_hidden_units * classes
        + classes
    )
    projection_parameters = hidden_neurons * projection_dim + projection_dim
    pyramid_input = hidden_neurons + projection_dim * sum(levels)
    per_bottleneck = pyramid_input + classes + 1
    available = baseline_parameters - projection_parameters - classes
    bottleneck = max(1, available // per_bottleneck)
    actual_parameters = (
        projection_parameters + bottleneck * per_bottleneck + classes
    )
    return int(bottleneck), int(actual_parameters), int(baseline_parameters)


class TemporalPyramidReadout(nn.Module):
    """Shared projection over multi-scale temporal windows plus an MLP head."""

    def __init__(
        self,
        *,
        hidden_neurons: int,
        classes: int,
        projection_dim: int,
        temporal_levels: Iterable[int],
        baseline_hidden_units: int,
    ) -> None:
        if torch is None:
            raise ImportError("Phase 36 temporal pyramid requires PyTorch")
        super().__init__()
        self.temporal_levels = tuple(int(level) for level in temporal_levels)
        bottleneck, actual, baseline = parameter_matched_bottleneck(
            hidden_neurons=hidden_neurons,
            classes=classes,
            projection_dim=projection_dim,
            temporal_levels=self.temporal_levels,
            baseline_hidden_units=baseline_hidden_units,
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_neurons, projection_dim),
            nn.ReLU(),
        )
        feature_dim = hidden_neurons + projection_dim * sum(self.temporal_levels)
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, classes),
        )
        self.feature_dim = int(feature_dim)
        self.bottleneck_units = int(bottleneck)
        self.actual_parameter_count = int(actual)
        self.baseline_parameter_count = int(baseline)

    def forward(self, hidden_trace, final_membrane):  # type: ignore[override]
        if hidden_trace.ndim != 3:
            raise ValueError("hidden trace must have shape [batch, time, hidden]")
        timesteps = int(hidden_trace.shape[1])
        features = []
        for level in self.temporal_levels:
            if level > timesteps:
                raise ValueError("temporal level cannot exceed the timestep count")
            for window in range(level):
                start = window * timesteps // level
                stop = (window + 1) * timesteps // level
                pooled = hidden_trace[:, start:stop, :].mean(dim=1)
                features.append(self.projection(pooled))
        features.append(final_membrane)
        return self.decoder(torch.cat(features, dim=1))


class SHDTemporalPyramidClassifier(SHDSparseClassifier):
    """No-delay SHD reservoir with a temporally ordered pyramid readout."""

    def __init__(
        self,
        config: SHDConfig,
        *,
        seed: int,
        surrogate_slope: float,
        projection_dim: int,
        temporal_levels: Iterable[int],
        baseline_hidden_units: int,
        temporal_order: str,
        device,
    ) -> None:
        if temporal_order not in {"ordered", "fixed_shuffle"}:
            raise ValueError("temporal order must be ordered or fixed_shuffle")
        super().__init__(
            config,
            seed=seed,
            delay_pattern="none",
            max_delay_steps=0,
            surrogate_slope=surrogate_slope,
            readout_kind="mlp",
            readout_hidden_units=baseline_hidden_units,
            device=device,
        )
        self.readout = TemporalPyramidReadout(
            hidden_neurons=config.hidden_neurons,
            classes=config.classes,
            projection_dim=projection_dim,
            temporal_levels=temporal_levels,
            baseline_hidden_units=baseline_hidden_units,
        )
        self.temporal_order = str(temporal_order)
        generator = torch.Generator(device="cpu").manual_seed(seed + 91_000)
        if temporal_order == "fixed_shuffle":
            permutation = torch.randperm(config.timesteps, generator=generator)
        else:
            permutation = torch.arange(config.timesteps)
        self.register_buffer("input_time_permutation", permutation.to(torch.long))

    def forward(self, events, *, return_event_rate: bool = False):  # type: ignore[override]
        if events.ndim != 3:
            raise ValueError("events must have shape [batch, time, channels]")
        if events.shape[1] != self.config.timesteps:
            raise ValueError(f"events must have {self.config.timesteps} timesteps")
        if events.shape[2] != self.input_neurons:
            raise ValueError(f"events must have {self.input_neurons} channels")
        events = events.index_select(1, self.input_time_permutation)
        membrane = events.new_zeros(
            (events.shape[0], self.hidden_neurons), dtype=torch.float32
        )
        hidden_spikes = torch.zeros_like(membrane)
        accumulated_spikes = torch.zeros_like(membrane)
        zero_state = events.new_zeros(
            (events.shape[0], self.neuron_count), dtype=torch.float32
        )
        hidden_trace = []
        history: list = []
        for step in range(events.shape[1]):
            sensor_events = events[:, step, :].to(torch.float32) * self.config.input_gain
            network_state = torch.cat((sensor_events, hidden_spikes), dim=1)
            history.insert(0, network_state)
            if len(history) > 1:
                history.pop()
            current = delayed_sparse_current(
                self.graph,
                history,
                zero_state=zero_state,
                max_delay_steps=0,
            )[:, self.input_neurons :]
            pre_reset = membrane * self.config.reservoir_leak + current
            hidden_spikes = SurrogateSpike.apply(
                pre_reset - self.config.reservoir_threshold,
                self.surrogate_slope,
            )
            membrane = pre_reset - hidden_spikes * self.config.reservoir_threshold
            accumulated_spikes = accumulated_spikes + hidden_spikes
            hidden_trace.append(hidden_spikes)
        mean_spikes = accumulated_spikes / events.shape[1]
        logits = self.readout(torch.stack(hidden_trace, dim=1), membrane)
        if return_event_rate:
            return logits, mean_spikes.mean()
        return logits


@dataclass
class SHDTemporalPyramidResult:
    config: SHDConfig
    device: str
    temporal_levels: tuple[int, ...]
    projection_dim: int
    baseline_readout_hidden_units: int
    arms: list[dict]
    records: list[dict]
    summary: list[dict]

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "shd_temporal_pyramid.json"
        records_path = output / "shd_temporal_pyramid_records.csv"
        summary_path = output / "shd_temporal_pyramid_summary.csv"
        payload = {
            "config": asdict(self.config),
            "device": self.device,
            "temporal_levels": list(self.temporal_levels),
            "projection_dim": self.projection_dim,
            "baseline_readout_hidden_units": self.baseline_readout_hidden_units,
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
            plot_path = output / "shd_temporal_pyramid_summary.png"
            plot_shd_temporal_pyramid(self.summary, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def run_shd_temporal_pyramid(
    config: SHDConfig,
    *,
    device="auto",
    surrogate_slope: float = 10.0,
    projection_dim: int = 32,
    temporal_levels: Iterable[int] = DEFAULT_TEMPORAL_LEVELS,
    baseline_readout_hidden_units: int = 128,
    ltw_minimum: float = 0.0,
    ltw_maximum: float = 1.0,
) -> SHDTemporalPyramidResult:
    if torch is None:
        raise ImportError("Phase 36 temporal pyramid requires PyTorch")
    levels = tuple(int(level) for level in temporal_levels)
    if not levels or any(level <= 0 or level > config.timesteps for level in levels):
        raise ValueError("temporal levels must be between one and timesteps")
    arms = default_shd_temporal_pyramid_arms()
    resolved = resolve_device(device)
    train_events, train_labels, test_events, test_labels = load_shd_tensors(config)
    records: list[dict] = []
    for seed in config.seeds:
        for arm in arms:
            seed_everything(seed, device=resolved)
            required_edges = (
                config.input_neurons * config.sensor_fanout
                + arm.hidden_neurons * config.recurrent_fanout
            )
            arm_config = replace(
                config,
                hidden_neurons=arm.hidden_neurons,
                max_edges=max(config.max_edges, _next_power_of_two(required_edges)),
            )
            if arm.readout_mode == "global":
                model = SHDSparseClassifier(
                    arm_config,
                    seed=seed,
                    delay_pattern="none",
                    max_delay_steps=0,
                    surrogate_slope=surrogate_slope,
                    readout_kind="mlp",
                    readout_hidden_units=baseline_readout_hidden_units,
                    device=resolved,
                ).to(resolved)
                feature_dim = 2 * arm.hidden_neurons
                bottleneck_units = baseline_readout_hidden_units
            else:
                model = SHDTemporalPyramidClassifier(
                    arm_config,
                    seed=seed,
                    surrogate_slope=surrogate_slope,
                    projection_dim=projection_dim,
                    temporal_levels=levels,
                    baseline_hidden_units=baseline_readout_hidden_units,
                    temporal_order=arm.temporal_order,
                    device=resolved,
                ).to(resolved)
                feature_dim = model.readout.feature_dim
                bottleneck_units = model.readout.bottleneck_units
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
            active = model.graph.active_mask
            active_edges = int(active.sum().item())
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
            readout_parameters = sum(
                parameter.numel() for parameter in model.readout.parameters()
            )
            allocated_parameters = sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            records.append(
                {
                    "seed": int(seed),
                    "arm": arm.name,
                    "hidden_neurons": int(arm.hidden_neurons),
                    "readout_mode": arm.readout_mode,
                    "temporal_order": arm.temporal_order,
                    "feature_dim": int(feature_dim),
                    "readout_bottleneck_units": int(bottleneck_units),
                    "train_accuracy": float(train_accuracy),
                    "test_accuracy": float(test_accuracy),
                    "active_edges": int(active_edges),
                    "readout_trainable_parameters": int(readout_parameters),
                    "effective_trainable_parameters": int(
                        readout_parameters + active_edges
                    ),
                    "allocated_trainable_parameters": int(allocated_parameters),
                    "initial_hidden_event_rate": float(initial_event_rate),
                    "final_hidden_event_rate": float(final_event_rate),
                    "event_rate_ratio": float(
                        final_event_rate / max(initial_event_rate, 1e-12)
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
    _attach_temporal_comparisons(records)
    return SHDTemporalPyramidResult(
        config=config,
        device=device_kind(resolved),
        temporal_levels=levels,
        projection_dim=int(projection_dim),
        baseline_readout_hidden_units=int(baseline_readout_hidden_units),
        arms=[asdict(arm) for arm in arms],
        records=records,
        summary=summarize_shd_temporal_pyramid(records, arms=arms),
    )


def summarize_shd_temporal_pyramid(
    records: Iterable[dict], *, arms: Iterable[SHDTemporalPyramidArm]
) -> list[dict]:
    rows = list(records)
    summary: list[dict] = []
    for arm in arms:
        group = [row for row in rows if row["arm"] == arm.name]
        if not group:
            continue
        global_gains = [float(row["gain_vs_same_width_global"]) for row in group]
        shuffled_gains = [float(row["gain_vs_same_arch_shuffled"]) for row in group]
        summary.append(
            {
                "arm": arm.name,
                "hidden_neurons": int(arm.hidden_neurons),
                "readout_mode": arm.readout_mode,
                "temporal_order": arm.temporal_order,
                "seeds": len(group),
                "mean_test_accuracy": statistics.fmean(
                    float(row["test_accuracy"]) for row in group
                ),
                "std_test_accuracy": statistics.pstdev(
                    float(row["test_accuracy"]) for row in group
                ),
                "mean_gain_vs_same_width_global": statistics.fmean(global_gains),
                "improved_seed_count_vs_global": sum(gain > 0 for gain in global_gains),
                "two_point_seed_count_vs_global": sum(
                    gain >= 0.02 for gain in global_gains
                ),
                "three_point_seed_count_vs_global": sum(
                    gain >= 0.03 for gain in global_gains
                ),
                "mean_gain_vs_same_arch_shuffled": statistics.fmean(shuffled_gains),
                "improved_seed_count_vs_shuffled": sum(
                    gain > 0 for gain in shuffled_gains
                ),
                "two_point_seed_count_vs_shuffled": sum(
                    gain >= 0.02 for gain in shuffled_gains
                ),
                "mean_final_hidden_event_rate": statistics.fmean(
                    float(row["final_hidden_event_rate"]) for row in group
                ),
                "mean_event_rate_vs_same_width_global": statistics.fmean(
                    float(row["event_rate_vs_same_width_global"]) for row in group
                ),
                "active_edges": int(group[0]["active_edges"]),
                "feature_dim": int(group[0]["feature_dim"]),
                "readout_bottleneck_units": int(
                    group[0]["readout_bottleneck_units"]
                ),
                "effective_trainable_parameters": int(
                    group[0]["effective_trainable_parameters"]
                ),
                "parameter_ratio_vs_same_width_global": statistics.fmean(
                    float(row["parameter_ratio_vs_same_width_global"])
                    for row in group
                ),
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


def plot_shd_temporal_pyramid(summary: list[dict], path: str | pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    labels = [row["arm"].replace("sparse", "").replace("_", "\n") for row in summary]
    accuracy = [100.0 * float(row["mean_test_accuracy"]) for row in summary]
    errors = [100.0 * float(row["std_test_accuracy"]) for row in summary]
    gains = [
        100.0 * float(row["mean_gain_vs_same_width_global"]) for row in summary
    ]
    parameters = [
        100.0 * float(row["parameter_ratio_vs_same_width_global"])
        for row in summary
    ]
    colors = ["#35b4f2" if row["readout_mode"] == "global" else "#48c78e" for row in summary]
    for index, row in enumerate(summary):
        if row["temporal_order"] == "fixed_shuffle":
            colors[index] = "#8b6fd6"
    figure, axes = plt.subplots(3, 1, figsize=(15, 13), constrained_layout=True)
    x = list(range(len(summary)))
    axes[0].bar(x, accuracy, yerr=errors, capsize=5, color=colors)
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_title("AMMC Gen-5 Phase 36: parameter-matched SHD temporal pyramid")
    axes[1].bar(x, gains, color=colors)
    axes[1].axhline(3.0, color="#bd3d3a", linestyle="--", label="+3 point gate")
    axes[1].set_ylabel("Gain vs same-width global (points)")
    axes[1].legend()
    axes[2].bar(x, parameters, color=colors)
    axes[2].axhline(100.0, color="#222222", linestyle="--")
    axes[2].set_ylabel("Parameters vs global (%)")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25)
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _attach_temporal_comparisons(records: list[dict]) -> None:
    globals_by_seed_width = {
        (int(row["seed"]), int(row["hidden_neurons"])): row
        for row in records
        if row["readout_mode"] == "global"
    }
    shuffled_by_seed_width = {
        (int(row["seed"]), int(row["hidden_neurons"])): row
        for row in records
        if row["readout_mode"] == "pyramid"
        and row["temporal_order"] == "fixed_shuffle"
    }
    for row in records:
        key = (int(row["seed"]), int(row["hidden_neurons"]))
        control = globals_by_seed_width[key]
        row["gain_vs_same_width_global"] = float(row["test_accuracy"]) - float(
            control["test_accuracy"]
        )
        row["event_rate_vs_same_width_global"] = float(
            row["final_hidden_event_rate"]
        ) / max(float(control["final_hidden_event_rate"]), 1e-12)
        row["parameter_ratio_vs_same_width_global"] = float(
            row["effective_trainable_parameters"]
        ) / max(float(control["effective_trainable_parameters"]), 1.0)
        shuffled = shuffled_by_seed_width.get(key)
        if (
            shuffled is not None
            and row["readout_mode"] == "pyramid"
            and row["temporal_order"] == "ordered"
        ):
            row["gain_vs_same_arch_shuffled"] = float(row["test_accuracy"]) - float(
                shuffled["test_accuracy"]
            )
        else:
            row["gain_vs_same_arch_shuffled"] = 0.0


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
