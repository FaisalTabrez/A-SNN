"""Phase 38 parameter-matched conventional baselines on SHD.

The Phase 37 random recurrent graph failed its causal gate. This suite compares
the retained sparse systems with a standard dense recurrent LIF trained through
time and a GRU at approximately the same effective parameter budget.
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
from .shd_temporal_controls import (
    BudgetMatchedTemporalReadout,
    SHDRawTemporalPyramidClassifier,
    disable_shd_recurrent_edges,
)
from .shd_temporal_pyramid import (
    DEFAULT_TEMPORAL_LEVELS,
    SHDTemporalPyramidClassifier,
    parameter_matched_bottleneck,
)
from .trainable_temporal_mnist import SurrogateSpike


@dataclass(frozen=True)
class SHDMatchedBaselineArm:
    name: str
    model_kind: str


SHD_MATCHED_BASELINE_ARMS = (
    SHDMatchedBaselineArm("event_count_mlp", "event_count"),
    SHDMatchedBaselineArm("raw_temporal_pyramid", "raw_temporal"),
    SHDMatchedBaselineArm("dense_lif_recurrent", "dense_lif"),
    SHDMatchedBaselineArm("gru_temporal", "gru"),
    SHDMatchedBaselineArm("sparse512_feedforward_pyramid", "sparse_feedforward"),
    SHDMatchedBaselineArm("sparse512_recurrent_pyramid", "sparse_recurrent"),
)


def available_shd_matched_baseline_arms() -> tuple[str, ...]:
    return tuple(arm.name for arm in SHD_MATCHED_BASELINE_ARMS)


def gru_parameter_count(input_neurons: int, hidden_units: int, classes: int) -> int:
    """PyTorch one-layer GRU plus linear classifier parameter count."""

    if input_neurons <= 0 or hidden_units <= 0 or classes <= 1:
        raise ValueError("invalid GRU dimensions")
    gru = (
        3 * hidden_units * input_neurons
        + 3 * hidden_units * hidden_units
        + 6 * hidden_units
    )
    classifier = hidden_units * classes + classes
    return int(gru + classifier)


def matched_gru_hidden_units(
    input_neurons: int, classes: int, target_parameters: int
) -> tuple[int, int]:
    """Largest GRU width whose parameter count does not exceed the target."""

    if target_parameters <= 0:
        raise ValueError("target parameters must be positive")
    hidden = 1
    while gru_parameter_count(input_neurons, hidden + 1, classes) <= target_parameters:
        hidden += 1
    return hidden, gru_parameter_count(input_neurons, hidden, classes)


class DenseLIFTemporalClassifier(nn.Module):
    """Conventional dense recurrent LIF with surrogate-gradient BPTT."""

    def __init__(
        self,
        config: SHDConfig,
        *,
        hidden_neurons: int,
        temporal_levels: Iterable[int],
        projection_dim: int,
        target_parameters: int,
        surrogate_slope: float,
    ) -> None:
        if torch is None:
            raise ImportError("Phase 38 matched baselines require PyTorch")
        super().__init__()
        if hidden_neurons <= 1:
            raise ValueError("dense LIF hidden width must exceed one")
        self.config = config
        self.hidden_neurons = int(hidden_neurons)
        self.surrogate_slope = float(surrogate_slope)
        self.input_projection = nn.Linear(
            config.input_neurons, hidden_neurons, bias=False
        )
        self.recurrent_projection = nn.Linear(
            hidden_neurons, hidden_neurons, bias=False
        )
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.orthogonal_(self.recurrent_projection.weight, gain=0.5)
        self.dynamics_parameter_count = (
            config.input_neurons * hidden_neurons + hidden_neurons * hidden_neurons
        )
        readout_budget = target_parameters - self.dynamics_parameter_count
        self.readout = BudgetMatchedTemporalReadout(
            trace_dim=hidden_neurons,
            final_dim=hidden_neurons,
            classes=config.classes,
            projection_dim=projection_dim,
            temporal_levels=temporal_levels,
            target_parameters=readout_budget,
        )

    def forward(self, events, *, return_event_rate: bool = False):  # type: ignore[override]
        if events.ndim != 3 or events.shape[2] != self.config.input_neurons:
            raise ValueError("events must have shape [batch, time, input_neurons]")
        membrane = events.new_zeros(
            (events.shape[0], self.hidden_neurons), dtype=torch.float32
        )
        spikes = torch.zeros_like(membrane)
        accumulated = torch.zeros_like(membrane)
        trace = []
        for step in range(events.shape[1]):
            sensor = events[:, step].to(torch.float32) * self.config.input_gain
            current = self.input_projection(sensor) + self.recurrent_projection(spikes)
            pre_reset = membrane * self.config.reservoir_leak + current
            spikes = SurrogateSpike.apply(
                pre_reset - self.config.reservoir_threshold,
                self.surrogate_slope,
            )
            membrane = pre_reset - spikes * self.config.reservoir_threshold
            accumulated = accumulated + spikes
            trace.append(spikes)
        mean_spikes = accumulated / events.shape[1]
        logits = self.readout(torch.stack(trace, dim=1), membrane)
        if return_event_rate:
            return logits, mean_spikes.mean()
        return logits


class GRUTemporalClassifier(nn.Module):
    """Conventional recurrent ANN control at the registered parameter budget."""

    def __init__(self, config: SHDConfig, *, hidden_units: int) -> None:
        if torch is None:
            raise ImportError("Phase 38 matched baselines require PyTorch")
        super().__init__()
        self.config = config
        self.hidden_units = int(hidden_units)
        self.gru = nn.GRU(
            input_size=config.input_neurons,
            hidden_size=hidden_units,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_units, config.classes)

    def forward(self, events, *, return_event_rate: bool = False):  # type: ignore[override]
        if events.ndim != 3 or events.shape[2] != self.config.input_neurons:
            raise ValueError("events must have shape [batch, time, input_neurons]")
        output, hidden = self.gru(events.to(torch.float32))
        logits = self.classifier(hidden[-1])
        if return_event_rate:
            return logits, output.abs().mean()
        return logits


@dataclass
class SHDMatchedBaselinesResult:
    config: SHDConfig
    device: str
    target_parameters: int
    sparse_hidden_neurons: int
    dense_lif_hidden_neurons: int
    gru_hidden_units: int
    temporal_levels: tuple[int, ...]
    arms: list[dict]
    records: list[dict]
    summary: list[dict]

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "shd_matched_baselines.json"
        records_path = output / "shd_matched_baselines_records.csv"
        summary_path = output / "shd_matched_baselines_summary.csv"
        payload = {
            "config": asdict(self.config),
            "device": self.device,
            "target_parameters": self.target_parameters,
            "sparse_hidden_neurons": self.sparse_hidden_neurons,
            "dense_lif_hidden_neurons": self.dense_lif_hidden_neurons,
            "gru_hidden_units": self.gru_hidden_units,
            "temporal_levels": list(self.temporal_levels),
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
            plot_path = output / "shd_matched_baselines_summary.png"
            plot_shd_matched_baselines(self.summary, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def run_shd_matched_baselines(
    config: SHDConfig,
    *,
    sparse_hidden_neurons: int = 512,
    dense_lif_hidden_neurons: int = 128,
    device="auto",
    surrogate_slope: float = 10.0,
    pyramid_projection_dim: int = 32,
    dense_lif_projection_dim: int = 16,
    temporal_levels: Iterable[int] = DEFAULT_TEMPORAL_LEVELS,
    readout_hidden_units: int = 128,
    ltw_minimum: float = 0.0,
    ltw_maximum: float = 1.0,
) -> SHDMatchedBaselinesResult:
    if torch is None:
        raise ImportError("Phase 38 matched baselines require PyTorch")
    levels = tuple(int(level) for level in temporal_levels)
    if not levels or any(level <= 0 or level > config.timesteps for level in levels):
        raise ValueError("temporal levels must be between one and timesteps")
    required_edges = (
        config.input_neurons * config.sensor_fanout
        + sparse_hidden_neurons * config.recurrent_fanout
    )
    _, pyramid_readout_parameters, _ = parameter_matched_bottleneck(
        hidden_neurons=sparse_hidden_neurons,
        classes=config.classes,
        projection_dim=pyramid_projection_dim,
        temporal_levels=levels,
        baseline_hidden_units=readout_hidden_units,
    )
    target_parameters = pyramid_readout_parameters + required_edges
    gru_hidden_units, _ = matched_gru_hidden_units(
        config.input_neurons, config.classes, target_parameters
    )
    arm_config = replace(
        config,
        hidden_neurons=sparse_hidden_neurons,
        max_edges=max(config.max_edges, _next_power_of_two(required_edges)),
    )
    resolved = resolve_device(device)
    train_events, train_labels, test_events, test_labels = load_shd_tensors(config)
    records: list[dict] = []
    for seed in config.seeds:
        for arm in SHD_MATCHED_BASELINE_ARMS:
            seed_everything(seed, device=resolved)
            if arm.model_kind == "event_count":
                model = SHDEventCountClassifier(arm_config, kind="mlp").to(resolved)
                topology = "raw_count"
                hidden_units = config.count_hidden_units
                dynamics_parameters = 0
            elif arm.model_kind == "raw_temporal":
                model = SHDRawTemporalPyramidClassifier(
                    arm_config,
                    projection_dim=pyramid_projection_dim,
                    temporal_levels=levels,
                    target_parameters=target_parameters,
                ).to(resolved)
                topology = "raw_temporal"
                hidden_units = model.readout.bottleneck_units
                dynamics_parameters = 0
            elif arm.model_kind == "dense_lif":
                model = DenseLIFTemporalClassifier(
                    arm_config,
                    hidden_neurons=dense_lif_hidden_neurons,
                    temporal_levels=levels,
                    projection_dim=dense_lif_projection_dim,
                    target_parameters=target_parameters,
                    surrogate_slope=surrogate_slope,
                ).to(resolved)
                topology = "dense_recurrent_lif"
                hidden_units = dense_lif_hidden_neurons
                dynamics_parameters = model.dynamics_parameter_count
            elif arm.model_kind == "gru":
                model = GRUTemporalClassifier(
                    arm_config, hidden_units=gru_hidden_units
                ).to(resolved)
                topology = "dense_gru"
                hidden_units = gru_hidden_units
                dynamics_parameters = sum(
                    parameter.numel() for parameter in model.gru.parameters()
                )
            else:
                model = SHDTemporalPyramidClassifier(
                    arm_config,
                    seed=seed,
                    surrogate_slope=surrogate_slope,
                    projection_dim=pyramid_projection_dim,
                    temporal_levels=levels,
                    baseline_hidden_units=readout_hidden_units,
                    temporal_order="ordered",
                    device=resolved,
                ).to(resolved)
                hidden_units = sparse_hidden_neurons
                dynamics_parameters = required_edges
                if arm.model_kind == "sparse_feedforward":
                    disable_shd_recurrent_edges(model)
                    topology = "sparse_feedforward_lif"
                else:
                    topology = "sparse_recurrent_lif"
            initial_ltw = None
            if isinstance(model, SHDSparseClassifier):
                initial_ltw = model.graph.long_term_weight.detach().clone()
            _, _, initial_activity = _measure(
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
            test_accuracy, inference_seconds, final_activity = _measure(
                model, test_events, test_labels, config.batch_size, resolved
            )
            active_edges = recurrent_edges = 0
            mean_ltw_change = upper_saturation = 0.0
            if isinstance(model, SHDSparseClassifier):
                active = model.graph.active_mask
                recurrent = active & (model.graph.sources >= config.input_neurons)
                active_edges = int(active.sum().item())
                recurrent_edges = int(recurrent.sum().item())
                final_ltw = model.graph.long_term_weight.detach()
                mean_ltw_change = float(
                    (final_ltw[active] - initial_ltw[active]).abs().mean().item()
                )
                upper_saturation = float(
                    (final_ltw[active] >= ltw_maximum - 1e-6)
                    .to(torch.float32).mean().item()
                )
                effective_parameters = (
                    sum(parameter.numel() for parameter in model.readout.parameters())
                    + active_edges
                )
                activity_kind = "hidden_spike_rate"
            else:
                effective_parameters = sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                )
                activity_kind = (
                    "hidden_spike_rate"
                    if isinstance(model, DenseLIFTemporalClassifier)
                    else "analog_activation" if isinstance(model, GRUTemporalClassifier)
                    else "input_event_rate"
                )
            records.append(
                {
                    "seed": int(seed),
                    "arm": arm.name,
                    "model_kind": arm.model_kind,
                    "topology": topology,
                    "hidden_units": int(hidden_units),
                    "train_accuracy": float(train_accuracy),
                    "test_accuracy": float(test_accuracy),
                    "active_edges": int(active_edges),
                    "active_recurrent_edges": int(recurrent_edges),
                    "dynamics_parameters": int(dynamics_parameters),
                    "effective_trainable_parameters": int(effective_parameters),
                    "parameter_ratio_vs_target": float(
                        effective_parameters / target_parameters
                    ),
                    "initial_activity": float(initial_activity),
                    "final_activity": float(final_activity),
                    "activity_kind": activity_kind,
                    "mean_absolute_ltw_change": float(mean_ltw_change),
                    "upper_ltw_saturation_rate": float(upper_saturation),
                    "train_seconds": float(train_seconds),
                    "inference_seconds": float(inference_seconds),
                    "test_examples_per_second": float(
                        test_events.shape[0] / max(inference_seconds, 1e-12)
                    ),
                }
            )
    _attach_baseline_comparisons(records)
    return SHDMatchedBaselinesResult(
        config=config,
        device=device_kind(resolved),
        target_parameters=int(target_parameters),
        sparse_hidden_neurons=int(sparse_hidden_neurons),
        dense_lif_hidden_neurons=int(dense_lif_hidden_neurons),
        gru_hidden_units=int(gru_hidden_units),
        temporal_levels=levels,
        arms=[asdict(arm) for arm in SHD_MATCHED_BASELINE_ARMS],
        records=records,
        summary=summarize_shd_matched_baselines(records),
    )


def summarize_shd_matched_baselines(records: Iterable[dict]) -> list[dict]:
    rows = list(records)
    summary: list[dict] = []
    for arm in SHD_MATCHED_BASELINE_ARMS:
        group = [row for row in rows if row["arm"] == arm.name]
        if not group:
            continue
        dense_gains = [float(row["gain_vs_dense_lif"]) for row in group]
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
                "mean_gain_vs_raw_temporal": statistics.fmean(
                    float(row["gain_vs_raw_temporal"]) for row in group
                ),
                "mean_gain_vs_dense_lif": statistics.fmean(dense_gains),
                "improved_seed_count_vs_dense_lif": sum(
                    gain > 0 for gain in dense_gains
                ),
                "one_point_seed_count_vs_dense_lif": sum(
                    gain >= 0.01 for gain in dense_gains
                ),
                "mean_gain_vs_gru": statistics.fmean(
                    float(row["gain_vs_gru"]) for row in group
                ),
                "mean_gain_vs_sparse_feedforward": statistics.fmean(
                    float(row["gain_vs_sparse_feedforward"]) for row in group
                ),
                "hidden_units": int(group[0]["hidden_units"]),
                "active_edges": int(group[0]["active_edges"]),
                "active_recurrent_edges": int(group[0]["active_recurrent_edges"]),
                "effective_trainable_parameters": int(
                    group[0]["effective_trainable_parameters"]
                ),
                "parameter_ratio_vs_target": statistics.fmean(
                    float(row["parameter_ratio_vs_target"]) for row in group
                ),
                "mean_final_activity": statistics.fmean(
                    float(row["final_activity"]) for row in group
                ),
                "activity_kind": group[0]["activity_kind"],
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


def plot_shd_matched_baselines(summary: list[dict], path: str | pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    labels = [row["arm"].replace("_", "\n") for row in summary]
    accuracy = [100.0 * float(row["mean_test_accuracy"]) for row in summary]
    errors = [100.0 * float(row["std_test_accuracy"]) for row in summary]
    dense_gains = [100.0 * float(row["mean_gain_vs_dense_lif"]) for row in summary]
    throughput = [float(row["mean_test_examples_per_second"]) for row in summary]
    colors = ("#8b6fd6", "#ffb31a", "#bd3d3a", "#666666", "#48c78e", "#167d55")
    x = list(range(len(summary)))
    figure, axes = plt.subplots(3, 1, figsize=(16, 13), constrained_layout=True)
    axes[0].bar(x, accuracy, yerr=errors, capsize=5, color=colors)
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_title("AMMC Gen-5 Phase 38: parameter-matched SHD baselines")
    axes[1].bar(x, dense_gains, color=colors)
    axes[1].axhline(2.0, color="#bd3d3a", linestyle="--", label="+2 point sparse gate")
    axes[1].set_ylabel("Gain vs dense LIF (points)")
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


def _attach_baseline_comparisons(records: list[dict]) -> None:
    by_seed_arm = {(int(row["seed"]), row["arm"]): row for row in records}
    for row in records:
        seed = int(row["seed"])
        raw = by_seed_arm[(seed, "raw_temporal_pyramid")]
        dense_lif = by_seed_arm[(seed, "dense_lif_recurrent")]
        gru = by_seed_arm[(seed, "gru_temporal")]
        feedforward = by_seed_arm[(seed, "sparse512_feedforward_pyramid")]
        row["gain_vs_raw_temporal"] = float(row["test_accuracy"]) - float(
            raw["test_accuracy"]
        )
        row["gain_vs_dense_lif"] = float(row["test_accuracy"]) - float(
            dense_lif["test_accuracy"]
        )
        row["gain_vs_gru"] = float(row["test_accuracy"]) - float(
            gru["test_accuracy"]
        )
        row["gain_vs_sparse_feedforward"] = float(row["test_accuracy"]) - float(
            feedforward["test_accuracy"]
        )


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
