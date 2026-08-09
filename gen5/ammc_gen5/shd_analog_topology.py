"""Phase 40 analog/topology controls following the Phase 39 falsification."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
import json
import pathlib
import statistics
from typing import Iterable

from .event_mnist import nn, torch
from .runtime import device_kind, resolve_device, seed_everything
from .shd_benchmark import SHDConfig, SHDSparseClassifier, _measure, _train_model, load_shd_tensors
from .shd_matched_baselines import DenseLIFTemporalClassifier
from .shd_sparse_mechanisms import SHDAnalogTemporalPyramidClassifier
from .shd_temporal_controls import BudgetMatchedTemporalReadout, SHDRawTemporalPyramidClassifier, disable_shd_recurrent_edges
from .shd_temporal_pyramid import DEFAULT_TEMPORAL_LEVELS, parameter_matched_bottleneck


@dataclass(frozen=True)
class SHDAnalogTopologyArm:
    name: str
    model_kind: str


SHD_ANALOG_TOPOLOGY_ARMS = (
    SHDAnalogTopologyArm("raw_temporal_pyramid", "raw"),
    SHDAnalogTopologyArm("dense_lif_recurrent", "dense_lif"),
    SHDAnalogTopologyArm("dense_analog_feedforward", "dense_analog_feedforward"),
    SHDAnalogTopologyArm("dense_analog_recurrent", "dense_analog_recurrent"),
    SHDAnalogTopologyArm("sparse_analog_instant", "sparse_analog_instant"),
    SHDAnalogTopologyArm("sparse_analog_leaky", "sparse_analog_leaky"),
)


def available_shd_analog_topology_arms() -> tuple[str, ...]:
    return tuple(arm.name for arm in SHD_ANALOG_TOPOLOGY_ARMS)


class DenseAnalogTemporalClassifier(nn.Module):
    """Dense leaky analog control with a matched temporal-pyramid readout."""

    def __init__(
        self,
        config: SHDConfig,
        *,
        hidden_neurons: int,
        recurrent: bool,
        temporal_levels: Iterable[int],
        projection_dim: int,
        target_parameters: int,
    ) -> None:
        if torch is None:
            raise ImportError("Phase 40 analog topology controls require PyTorch")
        super().__init__()
        self.config = config
        self.hidden_neurons = int(hidden_neurons)
        self.recurrent = bool(recurrent)
        self.input_projection = nn.Linear(config.input_neurons, hidden_neurons, bias=False)
        nn.init.xavier_uniform_(self.input_projection.weight)
        if recurrent:
            self.recurrent_projection = nn.Linear(hidden_neurons, hidden_neurons, bias=False)
            nn.init.orthogonal_(self.recurrent_projection.weight, gain=0.5)
        else:
            self.recurrent_projection = None
        self.dynamics_parameter_count = config.input_neurons * hidden_neurons
        if recurrent:
            self.dynamics_parameter_count += hidden_neurons * hidden_neurons
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
        membrane = events.new_zeros((events.shape[0], self.hidden_neurons), dtype=torch.float32)
        activation = torch.zeros_like(membrane)
        trace = []
        activity_sum = events.new_zeros((), dtype=torch.float32)
        for step in range(events.shape[1]):
            current = self.input_projection(events[:, step].to(torch.float32) * self.config.input_gain)
            if self.recurrent_projection is not None:
                current = current + self.recurrent_projection(activation)
            membrane = self.config.reservoir_leak * membrane + current
            activation = torch.tanh(membrane)
            trace.append(activation)
            activity_sum = activity_sum + activation.abs().mean()
        logits = self.readout(torch.stack(trace, dim=1), membrane)
        if return_event_rate:
            return logits, activity_sum / events.shape[1]
        return logits


@dataclass
class SHDAnalogTopologyResult:
    config: SHDConfig
    device: str
    sparse_hidden_neurons: int
    dense_hidden_neurons: int
    temporal_levels: tuple[int, ...]
    target_parameters: int
    arms: list[dict]
    records: list[dict]
    summary: list[dict]

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "shd_analog_topology.json"
        records_path = output / "shd_analog_topology_records.csv"
        summary_path = output / "shd_analog_topology_summary.csv"
        payload = {
            "config": asdict(self.config),
            "device": self.device,
            "sparse_hidden_neurons": self.sparse_hidden_neurons,
            "dense_hidden_neurons": self.dense_hidden_neurons,
            "temporal_levels": list(self.temporal_levels),
            "target_parameters": self.target_parameters,
            "arms": self.arms,
            "records": self.records,
            "summary": self.summary,
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        _write_csv(records_path, self.records)
        _write_csv(summary_path, self.summary)
        paths = {"json": str(json_path), "records_csv": str(records_path), "summary_csv": str(summary_path)}
        if plot:
            plot_path = output / "shd_analog_topology_summary.png"
            plot_shd_analog_topology(self.summary, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def run_shd_analog_topology(
    config: SHDConfig,
    *,
    sparse_hidden_neurons: int = 512,
    dense_hidden_neurons: int = 128,
    device="auto",
    surrogate_slope: float = 10.0,
    sparse_projection_dim: int = 32,
    dense_projection_dim: int = 16,
    temporal_levels: Iterable[int] = DEFAULT_TEMPORAL_LEVELS,
    readout_hidden_units: int = 128,
    ltw_minimum: float = 0.0,
    ltw_maximum: float = 1.0,
) -> SHDAnalogTopologyResult:
    if torch is None:
        raise ImportError("Phase 40 analog topology controls require PyTorch")
    levels = tuple(int(level) for level in temporal_levels)
    if not levels or any(level <= 0 or level > config.timesteps for level in levels):
        raise ValueError("temporal levels must be between one and timesteps")
    required_edges = config.input_neurons * config.sensor_fanout + sparse_hidden_neurons * config.recurrent_fanout
    _, readout_parameters, _ = parameter_matched_bottleneck(
        hidden_neurons=sparse_hidden_neurons,
        classes=config.classes,
        projection_dim=sparse_projection_dim,
        temporal_levels=levels,
        baseline_hidden_units=readout_hidden_units,
    )
    target_parameters = readout_parameters + config.input_neurons * config.sensor_fanout
    arm_config = replace(
        config,
        hidden_neurons=sparse_hidden_neurons,
        max_edges=max(config.max_edges, _next_power_of_two(required_edges)),
    )
    resolved = resolve_device(device)
    train_events, train_labels, test_events, test_labels = load_shd_tensors(config)
    records: list[dict] = []
    for seed in config.seeds:
        for arm in SHD_ANALOG_TOPOLOGY_ARMS:
            seed_everything(seed, device=resolved)
            topology = arm.model_kind
            dynamics_parameters = 0
            active_edges = 0
            if arm.model_kind == "raw":
                model = SHDRawTemporalPyramidClassifier(
                    arm_config,
                    projection_dim=sparse_projection_dim,
                    temporal_levels=levels,
                    target_parameters=target_parameters,
                ).to(resolved)
                activity_kind = "input_event_rate"
            elif arm.model_kind == "dense_lif":
                model = DenseLIFTemporalClassifier(
                    arm_config,
                    hidden_neurons=dense_hidden_neurons,
                    temporal_levels=levels,
                    projection_dim=dense_projection_dim,
                    target_parameters=target_parameters,
                    surrogate_slope=surrogate_slope,
                ).to(resolved)
                dynamics_parameters = model.dynamics_parameter_count
                activity_kind = "hidden_spike_rate"
            elif arm.model_kind.startswith("dense_analog"):
                recurrent = arm.model_kind.endswith("recurrent")
                model = DenseAnalogTemporalClassifier(
                    arm_config,
                    hidden_neurons=dense_hidden_neurons,
                    recurrent=recurrent,
                    temporal_levels=levels,
                    projection_dim=dense_projection_dim,
                    target_parameters=target_parameters,
                ).to(resolved)
                dynamics_parameters = model.dynamics_parameter_count
                activity_kind = "analog_activation"
            else:
                model_config = replace(
                    arm_config,
                    reservoir_leak=(0.0 if arm.model_kind.endswith("instant") else arm_config.reservoir_leak),
                )
                model = SHDAnalogTemporalPyramidClassifier(
                    model_config,
                    seed=seed,
                    surrogate_slope=surrogate_slope,
                    projection_dim=sparse_projection_dim,
                    temporal_levels=levels,
                    baseline_hidden_units=readout_hidden_units,
                    temporal_order="ordered",
                    device=resolved,
                ).to(resolved)
                disable_shd_recurrent_edges(model)
                active_edges = int(model.graph.active_mask.sum().item())
                dynamics_parameters = active_edges
                activity_kind = "analog_activation"
            initial_accuracy, _, initial_activity = _measure(model, test_events, test_labels, config.batch_size, resolved)
            training_config = replace(arm_config, reservoir_learning_rate=0.0)
            train_seconds = _train_model(
                model,
                train_events,
                train_labels,
                training_config,
                seed=seed,
                device=resolved,
                ltw_minimum=ltw_minimum,
                ltw_maximum=ltw_maximum,
            )
            train_accuracy, _, _ = _measure(model, train_events, train_labels, config.batch_size, resolved)
            test_accuracy, inference_seconds, final_activity = _measure(model, test_events, test_labels, config.batch_size, resolved)
            trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            if isinstance(model, SHDSparseClassifier):
                trainable_parameters = sum(parameter.numel() for parameter in model.readout.parameters())
            effective_parameters = trainable_parameters + (active_edges if isinstance(model, SHDSparseClassifier) else 0)
            records.append(
                {
                    "seed": int(seed),
                    "arm": arm.name,
                    "model_kind": arm.model_kind,
                    "topology": topology,
                    "initial_test_accuracy": float(initial_accuracy),
                    "train_accuracy": float(train_accuracy),
                    "test_accuracy": float(test_accuracy),
                    "hidden_neurons": int(0 if arm.model_kind == "raw" else dense_hidden_neurons if arm.model_kind.startswith("dense") else sparse_hidden_neurons),
                    "active_edges": int(active_edges),
                    "dynamics_parameters": int(dynamics_parameters),
                    "effective_model_parameters": int(effective_parameters),
                    "trainable_parameters": int(trainable_parameters),
                    "parameter_ratio_vs_target": float(effective_parameters / target_parameters),
                    "initial_activity": float(initial_activity),
                    "final_activity": float(final_activity),
                    "activity_kind": activity_kind,
                    "train_seconds": float(train_seconds),
                    "inference_seconds": float(inference_seconds),
                    "test_examples_per_second": float(test_events.shape[0] / max(inference_seconds, 1e-12)),
                }
            )
    _attach_comparisons(records)
    return SHDAnalogTopologyResult(
        config=config,
        device=device_kind(resolved),
        sparse_hidden_neurons=int(sparse_hidden_neurons),
        dense_hidden_neurons=int(dense_hidden_neurons),
        temporal_levels=levels,
        target_parameters=int(target_parameters),
        arms=[asdict(arm) for arm in SHD_ANALOG_TOPOLOGY_ARMS],
        records=records,
        summary=summarize_shd_analog_topology(records),
    )


def summarize_shd_analog_topology(records: Iterable[dict]) -> list[dict]:
    rows = list(records)
    summary: list[dict] = []
    for arm in SHD_ANALOG_TOPOLOGY_ARMS:
        group = [row for row in rows if row["arm"] == arm.name]
        if not group:
            continue
        analog_gains = [float(row["gain_vs_dense_lif"]) for row in group]
        sparse_gains = [float(row["gain_vs_dense_analog_feedforward"]) for row in group]
        leak_gains = [float(row["gain_vs_sparse_instant"]) for row in group]
        summary.append(
            {
                "arm": arm.name,
                "model_kind": arm.model_kind,
                "seeds": len(group),
                "mean_test_accuracy": statistics.fmean(float(row["test_accuracy"]) for row in group),
                "std_test_accuracy": statistics.pstdev(float(row["test_accuracy"]) for row in group),
                "mean_gain_vs_raw_temporal": statistics.fmean(float(row["gain_vs_raw_temporal"]) for row in group),
                "mean_gain_vs_dense_lif": statistics.fmean(analog_gains),
                "analog_one_point_seed_count": sum(gain >= 0.01 for gain in analog_gains),
                "mean_gain_vs_dense_analog_feedforward": statistics.fmean(sparse_gains),
                "sparse_one_point_seed_count": sum(gain >= 0.01 for gain in sparse_gains),
                "mean_gain_vs_sparse_instant": statistics.fmean(leak_gains),
                "leak_positive_seed_count": sum(gain > 0.0 for gain in leak_gains),
                "hidden_neurons": int(group[0]["hidden_neurons"]),
                "active_edges": int(group[0]["active_edges"]),
                "dynamics_parameters": int(group[0]["dynamics_parameters"]),
                "effective_model_parameters": int(group[0]["effective_model_parameters"]),
                "trainable_parameters": int(group[0]["trainable_parameters"]),
                "parameter_ratio_vs_target": statistics.fmean(float(row["parameter_ratio_vs_target"]) for row in group),
                "mean_final_activity": statistics.fmean(float(row["final_activity"]) for row in group),
                "activity_kind": group[0]["activity_kind"],
                "mean_train_seconds": statistics.fmean(float(row["train_seconds"]) for row in group),
                "mean_test_examples_per_second": statistics.fmean(float(row["test_examples_per_second"]) for row in group),
            }
        )
    return summary


def plot_shd_analog_topology(summary: list[dict], path: str | pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    labels = [row["arm"].replace("_", "\n") for row in summary]
    x = list(range(len(summary)))
    accuracy = [100.0 * float(row["mean_test_accuracy"]) for row in summary]
    errors = [100.0 * float(row["std_test_accuracy"]) for row in summary]
    raw_gains = [100.0 * float(row["mean_gain_vs_raw_temporal"]) for row in summary]
    throughput = [float(row["mean_test_examples_per_second"]) for row in summary]
    colors = ("#ffb31a", "#bd3d3a", "#8b6fd6", "#665191", "#35b4f2", "#167d55")
    figure, axes = plt.subplots(3, 1, figsize=(17, 13), constrained_layout=True)
    axes[0].bar(x, accuracy, yerr=errors, capsize=5, color=colors)
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_title("AMMC Gen-5 Phase 40: analog dynamics and sparse topology")
    axes[1].bar(x, raw_gains, color=colors)
    axes[1].axhline(2.0, color="#bd3d3a", linestyle="--", label="+2 point gate")
    axes[1].set_ylabel("Gain vs raw temporal (points)")
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


def _attach_comparisons(records: list[dict]) -> None:
    lookup = {(int(row["seed"]), row["arm"]): row for row in records}
    for row in records:
        seed = int(row["seed"])
        value = float(row["test_accuracy"])
        row["gain_vs_raw_temporal"] = value - float(lookup[(seed, "raw_temporal_pyramid")]["test_accuracy"])
        row["gain_vs_dense_lif"] = value - float(lookup[(seed, "dense_lif_recurrent")]["test_accuracy"])
        row["gain_vs_dense_analog_feedforward"] = value - float(lookup[(seed, "dense_analog_feedforward")]["test_accuracy"])
        row["gain_vs_sparse_instant"] = value - float(lookup[(seed, "sparse_analog_instant")]["test_accuracy"])


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
