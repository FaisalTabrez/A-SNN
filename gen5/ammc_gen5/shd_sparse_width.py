"""Phase 41 fixed-budget scaling of the supported sparse leaky analog model."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
import json
import pathlib
import statistics
from typing import Iterable

from .event_mnist import torch
from .runtime import device_kind, resolve_device, seed_everything
from .shd_benchmark import SHDConfig, SHDSparseClassifier, _measure, _train_model, load_shd_tensors
from .shd_sparse_mechanisms import SHDAnalogTemporalPyramidClassifier
from .shd_temporal_controls import BudgetMatchedTemporalReadout, SHDRawTemporalPyramidClassifier, disable_shd_recurrent_edges
from .shd_temporal_pyramid import DEFAULT_TEMPORAL_LEVELS


@dataclass(frozen=True)
class SHDSparseWidthArm:
    name: str
    hidden_neurons: int


def default_shd_sparse_width_arms(widths: Iterable[int] = (128, 256, 512, 1024)) -> tuple[SHDSparseWidthArm, ...]:
    values = tuple(int(width) for width in widths)
    if not values or any(width <= 1 for width in values) or len(set(values)) != len(values):
        raise ValueError("hidden widths must be unique integers greater than one")
    return (SHDSparseWidthArm("raw_temporal_pyramid", 0),) + tuple(
        SHDSparseWidthArm(f"sparse_analog_leaky_{width}", width) for width in values
    )


class FixedBudgetSparseAnalogClassifier(SHDAnalogTemporalPyramidClassifier):
    """Sparse analog classifier whose total effective parameter budget is fixed."""

    def __init__(
        self,
        config: SHDConfig,
        *,
        seed: int,
        surrogate_slope: float,
        projection_dim: int,
        temporal_levels: Iterable[int],
        readout_hidden_units: int,
        target_parameters: int,
        device,
    ) -> None:
        super().__init__(
            config,
            seed=seed,
            surrogate_slope=surrogate_slope,
            projection_dim=projection_dim,
            temporal_levels=temporal_levels,
            baseline_hidden_units=readout_hidden_units,
            temporal_order="ordered",
            device=device,
        )
        sensor_edges = config.input_neurons * config.sensor_fanout
        self.readout = BudgetMatchedTemporalReadout(
            trace_dim=config.hidden_neurons,
            final_dim=config.hidden_neurons,
            classes=config.classes,
            projection_dim=projection_dim,
            temporal_levels=temporal_levels,
            target_parameters=target_parameters - sensor_edges,
        )
        self.target_parameters = int(target_parameters)


@dataclass
class SHDSparseWidthResult:
    config: SHDConfig
    device: str
    widths: tuple[int, ...]
    temporal_levels: tuple[int, ...]
    target_parameters: int
    arms: list[dict]
    records: list[dict]
    summary: list[dict]

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "shd_sparse_width.json"
        records_path = output / "shd_sparse_width_records.csv"
        summary_path = output / "shd_sparse_width_summary.csv"
        payload = {
            "config": asdict(self.config), "device": self.device,
            "widths": list(self.widths), "temporal_levels": list(self.temporal_levels),
            "target_parameters": self.target_parameters, "arms": self.arms,
            "records": self.records, "summary": self.summary,
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        _write_csv(records_path, self.records)
        _write_csv(summary_path, self.summary)
        paths = {"json": str(json_path), "records_csv": str(records_path), "summary_csv": str(summary_path)}
        if plot:
            plot_path = output / "shd_sparse_width_summary.png"
            plot_shd_sparse_width(self.summary, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def run_shd_sparse_width(
    config: SHDConfig,
    *,
    widths: Iterable[int] = (128, 256, 512, 1024),
    target_parameters: int = 133_631,
    device="auto",
    surrogate_slope: float = 10.0,
    projection_dim: int = 32,
    temporal_levels: Iterable[int] = DEFAULT_TEMPORAL_LEVELS,
    readout_hidden_units: int = 128,
    ltw_minimum: float = 0.0,
    ltw_maximum: float = 1.0,
) -> SHDSparseWidthResult:
    if torch is None:
        raise ImportError("Phase 41 sparse width scaling requires PyTorch")
    arms = default_shd_sparse_width_arms(widths)
    width_values = tuple(arm.hidden_neurons for arm in arms if arm.hidden_neurons > 0)
    levels = tuple(int(level) for level in temporal_levels)
    if target_parameters <= 0 or not levels or any(level <= 0 or level > config.timesteps for level in levels):
        raise ValueError("invalid target budget or temporal levels")
    resolved = resolve_device(device)
    train_events, train_labels, test_events, test_labels = load_shd_tensors(config)
    records: list[dict] = []
    for seed in config.seeds:
        for arm in arms:
            seed_everything(seed, device=resolved)
            active_edges = connected_hidden = max_fanin = 0
            mean_fanin = occupancy = 0.0
            if arm.hidden_neurons == 0:
                model = SHDRawTemporalPyramidClassifier(
                    config,
                    projection_dim=projection_dim,
                    temporal_levels=levels,
                    target_parameters=target_parameters,
                ).to(resolved)
                model_config = config
                activity_kind = "input_event_rate"
            else:
                required_edges = config.input_neurons * config.sensor_fanout + arm.hidden_neurons * config.recurrent_fanout
                model_config = replace(
                    config,
                    hidden_neurons=arm.hidden_neurons,
                    max_edges=max(config.max_edges, _next_power_of_two(required_edges)),
                )
                model = FixedBudgetSparseAnalogClassifier(
                    model_config,
                    seed=seed,
                    surrogate_slope=surrogate_slope,
                    projection_dim=projection_dim,
                    temporal_levels=levels,
                    readout_hidden_units=readout_hidden_units,
                    target_parameters=target_parameters,
                    device=resolved,
                ).to(resolved)
                disable_shd_recurrent_edges(model)
                active = model.graph.active_mask
                active_edges = int(active.sum().item())
                targets = model.graph.targets[active] - config.input_neurons
                fanin = torch.bincount(targets, minlength=arm.hidden_neurons)
                connected_hidden = int((fanin > 0).sum().item())
                occupancy = connected_hidden / arm.hidden_neurons
                mean_fanin = float(fanin[fanin > 0].to(torch.float32).mean().item())
                max_fanin = int(fanin.max().item())
                activity_kind = "analog_activation"
            initial_accuracy, _, initial_activity = _measure(model, test_events, test_labels, config.batch_size, resolved)
            training_config = replace(model_config, reservoir_learning_rate=0.0)
            train_seconds = _train_model(
                model, train_events, train_labels, training_config,
                seed=seed, device=resolved, ltw_minimum=ltw_minimum, ltw_maximum=ltw_maximum,
            )
            train_accuracy, _, _ = _measure(model, train_events, train_labels, config.batch_size, resolved)
            test_accuracy, inference_seconds, final_activity = _measure(model, test_events, test_labels, config.batch_size, resolved)
            trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            if isinstance(model, SHDSparseClassifier):
                trainable_parameters = sum(parameter.numel() for parameter in model.readout.parameters())
            effective_parameters = trainable_parameters + active_edges
            records.append(
                {
                    "seed": int(seed), "arm": arm.name, "hidden_neurons": int(arm.hidden_neurons),
                    "test_accuracy": float(test_accuracy), "train_accuracy": float(train_accuracy),
                    "initial_test_accuracy": float(initial_accuracy), "active_edges": int(active_edges),
                    "connected_hidden_neurons": int(connected_hidden), "hidden_occupancy_rate": float(occupancy),
                    "mean_sensor_fanin_connected": float(mean_fanin), "max_sensor_fanin": int(max_fanin),
                    "effective_model_parameters": int(effective_parameters), "trainable_parameters": int(trainable_parameters),
                    "parameter_ratio_vs_target": float(effective_parameters / target_parameters),
                    "initial_activity": float(initial_activity), "final_activity": float(final_activity),
                    "activity_kind": activity_kind, "train_seconds": float(train_seconds),
                    "inference_seconds": float(inference_seconds),
                    "test_examples_per_second": float(test_events.shape[0] / max(inference_seconds, 1e-12)),
                }
            )
    _attach_width_comparisons(records, width_values)
    return SHDSparseWidthResult(
        config=config, device=device_kind(resolved), widths=width_values,
        temporal_levels=levels, target_parameters=int(target_parameters),
        arms=[asdict(arm) for arm in arms], records=records,
        summary=summarize_shd_sparse_width(records, arms=arms),
    )


def summarize_shd_sparse_width(records: Iterable[dict], *, arms: Iterable[SHDSparseWidthArm] | None = None) -> list[dict]:
    rows = list(records)
    arm_values = tuple(arms) if arms is not None else tuple(
        SHDSparseWidthArm(name, int(group[0]["hidden_neurons"]))
        for name in dict.fromkeys(row["arm"] for row in rows)
        for group in [[row for row in rows if row["arm"] == name]]
    )
    summary: list[dict] = []
    for arm in arm_values:
        group = [row for row in rows if row["arm"] == arm.name]
        if not group:
            continue
        raw_gains = [float(row["gain_vs_raw_temporal"]) for row in group]
        base_gains = [float(row["gain_vs_smallest_width"]) for row in group]
        previous_gains = [float(row["gain_vs_previous_width"]) for row in group]
        summary.append(
            {
                "arm": arm.name, "hidden_neurons": arm.hidden_neurons, "seeds": len(group),
                "mean_test_accuracy": statistics.fmean(float(row["test_accuracy"]) for row in group),
                "std_test_accuracy": statistics.pstdev(float(row["test_accuracy"]) for row in group),
                "mean_gain_vs_raw_temporal": statistics.fmean(raw_gains),
                "raw_one_point_seed_count": sum(gain >= 0.01 for gain in raw_gains),
                "mean_gain_vs_smallest_width": statistics.fmean(base_gains),
                "smallest_width_one_point_seed_count": sum(gain >= 0.01 for gain in base_gains),
                "mean_gain_vs_previous_width": statistics.fmean(previous_gains),
                "previous_width_positive_seed_count": sum(gain > 0.0 for gain in previous_gains),
                "active_edges": int(group[0]["active_edges"]),
                "mean_connected_hidden_neurons": statistics.fmean(int(row["connected_hidden_neurons"]) for row in group),
                "mean_hidden_occupancy_rate": statistics.fmean(float(row["hidden_occupancy_rate"]) for row in group),
                "mean_sensor_fanin_connected": statistics.fmean(float(row["mean_sensor_fanin_connected"]) for row in group),
                "effective_model_parameters": int(group[0]["effective_model_parameters"]),
                "parameter_ratio_vs_target": statistics.fmean(float(row["parameter_ratio_vs_target"]) for row in group),
                "mean_final_activity": statistics.fmean(float(row["final_activity"]) for row in group),
                "mean_train_seconds": statistics.fmean(float(row["train_seconds"]) for row in group),
                "mean_test_examples_per_second": statistics.fmean(float(row["test_examples_per_second"]) for row in group),
            }
        )
    return summary


def plot_shd_sparse_width(summary: list[dict], path: str | pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    sparse = [row for row in summary if int(row["hidden_neurons"]) > 0]
    widths = [int(row["hidden_neurons"]) for row in sparse]
    accuracy = [100.0 * float(row["mean_test_accuracy"]) for row in sparse]
    errors = [100.0 * float(row["std_test_accuracy"]) for row in sparse]
    occupancy = [100.0 * float(row["mean_hidden_occupancy_rate"]) for row in sparse]
    throughput = [float(row["mean_test_examples_per_second"]) for row in sparse]
    raw = next((100.0 * float(row["mean_test_accuracy"]) for row in summary if int(row["hidden_neurons"]) == 0), None)
    figure, axes = plt.subplots(3, 1, figsize=(14, 13), constrained_layout=True)
    axes[0].errorbar(widths, accuracy, yerr=errors, marker="o", capsize=5, color="#167d55")
    if raw is not None:
        axes[0].axhline(raw, color="#ffb31a", linestyle="--", label="Raw temporal")
        axes[0].legend()
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_title("AMMC Gen-5 Phase 41: fixed-budget sparse width scaling")
    axes[1].plot(widths, occupancy, marker="s", color="#8b6fd6")
    axes[1].set_ylabel("Connected hidden nodes (%)")
    axes[2].plot(widths, throughput, marker="o", color="#35b4f2")
    axes[2].set_ylabel("Test examples / second")
    axes[2].set_xlabel("Sparse hidden width")
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks(widths, [str(width) for width in widths])
        axis.grid(alpha=0.25)
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _attach_width_comparisons(records: list[dict], widths: tuple[int, ...]) -> None:
    ordered = tuple(sorted(widths))
    smallest = ordered[0]
    lookup = {(int(row["seed"]), int(row["hidden_neurons"])): row for row in records}
    for row in records:
        seed = int(row["seed"])
        width = int(row["hidden_neurons"])
        value = float(row["test_accuracy"])
        row["gain_vs_raw_temporal"] = value - float(lookup[(seed, 0)]["test_accuracy"])
        row["gain_vs_smallest_width"] = 0.0 if width == 0 else value - float(lookup[(seed, smallest)]["test_accuracy"])
        if width <= smallest:
            row["gain_vs_previous_width"] = 0.0
        else:
            previous = ordered[ordered.index(width) - 1]
            row["gain_vs_previous_width"] = value - float(lookup[(seed, previous)]["test_accuracy"])


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
