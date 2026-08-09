"""Phase 42 topology/readout seed decomposition for sparse SHD models."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
import json
import pathlib
import statistics
from typing import Iterable

from .event_mnist import torch
from .runtime import device_kind, resolve_device, seed_everything
from .shd_benchmark import SHDConfig, _measure, _train_model, load_shd_tensors
from .shd_sparse_width import FixedBudgetSparseAnalogClassifier
from .shd_temporal_controls import BudgetMatchedTemporalReadout, SHDRawTemporalPyramidClassifier, disable_shd_recurrent_edges
from .shd_temporal_pyramid import DEFAULT_TEMPORAL_LEVELS


@dataclass(frozen=True)
class SHDInitializationArm:
    name: str
    hidden_neurons: int


SHD_INITIALIZATION_ARMS = (
    SHDInitializationArm("raw_temporal_pyramid", 0),
    SHDInitializationArm("sparse_analog_leaky_512", 512),
    SHDInitializationArm("sparse_analog_leaky_1024", 1024),
)


def available_shd_initialization_arms() -> tuple[str, ...]:
    return tuple(arm.name for arm in SHD_INITIALIZATION_ARMS)


@dataclass
class SHDInitializationRobustnessResult:
    config: SHDConfig
    device: str
    topology_seeds: tuple[int, ...]
    readout_seeds: tuple[int, ...]
    temporal_levels: tuple[int, ...]
    target_parameters: int
    arms: list[dict]
    records: list[dict]
    summary: list[dict]

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "shd_initialization_robustness.json"
        records_path = output / "shd_initialization_robustness_records.csv"
        summary_path = output / "shd_initialization_robustness_summary.csv"
        payload = {
            "config": asdict(self.config), "device": self.device,
            "topology_seeds": list(self.topology_seeds), "readout_seeds": list(self.readout_seeds),
            "temporal_levels": list(self.temporal_levels), "target_parameters": self.target_parameters,
            "arms": self.arms, "records": self.records, "summary": self.summary,
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        _write_csv(records_path, self.records)
        _write_csv(summary_path, self.summary)
        paths = {"json": str(json_path), "records_csv": str(records_path), "summary_csv": str(summary_path)}
        if plot:
            plot_path = output / "shd_initialization_robustness_summary.png"
            plot_shd_initialization_robustness(self.summary, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def run_shd_initialization_robustness(
    config: SHDConfig,
    *,
    topology_seeds: Iterable[int] = (42, 43, 44),
    readout_seeds: Iterable[int] = (142, 143, 144),
    target_parameters: int = 133_631,
    device="auto",
    surrogate_slope: float = 10.0,
    projection_dim: int = 32,
    temporal_levels: Iterable[int] = DEFAULT_TEMPORAL_LEVELS,
    readout_hidden_units: int = 128,
    ltw_minimum: float = 0.0,
    ltw_maximum: float = 1.0,
) -> SHDInitializationRobustnessResult:
    if torch is None:
        raise ImportError("Phase 42 initialization robustness requires PyTorch")
    topology_values = tuple(int(seed) for seed in topology_seeds)
    readout_values = tuple(int(seed) for seed in readout_seeds)
    levels = tuple(int(level) for level in temporal_levels)
    if not topology_values or not readout_values or len(set(topology_values)) != len(topology_values) or len(set(readout_values)) != len(readout_values):
        raise ValueError("topology and readout seed lists must be non-empty and unique")
    if not levels or any(level <= 0 or level > config.timesteps for level in levels):
        raise ValueError("invalid temporal levels")
    resolved = resolve_device(device)
    train_events, train_labels, test_events, test_labels = load_shd_tensors(config)
    records: list[dict] = []

    for readout_seed in readout_values:
        seed_everything(readout_seed, device=resolved)
        model = SHDRawTemporalPyramidClassifier(
            config, projection_dim=projection_dim, temporal_levels=levels,
            target_parameters=target_parameters,
        ).to(resolved)
        records.append(
            _train_and_record(
                model, config, train_events, train_labels, test_events, test_labels,
                arm=SHD_INITIALIZATION_ARMS[0], topology_seed=-1,
                readout_seed=readout_seed, active_edges=0, connected_hidden=0,
                target_parameters=target_parameters, resolved=resolved,
                ltw_minimum=ltw_minimum, ltw_maximum=ltw_maximum,
            )
        )

    for topology_seed in topology_values:
        for readout_seed in readout_values:
            for arm in SHD_INITIALIZATION_ARMS[1:]:
                required_edges = config.input_neurons * config.sensor_fanout + arm.hidden_neurons * config.recurrent_fanout
                arm_config = replace(
                    config, hidden_neurons=arm.hidden_neurons,
                    max_edges=max(config.max_edges, _next_power_of_two(required_edges)),
                )
                seed_everything(topology_seed, device=resolved)
                model = FixedBudgetSparseAnalogClassifier(
                    arm_config, seed=topology_seed, surrogate_slope=surrogate_slope,
                    projection_dim=projection_dim, temporal_levels=levels,
                    readout_hidden_units=readout_hidden_units,
                    target_parameters=target_parameters, device=resolved,
                ).to(resolved)
                disable_shd_recurrent_edges(model)
                active = model.graph.active_mask
                active_edges = int(active.sum().item())
                targets = model.graph.targets[active] - config.input_neurons
                connected_hidden = int(torch.unique(targets).numel())

                # The final readout and optimizer seed are explicitly independent
                # of topology construction and graph allocator capacity.
                seed_everything(readout_seed, device=resolved)
                model.readout = BudgetMatchedTemporalReadout(
                    trace_dim=arm.hidden_neurons, final_dim=arm.hidden_neurons,
                    classes=config.classes, projection_dim=projection_dim,
                    temporal_levels=levels,
                    target_parameters=target_parameters - active_edges,
                ).to(resolved)
                records.append(
                    _train_and_record(
                        model, arm_config, train_events, train_labels, test_events, test_labels,
                        arm=arm, topology_seed=topology_seed,
                        readout_seed=readout_seed, active_edges=active_edges,
                        connected_hidden=connected_hidden, target_parameters=target_parameters,
                        resolved=resolved, ltw_minimum=ltw_minimum, ltw_maximum=ltw_maximum,
                    )
                )
    _attach_comparisons(records)
    return SHDInitializationRobustnessResult(
        config=config, device=device_kind(resolved), topology_seeds=topology_values,
        readout_seeds=readout_values, temporal_levels=levels,
        target_parameters=int(target_parameters),
        arms=[asdict(arm) for arm in SHD_INITIALIZATION_ARMS], records=records,
        summary=summarize_shd_initialization_robustness(records),
    )


def _train_and_record(
    model, config, train_events, train_labels, test_events, test_labels, *,
    arm: SHDInitializationArm, topology_seed: int, readout_seed: int,
    active_edges: int, connected_hidden: int, target_parameters: int,
    resolved, ltw_minimum: float, ltw_maximum: float,
) -> dict:
    initial_accuracy, _, initial_activity = _measure(model, test_events, test_labels, config.batch_size, resolved)
    training_config = replace(config, reservoir_learning_rate=0.0)
    train_seconds = _train_model(
        model, train_events, train_labels, training_config,
        seed=readout_seed, device=resolved,
        ltw_minimum=ltw_minimum, ltw_maximum=ltw_maximum,
    )
    train_accuracy, _, _ = _measure(model, train_events, train_labels, config.batch_size, resolved)
    test_accuracy, inference_seconds, final_activity = _measure(model, test_events, test_labels, config.batch_size, resolved)
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if active_edges:
        trainable_parameters = sum(parameter.numel() for parameter in model.readout.parameters())
    effective_parameters = trainable_parameters + active_edges
    return {
        "arm": arm.name, "hidden_neurons": arm.hidden_neurons,
        "topology_seed": int(topology_seed), "readout_seed": int(readout_seed),
        "initial_test_accuracy": float(initial_accuracy), "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy), "active_edges": int(active_edges),
        "connected_hidden_neurons": int(connected_hidden),
        "effective_model_parameters": int(effective_parameters),
        "parameter_ratio_vs_target": float(effective_parameters / target_parameters),
        "initial_activity": float(initial_activity), "final_activity": float(final_activity),
        "train_seconds": float(train_seconds), "inference_seconds": float(inference_seconds),
        "test_examples_per_second": float(test_events.shape[0] / max(inference_seconds, 1e-12)),
    }


def summarize_shd_initialization_robustness(records: Iterable[dict]) -> list[dict]:
    rows = list(records)
    summary: list[dict] = []
    for arm in SHD_INITIALIZATION_ARMS:
        group = [row for row in rows if row["arm"] == arm.name]
        if not group:
            continue
        raw_gains = [float(row["gain_vs_raw_temporal"]) for row in group]
        width_gains = [float(row["gain_vs_sparse_512"]) for row in group]
        topology_groups = {
            seed: [row for row in group if int(row["topology_seed"]) == seed]
            for seed in sorted(set(int(row["topology_seed"]) for row in group))
            if seed >= 0
        }
        topology_means = [statistics.fmean(float(row["test_accuracy"]) for row in values) for values in topology_groups.values()]
        within_stds = [statistics.pstdev(float(row["test_accuracy"]) for row in values) for values in topology_groups.values()]
        summary.append(
            {
                "arm": arm.name, "hidden_neurons": arm.hidden_neurons, "runs": len(group),
                "mean_test_accuracy": statistics.fmean(float(row["test_accuracy"]) for row in group),
                "std_test_accuracy": statistics.pstdev(float(row["test_accuracy"]) for row in group),
                "minimum_test_accuracy": min(float(row["test_accuracy"]) for row in group),
                "maximum_test_accuracy": max(float(row["test_accuracy"]) for row in group),
                "mean_gain_vs_raw_temporal": statistics.fmean(raw_gains),
                "positive_pair_count_vs_raw": sum(gain > 0.0 for gain in raw_gains),
                "two_point_pair_count_vs_raw": sum(gain >= 0.02 for gain in raw_gains),
                "mean_gain_vs_sparse_512": statistics.fmean(width_gains),
                "positive_pair_count_vs_512": sum(gain > 0.0 for gain in width_gains),
                "between_topology_std": statistics.pstdev(topology_means) if topology_means else 0.0,
                "mean_within_topology_readout_std": statistics.fmean(within_stds) if within_stds else statistics.pstdev(float(row["test_accuracy"]) for row in group),
                "mean_connected_hidden_neurons": statistics.fmean(int(row["connected_hidden_neurons"]) for row in group),
                "effective_model_parameters": int(group[0]["effective_model_parameters"]),
                "parameter_ratio_vs_target": statistics.fmean(float(row["parameter_ratio_vs_target"]) for row in group),
                "mean_final_activity": statistics.fmean(float(row["final_activity"]) for row in group),
                "mean_train_seconds": statistics.fmean(float(row["train_seconds"]) for row in group),
                "mean_test_examples_per_second": statistics.fmean(float(row["test_examples_per_second"]) for row in group),
            }
        )
    return summary


def plot_shd_initialization_robustness(summary: list[dict], path: str | pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    labels = [row["arm"].replace("_", "\n") for row in summary]
    x = list(range(len(summary)))
    accuracy = [100.0 * float(row["mean_test_accuracy"]) for row in summary]
    errors = [100.0 * float(row["std_test_accuracy"]) for row in summary]
    gain = [100.0 * float(row["mean_gain_vs_raw_temporal"]) for row in summary]
    within = [100.0 * float(row["mean_within_topology_readout_std"]) for row in summary]
    between = [100.0 * float(row["between_topology_std"]) for row in summary]
    colors = ("#ffb31a", "#167d55", "#35b4f2")
    figure, axes = plt.subplots(3, 1, figsize=(15, 13), constrained_layout=True)
    axes[0].bar(x, accuracy, yerr=errors, capsize=5, color=colors)
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_title("AMMC Gen-5 Phase 42: topology/readout robustness")
    axes[1].bar(x, gain, color=colors)
    axes[1].axhline(2.0, color="#bd3d3a", linestyle="--", label="+2 point gate")
    axes[1].set_ylabel("Gain vs paired raw (points)")
    axes[1].legend()
    axes[2].bar(x, within, label="Within-topology readout std", color="#8b6fd6")
    axes[2].bar(x, between, bottom=within, label="Between-topology std", color="#35b4f2")
    axes[2].set_ylabel("Accuracy standard deviation (points)")
    axes[2].legend()
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25)
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _attach_comparisons(records: list[dict]) -> None:
    raw = {int(row["readout_seed"]): row for row in records if row["hidden_neurons"] == 0}
    sparse = {
        (int(row["topology_seed"]), int(row["readout_seed"]), int(row["hidden_neurons"])): row
        for row in records if int(row["hidden_neurons"]) > 0
    }
    for row in records:
        value = float(row["test_accuracy"])
        readout_seed = int(row["readout_seed"])
        row["gain_vs_raw_temporal"] = value - float(raw[readout_seed]["test_accuracy"])
        if int(row["hidden_neurons"]) == 0:
            row["gain_vs_sparse_512"] = 0.0
        else:
            key = (int(row["topology_seed"]), readout_seed, 512)
            row["gain_vs_sparse_512"] = value - float(sparse[key]["test_accuracy"])


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
