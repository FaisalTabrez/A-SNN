"""Phase 32 diagnostic for the SHD sparse-representation bottleneck.

Phase 31 established that the official SHD pipeline is learnable while the
128-neuron sparse representation trails simple event-count controls. This
module changes one interpretable factor at a time: decoder nonlinearity,
heterogeneous delays under that decoder, hidden capacity, and firing threshold.
It is a diagnostic matrix, not a leaderboard configuration search.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
import json
import pathlib
import statistics
from typing import Iterable

from .event_mnist import torch
from .runtime import device_kind, resolve_device, seed_everything
from .shd_benchmark import (
    SHDConfig,
    SHDEventCountClassifier,
    SHDSparseClassifier,
    _measure,
    _train_model,
    load_shd_tensors,
)


@dataclass(frozen=True)
class SHDRepresentationArm:
    name: str
    model_kind: str
    readout_kind: str
    hidden_multiplier: int = 1
    delay_pattern: str = "none"
    max_delay_steps: int = 0
    threshold: float = 1.0


SHD_REPRESENTATION_ARMS = (
    SHDRepresentationArm("event_count_linear", "count", "linear"),
    SHDRepresentationArm("event_count_mlp", "count", "mlp"),
    SHDRepresentationArm("sparse128_linear_no_delay", "sparse", "linear"),
    SHDRepresentationArm("sparse128_mlp_no_delay", "sparse", "mlp"),
    SHDRepresentationArm(
        "sparse128_mlp_distance012",
        "sparse",
        "mlp",
        delay_pattern="distance_0_2",
        max_delay_steps=2,
    ),
    SHDRepresentationArm(
        "sparse256_mlp_distance012",
        "sparse",
        "mlp",
        hidden_multiplier=2,
        delay_pattern="distance_0_2",
        max_delay_steps=2,
    ),
    SHDRepresentationArm(
        "sparse128_mlp_distance012_threshold1p5",
        "sparse",
        "mlp",
        delay_pattern="distance_0_2",
        max_delay_steps=2,
        threshold=1.5,
    ),
)


def available_shd_representation_arms() -> tuple[str, ...]:
    return tuple(arm.name for arm in SHD_REPRESENTATION_ARMS)


@dataclass
class SHDRepresentationResult:
    config: SHDConfig
    device: str
    surrogate_slope: float
    arms: list[dict]
    records: list[dict]
    summary: list[dict]

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "shd_representation.json"
        records_path = output / "shd_representation_records.csv"
        summary_path = output / "shd_representation_summary.csv"
        payload = {
            "config": asdict(self.config),
            "device": self.device,
            "surrogate_slope": self.surrogate_slope,
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
            plot_path = output / "shd_representation_summary.png"
            plot_shd_representation(self.summary, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def run_shd_representation_diagnostic(
    config: SHDConfig,
    *,
    device="auto",
    surrogate_slope: float = 10.0,
    arm_names: Iterable[str] | None = None,
    readout_hidden_units: int = 128,
    ltw_minimum: float = 0.0,
    ltw_maximum: float = 1.0,
) -> SHDRepresentationResult:
    if torch is None:
        raise ImportError("Phase 32 SHD representation diagnostic requires PyTorch")
    if readout_hidden_units <= 0:
        raise ValueError("readout_hidden_units must be positive")
    arms = _select_arms(arm_names)
    resolved = resolve_device(device)
    train_events, train_labels, test_events, test_labels = load_shd_tensors(config)
    records: list[dict] = []
    for seed in config.seeds:
        for arm in arms:
            seed_everything(seed, device=resolved)
            arm_config = replace(
                config,
                hidden_neurons=config.hidden_neurons * arm.hidden_multiplier,
                reservoir_threshold=arm.threshold,
            )
            required_edges = (
                arm_config.input_neurons * arm_config.sensor_fanout
                + arm_config.hidden_neurons * arm_config.recurrent_fanout
            )
            if required_edges > arm_config.max_edges:
                arm_config = replace(arm_config, max_edges=_next_power_of_two(required_edges))
            if arm.model_kind == "count":
                model = SHDEventCountClassifier(config, kind=arm.readout_kind).to(resolved)
                initial_ltw = None
                initial_event_rate = 0.0
            else:
                model = SHDSparseClassifier(
                    arm_config,
                    seed=seed,
                    delay_pattern=arm.delay_pattern,
                    max_delay_steps=arm.max_delay_steps,
                    surrogate_slope=surrogate_slope,
                    readout_kind=arm.readout_kind,
                    readout_hidden_units=readout_hidden_units,
                    device=resolved,
                ).to(resolved)
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
            if arm.model_kind == "count":
                # Count controls have no hidden spiking state. Their input-event
                # density is not comparable to the sparse hidden event rate.
                final_event_rate = 0.0
            active_edges = delayed_edges = 0
            mean_ltw_change = lower_saturation = upper_saturation = 0.0
            if isinstance(model, SHDSparseClassifier):
                active = model.graph.active_mask
                recurrent = active & (model.graph.sources >= arm_config.input_neurons)
                active_edges = int(active.sum().item())
                delayed_edges = int((model.graph.delay_steps[recurrent] > 0).sum().item())
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
            allocated_parameters = sum(
                parameter.numel() for parameter in model.parameters()
                if parameter.requires_grad
            )
            effective_parameters = allocated_parameters
            if isinstance(model, SHDSparseClassifier):
                effective_parameters = sum(
                    parameter.numel() for parameter in model.readout.parameters()
                ) + active_edges
            records.append(
                {
                    "seed": int(seed),
                    "arm": arm.name,
                    "model_kind": arm.model_kind,
                    "readout_kind": arm.readout_kind,
                    "hidden_neurons": int(arm_config.hidden_neurons),
                    "reservoir_threshold": float(arm_config.reservoir_threshold),
                    "delay_pattern": arm.delay_pattern,
                    "train_accuracy": float(train_accuracy),
                    "test_accuracy": float(test_accuracy),
                    "active_edges": int(active_edges),
                    "delayed_edges": int(delayed_edges),
                    "effective_trainable_parameters": int(effective_parameters),
                    "allocated_trainable_parameters": int(allocated_parameters),
                    "initial_hidden_event_rate": float(initial_event_rate),
                    "final_hidden_event_rate": float(final_event_rate),
                    "event_rate_ratio": float(
                        final_event_rate / max(initial_event_rate, 1e-12)
                        if initial_event_rate > 0 else 0.0
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
    _attach_diagnostic_comparisons(records)
    return SHDRepresentationResult(
        config=config,
        device=device_kind(resolved),
        surrogate_slope=float(surrogate_slope),
        arms=[asdict(arm) for arm in arms],
        records=records,
        summary=summarize_shd_representation(records, arms=arms),
    )


def summarize_shd_representation(
    records: Iterable[dict],
    *,
    arms: Iterable[SHDRepresentationArm] = SHD_REPRESENTATION_ARMS,
) -> list[dict]:
    rows = list(records)
    summary: list[dict] = []
    for arm in arms:
        group = [row for row in rows if row["arm"] == arm.name]
        if not group:
            continue
        summary.append(
            {
                "arm": arm.name,
                "model_kind": arm.model_kind,
                "readout_kind": arm.readout_kind,
                "seeds": len(group),
                "hidden_neurons": int(group[0]["hidden_neurons"]),
                "reservoir_threshold": float(group[0]["reservoir_threshold"]),
                "mean_test_accuracy": statistics.fmean(
                    float(row["test_accuracy"]) for row in group
                ),
                "std_test_accuracy": statistics.pstdev(
                    float(row["test_accuracy"]) for row in group
                ),
                "mean_gain_vs_sparse_linear": statistics.fmean(
                    float(row["gain_vs_sparse_linear"]) for row in group
                ),
                "mean_gain_vs_mlp_no_delay": statistics.fmean(
                    float(row["gain_vs_mlp_no_delay"]) for row in group
                ),
                "mean_gain_vs_base_distance": statistics.fmean(
                    float(row["gain_vs_base_distance"]) for row in group
                ),
                "improved_seed_count_vs_relevant_control": sum(
                    float(row["gain_vs_relevant_control"]) > 0.0 for row in group
                ),
                "practical_seed_count_vs_relevant_control": sum(
                    float(row["gain_vs_relevant_control"])
                    >= _practical_threshold(arm.name)
                    for row in group
                ),
                "mean_final_hidden_event_rate": statistics.fmean(
                    float(row["final_hidden_event_rate"]) for row in group
                ),
                "mean_event_rate_ratio": statistics.fmean(
                    float(row["event_rate_ratio"]) for row in group
                ),
                "active_edges": int(group[0]["active_edges"]),
                "effective_trainable_parameters": int(
                    group[0]["effective_trainable_parameters"]
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
            }
        )
    return summary


def plot_shd_representation(summary: list[dict], path: str | pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    names = [row["arm"] for row in summary]
    accuracy = [100.0 * float(row["mean_test_accuracy"]) for row in summary]
    errors = [100.0 * float(row["std_test_accuracy"]) for row in summary]
    event_rates = [100.0 * float(row["mean_final_hidden_event_rate"]) for row in summary]
    parameters = [int(row["effective_trainable_parameters"]) for row in summary]
    figure, axes = plt.subplots(3, 1, figsize=(15, 13), constrained_layout=True)
    axes[0].bar(names, accuracy, yerr=errors, color="#35b4f2", capsize=5)
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_title("AMMC Gen-5 Phase 32: SHD representation diagnostic")
    axes[1].bar(names, event_rates, color="#ffb31a")
    axes[1].set_ylabel("Final hidden event rate (%)")
    axes[2].bar(names, parameters, color="#48c78e")
    axes[2].set_ylabel("Effective trainable parameters")
    axes[2].set_yscale("log")
    for axis in axes:
        axis.tick_params(axis="x", rotation=18)
        axis.grid(axis="y", alpha=0.25)
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _attach_diagnostic_comparisons(records: list[dict]) -> None:
    by_seed_arm = {(int(row["seed"]), row["arm"]): row for row in records}
    for row in records:
        seed = int(row["seed"])
        linear = by_seed_arm.get((seed, "sparse128_linear_no_delay"))
        mlp_no_delay = by_seed_arm.get((seed, "sparse128_mlp_no_delay"))
        base_distance = by_seed_arm.get((seed, "sparse128_mlp_distance012"))
        row["gain_vs_sparse_linear"] = _accuracy_delta(row, linear)
        row["gain_vs_mlp_no_delay"] = _accuracy_delta(row, mlp_no_delay)
        row["gain_vs_base_distance"] = _accuracy_delta(row, base_distance)
        if row["arm"] == "sparse128_mlp_no_delay":
            relevant = linear
        elif row["arm"] == "sparse128_mlp_distance012":
            relevant = mlp_no_delay
        elif row["arm"] in {
            "sparse256_mlp_distance012",
            "sparse128_mlp_distance012_threshold1p5",
        }:
            relevant = base_distance
        else:
            relevant = row
        row["gain_vs_relevant_control"] = _accuracy_delta(row, relevant)


def _accuracy_delta(row: dict, control: dict | None) -> float:
    if control is None:
        return 0.0
    return float(row["test_accuracy"]) - float(control["test_accuracy"])


def _practical_threshold(arm_name: str) -> float:
    if arm_name == "sparse128_mlp_distance012":
        return 0.01
    if arm_name in {
        "sparse128_mlp_no_delay",
        "sparse256_mlp_distance012",
    }:
        return 0.03
    if arm_name == "sparse128_mlp_distance012_threshold1p5":
        return 0.02
    return float("inf")


def _select_arms(names: Iterable[str] | None) -> tuple[SHDRepresentationArm, ...]:
    if names is None:
        return SHD_REPRESENTATION_ARMS
    lookup = {arm.name: arm for arm in SHD_REPRESENTATION_ARMS}
    selected = []
    for name in names:
        if name not in lookup:
            raise ValueError(f"unknown SHD representation arm: {name}")
        selected.append(lookup[name])
    if not selected:
        raise ValueError("at least one SHD representation arm is required")
    return tuple(selected)


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
