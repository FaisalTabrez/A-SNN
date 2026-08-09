"""Phase 34 no-delay hidden-capacity scaling on SHD.

The preceding representation diagnostic showed a large 128-to-256 neuron gain
but used fixed delays in the wide arm. This runner validates that finding under
no-delay controls, locates the capacity/efficiency knee, and retains exactly one
256-neuron delay comparator so the delay hypothesis remains falsifiable.
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
class SHDCapacityArm:
    name: str
    hidden_neurons: int
    delay_pattern: str = "none"
    max_delay_steps: int = 0
    is_count_control: bool = False


def default_shd_capacity_arms(
    hidden_counts: Iterable[int] = (128, 192, 256, 384, 512),
    *,
    delay_anchor: int = 256,
) -> tuple[SHDCapacityArm, ...]:
    counts = tuple(int(value) for value in hidden_counts)
    if not counts or any(value <= 1 for value in counts):
        raise ValueError("hidden counts must contain values greater than one")
    if len(set(counts)) != len(counts):
        raise ValueError("hidden counts must be unique")
    if delay_anchor not in counts:
        raise ValueError("delay anchor must be one of the hidden counts")
    arms = [SHDCapacityArm("event_count_mlp", 0, is_count_control=True)]
    arms.extend(
        SHDCapacityArm(f"sparse{count}_mlp_no_delay", count)
        for count in counts
    )
    arms.append(
        SHDCapacityArm(
            f"sparse{delay_anchor}_mlp_distance012",
            delay_anchor,
            delay_pattern="distance_0_2",
            max_delay_steps=2,
        )
    )
    return tuple(arms)


@dataclass
class SHDCapacityResult:
    config: SHDConfig
    device: str
    hidden_counts: tuple[int, ...]
    delay_anchor: int
    readout_hidden_units: int
    arms: list[dict]
    records: list[dict]
    summary: list[dict]

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "shd_capacity.json"
        records_path = output / "shd_capacity_records.csv"
        summary_path = output / "shd_capacity_summary.csv"
        payload = {
            "config": asdict(self.config),
            "device": self.device,
            "hidden_counts": list(self.hidden_counts),
            "delay_anchor": self.delay_anchor,
            "readout_hidden_units": self.readout_hidden_units,
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
            plot_path = output / "shd_capacity_summary.png"
            plot_shd_capacity(self.summary, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def run_shd_capacity_scaling(
    config: SHDConfig,
    *,
    hidden_counts: Iterable[int] = (128, 192, 256, 384, 512),
    delay_anchor: int = 256,
    device="auto",
    surrogate_slope: float = 10.0,
    readout_hidden_units: int = 128,
    ltw_minimum: float = 0.0,
    ltw_maximum: float = 1.0,
) -> SHDCapacityResult:
    if torch is None:
        raise ImportError("Phase 34 SHD capacity scaling requires PyTorch")
    counts = tuple(int(value) for value in hidden_counts)
    arms = default_shd_capacity_arms(counts, delay_anchor=delay_anchor)
    if readout_hidden_units <= 0:
        raise ValueError("readout_hidden_units must be positive")
    resolved = resolve_device(device)
    train_events, train_labels, test_events, test_labels = load_shd_tensors(config)
    records: list[dict] = []
    for seed in config.seeds:
        for arm in arms:
            seed_everything(seed, device=resolved)
            if arm.is_count_control:
                arm_config = config
                model = SHDEventCountClassifier(config, kind="mlp").to(resolved)
                initial_ltw = None
                initial_event_rate = 0.0
            else:
                required_edges = (
                    config.input_neurons * config.sensor_fanout
                    + arm.hidden_neurons * config.recurrent_fanout
                )
                arm_config = replace(
                    config,
                    hidden_neurons=arm.hidden_neurons,
                    max_edges=max(config.max_edges, _next_power_of_two(required_edges)),
                )
                model = SHDSparseClassifier(
                    arm_config,
                    seed=seed,
                    delay_pattern=arm.delay_pattern,
                    max_delay_steps=arm.max_delay_steps,
                    surrogate_slope=surrogate_slope,
                    readout_kind="mlp",
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
            if arm.is_count_control:
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
                    "hidden_neurons": int(arm.hidden_neurons),
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
    _attach_capacity_comparisons(records, delay_anchor=delay_anchor)
    return SHDCapacityResult(
        config=config,
        device=device_kind(resolved),
        hidden_counts=counts,
        delay_anchor=int(delay_anchor),
        readout_hidden_units=int(readout_hidden_units),
        arms=[asdict(arm) for arm in arms],
        records=records,
        summary=summarize_shd_capacity(records, arms=arms),
    )


def summarize_shd_capacity(
    records: Iterable[dict], *, arms: Iterable[SHDCapacityArm]
) -> list[dict]:
    rows = list(records)
    summary: list[dict] = []
    for arm in arms:
        group = [row for row in rows if row["arm"] == arm.name]
        if not group:
            continue
        parameters = int(group[0]["effective_trainable_parameters"])
        accuracy = statistics.fmean(float(row["test_accuracy"]) for row in group)
        summary.append(
            {
                "arm": arm.name,
                "hidden_neurons": int(arm.hidden_neurons),
                "delay_pattern": arm.delay_pattern,
                "seeds": len(group),
                "mean_test_accuracy": accuracy,
                "std_test_accuracy": statistics.pstdev(
                    float(row["test_accuracy"]) for row in group
                ),
                "mean_gain_vs_128_no_delay": statistics.fmean(
                    float(row["gain_vs_128_no_delay"]) for row in group
                ),
                "improved_seed_count_vs_128": sum(
                    float(row["gain_vs_128_no_delay"]) > 0 for row in group
                ),
                "primary_gain_seed_count_vs_128": sum(
                    float(row["gain_vs_128_no_delay"]) >= 0.08 for row in group
                ),
                "mean_gain_vs_256_no_delay": statistics.fmean(
                    float(row["gain_vs_256_no_delay"]) for row in group
                ),
                "mean_gain_vs_same_scale_no_delay": statistics.fmean(
                    float(row["gain_vs_same_scale_no_delay"]) for row in group
                ),
                "improved_seed_count_vs_256": sum(
                    float(row["gain_vs_256_no_delay"]) > 0 for row in group
                ),
                "practical_seed_count_vs_256": sum(
                    float(row["gain_vs_256_no_delay"]) >= 0.02 for row in group
                ),
                "mean_final_hidden_event_rate": statistics.fmean(
                    float(row["final_hidden_event_rate"]) for row in group
                ),
                "active_edges": int(group[0]["active_edges"]),
                "effective_trainable_parameters": parameters,
                "accuracy_per_1k_effective_parameters": float(
                    accuracy / max(parameters / 1000.0, 1e-12)
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


def plot_shd_capacity(summary: list[dict], path: str | pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    no_delay = [
        row for row in summary
        if int(row["hidden_neurons"]) > 0 and row["delay_pattern"] == "none"
    ]
    no_delay.sort(key=lambda row: int(row["hidden_neurons"]))
    count = next((row for row in summary if row["arm"] == "event_count_mlp"), None)
    delay = next(
        (row for row in summary if row["delay_pattern"] == "distance_0_2"), None
    )
    hidden = [int(row["hidden_neurons"]) for row in no_delay]
    accuracy = [100.0 * float(row["mean_test_accuracy"]) for row in no_delay]
    errors = [100.0 * float(row["std_test_accuracy"]) for row in no_delay]
    event_rate = [100.0 * float(row["mean_final_hidden_event_rate"]) for row in no_delay]
    efficiency = [
        float(row["accuracy_per_1k_effective_parameters"]) for row in no_delay
    ]
    figure, axes = plt.subplots(3, 1, figsize=(13, 13), constrained_layout=True)
    axes[0].errorbar(hidden, accuracy, yerr=errors, marker="o", capsize=5)
    if count is not None:
        axes[0].axhline(
            100.0 * float(count["mean_test_accuracy"]),
            color="#48c78e",
            linestyle="--",
            label="event-count MLP",
        )
    if delay is not None:
        axes[0].scatter(
            [int(delay["hidden_neurons"])],
            [100.0 * float(delay["mean_test_accuracy"])],
            color="#bd3d3a",
            marker="x",
            s=90,
            label="256 fixed delays",
        )
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_title("AMMC Gen-5 Phase 34: SHD capacity scaling")
    axes[0].legend()
    axes[1].plot(hidden, event_rate, marker="o", color="#ffb31a")
    axes[1].set_ylabel("Hidden event rate (%)")
    axes[2].plot(hidden, efficiency, marker="o", color="#48c78e")
    axes[2].set_ylabel("Accuracy / 1k parameters")
    axes[2].set_xlabel("Hidden neurons")
    for axis in axes:
        axis.grid(alpha=0.25)
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _attach_capacity_comparisons(records: list[dict], *, delay_anchor: int) -> None:
    by_seed_arm = {(int(row["seed"]), row["arm"]): row for row in records}
    for row in records:
        seed = int(row["seed"])
        base128 = by_seed_arm.get((seed, "sparse128_mlp_no_delay"))
        base256 = by_seed_arm.get((seed, "sparse256_mlp_no_delay"))
        same_scale = by_seed_arm.get(
            (seed, f"sparse{int(row['hidden_neurons'])}_mlp_no_delay")
        )
        row["gain_vs_128_no_delay"] = _accuracy_delta(row, base128)
        row["gain_vs_256_no_delay"] = _accuracy_delta(row, base256)
        row["gain_vs_same_scale_no_delay"] = _accuracy_delta(row, same_scale)
        row["is_delay_anchor"] = bool(
            row["delay_pattern"] == "distance_0_2"
            and int(row["hidden_neurons"]) == delay_anchor
        )


def _accuracy_delta(row: dict, control: dict | None) -> float:
    if control is None:
        return 0.0
    return float(row["test_accuracy"]) - float(control["test_accuracy"])


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
