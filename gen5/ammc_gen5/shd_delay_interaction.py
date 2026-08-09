"""Phase 35 factorial SHD capacity-by-delay interaction.

Phase 34 found a variable distance-delay gain at 256 neurons despite negligible
gains at 128. This runner pairs two capacity levels with uniform and two
heterogeneous delay patterns to distinguish a real capacity interaction from
generic slowing or seed noise.
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
    SHDSparseClassifier,
    _measure,
    _train_model,
    load_shd_tensors,
)


DELAY_PATTERNS = {
    "none": 0,
    "uniform_1": 1,
    "hash_0_2": 2,
    "distance_0_2": 2,
}


@dataclass(frozen=True)
class SHDDelayInteractionArm:
    name: str
    hidden_neurons: int
    delay_pattern: str
    max_delay_steps: int


def default_shd_delay_interaction_arms(
    hidden_counts: Iterable[int] = (256, 512),
) -> tuple[SHDDelayInteractionArm, ...]:
    counts = tuple(int(value) for value in hidden_counts)
    if not counts or any(value <= 1 for value in counts):
        raise ValueError("hidden counts must contain values greater than one")
    if len(set(counts)) != len(counts):
        raise ValueError("hidden counts must be unique")
    return tuple(
        SHDDelayInteractionArm(
            name=f"sparse{hidden}_mlp_{pattern}",
            hidden_neurons=hidden,
            delay_pattern=pattern,
            max_delay_steps=max_delay,
        )
        for hidden in counts
        for pattern, max_delay in DELAY_PATTERNS.items()
    )


@dataclass
class SHDDelayInteractionResult:
    config: SHDConfig
    device: str
    hidden_counts: tuple[int, ...]
    readout_hidden_units: int
    arms: list[dict]
    records: list[dict]
    summary: list[dict]

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "shd_delay_interaction.json"
        records_path = output / "shd_delay_interaction_records.csv"
        summary_path = output / "shd_delay_interaction_summary.csv"
        payload = {
            "config": asdict(self.config),
            "device": self.device,
            "hidden_counts": list(self.hidden_counts),
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
            plot_path = output / "shd_delay_interaction_summary.png"
            plot_shd_delay_interaction(self.summary, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def run_shd_delay_interaction(
    config: SHDConfig,
    *,
    hidden_counts: Iterable[int] = (256, 512),
    device="auto",
    surrogate_slope: float = 10.0,
    readout_hidden_units: int = 128,
    ltw_minimum: float = 0.0,
    ltw_maximum: float = 1.0,
) -> SHDDelayInteractionResult:
    if torch is None:
        raise ImportError("Phase 35 SHD delay interaction requires PyTorch")
    counts = tuple(int(value) for value in hidden_counts)
    arms = default_shd_delay_interaction_arms(counts)
    if readout_hidden_units <= 0:
        raise ValueError("readout_hidden_units must be positive")
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
    _attach_no_delay_comparisons(records)
    return SHDDelayInteractionResult(
        config=config,
        device=device_kind(resolved),
        hidden_counts=counts,
        readout_hidden_units=int(readout_hidden_units),
        arms=[asdict(arm) for arm in arms],
        records=records,
        summary=summarize_shd_delay_interaction(records, arms=arms),
    )


def summarize_shd_delay_interaction(
    records: Iterable[dict], *, arms: Iterable[SHDDelayInteractionArm]
) -> list[dict]:
    rows = list(records)
    summary: list[dict] = []
    for arm in arms:
        group = [row for row in rows if row["arm"] == arm.name]
        if not group:
            continue
        gains = [float(row["gain_vs_same_width_no_delay"]) for row in group]
        rate_ratios = [float(row["event_rate_vs_same_width_no_delay"]) for row in group]
        summary.append(
            {
                "arm": arm.name,
                "hidden_neurons": int(arm.hidden_neurons),
                "delay_pattern": arm.delay_pattern,
                "seeds": len(group),
                "mean_test_accuracy": statistics.fmean(
                    float(row["test_accuracy"]) for row in group
                ),
                "std_test_accuracy": statistics.pstdev(
                    float(row["test_accuracy"]) for row in group
                ),
                "mean_gain_vs_same_width_no_delay": statistics.fmean(gains),
                "improved_seed_count": sum(gain > 0 for gain in gains),
                "one_point_seed_count": sum(gain >= 0.01 for gain in gains),
                "two_point_seed_count": sum(gain >= 0.02 for gain in gains),
                "mean_final_hidden_event_rate": statistics.fmean(
                    float(row["final_hidden_event_rate"]) for row in group
                ),
                "mean_event_rate_vs_same_width_no_delay": statistics.fmean(
                    rate_ratios
                ),
                "active_edges": int(group[0]["active_edges"]),
                "mean_delayed_edges": statistics.fmean(
                    int(row["delayed_edges"]) for row in group
                ),
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
                "mean_test_examples_per_second": statistics.fmean(
                    float(row["test_examples_per_second"]) for row in group
                ),
            }
        )
    return summary


def plot_shd_delay_interaction(summary: list[dict], path: str | pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    widths = sorted({int(row["hidden_neurons"]) for row in summary})
    patterns = tuple(DELAY_PATTERNS)
    lookup = {
        (int(row["hidden_neurons"]), row["delay_pattern"]): row for row in summary
    }
    colors = ("#35b4f2", "#8b6fd6", "#48c78e", "#ffb31a")
    figure, axes = plt.subplots(3, 1, figsize=(14, 13), constrained_layout=True)
    x = list(range(len(widths)))
    bar_width = 0.19
    for index, (pattern, color) in enumerate(zip(patterns, colors)):
        offsets = [value + (index - 1.5) * bar_width for value in x]
        axes[0].bar(
            offsets,
            [100.0 * float(lookup[(width, pattern)]["mean_test_accuracy"]) for width in widths],
            bar_width,
            label=pattern,
            color=color,
        )
        axes[1].bar(
            offsets,
            [
                100.0 * float(
                    lookup[(width, pattern)]["mean_gain_vs_same_width_no_delay"]
                )
                for width in widths
            ],
            bar_width,
            color=color,
        )
        axes[2].bar(
            offsets,
            [
                float(lookup[(width, pattern)]["mean_test_examples_per_second"])
                for width in widths
            ],
            bar_width,
            color=color,
        )
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_title("AMMC Gen-5 Phase 35: SHD capacity-delay interaction")
    axes[0].legend(ncol=2)
    axes[1].set_ylabel("Gain vs same-width no delay (points)")
    axes[1].axhline(2.0, color="#bd3d3a", linestyle="--", label="+2 point gate")
    axes[1].legend()
    axes[2].set_ylabel("Test examples / second")
    for axis in axes:
        axis.set_xticks(x, [str(width) for width in widths])
        axis.grid(axis="y", alpha=0.25)
    axes[2].set_xlabel("Hidden neurons")
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _attach_no_delay_comparisons(records: list[dict]) -> None:
    controls = {
        (int(row["seed"]), int(row["hidden_neurons"])): row
        for row in records
        if row["delay_pattern"] == "none"
    }
    for row in records:
        control = controls[(int(row["seed"]), int(row["hidden_neurons"]))]
        row["gain_vs_same_width_no_delay"] = float(row["test_accuracy"]) - float(
            control["test_accuracy"]
        )
        row["event_rate_vs_same_width_no_delay"] = float(
            row["final_hidden_event_rate"]
        ) / max(float(control["final_hidden_event_rate"]), 1e-12)


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
