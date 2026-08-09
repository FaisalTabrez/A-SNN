"""Phase 39 causal ablation of the sparse feedforward SHD advantage.

Phase 38 showed that the sparse feedforward and recurrent systems both beat a
parameter-matched dense recurrent LIF.  Since recurrence did not explain the
gain, this experiment independently tests hard spiking and LTW optimization.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
import json
import pathlib
import statistics
from typing import Iterable

from .delayed_sequential_mnist import delayed_sparse_current
from .event_mnist import torch
from .runtime import device_kind, resolve_device, seed_everything
from .shd_benchmark import (
    SHDConfig,
    SHDSparseClassifier,
    _measure,
    _train_model,
    load_shd_tensors,
)
from .shd_temporal_controls import (
    SHDRawTemporalPyramidClassifier,
    disable_shd_recurrent_edges,
)
from .shd_temporal_pyramid import (
    DEFAULT_TEMPORAL_LEVELS,
    SHDTemporalPyramidClassifier,
    parameter_matched_bottleneck,
)


@dataclass(frozen=True)
class SHDSparseMechanismArm:
    name: str
    dynamics: str
    train_ltw: bool


SHD_SPARSE_MECHANISM_ARMS = (
    SHDSparseMechanismArm("raw_temporal_pyramid", "raw", False),
    SHDSparseMechanismArm("sparse_lif_frozen_ltw", "lif", False),
    SHDSparseMechanismArm("sparse_lif_trainable_ltw", "lif", True),
    SHDSparseMechanismArm("sparse_analog_frozen_ltw", "analog", False),
    SHDSparseMechanismArm("sparse_analog_trainable_ltw", "analog", True),
)


def available_shd_sparse_mechanism_arms() -> tuple[str, ...]:
    return tuple(arm.name for arm in SHD_SPARSE_MECHANISM_ARMS)


class SHDAnalogTemporalPyramidClassifier(SHDTemporalPyramidClassifier):
    """Sparse feedforward leaky analog control with the identical graph/readout."""

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
        zero_hidden = torch.zeros_like(membrane)
        zero_state = events.new_zeros(
            (events.shape[0], self.neuron_count), dtype=torch.float32
        )
        trace = []
        activity_sum = events.new_zeros((), dtype=torch.float32)
        for step in range(events.shape[1]):
            sensor = events[:, step].to(torch.float32) * self.config.input_gain
            network_state = torch.cat((sensor, zero_hidden), dim=1)
            current = delayed_sparse_current(
                self.graph,
                [network_state],
                zero_state=zero_state,
                max_delay_steps=0,
            )[:, self.input_neurons :]
            membrane = self.config.reservoir_leak * membrane + current
            activation = torch.tanh(membrane)
            trace.append(activation)
            activity_sum = activity_sum + activation.abs().mean()
        logits = self.readout(torch.stack(trace, dim=1), membrane)
        if return_event_rate:
            return logits, activity_sum / events.shape[1]
        return logits


@dataclass
class SHDSparseMechanismsResult:
    config: SHDConfig
    device: str
    hidden_neurons: int
    temporal_levels: tuple[int, ...]
    target_parameters: int
    arms: list[dict]
    records: list[dict]
    summary: list[dict]

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "shd_sparse_mechanisms.json"
        records_path = output / "shd_sparse_mechanisms_records.csv"
        summary_path = output / "shd_sparse_mechanisms_summary.csv"
        payload = {
            "config": asdict(self.config),
            "device": self.device,
            "hidden_neurons": self.hidden_neurons,
            "temporal_levels": list(self.temporal_levels),
            "target_parameters": self.target_parameters,
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
            plot_path = output / "shd_sparse_mechanisms_summary.png"
            plot_shd_sparse_mechanisms(self.summary, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def run_shd_sparse_mechanisms(
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
) -> SHDSparseMechanismsResult:
    if torch is None:
        raise ImportError("Phase 39 sparse mechanism ablation requires PyTorch")
    levels = tuple(int(level) for level in temporal_levels)
    if not levels or any(level <= 0 or level > config.timesteps for level in levels):
        raise ValueError("temporal levels must be between one and timesteps")
    required_edges = config.input_neurons * config.sensor_fanout + hidden_neurons * config.recurrent_fanout
    _, readout_parameters, _ = parameter_matched_bottleneck(
        hidden_neurons=hidden_neurons,
        classes=config.classes,
        projection_dim=projection_dim,
        temporal_levels=levels,
        baseline_hidden_units=readout_hidden_units,
    )
    target_parameters = readout_parameters + config.input_neurons * config.sensor_fanout
    arm_config = replace(
        config,
        hidden_neurons=hidden_neurons,
        max_edges=max(config.max_edges, _next_power_of_two(required_edges)),
    )
    resolved = resolve_device(device)
    train_events, train_labels, test_events, test_labels = load_shd_tensors(config)
    records: list[dict] = []
    for seed in config.seeds:
        for arm in SHD_SPARSE_MECHANISM_ARMS:
            seed_everything(seed, device=resolved)
            topology = "raw_events"
            activity_kind = "input_event_rate"
            initial_ltw = None
            if arm.dynamics == "raw":
                model = SHDRawTemporalPyramidClassifier(
                    arm_config,
                    projection_dim=projection_dim,
                    temporal_levels=levels,
                    target_parameters=target_parameters,
                ).to(resolved)
            else:
                model_class = (
                    SHDTemporalPyramidClassifier
                    if arm.dynamics == "lif"
                    else SHDAnalogTemporalPyramidClassifier
                )
                model = model_class(
                    arm_config,
                    seed=seed,
                    surrogate_slope=surrogate_slope,
                    projection_dim=projection_dim,
                    temporal_levels=levels,
                    baseline_hidden_units=readout_hidden_units,
                    temporal_order="ordered",
                    device=resolved,
                ).to(resolved)
                disable_shd_recurrent_edges(model)
                initial_ltw = model.graph.long_term_weight.detach().clone()
                topology = "sparse_feedforward"
                activity_kind = (
                    "hidden_spike_rate" if arm.dynamics == "lif" else "analog_activation"
                )
            initial_accuracy, _, initial_activity = _measure(
                model, test_events, test_labels, config.batch_size, resolved
            )
            training_config = replace(
                arm_config,
                reservoir_learning_rate=(
                    arm_config.reservoir_learning_rate if arm.train_ltw else 0.0
                ),
            )
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
            train_accuracy, _, _ = _measure(
                model, train_events, train_labels, config.batch_size, resolved
            )
            test_accuracy, inference_seconds, final_activity = _measure(
                model, test_events, test_labels, config.batch_size, resolved
            )
            active_edges = 0
            mean_ltw_change = 0.0
            effective_parameters = sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            )
            if isinstance(model, SHDSparseClassifier):
                active = model.graph.active_mask
                active_edges = int(active.sum().item())
                final_ltw = model.graph.long_term_weight.detach()
                mean_ltw_change = float(
                    (final_ltw[active] - initial_ltw[active]).abs().mean().item()
                )
                effective_parameters = sum(
                    parameter.numel() for parameter in model.readout.parameters()
                ) + (active_edges if arm.train_ltw else 0)
            records.append(
                {
                    "seed": int(seed),
                    "arm": arm.name,
                    "dynamics": arm.dynamics,
                    "train_ltw": bool(arm.train_ltw),
                    "topology": topology,
                    "initial_test_accuracy": float(initial_accuracy),
                    "train_accuracy": float(train_accuracy),
                    "test_accuracy": float(test_accuracy),
                    "active_edges": int(active_edges),
                    "effective_trainable_parameters": int(effective_parameters),
                    "parameter_ratio_vs_target": float(effective_parameters / target_parameters),
                    "initial_activity": float(initial_activity),
                    "final_activity": float(final_activity),
                    "activity_kind": activity_kind,
                    "mean_absolute_ltw_change": float(mean_ltw_change),
                    "train_seconds": float(train_seconds),
                    "inference_seconds": float(inference_seconds),
                    "test_examples_per_second": float(
                        test_events.shape[0] / max(inference_seconds, 1e-12)
                    ),
                }
            )
    _attach_mechanism_comparisons(records)
    return SHDSparseMechanismsResult(
        config=config,
        device=device_kind(resolved),
        hidden_neurons=int(hidden_neurons),
        temporal_levels=levels,
        target_parameters=int(target_parameters),
        arms=[asdict(arm) for arm in SHD_SPARSE_MECHANISM_ARMS],
        records=records,
        summary=summarize_shd_sparse_mechanisms(records),
    )


def summarize_shd_sparse_mechanisms(records: Iterable[dict]) -> list[dict]:
    rows = list(records)
    summary: list[dict] = []
    for arm in SHD_SPARSE_MECHANISM_ARMS:
        group = [row for row in rows if row["arm"] == arm.name]
        if not group:
            continue
        spiking_gains = [float(row["gain_vs_matched_analog"]) for row in group]
        ltw_gains = [float(row["gain_vs_frozen_ltw"]) for row in group]
        raw_gains = [float(row["gain_vs_raw_temporal"]) for row in group]
        summary.append(
            {
                "arm": arm.name,
                "dynamics": arm.dynamics,
                "train_ltw": arm.train_ltw,
                "seeds": len(group),
                "mean_test_accuracy": statistics.fmean(float(row["test_accuracy"]) for row in group),
                "std_test_accuracy": statistics.pstdev(float(row["test_accuracy"]) for row in group),
                "mean_gain_vs_raw_temporal": statistics.fmean(raw_gains),
                "mean_gain_vs_matched_analog": statistics.fmean(spiking_gains),
                "spiking_positive_seed_count": sum(gain > 0.0 for gain in spiking_gains),
                "spiking_one_point_seed_count": sum(gain >= 0.01 for gain in spiking_gains),
                "mean_gain_vs_frozen_ltw": statistics.fmean(ltw_gains),
                "ltw_positive_seed_count": sum(gain > 0.0 for gain in ltw_gains),
                "ltw_one_point_seed_count": sum(gain >= 0.01 for gain in ltw_gains),
                "active_edges": int(group[0]["active_edges"]),
                "effective_trainable_parameters": int(group[0]["effective_trainable_parameters"]),
                "parameter_ratio_vs_target": statistics.fmean(float(row["parameter_ratio_vs_target"]) for row in group),
                "mean_final_activity": statistics.fmean(float(row["final_activity"]) for row in group),
                "activity_kind": group[0]["activity_kind"],
                "mean_absolute_ltw_change": statistics.fmean(float(row["mean_absolute_ltw_change"]) for row in group),
                "mean_train_seconds": statistics.fmean(float(row["train_seconds"]) for row in group),
                "mean_test_examples_per_second": statistics.fmean(float(row["test_examples_per_second"]) for row in group),
            }
        )
    return summary


def plot_shd_sparse_mechanisms(summary: list[dict], path: str | pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    labels = [row["arm"].replace("_", "\n") for row in summary]
    accuracy = [100.0 * float(row["mean_test_accuracy"]) for row in summary]
    errors = [100.0 * float(row["std_test_accuracy"]) for row in summary]
    raw_gains = [100.0 * float(row["mean_gain_vs_raw_temporal"]) for row in summary]
    ltw_change = [float(row["mean_absolute_ltw_change"]) for row in summary]
    colors = ("#ffb31a", "#8b6fd6", "#167d55", "#9c7b5b", "#35b4f2")
    x = list(range(len(summary)))
    figure, axes = plt.subplots(3, 1, figsize=(16, 13), constrained_layout=True)
    axes[0].bar(x, accuracy, yerr=errors, capsize=5, color=colors)
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_title("AMMC Gen-5 Phase 39: sparse SHD mechanism ablation")
    axes[1].bar(x, raw_gains, color=colors)
    axes[1].axhline(2.0, color="#bd3d3a", linestyle="--", label="+2 point sparse gate")
    axes[1].set_ylabel("Gain vs raw temporal (points)")
    axes[1].legend()
    axes[2].bar(x, ltw_change, color=colors)
    axes[2].set_ylabel("Mean absolute LTW change")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25)
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _attach_mechanism_comparisons(records: list[dict]) -> None:
    lookup = {(int(row["seed"]), row["arm"]): row for row in records}
    for row in records:
        seed = int(row["seed"])
        raw = lookup[(seed, "raw_temporal_pyramid")]
        row["gain_vs_raw_temporal"] = float(row["test_accuracy"]) - float(raw["test_accuracy"])
        if row["dynamics"] in {"lif", "analog"}:
            suffix = "trainable_ltw" if row["train_ltw"] else "frozen_ltw"
            analog = lookup[(seed, f"sparse_analog_{suffix}")]
            frozen = lookup[(seed, f"sparse_{row['dynamics']}_frozen_ltw")]
            row["gain_vs_matched_analog"] = float(row["test_accuracy"]) - float(analog["test_accuracy"])
            row["gain_vs_frozen_ltw"] = float(row["test_accuracy"]) - float(frozen["test_accuracy"])
        else:
            row["gain_vs_matched_analog"] = 0.0
            row["gain_vs_frozen_ltw"] = 0.0


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
