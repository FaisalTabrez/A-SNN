"""Phase 43 validation-selected final audit of raw versus sparse SHD models."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
import json
import pathlib
import statistics
import time
from typing import Iterable

from .event_mnist import torch
from .runtime import device_kind, mark_step, resolve_device, seed_everything, sync
from .shd_benchmark import SHDConfig, _measure, load_shd_tensors
from .shd_sparse_width import FixedBudgetSparseAnalogClassifier
from .shd_temporal_controls import BudgetMatchedTemporalReadout, SHDRawTemporalPyramidClassifier, disable_shd_recurrent_edges
from .shd_temporal_pyramid import DEFAULT_TEMPORAL_LEVELS


@dataclass(frozen=True)
class SHDValidationCheckpointArm:
    name: str
    hidden_neurons: int


SHD_VALIDATION_CHECKPOINT_ARMS = (
    SHDValidationCheckpointArm("raw_temporal_pyramid", 0),
    SHDValidationCheckpointArm("sparse_analog_leaky_512", 512),
)


def available_shd_validation_checkpoint_arms() -> tuple[str, ...]:
    return tuple(arm.name for arm in SHD_VALIDATION_CHECKPOINT_ARMS)


@dataclass
class SHDValidationCheckpointResult:
    config: SHDConfig
    device: str
    topology_seeds: tuple[int, ...]
    readout_seeds: tuple[int, ...]
    validation_fraction: float
    temporal_levels: tuple[int, ...]
    target_parameters: int
    arms: list[dict]
    records: list[dict]
    summary: list[dict]

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "shd_validation_checkpoint.json"
        records_path = output / "shd_validation_checkpoint_records.csv"
        summary_path = output / "shd_validation_checkpoint_summary.csv"
        payload = {
            "config": asdict(self.config), "device": self.device,
            "topology_seeds": list(self.topology_seeds), "readout_seeds": list(self.readout_seeds),
            "validation_fraction": self.validation_fraction,
            "temporal_levels": list(self.temporal_levels), "target_parameters": self.target_parameters,
            "arms": self.arms, "records": self.records, "summary": self.summary,
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        _write_csv(records_path, self.records)
        _write_csv(summary_path, self.summary)
        paths = {"json": str(json_path), "records_csv": str(records_path), "summary_csv": str(summary_path)}
        if plot:
            plot_path = output / "shd_validation_checkpoint_summary.png"
            plot_shd_validation_checkpoint(self.summary, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def run_shd_validation_checkpoint(
    config: SHDConfig,
    *,
    topology_seeds: Iterable[int] = (42, 43, 44),
    readout_seeds: Iterable[int] = (142, 143, 144),
    validation_fraction: float = 0.10,
    target_parameters: int = 133_631,
    device="auto",
    surrogate_slope: float = 10.0,
    projection_dim: int = 32,
    temporal_levels: Iterable[int] = DEFAULT_TEMPORAL_LEVELS,
    readout_hidden_units: int = 128,
) -> SHDValidationCheckpointResult:
    if torch is None:
        raise ImportError("Phase 43 validation checkpointing requires PyTorch")
    topology_values = tuple(int(seed) for seed in topology_seeds)
    readout_values = tuple(int(seed) for seed in readout_seeds)
    levels = tuple(int(level) for level in temporal_levels)
    if not topology_values or not readout_values or not 0.0 < validation_fraction < 0.5:
        raise ValueError("invalid seed matrix or validation fraction")
    resolved = resolve_device(device)
    all_train_events, all_train_labels, test_events, test_labels = load_shd_tensors(config)
    train_events, train_labels, validation_events, validation_labels = _stratified_split(
        all_train_events, all_train_labels,
        fraction=validation_fraction, seed=config.data_seed + 43_000,
    )
    records: list[dict] = []

    for readout_seed in readout_values:
        seed_everything(readout_seed, device=resolved)
        model = SHDRawTemporalPyramidClassifier(
            config, projection_dim=projection_dim, temporal_levels=levels,
            target_parameters=target_parameters,
        ).to(resolved)
        records.append(
            _fit_record(
                model, config, train_events, train_labels, validation_events,
                validation_labels, test_events, test_labels,
                arm=SHD_VALIDATION_CHECKPOINT_ARMS[0], topology_seed=-1,
                readout_seed=readout_seed, target_parameters=target_parameters,
                active_edges=0, resolved=resolved,
            )
        )

    arm = SHD_VALIDATION_CHECKPOINT_ARMS[1]
    for topology_seed in topology_values:
        for readout_seed in readout_values:
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
            active_edges = int(model.graph.active_mask.sum().item())
            model.graph.long_term_weight.requires_grad_(False)
            seed_everything(readout_seed, device=resolved)
            model.readout = BudgetMatchedTemporalReadout(
                trace_dim=arm.hidden_neurons, final_dim=arm.hidden_neurons,
                classes=config.classes, projection_dim=projection_dim,
                temporal_levels=levels,
                target_parameters=target_parameters - active_edges,
            ).to(resolved)
            records.append(
                _fit_record(
                    model, arm_config, train_events, train_labels,
                    validation_events, validation_labels, test_events, test_labels,
                    arm=arm, topology_seed=topology_seed,
                    readout_seed=readout_seed, target_parameters=target_parameters,
                    active_edges=active_edges, resolved=resolved,
                )
            )
    _attach_comparisons(records)
    return SHDValidationCheckpointResult(
        config=config, device=device_kind(resolved), topology_seeds=topology_values,
        readout_seeds=readout_values, validation_fraction=float(validation_fraction),
        temporal_levels=levels, target_parameters=int(target_parameters),
        arms=[asdict(arm) for arm in SHD_VALIDATION_CHECKPOINT_ARMS], records=records,
        summary=summarize_shd_validation_checkpoint(records),
    )


def _fit_record(
    model, config, train_events, train_labels, validation_events,
    validation_labels, test_events, test_labels, *,
    arm: SHDValidationCheckpointArm, topology_seed: int, readout_seed: int,
    target_parameters: int, active_edges: int, resolved,
) -> dict:
    training = _train_validation_selected(
        model, train_events, train_labels, validation_events, validation_labels,
        config, seed=readout_seed, device=resolved,
    )
    final_test_accuracy, final_inference_seconds, final_activity = _measure(
        model, test_events, test_labels, config.batch_size, resolved
    )
    model.load_state_dict(training["best_state"])
    checkpoint_test_accuracy, checkpoint_inference_seconds, checkpoint_activity = _measure(
        model, test_events, test_labels, config.batch_size, resolved
    )
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    effective_parameters = trainable_parameters + active_edges
    return {
        "arm": arm.name, "hidden_neurons": arm.hidden_neurons,
        "topology_seed": int(topology_seed), "readout_seed": int(readout_seed),
        "train_samples": int(train_events.shape[0]), "validation_samples": int(validation_events.shape[0]),
        "best_epoch": int(training["best_epoch"]), "best_validation_accuracy": float(training["best_validation_accuracy"]),
        "final_validation_accuracy": float(training["final_validation_accuracy"]),
        "final_test_accuracy": float(final_test_accuracy),
        "checkpoint_test_accuracy": float(checkpoint_test_accuracy),
        "checkpoint_gain_vs_final": float(checkpoint_test_accuracy - final_test_accuracy),
        "active_edges": int(active_edges), "effective_model_parameters": int(effective_parameters),
        "parameter_ratio_vs_target": float(effective_parameters / target_parameters),
        "final_activity": float(final_activity), "checkpoint_activity": float(checkpoint_activity),
        "train_seconds": float(training["train_seconds"]),
        "final_inference_seconds": float(final_inference_seconds),
        "checkpoint_inference_seconds": float(checkpoint_inference_seconds),
    }


def _train_validation_selected(model, train_events, train_labels, validation_events, validation_labels, config, *, seed: int, device) -> dict:
    parameters = list(model.parameters())
    if hasattr(model, "readout") and hasattr(model, "graph"):
        parameters = list(model.readout.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(seed + 50_000)
    best_accuracy = -1.0
    best_epoch = 0
    best_state = None
    final_validation = 0.0
    sync(device)
    start = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        order = torch.randperm(train_events.shape[0], generator=generator)
        for offset in range(0, train_events.shape[0], config.batch_size):
            index = order[offset : offset + config.batch_size]
            batch_events = train_events.index_select(0, index).to(device)
            batch_labels = train_labels.index_select(0, index).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(batch_events), batch_labels)
            loss.backward()
            optimizer.step()
            mark_step(device)
        final_validation, _, _ = _measure(
            model, validation_events, validation_labels, config.batch_size, device
        )
        if final_validation > best_accuracy:
            best_accuracy = float(final_validation)
            best_epoch = epoch + 1
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    sync(device)
    return {
        "best_state": best_state, "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "final_validation_accuracy": float(final_validation),
        "train_seconds": time.perf_counter() - start,
    }


def _stratified_split(events, labels, *, fraction: float, seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    train_indices = []
    validation_indices = []
    for label in torch.unique(labels).tolist():
        indices = torch.nonzero(labels == int(label), as_tuple=False).flatten()
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        count = max(1, int(round(indices.numel() * fraction)))
        validation_indices.append(indices[:count])
        train_indices.append(indices[count:])
    train_index = torch.cat(train_indices)
    validation_index = torch.cat(validation_indices)
    return (
        events.index_select(0, train_index), labels.index_select(0, train_index),
        events.index_select(0, validation_index), labels.index_select(0, validation_index),
    )


def summarize_shd_validation_checkpoint(records: Iterable[dict]) -> list[dict]:
    rows = list(records)
    summary: list[dict] = []
    for arm in SHD_VALIDATION_CHECKPOINT_ARMS:
        group = [row for row in rows if row["arm"] == arm.name]
        if not group:
            continue
        gains = [float(row["checkpoint_gain_vs_raw"]) for row in group]
        final_values = [float(row["final_test_accuracy"]) for row in group]
        checkpoint_values = [float(row["checkpoint_test_accuracy"]) for row in group]
        summary.append(
            {
                "arm": arm.name, "hidden_neurons": arm.hidden_neurons, "runs": len(group),
                "mean_final_test_accuracy": statistics.fmean(final_values),
                "std_final_test_accuracy": statistics.pstdev(final_values),
                "mean_checkpoint_test_accuracy": statistics.fmean(checkpoint_values),
                "std_checkpoint_test_accuracy": statistics.pstdev(checkpoint_values),
                "mean_checkpoint_gain_vs_final": statistics.fmean(float(row["checkpoint_gain_vs_final"]) for row in group),
                "mean_checkpoint_gain_vs_raw": statistics.fmean(gains),
                "positive_pair_count_vs_raw": sum(gain > 0.0 for gain in gains),
                "two_point_pair_count_vs_raw": sum(gain >= 0.02 for gain in gains),
                "mean_best_epoch": statistics.fmean(int(row["best_epoch"]) for row in group),
                "mean_best_validation_accuracy": statistics.fmean(float(row["best_validation_accuracy"]) for row in group),
                "effective_model_parameters": int(group[0]["effective_model_parameters"]),
                "parameter_ratio_vs_target": statistics.fmean(float(row["parameter_ratio_vs_target"]) for row in group),
                "mean_train_seconds": statistics.fmean(float(row["train_seconds"]) for row in group),
            }
        )
    return summary


def plot_shd_validation_checkpoint(summary: list[dict], path: str | pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    labels = [row["arm"].replace("_", "\n") for row in summary]
    x = list(range(len(summary)))
    final = [100.0 * float(row["mean_final_test_accuracy"]) for row in summary]
    checkpoint = [100.0 * float(row["mean_checkpoint_test_accuracy"]) for row in summary]
    errors = [100.0 * float(row["std_checkpoint_test_accuracy"]) for row in summary]
    gains = [100.0 * float(row["mean_checkpoint_gain_vs_raw"]) for row in summary]
    epochs = [float(row["mean_best_epoch"]) for row in summary]
    figure, axes = plt.subplots(3, 1, figsize=(14, 13), constrained_layout=True)
    width = 0.35
    axes[0].bar([value - width / 2 for value in x], final, width, label="Final epoch", color="#8b6fd6")
    axes[0].bar([value + width / 2 for value in x], checkpoint, width, yerr=errors, capsize=5, label="Best validation", color="#167d55")
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_title("AMMC Gen-5 Phase 43: validation-selected checkpointing")
    axes[0].legend()
    axes[1].bar(x, gains, color=("#ffb31a", "#167d55"))
    axes[1].axhline(2.0, color="#bd3d3a", linestyle="--", label="+2 point sparse gate")
    axes[1].set_ylabel("Checkpoint gain vs paired raw (points)")
    axes[1].legend()
    axes[2].bar(x, epochs, color=("#ffb31a", "#35b4f2"))
    axes[2].set_ylabel("Mean best epoch")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25)
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _attach_comparisons(records: list[dict]) -> None:
    raw = {int(row["readout_seed"]): row for row in records if int(row["hidden_neurons"]) == 0}
    for row in records:
        reference = raw[int(row["readout_seed"])]
        row["checkpoint_gain_vs_raw"] = float(row["checkpoint_test_accuracy"]) - float(reference["checkpoint_test_accuracy"])


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
