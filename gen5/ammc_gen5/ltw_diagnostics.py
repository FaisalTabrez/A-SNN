"""Phase 22 diagnostics for stable fixed-topology LTW optimization.

Phase 21 showed that active LTWs receive gradient and move, but unrestricted
joint training did not reliably improve accuracy. This module keeps topology
fixed and diagnoses optimization schedule, learning rate, surrogate slope,
and sensor-versus-recurrent LTW scope with paired initializations.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import pathlib
import statistics
import time
from typing import Iterable

from .event_mnist import EventMNISTConfig, _matched_raw_hidden_units, load_mnist_tensors, nn, torch
from .runtime import device_kind, mark_step, resolve_device, seed_everything, sync
from .trainable_temporal_mnist import SparseTemporalClassifier, _measure, _readout_parameter_count


@dataclass(frozen=True)
class LTWDiagnosticArm:
    """One pre-registered LTW optimization intervention."""

    name: str
    schedule: str
    scope: str
    reservoir_learning_rate: float
    surrogate_slope: float


LTW_DIAGNOSTIC_ARMS = (
    LTWDiagnosticArm("frozen", "frozen", "none", 0.0, 10.0),
    LTWDiagnosticArm("joint_all_1em3_s10", "joint", "all", 1e-3, 10.0),
    LTWDiagnosticArm("warm_all_1em4_s5", "warmup", "all", 1e-4, 5.0),
    LTWDiagnosticArm("warm_all_3em4_s5", "warmup", "all", 3e-4, 5.0),
    LTWDiagnosticArm("warm_all_1em4_s10", "warmup", "all", 1e-4, 10.0),
    LTWDiagnosticArm("warm_all_3em4_s10", "warmup", "all", 3e-4, 10.0),
    LTWDiagnosticArm("warm_sensor_3em4_s10", "warmup", "sensor", 3e-4, 10.0),
    LTWDiagnosticArm("warm_recurrent_3em4_s10", "warmup", "recurrent", 3e-4, 10.0),
)


def available_ltw_diagnostic_arms() -> tuple[str, ...]:
    return tuple(arm.name for arm in LTW_DIAGNOSTIC_ARMS)


@dataclass
class LTWDiagnosticResult:
    config: EventMNISTConfig
    device: str
    warmup_epochs: int
    arms: list[dict]
    records: list[dict]
    summary: list[dict]

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "ltw_optimization_diagnostic.json"
        records_path = output / "ltw_optimization_diagnostic_records.csv"
        summary_path = output / "ltw_optimization_diagnostic_summary.csv"
        payload = {
            "config": asdict(self.config),
            "device": self.device,
            "warmup_epochs": self.warmup_epochs,
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
            plot_path = output / "ltw_optimization_diagnostic_summary.png"
            plot_ltw_diagnostic(self.summary, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def run_ltw_optimization_diagnostic(
    config: EventMNISTConfig,
    *,
    device="auto",
    warmup_epochs: int = 10,
    arm_names: Iterable[str] | None = None,
    ltw_minimum: float = 0.0,
    ltw_maximum: float = 1.0,
) -> LTWDiagnosticResult:
    """Run paired fixed-topology LTW optimization interventions."""

    if torch is None:
        raise ImportError("Phase 22 LTW diagnostics require PyTorch")
    arms = _select_arms(arm_names)
    _validate(config, arms, warmup_epochs, ltw_minimum, ltw_maximum)
    resolved = resolve_device(device)
    train_pixels, train_labels, test_pixels, test_labels = load_mnist_tensors(config)
    records: list[dict] = []

    for seed in config.seeds:
        for classifier_index, classifier in enumerate(("linear", "mlp")):
            for arm in arms:
                # Every arm in a classifier/seed block receives the same graph
                # and readout initialization, making gains paired interventions.
                seed_everything(seed + classifier_index * 10_000, device=resolved)
                feature_dim = config.timesteps * config.neuron_count
                hidden_units = _matched_raw_hidden_units(
                    feature_dim,
                    config.neuron_count * 2,
                    config.readout_hidden_units,
                )
                train_ltw = arm.schedule != "frozen"
                model = SparseTemporalClassifier(
                    config,
                    seed=seed,
                    classifier=classifier,
                    hidden_units=hidden_units,
                    train_ltw=train_ltw,
                    surrogate_slope=arm.surrogate_slope,
                    device=resolved,
                ).to(resolved)
                initial_ltw = model.graph.long_term_weight.detach().clone()
                _, _, initial_event_rate = _measure(
                    model, test_pixels, test_labels, config.batch_size, resolved
                )
                scope_mask = ltw_scope_mask(model, arm.scope)
                train_seconds = _train_arm(
                    model,
                    train_pixels,
                    train_labels,
                    config,
                    arm=arm,
                    seed=seed,
                    device=resolved,
                    warmup_epochs=warmup_epochs,
                    scope_mask=scope_mask,
                    ltw_minimum=ltw_minimum,
                    ltw_maximum=ltw_maximum,
                )
                train_accuracy, _, _ = _measure(
                    model, train_pixels, train_labels, config.batch_size, resolved
                )
                test_accuracy, inference_seconds, final_event_rate = _measure(
                    model, test_pixels, test_labels, config.batch_size, resolved
                )
                current_ltw = model.graph.long_term_weight.detach()
                active = model.graph.active_mask
                sensor = active & (model.graph.sources < config.sensor_neurons)
                recurrent = active & ~sensor
                records.append(
                    {
                        "seed": int(seed),
                        "arm": arm.name,
                        "classifier": classifier,
                        "schedule": arm.schedule,
                        "scope": arm.scope,
                        "reservoir_learning_rate": float(arm.reservoir_learning_rate),
                        "surrogate_slope": float(arm.surrogate_slope),
                        "warmup_epochs": int(warmup_epochs if arm.schedule == "warmup" else 0),
                        "train_accuracy": float(train_accuracy),
                        "test_accuracy": float(test_accuracy),
                        "active_edges": int(active.sum().item()),
                        "scope_trainable_edges": int(scope_mask.sum().item()),
                        "readout_parameters": int(_readout_parameter_count(model)),
                        "optimizer_parameters": int(
                            _readout_parameter_count(model) + (model.graph.max_edges if train_ltw else 0)
                        ),
                        "effective_trainable_parameters": int(
                            _readout_parameter_count(model) + (scope_mask.sum().item() if train_ltw else 0)
                        ),
                        "initial_hidden_event_rate": float(initial_event_rate),
                        "final_hidden_event_rate": float(final_event_rate),
                        "event_rate_ratio": float(final_event_rate / max(initial_event_rate, 1e-12)),
                        "mean_absolute_ltw_change": _mean_absolute_change(
                            current_ltw, initial_ltw, active
                        ),
                        "mean_sensor_ltw_change": _mean_absolute_change(
                            current_ltw, initial_ltw, sensor
                        ),
                        "mean_recurrent_ltw_change": _mean_absolute_change(
                            current_ltw, initial_ltw, recurrent
                        ),
                        "lower_ltw_saturation_rate": _saturation_rate(
                            current_ltw, active, ltw_minimum, lower=True
                        ),
                        "upper_ltw_saturation_rate": _saturation_rate(
                            current_ltw, active, ltw_maximum, lower=False
                        ),
                        "train_seconds": float(train_seconds),
                        "inference_seconds": float(inference_seconds),
                        "end_to_end_examples_per_second": float(
                            test_pixels.shape[0] / max(inference_seconds, 1e-12)
                        ),
                    }
                )

    _attach_frozen_comparisons(records)
    return LTWDiagnosticResult(
        config=config,
        device=device_kind(resolved),
        warmup_epochs=int(warmup_epochs),
        arms=[asdict(arm) for arm in arms],
        records=records,
        summary=summarize_ltw_diagnostic(records, arms=arms),
    )


def ltw_scope_mask(model: SparseTemporalClassifier, scope: str):
    """Return active LTW slots belonging to a diagnostic training scope."""

    active = model.graph.active_mask
    if scope == "none":
        return torch.zeros_like(active)
    sensor = model.graph.sources < model.config.sensor_neurons
    if scope == "all":
        return active.clone()
    if scope == "sensor":
        return active & sensor
    if scope == "recurrent":
        return active & ~sensor
    raise ValueError(f"unknown LTW scope: {scope}")


def summarize_ltw_diagnostic(
    records: Iterable[dict], *, arms: Iterable[LTWDiagnosticArm] | None = None
) -> list[dict]:
    rows = list(records)
    arm_order = list(arms or LTW_DIAGNOSTIC_ARMS)
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((str(row["arm"]), str(row["classifier"])), []).append(row)
    summary: list[dict] = []
    for arm in arm_order:
        for classifier in ("linear", "mlp"):
            group = grouped.get((arm.name, classifier), [])
            if not group:
                continue
            accuracy = [float(row["test_accuracy"]) for row in group]
            gains = [float(row["accuracy_gain_vs_frozen"]) for row in group]
            summary.append(
                {
                    "arm": arm.name,
                    "classifier": classifier,
                    "schedule": arm.schedule,
                    "scope": arm.scope,
                    "reservoir_learning_rate": arm.reservoir_learning_rate,
                    "surrogate_slope": arm.surrogate_slope,
                    "seeds": len(group),
                    "mean_test_accuracy": statistics.fmean(accuracy),
                    "std_test_accuracy": statistics.pstdev(accuracy),
                    "mean_accuracy_gain_vs_frozen": statistics.fmean(gains),
                    "improved_seed_count": sum(gain > 0 for gain in gains),
                    "practical_gain_seed_count": sum(gain >= 0.005 for gain in gains),
                    "active_edges": int(group[0]["active_edges"]),
                    "scope_trainable_edges": int(group[0]["scope_trainable_edges"]),
                    "effective_trainable_parameters": int(group[0]["effective_trainable_parameters"]),
                    "mean_event_rate_ratio": statistics.fmean(
                        float(row["event_rate_ratio"]) for row in group
                    ),
                    "mean_absolute_ltw_change": statistics.fmean(
                        float(row["mean_absolute_ltw_change"]) for row in group
                    ),
                    "mean_sensor_ltw_change": statistics.fmean(
                        float(row["mean_sensor_ltw_change"]) for row in group
                    ),
                    "mean_recurrent_ltw_change": statistics.fmean(
                        float(row["mean_recurrent_ltw_change"]) for row in group
                    ),
                    "mean_lower_ltw_saturation_rate": statistics.fmean(
                        float(row["lower_ltw_saturation_rate"]) for row in group
                    ),
                    "mean_upper_ltw_saturation_rate": statistics.fmean(
                        float(row["upper_ltw_saturation_rate"]) for row in group
                    ),
                    "mean_train_seconds": statistics.fmean(
                        float(row["train_seconds"]) for row in group
                    ),
                    "mean_end_to_end_examples_per_second": statistics.fmean(
                        float(row["end_to_end_examples_per_second"]) for row in group
                    ),
                }
            )
    return summary


def plot_ltw_diagnostic(summary: list[dict], path: str | pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    arms = list(dict.fromkeys(row["arm"] for row in summary))
    lookup = {(row["arm"], row["classifier"]): row for row in summary}
    positions = list(range(len(arms)))
    width = 0.38
    figure, axes = plt.subplots(2, 1, figsize=(16, 10), constrained_layout=True)
    for offset, classifier, color in ((-width / 2, "linear", "#35b4f2"), (width / 2, "mlp", "#ffb31a")):
        axes[0].bar(
            [position + offset for position in positions],
            [100.0 * float(lookup[(arm, classifier)]["mean_accuracy_gain_vs_frozen"]) for arm in arms],
            width,
            label=classifier,
            color=color,
        )
        axes[1].bar(
            [position + offset for position in positions],
            [float(lookup[(arm, classifier)]["mean_event_rate_ratio"]) for arm in arms],
            width,
            label=classifier,
            color=color,
        )
    labels = [arm.replace("_", "\n") for arm in arms]
    axes[0].axhline(0.0, color="#222222", linewidth=1)
    axes[0].set_ylabel("Accuracy gain over paired frozen (points)")
    axes[0].set_title("AMMC Gen-5 Phase 22: LTW Optimization Diagnostic")
    axes[0].set_xticks(positions, labels)
    axes[0].legend()
    axes[1].axhline(1.0, color="#222222", linewidth=1)
    axes[1].set_ylabel("Final / initial hidden event rate")
    axes[1].set_xticks(positions, labels)
    axes[1].legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _train_arm(
    model,
    pixels,
    labels,
    config,
    *,
    arm,
    seed,
    device,
    warmup_epochs,
    scope_mask,
    ltw_minimum,
    ltw_maximum,
):
    train_ltw = arm.schedule != "frozen"
    parameter_groups = [
        {
            "params": list(model.readout.parameters()),
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
        }
    ]
    if train_ltw:
        parameter_groups.append(
            {
                "params": [model.graph.long_term_weight],
                "lr": arm.reservoir_learning_rate,
                "weight_decay": 0.0,
            }
        )
    optimizer = torch.optim.AdamW(parameter_groups)
    criterion = nn.CrossEntropyLoss()
    model.train()
    start_time = time.perf_counter()
    for epoch in range(config.epochs):
        ltw_active = arm.schedule == "joint" or (
            arm.schedule == "warmup" and epoch >= warmup_epochs
        )
        generator = torch.Generator().manual_seed(seed * 1000 + epoch)
        order = torch.randperm(pixels.shape[0], generator=generator)
        for start in range(0, order.numel(), config.batch_size):
            index = order[start : start + config.batch_size]
            batch = pixels.index_select(0, index).to(device)
            target = labels.index_select(0, index).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), target)
            loss.backward()
            gradient = model.graph.long_term_weight.grad
            if train_ltw and gradient is not None:
                if ltw_active:
                    gradient.mul_(scope_mask.to(gradient.dtype))
                else:
                    gradient.zero_()
            optimizer.step()
            if train_ltw:
                model.clamp_ltw(ltw_minimum, ltw_maximum)
            mark_step(device)
    sync(device)
    return time.perf_counter() - start_time


def _attach_frozen_comparisons(records: list[dict]) -> None:
    frozen = {
        (int(row["seed"]), str(row["classifier"])): float(row["test_accuracy"])
        for row in records
        if row["arm"] == "frozen"
    }
    for row in records:
        baseline = frozen[(int(row["seed"]), str(row["classifier"]))]
        row["paired_frozen_test_accuracy"] = baseline
        row["accuracy_gain_vs_frozen"] = float(row["test_accuracy"]) - baseline


def _select_arms(names: Iterable[str] | None) -> tuple[LTWDiagnosticArm, ...]:
    registry = {arm.name: arm for arm in LTW_DIAGNOSTIC_ARMS}
    if names is None:
        return LTW_DIAGNOSTIC_ARMS
    selected = tuple(names)
    unknown = [name for name in selected if name not in registry]
    if unknown:
        raise ValueError(f"unknown LTW diagnostic arms: {', '.join(unknown)}")
    if "frozen" not in selected:
        selected = ("frozen",) + selected
    return tuple(registry[name] for name in selected)


def _mean_absolute_change(current, initial, mask) -> float:
    if not bool(mask.any().item()):
        return 0.0
    return float((current[mask] - initial[mask]).abs().mean().item())


def _saturation_rate(values, mask, boundary: float, *, lower: bool) -> float:
    if not bool(mask.any().item()):
        return 0.0
    selected = values[mask]
    saturated = selected <= boundary + 1e-6 if lower else selected >= boundary - 1e-6
    return float(saturated.to(values.dtype).mean().item())


def _validate(config, arms, warmup_epochs, ltw_minimum, ltw_maximum) -> None:
    if not config.seeds or config.epochs <= 0:
        raise ValueError("at least one seed and positive epochs are required")
    if not 0 <= warmup_epochs < config.epochs:
        raise ValueError("warmup_epochs must be in [0, epochs)")
    if ltw_minimum < 0 or ltw_maximum <= ltw_minimum:
        raise ValueError("LTW bounds must satisfy 0 <= minimum < maximum")
    required = config.sensor_neurons * config.sensor_fanout + config.hidden_neurons * config.recurrent_fanout
    if required > config.max_edges:
        raise ValueError(f"topology requires {required} edges but max_edges is {config.max_edges}")
    for arm in arms:
        if arm.schedule not in {"frozen", "joint", "warmup"}:
            raise ValueError(f"unknown LTW schedule: {arm.schedule}")
        if arm.scope not in {"none", "all", "sensor", "recurrent"}:
            raise ValueError(f"unknown LTW scope: {arm.scope}")
        if arm.schedule != "frozen" and arm.reservoir_learning_rate <= 0:
            raise ValueError("trainable LTW arms require a positive learning rate")
        if arm.surrogate_slope <= 0:
            raise ValueError("surrogate slope must be positive")


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
