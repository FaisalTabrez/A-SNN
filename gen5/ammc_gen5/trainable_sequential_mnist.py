"""Phase 25 fixed-topology LTW training on row-sequential MNIST.

Phase 24 established a large causal recurrence effect when images arrive one
row at a time. This module keeps that topology fixed and asks whether stable,
warm-started LTW optimization improves the useful recurrent representation.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import pathlib
import statistics
import time
from typing import Iterable

from .dynamic_sparse import DynamicSparseLinear
from .event_mnist import (
    EventMNISTConfig,
    _Classifier,
    _matched_raw_hidden_units,
    build_event_reservoir_edges,
    load_mnist_tensors,
    nn,
    torch,
)
from .runtime import device_kind, mark_step, resolve_device, seed_everything, sync
from .trainable_temporal_mnist import SurrogateSpike


@dataclass(frozen=True)
class SequentialLTWArm:
    name: str
    schedule: str
    scope: str
    reservoir_learning_rate: float


SEQUENTIAL_LTW_ARMS = (
    SequentialLTWArm("raw", "raw", "none", 0.0),
    SequentialLTWArm("frozen_recurrent", "frozen", "none", 0.0),
    SequentialLTWArm("warm_all_3em4", "warmup", "all", 3e-4),
    SequentialLTWArm("warm_recurrent_3em4", "warmup", "recurrent", 3e-4),
)


def available_sequential_ltw_arms() -> tuple[str, ...]:
    return tuple(arm.name for arm in SEQUENTIAL_LTW_ARMS)


class TrainableSequentialClassifier(nn.Module):
    """Final-state row-sequential classifier with a fixed sparse topology."""

    def __init__(
        self,
        config: EventMNISTConfig,
        *,
        seed: int,
        classifier: str,
        train_ltw: bool,
        surrogate_slope: float,
        device,
    ) -> None:
        if torch is None:
            raise ImportError("Phase 25 sequential LTW training requires PyTorch")
        super().__init__()
        self.config = config
        self.input_neurons = config.image_size
        self.hidden_neurons = config.hidden_neurons
        self.neuron_count = self.input_neurons + self.hidden_neurons
        self.surrogate_slope = float(surrogate_slope)
        edges = build_event_reservoir_edges(
            self.input_neurons,
            self.hidden_neurons,
            sensor_fanout=config.sensor_fanout,
            recurrent_fanout=config.recurrent_fanout,
            seed=seed,
        )
        if len(edges) > config.max_edges:
            raise ValueError(f"sequential reservoir requires {len(edges)} edges")
        self.graph = DynamicSparseLinear(
            self.neuron_count,
            self.neuron_count,
            config.max_edges,
            device=device,
        )
        self.graph.load_edges(edges)
        self.graph.short_term_weight.requires_grad_(False)
        self.graph.long_term_weight.requires_grad_(bool(train_ltw))
        hidden_units = config.readout_hidden_units if classifier == "mlp" else 1
        self.readout = _Classifier(
            self.hidden_neurons * 2,
            kind=classifier,
            hidden_units=hidden_units,
        )

    @property
    def active_edge_count(self) -> int:
        return self.graph.active_edge_count

    def forward(self, pixels, *, return_event_rate: bool = False):  # type: ignore[override]
        if pixels.ndim != 2 or pixels.shape[1] != self.config.image_size**2:
            raise ValueError("pixels must have shape [batch, image_size ** 2]")
        frames = pixels.reshape(
            pixels.shape[0], self.config.image_size, self.config.image_size
        )
        membrane = pixels.new_zeros((pixels.shape[0], self.hidden_neurons))
        hidden_spikes = torch.zeros_like(membrane)
        hidden_event_sum = pixels.new_zeros(())
        for step in range(self.config.image_size):
            sensor_events = frames[:, step, :] * self.config.input_gain
            network_state = torch.cat((sensor_events, hidden_spikes), dim=1)
            current = self.graph(network_state)[:, self.input_neurons :]
            pre_reset = membrane * self.config.reservoir_leak + current
            hidden_spikes = SurrogateSpike.apply(
                pre_reset - self.config.reservoir_threshold,
                self.surrogate_slope,
            )
            membrane = pre_reset - hidden_spikes * self.config.reservoir_threshold
            hidden_event_sum = hidden_event_sum + hidden_spikes.mean()
        # Only state available after the final row reaches the readout.
        logits = self.readout(torch.cat((hidden_spikes, membrane), dim=1))
        if return_event_rate:
            return logits, hidden_event_sum / self.config.image_size
        return logits

    def clamp_ltw(self, minimum: float, maximum: float) -> None:
        with torch.no_grad():
            self.graph.long_term_weight.clamp_(minimum, maximum)
            self.graph.long_term_weight.mul_(
                self.graph.active_mask.to(self.graph.long_term_weight.dtype)
            )
            self.graph.short_term_weight.zero_()


@dataclass
class TrainableSequentialMNISTResult:
    config: EventMNISTConfig
    device: str
    warmup_epochs: int
    surrogate_slope: float
    arms: list[dict]
    records: list[dict]
    summary: list[dict]

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "trainable_sequential_mnist.json"
        records_path = output / "trainable_sequential_mnist_records.csv"
        summary_path = output / "trainable_sequential_mnist_summary.csv"
        payload = {
            "config": asdict(self.config),
            "device": self.device,
            "warmup_epochs": self.warmup_epochs,
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
            plot_path = output / "trainable_sequential_mnist_summary.png"
            plot_trainable_sequential_mnist(self.summary, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def run_trainable_sequential_mnist(
    config: EventMNISTConfig,
    *,
    device="auto",
    warmup_epochs: int = 10,
    surrogate_slope: float = 10.0,
    arm_names: Iterable[str] | None = None,
    ltw_minimum: float = 0.0,
    ltw_maximum: float = 1.0,
) -> TrainableSequentialMNISTResult:
    if torch is None:
        raise ImportError("Phase 25 sequential LTW training requires PyTorch")
    arms = _select_arms(arm_names)
    _validate(
        config,
        arms,
        warmup_epochs,
        surrogate_slope,
        ltw_minimum,
        ltw_maximum,
    )
    resolved = resolve_device(device)
    train_pixels, train_labels, test_pixels, test_labels = load_mnist_tensors(config)
    records: list[dict] = []

    for seed in config.seeds:
        for classifier_index, classifier in enumerate(("linear", "mlp")):
            for arm in arms:
                # Frozen and trainable interventions receive identical graph
                # and readout initialization inside each seed/classifier pair.
                seed_everything(seed + classifier_index * 10_000, device=resolved)
                train_ltw = arm.schedule == "warmup"
                if arm.schedule == "raw":
                    hidden_units = 1
                    if classifier == "mlp":
                        hidden_units = _matched_raw_hidden_units(
                            config.image_size**2,
                            config.hidden_neurons * 2,
                            config.readout_hidden_units,
                        )
                    model = _Classifier(
                        config.image_size**2,
                        kind=classifier,
                        hidden_units=hidden_units,
                    ).to(resolved)
                    initial_ltw = None
                    initial_event_rate = 0.0
                    scope_mask = None
                else:
                    model = TrainableSequentialClassifier(
                        config,
                        seed=seed,
                        classifier=classifier,
                        train_ltw=train_ltw,
                        surrogate_slope=surrogate_slope,
                        device=resolved,
                    ).to(resolved)
                    initial_ltw = model.graph.long_term_weight.detach().clone()
                    _, _, initial_event_rate = _measure(
                        model,
                        test_pixels,
                        test_labels,
                        config.batch_size,
                        resolved,
                    )
                    scope_mask = sequential_ltw_scope_mask(model, arm.scope)

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

                active_edges = 0
                scope_edges = 0
                mean_ltw_change = 0.0
                sensor_ltw_change = 0.0
                recurrent_ltw_change = 0.0
                lower_saturation = 0.0
                upper_saturation = 0.0
                if isinstance(model, TrainableSequentialClassifier):
                    active = model.graph.active_mask
                    sensor = active & (model.graph.sources < model.input_neurons)
                    recurrent = active & ~sensor
                    current = model.graph.long_term_weight.detach()
                    active_edges = int(active.sum().item())
                    scope_edges = int(scope_mask.sum().item()) if scope_mask is not None else 0
                    mean_ltw_change = _mean_absolute_change(current, initial_ltw, active)
                    sensor_ltw_change = _mean_absolute_change(current, initial_ltw, sensor)
                    recurrent_ltw_change = _mean_absolute_change(current, initial_ltw, recurrent)
                    lower_saturation = _saturation_rate(
                        current, active, ltw_minimum, lower=True
                    )
                    upper_saturation = _saturation_rate(
                        current, active, ltw_maximum, lower=False
                    )

                readout_parameters = _readout_parameter_count(model)
                optimizer_parameters = sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                )
                effective_trainable = readout_parameters + (
                    scope_edges if train_ltw else 0
                )
                records.append(
                    {
                        "seed": int(seed),
                        "arm": arm.name,
                        "classifier": classifier,
                        "schedule": arm.schedule,
                        "scope": arm.scope,
                        "reservoir_learning_rate": float(arm.reservoir_learning_rate),
                        "surrogate_slope": float(surrogate_slope),
                        "warmup_epochs": int(warmup_epochs if train_ltw else 0),
                        "train_accuracy": float(train_accuracy),
                        "test_accuracy": float(test_accuracy),
                        "active_edges": active_edges,
                        "scope_trainable_edges": scope_edges,
                        "readout_parameters": int(readout_parameters),
                        "optimizer_parameters": int(optimizer_parameters),
                        "effective_trainable_parameters": int(effective_trainable),
                        "initial_hidden_event_rate": float(initial_event_rate),
                        "final_hidden_event_rate": float(final_event_rate),
                        "event_rate_ratio": float(
                            final_event_rate / max(initial_event_rate, 1e-12)
                            if initial_event_rate > 0
                            else 0.0
                        ),
                        "mean_absolute_ltw_change": mean_ltw_change,
                        "mean_sensor_ltw_change": sensor_ltw_change,
                        "mean_recurrent_ltw_change": recurrent_ltw_change,
                        "lower_ltw_saturation_rate": lower_saturation,
                        "upper_ltw_saturation_rate": upper_saturation,
                        "train_seconds": float(train_seconds),
                        "inference_seconds": float(inference_seconds),
                        "end_to_end_examples_per_second": float(
                            test_pixels.shape[0] / max(inference_seconds, 1e-12)
                        ),
                    }
                )

    _attach_frozen_comparisons(records)
    return TrainableSequentialMNISTResult(
        config=config,
        device=device_kind(resolved),
        warmup_epochs=int(warmup_epochs),
        surrogate_slope=float(surrogate_slope),
        arms=[asdict(arm) for arm in arms],
        records=records,
        summary=summarize_trainable_sequential_mnist(records, arms=arms),
    )


def sequential_ltw_scope_mask(model: TrainableSequentialClassifier, scope: str):
    active = model.graph.active_mask
    if scope == "none":
        return torch.zeros_like(active)
    if scope == "all":
        return active.clone()
    if scope == "recurrent":
        return active & (model.graph.sources >= model.input_neurons)
    raise ValueError(f"unsupported sequential LTW scope: {scope}")


def summarize_trainable_sequential_mnist(
    records: Iterable[dict],
    *,
    arms: Iterable[SequentialLTWArm] = SEQUENTIAL_LTW_ARMS,
) -> list[dict]:
    rows = list(records)
    summary: list[dict] = []
    for arm in arms:
        for classifier in ("linear", "mlp"):
            group = [
                row
                for row in rows
                if row["arm"] == arm.name and row["classifier"] == classifier
            ]
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
                    "seeds": len(group),
                    "mean_test_accuracy": statistics.fmean(accuracy),
                    "std_test_accuracy": statistics.pstdev(accuracy),
                    "mean_accuracy_gain_vs_frozen": statistics.fmean(gains),
                    "improved_seed_count": sum(gain > 0 for gain in gains),
                    "practical_gain_seed_count": sum(gain >= 0.005 for gain in gains),
                    "active_edges": int(group[0]["active_edges"]),
                    "scope_trainable_edges": int(group[0]["scope_trainable_edges"]),
                    "effective_trainable_parameters": int(
                        group[0]["effective_trainable_parameters"]
                    ),
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


def plot_trainable_sequential_mnist(summary: list[dict], path: str | pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    arms = list(dict.fromkeys(row["arm"] for row in summary))
    lookup = {(row["arm"], row["classifier"]): row for row in summary}
    positions = list(range(len(arms)))
    width = 0.38
    figure, axes = plt.subplots(3, 1, figsize=(15, 13), constrained_layout=True)
    for offset, classifier, color in (
        (-width / 2, "linear", "#35b4f2"),
        (width / 2, "mlp", "#ffb31a"),
    ):
        axes[0].bar(
            [position + offset for position in positions],
            [100.0 * float(lookup[(arm, classifier)]["mean_test_accuracy"]) for arm in arms],
            width,
            label=classifier,
            color=color,
        )
        axes[1].bar(
            [position + offset for position in positions],
            [
                100.0 * float(lookup[(arm, classifier)]["mean_accuracy_gain_vs_frozen"])
                for arm in arms
            ],
            width,
            label=classifier,
            color=color,
        )
        axes[2].bar(
            [position + offset for position in positions],
            [float(lookup[(arm, classifier)]["mean_event_rate_ratio"]) for arm in arms],
            width,
            label=classifier,
            color=color,
        )
    labels = [arm.replace("_", "\n") for arm in arms]
    axes[0].set_ylabel("Engineering-validation accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_title("AMMC Gen-5 Phase 25: Sequential LTW Training")
    axes[1].axhline(0.0, color="#222222", linewidth=1)
    axes[1].set_ylabel("Gain over paired frozen (points)")
    axes[2].axhline(1.0, color="#222222", linewidth=1)
    axes[2].set_ylabel("Final / initial hidden event rate")
    for axis in axes:
        axis.set_xticks(positions, labels)
        axis.legend()
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
    train_ltw = arm.schedule == "warmup"
    if isinstance(model, TrainableSequentialClassifier):
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
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    criterion = nn.CrossEntropyLoss()
    model.train()
    start_time = time.perf_counter()
    for epoch in range(config.epochs):
        ltw_active = train_ltw and epoch >= warmup_epochs
        generator = torch.Generator().manual_seed(seed * 1000 + epoch)
        order = torch.randperm(pixels.shape[0], generator=generator)
        for start in range(0, order.numel(), config.batch_size):
            index = order[start : start + config.batch_size]
            batch = pixels.index_select(0, index).to(device)
            target = labels.index_select(0, index).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), target)
            loss.backward()
            if train_ltw:
                gradient = model.graph.long_term_weight.grad
                if gradient is not None:
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


def _measure(model, pixels, labels, batch_size, device):
    model.eval()
    correct = 0
    weighted_event_rate = 0.0
    start_time = time.perf_counter()
    with torch.no_grad():
        for start in range(0, pixels.shape[0], batch_size):
            batch = pixels[start : start + batch_size].to(device)
            if isinstance(model, TrainableSequentialClassifier):
                logits, event_rate = model(batch, return_event_rate=True)
                weighted_event_rate += float(event_rate.item()) * batch.shape[0]
            else:
                logits = model(batch)
            prediction = logits.argmax(dim=1).cpu()
            correct += int((prediction == labels[start : start + batch_size]).sum().item())
            mark_step(device)
    sync(device)
    seconds = time.perf_counter() - start_time
    return (
        correct / pixels.shape[0],
        seconds,
        weighted_event_rate / pixels.shape[0],
    )


def _readout_parameter_count(model) -> int:
    parameters = (
        model.readout.parameters()
        if isinstance(model, TrainableSequentialClassifier)
        else model.parameters()
    )
    return sum(parameter.numel() for parameter in parameters)


def _attach_frozen_comparisons(records: list[dict]) -> None:
    frozen = {
        (int(row["seed"]), str(row["classifier"])): float(row["test_accuracy"])
        for row in records
        if row["arm"] == "frozen_recurrent"
    }
    for row in records:
        baseline = frozen[(int(row["seed"]), str(row["classifier"]))]
        row["paired_frozen_test_accuracy"] = baseline
        row["accuracy_gain_vs_frozen"] = float(row["test_accuracy"]) - baseline


def _select_arms(names: Iterable[str] | None) -> tuple[SequentialLTWArm, ...]:
    registry = {arm.name: arm for arm in SEQUENTIAL_LTW_ARMS}
    if names is None:
        return SEQUENTIAL_LTW_ARMS
    selected = tuple(names)
    unknown = [name for name in selected if name not in registry]
    if unknown:
        raise ValueError(f"unknown sequential LTW arms: {', '.join(unknown)}")
    required = [name for name in ("raw", "frozen_recurrent") if name not in selected]
    return tuple(registry[name] for name in (*required, *selected))


def _mean_absolute_change(current, initial, mask) -> float:
    if initial is None or not bool(mask.any().item()):
        return 0.0
    return float((current[mask] - initial[mask]).abs().mean().item())


def _saturation_rate(values, mask, boundary: float, *, lower: bool) -> float:
    if not bool(mask.any().item()):
        return 0.0
    selected = values[mask]
    saturated = selected <= boundary + 1e-6 if lower else selected >= boundary - 1e-6
    return float(saturated.to(values.dtype).mean().item())


def _validate(config, arms, warmup_epochs, surrogate_slope, ltw_minimum, ltw_maximum):
    if not config.seeds or config.epochs <= 0:
        raise ValueError("at least one seed and positive epochs are required")
    if not 0 <= warmup_epochs < config.epochs:
        raise ValueError("warmup_epochs must be in [0, epochs)")
    if surrogate_slope <= 0:
        raise ValueError("surrogate slope must be positive")
    if ltw_minimum < 0 or ltw_maximum <= ltw_minimum:
        raise ValueError("LTW bounds must satisfy 0 <= minimum < maximum")
    required_edges = (
        config.image_size * config.sensor_fanout
        + config.hidden_neurons * config.recurrent_fanout
    )
    if required_edges > config.max_edges:
        raise ValueError(
            f"sequential topology requires {required_edges} edges but max_edges is {config.max_edges}"
        )
    if not any(arm.name == "frozen_recurrent" for arm in arms):
        raise ValueError("the paired frozen recurrent control is required")


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
