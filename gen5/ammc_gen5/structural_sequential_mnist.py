"""Phase 26 targeted synaptogenesis on row-sequential MNIST.

Phase 25 localized useful LTW adaptation to the sensor projection. This module
preserves the proven recurrent core and introduces deterministic mid-training
sensor or recurrent sprouting so topology growth can be tested causally before
any pruning or general structural churn is allowed.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import pathlib
import random
import statistics
import time
from typing import Iterable

from .event_mnist import (
    EventMNISTConfig,
    _Classifier,
    _matched_raw_hidden_units,
    load_mnist_tensors,
    nn,
    torch,
)
from .runtime import device_kind, mark_step, resolve_device, seed_everything, sync
from .trainable_sequential_mnist import (
    TrainableSequentialClassifier,
    _mean_absolute_change,
    _measure,
    _readout_parameter_count,
    _saturation_rate,
)


@dataclass(frozen=True)
class StructuralSequentialArm:
    name: str
    schedule: str
    sprout_mode: str
    sprout_edges: int
    reservoir_learning_rate: float
    birth_weight: float = 0.1


STRUCTURAL_SEQUENTIAL_ARMS = (
    StructuralSequentialArm("raw", "raw", "none", 0, 0.0),
    StructuralSequentialArm("frozen_recurrent", "frozen", "none", 0, 0.0),
    StructuralSequentialArm("fixed_warm_all", "warmup", "none", 0, 3e-4),
    StructuralSequentialArm("sensor_sprout_16", "warmup", "sensor", 16, 3e-4),
    StructuralSequentialArm("sensor_sprout_48", "warmup", "sensor", 48, 3e-4),
    StructuralSequentialArm("recurrent_sprout_64", "warmup", "recurrent", 64, 3e-4),
)


def available_structural_sequential_arms() -> tuple[str, ...]:
    return tuple(arm.name for arm in STRUCTURAL_SEQUENTIAL_ARMS)


@dataclass
class StructuralSequentialMNISTResult:
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
        json_path = output / "structural_sequential_mnist.json"
        records_path = output / "structural_sequential_mnist_records.csv"
        summary_path = output / "structural_sequential_mnist_summary.csv"
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
            plot_path = output / "structural_sequential_mnist_summary.png"
            plot_structural_sequential_mnist(self.summary, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def run_structural_sequential_mnist(
    config: EventMNISTConfig,
    *,
    device="auto",
    warmup_epochs: int = 10,
    surrogate_slope: float = 10.0,
    arm_names: Iterable[str] | None = None,
    ltw_minimum: float = 0.0,
    ltw_maximum: float = 1.0,
) -> StructuralSequentialMNISTResult:
    if torch is None:
        raise ImportError("Phase 26 targeted synaptogenesis requires PyTorch")
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
                    core_mask = None
                    initial_event_rate = 0.0
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
                    core_mask = model.graph.active_mask.clone()
                    _, _, initial_event_rate = _measure(
                        model,
                        test_pixels,
                        test_labels,
                        config.batch_size,
                        resolved,
                    )

                train_seconds, sprouted_slots, sprout_reference = _train_structural_arm(
                    model,
                    train_pixels,
                    train_labels,
                    config,
                    arm=arm,
                    seed=seed,
                    device=resolved,
                    warmup_epochs=warmup_epochs,
                    ltw_minimum=ltw_minimum,
                    ltw_maximum=ltw_maximum,
                )
                train_accuracy, _, _ = _measure(
                    model, train_pixels, train_labels, config.batch_size, resolved
                )
                test_accuracy, inference_seconds, final_event_rate = _measure(
                    model, test_pixels, test_labels, config.batch_size, resolved
                )

                initial_edges = 0
                final_edges = 0
                core_ltw_change = 0.0
                sprouted_ltw_change = 0.0
                lower_saturation = 0.0
                upper_saturation = 0.0
                if isinstance(model, TrainableSequentialClassifier):
                    active = model.graph.active_mask
                    current = model.graph.long_term_weight.detach()
                    initial_edges = int(core_mask.sum().item())
                    final_edges = int(active.sum().item())
                    core_ltw_change = _mean_absolute_change(
                        current, initial_ltw, core_mask
                    )
                    if sprouted_slots:
                        sprout_mask = torch.zeros_like(active)
                        sprout_mask[sprouted_slots] = True
                        sprouted_ltw_change = _mean_absolute_change(
                            current, sprout_reference, sprout_mask
                        )
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
                    final_edges if train_ltw else 0
                )
                records.append(
                    {
                        "seed": int(seed),
                        "arm": arm.name,
                        "classifier": classifier,
                        "schedule": arm.schedule,
                        "sprout_mode": arm.sprout_mode,
                        "requested_sprout_edges": int(arm.sprout_edges),
                        "sprouted_edges": len(sprouted_slots),
                        "birth_weight": float(arm.birth_weight),
                        "reservoir_learning_rate": float(arm.reservoir_learning_rate),
                        "surrogate_slope": float(surrogate_slope),
                        "warmup_epochs": int(warmup_epochs if train_ltw else 0),
                        "train_accuracy": float(train_accuracy),
                        "test_accuracy": float(test_accuracy),
                        "initial_active_edges": initial_edges,
                        "final_active_edges": final_edges,
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
                        "mean_core_ltw_change": core_ltw_change,
                        "mean_sprouted_ltw_change_from_birth": sprouted_ltw_change,
                        "lower_ltw_saturation_rate": lower_saturation,
                        "upper_ltw_saturation_rate": upper_saturation,
                        "train_seconds": float(train_seconds),
                        "inference_seconds": float(inference_seconds),
                        "end_to_end_examples_per_second": float(
                            test_pixels.shape[0] / max(inference_seconds, 1e-12)
                        ),
                    }
                )

    _attach_structural_comparisons(records)
    return StructuralSequentialMNISTResult(
        config=config,
        device=device_kind(resolved),
        warmup_epochs=int(warmup_epochs),
        surrogate_slope=float(surrogate_slope),
        arms=[asdict(arm) for arm in arms],
        records=records,
        summary=summarize_structural_sequential_mnist(records, arms=arms),
    )


def sprout_targeted_edges(
    model: TrainableSequentialClassifier,
    mode: str,
    count: int,
    *,
    seed: int,
    birth_weight: float = 0.1,
) -> list[int]:
    """Sprout unique targeted edges while preserving every existing edge."""

    if mode not in {"sensor", "recurrent"}:
        raise ValueError("sprout mode must be sensor or recurrent")
    if count < 0:
        raise ValueError("sprout count must be non-negative")
    if model.active_edge_count + count > model.graph.max_edges:
        raise ValueError("sprouting exceeds the fixed edge-pool capacity")
    # The sensor-16 topology is a strict prefix of sensor-48 for the same seed,
    # making the dose comparison nested instead of comparing unrelated draws.
    rng = random.Random(int(seed) * 1009 + (1 if mode == "sensor" else 2))
    active_slots = model.graph.active_mask.nonzero(as_tuple=False).flatten().tolist()
    existing = {
        (int(model.graph.sources[slot].item()), int(model.graph.targets[slot].item()))
        for slot in active_slots
    }
    hidden_ids = list(range(model.input_neurons, model.neuron_count))
    slots: list[int] = []
    for index in range(count):
        if mode == "sensor":
            source = index % model.input_neurons
            candidates = [
                target for target in hidden_ids if (source, target) not in existing
            ]
            sign = 1.0
        else:
            source = hidden_ids[index % model.hidden_neurons]
            candidates = [
                target
                for target in hidden_ids
                if target != source and (source, target) not in existing
            ]
            sign = -1.0 if rng.random() < 0.2 else 1.0
        if not candidates:
            raise RuntimeError(f"no unique {mode} target remains for source {source}")
        target = rng.choice(candidates)
        slot = model.graph.sprout(
            source,
            target,
            short_term_weight=0.0,
            long_term_weight=birth_weight,
            sign=sign,
            delay_steps=0,
        )
        existing.add((source, target))
        slots.append(slot)
    return slots


def summarize_structural_sequential_mnist(
    records: Iterable[dict],
    *,
    arms: Iterable[StructuralSequentialArm] = STRUCTURAL_SEQUENTIAL_ARMS,
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
            fixed_gains = [float(row["accuracy_gain_vs_fixed_warm_all"]) for row in group]
            frozen_gains = [float(row["accuracy_gain_vs_frozen"]) for row in group]
            summary.append(
                {
                    "arm": arm.name,
                    "classifier": classifier,
                    "schedule": arm.schedule,
                    "sprout_mode": arm.sprout_mode,
                    "seeds": len(group),
                    "mean_test_accuracy": statistics.fmean(accuracy),
                    "std_test_accuracy": statistics.pstdev(accuracy),
                    "mean_accuracy_gain_vs_fixed_warm_all": statistics.fmean(
                        fixed_gains
                    ),
                    "fixed_improved_seed_count": sum(gain > 0 for gain in fixed_gains),
                    "fixed_practical_gain_seed_count": sum(
                        gain >= 0.005 for gain in fixed_gains
                    ),
                    "mean_accuracy_gain_vs_frozen": statistics.fmean(frozen_gains),
                    "initial_active_edges": int(group[0]["initial_active_edges"]),
                    "final_active_edges": int(group[0]["final_active_edges"]),
                    "sprouted_edges": int(group[0]["sprouted_edges"]),
                    "effective_trainable_parameters": int(
                        group[0]["effective_trainable_parameters"]
                    ),
                    "mean_event_rate_ratio": statistics.fmean(
                        float(row["event_rate_ratio"]) for row in group
                    ),
                    "mean_core_ltw_change": statistics.fmean(
                        float(row["mean_core_ltw_change"]) for row in group
                    ),
                    "mean_sprouted_ltw_change_from_birth": statistics.fmean(
                        float(row["mean_sprouted_ltw_change_from_birth"])
                        for row in group
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


def plot_structural_sequential_mnist(summary: list[dict], path: str | pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    arms = list(dict.fromkeys(row["arm"] for row in summary))
    lookup = {(row["arm"], row["classifier"]): row for row in summary}
    positions = list(range(len(arms)))
    width = 0.38
    figure, axes = plt.subplots(3, 1, figsize=(16, 13), constrained_layout=True)
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
                100.0
                * float(
                    lookup[(arm, classifier)]["mean_accuracy_gain_vs_fixed_warm_all"]
                )
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
    axes[0].set_title("AMMC Gen-5 Phase 26: Targeted Synaptogenesis")
    axes[1].axhline(0.0, color="#222222", linewidth=1)
    axes[1].set_ylabel("Gain over fixed warm-all (points)")
    axes[2].axhline(1.0, color="#222222", linewidth=1)
    axes[2].set_ylabel("Final / initial hidden event rate")
    for axis in axes:
        axis.set_xticks(positions, labels)
        axis.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _train_structural_arm(
    model,
    pixels,
    labels,
    config,
    *,
    arm,
    seed,
    device,
    warmup_epochs,
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
    sprouted_slots: list[int] = []
    sprout_reference = None
    start_time = time.perf_counter()
    for epoch in range(config.epochs):
        if (
            train_ltw
            and epoch == warmup_epochs
            and arm.sprout_mode != "none"
            and arm.sprout_edges > 0
        ):
            sprouted_slots = sprout_targeted_edges(
                model,
                arm.sprout_mode,
                arm.sprout_edges,
                seed=seed,
                birth_weight=arm.birth_weight,
            )
            sprout_reference = model.graph.long_term_weight.detach().clone()
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
                        gradient.mul_(model.graph.active_mask.to(gradient.dtype))
                    else:
                        gradient.zero_()
            optimizer.step()
            if train_ltw:
                model.clamp_ltw(ltw_minimum, ltw_maximum)
            mark_step(device)
    sync(device)
    return time.perf_counter() - start_time, sprouted_slots, sprout_reference


def _attach_structural_comparisons(records: list[dict]) -> None:
    frozen = {
        (int(row["seed"]), str(row["classifier"])): float(row["test_accuracy"])
        for row in records
        if row["arm"] == "frozen_recurrent"
    }
    fixed = {
        (int(row["seed"]), str(row["classifier"])): float(row["test_accuracy"])
        for row in records
        if row["arm"] == "fixed_warm_all"
    }
    for row in records:
        key = (int(row["seed"]), str(row["classifier"]))
        row["paired_frozen_test_accuracy"] = frozen[key]
        row["accuracy_gain_vs_frozen"] = float(row["test_accuracy"]) - frozen[key]
        row["paired_fixed_warm_all_test_accuracy"] = fixed[key]
        row["accuracy_gain_vs_fixed_warm_all"] = (
            float(row["test_accuracy"]) - fixed[key]
        )


def _select_arms(names: Iterable[str] | None) -> tuple[StructuralSequentialArm, ...]:
    registry = {arm.name: arm for arm in STRUCTURAL_SEQUENTIAL_ARMS}
    if names is None:
        return STRUCTURAL_SEQUENTIAL_ARMS
    selected = tuple(names)
    unknown = [name for name in selected if name not in registry]
    if unknown:
        raise ValueError(f"unknown structural sequential arms: {', '.join(unknown)}")
    required = [
        name
        for name in ("raw", "frozen_recurrent", "fixed_warm_all")
        if name not in selected
    ]
    return tuple(registry[name] for name in (*required, *selected))


def _validate(config, arms, warmup_epochs, surrogate_slope, ltw_minimum, ltw_maximum):
    if not config.seeds or config.epochs <= 0:
        raise ValueError("at least one seed and positive epochs are required")
    if not 0 <= warmup_epochs < config.epochs:
        raise ValueError("warmup_epochs must be in [0, epochs)")
    if surrogate_slope <= 0:
        raise ValueError("surrogate slope must be positive")
    if ltw_minimum < 0 or ltw_maximum <= ltw_minimum:
        raise ValueError("LTW bounds must satisfy 0 <= minimum < maximum")
    base_edges = (
        config.image_size * config.sensor_fanout
        + config.hidden_neurons * config.recurrent_fanout
    )
    largest_growth = max(arm.sprout_edges for arm in arms)
    if base_edges + largest_growth > config.max_edges:
        raise ValueError(
            f"structural topology requires {base_edges + largest_growth} slots but max_edges is {config.max_edges}"
        )
    required_names = {"frozen_recurrent", "fixed_warm_all"}
    if not required_names.issubset(arm.name for arm in arms):
        raise ValueError("frozen and fixed warm-all paired controls are required")


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
