"""Phase 27 utility-gated structural plasticity on sequential MNIST.

Phase 26 showed that random sensor growth can help a linear readout but is not
seed-robust and does not help the MLP readout. This module tests whether task
gradient ranking can select better sensor routes. The original recurrent graph
is immutable; the optional pruning arm can deactivate only newly grown edges.
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
from .structural_sequential_mnist import sprout_targeted_edges
from .trainable_sequential_mnist import (
    TrainableSequentialClassifier,
    _mean_absolute_change,
    _measure,
    _readout_parameter_count,
    _saturation_rate,
)


@dataclass(frozen=True)
class UtilityGatedArm:
    name: str
    schedule: str
    selection: str
    sprout_edges: int
    candidate_edges: int
    reservoir_learning_rate: float
    birth_weight: float = 0.1
    prune_fraction: float = 0.0
    prune_threshold_ratio: float = 0.95


UTILITY_GATED_ARMS = (
    UtilityGatedArm("raw", "raw", "none", 0, 0, 0.0),
    UtilityGatedArm("frozen_recurrent", "frozen", "none", 0, 0, 0.0),
    UtilityGatedArm("fixed_warm_all", "warmup", "none", 0, 0, 3e-4),
    UtilityGatedArm("random_sensor_48", "warmup", "random", 48, 48, 3e-4),
    UtilityGatedArm("gradient_sensor_16", "warmup", "gradient", 16, 192, 3e-4),
    UtilityGatedArm("gradient_sensor_48", "warmup", "gradient", 48, 192, 3e-4),
    UtilityGatedArm(
        "gradient_sensor_48_prune",
        "warmup",
        "gradient",
        48,
        192,
        3e-4,
        prune_fraction=0.5,
    ),
)


def available_utility_gated_arms() -> tuple[str, ...]:
    return tuple(arm.name for arm in UTILITY_GATED_ARMS)


@dataclass
class UtilityGatedStructuralMNISTResult:
    config: EventMNISTConfig
    device: str
    warmup_epochs: int
    scoring_batches: int
    prune_after_epochs: int
    surrogate_slope: float
    arms: list[dict]
    records: list[dict]
    summary: list[dict]

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "utility_gated_structural_mnist.json"
        records_path = output / "utility_gated_structural_mnist_records.csv"
        summary_path = output / "utility_gated_structural_mnist_summary.csv"
        payload = {
            "config": asdict(self.config),
            "device": self.device,
            "warmup_epochs": self.warmup_epochs,
            "scoring_batches": self.scoring_batches,
            "prune_after_epochs": self.prune_after_epochs,
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
            plot_path = output / "utility_gated_structural_mnist_summary.png"
            plot_utility_gated_structural_mnist(self.summary, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def run_utility_gated_structural_mnist(
    config: EventMNISTConfig,
    *,
    device="auto",
    warmup_epochs: int = 10,
    scoring_batches: int = 4,
    prune_after_epochs: int = 3,
    surrogate_slope: float = 10.0,
    arm_names: Iterable[str] | None = None,
    ltw_minimum: float = 0.0,
    ltw_maximum: float = 1.0,
) -> UtilityGatedStructuralMNISTResult:
    if torch is None:
        raise ImportError("Phase 27 utility-gated structural MNIST requires PyTorch")
    arms = _select_arms(arm_names)
    _validate(
        config,
        arms,
        warmup_epochs,
        scoring_batches,
        prune_after_epochs,
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

                training = _train_utility_arm(
                    model,
                    train_pixels,
                    train_labels,
                    config,
                    arm=arm,
                    seed=seed,
                    device=resolved,
                    warmup_epochs=warmup_epochs,
                    scoring_batches=scoring_batches,
                    prune_after_epochs=prune_after_epochs,
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
                final_sprouted_ltw = 0.0
                lower_saturation = 0.0
                upper_saturation = 0.0
                if isinstance(model, TrainableSequentialClassifier):
                    active = model.graph.active_mask
                    current = model.graph.long_term_weight.detach()
                    initial_edges = int(core_mask.sum().item())
                    final_edges = int(active.sum().item())
                    core_ltw_change = _mean_absolute_change(current, initial_ltw, core_mask)
                    retained = [
                        slot for slot in training["sprouted_slots"] if bool(active[slot].item())
                    ]
                    if retained:
                        retained_mask = torch.zeros_like(active)
                        retained_mask[retained] = True
                        sprouted_ltw_change = _mean_absolute_change(
                            current, training["sprout_reference"], retained_mask
                        )
                        final_sprouted_ltw = float(current[retained].mean().item())
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
                        "selection": arm.selection,
                        "candidate_edges": int(arm.candidate_edges),
                        "requested_sprout_edges": int(arm.sprout_edges),
                        "sprouted_edges": len(training["sprouted_slots"]),
                        "pruned_sprouted_edges": len(training["pruned_slots"]),
                        "retained_sprouted_edges": len(training["sprouted_slots"])
                        - len(training["pruned_slots"]),
                        "birth_weight": float(arm.birth_weight),
                        "prune_fraction": float(arm.prune_fraction),
                        "prune_threshold_ratio": float(arm.prune_threshold_ratio),
                        "mean_selected_gradient_score": float(
                            statistics.fmean(training["selected_scores"])
                            if training["selected_scores"]
                            else 0.0
                        ),
                        "mean_pruned_ltw_before_removal": float(
                            statistics.fmean(training["pruned_weights"])
                            if training["pruned_weights"]
                            else 0.0
                        ),
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
                        "mean_retained_sprouted_ltw_change_from_birth": sprouted_ltw_change,
                        "mean_final_retained_sprouted_ltw": final_sprouted_ltw,
                        "lower_ltw_saturation_rate": lower_saturation,
                        "upper_ltw_saturation_rate": upper_saturation,
                        "train_seconds": float(training["train_seconds"]),
                        "inference_seconds": float(inference_seconds),
                        "end_to_end_examples_per_second": float(
                            test_pixels.shape[0] / max(inference_seconds, 1e-12)
                        ),
                    }
                )

    _attach_comparisons(records)
    return UtilityGatedStructuralMNISTResult(
        config=config,
        device=device_kind(resolved),
        warmup_epochs=int(warmup_epochs),
        scoring_batches=int(scoring_batches),
        prune_after_epochs=int(prune_after_epochs),
        surrogate_slope=float(surrogate_slope),
        arms=[asdict(arm) for arm in arms],
        records=records,
        summary=summarize_utility_gated_structural_mnist(records, arms=arms),
    )


def build_sensor_candidate_pool(
    input_neurons: int,
    hidden_neurons: int,
    existing_edges: Iterable[tuple[int, int]],
    count: int,
    *,
    seed: int,
) -> list[tuple[int, int]]:
    """Return a deterministic, unique pool of inactive sensor edges."""

    if input_neurons <= 0 or hidden_neurons <= 0:
        raise ValueError("input and hidden neuron counts must be positive")
    if count < 0:
        raise ValueError("candidate count must be non-negative")
    existing = {(int(source), int(target)) for source, target in existing_edges}
    hidden_ids = range(input_neurons, input_neurons + hidden_neurons)
    available = [
        (source, target)
        for source in range(input_neurons)
        for target in hidden_ids
        if (source, target) not in existing
    ]
    if count > len(available):
        raise ValueError(
            f"requested {count} candidates but only {len(available)} are available"
        )
    random.Random(int(seed) * 2027 + 17).shuffle(available)
    return available[:count]


def select_gradient_sensor_edges(
    model: TrainableSequentialClassifier,
    pixels,
    labels,
    config: EventMNISTConfig,
    *,
    count: int,
    candidate_count: int,
    scoring_batches: int,
    seed: int,
    birth_weight: float,
    device,
) -> tuple[list[int], list[float]]:
    """Rank inactive sensor edges by absolute task-loss gradient and grow top-k."""

    if count <= 0 or candidate_count < count:
        raise ValueError("gradient selection requires candidate_count >= count > 0")
    active_slots = model.graph.active_mask.nonzero(as_tuple=False).flatten().tolist()
    existing = {
        (int(model.graph.sources[slot].item()), int(model.graph.targets[slot].item()))
        for slot in active_slots
    }
    candidates = build_sensor_candidate_pool(
        model.input_neurons,
        model.hidden_neurons,
        existing,
        candidate_count,
        seed=seed,
    )
    if model.active_edge_count + candidate_count > model.graph.max_edges:
        raise ValueError("candidate scoring pool exceeds edge capacity")

    candidate_slots = [
        model.graph.sprout(source, target, long_term_weight=0.0, sign=1.0)
        for source, target in candidates
    ]
    criterion = nn.CrossEntropyLoss()
    scores = torch.zeros(candidate_count, device=device)
    generator = torch.Generator().manual_seed(seed * 4099 + 23)
    order = torch.randperm(pixels.shape[0], generator=generator)
    model.train()
    for batch_index in range(scoring_batches):
        start = batch_index * config.batch_size
        index = order[start : start + config.batch_size]
        if index.numel() == 0:
            break
        model.zero_grad(set_to_none=True)
        batch = pixels.index_select(0, index).to(device)
        target = labels.index_select(0, index).to(device)
        loss = criterion(model(batch), target)
        loss.backward()
        gradient = model.graph.long_term_weight.grad
        if gradient is None:
            raise RuntimeError("LTW gradient is unavailable during candidate scoring")
        scores.add_(gradient[candidate_slots].detach().abs())
        mark_step(device)
    model.zero_grad(set_to_none=True)
    score_values = [float(value) for value in scores.detach().cpu().tolist()]
    ranking = sorted(
        range(candidate_count),
        key=lambda index: (-score_values[index], candidates[index][0], candidates[index][1]),
    )
    selected = ranking[:count]
    _deactivate_slots(model, candidate_slots)

    selected_slots: list[int] = []
    selected_scores: list[float] = []
    for index in selected:
        source, target = candidates[index]
        selected_slots.append(
            model.graph.sprout(
                source,
                target,
                short_term_weight=0.0,
                long_term_weight=birth_weight,
                sign=1.0,
            )
        )
        selected_scores.append(score_values[index])
    return selected_slots, selected_scores


def prune_weak_sprouted_edges(
    model: TrainableSequentialClassifier,
    slots: Iterable[int],
    *,
    birth_weight: float,
    threshold_ratio: float,
    maximum_fraction: float,
) -> tuple[list[int], list[float]]:
    """Prune only weak peripheral sprouts; the seed/core mask is never touched."""

    if not 0 <= maximum_fraction <= 1:
        raise ValueError("maximum_fraction must be in [0, 1]")
    if not 0 <= threshold_ratio <= 1:
        raise ValueError("threshold_ratio must be in [0, 1]")
    active_slots = [slot for slot in slots if bool(model.graph.active_mask[slot].item())]
    limit = int(len(active_slots) * maximum_fraction)
    threshold = birth_weight * threshold_ratio
    weak = sorted(
        (
            (float(model.graph.long_term_weight[slot].detach().item()), slot)
            for slot in active_slots
            if float(model.graph.long_term_weight[slot].detach().item()) < threshold
        ),
        key=lambda pair: (pair[0], pair[1]),
    )[:limit]
    pruned_weights = [weight for weight, _ in weak]
    pruned_slots = [slot for _, slot in weak]
    _deactivate_slots(model, pruned_slots)
    return pruned_slots, pruned_weights


def summarize_utility_gated_structural_mnist(
    records: Iterable[dict],
    *,
    arms: Iterable[UtilityGatedArm] = UTILITY_GATED_ARMS,
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
            random_gains = [float(row["accuracy_gain_vs_random_sensor_48"]) for row in group]
            summary.append(
                {
                    "arm": arm.name,
                    "classifier": classifier,
                    "selection": arm.selection,
                    "seeds": len(group),
                    "mean_test_accuracy": statistics.fmean(accuracy),
                    "std_test_accuracy": statistics.pstdev(accuracy),
                    "mean_accuracy_gain_vs_fixed_warm_all": statistics.fmean(fixed_gains),
                    "fixed_improved_seed_count": sum(gain > 0 for gain in fixed_gains),
                    "mean_accuracy_gain_vs_random_sensor_48": statistics.fmean(random_gains),
                    "random_improved_seed_count": sum(gain > 0 for gain in random_gains),
                    "random_practical_gain_seed_count": sum(
                        gain >= 0.005 for gain in random_gains
                    ),
                    "initial_active_edges": int(group[0]["initial_active_edges"]),
                    "mean_final_active_edges": statistics.fmean(
                        float(row["final_active_edges"]) for row in group
                    ),
                    "sprouted_edges": int(group[0]["sprouted_edges"]),
                    "mean_pruned_sprouted_edges": statistics.fmean(
                        float(row["pruned_sprouted_edges"]) for row in group
                    ),
                    "mean_retained_sprouted_edges": statistics.fmean(
                        float(row["retained_sprouted_edges"]) for row in group
                    ),
                    "mean_selected_gradient_score": statistics.fmean(
                        float(row["mean_selected_gradient_score"]) for row in group
                    ),
                    "mean_event_rate_ratio": statistics.fmean(
                        float(row["event_rate_ratio"]) for row in group
                    ),
                    "mean_core_ltw_change": statistics.fmean(
                        float(row["mean_core_ltw_change"]) for row in group
                    ),
                    "mean_retained_sprouted_ltw_change_from_birth": statistics.fmean(
                        float(row["mean_retained_sprouted_ltw_change_from_birth"])
                        for row in group
                    ),
                    "mean_final_retained_sprouted_ltw": statistics.fmean(
                        float(row["mean_final_retained_sprouted_ltw"]) for row in group
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
                }
            )
    return summary


def plot_utility_gated_structural_mnist(
    summary: list[dict], path: str | pathlib.Path
) -> None:
    import matplotlib.pyplot as plt

    arms = list(dict.fromkeys(row["arm"] for row in summary))
    lookup = {(row["arm"], row["classifier"]): row for row in summary}
    positions = list(range(len(arms)))
    width = 0.38
    figure, axes = plt.subplots(3, 1, figsize=(17, 13), constrained_layout=True)
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
                * float(lookup[(arm, classifier)]["mean_accuracy_gain_vs_random_sensor_48"])
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
    axes[0].set_title("AMMC Gen-5 Phase 27: Utility-Gated Structural Plasticity")
    axes[1].axhline(0.0, color="#222222", linewidth=1)
    axes[1].set_ylabel("Gain over random sensor-48 (points)")
    axes[2].axhline(1.0, color="#222222", linewidth=1)
    axes[2].set_ylabel("Final / initial hidden event rate")
    for axis in axes:
        axis.set_xticks(positions, labels)
        axis.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _train_utility_arm(
    model,
    pixels,
    labels,
    config,
    *,
    arm,
    seed,
    device,
    warmup_epochs,
    scoring_batches,
    prune_after_epochs,
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
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
    criterion = nn.CrossEntropyLoss()
    model.train()
    sprouted_slots: list[int] = []
    selected_scores: list[float] = []
    pruned_slots: list[int] = []
    pruned_weights: list[float] = []
    sprout_reference = None
    start_time = time.perf_counter()
    for epoch in range(config.epochs):
        if train_ltw and epoch == warmup_epochs and arm.sprout_edges > 0:
            if arm.selection == "random":
                sprouted_slots = sprout_targeted_edges(
                    model,
                    "sensor",
                    arm.sprout_edges,
                    seed=seed,
                    birth_weight=arm.birth_weight,
                )
            elif arm.selection == "gradient":
                sprouted_slots, selected_scores = select_gradient_sensor_edges(
                    model,
                    pixels,
                    labels,
                    config,
                    count=arm.sprout_edges,
                    candidate_count=arm.candidate_edges,
                    scoring_batches=scoring_batches,
                    seed=seed,
                    birth_weight=arm.birth_weight,
                    device=device,
                )
            else:
                raise ValueError(f"unsupported structural selection: {arm.selection}")
            sprout_reference = model.graph.long_term_weight.detach().clone()
        if (
            sprouted_slots
            and arm.prune_fraction > 0
            and epoch == warmup_epochs + prune_after_epochs
        ):
            pruned_slots, pruned_weights = prune_weak_sprouted_edges(
                model,
                sprouted_slots,
                birth_weight=arm.birth_weight,
                threshold_ratio=arm.prune_threshold_ratio,
                maximum_fraction=arm.prune_fraction,
            )
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
    return {
        "train_seconds": time.perf_counter() - start_time,
        "sprouted_slots": sprouted_slots,
        "selected_scores": selected_scores,
        "pruned_slots": pruned_slots,
        "pruned_weights": pruned_weights,
        "sprout_reference": sprout_reference,
    }


def _deactivate_slots(model: TrainableSequentialClassifier, slots: Iterable[int]) -> None:
    slot_list = list(slots)
    if not slot_list:
        return
    with torch.no_grad():
        model.graph.active_mask[slot_list] = False
        model.graph.short_term_weight[slot_list] = 0
        model.graph.long_term_weight[slot_list] = 0
        model.graph.signs[slot_list] = 1
        model.graph.delay_steps[slot_list] = 0


def _attach_comparisons(records: list[dict]) -> None:
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
    random_control = {
        (int(row["seed"]), str(row["classifier"])): float(row["test_accuracy"])
        for row in records
        if row["arm"] == "random_sensor_48"
    }
    for row in records:
        key = (int(row["seed"]), str(row["classifier"]))
        row["paired_frozen_test_accuracy"] = frozen[key]
        row["accuracy_gain_vs_frozen"] = float(row["test_accuracy"]) - frozen[key]
        row["paired_fixed_warm_all_test_accuracy"] = fixed[key]
        row["accuracy_gain_vs_fixed_warm_all"] = float(row["test_accuracy"]) - fixed[key]
        row["paired_random_sensor_48_test_accuracy"] = random_control[key]
        row["accuracy_gain_vs_random_sensor_48"] = (
            float(row["test_accuracy"]) - random_control[key]
        )


def _select_arms(names: Iterable[str] | None) -> tuple[UtilityGatedArm, ...]:
    registry = {arm.name: arm for arm in UTILITY_GATED_ARMS}
    if names is None:
        return UTILITY_GATED_ARMS
    selected = tuple(names)
    unknown = [name for name in selected if name not in registry]
    if unknown:
        raise ValueError(f"unknown utility-gated arms: {', '.join(unknown)}")
    required = [
        name
        for name in (
            "raw",
            "frozen_recurrent",
            "fixed_warm_all",
            "random_sensor_48",
        )
        if name not in selected
    ]
    return tuple(registry[name] for name in (*required, *selected))


def _validate(
    config,
    arms,
    warmup_epochs,
    scoring_batches,
    prune_after_epochs,
    surrogate_slope,
    ltw_minimum,
    ltw_maximum,
):
    if not config.seeds or config.epochs <= 0:
        raise ValueError("at least one seed and positive epochs are required")
    if not 0 <= warmup_epochs < config.epochs:
        raise ValueError("warmup_epochs must be in [0, epochs)")
    if scoring_batches <= 0:
        raise ValueError("scoring_batches must be positive")
    if not 0 < prune_after_epochs < config.epochs - warmup_epochs:
        raise ValueError("prune_after_epochs must leave at least one post-prune epoch")
    if surrogate_slope <= 0:
        raise ValueError("surrogate slope must be positive")
    if ltw_minimum < 0 or ltw_maximum <= ltw_minimum:
        raise ValueError("LTW bounds must satisfy 0 <= minimum < maximum")
    base_edges = (
        config.image_size * config.sensor_fanout
        + config.hidden_neurons * config.recurrent_fanout
    )
    largest_temporary_pool = max(arm.candidate_edges for arm in arms)
    if base_edges + largest_temporary_pool > config.max_edges:
        raise ValueError(
            "gradient candidate pool exceeds the fixed edge-pool capacity: "
            f"{base_edges + largest_temporary_pool} > {config.max_edges}"
        )
    required = {"frozen_recurrent", "fixed_warm_all", "random_sensor_48"}
    if not required.issubset(arm.name for arm in arms):
        raise ValueError("frozen, fixed, and random-growth paired controls are required")


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
