"""Phase 30 trainable delay assignment on row-sequential MNIST.

Phase 29 established a large, seed-consistent benefit from heterogeneous fixed
recurrent delays.  This final MNIST diagnostic asks whether delay assignments
can be optimized without changing the proven sparse topology.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import pathlib
import statistics
import time
from typing import Iterable

from .delayed_sequential_mnist import (
    DelayedSequentialClassifier,
)
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
    _mean_absolute_change,
    _measure,
    _readout_parameter_count,
    _saturation_rate,
)
from .trainable_temporal_mnist import SurrogateSpike


@dataclass(frozen=True)
class TrainableDelayArm:
    name: str
    model_kind: str
    delay_pattern: str
    gate_mode: str
    delay_initialization: str
    train_delays: bool
    reservoir_learning_rate: float = 3e-4


TRAINABLE_DELAY_ARMS = (
    TrainableDelayArm("raw", "raw", "none", "fixed", "none", False, 0.0),
    TrainableDelayArm(
        "lif_no_delay_warm_all", "fixed", "none", "fixed", "none", False
    ),
    TrainableDelayArm(
        "fixed_distance012_warm_all",
        "fixed",
        "distance_0_2",
        "fixed",
        "distance",
        False,
    ),
    TrainableDelayArm(
        "learned_soft_distance_init",
        "learned",
        "distance_0_2",
        "soft",
        "distance",
        True,
    ),
    TrainableDelayArm(
        "learned_st_distance_init",
        "learned",
        "distance_0_2",
        "straight_through",
        "distance",
        True,
    ),
    TrainableDelayArm(
        "learned_soft_flat_init",
        "learned",
        "distance_0_2",
        "soft",
        "flat",
        True,
    ),
)


def available_trainable_delay_arms() -> tuple[str, ...]:
    return tuple(arm.name for arm in TRAINABLE_DELAY_ARMS)


class TrainableDelaySequentialClassifier(DelayedSequentialClassifier):
    """Sparse LIF classifier with differentiable delay-bucket gates."""

    delay_bucket_count = 3

    def __init__(
        self,
        config: EventMNISTConfig,
        *,
        seed: int,
        classifier: str,
        surrogate_slope: float,
        gate_mode: str,
        delay_initialization: str,
        train_delays: bool,
        device,
    ) -> None:
        if gate_mode not in {"soft", "straight_through"}:
            raise ValueError("gate_mode must be soft or straight_through")
        if delay_initialization not in {"distance", "flat"}:
            raise ValueError("delay_initialization must be distance or flat")
        super().__init__(
            config,
            seed=seed,
            classifier=classifier,
            train_ltw=True,
            surrogate_slope=surrogate_slope,
            delay_pattern="distance_0_2",
            max_delay_steps=2,
            device=device,
        )
        self.gate_mode = gate_mode
        self.delay_initialization = delay_initialization
        logits = torch.zeros(
            (self.graph.max_edges, self.delay_bucket_count),
            dtype=self.graph.long_term_weight.dtype,
            device=device,
        )
        recurrent = self.recurrent_edge_mask
        if delay_initialization == "distance":
            logits.fill_(-4.0)
            logits[:, 0] = 4.0
            slots = recurrent.nonzero(as_tuple=False).flatten()
            if slots.numel():
                # Moderate recurrent logits preserve the winning assignment
                # while leaving enough softmax gradient to change it.
                logits[slots] = -1.0
                logits[slots, self.graph.delay_steps[slots]] = 1.0
        else:
            logits[:, 0] = 4.0
            logits[:, 1:] = -4.0
            logits[recurrent] = 0.0
        self.delay_logits = nn.Parameter(logits, requires_grad=bool(train_delays))

    @property
    def recurrent_edge_mask(self):
        return self.graph.active_mask & (self.graph.sources >= self.input_neurons)

    def delay_probabilities(self):
        return torch.softmax(self.delay_logits, dim=1)

    def delay_gates(self):
        probabilities = self.delay_probabilities()
        if self.gate_mode == "straight_through":
            indices = probabilities.argmax(dim=1)
            hard = torch.nn.functional.one_hot(
                indices, num_classes=self.delay_bucket_count
            ).to(probabilities.dtype)
            gates = hard - probabilities.detach() + probabilities
        else:
            gates = probabilities
        immediate = torch.zeros_like(gates)
        immediate[:, 0] = 1.0
        return torch.where(self.recurrent_edge_mask.unsqueeze(1), gates, immediate)

    def selected_delays(self):
        return self.delay_probabilities().argmax(dim=1)

    def recurrent_delay_entropy(self):
        probabilities = self.delay_probabilities()[self.recurrent_edge_mask]
        if probabilities.numel() == 0:
            return self.delay_logits.new_zeros(())
        return -(
            probabilities * torch.log(probabilities.clamp_min(1e-8))
        ).sum(dim=1).mean()

    def forward(self, pixels, *, return_event_rate: bool = False):  # type: ignore[override]
        if pixels.ndim != 2 or pixels.shape[1] != self.config.image_size**2:
            raise ValueError("pixels must have shape [batch, image_size ** 2]")
        frames = pixels.reshape(
            pixels.shape[0], self.config.image_size, self.config.image_size
        )
        membrane = pixels.new_zeros((pixels.shape[0], self.hidden_neurons))
        hidden_spikes = torch.zeros_like(membrane)
        hidden_event_sum = pixels.new_zeros(())
        zero_state = pixels.new_zeros((pixels.shape[0], self.neuron_count))
        history: list = []
        for step in range(self.config.image_size):
            sensor_events = frames[:, step, :] * self.config.input_gain
            network_state = torch.cat((sensor_events, hidden_spikes), dim=1)
            history.insert(0, network_state)
            if len(history) > self.delay_bucket_count:
                history.pop()
            current = trainable_delay_current(
                self.graph,
                history,
                self.delay_gates(),
                zero_state=zero_state,
            )[:, self.input_neurons :]
            pre_reset = membrane * self.config.reservoir_leak + current
            hidden_spikes = SurrogateSpike.apply(
                pre_reset - self.config.reservoir_threshold,
                self.surrogate_slope,
            )
            membrane = pre_reset - hidden_spikes * self.config.reservoir_threshold
            hidden_event_sum = hidden_event_sum + hidden_spikes.mean()
        logits = self.readout(torch.cat((hidden_spikes, membrane), dim=1))
        if return_event_rate:
            return logits, hidden_event_sum / self.config.image_size
        return logits


def trainable_delay_current(graph, history: list, gates, *, zero_state):
    """Mix source states across delay buckets with per-edge differentiable gates."""

    signed_weight = (
        (graph.short_term_weight + graph.long_term_weight)
        * graph.signs
        * graph.active_mask.to(graph.long_term_weight.dtype)
    )
    output = zero_state.new_zeros((zero_state.shape[0], graph.out_features))
    for delay in range(gates.shape[1]):
        state = history[delay] if delay < len(history) else zero_state
        edge_current = (
            state.index_select(1, graph.sources)
            * (signed_weight * gates[:, delay]).unsqueeze(0)
        )
        output = output.index_add(1, graph.targets, edge_current)
    return output


@dataclass
class TrainableDelaysMNISTResult:
    config: EventMNISTConfig
    device: str
    warmup_epochs: int
    delay_learning_rate: float
    entropy_regularization: float
    arms: list[dict]
    records: list[dict]
    summary: list[dict]

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "trainable_delays_mnist.json"
        records_path = output / "trainable_delays_mnist_records.csv"
        summary_path = output / "trainable_delays_mnist_summary.csv"
        payload = {
            "config": asdict(self.config),
            "device": self.device,
            "warmup_epochs": self.warmup_epochs,
            "delay_learning_rate": self.delay_learning_rate,
            "entropy_regularization": self.entropy_regularization,
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
            plot_path = output / "trainable_delays_mnist_summary.png"
            plot_trainable_delays_mnist(self.summary, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def run_trainable_delays_mnist(
    config: EventMNISTConfig,
    *,
    device="auto",
    warmup_epochs: int = 10,
    surrogate_slope: float = 10.0,
    delay_learning_rate: float = 3e-3,
    entropy_regularization: float = 1e-3,
    arm_names: Iterable[str] | None = None,
    ltw_minimum: float = 0.0,
    ltw_maximum: float = 1.0,
) -> TrainableDelaysMNISTResult:
    if torch is None:
        raise ImportError("Phase 30 trainable delays require PyTorch")
    arms = _select_arms(arm_names)
    _validate(
        config,
        arms,
        warmup_epochs,
        surrogate_slope,
        delay_learning_rate,
        entropy_regularization,
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
                if arm.model_kind == "raw":
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
                    initial_delays = None
                elif arm.model_kind == "fixed":
                    model = DelayedSequentialClassifier(
                        config,
                        seed=seed,
                        classifier=classifier,
                        train_ltw=True,
                        surrogate_slope=surrogate_slope,
                        delay_pattern=arm.delay_pattern,
                        max_delay_steps=2 if arm.delay_pattern != "none" else 0,
                        device=resolved,
                    ).to(resolved)
                    initial_ltw = model.graph.long_term_weight.detach().clone()
                    initial_delays = model.graph.delay_steps.detach().clone()
                    _, _, initial_event_rate = _measure(
                        model, test_pixels, test_labels, config.batch_size, resolved
                    )
                else:
                    model = TrainableDelaySequentialClassifier(
                        config,
                        seed=seed,
                        classifier=classifier,
                        surrogate_slope=surrogate_slope,
                        gate_mode=arm.gate_mode,
                        delay_initialization=arm.delay_initialization,
                        train_delays=arm.train_delays,
                        device=resolved,
                    ).to(resolved)
                    initial_ltw = model.graph.long_term_weight.detach().clone()
                    initial_delays = model.selected_delays().detach().clone()
                    _, _, initial_event_rate = _measure(
                        model, test_pixels, test_labels, config.batch_size, resolved
                    )

                train_seconds = _train_arm(
                    model,
                    train_pixels,
                    train_labels,
                    config,
                    arm=arm,
                    seed=seed,
                    device=resolved,
                    warmup_epochs=warmup_epochs,
                    delay_learning_rate=delay_learning_rate,
                    entropy_regularization=entropy_regularization,
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
                recurrent_edges = 0
                delayed_edges = 0
                changed_delays = 0
                mean_delay = 0.0
                delay_entropy = 0.0
                mean_ltw_change = 0.0
                lower_saturation = 0.0
                upper_saturation = 0.0
                if isinstance(model, DelayedSequentialClassifier):
                    active = model.graph.active_mask
                    recurrent = active & (model.graph.sources >= model.input_neurons)
                    if isinstance(model, TrainableDelaySequentialClassifier):
                        final_delays = model.selected_delays().detach()
                        delay_entropy = float(model.recurrent_delay_entropy().item())
                    else:
                        final_delays = model.graph.delay_steps.detach()
                    selected = final_delays[recurrent]
                    active_edges = int(active.sum().item())
                    recurrent_edges = int(recurrent.sum().item())
                    delayed_edges = int((selected > 0).sum().item())
                    mean_delay = float(selected.to(torch.float32).mean().item())
                    changed_delays = int(
                        (final_delays[recurrent] != initial_delays[recurrent]).sum().item()
                    )
                    current = model.graph.long_term_weight.detach()
                    mean_ltw_change = _mean_absolute_change(current, initial_ltw, active)
                    lower_saturation = _saturation_rate(
                        current, active, ltw_minimum, lower=True
                    )
                    upper_saturation = _saturation_rate(
                        current, active, ltw_maximum, lower=False
                    )

                readout_parameters = _readout_parameter_count(model)
                effective_trainable = readout_parameters
                if isinstance(model, DelayedSequentialClassifier):
                    effective_trainable += active_edges
                if isinstance(model, TrainableDelaySequentialClassifier) and arm.train_delays:
                    effective_trainable += recurrent_edges * model.delay_bucket_count
                records.append(
                    {
                        "seed": int(seed),
                        "arm": arm.name,
                        "classifier": classifier,
                        "model_kind": arm.model_kind,
                        "gate_mode": arm.gate_mode,
                        "delay_initialization": arm.delay_initialization,
                        "train_delays": bool(arm.train_delays),
                        "train_accuracy": float(train_accuracy),
                        "test_accuracy": float(test_accuracy),
                        "active_edges": int(active_edges),
                        "recurrent_edges": int(recurrent_edges),
                        "delayed_edges": int(delayed_edges),
                        "changed_delay_assignments": int(changed_delays),
                        "mean_recurrent_delay": float(mean_delay),
                        "final_delay_entropy": float(delay_entropy),
                        "readout_parameters": int(readout_parameters),
                        "effective_trainable_parameters": int(effective_trainable),
                        "initial_hidden_event_rate": float(initial_event_rate),
                        "final_hidden_event_rate": float(final_event_rate),
                        "event_rate_ratio": float(
                            final_event_rate / max(initial_event_rate, 1e-12)
                            if initial_event_rate > 0.0 else 0.0
                        ),
                        "mean_absolute_ltw_change": float(mean_ltw_change),
                        "lower_ltw_saturation_rate": float(lower_saturation),
                        "upper_ltw_saturation_rate": float(upper_saturation),
                        "train_seconds": float(train_seconds),
                        "inference_seconds": float(inference_seconds),
                    }
                )

    _attach_fixed_delay_comparisons(records)
    return TrainableDelaysMNISTResult(
        config=config,
        device=device_kind(resolved),
        warmup_epochs=int(warmup_epochs),
        delay_learning_rate=float(delay_learning_rate),
        entropy_regularization=float(entropy_regularization),
        arms=[asdict(arm) for arm in arms],
        records=records,
        summary=summarize_trainable_delays_mnist(records, arms=arms),
    )


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
    delay_learning_rate,
    entropy_regularization,
    ltw_minimum,
    ltw_maximum,
):
    if isinstance(model, DelayedSequentialClassifier):
        groups = [
            {
                "params": list(model.readout.parameters()),
                "lr": config.learning_rate,
                "weight_decay": config.weight_decay,
            },
            {
                "params": [model.graph.long_term_weight],
                "lr": arm.reservoir_learning_rate,
                "weight_decay": 0.0,
            },
        ]
        if isinstance(model, TrainableDelaySequentialClassifier) and arm.train_delays:
            groups.append(
                {
                    "params": [model.delay_logits],
                    "lr": delay_learning_rate,
                    "weight_decay": 0.0,
                }
            )
        optimizer = torch.optim.AdamW(groups)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
    criterion = nn.CrossEntropyLoss()
    model.train()
    start_time = time.perf_counter()
    for epoch in range(config.epochs):
        dynamics_active = epoch >= warmup_epochs
        generator = torch.Generator().manual_seed(seed * 1000 + epoch)
        order = torch.randperm(pixels.shape[0], generator=generator)
        for start in range(0, order.numel(), config.batch_size):
            index = order[start : start + config.batch_size]
            batch = pixels.index_select(0, index).to(device)
            target = labels.index_select(0, index).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), target)
            if (
                dynamics_active
                and isinstance(model, TrainableDelaySequentialClassifier)
                and arm.train_delays
                and entropy_regularization > 0.0
            ):
                loss = loss + entropy_regularization * model.recurrent_delay_entropy()
            loss.backward()
            if isinstance(model, DelayedSequentialClassifier):
                ltw_gradient = model.graph.long_term_weight.grad
                if ltw_gradient is not None:
                    if dynamics_active:
                        ltw_gradient.mul_(model.graph.active_mask.to(ltw_gradient.dtype))
                    else:
                        ltw_gradient.zero_()
                if isinstance(model, TrainableDelaySequentialClassifier):
                    delay_gradient = model.delay_logits.grad
                    if delay_gradient is not None:
                        if dynamics_active and arm.train_delays:
                            delay_gradient.mul_(
                                model.recurrent_edge_mask.unsqueeze(1).to(
                                    delay_gradient.dtype
                                )
                            )
                        else:
                            delay_gradient.zero_()
            optimizer.step()
            if isinstance(model, DelayedSequentialClassifier):
                model.clamp_ltw(ltw_minimum, ltw_maximum)
            mark_step(device)
    sync(device)
    return time.perf_counter() - start_time


def summarize_trainable_delays_mnist(
    records: Iterable[dict],
    *,
    arms: Iterable[TrainableDelayArm] = TRAINABLE_DELAY_ARMS,
) -> list[dict]:
    rows = list(records)
    summary: list[dict] = []
    for arm in arms:
        for classifier in ("linear", "mlp"):
            group = [
                row for row in rows
                if row["arm"] == arm.name and row["classifier"] == classifier
            ]
            if not group:
                continue
            gains = [float(row["accuracy_gain_vs_fixed_distance"]) for row in group]
            summary.append(
                {
                    "arm": arm.name,
                    "classifier": classifier,
                    "model_kind": arm.model_kind,
                    "gate_mode": arm.gate_mode,
                    "seeds": len(group),
                    "mean_test_accuracy": statistics.fmean(
                        float(row["test_accuracy"]) for row in group
                    ),
                    "std_test_accuracy": statistics.pstdev(
                        float(row["test_accuracy"]) for row in group
                    ),
                    "mean_accuracy_gain_vs_fixed_distance": statistics.fmean(gains),
                    "improved_seed_count": sum(gain > 0.0 for gain in gains),
                    "practical_gain_seed_count": sum(gain >= 0.005 for gain in gains),
                    "effective_trainable_parameters": int(
                        group[0]["effective_trainable_parameters"]
                    ),
                    "mean_changed_delay_assignments": statistics.fmean(
                        int(row["changed_delay_assignments"]) for row in group
                    ),
                    "mean_recurrent_delay": statistics.fmean(
                        float(row["mean_recurrent_delay"]) for row in group
                    ),
                    "mean_final_delay_entropy": statistics.fmean(
                        float(row["final_delay_entropy"]) for row in group
                    ),
                    "mean_event_rate_ratio": statistics.fmean(
                        float(row["event_rate_ratio"]) for row in group
                    ),
                    "mean_absolute_ltw_change": statistics.fmean(
                        float(row["mean_absolute_ltw_change"]) for row in group
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


def plot_trainable_delays_mnist(summary: list[dict], path: str | pathlib.Path) -> None:
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
                100.0 * float(
                    lookup[(arm, classifier)]["mean_accuracy_gain_vs_fixed_distance"]
                ) for arm in arms
            ],
            width,
            label=classifier,
            color=color,
        )
        axes[2].bar(
            [position + offset for position in positions],
            [
                float(lookup[(arm, classifier)]["mean_changed_delay_assignments"])
                for arm in arms
            ],
            width,
            label=classifier,
            color=color,
        )
    labels = [arm.replace("_", "\n") for arm in arms]
    axes[0].set_ylabel("Engineering-validation accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_title("AMMC Gen-5 Phase 30: Trainable Delay Assignment")
    axes[1].axhline(0.0, color="#222222", linewidth=1)
    axes[1].set_ylabel("Gain over fixed distance delays (points)")
    axes[2].set_ylabel("Changed recurrent delay assignments")
    for axis in axes:
        axis.set_xticks(positions, labels)
        axis.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _attach_fixed_delay_comparisons(records: list[dict]) -> None:
    fixed = {
        (int(row["seed"]), str(row["classifier"])): float(row["test_accuracy"])
        for row in records if row["arm"] == "fixed_distance012_warm_all"
    }
    for row in records:
        baseline = fixed[(int(row["seed"]), str(row["classifier"]))]
        row["paired_fixed_distance_test_accuracy"] = baseline
        row["accuracy_gain_vs_fixed_distance"] = float(row["test_accuracy"]) - baseline


def _select_arms(names: Iterable[str] | None) -> tuple[TrainableDelayArm, ...]:
    registry = {arm.name: arm for arm in TRAINABLE_DELAY_ARMS}
    if names is None:
        return TRAINABLE_DELAY_ARMS
    selected = tuple(names)
    unknown = [name for name in selected if name not in registry]
    if unknown:
        raise ValueError(f"unknown trainable delay arms: {', '.join(unknown)}")
    required = [
        name for name in ("raw", "fixed_distance012_warm_all")
        if name not in selected
    ]
    return tuple(registry[name] for name in (*required, *selected))


def _validate(
    config,
    arms,
    warmup_epochs,
    surrogate_slope,
    delay_learning_rate,
    entropy_regularization,
    ltw_minimum,
    ltw_maximum,
):
    if not config.seeds or config.epochs <= 0:
        raise ValueError("at least one seed and positive epochs are required")
    if not 0 <= warmup_epochs < config.epochs:
        raise ValueError("warmup_epochs must be in [0, epochs)")
    if surrogate_slope <= 0.0 or delay_learning_rate <= 0.0:
        raise ValueError("surrogate slope and delay learning rate must be positive")
    if entropy_regularization < 0.0:
        raise ValueError("entropy_regularization must be non-negative")
    if ltw_minimum < 0.0 or ltw_maximum <= ltw_minimum:
        raise ValueError("LTW bounds must satisfy 0 <= minimum < maximum")
    if not any(arm.name == "fixed_distance012_warm_all" for arm in arms):
        raise ValueError("the paired fixed-distance control is required")


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
