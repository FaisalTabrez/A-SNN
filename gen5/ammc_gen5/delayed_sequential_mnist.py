"""Phase 29 executable axonal-delay ablation on row-sequential MNIST.

The sparse backend has always serialized ``delay_steps`` metadata, but the
sequential forward path previously ignored it.  This module makes those delays
causal through fixed history buckets while holding topology, LTWs, neurons, and
readout dimensions constant inside paired comparisons.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import pathlib
import statistics
from typing import Iterable

from .event_mnist import (
    EventMNISTConfig,
    _Classifier,
    _matched_raw_hidden_units,
    load_mnist_tensors,
    nn,
    torch,
)
from .runtime import device_kind, resolve_device, seed_everything
from .trainable_sequential_mnist import (
    TrainableSequentialClassifier,
    _mean_absolute_change,
    _measure,
    _readout_parameter_count,
    _saturation_rate,
    _train_arm,
)
from .trainable_temporal_mnist import SurrogateSpike


@dataclass(frozen=True)
class DelayedSequentialArm:
    name: str
    schedule: str
    scope: str
    reservoir_learning_rate: float
    delay_pattern: str
    max_delay_steps: int


DELAYED_SEQUENTIAL_ARMS = (
    DelayedSequentialArm("raw", "raw", "none", 0.0, "none", 0),
    DelayedSequentialArm(
        "lif_no_delay_frozen", "frozen", "none", 0.0, "none", 0
    ),
    DelayedSequentialArm(
        "lif_no_delay_warm_all", "warmup", "all", 3e-4, "none", 0
    ),
    DelayedSequentialArm(
        "recurrent_delay1_frozen", "frozen", "none", 0.0, "uniform_1", 1
    ),
    DelayedSequentialArm(
        "recurrent_delay1_warm_all", "warmup", "all", 3e-4, "uniform_1", 1
    ),
    DelayedSequentialArm(
        "recurrent_hash012_warm_all", "warmup", "all", 3e-4, "hash_0_2", 2
    ),
    DelayedSequentialArm(
        "recurrent_distance012_warm_all",
        "warmup",
        "all",
        3e-4,
        "distance_0_2",
        2,
    ),
)


def available_delayed_sequential_arms() -> tuple[str, ...]:
    return tuple(arm.name for arm in DELAYED_SEQUENTIAL_ARMS)


class DelayedSequentialClassifier(TrainableSequentialClassifier):
    """Final-state LIF classifier with executable fixed delay buckets."""

    def __init__(
        self,
        config: EventMNISTConfig,
        *,
        seed: int,
        classifier: str,
        train_ltw: bool,
        surrogate_slope: float,
        delay_pattern: str,
        max_delay_steps: int,
        device,
    ) -> None:
        if max_delay_steps < 0:
            raise ValueError("max_delay_steps must be non-negative")
        super().__init__(
            config,
            seed=seed,
            classifier=classifier,
            train_ltw=train_ltw,
            surrogate_slope=surrogate_slope,
            device=device,
        )
        self.delay_pattern = str(delay_pattern)
        self.max_delay_steps = int(max_delay_steps)
        assign_fixed_delays(
            self,
            pattern=self.delay_pattern,
            max_delay_steps=self.max_delay_steps,
            seed=seed,
        )

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
            if len(history) > self.max_delay_steps + 1:
                history.pop()
            current = delayed_sparse_current(
                self.graph,
                history,
                zero_state=zero_state,
                max_delay_steps=self.max_delay_steps,
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


def delayed_sparse_current(graph, history: list, *, zero_state, max_delay_steps: int):
    """Route each edge from the source-state bucket selected by delay metadata."""

    signed_weight = (
        (graph.short_term_weight + graph.long_term_weight)
        * graph.signs
        * graph.active_mask.to(graph.long_term_weight.dtype)
    )
    output = zero_state.new_zeros((zero_state.shape[0], graph.out_features))
    for delay in range(max_delay_steps + 1):
        state = history[delay] if delay < len(history) else zero_state
        selected = (
            graph.active_mask & (graph.delay_steps == delay)
        ).to(signed_weight.dtype)
        edge_current = (
            state.index_select(1, graph.sources)
            * (signed_weight * selected).unsqueeze(0)
        )
        output = output.index_add(1, graph.targets, edge_current)
    return output


def assign_fixed_delays(
    model: TrainableSequentialClassifier,
    *,
    pattern: str,
    max_delay_steps: int,
    seed: int,
) -> None:
    """Assign delays only to recurrent edges; sensor timing remains unchanged."""

    supported = {"none", "uniform_1", "hash_0_2", "distance_0_2"}
    if pattern not in supported:
        raise ValueError(f"unsupported delay pattern: {pattern}")
    if max_delay_steps < 0:
        raise ValueError("max_delay_steps must be non-negative")
    with torch.no_grad():
        model.graph.delay_steps.zero_()
        recurrent = model.graph.active_mask & (
            model.graph.sources >= model.input_neurons
        )
        slots = recurrent.nonzero(as_tuple=False).flatten().tolist()
        for slot in slots:
            source = int(model.graph.sources[slot].item()) - model.input_neurons
            target = int(model.graph.targets[slot].item()) - model.input_neurons
            if pattern == "none":
                delay = 0
            elif pattern == "uniform_1":
                delay = 1
            elif pattern == "hash_0_2":
                delay = (source * 31 + target * 17 + int(seed)) % 3
            else:
                direct = abs(source - target)
                circular = min(direct, model.hidden_neurons - direct)
                maximum = max(1, model.hidden_neurons // 2)
                delay = min(2, (3 * circular) // (maximum + 1))
            model.graph.delay_steps[slot] = min(int(delay), max_delay_steps)


@dataclass
class DelayedSequentialMNISTResult:
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
        json_path = output / "delayed_sequential_mnist.json"
        records_path = output / "delayed_sequential_mnist_records.csv"
        summary_path = output / "delayed_sequential_mnist_summary.csv"
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
            plot_path = output / "delayed_sequential_mnist_summary.png"
            plot_delayed_sequential_mnist(self.summary, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def run_delayed_sequential_mnist(
    config: EventMNISTConfig,
    *,
    device="auto",
    warmup_epochs: int = 10,
    surrogate_slope: float = 10.0,
    arm_names: Iterable[str] | None = None,
    ltw_minimum: float = 0.0,
    ltw_maximum: float = 1.0,
) -> DelayedSequentialMNISTResult:
    if torch is None:
        raise ImportError("Phase 29 delayed sequential MNIST requires PyTorch")
    arms = _select_arms(arm_names)
    _validate(config, arms, warmup_epochs, surrogate_slope, ltw_minimum, ltw_maximum)
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
                    initial_event_rate = 0.0
                    scope_mask = None
                else:
                    model = DelayedSequentialClassifier(
                        config,
                        seed=seed,
                        classifier=classifier,
                        train_ltw=train_ltw,
                        surrogate_slope=surrogate_slope,
                        delay_pattern=arm.delay_pattern,
                        max_delay_steps=arm.max_delay_steps,
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
                    scope_mask = (
                        model.graph.active_mask.clone()
                        if arm.scope == "all"
                        else torch.zeros_like(model.graph.active_mask)
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
                recurrent_edges = 0
                delayed_edges = 0
                delay_counts = [0, 0, 0]
                mean_recurrent_delay = 0.0
                scope_edges = 0
                mean_ltw_change = 0.0
                lower_saturation = 0.0
                upper_saturation = 0.0
                if isinstance(model, DelayedSequentialClassifier):
                    active = model.graph.active_mask
                    recurrent = active & (model.graph.sources >= model.input_neurons)
                    delays = model.graph.delay_steps[recurrent]
                    active_edges = int(active.sum().item())
                    recurrent_edges = int(recurrent.sum().item())
                    delayed_edges = int((delays > 0).sum().item())
                    mean_recurrent_delay = float(delays.to(torch.float32).mean().item())
                    for delay in range(3):
                        delay_counts[delay] = int((delays == delay).sum().item())
                    scope_edges = int(scope_mask.sum().item())
                    current = model.graph.long_term_weight.detach()
                    mean_ltw_change = _mean_absolute_change(current, initial_ltw, active)
                    lower_saturation = _saturation_rate(
                        current, active, ltw_minimum, lower=True
                    )
                    upper_saturation = _saturation_rate(
                        current, active, ltw_maximum, lower=False
                    )

                readout_parameters = _readout_parameter_count(model)
                effective_trainable = readout_parameters + (
                    scope_edges if train_ltw else 0
                )
                records.append(
                    {
                        "seed": int(seed),
                        "arm": arm.name,
                        "classifier": classifier,
                        "schedule": arm.schedule,
                        "delay_pattern": arm.delay_pattern,
                        "max_delay_steps": int(arm.max_delay_steps),
                        "warmup_epochs": int(warmup_epochs if train_ltw else 0),
                        "train_accuracy": float(train_accuracy),
                        "test_accuracy": float(test_accuracy),
                        "active_edges": int(active_edges),
                        "recurrent_edges": int(recurrent_edges),
                        "delayed_edges": int(delayed_edges),
                        "delay_0_recurrent_edges": int(delay_counts[0]),
                        "delay_1_recurrent_edges": int(delay_counts[1]),
                        "delay_2_recurrent_edges": int(delay_counts[2]),
                        "mean_recurrent_delay": float(mean_recurrent_delay),
                        "scope_trainable_edges": int(scope_edges),
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
                        "end_to_end_examples_per_second": float(
                            test_pixels.shape[0] / max(inference_seconds, 1e-12)
                        ),
                    }
                )

    _attach_no_delay_comparisons(records)
    return DelayedSequentialMNISTResult(
        config=config,
        device=device_kind(resolved),
        warmup_epochs=int(warmup_epochs),
        surrogate_slope=float(surrogate_slope),
        arms=[asdict(arm) for arm in arms],
        records=records,
        summary=summarize_delayed_sequential_mnist(records, arms=arms),
    )


def summarize_delayed_sequential_mnist(
    records: Iterable[dict],
    *,
    arms: Iterable[DelayedSequentialArm] = DELAYED_SEQUENTIAL_ARMS,
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
            gains = [float(row["accuracy_gain_vs_no_delay_control"]) for row in group]
            event_ratios = [
                float(row["event_rate_vs_no_delay_control"]) for row in group
                if float(row["event_rate_vs_no_delay_control"]) > 0.0
            ]
            summary.append(
                {
                    "arm": arm.name,
                    "classifier": classifier,
                    "schedule": arm.schedule,
                    "delay_pattern": arm.delay_pattern,
                    "seeds": len(group),
                    "mean_test_accuracy": statistics.fmean(
                        float(row["test_accuracy"]) for row in group
                    ),
                    "std_test_accuracy": statistics.pstdev(
                        float(row["test_accuracy"]) for row in group
                    ),
                    "mean_accuracy_gain_vs_no_delay_control": statistics.fmean(gains),
                    "improved_seed_count": sum(gain > 0.0 for gain in gains),
                    "practical_gain_seed_count": sum(gain >= 0.005 for gain in gains),
                    "active_edges": int(group[0]["active_edges"]),
                    "delayed_edges": statistics.fmean(
                        int(row["delayed_edges"]) for row in group
                    ),
                    "mean_recurrent_delay": statistics.fmean(
                        float(row["mean_recurrent_delay"]) for row in group
                    ),
                    "effective_trainable_parameters": int(
                        group[0]["effective_trainable_parameters"]
                    ),
                    "mean_event_rate_ratio": statistics.fmean(
                        float(row["event_rate_ratio"]) for row in group
                    ),
                    "mean_event_rate_vs_no_delay_control": (
                        statistics.fmean(event_ratios) if event_ratios else 0.0
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


def plot_delayed_sequential_mnist(summary: list[dict], path: str | pathlib.Path) -> None:
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
                    lookup[(arm, classifier)]["mean_accuracy_gain_vs_no_delay_control"]
                ) for arm in arms
            ],
            width,
            label=classifier,
            color=color,
        )
        axes[2].bar(
            [position + offset for position in positions],
            [
                float(
                    lookup[(arm, classifier)]["mean_event_rate_vs_no_delay_control"]
                ) for arm in arms
            ],
            width,
            label=classifier,
            color=color,
        )
    labels = [arm.replace("_", "\n") for arm in arms]
    axes[0].set_ylabel("Engineering-validation accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_title("AMMC Gen-5 Phase 29: Executable Axonal Delays")
    axes[1].axhline(0.0, color="#222222", linewidth=1)
    axes[1].set_ylabel("Gain over paired no-delay control (points)")
    axes[2].axhline(1.0, color="#222222", linewidth=1)
    axes[2].set_ylabel("Event rate / paired no-delay LIF")
    for axis in axes:
        axis.set_xticks(positions, labels)
        axis.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _attach_no_delay_comparisons(records: list[dict]) -> None:
    warm = {
        (int(row["seed"]), str(row["classifier"])): row
        for row in records if row["arm"] == "lif_no_delay_warm_all"
    }
    frozen = {
        (int(row["seed"]), str(row["classifier"])): row
        for row in records if row["arm"] == "lif_no_delay_frozen"
    }
    for row in records:
        key = (int(row["seed"]), str(row["classifier"]))
        control = frozen[key] if row["schedule"] == "frozen" else warm[key]
        row["paired_no_delay_test_accuracy"] = float(control["test_accuracy"])
        row["accuracy_gain_vs_no_delay_control"] = (
            float(row["test_accuracy"]) - float(control["test_accuracy"])
        )
        control_rate = float(control["final_hidden_event_rate"])
        row["event_rate_vs_no_delay_control"] = (
            float(row["final_hidden_event_rate"]) / max(control_rate, 1e-12)
            if control_rate > 0.0 else 0.0
        )


def _select_arms(names: Iterable[str] | None) -> tuple[DelayedSequentialArm, ...]:
    registry = {arm.name: arm for arm in DELAYED_SEQUENTIAL_ARMS}
    if names is None:
        return DELAYED_SEQUENTIAL_ARMS
    selected = tuple(names)
    unknown = [name for name in selected if name not in registry]
    if unknown:
        raise ValueError(f"unknown delayed sequential arms: {', '.join(unknown)}")
    required = [
        name for name in ("raw", "lif_no_delay_frozen", "lif_no_delay_warm_all")
        if name not in selected
    ]
    return tuple(registry[name] for name in (*required, *selected))


def _validate(config, arms, warmup_epochs, surrogate_slope, ltw_minimum, ltw_maximum):
    if not config.seeds or config.epochs <= 0:
        raise ValueError("at least one seed and positive epochs are required")
    if not 0 <= warmup_epochs < config.epochs:
        raise ValueError("warmup_epochs must be in [0, epochs)")
    if surrogate_slope <= 0.0:
        raise ValueError("surrogate_slope must be positive")
    if ltw_minimum < 0.0 or ltw_maximum <= ltw_minimum:
        raise ValueError("LTW bounds must satisfy 0 <= minimum < maximum")
    required_edges = (
        config.image_size * config.sensor_fanout
        + config.hidden_neurons * config.recurrent_fanout
    )
    if required_edges > config.max_edges:
        raise ValueError(
            f"sequential topology requires {required_edges} edges but max_edges is {config.max_edges}"
        )
    required = {"lif_no_delay_frozen", "lif_no_delay_warm_all"}
    if not required.issubset(arm.name for arm in arms):
        raise ValueError("paired frozen and warm no-delay controls are required")


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
