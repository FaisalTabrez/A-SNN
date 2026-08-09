"""Phase 28 adaptive-neuron ablation on row-sequential MNIST.

Phase 27 showed that one-shot absolute-gradient sprouting was worse than a
paired random-growth control.  This phase freezes topology again and tests a
different temporal-capacity mechanism: a slow, activity-dependent firing
threshold on a controlled fraction of hidden neurons.  The sparse graph,
readout shape, LTW schedule, and seed are held fixed inside every comparison.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import pathlib
import random
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
from .runtime import device_kind, mark_step, resolve_device, seed_everything, sync
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
class AdaptiveSequentialArm:
    name: str
    schedule: str
    scope: str
    reservoir_learning_rate: float
    adaptive_fraction: float


ADAPTIVE_SEQUENTIAL_ARMS = (
    AdaptiveSequentialArm("raw", "raw", "none", 0.0, 0.0),
    AdaptiveSequentialArm("lif_frozen", "frozen", "none", 0.0, 0.0),
    AdaptiveSequentialArm("lif_warm_all", "warmup", "all", 3e-4, 0.0),
    AdaptiveSequentialArm("alif50_frozen", "frozen", "none", 0.0, 0.5),
    AdaptiveSequentialArm("alif25_warm_all", "warmup", "all", 3e-4, 0.25),
    AdaptiveSequentialArm("alif50_warm_all", "warmup", "all", 3e-4, 0.5),
    AdaptiveSequentialArm("alif100_warm_all", "warmup", "all", 3e-4, 1.0),
)


def available_adaptive_sequential_arms() -> tuple[str, ...]:
    return tuple(arm.name for arm in ADAPTIVE_SEQUENTIAL_ARMS)


class AdaptiveSequentialClassifier(TrainableSequentialClassifier):
    """Fixed sparse sequential SNN with optional adaptive thresholds.

    Adaptation is a non-trainable neuron state.  Keeping it fixed avoids adding
    optimizer parameters, and the readout still receives exactly the same
    ``[final spikes, final membrane]`` features as the LIF control.
    """

    def __init__(
        self,
        config: EventMNISTConfig,
        *,
        seed: int,
        classifier: str,
        train_ltw: bool,
        surrogate_slope: float,
        adaptive_fraction: float,
        adaptation_decay: float,
        adaptation_strength: float,
        device,
    ) -> None:
        if not 0.0 <= adaptive_fraction <= 1.0:
            raise ValueError("adaptive_fraction must be in [0, 1]")
        if not 0.0 <= adaptation_decay < 1.0:
            raise ValueError("adaptation_decay must be in [0, 1)")
        if adaptation_strength < 0.0:
            raise ValueError("adaptation_strength must be non-negative")
        super().__init__(
            config,
            seed=seed,
            classifier=classifier,
            train_ltw=train_ltw,
            surrogate_slope=surrogate_slope,
            device=device,
        )
        self.adaptive_fraction = float(adaptive_fraction)
        self.adaptation_decay = float(adaptation_decay)
        self.adaptation_strength = float(adaptation_strength)
        adaptive_count = round(self.hidden_neurons * self.adaptive_fraction)
        indices = list(range(self.hidden_neurons))
        random.Random(seed + 28_000).shuffle(indices)
        mask = torch.zeros(self.hidden_neurons, dtype=torch.bool, device=device)
        if adaptive_count:
            mask[indices[:adaptive_count]] = True
        self.register_buffer("adaptive_mask", mask)

    @property
    def adaptive_neuron_count(self) -> int:
        return int(self.adaptive_mask.sum().item())

    def forward(
        self,
        pixels,
        *,
        return_event_rate: bool = False,
        return_diagnostics: bool = False,
    ):  # type: ignore[override]
        if pixels.ndim != 2 or pixels.shape[1] != self.config.image_size**2:
            raise ValueError("pixels must have shape [batch, image_size ** 2]")
        frames = pixels.reshape(
            pixels.shape[0], self.config.image_size, self.config.image_size
        )
        membrane = pixels.new_zeros((pixels.shape[0], self.hidden_neurons))
        hidden_spikes = torch.zeros_like(membrane)
        adaptation = torch.zeros_like(membrane)
        hidden_event_sum = pixels.new_zeros(())
        adaptive_mask = self.adaptive_mask.to(pixels.dtype).unsqueeze(0)
        for step in range(self.config.image_size):
            sensor_events = frames[:, step, :] * self.config.input_gain
            network_state = torch.cat((sensor_events, hidden_spikes), dim=1)
            current = self.graph(network_state)[:, self.input_neurons :]
            threshold = (
                self.config.reservoir_threshold
                + self.adaptation_strength * adaptation
            )
            pre_reset = membrane * self.config.reservoir_leak + current
            hidden_spikes = SurrogateSpike.apply(
                pre_reset - threshold,
                self.surrogate_slope,
            )
            membrane = pre_reset - hidden_spikes * threshold
            adaptation = (
                adaptation * self.adaptation_decay
                + hidden_spikes * adaptive_mask
            )
            hidden_event_sum = hidden_event_sum + hidden_spikes.mean()
        logits = self.readout(torch.cat((hidden_spikes, membrane), dim=1))
        event_rate = hidden_event_sum / self.config.image_size
        if self.adaptive_neuron_count:
            final_adaptation = adaptation[:, self.adaptive_mask].mean()
        else:
            final_adaptation = pixels.new_zeros(())
        mean_threshold = (
            self.config.reservoir_threshold
            + self.adaptation_strength * final_adaptation
        )
        if return_diagnostics:
            return logits, event_rate, final_adaptation, mean_threshold
        if return_event_rate:
            return logits, event_rate
        return logits


@dataclass
class AdaptiveSequentialMNISTResult:
    config: EventMNISTConfig
    device: str
    warmup_epochs: int
    surrogate_slope: float
    adaptation_decay: float
    adaptation_strength: float
    arms: list[dict]
    records: list[dict]
    summary: list[dict]

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "adaptive_sequential_mnist.json"
        records_path = output / "adaptive_sequential_mnist_records.csv"
        summary_path = output / "adaptive_sequential_mnist_summary.csv"
        payload = {
            "config": asdict(self.config),
            "device": self.device,
            "warmup_epochs": self.warmup_epochs,
            "surrogate_slope": self.surrogate_slope,
            "adaptation_decay": self.adaptation_decay,
            "adaptation_strength": self.adaptation_strength,
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
            plot_path = output / "adaptive_sequential_mnist_summary.png"
            plot_adaptive_sequential_mnist(self.summary, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def run_adaptive_sequential_mnist(
    config: EventMNISTConfig,
    *,
    device="auto",
    warmup_epochs: int = 10,
    surrogate_slope: float = 10.0,
    adaptation_decay: float = 0.95,
    adaptation_strength: float = 0.5,
    arm_names: Iterable[str] | None = None,
    ltw_minimum: float = 0.0,
    ltw_maximum: float = 1.0,
) -> AdaptiveSequentialMNISTResult:
    if torch is None:
        raise ImportError("Phase 28 adaptive sequential MNIST requires PyTorch")
    arms = _select_arms(arm_names)
    _validate(
        config,
        arms,
        warmup_epochs,
        surrogate_slope,
        adaptation_decay,
        adaptation_strength,
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
                    initial_event_rate = 0.0
                    initial_adaptation = 0.0
                    scope_mask = None
                else:
                    model = AdaptiveSequentialClassifier(
                        config,
                        seed=seed,
                        classifier=classifier,
                        train_ltw=train_ltw,
                        surrogate_slope=surrogate_slope,
                        adaptive_fraction=arm.adaptive_fraction,
                        adaptation_decay=adaptation_decay,
                        adaptation_strength=adaptation_strength,
                        device=resolved,
                    ).to(resolved)
                    initial_ltw = model.graph.long_term_weight.detach().clone()
                    _, _, initial_event_rate, initial_adaptation, _ = _measure_adaptive(
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
                if isinstance(model, AdaptiveSequentialClassifier):
                    (
                        test_accuracy,
                        inference_seconds,
                        final_event_rate,
                        final_adaptation,
                        final_threshold,
                    ) = _measure_adaptive(
                        model,
                        test_pixels,
                        test_labels,
                        config.batch_size,
                        resolved,
                    )
                else:
                    test_accuracy, inference_seconds, _ = _measure(
                        model, test_pixels, test_labels, config.batch_size, resolved
                    )
                    final_event_rate = 0.0
                    final_adaptation = 0.0
                    final_threshold = config.reservoir_threshold

                active_edges = 0
                scope_edges = 0
                mean_ltw_change = 0.0
                lower_saturation = 0.0
                upper_saturation = 0.0
                adaptive_neurons = 0
                if isinstance(model, AdaptiveSequentialClassifier):
                    active = model.graph.active_mask
                    current = model.graph.long_term_weight.detach()
                    active_edges = int(active.sum().item())
                    scope_edges = int(scope_mask.sum().item())
                    adaptive_neurons = model.adaptive_neuron_count
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
                        "adaptive_fraction": float(arm.adaptive_fraction),
                        "adaptive_neurons": int(adaptive_neurons),
                        "adaptation_decay": float(adaptation_decay),
                        "adaptation_strength": float(adaptation_strength),
                        "warmup_epochs": int(warmup_epochs if train_ltw else 0),
                        "train_accuracy": float(train_accuracy),
                        "test_accuracy": float(test_accuracy),
                        "active_edges": int(active_edges),
                        "scope_trainable_edges": int(scope_edges),
                        "readout_parameters": int(readout_parameters),
                        "effective_trainable_parameters": int(effective_trainable),
                        "initial_hidden_event_rate": float(initial_event_rate),
                        "final_hidden_event_rate": float(final_event_rate),
                        "event_rate_ratio": float(
                            final_event_rate / max(initial_event_rate, 1e-12)
                            if initial_event_rate > 0.0
                            else 0.0
                        ),
                        "initial_mean_adaptation": float(initial_adaptation),
                        "final_mean_adaptation": float(final_adaptation),
                        "final_mean_adaptive_threshold": float(final_threshold),
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

    _attach_lif_comparisons(records)
    return AdaptiveSequentialMNISTResult(
        config=config,
        device=device_kind(resolved),
        warmup_epochs=int(warmup_epochs),
        surrogate_slope=float(surrogate_slope),
        adaptation_decay=float(adaptation_decay),
        adaptation_strength=float(adaptation_strength),
        arms=[asdict(arm) for arm in arms],
        records=records,
        summary=summarize_adaptive_sequential_mnist(records, arms=arms),
    )


def summarize_adaptive_sequential_mnist(
    records: Iterable[dict],
    *,
    arms: Iterable[AdaptiveSequentialArm] = ADAPTIVE_SEQUENTIAL_ARMS,
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
            gains = [float(row["accuracy_gain_vs_lif_control"]) for row in group]
            event_ratios = [
                float(row["event_rate_vs_lif_control"]) for row in group
                if float(row["event_rate_vs_lif_control"]) > 0.0
            ]
            summary.append(
                {
                    "arm": arm.name,
                    "classifier": classifier,
                    "schedule": arm.schedule,
                    "adaptive_fraction": float(arm.adaptive_fraction),
                    "seeds": len(group),
                    "mean_test_accuracy": statistics.fmean(
                        float(row["test_accuracy"]) for row in group
                    ),
                    "std_test_accuracy": statistics.pstdev(
                        float(row["test_accuracy"]) for row in group
                    ),
                    "mean_accuracy_gain_vs_lif_control": statistics.fmean(gains),
                    "improved_seed_count": sum(gain > 0.0 for gain in gains),
                    "practical_gain_seed_count": sum(gain >= 0.005 for gain in gains),
                    "active_edges": int(group[0]["active_edges"]),
                    "adaptive_neurons": int(group[0]["adaptive_neurons"]),
                    "effective_trainable_parameters": int(
                        group[0]["effective_trainable_parameters"]
                    ),
                    "mean_event_rate_ratio": statistics.fmean(
                        float(row["event_rate_ratio"]) for row in group
                    ),
                    "mean_event_rate_vs_lif_control": (
                        statistics.fmean(event_ratios) if event_ratios else 0.0
                    ),
                    "mean_final_adaptation": statistics.fmean(
                        float(row["final_mean_adaptation"]) for row in group
                    ),
                    "mean_final_adaptive_threshold": statistics.fmean(
                        float(row["final_mean_adaptive_threshold"]) for row in group
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


def plot_adaptive_sequential_mnist(summary: list[dict], path: str | pathlib.Path) -> None:
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
                    lookup[(arm, classifier)]["mean_accuracy_gain_vs_lif_control"]
                )
                for arm in arms
            ],
            width,
            label=classifier,
            color=color,
        )
        axes[2].bar(
            [position + offset for position in positions],
            [
                float(lookup[(arm, classifier)]["mean_event_rate_vs_lif_control"])
                for arm in arms
            ],
            width,
            label=classifier,
            color=color,
        )
    labels = [arm.replace("_", "\n") for arm in arms]
    axes[0].set_ylabel("Engineering-validation accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_title("AMMC Gen-5 Phase 28: Adaptive-Neuron Ablation")
    axes[1].axhline(0.0, color="#222222", linewidth=1)
    axes[1].set_ylabel("Gain over paired LIF control (points)")
    axes[2].axhline(1.0, color="#222222", linewidth=1)
    axes[2].set_ylabel("Event rate / paired LIF")
    for axis in axes:
        axis.set_xticks(positions, labels)
        axis.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _measure_adaptive(model, pixels, labels, batch_size, device):
    model.eval()
    correct = 0
    event_total = 0.0
    adaptation_total = 0.0
    threshold_total = 0.0
    import time

    start_time = time.perf_counter()
    with torch.no_grad():
        for start in range(0, pixels.shape[0], batch_size):
            batch = pixels[start : start + batch_size].to(device)
            logits, event_rate, adaptation, threshold = model(
                batch, return_diagnostics=True
            )
            count = batch.shape[0]
            event_total += float(event_rate.item()) * count
            adaptation_total += float(adaptation.item()) * count
            threshold_total += float(threshold.item()) * count
            prediction = logits.argmax(dim=1).cpu()
            correct += int((prediction == labels[start : start + count]).sum().item())
            mark_step(device)
    sync(device)
    seconds = time.perf_counter() - start_time
    count = pixels.shape[0]
    return (
        correct / count,
        seconds,
        event_total / count,
        adaptation_total / count,
        threshold_total / count,
    )


def _attach_lif_comparisons(records: list[dict]) -> None:
    warm = {
        (int(row["seed"]), str(row["classifier"])): row
        for row in records
        if row["arm"] == "lif_warm_all"
    }
    frozen = {
        (int(row["seed"]), str(row["classifier"])): row
        for row in records
        if row["arm"] == "lif_frozen"
    }
    for row in records:
        key = (int(row["seed"]), str(row["classifier"]))
        control = frozen[key] if row["schedule"] == "frozen" else warm[key]
        row["paired_lif_test_accuracy"] = float(control["test_accuracy"])
        row["accuracy_gain_vs_lif_control"] = (
            float(row["test_accuracy"]) - float(control["test_accuracy"])
        )
        control_rate = float(control["final_hidden_event_rate"])
        row["event_rate_vs_lif_control"] = (
            float(row["final_hidden_event_rate"]) / max(control_rate, 1e-12)
            if control_rate > 0.0 else 0.0
        )


def _select_arms(names: Iterable[str] | None) -> tuple[AdaptiveSequentialArm, ...]:
    registry = {arm.name: arm for arm in ADAPTIVE_SEQUENTIAL_ARMS}
    if names is None:
        return ADAPTIVE_SEQUENTIAL_ARMS
    selected = tuple(names)
    unknown = [name for name in selected if name not in registry]
    if unknown:
        raise ValueError(f"unknown adaptive sequential arms: {', '.join(unknown)}")
    required = [
        name for name in ("raw", "lif_frozen", "lif_warm_all")
        if name not in selected
    ]
    return tuple(registry[name] for name in (*required, *selected))


def _validate(
    config,
    arms,
    warmup_epochs,
    surrogate_slope,
    adaptation_decay,
    adaptation_strength,
    ltw_minimum,
    ltw_maximum,
):
    if not config.seeds or config.epochs <= 0:
        raise ValueError("at least one seed and positive epochs are required")
    if not 0 <= warmup_epochs < config.epochs:
        raise ValueError("warmup_epochs must be in [0, epochs)")
    if surrogate_slope <= 0.0:
        raise ValueError("surrogate_slope must be positive")
    if not 0.0 <= adaptation_decay < 1.0:
        raise ValueError("adaptation_decay must be in [0, 1)")
    if adaptation_strength < 0.0:
        raise ValueError("adaptation_strength must be non-negative")
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
    required = {"lif_frozen", "lif_warm_all"}
    if not required.issubset(arm.name for arm in arms):
        raise ValueError("paired frozen and warm LIF controls are required")


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
