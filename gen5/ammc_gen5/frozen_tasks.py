"""Frozen diversified task benchmarks for AMMC Gen-5.

This module deliberately disables evolution, plasticity, and optimizer updates.
It asks a narrower question: how does the current sparse AMMC substrate behave
when the same frozen wiring is exposed to non-foraging temporal tasks?

The first benchmark set is synthetic and download-free so it can run in Colab,
CUDA, TPU/XLA, or local CPU without dataset setup. It keeps the existing
8-sensor / 4-motor convention:

- sensor channels 0:4 are directional "food-like" channels
- sensor channels 4:8 are auxiliary/toxin/cue channels
- motor channels 8:12 represent north/east/south/west class decisions
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

try:  # pragma: no cover - exercised in accelerator runtimes
    import torch
except Exception:  # pragma: no cover
    torch = None

from .dynamic_sparse import EdgeRecord
from .evaluation import default_foraging_seed_edges
from .evolver import TensorEvolver, TensorEvolverConfig
from .runtime import make_generator, mark_step, resolve_device, seed_everything, sync
from .transducer import TransducerConfig, VectorizedTransducer


def _require_torch() -> None:
    if torch is None:
        raise ImportError("AMMC Gen-5 frozen task benchmarks require PyTorch")


@dataclass(frozen=True)
class FrozenTaskConfig:
    """Configuration for frozen diversified task evaluation."""

    tasks: tuple[str, ...] = (
        "direction_copy",
        "anti_toxin",
        "cue_switch",
        "delayed_recall",
        "two_pulse_sum",
    )
    sample_count: int = 4096
    timesteps: int = 8
    seed: int = 42
    neuron_count: int = 16
    max_edges: int = 128
    sensor_channels: int = 8
    motor_channels: int = 4
    sensor_gain: float = 1.0
    motor_gain: float = 1.0
    leak: float = 0.9
    threshold: float = 1.0
    input_amplitude: float = 0.75
    device: str = "auto"
    seed_edges: tuple[EdgeRecord, ...] = field(default_factory=default_foraging_seed_edges)


@dataclass(frozen=True)
class FrozenTaskSummaryRecord:
    """One row summarizing a frozen model on one task."""

    task: str
    samples: int
    timesteps: int
    target_rule: str
    frozen_ammc_accuracy: float
    random_accuracy: float
    instant_reflex_accuracy: float
    integrated_reflex_accuracy: float
    oracle_accuracy: float
    inactive_output_rate: float
    mean_correct_evidence_margin: float


@dataclass(frozen=True)
class FrozenTaskResult:
    """Complete frozen diversified task result."""

    config: dict
    summary: list[FrozenTaskSummaryRecord]

    def to_json_dict(self) -> dict:
        return {
            "config": self.config,
            "summary": [asdict(row) for row in self.summary],
        }


@dataclass(frozen=True)
class FrozenProbeConfig:
    """Configuration for linear probes over frozen AMMC traces."""

    task_config: FrozenTaskConfig = field(default_factory=FrozenTaskConfig)
    train_fraction: float = 0.7
    epochs: int = 200
    learning_rate: float = 0.05
    weight_decay: float = 0.001
    standardize_features: bool = True


@dataclass(frozen=True)
class FrozenProbeSummaryRecord:
    """One row summarizing a linear probe for one frozen task."""

    task: str
    samples: int
    train_samples: int
    test_samples: int
    timesteps: int
    target_rule: str
    feature_dim: int
    frozen_ammc_accuracy: float
    linear_probe_accuracy: float
    linear_probe_train_accuracy: float
    random_accuracy: float
    instant_reflex_accuracy: float
    integrated_reflex_accuracy: float
    inactive_output_rate: float
    representation_gain_over_frozen: float
    representation_gain_over_best_reflex: float
    final_probe_loss: float


@dataclass(frozen=True)
class FrozenProbeResult:
    """Complete frozen representation probe result."""

    config: dict
    summary: list[FrozenProbeSummaryRecord]

    def to_json_dict(self) -> dict:
        return {
            "config": self.config,
            "summary": [asdict(row) for row in self.summary],
        }


@dataclass(frozen=True)
class _TaskBatch:
    inputs: object
    targets: object
    target_rule: str


class FrozenTaskRunner:
    """Evaluate a frozen sparse AMMC brain on small diversified tasks."""

    def __init__(self, config: FrozenTaskConfig | None = None) -> None:
        self.config = config or FrozenTaskConfig()
        if self.config.sample_count < 2:
            raise ValueError("sample_count must be at least 2")
        if self.config.timesteps < 2:
            raise ValueError("timesteps must be at least 2")
        if self.config.neuron_count < self.config.sensor_channels + self.config.motor_channels:
            raise ValueError("neuron_count must fit sensor and motor channels")
        if self.config.max_edges < len(self.config.seed_edges):
            raise ValueError("max_edges must fit seed_edges")

    def run(self) -> FrozenTaskResult:
        _require_torch()
        device = resolve_device(self.config.device)
        seed_everything(self.config.seed, device=device)
        generator = make_generator(self.config.seed, device=device)

        summary: list[FrozenTaskSummaryRecord] = []
        for offset, task_name in enumerate(self.config.tasks):
            task_generator = make_generator(self.config.seed + offset + 1, device=device)
            batch = self._make_task(task_name, task_generator, device)
            summary.append(self._evaluate_task(task_name, batch, generator, device))
            sync(device)

        return FrozenTaskResult(
            config=self._jsonable_config(device),
            summary=summary,
        )

    def save_outputs(
        self,
        result: FrozenTaskResult,
        output_dir: str | Path,
        *,
        plot: bool = True,
    ) -> dict[str, str]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        json_path = output / "frozen_diversified_tasks.json"
        json_path.write_text(json.dumps(result.to_json_dict(), indent=2) + "\n", encoding="utf-8")

        summary_csv = output / "frozen_diversified_tasks_summary.csv"
        _write_csv(summary_csv, [asdict(row) for row in result.summary])

        paths = {
            "json": str(json_path),
            "summary_csv": str(summary_csv),
        }
        if plot:
            try:
                plot_path = output / "frozen_diversified_tasks_summary.png"
                plot_frozen_task_result(result, plot_path)
                paths["plot"] = str(plot_path)
            except Exception as exc:  # pragma: no cover - optional plotting
                paths["plot"] = f"skipped: {exc}"
        return paths

    def _evaluate_task(self, task_name: str, batch: _TaskBatch, generator, device) -> FrozenTaskSummaryRecord:
        cfg = self.config
        inputs = batch.inputs
        targets = batch.targets
        evidence = self._frozen_ammc_evidence(inputs, device)
        predictions = evidence.argmax(dim=1)
        frozen_accuracy = _accuracy(predictions, targets)
        inactive_rate = float((evidence.max(dim=1).values <= 1e-8).to(torch.float32).mean().item())
        correct_evidence = evidence.gather(1, targets.unsqueeze(1)).squeeze(1)
        masked = evidence.clone()
        masked.scatter_(1, targets.unsqueeze(1), float("-inf"))
        competitor = masked.max(dim=1).values
        mean_margin = float((correct_evidence - competitor).mean().item())

        random_predictions = torch.randint(
            0,
            cfg.motor_channels,
            targets.shape,
            device=targets.device,
            generator=generator,
        )
        instant_reflex = _instant_reflex_predictions(inputs, cfg.motor_channels)
        integrated_reflex = _integrated_reflex_predictions(inputs, cfg.motor_channels)

        return FrozenTaskSummaryRecord(
            task=task_name,
            samples=int(targets.numel()),
            timesteps=cfg.timesteps,
            target_rule=batch.target_rule,
            frozen_ammc_accuracy=frozen_accuracy,
            random_accuracy=_accuracy(random_predictions, targets),
            instant_reflex_accuracy=_accuracy(instant_reflex, targets),
            integrated_reflex_accuracy=_accuracy(integrated_reflex, targets),
            oracle_accuracy=1.0,
            inactive_output_rate=inactive_rate,
            mean_correct_evidence_margin=mean_margin,
        )

    def _frozen_ammc_evidence(self, inputs, device):
        return self._frozen_ammc_trace(inputs, device)["evidence"]

    def _frozen_ammc_trace(self, inputs, device) -> dict:
        """Run frozen AMMC dynamics and return readout plus probe features."""

        cfg = self.config
        batch_size = int(inputs.shape[0])
        brain = TensorEvolver(
            TensorEvolverConfig(
                population_size=batch_size,
                neuron_count=cfg.neuron_count,
                max_edges=cfg.max_edges,
                survivor_fraction=0.5,
                ltw_noise_std=0.0,
                sprout_probability=0.0,
                prune_probability=0.0,
            ),
            device=device,
            dtype=inputs.dtype,
        )
        brain.seed_from_edges(cfg.seed_edges)
        transducer = VectorizedTransducer(
            TransducerConfig(
                neuron_count=cfg.neuron_count,
                sensor_channels=cfg.sensor_channels,
                motor_channels=cfg.motor_channels,
                sensor_gain=cfg.sensor_gain,
                motor_gain=cfg.motor_gain,
                leak=cfg.leak,
                threshold=cfg.threshold,
            )
        )
        membrane = inputs.new_zeros((batch_size, cfg.neuron_count))
        spike_counts = inputs.new_zeros((batch_size, cfg.neuron_count))
        motor_start = cfg.sensor_channels
        motor_end = motor_start + cfg.motor_channels
        evidence = inputs.new_zeros((batch_size, cfg.motor_channels))
        for step in range(cfg.timesteps):
            neural_input = transducer.encode_sensors(inputs[:, step, :])
            recurrent_current = brain(membrane)
            spikes, membrane = transducer.lif_step(neural_input + recurrent_current, membrane)
            spike_counts = spike_counts + spikes
            evidence = evidence + spikes[:, motor_start:motor_end]

        # If the frozen circuit stays subthreshold, retain an analog readout of
        # final motor membrane. This keeps the benchmark about directional
        # evidence rather than only hard spike emission.
        evidence = evidence + torch.clamp(membrane[:, motor_start:motor_end], min=0.0)
        features = torch.cat([membrane, spike_counts], dim=1)
        return {
            "evidence": evidence,
            "final_membrane": membrane,
            "spike_counts": spike_counts,
            "features": features,
        }

    def _make_task(self, task_name: str, generator, device) -> _TaskBatch:
        builders: dict[str, Callable] = {
            "direction_copy": self._direction_copy,
            "anti_toxin": self._anti_toxin,
            "cue_switch": self._cue_switch,
            "delayed_recall": self._delayed_recall,
            "two_pulse_sum": self._two_pulse_sum,
        }
        try:
            return builders[task_name](generator, device)
        except KeyError as exc:
            raise ValueError(f"unknown frozen task: {task_name}") from exc

    def _empty_inputs(self, device):
        return torch.zeros(
            (self.config.sample_count, self.config.timesteps, self.config.sensor_channels),
            dtype=torch.float32,
            device=device,
        )

    def _balanced_classes(self, generator, device):
        labels = torch.arange(self.config.sample_count, device=device) % self.config.motor_channels
        order = torch.randperm(self.config.sample_count, generator=generator, device=device)
        return labels.index_select(0, order)

    def _direction_copy(self, generator, device) -> _TaskBatch:
        labels = self._balanced_classes(generator, device)
        inputs = self._empty_inputs(device)
        _write_direction(inputs, labels, channels_offset=0, amplitude=self.config.input_amplitude)
        return _TaskBatch(inputs, labels, "choose the active food-direction channel")

    def _anti_toxin(self, generator, device) -> _TaskBatch:
        hazard = self._balanced_classes(generator, device)
        targets = _opposite_direction(hazard)
        inputs = self._empty_inputs(device)
        _write_direction(inputs, hazard, channels_offset=4, amplitude=self.config.input_amplitude)
        return _TaskBatch(inputs, targets, "choose the opposite direction of the active toxin channel")

    def _cue_switch(self, generator, device) -> _TaskBatch:
        direction = self._balanced_classes(generator, device)
        cue = torch.randint(0, 2, (self.config.sample_count,), device=device, generator=generator)
        targets = torch.where(cue.bool(), _opposite_direction(direction), direction)
        inputs = self._empty_inputs(device)
        _write_direction(inputs, direction, channels_offset=0, amplitude=self.config.input_amplitude)
        # Channel 4 is treated as a binary context cue here.
        inputs[:, :, 4] = cue.to(inputs.dtype).unsqueeze(1) * self.config.input_amplitude
        return _TaskBatch(inputs, targets, "if cue channel is active choose opposite, else choose direction")

    def _delayed_recall(self, generator, device) -> _TaskBatch:
        labels = self._balanced_classes(generator, device)
        inputs = self._empty_inputs(device)
        _write_direction(inputs[:, :2, :], labels, channels_offset=0, amplitude=self.config.input_amplitude)
        return _TaskBatch(inputs, labels, "remember the initial food direction after blank delay")

    def _two_pulse_sum(self, generator, device) -> _TaskBatch:
        first = self._balanced_classes(generator, device)
        second = torch.randint(0, self.config.motor_channels, first.shape, device=device, generator=generator)
        targets = (first + second) % self.config.motor_channels
        inputs = self._empty_inputs(device)
        _write_direction(inputs[:, :2, :], first, channels_offset=0, amplitude=self.config.input_amplitude)
        _write_direction(inputs[:, -2:, :], second, channels_offset=0, amplitude=self.config.input_amplitude)
        return _TaskBatch(inputs, targets, "classify modular sum of first and final directional pulses")

    def _jsonable_config(self, device) -> dict:
        cfg = asdict(self.config)
        cfg["device"] = str(device)
        cfg["seed_edges"] = [asdict(edge) for edge in self.config.seed_edges]
        return cfg


class FrozenRepresentationProbeRunner:
    """Train tiny linear readouts on frozen AMMC trace features.

    The sparse AMMC substrate is run once per task and then held fixed. Only a
    linear classifier over final membrane and spike-count features is trained.
    """

    def __init__(self, config: FrozenProbeConfig | None = None) -> None:
        self.config = config or FrozenProbeConfig()
        if not 0.0 < self.config.train_fraction < 1.0:
            raise ValueError("train_fraction must be in (0, 1)")
        if self.config.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.config.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.config.task_config.sample_count < 4:
            raise ValueError("sample_count must be at least 4 for train/test probing")
        self.task_runner = FrozenTaskRunner(self.config.task_config)

    def run(self) -> FrozenProbeResult:
        _require_torch()
        task_cfg = self.config.task_config
        device = resolve_device(task_cfg.device)
        seed_everything(task_cfg.seed, device=device)
        split_generator = make_generator(task_cfg.seed + 10_000, device=device)

        summary: list[FrozenProbeSummaryRecord] = []
        for offset, task_name in enumerate(task_cfg.tasks):
            task_generator = make_generator(task_cfg.seed + offset + 1, device=device)
            batch = self.task_runner._make_task(task_name, task_generator, device)
            trace = self.task_runner._frozen_ammc_trace(batch.inputs, device)
            summary.append(
                self._probe_task(
                    task_name,
                    batch,
                    trace,
                    split_generator,
                    device,
                )
            )
            sync(device)

        return FrozenProbeResult(
            config=self._jsonable_config(device),
            summary=summary,
        )

    def save_outputs(
        self,
        result: FrozenProbeResult,
        output_dir: str | Path,
        *,
        plot: bool = True,
    ) -> dict[str, str]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        json_path = output / "frozen_representation_probe.json"
        json_path.write_text(json.dumps(result.to_json_dict(), indent=2) + "\n", encoding="utf-8")

        summary_csv = output / "frozen_representation_probe_summary.csv"
        _write_csv(summary_csv, [asdict(row) for row in result.summary])

        paths = {
            "json": str(json_path),
            "summary_csv": str(summary_csv),
        }
        if plot:
            try:
                plot_path = output / "frozen_representation_probe_summary.png"
                plot_frozen_probe_result(result, plot_path)
                paths["plot"] = str(plot_path)
            except Exception as exc:  # pragma: no cover - optional plotting
                paths["plot"] = f"skipped: {exc}"
        return paths

    def _probe_task(self, task_name: str, batch: _TaskBatch, trace: dict, split_generator, device) -> FrozenProbeSummaryRecord:
        task_cfg = self.config.task_config
        features = trace["features"].detach()
        targets = batch.targets
        evidence = trace["evidence"].detach()
        frozen_predictions = evidence.argmax(dim=1)
        frozen_accuracy = _accuracy(frozen_predictions, targets)
        inactive_rate = float((evidence.max(dim=1).values <= 1e-8).to(torch.float32).mean().item())

        order = torch.randperm(targets.numel(), device=device, generator=split_generator)
        train_count = max(1, int(round(targets.numel() * self.config.train_fraction)))
        train_count = min(train_count, targets.numel() - 1)
        train_idx = order[:train_count]
        test_idx = order[train_count:]

        x_train = features.index_select(0, train_idx)
        y_train = targets.index_select(0, train_idx)
        x_test = features.index_select(0, test_idx)
        y_test = targets.index_select(0, test_idx)
        if self.config.standardize_features:
            mean = x_train.mean(dim=0, keepdim=True)
            std = x_train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
            x_train = (x_train - mean) / std
            x_test = (x_test - mean) / std

        train_accuracy, test_accuracy, final_loss = self._train_linear_readout(x_train, y_train, x_test, y_test, device)

        random_predictions = torch.randint(
            0,
            task_cfg.motor_channels,
            y_test.shape,
            device=device,
            generator=split_generator,
        )
        instant_reflex = _instant_reflex_predictions(batch.inputs, task_cfg.motor_channels).index_select(0, test_idx)
        integrated_reflex = _integrated_reflex_predictions(batch.inputs, task_cfg.motor_channels).index_select(0, test_idx)
        best_reflex = max(_accuracy(instant_reflex, y_test), _accuracy(integrated_reflex, y_test))

        return FrozenProbeSummaryRecord(
            task=task_name,
            samples=int(targets.numel()),
            train_samples=int(y_train.numel()),
            test_samples=int(y_test.numel()),
            timesteps=task_cfg.timesteps,
            target_rule=batch.target_rule,
            feature_dim=int(features.shape[1]),
            frozen_ammc_accuracy=frozen_accuracy,
            linear_probe_accuracy=test_accuracy,
            linear_probe_train_accuracy=train_accuracy,
            random_accuracy=_accuracy(random_predictions, y_test),
            instant_reflex_accuracy=_accuracy(instant_reflex, y_test),
            integrated_reflex_accuracy=_accuracy(integrated_reflex, y_test),
            inactive_output_rate=inactive_rate,
            representation_gain_over_frozen=test_accuracy - frozen_accuracy,
            representation_gain_over_best_reflex=test_accuracy - best_reflex,
            final_probe_loss=final_loss,
        )

    def _train_linear_readout(self, x_train, y_train, x_test, y_test, device) -> tuple[float, float, float]:
        classifier = torch.nn.Linear(x_train.shape[1], self.config.task_config.motor_channels).to(device)
        optimizer = torch.optim.AdamW(
            classifier.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        final_loss = 0.0
        for _ in range(self.config.epochs):
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(x_train)
            loss = torch.nn.functional.cross_entropy(logits, y_train)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().item())
            mark_step(device)

        with torch.no_grad():
            train_predictions = classifier(x_train).argmax(dim=1)
            test_predictions = classifier(x_test).argmax(dim=1)
        return _accuracy(train_predictions, y_train), _accuracy(test_predictions, y_test), final_loss

    def _jsonable_config(self, device) -> dict:
        cfg = asdict(self.config)
        cfg["task_config"]["device"] = str(device)
        cfg["task_config"]["seed_edges"] = [asdict(edge) for edge in self.config.task_config.seed_edges]
        return cfg


def available_frozen_tasks() -> tuple[str, ...]:
    """Return available synthetic frozen task names."""

    return (
        "direction_copy",
        "anti_toxin",
        "cue_switch",
        "delayed_recall",
        "two_pulse_sum",
    )


def plot_frozen_task_result(result: FrozenTaskResult, output_path: str | Path) -> None:
    """Plot frozen AMMC accuracy against simple baselines."""

    import matplotlib.pyplot as plt  # type: ignore

    labels = [row.task for row in result.summary]
    x = list(range(len(labels)))
    width = 0.22
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))
    ax.bar([i - width for i in x], [row.frozen_ammc_accuracy for row in result.summary], width, label="Frozen AMMC")
    ax.bar([i for i in x], [row.instant_reflex_accuracy for row in result.summary], width, label="Instant reflex")
    ax.bar([i + width for i in x], [row.integrated_reflex_accuracy for row in result.summary], width, label="Integrated reflex")
    ax.axhline(0.25, color="gray", linestyle="--", linewidth=1, label="Random 4-way chance")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("AMMC Gen-5 frozen diversified tasks")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_frozen_probe_result(result: FrozenProbeResult, output_path: str | Path) -> None:
    """Plot linear probe accuracy against frozen and reflex baselines."""

    import matplotlib.pyplot as plt  # type: ignore

    labels = [row.task for row in result.summary]
    x = list(range(len(labels)))
    width = 0.18
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.6), 5))
    ax.bar([i - 1.5 * width for i in x], [row.frozen_ammc_accuracy for row in result.summary], width, label="Frozen motor")
    ax.bar([i - 0.5 * width for i in x], [row.linear_probe_accuracy for row in result.summary], width, label="Linear probe")
    ax.bar([i + 0.5 * width for i in x], [row.instant_reflex_accuracy for row in result.summary], width, label="Instant reflex")
    ax.bar([i + 1.5 * width for i in x], [row.integrated_reflex_accuracy for row in result.summary], width, label="Integrated reflex")
    ax.axhline(0.25, color="gray", linestyle="--", linewidth=1, label="Random 4-way chance")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("AMMC Gen-5 frozen representation probe")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_direction(inputs, labels, *, channels_offset: int, amplitude: float) -> None:
    channel = labels + channels_offset
    if inputs.dim() != 3:
        raise ValueError("inputs must be [batch, time, channels]")
    batch = inputs.shape[0]
    time = inputs.shape[1]
    batch_index = torch.arange(batch, device=inputs.device).unsqueeze(1).expand(batch, time)
    time_index = torch.arange(time, device=inputs.device).unsqueeze(0).expand(batch, time)
    channel_index = channel.unsqueeze(1).expand(batch, time)
    inputs[batch_index, time_index, channel_index] = amplitude


def _opposite_direction(labels):
    return (labels + 2) % 4


def _accuracy(predictions, targets) -> float:
    return float((predictions == targets).to(torch.float32).mean().item())


def _instant_reflex_predictions(inputs, motor_channels: int):
    final_food = inputs[:, -1, :motor_channels]
    return final_food.argmax(dim=1)


def _integrated_reflex_predictions(inputs, motor_channels: int):
    integrated_food = inputs[:, :, :motor_channels].sum(dim=1)
    return integrated_food.argmax(dim=1)
