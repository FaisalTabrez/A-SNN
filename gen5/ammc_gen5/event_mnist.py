"""Frozen sparse-reservoir benchmark for event-coded MNIST.

Phase 18 is deliberately an external representation test, not an end-to-end
learning claim. MNIST pixels are converted to latency-coded events, propagated
through a frozen :class:`DynamicSparseLinear` reservoir, and decoded by small
trainable heads. Raw-pixel linear and MLP classifiers provide matched controls.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import pathlib
import random
import statistics
import time
from types import SimpleNamespace
from typing import Iterable

try:  # pragma: no cover - exercised in PyTorch runtimes
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - keeps import/contract checks available
    torch = None

    class _MissingModule:
        pass

    nn = SimpleNamespace(Module=_MissingModule)

from .dynamic_sparse import DynamicSparseLinear, EdgeRecord
from .runtime import device_kind, mark_step, resolve_device, seed_everything, sync


MODEL_NAMES = (
    "raw_pixel_linear",
    "raw_pixel_mlp",
    "frozen_ammc_linear",
    "frozen_ammc_mlp",
)


def _require_torch() -> None:
    if torch is None:
        raise ImportError("Phase 18 event-coded MNIST requires PyTorch")


@dataclass(frozen=True)
class EventMNISTConfig:
    """Configuration for the frozen event-coded MNIST benchmark."""

    seeds: tuple[int, ...] = (42, 43, 44)
    train_samples: int = 20_000
    test_samples: int = 5_000
    image_size: int = 8
    timesteps: int = 8
    event_threshold: float = 0.05
    hidden_neurons: int = 64
    sensor_fanout: int = 2
    recurrent_fanout: int = 4
    max_edges: int = 512
    reservoir_leak: float = 0.85
    reservoir_threshold: float = 1.0
    input_gain: float = 1.25
    readout_hidden_units: int = 128
    epochs: int = 15
    learning_rate: float = 0.003
    weight_decay: float = 0.0001
    batch_size: int = 512
    data_seed: int = 2026
    data_root: str = "gen5_data"
    download: bool = True

    @property
    def sensor_neurons(self) -> int:
        return self.image_size * self.image_size

    @property
    def neuron_count(self) -> int:
        return self.sensor_neurons + self.hidden_neurons


@dataclass
class EventMNISTResult:
    """Serializable records and aggregates from a Phase 18 run."""

    config: EventMNISTConfig
    device: str
    active_edges: int
    records: list[dict]
    summary: list[dict]

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "event_mnist.json"
        records_path = output / "event_mnist_records.csv"
        summary_path = output / "event_mnist_summary.csv"
        payload = {
            "config": asdict(self.config),
            "device": self.device,
            "active_edges": self.active_edges,
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
            plot_path = output / "event_mnist_summary.png"
            plot_event_mnist_result(self.summary, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def build_event_reservoir_edges(
    sensor_neurons: int,
    hidden_neurons: int,
    *,
    sensor_fanout: int = 2,
    recurrent_fanout: int = 4,
    seed: int = 42,
) -> list[EdgeRecord]:
    """Build a reproducible sparse sensor/recurrent reservoir topology.

    Every sensor projects into a small sample of hidden neurons. Every hidden
    neuron projects to distinct hidden peers. Approximately 20% of edges are
    inhibitory, matching the broad excitatory/inhibitory balance used in the
    rest of the AMMC prototype.
    """

    if sensor_neurons <= 0 or hidden_neurons <= 1:
        raise ValueError("sensor_neurons must be positive and hidden_neurons must exceed one")
    if not 1 <= sensor_fanout <= hidden_neurons:
        raise ValueError("sensor_fanout must be in [1, hidden_neurons]")
    if not 1 <= recurrent_fanout < hidden_neurons:
        raise ValueError("recurrent_fanout must be in [1, hidden_neurons - 1]")

    rng = random.Random(int(seed))
    hidden_ids = list(range(sensor_neurons, sensor_neurons + hidden_neurons))
    edges: list[EdgeRecord] = []

    def edge(source: int, target: int, *, sensor_edge: bool) -> EdgeRecord:
        sign = 1.0 if sensor_edge or rng.random() >= 0.2 else -1.0
        weight = rng.uniform(0.55, 0.95) if sensor_edge else rng.uniform(0.08, 0.30)
        return EdgeRecord(
            source=source,
            target=target,
            short_term_weight=0.0,
            long_term_weight=weight,
            sign=sign,
            delay_steps=0,
        )

    for source in range(sensor_neurons):
        for target in rng.sample(hidden_ids, sensor_fanout):
            edges.append(edge(source, target, sensor_edge=True))
    for source in hidden_ids:
        candidates = [target for target in hidden_ids if target != source]
        for target in rng.sample(candidates, recurrent_fanout):
            edges.append(edge(source, target, sensor_edge=False))
    return edges


def latency_encode(pixels, timesteps: int, event_threshold: float = 0.05):
    """Convert normalized pixels ``[batch, pixels]`` into one-spike latency events.

    Bright pixels fire earlier. Pixels at or below ``event_threshold`` remain
    silent. The result has shape ``[timesteps, batch, pixels]``.
    """

    _require_torch()
    if pixels.ndim != 2:
        raise ValueError("pixels must have shape [batch, pixels]")
    if timesteps < 2:
        raise ValueError("timesteps must be at least two")
    if not 0 <= event_threshold < 1:
        raise ValueError("event_threshold must be in [0, 1)")
    clipped = pixels.clamp(0.0, 1.0)
    slots = torch.round((1.0 - clipped) * (timesteps - 1)).to(torch.long)
    active = clipped > event_threshold
    encoded = pixels.new_zeros((timesteps, pixels.shape[0], pixels.shape[1]))
    encoded.scatter_(0, slots.unsqueeze(0), active.to(pixels.dtype).unsqueeze(0))
    return encoded


class FrozenEventReservoir(nn.Module):
    """Frozen sparse LIF reservoir used as the AMMC representation substrate."""

    def __init__(self, config: EventMNISTConfig, *, seed: int, device=None) -> None:
        _require_torch()
        super().__init__()
        self.config = config
        self.device = resolve_device(device or "auto")
        edges = build_event_reservoir_edges(
            config.sensor_neurons,
            config.hidden_neurons,
            sensor_fanout=config.sensor_fanout,
            recurrent_fanout=config.recurrent_fanout,
            seed=seed,
        )
        if len(edges) > config.max_edges:
            raise ValueError(f"reservoir requires {len(edges)} edges, capacity is {config.max_edges}")
        self.graph = DynamicSparseLinear(
            config.neuron_count,
            config.neuron_count,
            config.max_edges,
            device=self.device,
        )
        self.graph.load_edges(edges)
        self.graph.requires_grad_(False)

    @property
    def feature_dim(self) -> int:
        return self.config.neuron_count * 2

    @property
    def active_edge_count(self) -> int:
        return self.graph.active_edge_count

    def forward(self, pixels):  # type: ignore[override]
        events = latency_encode(pixels, self.config.timesteps, self.config.event_threshold)
        batch = pixels.shape[0]
        membrane = pixels.new_zeros((batch, self.config.neuron_count))
        spikes = torch.zeros_like(membrane)
        spike_counts = torch.zeros_like(membrane)
        for step in range(self.config.timesteps):
            injected = torch.zeros_like(membrane)
            injected[:, : self.config.sensor_neurons] = events[step] * self.config.input_gain
            membrane = membrane * self.config.reservoir_leak + injected + self.graph(spikes)
            spikes = (membrane >= self.config.reservoir_threshold).to(membrane.dtype)
            membrane = membrane - spikes * self.config.reservoir_threshold
            spike_counts = spike_counts + spikes
        return torch.cat((spike_counts / self.config.timesteps, membrane), dim=1)


class _Classifier(nn.Module):
    def __init__(self, in_features: int, *, kind: str, hidden_units: int) -> None:
        _require_torch()
        super().__init__()
        if kind == "linear":
            self.network = nn.Linear(in_features, 10)
        elif kind == "mlp":
            self.network = nn.Sequential(
                nn.Linear(in_features, hidden_units),
                nn.ReLU(),
                nn.Linear(hidden_units, 10),
            )
        else:
            raise ValueError(f"unknown classifier kind: {kind}")

    def forward(self, features):  # type: ignore[override]
        return self.network(features)


def run_event_mnist(config: EventMNISTConfig, *, device="auto") -> EventMNISTResult:
    """Run all raw-pixel and frozen-AMMC MNIST comparisons."""

    _require_torch()
    _validate_config(config)
    resolved = resolve_device(device)
    train_pixels, train_labels, test_pixels, test_labels = load_mnist_tensors(config)
    records: list[dict] = []
    expected_edges = config.sensor_neurons * config.sensor_fanout + config.hidden_neurons * config.recurrent_fanout

    for seed in config.seeds:
        seed_everything(seed, device=resolved)
        reservoir = FrozenEventReservoir(config, seed=seed, device=resolved)
        reservoir.eval()
        feature_start = time.perf_counter()
        train_reservoir = _extract_reservoir_features(reservoir, train_pixels, config.batch_size, resolved)
        test_reservoir = _extract_reservoir_features(reservoir, test_pixels, config.batch_size, resolved)
        sync(resolved)
        feature_seconds = time.perf_counter() - feature_start

        matched_raw_hidden = _matched_raw_hidden_units(
            config.sensor_neurons,
            reservoir.feature_dim,
            config.readout_hidden_units,
        )
        feature_sets = {
            "raw_pixel_linear": (train_pixels, test_pixels, "linear", 0.0, 0),
            "raw_pixel_mlp": (train_pixels, test_pixels, "mlp", 0.0, matched_raw_hidden),
            "frozen_ammc_linear": (train_reservoir, test_reservoir, "linear", feature_seconds, 0),
            "frozen_ammc_mlp": (
                train_reservoir,
                test_reservoir,
                "mlp",
                feature_seconds,
                config.readout_hidden_units,
            ),
        }
        for model_name in MODEL_NAMES:
            train_features, test_features, kind, extraction_seconds, hidden_units = feature_sets[model_name]
            record = _fit_and_measure(
                model_name,
                kind,
                train_features,
                train_labels,
                test_features,
                test_labels,
                config,
                seed,
                resolved,
                extraction_seconds,
                reservoir.active_edge_count if model_name.startswith("frozen_") else 0,
                hidden_units,
            )
            records.append(record)

    return EventMNISTResult(
        config=config,
        device=device_kind(resolved),
        active_edges=expected_edges,
        records=records,
        summary=summarize_event_mnist_records(records),
    )


def load_mnist_tensors(config: EventMNISTConfig):
    """Load deterministic subsets from the official torchvision MNIST splits."""

    _require_torch()
    try:
        from torch.utils.data import DataLoader, Subset
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover - dependency specific
        raise ImportError(
            "Phase 18 requires torchvision. Install a torchvision build that "
            "matches the active PyTorch version before running."
        ) from exc

    transform = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size), antialias=True),
            transforms.ToTensor(),
        ]
    )
    root = pathlib.Path(config.data_root)
    train_dataset = datasets.MNIST(root=str(root), train=True, download=config.download, transform=transform)
    test_dataset = datasets.MNIST(root=str(root), train=False, download=config.download, transform=transform)
    train_subset = _deterministic_subset(train_dataset, config.train_samples, config.data_seed, Subset)
    test_subset = _deterministic_subset(test_dataset, config.test_samples, config.data_seed + 1, Subset)
    train_pixels, train_labels = _materialize_dataset(train_subset, config.batch_size, DataLoader)
    test_pixels, test_labels = _materialize_dataset(test_subset, config.batch_size, DataLoader)
    return train_pixels, train_labels, test_pixels, test_labels


def summarize_event_mnist_records(records: Iterable[dict]) -> list[dict]:
    """Aggregate seed-level records into mean/std model summaries."""

    grouped: dict[str, list[dict]] = {}
    for row in records:
        grouped.setdefault(str(row["model"]), []).append(row)
    summary: list[dict] = []
    for model in MODEL_NAMES:
        rows = grouped.get(model, [])
        if not rows:
            continue
        accuracy = [float(row["test_accuracy"]) for row in rows]
        summary.append(
            {
                "model": model,
                "seeds": len(rows),
                "mean_test_accuracy": statistics.fmean(accuracy),
                "std_test_accuracy": statistics.pstdev(accuracy),
                "mean_train_accuracy": statistics.fmean(float(row["train_accuracy"]) for row in rows),
                "feature_dim": int(rows[0]["feature_dim"]),
                "trainable_parameters": int(rows[0]["trainable_parameters"]),
                "classifier_hidden_units": int(rows[0]["classifier_hidden_units"]),
                "frozen_active_edges": int(rows[0]["frozen_active_edges"]),
                "mean_hidden_spike_rate": statistics.fmean(float(row["mean_hidden_spike_rate"]) for row in rows),
                "mean_feature_seconds": statistics.fmean(float(row["feature_seconds"]) for row in rows),
                "mean_train_seconds": statistics.fmean(float(row["train_seconds"]) for row in rows),
                "mean_inference_examples_per_second": statistics.fmean(
                    float(row["inference_examples_per_second"]) for row in rows
                ),
            }
        )
    return summary


def plot_event_mnist_result(summary: list[dict], path: str | pathlib.Path) -> None:
    """Plot accuracy and trainable-parameter comparisons."""

    import matplotlib.pyplot as plt

    labels = [row["model"].replace("_", "\n") for row in summary]
    accuracy = [100.0 * float(row["mean_test_accuracy"]) for row in summary]
    errors = [100.0 * float(row["std_test_accuracy"]) for row in summary]
    parameters = [int(row["trainable_parameters"]) for row in summary]
    figure, axes = plt.subplots(2, 1, figsize=(11, 9), constrained_layout=True)
    axes[0].bar(labels, accuracy, yerr=errors, capsize=5, color="#35b4f2")
    axes[0].axhline(10.0, color="#d84a4a", linestyle="--", linewidth=1, label="chance")
    axes[0].set_ylabel("Official test accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[0].legend()
    axes[1].bar(labels, parameters, color="#ffb31a")
    axes[1].set_ylabel("Trainable parameters")
    axes[1].set_yscale("log")
    figure.suptitle("AMMC Gen-5 Phase 18: Frozen Event-Coded MNIST")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _validate_config(config: EventMNISTConfig) -> None:
    if not config.seeds:
        raise ValueError("at least one seed is required")
    if config.train_samples <= 0 or config.test_samples <= 0:
        raise ValueError("train_samples and test_samples must be positive")
    if config.image_size <= 0 or config.timesteps < 2 or config.hidden_neurons <= 1:
        raise ValueError("image_size must be positive, timesteps >= 2, and hidden_neurons > 1")
    required = config.sensor_neurons * config.sensor_fanout + config.hidden_neurons * config.recurrent_fanout
    if required > config.max_edges:
        raise ValueError(f"topology requires {required} edges but max_edges is {config.max_edges}")
    if config.epochs <= 0 or config.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")


def _deterministic_subset(dataset, requested: int, seed: int, subset_type):
    if requested > len(dataset):
        raise ValueError(f"requested {requested} samples from a split containing {len(dataset)}")
    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(len(dataset), generator=generator)[:requested].tolist()
    return subset_type(dataset, indices)


def _materialize_dataset(dataset, batch_size: int, loader_type):
    pixels: list = []
    labels: list = []
    for images, target in loader_type(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        pixels.append(images.flatten(1).to(torch.float32))
        labels.append(target.to(torch.long))
    return torch.cat(pixels, dim=0), torch.cat(labels, dim=0)


def _extract_reservoir_features(reservoir, pixels, batch_size: int, device):
    outputs = []
    with torch.no_grad():
        for start in range(0, pixels.shape[0], batch_size):
            batch = pixels[start : start + batch_size].to(device)
            outputs.append(reservoir(batch).cpu())
            mark_step(device)
    sync(device)
    return torch.cat(outputs, dim=0)


def _fit_and_measure(
    model_name,
    kind,
    train_features,
    train_labels,
    test_features,
    test_labels,
    config,
    seed,
    device,
    feature_seconds,
    frozen_active_edges,
    hidden_units,
):
    seed_everything(seed + MODEL_NAMES.index(model_name) * 10_000, device=device)
    model = _Classifier(
        train_features.shape[1],
        kind=kind,
        hidden_units=hidden_units or config.readout_hidden_units,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    train_start = time.perf_counter()
    model.train()
    for epoch in range(config.epochs):
        generator = torch.Generator().manual_seed(seed * 1000 + epoch)
        order = torch.randperm(train_features.shape[0], generator=generator)
        for start in range(0, order.numel(), config.batch_size):
            index = order[start : start + config.batch_size]
            features = train_features.index_select(0, index).to(device)
            labels = train_labels.index_select(0, index).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            mark_step(device)
    sync(device)
    train_seconds = time.perf_counter() - train_start
    train_accuracy, _ = _measure_accuracy(model, train_features, train_labels, config.batch_size, device)
    test_accuracy, inference_seconds = _measure_accuracy(model, test_features, test_labels, config.batch_size, device)
    parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    hidden_spike_rate = 0.0
    if model_name.startswith("frozen_"):
        hidden_spike_rate = float(
            train_features[:, config.sensor_neurons : config.neuron_count].mean().item()
        )
    return {
        "seed": int(seed),
        "model": model_name,
        "train_samples": int(train_features.shape[0]),
        "test_samples": int(test_features.shape[0]),
        "feature_dim": int(train_features.shape[1]),
        "trainable_parameters": int(parameters),
        "classifier_hidden_units": int(hidden_units if kind == "mlp" else 0),
        "frozen_active_edges": int(frozen_active_edges),
        "mean_hidden_spike_rate": hidden_spike_rate,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "feature_seconds": float(feature_seconds),
        "train_seconds": float(train_seconds),
        "inference_seconds": float(inference_seconds),
        "inference_examples_per_second": float(test_features.shape[0] / max(inference_seconds, 1e-12)),
    }


def _matched_raw_hidden_units(raw_dim: int, reservoir_dim: int, reservoir_hidden_units: int) -> int:
    """Choose a raw-input MLP width with approximately equal trainable parameters."""

    target_parameters = (reservoir_dim + 1) * reservoir_hidden_units + (reservoir_hidden_units + 1) * 10
    per_hidden = raw_dim + 1 + 10
    return max(1, round((target_parameters - 10) / per_hidden))


def _measure_accuracy(model, features, labels, batch_size: int, device):
    model.eval()
    correct = 0
    start_time = time.perf_counter()
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            logits = model(features[start : start + batch_size].to(device))
            prediction = logits.argmax(dim=1).cpu()
            correct += int((prediction == labels[start : start + batch_size]).sum().item())
            mark_step(device)
    sync(device)
    seconds = time.perf_counter() - start_time
    return correct / features.shape[0], seconds


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
