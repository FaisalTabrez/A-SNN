"""Phase 24 streaming row-sequential MNIST benchmark.

Static MNIST did not require recurrent dynamics. This benchmark presents one
image row per simulation step and exposes only final hidden state to the sparse
readout, creating a genuine memory requirement while retaining raw and
order-insensitive controls.
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
    _fit_and_measure,
    _matched_raw_hidden_units,
    build_event_reservoir_edges,
    load_mnist_tensors,
    nn,
    torch,
)
from .runtime import device_kind, mark_step, resolve_device, seed_everything, sync


SEQUENTIAL_MNIST_FEATURES = (
    "raw_flattened",
    "last_row",
    "integrated_rows",
    "hidden_feedforward_final",
    "hidden_recurrent_final",
)


class StreamingMNISTReservoir(nn.Module):
    """Sparse hidden reservoir receiving one downsampled image row per step."""

    def __init__(
        self,
        config: EventMNISTConfig,
        *,
        seed: int,
        recurrent: bool,
        device=None,
    ) -> None:
        if torch is None:
            raise ImportError("Phase 24 sequential MNIST requires PyTorch")
        super().__init__()
        self.config = config
        self.input_neurons = config.image_size
        self.hidden_neurons = config.hidden_neurons
        self.neuron_count = self.input_neurons + self.hidden_neurons
        self.recurrent = bool(recurrent)
        edges = build_event_reservoir_edges(
            self.input_neurons,
            self.hidden_neurons,
            sensor_fanout=config.sensor_fanout,
            recurrent_fanout=config.recurrent_fanout,
            seed=seed,
        )
        if len(edges) > config.max_edges:
            raise ValueError(f"streaming reservoir requires {len(edges)} edges")
        self.graph = DynamicSparseLinear(
            self.neuron_count,
            self.neuron_count,
            config.max_edges,
            device=device,
        )
        self.graph.load_edges(edges)
        if not recurrent:
            self._disable_recurrence()
        self.graph.requires_grad_(False)

    @property
    def active_edge_count(self) -> int:
        return self.graph.active_edge_count

    @property
    def feature_dim(self) -> int:
        return self.hidden_neurons * 2

    def forward(self, pixels):  # type: ignore[override]
        if pixels.ndim != 2 or pixels.shape[1] != self.config.image_size**2:
            raise ValueError("pixels must have shape [batch, image_size ** 2]")
        frames = pixels.reshape(pixels.shape[0], self.config.image_size, self.config.image_size)
        hidden_membrane = pixels.new_zeros((pixels.shape[0], self.hidden_neurons))
        hidden_spikes = torch.zeros_like(hidden_membrane)
        hidden_counts = torch.zeros_like(hidden_membrane)
        for step in range(self.config.image_size):
            network_spikes = pixels.new_zeros((pixels.shape[0], self.neuron_count))
            # The transducer emits graded row events. Hidden neurons remain
            # hard-threshold LIF units; recurrence uses the previous hidden
            # spike state in the same sparse edge graph.
            network_spikes[:, : self.input_neurons] = (
                frames[:, step, :] * self.config.input_gain
            )
            network_spikes[:, self.input_neurons :] = hidden_spikes
            current = self.graph(network_spikes)[:, self.input_neurons :]
            pre_reset = hidden_membrane * self.config.reservoir_leak + current
            hidden_spikes = (pre_reset >= self.config.reservoir_threshold).to(pre_reset.dtype)
            hidden_membrane = pre_reset - hidden_spikes * self.config.reservoir_threshold
            hidden_counts = hidden_counts + hidden_spikes
        # Causal readouts receive only state available after the final row.
        # Cumulative spike counts are retained solely for the activity
        # diagnostic; exposing them would create an external memory bypass.
        features = torch.cat((hidden_spikes, hidden_membrane), dim=1)
        event_rate = hidden_counts.mean() / self.config.image_size
        return features, event_rate

    def _disable_recurrence(self) -> None:
        recurrent = self.graph.active_mask & (self.graph.sources >= self.input_neurons)
        with torch.no_grad():
            self.graph.active_mask[recurrent] = False
            self.graph.short_term_weight[recurrent] = 0.0
            self.graph.long_term_weight[recurrent] = 0.0


@dataclass
class SequentialMNISTResult:
    config: EventMNISTConfig
    device: str
    input_edges: int
    recurrent_edges: int
    records: list[dict]
    summary: list[dict]

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "sequential_mnist.json"
        records_path = output / "sequential_mnist_records.csv"
        summary_path = output / "sequential_mnist_summary.csv"
        payload = {
            "config": asdict(self.config),
            "device": self.device,
            "input_edges": self.input_edges,
            "recurrent_edges": self.recurrent_edges,
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
            plot_path = output / "sequential_mnist_summary.png"
            plot_sequential_mnist(self.summary, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def run_sequential_mnist(
    config: EventMNISTConfig,
    *,
    device="auto",
) -> SequentialMNISTResult:
    """Evaluate final-state memory with and without hidden recurrence."""

    if torch is None:
        raise ImportError("Phase 24 sequential MNIST requires PyTorch")
    _validate(config)
    resolved = resolve_device(device)
    train_pixels, train_labels, test_pixels, test_labels = load_mnist_tensors(config)
    records: list[dict] = []
    input_edges = config.image_size * config.sensor_fanout
    recurrent_edges = config.hidden_neurons * config.recurrent_fanout
    target_dim = config.hidden_neurons * 2

    train_last = train_pixels.reshape(-1, config.image_size, config.image_size)[:, -1, :]
    test_last = test_pixels.reshape(-1, config.image_size, config.image_size)[:, -1, :]
    train_integrated = train_pixels.reshape(-1, config.image_size, config.image_size).mean(dim=1)
    test_integrated = test_pixels.reshape(-1, config.image_size, config.image_size).mean(dim=1)

    for seed in config.seeds:
        seed_everything(seed, device=resolved)
        feedforward = StreamingMNISTReservoir(
            config, seed=seed, recurrent=False, device=resolved
        ).to(resolved)
        recurrent = StreamingMNISTReservoir(
            config, seed=seed, recurrent=True, device=resolved
        ).to(resolved)
        feedforward.eval()
        recurrent.eval()

        feedforward_start = time.perf_counter()
        train_feedforward, feedforward_train_rate = _extract_streaming_features(
            feedforward, train_pixels, config.batch_size, resolved
        )
        test_feedforward, _ = _extract_streaming_features(
            feedforward, test_pixels, config.batch_size, resolved
        )
        sync(resolved)
        feedforward_seconds = time.perf_counter() - feedforward_start

        recurrent_start = time.perf_counter()
        train_recurrent, recurrent_train_rate = _extract_streaming_features(
            recurrent, train_pixels, config.batch_size, resolved
        )
        test_recurrent, _ = _extract_streaming_features(
            recurrent, test_pixels, config.batch_size, resolved
        )
        sync(resolved)
        recurrent_seconds = time.perf_counter() - recurrent_start

        feature_sets = {
            "raw_flattened": (train_pixels, test_pixels, 0.0, 0, 0.0, "raw"),
            "last_row": (train_last, test_last, 0.0, 0, 0.0, "last_row"),
            "integrated_rows": (
                train_integrated,
                test_integrated,
                0.0,
                0,
                0.0,
                "orderless",
            ),
            "hidden_feedforward_final": (
                train_feedforward,
                test_feedforward,
                feedforward_seconds,
                feedforward.active_edge_count,
                feedforward_train_rate,
                "feedforward",
            ),
            "hidden_recurrent_final": (
                train_recurrent,
                test_recurrent,
                recurrent_seconds,
                recurrent.active_edge_count,
                recurrent_train_rate,
                "recurrent",
            ),
        }

        for feature in SEQUENTIAL_MNIST_FEATURES:
            train_features, test_features, feature_seconds, active_edges, event_rate, topology = (
                feature_sets[feature]
            )
            for classifier_index, classifier in enumerate(("linear", "mlp")):
                hidden_units = 0
                if classifier == "mlp":
                    hidden_units = _matched_raw_hidden_units(
                        train_features.shape[1], target_dim, config.readout_hidden_units
                    )
                record = _fit_and_measure(
                    f"{feature}_{classifier}",
                    classifier,
                    train_features,
                    train_labels,
                    test_features,
                    test_labels,
                    config,
                    seed,
                    resolved,
                    feature_seconds,
                    active_edges,
                    hidden_units,
                    model_seed_offset=classifier_index,
                    hidden_spike_rate=event_rate,
                )
                record["feature"] = feature
                record["classifier"] = classifier
                record["topology"] = topology
                records.append(record)

    _attach_sequence_deltas(records)
    return SequentialMNISTResult(
        config=config,
        device=device_kind(resolved),
        input_edges=input_edges,
        recurrent_edges=recurrent_edges,
        records=records,
        summary=summarize_sequential_mnist(records),
    )


def summarize_sequential_mnist(records: Iterable[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in records:
        grouped.setdefault((str(row["feature"]), str(row["classifier"])), []).append(row)
    summary: list[dict] = []
    for feature in SEQUENTIAL_MNIST_FEATURES:
        for classifier in ("linear", "mlp"):
            rows = grouped.get((feature, classifier), [])
            if not rows:
                continue
            accuracy = [float(row["test_accuracy"]) for row in rows]
            recurrence_gains = [float(row["recurrence_gain"]) for row in rows]
            summary.append(
                {
                    "feature": feature,
                    "classifier": classifier,
                    "topology": str(rows[0]["topology"]),
                    "seeds": len(rows),
                    "mean_test_accuracy": statistics.fmean(accuracy),
                    "std_test_accuracy": statistics.pstdev(accuracy),
                    "mean_gain_vs_last_row": statistics.fmean(
                        float(row["accuracy_gain_vs_last_row"]) for row in rows
                    ),
                    "mean_gain_vs_integrated_rows": statistics.fmean(
                        float(row["accuracy_gain_vs_integrated_rows"]) for row in rows
                    ),
                    "mean_recurrence_gain": statistics.fmean(recurrence_gains),
                    "recurrence_improved_seed_count": sum(gain > 0 for gain in recurrence_gains),
                    "active_edges": int(rows[0]["frozen_active_edges"]),
                    "feature_dim": int(rows[0]["feature_dim"]),
                    "trainable_parameters": int(rows[0]["trainable_parameters"]),
                    "mean_hidden_event_rate": statistics.fmean(
                        float(row["mean_hidden_spike_rate"]) for row in rows
                    ),
                    "mean_feature_seconds": statistics.fmean(
                        float(row["feature_seconds"]) for row in rows
                    ),
                    "mean_feature_examples_per_second": statistics.fmean(
                        float(row["feature_examples_per_second"]) for row in rows
                    ),
                    "mean_train_seconds": statistics.fmean(
                        float(row["train_seconds"]) for row in rows
                    ),
                }
            )
    return summary


def plot_sequential_mnist(summary: list[dict], path: str | pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    lookup = {(row["feature"], row["classifier"]): row for row in summary}
    features = [feature for feature in SEQUENTIAL_MNIST_FEATURES if (feature, "linear") in lookup]
    positions = list(range(len(features)))
    width = 0.38
    figure, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)
    for offset, classifier, color in ((-width / 2, "linear", "#35b4f2"), (width / 2, "mlp", "#ffb31a")):
        axes[0].bar(
            [position + offset for position in positions],
            [100.0 * float(lookup[(feature, classifier)]["mean_test_accuracy"]) for feature in features],
            width,
            label=classifier,
            color=color,
        )
        axes[1].bar(
            [position + offset for position in positions],
            [100.0 * float(lookup[(feature, classifier)]["mean_recurrence_gain"]) for feature in features],
            width,
            label=classifier,
            color=color,
        )
    labels = [feature.replace("_", "\n") for feature in features]
    axes[0].set_ylabel("Engineering-validation accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_title("AMMC Gen-5 Phase 24: Streaming Row-Sequential MNIST")
    axes[0].set_xticks(positions, labels)
    axes[0].legend()
    axes[1].axhline(0.0, color="#222222", linewidth=1)
    axes[1].set_ylabel("Recurrent minus feedforward accuracy (points)")
    axes[1].set_xticks(positions, labels)
    axes[1].legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _extract_streaming_features(reservoir, pixels, batch_size, device):
    chunks = []
    weighted_rate = 0.0
    with torch.no_grad():
        for start in range(0, pixels.shape[0], batch_size):
            batch = pixels[start : start + batch_size].to(device)
            features, event_rate = reservoir(batch)
            chunks.append(features.cpu())
            weighted_rate += float(event_rate.item()) * batch.shape[0]
            mark_step(device)
    sync(device)
    return torch.cat(chunks, dim=0), weighted_rate / pixels.shape[0]


def _attach_sequence_deltas(records: list[dict]) -> None:
    lookup = {
        (int(row["seed"]), str(row["classifier"]), str(row["feature"])): float(
            row["test_accuracy"]
        )
        for row in records
    }
    for row in records:
        seed = int(row["seed"])
        classifier = str(row["classifier"])
        accuracy = float(row["test_accuracy"])
        last_row = lookup[(seed, classifier, "last_row")]
        integrated = lookup[(seed, classifier, "integrated_rows")]
        row["accuracy_gain_vs_last_row"] = accuracy - last_row
        row["accuracy_gain_vs_integrated_rows"] = accuracy - integrated
        if row["feature"] == "hidden_recurrent_final":
            baseline = lookup[(seed, classifier, "hidden_feedforward_final")]
            row["paired_feedforward_test_accuracy"] = baseline
            row["recurrence_gain"] = accuracy - baseline
        else:
            row["paired_feedforward_test_accuracy"] = accuracy
            row["recurrence_gain"] = 0.0


def _validate(config: EventMNISTConfig) -> None:
    if not config.seeds or config.train_samples <= 0 or config.test_samples <= 0:
        raise ValueError("seeds and positive sample counts are required")
    if config.image_size < 2 or config.epochs <= 0 or config.batch_size <= 0:
        raise ValueError("image size >= 2 and positive epochs/batch size are required")
    required = config.image_size * config.sensor_fanout + config.hidden_neurons * config.recurrent_fanout
    if required > config.max_edges:
        raise ValueError(f"streaming topology requires {required} edges but max_edges is {config.max_edges}")


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
