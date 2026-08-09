"""Phase 23 causal recurrence ablation for temporal-state MNIST.

Phase 22 stabilized LTW optimization but found no practical accuracy gain. The
next unresolved question is whether recurrent edges causally contribute to the
frozen temporal representation, or whether sensor-to-hidden random expansion
accounts for the useful features.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import pathlib
import statistics
import time
from typing import Iterable

from .event_mnist import (
    EventMNISTConfig,
    FrozenEventReservoir,
    _fit_and_measure,
    _matched_raw_hidden_units,
    load_mnist_tensors,
    torch,
)
from .runtime import device_kind, resolve_device, seed_everything, sync
from .temporal_mnist import _extract_temporal


RECURRENCE_ABLATION_FEATURES = (
    "raw_intensity",
    "sensor_temporal",
    "hidden_feedforward_temporal",
    "full_feedforward_temporal",
    "hidden_recurrent_temporal",
    "full_recurrent_temporal",
)


@dataclass
class RecurrenceAblationResult:
    config: EventMNISTConfig
    device: str
    feedforward_edges: int
    recurrent_edges: int
    records: list[dict]
    summary: list[dict]

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "recurrence_ablation.json"
        records_path = output / "recurrence_ablation_records.csv"
        summary_path = output / "recurrence_ablation_summary.csv"
        payload = {
            "config": asdict(self.config),
            "device": self.device,
            "feedforward_edges": self.feedforward_edges,
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
            plot_path = output / "recurrence_ablation_summary.png"
            plot_recurrence_ablation(self.summary, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def run_recurrence_ablation(
    config: EventMNISTConfig,
    *,
    device="auto",
) -> RecurrenceAblationResult:
    """Compare sensor-only, feedforward expansion, and recurrent expansion."""

    if torch is None:
        raise ImportError("Phase 23 recurrence ablation requires PyTorch")
    _validate(config)
    resolved = resolve_device(device)
    train_pixels, train_labels, test_pixels, test_labels = load_mnist_tensors(config)
    records: list[dict] = []
    feedforward_edge_count = config.sensor_neurons * config.sensor_fanout
    recurrent_edge_count = config.hidden_neurons * config.recurrent_fanout
    target_dim = config.neuron_count * 2

    for seed in config.seeds:
        seed_everything(seed, device=resolved)
        recurrent = FrozenEventReservoir(config, seed=seed, device=resolved)
        feedforward = FrozenEventReservoir(config, seed=seed, device=resolved)
        disabled = disable_recurrent_edges(feedforward)
        if disabled != recurrent_edge_count:
            raise RuntimeError(
                f"disabled {disabled} recurrent edges, expected {recurrent_edge_count}"
            )
        recurrent.eval()
        feedforward.eval()

        feedforward_start = time.perf_counter()
        train_feedforward = _extract_temporal(
            feedforward, train_pixels, config.batch_size, resolved
        )
        test_feedforward = _extract_temporal(
            feedforward, test_pixels, config.batch_size, resolved
        )
        sync(resolved)
        feedforward_seconds = time.perf_counter() - feedforward_start

        recurrent_start = time.perf_counter()
        train_recurrent = _extract_temporal(
            recurrent, train_pixels, config.batch_size, resolved
        )
        test_recurrent = _extract_temporal(
            recurrent, test_pixels, config.batch_size, resolved
        )
        sync(resolved)
        recurrent_seconds = time.perf_counter() - recurrent_start

        feedforward_event_rate = _hidden_event_rate(
            train_feedforward["hidden_temporal"], config.reservoir_threshold
        )
        recurrent_event_rate = _hidden_event_rate(
            train_recurrent["hidden_temporal"], config.reservoir_threshold
        )
        feature_sets = {
            "raw_intensity": (train_pixels, test_pixels, 0.0, 0, 0.0, "raw"),
            "sensor_temporal": (
                train_recurrent["sensor_temporal"],
                test_recurrent["sensor_temporal"],
                recurrent_seconds,
                0,
                0.0,
                "sensor",
            ),
            "hidden_feedforward_temporal": (
                train_feedforward["hidden_temporal"],
                test_feedforward["hidden_temporal"],
                feedforward_seconds,
                feedforward.active_edge_count,
                feedforward_event_rate,
                "feedforward",
            ),
            "full_feedforward_temporal": (
                train_feedforward["full_temporal"],
                test_feedforward["full_temporal"],
                feedforward_seconds,
                feedforward.active_edge_count,
                feedforward_event_rate,
                "feedforward",
            ),
            "hidden_recurrent_temporal": (
                train_recurrent["hidden_temporal"],
                test_recurrent["hidden_temporal"],
                recurrent_seconds,
                recurrent.active_edge_count,
                recurrent_event_rate,
                "recurrent",
            ),
            "full_recurrent_temporal": (
                train_recurrent["full_temporal"],
                test_recurrent["full_temporal"],
                recurrent_seconds,
                recurrent.active_edge_count,
                recurrent_event_rate,
                "recurrent",
            ),
        }

        for feature in RECURRENCE_ABLATION_FEATURES:
            train_features, test_features, feature_seconds, active_edges, event_rate, topology = (
                feature_sets[feature]
            )
            for classifier_index, classifier in enumerate(("linear", "mlp")):
                hidden_units = 0
                if classifier == "mlp":
                    hidden_units = _matched_raw_hidden_units(
                        train_features.shape[1], target_dim, config.readout_hidden_units
                    )
                # The classifier-only seed offset is shared across all feature
                # arms. Equal-dimensional causal pairs therefore receive
                # identical readout initialization and minibatch order.
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

    _attach_causal_deltas(records)
    return RecurrenceAblationResult(
        config=config,
        device=device_kind(resolved),
        feedforward_edges=feedforward_edge_count,
        recurrent_edges=recurrent_edge_count,
        records=records,
        summary=summarize_recurrence_ablation(records),
    )


def disable_recurrent_edges(reservoir: FrozenEventReservoir) -> int:
    """Deactivate hidden-source edges while preserving sensor projections."""

    recurrent = reservoir.graph.active_mask & (
        reservoir.graph.sources >= reservoir.config.sensor_neurons
    )
    count = int(recurrent.sum().item())
    with torch.no_grad():
        reservoir.graph.active_mask[recurrent] = False
        reservoir.graph.short_term_weight[recurrent] = 0.0
        reservoir.graph.long_term_weight[recurrent] = 0.0
    return count


def summarize_recurrence_ablation(records: Iterable[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in records:
        grouped.setdefault((str(row["feature"]), str(row["classifier"])), []).append(row)
    summary: list[dict] = []
    for feature in RECURRENCE_ABLATION_FEATURES:
        for classifier in ("linear", "mlp"):
            rows = grouped.get((feature, classifier), [])
            if not rows:
                continue
            accuracy = [float(row["test_accuracy"]) for row in rows]
            sensor_gains = [float(row["accuracy_gain_vs_sensor"]) for row in rows]
            recurrence_gains = [float(row["recurrence_gain"]) for row in rows]
            summary.append(
                {
                    "feature": feature,
                    "classifier": classifier,
                    "topology": str(rows[0]["topology"]),
                    "seeds": len(rows),
                    "mean_test_accuracy": statistics.fmean(accuracy),
                    "std_test_accuracy": statistics.pstdev(accuracy),
                    "mean_accuracy_gain_vs_sensor": statistics.fmean(sensor_gains),
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


def plot_recurrence_ablation(summary: list[dict], path: str | pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    lookup = {(row["feature"], row["classifier"]): row for row in summary}
    features = [feature for feature in RECURRENCE_ABLATION_FEATURES if (feature, "linear") in lookup]
    positions = list(range(len(features)))
    width = 0.38
    figure, axes = plt.subplots(2, 1, figsize=(15, 10), constrained_layout=True)
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
    axes[0].set_title("AMMC Gen-5 Phase 23: Causal Recurrence Ablation")
    axes[0].set_xticks(positions, labels)
    axes[0].legend()
    axes[1].axhline(0.0, color="#222222", linewidth=1)
    axes[1].set_ylabel("Recurrent minus feedforward accuracy (points)")
    axes[1].set_xticks(positions, labels)
    axes[1].legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _attach_causal_deltas(records: list[dict]) -> None:
    lookup = {
        (int(row["seed"]), str(row["classifier"]), str(row["feature"])): float(
            row["test_accuracy"]
        )
        for row in records
    }
    pairs = {
        "hidden_recurrent_temporal": "hidden_feedforward_temporal",
        "full_recurrent_temporal": "full_feedforward_temporal",
    }
    for row in records:
        key = (int(row["seed"]), str(row["classifier"]))
        sensor = lookup[(key[0], key[1], "sensor_temporal")]
        row["paired_sensor_test_accuracy"] = sensor
        row["accuracy_gain_vs_sensor"] = float(row["test_accuracy"]) - sensor
        comparison = pairs.get(str(row["feature"]))
        if comparison is None:
            row["paired_feedforward_test_accuracy"] = float(row["test_accuracy"])
            row["recurrence_gain"] = 0.0
        else:
            baseline = lookup[(key[0], key[1], comparison)]
            row["paired_feedforward_test_accuracy"] = baseline
            row["recurrence_gain"] = float(row["test_accuracy"]) - baseline


def _hidden_event_rate(features, threshold: float) -> float:
    return float((features >= threshold).to(torch.float32).mean().item())


def _validate(config: EventMNISTConfig) -> None:
    if not config.seeds or config.train_samples <= 0 or config.test_samples <= 0:
        raise ValueError("seeds and positive sample counts are required")
    if config.timesteps < 2 or config.epochs <= 0 or config.batch_size <= 0:
        raise ValueError("timesteps >= 2 and positive epochs/batch size are required")
    required = config.sensor_neurons * config.sensor_fanout + config.hidden_neurons * config.recurrent_fanout
    if required > config.max_edges:
        raise ValueError(f"topology requires {required} edges but max_edges is {config.max_edges}")


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
