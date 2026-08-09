"""Phase 20 temporal-state preservation benchmark for frozen event MNIST."""

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
    _extract_latency_features,
    _fit_and_measure,
    _matched_raw_hidden_units,
    load_mnist_tensors,
    torch,
)
from .runtime import device_kind, mark_step, resolve_device, seed_everything, sync


TEMPORAL_FEATURES = (
    "raw_intensity",
    "flattened_latency",
    "full_summary",
    "sensor_temporal",
    "hidden_temporal",
    "full_temporal",
    "raw_plus_hidden_temporal",
)


def _require_torch() -> None:
    if torch is None:
        raise ImportError("Phase 20 temporal-state MNIST requires PyTorch")


@dataclass
class TemporalMNISTResult:
    """Seed records and aggregates for temporal-state preservation."""

    config: EventMNISTConfig
    device: str
    active_edges: int
    records: list[dict]
    summary: list[dict]

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "event_mnist_temporal.json"
        records_path = output / "event_mnist_temporal_records.csv"
        summary_path = output / "event_mnist_temporal_summary.csv"
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
            plot_path = output / "event_mnist_temporal_summary.png"
            plot_temporal_mnist_result(self.summary, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def run_temporal_mnist(config: EventMNISTConfig, *, device="auto") -> TemporalMNISTResult:
    """Compare final summaries against preserved per-timestep neuron states."""

    _require_torch()
    _validate_config(config)
    resolved = resolve_device(device)
    train_pixels, train_labels, test_pixels, test_labels = load_mnist_tensors(config)
    latency_start = time.perf_counter()
    train_latency = _extract_latency_features(train_pixels, config, config.batch_size, resolved)
    test_latency = _extract_latency_features(test_pixels, config, config.batch_size, resolved)
    sync(resolved)
    latency_seconds = time.perf_counter() - latency_start
    records: list[dict] = []
    expected_edges = config.sensor_neurons * config.sensor_fanout + config.hidden_neurons * config.recurrent_fanout
    target_dim = config.neuron_count * 2

    for seed in config.seeds:
        seed_everything(seed, device=resolved)
        reservoir = FrozenEventReservoir(config, seed=seed, device=resolved)
        reservoir.eval()
        summary_start = time.perf_counter()
        train_summary = _extract_summary(reservoir, train_pixels, config.batch_size, resolved)
        test_summary = _extract_summary(reservoir, test_pixels, config.batch_size, resolved)
        sync(resolved)
        summary_seconds = time.perf_counter() - summary_start
        temporal_start = time.perf_counter()
        train_temporal = _extract_temporal(reservoir, train_pixels, config.batch_size, resolved)
        test_temporal = _extract_temporal(reservoir, test_pixels, config.batch_size, resolved)
        sync(resolved)
        temporal_seconds = time.perf_counter() - temporal_start
        train_temporal["raw_plus_hidden_temporal"] = torch.cat(
            (train_pixels, train_temporal["hidden_temporal"]), dim=1
        )
        test_temporal["raw_plus_hidden_temporal"] = torch.cat(
            (test_pixels, test_temporal["hidden_temporal"]), dim=1
        )
        hidden_event_rate = float(
            (train_temporal["hidden_temporal"] >= config.reservoir_threshold).to(torch.float32).mean().item()
        )
        feature_sets = {
            "raw_intensity": (train_pixels, test_pixels, 0.0, 0),
            "flattened_latency": (train_latency, test_latency, latency_seconds, 0),
            "full_summary": (train_summary, test_summary, summary_seconds, reservoir.active_edge_count),
            "sensor_temporal": (
                train_temporal["sensor_temporal"],
                test_temporal["sensor_temporal"],
                temporal_seconds,
                0,
            ),
            "hidden_temporal": (
                train_temporal["hidden_temporal"],
                test_temporal["hidden_temporal"],
                temporal_seconds,
                reservoir.active_edge_count,
            ),
            "full_temporal": (
                train_temporal["full_temporal"],
                test_temporal["full_temporal"],
                temporal_seconds,
                reservoir.active_edge_count,
            ),
            "raw_plus_hidden_temporal": (
                train_temporal["raw_plus_hidden_temporal"],
                test_temporal["raw_plus_hidden_temporal"],
                temporal_seconds,
                reservoir.active_edge_count,
            ),
        }
        for feature_index, feature_name in enumerate(TEMPORAL_FEATURES):
            train_features, test_features, feature_seconds, active_edges = feature_sets[feature_name]
            for classifier_index, classifier in enumerate(("linear", "mlp")):
                hidden_units = 0
                if classifier == "mlp":
                    hidden_units = _matched_raw_hidden_units(
                        train_features.shape[1], target_dim, config.readout_hidden_units
                    )
                record = _fit_and_measure(
                    f"{feature_name}_{classifier}",
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
                    model_seed_offset=feature_index * 2 + classifier_index,
                    hidden_spike_rate=hidden_event_rate if active_edges else 0.0,
                )
                record["feature"] = feature_name
                record["classifier"] = classifier
                records.append(record)

    return TemporalMNISTResult(
        config=config,
        device=device_kind(resolved),
        active_edges=expected_edges,
        records=records,
        summary=summarize_temporal_mnist(records),
    )


def summarize_temporal_mnist(records: Iterable[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in records:
        grouped.setdefault((str(row["feature"]), str(row["classifier"])), []).append(row)
    summary: list[dict] = []
    for feature in TEMPORAL_FEATURES:
        for classifier in ("linear", "mlp"):
            rows = grouped.get((feature, classifier), [])
            if not rows:
                continue
            accuracy = [float(row["test_accuracy"]) for row in rows]
            summary.append(
                {
                    "feature": feature,
                    "classifier": classifier,
                    "seeds": len(rows),
                    "mean_test_accuracy": statistics.fmean(accuracy),
                    "std_test_accuracy": statistics.pstdev(accuracy),
                    "mean_train_accuracy": statistics.fmean(float(row["train_accuracy"]) for row in rows),
                    "feature_dim": int(rows[0]["feature_dim"]),
                    "trainable_parameters": int(rows[0]["trainable_parameters"]),
                    "classifier_hidden_units": int(rows[0]["classifier_hidden_units"]),
                    "frozen_active_edges": int(rows[0]["frozen_active_edges"]),
                    "mean_hidden_event_rate": statistics.fmean(
                        float(row["mean_hidden_spike_rate"]) for row in rows
                    ),
                    "mean_feature_seconds": statistics.fmean(float(row["feature_seconds"]) for row in rows),
                    "mean_feature_examples_per_second": statistics.fmean(
                        float(row["feature_examples_per_second"]) for row in rows
                    ),
                    "mean_train_seconds": statistics.fmean(float(row["train_seconds"]) for row in rows),
                }
            )
    return summary


def plot_temporal_mnist_result(summary: list[dict], path: str | pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    rows = {(row["feature"], row["classifier"]): row for row in summary}
    features = [feature for feature in TEMPORAL_FEATURES if (feature, "linear") in rows]
    positions = list(range(len(features)))
    width = 0.38
    linear = [100.0 * float(rows[(feature, "linear")]["mean_test_accuracy"]) for feature in features]
    mlp = [100.0 * float(rows[(feature, "mlp")]["mean_test_accuracy"]) for feature in features]
    linear_error = [100.0 * float(rows[(feature, "linear")]["std_test_accuracy"]) for feature in features]
    mlp_error = [100.0 * float(rows[(feature, "mlp")]["std_test_accuracy"]) for feature in features]
    figure, axis = plt.subplots(figsize=(15, 7), constrained_layout=True)
    axis.bar([x - width / 2 for x in positions], linear, width, yerr=linear_error, capsize=4, label="linear")
    axis.bar([x + width / 2 for x in positions], mlp, width, yerr=mlp_error, capsize=4, label="parameter-matched MLP")
    axis.set_xticks(positions, [feature.replace("_", "\n") for feature in features])
    axis.set_ylabel("Engineering-validation accuracy (%)")
    axis.set_ylim(0, 100)
    axis.set_title("AMMC Gen-5 Phase 20: Temporal-State Preservation")
    axis.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _extract_summary(reservoir, pixels, batch_size: int, device):
    outputs = []
    with torch.no_grad():
        for start in range(0, pixels.shape[0], batch_size):
            outputs.append(reservoir(pixels[start : start + batch_size].to(device)).cpu())
            mark_step(device)
    sync(device)
    return torch.cat(outputs, dim=0)


def _extract_temporal(reservoir, pixels, batch_size: int, device):
    outputs: dict[str, list] = {"sensor_temporal": [], "hidden_temporal": [], "full_temporal": []}
    with torch.no_grad():
        for start in range(0, pixels.shape[0], batch_size):
            components = reservoir.temporal_components(pixels[start : start + batch_size].to(device))
            for name in outputs:
                outputs[name].append(components[name].cpu())
            mark_step(device)
    sync(device)
    return {name: torch.cat(chunks, dim=0) for name, chunks in outputs.items()}


def _validate_config(config: EventMNISTConfig) -> None:
    if not config.seeds or config.train_samples <= 0 or config.test_samples <= 0:
        raise ValueError("seeds and positive train/test sample counts are required")
    required = config.sensor_neurons * config.sensor_fanout + config.hidden_neurons * config.recurrent_fanout
    if required > config.max_edges:
        raise ValueError(f"topology requires {required} edges but max_edges is {config.max_edges}")
    if config.timesteps < 2 or config.epochs <= 0 or config.batch_size <= 0:
        raise ValueError("timesteps >= 2 and positive epochs/batch_size are required")


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
