"""Gen-19 external event-vision replication on N-MNIST."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import pathlib
import statistics
from typing import Iterable
import zipfile

try:  # pragma: no cover
    import numpy as np
    import torch
except Exception:  # pragma: no cover
    np = None
    torch = None

from .runtime import device_kind, resolve_device, seed_everything
from .shd_benchmark import SHDConfig, _measure
from .shd_calibrated_baselines import TemporalConvClassifier, matched_temporal_conv_channels
from .shd_residual_state_contribution import RESIDUAL_ABLATION_MODES
from .shd_state_placement_diagnostic import (
    ResidualTemporalConvStateClassifier,
    matched_temporal_conv_residual_channels,
)
from .shd_temporal_pyramid import DEFAULT_TEMPORAL_LEVELS
from .shd_validation_checkpoint import _stratified_split, _train_validation_selected


GEN19_ARMS = ("temporal_conv1d", "residual_lif")


@dataclass(frozen=True)
class Gen19Config:
    seeds: tuple[int, ...] = (190, 191, 192)
    train_samples: int = 0
    test_samples: int = 0
    timesteps: int = 30
    spatial_bins: int = 8
    duration_us: int = 300_000
    classes: int = 10
    epochs: int = 8
    learning_rate: float = 0.003
    weight_decay: float = 0.0001
    batch_size: int = 256
    data_seed: int = 2026
    data_root: str = "gen5_data/nmnist"
    download: bool = True
    validation_fraction: float = 0.10
    target_parameters: int = 133_631
    temporal_levels: tuple[int, ...] = DEFAULT_TEMPORAL_LEVELS
    temporal_conv_kernel_size: int = 5
    surrogate_slope: float = 10.0
    minimum_reference_accuracy: float = 0.90
    maximum_accuracy_gap_vs_conv: float = 0.01
    minimum_state_effect: float = 0.005
    minimum_effect_seed_count: int = 2
    minimum_spike_activity: float = 0.01
    maximum_spike_activity: float = 0.30

    @property
    def input_neurons(self) -> int:
        return 2 * self.spatial_bins * self.spatial_bins


@dataclass
class Gen19NMNISTStateReplicationResult:
    config: dict
    device: str
    direct_channels: int
    residual_channels: int
    dataset: dict
    records: list[dict]
    summary: dict
    decision: dict

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "gen19_nmnist_state_replication.json"
        records_path = output / "gen19_nmnist_state_replication_records.csv"
        summary_path = output / "gen19_nmnist_state_replication_summary.csv"
        json_path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        _write_csv(records_path, self.records)
        _write_csv(summary_path, [self.summary])
        paths = {
            "json": str(json_path),
            "records_csv": str(records_path),
            "summary_csv": str(summary_path),
        }
        if plot:
            plot_path = output / "gen19_nmnist_state_replication.png"
            plot_gen19_nmnist_state_replication(self, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def available_gen19_arms() -> tuple[str, ...]:
    return GEN19_ARMS


def bundle_gen19_artifacts(
    paths: dict[str, str], output_dir: str | pathlib.Path
) -> dict[str, str]:
    """Write a checksummed manifest and single-file archive for Colab retrieval."""
    output = pathlib.Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = []
    existing_paths: list[pathlib.Path] = []
    for label, raw_path in paths.items():
        source = pathlib.Path(raw_path)
        if not source.exists() or not source.is_file():
            continue
        existing_paths.append(source)
        artifacts.append({
            "label": label,
            "filename": source.name,
            "bytes": source.stat().st_size,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        })
    if not existing_paths:
        raise FileNotFoundError("no Gen-19 artifacts exist to bundle")

    manifest_path = output / "gen19_nmnist_state_replication_manifest.json"
    manifest_payload = {
        "schema": "ammc-gen19-artifact-manifest-v1",
        "artifacts": artifacts,
    }
    manifest_temp = manifest_path.with_suffix(".json.tmp")
    manifest_temp.write_text(
        json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8"
    )
    manifest_temp.replace(manifest_path)

    bundle_path = output / "gen19_nmnist_state_replication_bundle.zip"
    bundle_temp = bundle_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(bundle_temp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in existing_paths:
            archive.write(source, arcname=source.name)
        archive.write(manifest_path, arcname=manifest_path.name)
    bundle_temp.replace(bundle_path)
    return {"manifest": str(manifest_path), "bundle": str(bundle_path)}


def run_gen19_nmnist_state_replication(
    config: Gen19Config | None = None,
    *,
    device: str = "auto",
    progress_path: str | pathlib.Path | None = None,
) -> Gen19NMNISTStateReplicationResult:
    if torch is None or np is None:  # pragma: no cover
        raise ImportError("Gen-19 requires PyTorch and NumPy")
    cfg = config or Gen19Config()
    _validate_config(cfg)
    resolved = resolve_device(device)
    model_config = _model_config(cfg)
    all_train_events, all_train_labels, test_events, test_labels, dataset = load_nmnist_tensors(cfg)
    train_events, train_labels, validation_events, validation_labels = _stratified_split(
        all_train_events,
        all_train_labels,
        fraction=cfg.validation_fraction,
        seed=cfg.data_seed + 19_000,
    )
    direct_channels, direct_parameters = matched_temporal_conv_channels(
        cfg.input_neurons,
        cfg.classes,
        cfg.target_parameters,
        kernel_size=cfg.temporal_conv_kernel_size,
        temporal_levels=cfg.temporal_levels,
    )
    residual_channels, residual_parameters = matched_temporal_conv_residual_channels(
        cfg.input_neurons,
        cfg.classes,
        cfg.target_parameters,
        kernel_size=cfg.temporal_conv_kernel_size,
        temporal_levels=cfg.temporal_levels,
    )
    dataset.update({
        "training_split_samples": int(train_events.shape[0]),
        "validation_samples": int(validation_events.shape[0]),
        "test_samples": int(test_events.shape[0]),
        "input_neurons": int(cfg.input_neurons),
        "direct_parameters": int(direct_parameters),
        "residual_parameters": int(residual_parameters),
    })
    records = _load_progress(progress_path, cfg) if progress_path is not None else []
    completed_seeds = {int(row["seed"]) for row in records}
    for seed in cfg.seeds:
        if seed in completed_seeds:
            continue
        seed_everything(seed, device=resolved)
        direct_model = TemporalConvClassifier(
            model_config,
            channels=direct_channels,
            kernel_size=cfg.temporal_conv_kernel_size,
            temporal_levels=cfg.temporal_levels,
        ).to(resolved)
        direct_training = _train_validation_selected(
            direct_model,
            train_events,
            train_labels,
            validation_events,
            validation_labels,
            model_config,
            seed=seed,
            device=resolved,
        )
        direct_model.load_state_dict(direct_training["best_state"])
        direct_accuracy, direct_seconds, direct_activity = _measure(
            direct_model, test_events, test_labels, cfg.batch_size, resolved
        )

        seed_everything(seed, device=resolved)
        residual_model = ResidualTemporalConvStateClassifier(
            model_config,
            channels=residual_channels,
            kernel_size=cfg.temporal_conv_kernel_size,
            temporal_levels=cfg.temporal_levels,
            dynamics="lif",
            surrogate_slope=cfg.surrogate_slope,
        ).to(resolved)
        residual_training = _train_validation_selected(
            residual_model,
            train_events,
            train_labels,
            validation_events,
            validation_labels,
            model_config,
            seed=seed,
            device=resolved,
        )
        residual_model.load_state_dict(residual_training["best_state"])
        measurements = {}
        for mode in RESIDUAL_ABLATION_MODES:
            residual_model.set_ablation_mode(mode)
            accuracy, seconds, activity = _measure(
                residual_model, test_events, test_labels, cfg.batch_size, resolved
            )
            measurements[mode] = {
                "accuracy": float(accuracy),
                "seconds": float(seconds),
                "activity": float(activity),
            }
        residual_model.set_ablation_mode("full")
        full = measurements["full"]["accuracy"]
        direct_only = measurements["direct_only"]["accuracy"]
        state_only = measurements["state_only"]["accuracy"]
        shuffled = measurements["shuffled_state"]["accuracy"]
        records.append({
            "seed": int(seed),
            "conv_best_epoch": int(direct_training["best_epoch"]),
            "conv_best_validation_accuracy": float(direct_training["best_validation_accuracy"]),
            "conv_test_accuracy": float(direct_accuracy),
            "conv_activity": float(direct_activity),
            "conv_test_examples_per_second": float(test_events.shape[0] / max(direct_seconds, 1e-12)),
            "residual_best_epoch": int(residual_training["best_epoch"]),
            "residual_best_validation_accuracy": float(residual_training["best_validation_accuracy"]),
            "full_accuracy": float(full),
            "direct_only_accuracy": float(direct_only),
            "state_only_accuracy": float(state_only),
            "shuffled_state_accuracy": float(shuffled),
            "full_gain_vs_conv": float(full - direct_accuracy),
            "state_contribution_vs_direct_only": float(full - direct_only),
            "state_specificity_vs_shuffled": float(full - shuffled),
            "direct_contribution_vs_state_only": float(full - state_only),
            "full_spike_activity": float(measurements["full"]["activity"]),
            "residual_test_examples_per_second": float(
                test_events.shape[0] / max(measurements["full"]["seconds"], 1e-12)
            ),
            "conv_train_seconds": float(direct_training["train_seconds"]),
            "residual_train_seconds": float(residual_training["train_seconds"]),
        })
        if progress_path is not None:
            _save_progress(progress_path, cfg, dataset, records)
    summary = summarize_gen19(records, minimum_state_effect=cfg.minimum_state_effect)
    decision = decide_gen19(summary, cfg)
    return Gen19NMNISTStateReplicationResult(
        config=asdict(cfg),
        device=device_kind(resolved),
        direct_channels=int(direct_channels),
        residual_channels=int(residual_channels),
        dataset=dataset,
        records=records,
        summary=summary,
        decision=decision,
    )


def encode_nmnist_events(
    events,
    *,
    timesteps: int,
    spatial_bins: int,
    duration_us: int,
):
    """Bin one N-MNIST event stream into binary time/polarity/spatial cells."""

    if np is None:  # pragma: no cover
        raise ImportError("event encoding requires NumPy")
    if timesteps <= 0 or spatial_bins <= 0 or duration_us <= 0:
        raise ValueError("encoder dimensions and duration must be positive")
    frame = np.zeros((timesteps, 2 * spatial_bins * spatial_bins), dtype=np.uint8)
    if len(events) == 0:
        return frame
    if getattr(events.dtype, "names", None):
        x = np.asarray(events["x"], dtype=np.int64)
        y = np.asarray(events["y"], dtype=np.int64)
        timestamp = np.asarray(events["t"], dtype=np.int64)
        polarity = np.asarray(events["p"], dtype=np.int64)
    else:
        values = np.asarray(events)
        if values.ndim != 2 or values.shape[1] < 4:
            raise ValueError("events must expose x, y, t, and p")
        x, y, timestamp, polarity = (
            values[:, 0].astype(np.int64),
            values[:, 1].astype(np.int64),
            values[:, 2].astype(np.int64),
            values[:, 3].astype(np.int64),
        )
    valid = (
        (x >= 0) & (x < 34) & (y >= 0) & (y < 34)
        & (timestamp >= 0) & (timestamp < duration_us)
        & (polarity >= 0) & (polarity < 2)
    )
    if not bool(valid.any()):
        return frame
    x_bin = np.minimum(spatial_bins - 1, x[valid] * spatial_bins // 34)
    y_bin = np.minimum(spatial_bins - 1, y[valid] * spatial_bins // 34)
    time_bin = np.minimum(timesteps - 1, timestamp[valid] * timesteps // duration_us)
    feature = polarity[valid] * spatial_bins * spatial_bins + y_bin * spatial_bins + x_bin
    frame[time_bin, feature] = 1
    return frame


def load_nmnist_tensors(config: Gen19Config):
    root = pathlib.Path(config.data_root)
    cache = root / "ammc_cache"
    cache.mkdir(parents=True, exist_ok=True)
    train_events, train_labels = _load_nmnist_split(config, train=True, cache=cache)
    test_events, test_labels = _load_nmnist_split(config, train=False, cache=cache)
    return train_events, train_labels, test_events, test_labels, {
        "name": "N-MNIST",
        "sensor_size": [34, 34, 2],
        "timesteps": int(config.timesteps),
        "spatial_bins": int(config.spatial_bins),
        "duration_us": int(config.duration_us),
        "train_event_density": float(torch.count_nonzero(train_events).item() / train_events.numel()),
        "test_event_density": float(torch.count_nonzero(test_events).item() / test_events.numel()),
    }


def summarize_gen19(
    records: Iterable[dict], *, minimum_state_effect: float = 0.005
) -> dict:
    rows = list(records)
    if not rows:
        return {}
    contribution = [float(row["state_contribution_vs_direct_only"]) for row in rows]
    specificity = [float(row["state_specificity_vs_shuffled"]) for row in rows]
    return {
        "runs": len(rows),
        "mean_conv_accuracy": statistics.fmean(float(row["conv_test_accuracy"]) for row in rows),
        "std_conv_accuracy": statistics.pstdev(float(row["conv_test_accuracy"]) for row in rows),
        "mean_full_accuracy": statistics.fmean(float(row["full_accuracy"]) for row in rows),
        "std_full_accuracy": statistics.pstdev(float(row["full_accuracy"]) for row in rows),
        "mean_direct_only_accuracy": statistics.fmean(float(row["direct_only_accuracy"]) for row in rows),
        "mean_state_only_accuracy": statistics.fmean(float(row["state_only_accuracy"]) for row in rows),
        "mean_shuffled_state_accuracy": statistics.fmean(float(row["shuffled_state_accuracy"]) for row in rows),
        "mean_gain_vs_conv": statistics.fmean(float(row["full_gain_vs_conv"]) for row in rows),
        "mean_state_contribution_vs_direct_only": statistics.fmean(contribution),
        "state_contribution_seed_count": sum(value >= minimum_state_effect for value in contribution),
        "mean_state_specificity_vs_shuffled": statistics.fmean(specificity),
        "state_specificity_seed_count": sum(value >= minimum_state_effect for value in specificity),
        "mean_direct_contribution_vs_state_only": statistics.fmean(
            float(row["direct_contribution_vs_state_only"]) for row in rows
        ),
        "mean_spike_activity": statistics.fmean(float(row["full_spike_activity"]) for row in rows),
        "mean_conv_test_examples_per_second": statistics.fmean(
            float(row["conv_test_examples_per_second"]) for row in rows
        ),
        "mean_residual_test_examples_per_second": statistics.fmean(
            float(row["residual_test_examples_per_second"]) for row in rows
        ),
    }


def decide_gen19(summary: dict, config: Gen19Config) -> dict:
    if not summary:
        return {"status": "stop", "next_milestone": "invalid_empty_result"}
    reference_gate = float(summary["mean_conv_accuracy"]) >= config.minimum_reference_accuracy
    accuracy_gate = (
        float(summary["mean_full_accuracy"]) - float(summary["mean_conv_accuracy"])
        >= -config.maximum_accuracy_gap_vs_conv
    )
    contribution_gate = (
        float(summary["mean_state_contribution_vs_direct_only"]) >= config.minimum_state_effect
        and int(summary["state_contribution_seed_count"]) >= config.minimum_effect_seed_count
    )
    specificity_gate = (
        float(summary["mean_state_specificity_vs_shuffled"]) >= config.minimum_state_effect
        and int(summary["state_specificity_seed_count"]) >= config.minimum_effect_seed_count
    )
    activity = float(summary["mean_spike_activity"])
    activity_gate = config.minimum_spike_activity <= activity <= config.maximum_spike_activity
    passed = all((reference_gate, accuracy_gate, contribution_gate, specificity_gate, activity_gate))
    return {
        "status": "pass" if passed else "stop",
        "dataset_learnability_gate": reference_gate,
        "matched_accuracy_gate": accuracy_gate,
        "state_contribution_gate": contribution_gate,
        "state_identity_gate": specificity_gate,
        "spike_activity_gate": activity_gate,
        "next_milestone": (
            "freeze_cross_modal_residual_state_mechanism"
            if passed else "limit_residual_state_claim_to_event_audio"
        ),
    }


def plot_gen19_nmnist_state_replication(
    result: Gen19NMNISTStateReplicationResult, path: str | pathlib.Path
) -> None:
    import matplotlib.pyplot as plt

    summary = result.summary
    figure, axes = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)
    axes[0].bar(
        ("Conv1D", "Residual full", "Direct only", "State only", "Shuffled state"),
        [
            100.0 * float(summary["mean_conv_accuracy"]),
            100.0 * float(summary["mean_full_accuracy"]),
            100.0 * float(summary["mean_direct_only_accuracy"]),
            100.0 * float(summary["mean_state_only_accuracy"]),
            100.0 * float(summary["mean_shuffled_state_accuracy"]),
        ],
        color=("#8b6fd6", "#167d55", "#35b4f2", "#d88935", "#bd3d3a"),
    )
    axes[0].set_ylabel("N-MNIST test accuracy (%)")
    axes[0].set_title("Gen-19 real event-vision residual-state replication")
    axes[1].bar(
        ("Full - Conv1D", "Full - direct only", "Full - shuffled state"),
        [
            100.0 * float(summary["mean_gain_vs_conv"]),
            100.0 * float(summary["mean_state_contribution_vs_direct_only"]),
            100.0 * float(summary["mean_state_specificity_vs_shuffled"]),
        ],
        color=("#8b6fd6", "#35b4f2", "#167d55"),
    )
    axes[1].axhline(0.5, color="#bd3d3a", linestyle="--", label="registered causal threshold")
    axes[1].set_ylabel("Accuracy difference (points)")
    axes[1].legend()
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _load_nmnist_split(config: Gen19Config, *, train: bool, cache: pathlib.Path):
    split = "train" if train else "test"
    limit = config.train_samples if train else config.test_samples
    cache_path = cache / (
        f"{split}_t{config.timesteps}_s{config.spatial_bins}_d{config.duration_us}_"
        f"n{limit or 'all'}_seed{config.data_seed}.pt"
    )
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        return payload["events"], payload["labels"]
    if not config.download:
        raise FileNotFoundError(
            f"N-MNIST cache missing: {cache_path}. Remove --no-download to build it."
        )
    try:
        import tonic
    except ImportError as error:  # pragma: no cover
        raise ImportError(
            "Gen-19 N-MNIST download requires tonic. Install it with `pip install tonic`."
        ) from error
    dataset = tonic.datasets.NMNIST(save_to=str(pathlib.Path(config.data_root)), train=train)
    indices = _subset_indices(len(dataset), limit, config.data_seed + (0 if train else 1))
    encoded = torch.empty(
        (len(indices), config.timesteps, config.input_neurons), dtype=torch.uint8
    )
    labels = torch.empty((len(indices),), dtype=torch.long)
    for output_index, dataset_index in enumerate(indices):
        events, label = dataset[int(dataset_index)]
        encoded[output_index] = torch.from_numpy(encode_nmnist_events(
            events,
            timesteps=config.timesteps,
            spatial_bins=config.spatial_bins,
            duration_us=config.duration_us,
        ))
        labels[output_index] = int(label)
    torch.save({"events": encoded, "labels": labels}, cache_path)
    return encoded, labels


def _subset_indices(length: int, limit: int, seed: int) -> list[int]:
    if limit <= 0 or limit >= length:
        return list(range(length))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(length, generator=generator)[:limit].tolist()


def _model_config(config: Gen19Config) -> SHDConfig:
    return SHDConfig(
        seeds=config.seeds,
        train_samples=config.train_samples,
        test_samples=config.test_samples,
        input_neurons=config.input_neurons,
        classes=config.classes,
        timesteps=config.timesteps,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        batch_size=config.batch_size,
        data_seed=config.data_seed,
        data_root=config.data_root,
        download=config.download,
    )


def _save_progress(path, config, dataset, records):
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({
        "config": asdict(config),
        "dataset": dataset,
        "records": records,
    }, indent=2) + "\n", encoding="utf-8")


def _load_progress(path, config) -> list[dict]:
    source = pathlib.Path(path)
    if not source.exists():
        return []
    payload = json.loads(source.read_text(encoding="utf-8"))
    expected = json.loads(json.dumps(asdict(config)))
    if payload.get("config") != expected:
        raise ValueError(
            "Gen-19 progress configuration differs from the requested run; use a new output directory."
        )
    records = list(payload.get("records", []))
    seeds = [int(row["seed"]) for row in records]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Gen-19 progress contains duplicate seed records")
    return records


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validate_config(config: Gen19Config) -> None:
    if len(config.seeds) != 3 or len(set(config.seeds)) != 3:
        raise ValueError("Gen-19 requires exactly three unique seeds")
    if config.train_samples < 0 or config.test_samples < 0:
        raise ValueError("sample limits cannot be negative")
    if config.timesteps <= 0 or config.spatial_bins <= 0 or config.duration_us <= 0:
        raise ValueError("event dimensions and duration must be positive")
    if not 0.0 < config.validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5)")
    if config.epochs <= 0 or config.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if config.target_parameters <= 0:
        raise ValueError("target_parameters must be positive")
    if config.minimum_effect_seed_count > len(config.seeds):
        raise ValueError("minimum_effect_seed_count exceeds seed count")
    if not 0.0 <= config.minimum_spike_activity < config.maximum_spike_activity <= 1.0:
        raise ValueError("invalid spike activity interval")
