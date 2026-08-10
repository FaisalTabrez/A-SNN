"""Bounded full-resolution N-MNIST accuracy benchmark."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import pathlib
import statistics
import time
from typing import Iterable
import zipfile

try:  # pragma: no cover - dependency availability is environment specific
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    np = None
    torch = None
    nn = None
    F = None

from .runtime import device_kind, mark_step, resolve_device, seed_everything, sync
from .trainable_temporal_mnist import SurrogateSpike


NMNIST_ACCURACY_ARMS = (
    "frame_cnn",
    "spatiotemporal_cnn",
    "conv_plif",
)


@dataclass(frozen=True)
class NMNISTAccuracyConfig:
    screen_seed: int = 210
    confirmation_seeds: tuple[int, ...] = (211, 212, 213)
    timesteps: int = 10
    duration_us: int = 300_000
    sensor_size: int = 34
    classes: int = 10
    screen_train_samples: int = 20_000
    train_samples: int = 0
    test_samples: int = 0
    screen_epochs: int = 4
    confirmation_epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    validation_fraction: float = 0.10
    data_seed: int = 2026
    data_root: str = "gen5_data/nmnist"
    download: bool = True
    promotion_gap: float = 0.01
    maximum_promoted_arms: int = 2
    practical_accuracy: float = 0.99
    stretch_accuracy: float = 0.994
    event_dropout: float = 0.02
    maximum_shift: int = 2
    surrogate_slope: float = 10.0


@dataclass
class NMNISTAccuracyBenchmarkResult:
    config: dict
    device: str
    dataset: dict
    screen_records: list[dict]
    promoted_arms: list[str]
    confirmation_records: list[dict]
    summary: list[dict]
    decision: dict

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "nmnist_accuracy_benchmark.json"
        screen_path = output / "nmnist_accuracy_benchmark_screen.csv"
        records_path = output / "nmnist_accuracy_benchmark_records.csv"
        summary_path = output / "nmnist_accuracy_benchmark_summary.csv"
        json_path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        _write_csv(screen_path, self.screen_records)
        _write_csv(records_path, self.confirmation_records)
        _write_csv(summary_path, self.summary)
        paths = {
            "json": str(json_path),
            "screen_csv": str(screen_path),
            "records_csv": str(records_path),
            "summary_csv": str(summary_path),
        }
        if plot:
            plot_path = output / "nmnist_accuracy_benchmark.png"
            plot_nmnist_accuracy_benchmark(self, plot_path)
            paths["plot"] = str(plot_path)
        return paths


_ModuleBase = nn.Module if nn is not None else object


class FrameCNN(_ModuleBase):
    """Strong static-frame ceiling using all full-resolution event bins."""

    def __init__(self, classes: int = 10) -> None:
        _require_torch()
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(256, classes),
        )

    def forward(self, events):  # type: ignore[override]
        counts = torch.log1p(events.sum(dim=1)) / math.log(events.shape[1] + 1.0)
        return self.classifier(self.features(counts)), events.new_zeros(())


class SpatiotemporalCNN(_ModuleBase):
    """Dense 3-D convolutional reference retaining spatial and temporal axes."""

    def __init__(self, classes: int = 10) -> None:
        _require_torch()
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(2, 32, 3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.SiLU(),
            nn.Conv3d(32, 48, 3, padding=1, bias=False),
            nn.BatchNorm3d(48),
            nn.SiLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(48, 96, 3, padding=1, bias=False),
            nn.BatchNorm3d(96),
            nn.SiLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(96, 128, 3, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d((2, 4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 4 * 4, 256),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(256, classes),
        )

    def forward(self, events):  # type: ignore[override]
        volume = events.transpose(1, 2)
        return self.classifier(self.features(volume)), events.new_zeros(())


class LearnableLIF2d(_ModuleBase):
    """Channel-wise parametric LIF cell with surrogate gradients."""

    def __init__(self, channels: int, *, slope: float = 10.0) -> None:
        _require_torch()
        super().__init__()
        self.beta_logit = nn.Parameter(torch.full((1, channels, 1, 1), 1.4))
        self.threshold_log = nn.Parameter(torch.zeros((1, channels, 1, 1)))
        self.slope = float(slope)

    def forward(self, current, membrane):  # type: ignore[override]
        beta = 0.5 + 0.49 * torch.sigmoid(self.beta_logit)
        threshold = 0.5 + F.softplus(self.threshold_log)
        pre_reset = beta * membrane + current
        spikes = SurrogateSpike.apply(pre_reset - threshold, self.slope)
        return spikes, pre_reset - spikes * threshold


class ConvPLIFClassifier(_ModuleBase):
    """Full-resolution convolutional SNN with learnable membrane dynamics."""

    def __init__(self, classes: int = 10, *, surrogate_slope: float = 10.0) -> None:
        _require_torch()
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.lif1 = LearnableLIF2d(32, slope=surrogate_slope)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.lif2 = LearnableLIF2d(64, slope=surrogate_slope)
        self.conv3 = nn.Conv2d(64, 96, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(96)
        self.lif3 = LearnableLIF2d(96, slope=surrogate_slope)
        self.readout = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 4 * 4, 256),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(256, classes),
        )

    def forward(self, events):  # type: ignore[override]
        batch = events.shape[0]
        mem1 = events.new_zeros((batch, 32, 34, 34))
        mem2 = events.new_zeros((batch, 64, 17, 17))
        mem3 = events.new_zeros((batch, 96, 8, 8))
        logits = events.new_zeros((batch, self.readout[-1].out_features))
        spike_sum = events.new_zeros(())
        for step in range(events.shape[1]):
            spike1, mem1 = self.lif1(self.bn1(self.conv1(events[:, step])), mem1)
            pooled1 = F.max_pool2d(spike1, 2)
            spike2, mem2 = self.lif2(self.bn2(self.conv2(pooled1)), mem2)
            pooled2 = F.max_pool2d(spike2, 2)
            spike3, mem3 = self.lif3(self.bn3(self.conv3(pooled2)), mem3)
            features = F.adaptive_avg_pool2d(spike3, (4, 4))
            logits = logits + self.readout(features)
            spike_sum = spike_sum + (spike1.mean() + spike2.mean() + spike3.mean()) / 3.0
        return logits / events.shape[1], spike_sum / events.shape[1]


def available_nmnist_accuracy_arms() -> tuple[str, ...]:
    return NMNIST_ACCURACY_ARMS


def build_nmnist_accuracy_model(
    arm: str, *, classes: int = 10, surrogate_slope: float = 10.0
):
    if arm == "frame_cnn":
        return FrameCNN(classes)
    if arm == "spatiotemporal_cnn":
        return SpatiotemporalCNN(classes)
    if arm == "conv_plif":
        return ConvPLIFClassifier(classes, surrogate_slope=surrogate_slope)
    raise ValueError(f"unknown N-MNIST accuracy arm: {arm}")


def encode_nmnist_full_resolution(events, *, timesteps: int, duration_us: int):
    """Encode an N-MNIST sample as binary [time, polarity, y, x] occupancy."""
    if np is None:  # pragma: no cover
        raise ImportError("N-MNIST encoding requires NumPy")
    if timesteps <= 0 or duration_us <= 0:
        raise ValueError("timesteps and duration_us must be positive")
    output = np.zeros((timesteps, 2, 34, 34), dtype=np.uint8)
    if len(events) == 0:
        return output
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
    if bool(valid.any()):
        time_bin = np.minimum(timesteps - 1, timestamp[valid] * timesteps // duration_us)
        output[time_bin, polarity[valid], y[valid], x[valid]] = 1
    return output


def select_nmnist_accuracy_promoted_arms(
    screen_records: Iterable[dict], *, gap: float, maximum: int
) -> list[str]:
    rows = sorted(
        screen_records,
        key=lambda row: (-float(row["best_validation_accuracy"]), str(row["arm"])),
    )
    if not rows or maximum <= 0:
        return []
    best = float(rows[0]["best_validation_accuracy"])
    return [
        str(row["arm"])
        for row in rows
        if float(row["best_validation_accuracy"]) >= best - gap
    ][:maximum]


def run_nmnist_accuracy_benchmark(
    config: NMNISTAccuracyConfig | None = None,
    *,
    device: str = "auto",
    progress_path: str | pathlib.Path | None = None,
) -> NMNISTAccuracyBenchmarkResult:
    _require_torch()
    cfg = config or NMNISTAccuracyConfig()
    _validate_config(cfg)
    resolved = resolve_device(device)
    all_train, all_labels, test_events, test_labels, dataset = load_nmnist_accuracy_tensors(cfg)
    train_events, train_labels, validation_events, validation_labels = _stratified_split(
        all_train,
        all_labels,
        fraction=cfg.validation_fraction,
        seed=cfg.data_seed + 20_000,
    )
    screen_events, screen_labels = _stratified_limit(
        train_events, train_labels, cfg.screen_train_samples, cfg.data_seed + 20_001
    )
    progress = _load_progress(progress_path, cfg) if progress_path is not None else {}
    screen_records = list(progress.get("screen_records", []))
    confirmation_records = list(progress.get("confirmation_records", []))

    completed_screen = {str(row["arm"]) for row in screen_records}
    for arm in NMNIST_ACCURACY_ARMS:
        if arm in completed_screen:
            continue
        seed_everything(cfg.screen_seed, device=resolved)
        model = build_nmnist_accuracy_model(
            arm, classes=cfg.classes, surrogate_slope=cfg.surrogate_slope
        ).to(resolved)
        parameters = _parameter_count(model)
        macs = estimate_nmnist_dense_macs(model, cfg.timesteps, cfg.sensor_size)
        training = _train_model(
            model,
            screen_events,
            screen_labels,
            validation_events,
            validation_labels,
            epochs=cfg.screen_epochs,
            config=cfg,
            seed=cfg.screen_seed,
            device=resolved,
        )
        screen_records.append({
            "arm": arm,
            "seed": cfg.screen_seed,
            "parameters": parameters,
            "dense_macs_per_sample": macs,
            "best_epoch": training["best_epoch"],
            "best_validation_accuracy": training["best_validation_accuracy"],
            "validation_activity": training["best_validation_activity"],
            "train_seconds": training["train_seconds"],
        })
        print(
            f"[screen] arm={arm} validation={training['best_validation_accuracy']:.4f} "
            f"epoch={training['best_epoch']}",
            flush=True,
        )
        del model
        _save_progress(progress_path, cfg, dataset, screen_records, confirmation_records)

    promoted = select_nmnist_accuracy_promoted_arms(
        screen_records, gap=cfg.promotion_gap, maximum=cfg.maximum_promoted_arms
    )
    completed_confirmation = {
        (str(row["arm"]), int(row["seed"])) for row in confirmation_records
    }
    for arm in promoted:
        for seed in cfg.confirmation_seeds:
            if (arm, int(seed)) in completed_confirmation:
                continue
            seed_everything(seed, device=resolved)
            model = build_nmnist_accuracy_model(
                arm, classes=cfg.classes, surrogate_slope=cfg.surrogate_slope
            ).to(resolved)
            parameters = _parameter_count(model)
            macs = estimate_nmnist_dense_macs(model, cfg.timesteps, cfg.sensor_size)
            training = _train_model(
                model,
                train_events,
                train_labels,
                validation_events,
                validation_labels,
                epochs=cfg.confirmation_epochs,
                config=cfg,
                seed=seed,
                device=resolved,
            )
            model.load_state_dict(training["best_state"])
            accuracy, activity, seconds = _measure(
                model, test_events, test_labels, cfg.batch_size, resolved
            )
            confirmation_records.append({
                "arm": arm,
                "seed": int(seed),
                "parameters": parameters,
                "dense_macs_per_sample": macs,
                "activity_scaled_ops_proxy": (
                    macs * activity if arm == "conv_plif" else macs
                ),
                "best_epoch": training["best_epoch"],
                "best_validation_accuracy": training["best_validation_accuracy"],
                "test_accuracy": accuracy,
                "test_activity": activity,
                "test_examples_per_second": len(test_events) / max(seconds, 1e-12),
                "train_seconds": training["train_seconds"],
            })
            print(
                f"[confirm] arm={arm} seed={seed} test={accuracy:.4f} "
                f"validation={training['best_validation_accuracy']:.4f}",
                flush=True,
            )
            del model
            _save_progress(progress_path, cfg, dataset, screen_records, confirmation_records)

    summary = summarize_nmnist_accuracy(confirmation_records)
    decision = decide_nmnist_accuracy(summary, cfg)
    dataset.update({
        "training_split_samples": int(train_events.shape[0]),
        "validation_samples": int(validation_events.shape[0]),
        "screen_training_samples": int(screen_events.shape[0]),
        "test_samples": int(test_events.shape[0]),
    })
    return NMNISTAccuracyBenchmarkResult(
        config=asdict(cfg),
        device=device_kind(resolved),
        dataset=dataset,
        screen_records=screen_records,
        promoted_arms=promoted,
        confirmation_records=confirmation_records,
        summary=summary,
        decision=decision,
    )


def summarize_nmnist_accuracy(records: Iterable[dict]) -> list[dict]:
    rows = list(records)
    summary = []
    for arm in NMNIST_ACCURACY_ARMS:
        group = [row for row in rows if row["arm"] == arm]
        if not group:
            continue
        accuracies = [float(row["test_accuracy"]) for row in group]
        summary.append({
            "arm": arm,
            "runs": len(group),
            "mean_test_accuracy": statistics.fmean(accuracies),
            "std_test_accuracy": statistics.pstdev(accuracies),
            "minimum_test_accuracy": min(accuracies),
            "mean_best_validation_accuracy": statistics.fmean(
                float(row["best_validation_accuracy"]) for row in group
            ),
            "mean_test_activity": statistics.fmean(float(row["test_activity"]) for row in group),
            "mean_test_examples_per_second": statistics.fmean(
                float(row["test_examples_per_second"]) for row in group
            ),
            "parameters": int(group[0]["parameters"]),
            "dense_macs_per_sample": int(group[0]["dense_macs_per_sample"]),
            "mean_activity_scaled_ops_proxy": statistics.fmean(
                float(row["activity_scaled_ops_proxy"]) for row in group
            ),
        })
    return summary


def decide_nmnist_accuracy(summary: list[dict], config: NMNISTAccuracyConfig) -> dict:
    if not summary:
        return {"status": "stop", "next_milestone": "return_to_gen20"}
    best = max(summary, key=lambda row: float(row["mean_test_accuracy"]))
    spiking = next((row for row in summary if row["arm"] == "conv_plif"), None)
    mean_accuracy = float(best["mean_test_accuracy"])
    return {
        "status": "pass" if mean_accuracy >= config.practical_accuracy else "stop",
        "best_arm": best["arm"],
        "best_mean_test_accuracy": mean_accuracy,
        "practical_accuracy_gate": mean_accuracy >= config.practical_accuracy,
        "stretch_accuracy_gate": mean_accuracy >= config.stretch_accuracy,
        "spiking_confirmed": spiking is not None,
        "spiking_mean_test_accuracy": (
            float(spiking["mean_test_accuracy"]) if spiking is not None else None
        ),
        "next_milestone": "return_to_gen20",
    }


def load_nmnist_accuracy_tensors(config: NMNISTAccuracyConfig):
    root = pathlib.Path(config.data_root)
    cache = root / "ammc_accuracy_cache"
    cache.mkdir(parents=True, exist_ok=True)
    train_events, train_labels = _load_split(config, train=True, cache=cache)
    test_events, test_labels = _load_split(config, train=False, cache=cache)
    return train_events, train_labels, test_events, test_labels, {
        "name": "N-MNIST",
        "sensor_size": [34, 34, 2],
        "timesteps": config.timesteps,
        "duration_us": config.duration_us,
        "encoding": "binary_full_resolution_time_polarity_occupancy",
        "train_event_density": float(torch.count_nonzero(train_events) / train_events.numel()),
        "test_event_density": float(torch.count_nonzero(test_events) / test_events.numel()),
    }


def estimate_nmnist_dense_macs(model, timesteps: int, sensor_size: int = 34) -> int:
    """Count dense Conv/Linear multiply-accumulates for one forward pass."""
    _require_torch()
    total = 0

    def hook(module, inputs, output):
        nonlocal total
        value = output[0] if isinstance(output, tuple) else output
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            kernel = math.prod(module.kernel_size)
            total += int(value.numel() * kernel * module.in_channels / module.groups)
        elif isinstance(module, nn.Linear):
            total += int(value.numel() * module.in_features)

    handles = [
        module.register_forward_hook(hook)
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear))
    ]
    original_device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    with torch.no_grad():
        sample = torch.zeros((1, timesteps, 2, sensor_size, sensor_size), device=original_device)
        model(sample)
    for handle in handles:
        handle.remove()
    model.train(was_training)
    return total


def bundle_nmnist_accuracy_artifacts(
    paths: dict[str, str], output_dir: str | pathlib.Path
) -> dict[str, str]:
    output = pathlib.Path(output_dir)
    files = []
    manifest_rows = []
    for label, raw_path in paths.items():
        source = pathlib.Path(raw_path)
        if source.is_file():
            files.append(source)
            manifest_rows.append({
                "label": label,
                "filename": source.name,
                "bytes": source.stat().st_size,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            })
    if not files:
        raise FileNotFoundError("no N-MNIST accuracy artifacts exist to bundle")
    manifest = output / "nmnist_accuracy_benchmark_manifest.json"
    manifest_temp = manifest.with_suffix(".json.tmp")
    manifest_temp.write_text(json.dumps({
        "schema": "ammc-nmnist-accuracy-benchmark-v1",
        "artifacts": manifest_rows,
    }, indent=2) + "\n", encoding="utf-8")
    manifest_temp.replace(manifest)
    bundle = output / "nmnist_accuracy_benchmark_bundle.zip"
    bundle_temp = bundle.with_suffix(".zip.tmp")
    with zipfile.ZipFile(bundle_temp, "w", zipfile.ZIP_DEFLATED) as archive:
        for source in files:
            archive.write(source, arcname=source.name)
        archive.write(manifest, arcname=manifest.name)
    bundle_temp.replace(bundle)
    return {"manifest": str(manifest), "bundle": str(bundle)}


def plot_nmnist_accuracy_benchmark(
    result: NMNISTAccuracyBenchmarkResult, path: str | pathlib.Path
) -> None:
    import matplotlib.pyplot as plt

    labels = [row["arm"].replace("_", "\n") for row in result.summary]
    accuracy = [100.0 * float(row["mean_test_accuracy"]) for row in result.summary]
    errors = [100.0 * float(row["std_test_accuracy"]) for row in result.summary]
    throughput = [float(row["mean_test_examples_per_second"]) for row in result.summary]
    ops = [float(row["mean_activity_scaled_ops_proxy"]) for row in result.summary]
    figure, axes = plt.subplots(3, 1, figsize=(12, 14), constrained_layout=True)
    axes[0].bar(labels, accuracy, yerr=errors, capsize=5, color="#35b4f2")
    axes[0].axhline(99.0, color="#d88935", linestyle="--", label="practical gate")
    axes[0].axhline(99.4, color="#167d55", linestyle=":", label="stretch target")
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].legend()
    axes[1].bar(labels, throughput, color="#8b6fd6")
    axes[1].set_ylabel("Examples / second")
    axes[2].bar(labels, ops, color="#ffb31a")
    axes[2].set_ylabel("Activity-scaled operation proxy")
    axes[2].set_yscale("log")
    axes[0].set_title("AMMC bounded full-resolution N-MNIST accuracy benchmark")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _train_model(
    model,
    train_events,
    train_labels,
    validation_events,
    validation_labels,
    *,
    epochs: int,
    config: NMNISTAccuracyConfig,
    seed: int,
    device,
) -> dict:
    seed_everything(seed, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = getattr(device, "type", None) == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_accuracy = -1.0
    best_activity = 0.0
    best_epoch = 0
    best_state = None
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        generator = torch.Generator().manual_seed(seed * 10_000 + epoch)
        order = torch.randperm(len(train_events), generator=generator)
        for start in range(0, len(order), config.batch_size):
            indices = order[start : start + config.batch_size]
            events = train_events[indices].to(device=device, dtype=torch.float32)
            labels = train_labels[indices].to(device=device)
            events = _augment_events(
                events,
                dropout=config.event_dropout,
                maximum_shift=config.maximum_shift,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits, _ = model(events)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            mark_step(device)
        scheduler.step()
        accuracy, activity, _ = _measure(
            model, validation_events, validation_labels, config.batch_size, device
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_activity = activity
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
    sync(device)
    return {
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "best_validation_activity": best_activity,
        "best_state": best_state,
        "train_seconds": time.perf_counter() - started,
    }


def _measure(model, events, labels, batch_size: int, device) -> tuple[float, float, float]:
    model.eval()
    correct = 0
    activity_sum = 0.0
    batches = 0
    sync(device)
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(events), batch_size):
            batch = events[start : start + batch_size].to(device=device, dtype=torch.float32)
            target = labels[start : start + batch_size].to(device=device)
            logits, activity = model(batch)
            correct += int((logits.argmax(dim=1) == target).sum().item())
            activity_sum += float(activity.detach().item())
            batches += 1
            mark_step(device)
    sync(device)
    seconds = time.perf_counter() - started
    return correct / max(len(events), 1), activity_sum / max(batches, 1), seconds


def _augment_events(events, *, dropout: float, maximum_shift: int):
    if dropout > 0.0:
        events = events * (torch.rand_like(events) >= dropout)
    if maximum_shift > 0:
        shift_y = int(torch.randint(-maximum_shift, maximum_shift + 1, ()).item())
        shift_x = int(torch.randint(-maximum_shift, maximum_shift + 1, ()).item())
        padded = F.pad(events, (maximum_shift, maximum_shift, maximum_shift, maximum_shift))
        start_y = maximum_shift + shift_y
        start_x = maximum_shift + shift_x
        events = padded[..., start_y : start_y + 34, start_x : start_x + 34]
    return events


def _stratified_split(events, labels, *, fraction: float, seed: int):
    generator = torch.Generator().manual_seed(seed)
    train_indices = []
    validation_indices = []
    for label in torch.unique(labels, sorted=True):
        indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        order = indices[torch.randperm(len(indices), generator=generator)]
        validation_count = max(1, int(round(len(indices) * fraction)))
        validation_indices.append(order[:validation_count])
        train_indices.append(order[validation_count:])
    train_index = torch.cat(train_indices)
    validation_index = torch.cat(validation_indices)
    return events[train_index], labels[train_index], events[validation_index], labels[validation_index]


def _stratified_limit(events, labels, limit: int, seed: int):
    if limit <= 0 or limit >= len(events):
        return events, labels
    generator = torch.Generator().manual_seed(seed)
    selected = []
    per_class = max(1, limit // len(torch.unique(labels)))
    for label in torch.unique(labels, sorted=True):
        indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        selected.append(indices[torch.randperm(len(indices), generator=generator)[:per_class]])
    index = torch.cat(selected)[:limit]
    return events[index], labels[index]


def _load_split(config: NMNISTAccuracyConfig, *, train: bool, cache: pathlib.Path):
    split = "train" if train else "test"
    limit = config.train_samples if train else config.test_samples
    suffix = "all" if limit <= 0 else str(limit)
    cache_path = cache / (
        f"{split}_t{config.timesteps}_full34_d{config.duration_us}_n{suffix}_seed{config.data_seed}.pt"
    )
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        return payload["events"], payload["labels"]
    if not config.download:
        raise FileNotFoundError(f"N-MNIST accuracy cache missing: {cache_path}")
    try:
        import tonic
    except ImportError as error:  # pragma: no cover
        raise ImportError("Install N-MNIST support with `pip install tonic`.") from error
    dataset = tonic.datasets.NMNIST(save_to=str(pathlib.Path(config.data_root)), train=train)
    indices = _subset_indices(len(dataset), limit, config.data_seed + (0 if train else 1))
    encoded = torch.empty(
        (len(indices), config.timesteps, 2, 34, 34), dtype=torch.uint8
    )
    labels = torch.empty((len(indices),), dtype=torch.long)
    for output_index, dataset_index in enumerate(indices):
        raw_events, label = dataset[int(dataset_index)]
        encoded[output_index] = torch.from_numpy(encode_nmnist_full_resolution(
            raw_events, timesteps=config.timesteps, duration_us=config.duration_us
        ))
        labels[output_index] = int(label)
        if (output_index + 1) % 5000 == 0 or output_index + 1 == len(indices):
            print(
                f"[encode] split={split} samples={output_index + 1}/{len(indices)}",
                flush=True,
            )
    torch.save({"events": encoded, "labels": labels}, cache_path)
    return encoded, labels


def _subset_indices(length: int, limit: int, seed: int) -> list[int]:
    if limit <= 0 or limit >= length:
        return list(range(length))
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(length, generator=generator)[:limit].tolist()


def _parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _save_progress(path, config, dataset, screen_records, confirmation_records) -> None:
    if path is None:
        return
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({
        "config": asdict(config),
        "dataset": dataset,
        "screen_records": screen_records,
        "confirmation_records": confirmation_records,
    }, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def _load_progress(path, config) -> dict:
    source = pathlib.Path(path)
    if not source.exists():
        return {}
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("config") != json.loads(json.dumps(asdict(config))):
        raise ValueError("N-MNIST benchmark progress uses a different configuration")
    return payload


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validate_config(config: NMNISTAccuracyConfig) -> None:
    if config.sensor_size != 34:
        raise ValueError("the standard N-MNIST benchmark requires the native 34x34 sensor")
    if len(config.confirmation_seeds) != 3 or len(set(config.confirmation_seeds)) != 3:
        raise ValueError("confirmation requires exactly three unique seeds")
    if config.screen_seed in config.confirmation_seeds:
        raise ValueError("screen and confirmation seeds must be disjoint")
    if config.timesteps <= 0 or config.duration_us <= 0:
        raise ValueError("event dimensions must be positive")
    if config.screen_epochs <= 0 or config.confirmation_epochs <= 0:
        raise ValueError("epoch counts must be positive")
    if config.batch_size <= 0 or config.maximum_promoted_arms <= 0:
        raise ValueError("batch size and promotion count must be positive")
    if not 0.0 < config.validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5)")
    if not 0.0 <= config.event_dropout < 1.0 or config.maximum_shift < 0:
        raise ValueError("invalid augmentation settings")
    if not 0.0 < config.practical_accuracy <= config.stretch_accuracy <= 1.0:
        raise ValueError("invalid accuracy gates")


def _require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise ImportError("the N-MNIST accuracy benchmark requires PyTorch")
