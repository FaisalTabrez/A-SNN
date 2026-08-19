"""Preregistered Gen-20 spiking spatial-temporal N-MNIST translation."""

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

try:  # pragma: no cover - environment dependent
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None

from .nmnist_accuracy_benchmark import (
    ConvPLIFClassifier,
    SpatiotemporalCNN,
    _augment_events,
    _parameter_count,
    _stratified_limit,
    _stratified_split,
    estimate_nmnist_dense_macs,
    load_nmnist_accuracy_tensors,
)
from .runtime import device_kind, mark_step, resolve_device, seed_everything, sync
from .trainable_temporal_mnist import SurrogateSpike


GEN20_ARMS = (
    "spatiotemporal_cnn",
    "conv_plif",
    "multiscale_residual_plif",
    "distilled_multiscale_plif",
)
GEN20_NEW_SPIKING_ARMS = (
    "multiscale_residual_plif",
    "distilled_multiscale_plif",
)


@dataclass(frozen=True)
class Gen20Config:
    screen_seed: int = 220
    confirmation_seeds: tuple[int, ...] = (221, 222, 223)
    timesteps: int = 10
    duration_us: int = 300_000
    sensor_size: int = 34
    classes: int = 10
    screen_train_samples: int = 20_000
    train_samples: int = 0
    test_samples: int = 0
    screen_epochs: int = 6
    confirmation_epochs: int = 12
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    validation_fraction: float = 0.10
    data_seed: int = 2026
    data_root: str = "gen5_data/nmnist"
    download: bool = True
    minimum_screen_accuracy: float = 0.975
    minimum_activity: float = 0.01
    maximum_activity: float = 0.30
    maximum_promoted_arms: int = 2
    minimum_test_accuracy: float = 0.99
    maximum_dense_gap: float = 0.0075
    minimum_seed_accuracy: float = 0.987
    minimum_causal_margin: float = 0.01
    minimum_causal_seed_count: int = 2
    minimum_ops_reduction: float = 5.0
    event_dropout: float = 0.02
    maximum_shift: int = 2
    surrogate_slope: float = 10.0
    distillation_temperature: float = 2.0
    distillation_weight: float = 0.35
    feature_distillation_weight: float = 0.10


@dataclass
class Gen20Result:
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
        json_path = output / "gen20_spiking_spatiotemporal.json"
        screen_path = output / "gen20_spiking_spatiotemporal_screen.csv"
        records_path = output / "gen20_spiking_spatiotemporal_records.csv"
        summary_path = output / "gen20_spiking_spatiotemporal_summary.csv"
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
            plot_path = output / "gen20_spiking_spatiotemporal.png"
            plot_gen20(self, plot_path)
            paths["plot"] = str(plot_path)
        return paths


_ModuleBase = nn.Module if nn is not None else object


class Gen20SpatiotemporalTeacher(SpatiotemporalCNN):
    """Dense upper control exposing its penultimate representation."""

    def forward_features(self, events):
        encoded = self.features(events.transpose(1, 2))
        hidden = self.classifier[2](self.classifier[1](self.classifier[0](encoded)))
        return self.classifier[4](self.classifier[3](hidden)), hidden

    def forward(self, events):  # type: ignore[override]
        logits, _ = self.forward_features(events)
        return logits, events.new_zeros(())


class TimescaleLIF2d(_ModuleBase):
    """Parametric LIF cell initialized to a specified membrane timescale."""

    def __init__(
        self, channels: int, *, initial_beta: float, slope: float = 10.0
    ) -> None:
        _require_torch()
        super().__init__()
        normalized = min(max((initial_beta - 0.5) / 0.49, 1e-4), 1.0 - 1e-4)
        logit = math.log(normalized / (1.0 - normalized))
        self.beta_logit = nn.Parameter(torch.full((1, channels, 1, 1), logit))
        self.threshold_log = nn.Parameter(torch.zeros((1, channels, 1, 1)))
        self.slope = float(slope)

    def forward(self, current, membrane):  # type: ignore[override]
        beta = 0.5 + 0.49 * torch.sigmoid(self.beta_logit)
        threshold = 0.5 + F.softplus(self.threshold_log)
        pre_reset = beta * membrane + current
        spikes = SurrogateSpike.apply(pre_reset - threshold, self.slope)
        return spikes, pre_reset - spikes * threshold


class MultiTimescaleResidualPLIF(_ModuleBase):
    """Spatial stem plus fast/medium/slow LIF banks and spiking fusion state."""

    def __init__(self, classes: int = 10, *, surrogate_slope: float = 10.0) -> None:
        _require_torch()
        super().__init__()
        self.spatial_stem = nn.Sequential(
            nn.Conv2d(2, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
        )
        self.fast_lif = TimescaleLIF2d(48, initial_beta=0.65, slope=surrogate_slope)
        self.medium_lif = TimescaleLIF2d(48, initial_beta=0.82, slope=surrogate_slope)
        self.slow_lif = TimescaleLIF2d(48, initial_beta=0.94, slope=surrogate_slope)
        self.fusion = nn.Sequential(
            nn.Conv2d(48 * 6, 96, 1, bias=False),
            nn.BatchNorm2d(96),
        )
        self.output_lif = TimescaleLIF2d(96, initial_beta=0.88, slope=surrogate_slope)
        self.hidden = nn.Linear(96 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.15)
        self.readout = nn.Linear(256, classes)

    def forward_features(self, events, *, state_mode: str = "normal"):
        if state_mode not in {"normal", "removed"}:
            raise ValueError("state_mode must be 'normal' or 'removed'")
        batch = events.shape[0]
        bank_shape = (batch, 48, 9, 9)
        output_shape = (batch, 96, 9, 9)
        fast_mem = events.new_zeros(bank_shape)
        medium_mem = events.new_zeros(bank_shape)
        slow_mem = events.new_zeros(bank_shape)
        output_mem = events.new_zeros(output_shape)
        logits = events.new_zeros((batch, self.readout.out_features))
        hidden_sum = events.new_zeros((batch, self.hidden.out_features))
        activity_sum = events.new_zeros(())
        for step in range(events.shape[1]):
            if state_mode == "removed":
                fast_mem = torch.zeros_like(fast_mem)
                medium_mem = torch.zeros_like(medium_mem)
                slow_mem = torch.zeros_like(slow_mem)
                output_mem = torch.zeros_like(output_mem)
            current = self.spatial_stem(events[:, step])
            fast_spike, fast_mem = self.fast_lif(current, fast_mem)
            medium_spike, medium_mem = self.medium_lif(current, medium_mem)
            slow_spike, slow_mem = self.slow_lif(current, slow_mem)
            fused = self.fusion(torch.cat((
                fast_spike,
                medium_spike,
                slow_spike,
                torch.tanh(fast_mem),
                torch.tanh(medium_mem),
                torch.tanh(slow_mem),
            ), dim=1))
            output_spike, output_mem = self.output_lif(fused, output_mem)
            pooled = F.adaptive_avg_pool2d(output_spike, (4, 4)).flatten(1)
            hidden = F.silu(self.hidden(pooled))
            logits = logits + self.readout(self.dropout(hidden))
            hidden_sum = hidden_sum + hidden
            activity_sum = activity_sum + (
                fast_spike.mean()
                + medium_spike.mean()
                + slow_spike.mean()
                + output_spike.mean()
            ) / 4.0
        scale = float(events.shape[1])
        return logits / scale, activity_sum / scale, hidden_sum / scale

    def forward(self, events, *, state_mode: str = "normal"):  # type: ignore[override]
        logits, activity, _ = self.forward_features(events, state_mode=state_mode)
        return logits, activity


def available_gen20_arms() -> tuple[str, ...]:
    return GEN20_ARMS


def build_gen20_model(
    arm: str, *, classes: int = 10, surrogate_slope: float = 10.0
):
    if arm == "spatiotemporal_cnn":
        return Gen20SpatiotemporalTeacher(classes)
    if arm == "conv_plif":
        return ConvPLIFClassifier(classes, surrogate_slope=surrogate_slope)
    if arm in GEN20_NEW_SPIKING_ARMS:
        return MultiTimescaleResidualPLIF(classes, surrogate_slope=surrogate_slope)
    raise ValueError(f"unknown Gen-20 arm: {arm}")


def select_gen20_promoted_arms(
    screen_records: Iterable[dict], config: Gen20Config
) -> list[str]:
    eligible = [
        row for row in screen_records
        if row["arm"] in GEN20_NEW_SPIKING_ARMS
        and float(row["best_validation_accuracy"]) >= config.minimum_screen_accuracy
        and config.minimum_activity <= float(row["validation_activity"]) <= config.maximum_activity
        and bool(row.get("numerically_stable", True))
    ]
    eligible.sort(
        key=lambda row: (-float(row["best_validation_accuracy"]), str(row["arm"]))
    )
    return [str(row["arm"]) for row in eligible[: config.maximum_promoted_arms]]


def run_gen20(
    config: Gen20Config | None = None,
    *,
    device: str = "auto",
    progress_path: str | pathlib.Path | None = None,
) -> Gen20Result:
    _require_torch()
    cfg = config or Gen20Config()
    _validate_config(cfg)
    resolved = resolve_device(device)
    all_train, all_labels, test_events, test_labels, dataset = load_nmnist_accuracy_tensors(cfg)
    train_events, train_labels, validation_events, validation_labels = _stratified_split(
        all_train, all_labels, fraction=cfg.validation_fraction, seed=cfg.data_seed + 20_000
    )
    screen_events, screen_labels = _stratified_limit(
        train_events, train_labels, cfg.screen_train_samples, cfg.data_seed + 20_001
    )
    progress = _load_progress(progress_path, cfg)
    screen_records = list(progress.get("screen_records", []))
    confirmation_records = list(progress.get("confirmation_records", []))

    for arm in GEN20_ARMS:
        if any(row["arm"] == arm for row in screen_records):
            continue
        teacher = None
        if arm == "distilled_multiscale_plif":
            teacher = _load_teacher(cfg, progress_path, "screen", cfg.screen_seed, resolved)
        seed_everything(cfg.screen_seed, device=resolved)
        model = build_gen20_model(
            arm, classes=cfg.classes, surrogate_slope=cfg.surrogate_slope
        ).to(resolved)
        dense_macs, analog_macs = estimate_gen20_operations(
            model, cfg.timesteps, cfg.sensor_size
        )
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
            teacher=teacher,
        )
        checkpoint = _save_checkpoint(
            training["best_state"], progress_path, "screen", arm, cfg.screen_seed
        )
        screen_records.append({
            "arm": arm,
            "seed": cfg.screen_seed,
            "parameters": _parameter_count(model),
            "dense_macs_per_sample": dense_macs,
            "analog_dense_macs_per_sample": analog_macs,
            "best_epoch": training["best_epoch"],
            "best_validation_accuracy": training["best_validation_accuracy"],
            "validation_activity": training["best_validation_activity"],
            "numerically_stable": training["numerically_stable"],
            "train_seconds": training["train_seconds"],
            "checkpoint": checkpoint,
        })
        print(
            f"[screen] arm={arm} validation={training['best_validation_accuracy']:.4f} "
            f"activity={training['best_validation_activity']:.4f}",
            flush=True,
        )
        del model, teacher
        _save_progress(progress_path, cfg, dataset, screen_records, confirmation_records)

    promoted = select_gen20_promoted_arms(screen_records, cfg)
    if promoted:
        for seed in cfg.confirmation_seeds:
            if not _has_record(confirmation_records, "spatiotemporal_cnn", seed):
                seed_everything(seed, device=resolved)
                teacher = build_gen20_model("spatiotemporal_cnn", classes=cfg.classes).to(resolved)
                training = _train_model(
                    teacher,
                    train_events,
                    train_labels,
                    validation_events,
                    validation_labels,
                    epochs=cfg.confirmation_epochs,
                    config=cfg,
                    seed=seed,
                    device=resolved,
                )
                teacher.load_state_dict(training["best_state"])
                checkpoint = _save_checkpoint(
                    training["best_state"], progress_path, "confirmation", "spatiotemporal_cnn", seed
                )
                accuracy, activity, seconds = _measure(
                    teacher, test_events, test_labels, cfg.batch_size, resolved
                )
                dense_macs, analog_macs = estimate_gen20_operations(
                    teacher, cfg.timesteps, cfg.sensor_size
                )
                confirmation_records.append(_record(
                    "spatiotemporal_cnn", seed, teacher, training, accuracy, activity,
                    seconds, len(test_events), dense_macs, analog_macs, checkpoint,
                ))
                print(f"[confirm] arm=spatiotemporal_cnn seed={seed} test={accuracy:.4f}", flush=True)
                del teacher
                _save_progress(progress_path, cfg, dataset, screen_records, confirmation_records)

        for arm in promoted:
            for seed in cfg.confirmation_seeds:
                if _has_record(confirmation_records, arm, seed):
                    continue
                teacher = None
                if arm == "distilled_multiscale_plif":
                    teacher = _load_teacher(cfg, progress_path, "confirmation", seed, resolved)
                seed_everything(seed, device=resolved)
                model = build_gen20_model(
                    arm, classes=cfg.classes, surrogate_slope=cfg.surrogate_slope
                ).to(resolved)
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
                    teacher=teacher,
                )
                model.load_state_dict(training["best_state"])
                checkpoint = _save_checkpoint(
                    training["best_state"], progress_path, "confirmation", arm, seed
                )
                accuracy, activity, seconds = _measure(
                    model, test_events, test_labels, cfg.batch_size, resolved
                )
                removed_accuracy, _, _ = _measure(
                    model, test_events, test_labels, cfg.batch_size, resolved,
                    state_mode="removed",
                )
                shuffled_accuracy, _, _ = _measure(
                    model, test_events, test_labels, cfg.batch_size, resolved,
                    shuffle_seed=seed + 50_000,
                )
                dense_macs, analog_macs = estimate_gen20_operations(
                    model, cfg.timesteps, cfg.sensor_size
                )
                confirmation_records.append(_record(
                    arm, seed, model, training, accuracy, activity, seconds,
                    len(test_events), dense_macs, analog_macs, checkpoint,
                    state_removed_accuracy=removed_accuracy,
                    time_shuffled_accuracy=shuffled_accuracy,
                ))
                print(
                    f"[confirm] arm={arm} seed={seed} test={accuracy:.4f} "
                    f"removed={removed_accuracy:.4f} shuffled={shuffled_accuracy:.4f}",
                    flush=True,
                )
                del model, teacher
                _save_progress(progress_path, cfg, dataset, screen_records, confirmation_records)

    summary = summarize_gen20(confirmation_records, cfg)
    decision = decide_gen20(summary, promoted, cfg)
    dataset.update({
        "training_split_samples": int(train_events.shape[0]),
        "validation_samples": int(validation_events.shape[0]),
        "screen_training_samples": int(screen_events.shape[0]),
        "test_samples": int(test_events.shape[0]),
    })
    return Gen20Result(
        config=asdict(cfg),
        device=device_kind(resolved),
        dataset=dataset,
        screen_records=screen_records,
        promoted_arms=promoted,
        confirmation_records=confirmation_records,
        summary=summary,
        decision=decision,
    )


def summarize_gen20(records: Iterable[dict], config: Gen20Config) -> list[dict]:
    rows = list(records)
    teacher_rows = [row for row in rows if row["arm"] == "spatiotemporal_cnn"]
    teacher_ops = (
        statistics.fmean(float(row["activity_scaled_ops_proxy"]) for row in teacher_rows)
        if teacher_rows else 0.0
    )
    output = []
    for arm in ("spatiotemporal_cnn",) + GEN20_NEW_SPIKING_ARMS:
        group = [row for row in rows if row["arm"] == arm]
        if not group:
            continue
        accuracies = [float(row["test_accuracy"]) for row in group]
        is_spiking = arm in GEN20_NEW_SPIKING_ARMS
        state_margins = [float(row["state_contribution"]) for row in group] if is_spiking else []
        order_margins = [float(row["temporal_order_contribution"]) for row in group] if is_spiking else []
        proxy = statistics.fmean(float(row["activity_scaled_ops_proxy"]) for row in group)
        output.append({
            "arm": arm,
            "runs": len(group),
            "mean_test_accuracy": statistics.fmean(accuracies),
            "std_test_accuracy": statistics.pstdev(accuracies),
            "minimum_test_accuracy": min(accuracies),
            "mean_test_activity": statistics.fmean(float(row["test_activity"]) for row in group),
            "mean_state_contribution": statistics.fmean(state_margins) if state_margins else None,
            "state_contribution_seed_count": sum(
                margin >= config.minimum_causal_margin for margin in state_margins
            ),
            "mean_temporal_order_contribution": statistics.fmean(order_margins) if order_margins else None,
            "temporal_order_seed_count": sum(
                margin >= config.minimum_causal_margin for margin in order_margins
            ),
            "mean_test_examples_per_second": statistics.fmean(
                float(row["test_examples_per_second"]) for row in group
            ),
            "parameters": int(group[0]["parameters"]),
            "dense_macs_per_sample": int(group[0]["dense_macs_per_sample"]),
            "analog_dense_macs_per_sample": int(group[0]["analog_dense_macs_per_sample"]),
            "mean_activity_scaled_ops_proxy": proxy,
            "ops_reduction_vs_dense_teacher": teacher_ops / proxy if is_spiking and proxy > 0 else 1.0,
        })
    return output


def decide_gen20(summary: list[dict], promoted: list[str], config: Gen20Config) -> dict:
    teacher = next((row for row in summary if row["arm"] == "spatiotemporal_cnn"), None)
    candidates = [row for row in summary if row["arm"] in GEN20_NEW_SPIKING_ARMS]
    if not promoted or teacher is None or not candidates:
        return {
            "status": "stop",
            "reason": "no_new_spiking_arm_passed_screen" if not promoted else "confirmation_incomplete",
            "qualified_arms": [],
            "next_milestone": "evidence_synthesis",
        }
    qualified = []
    gates_by_arm = {}
    for row in candidates:
        gates = {
            "accuracy_gate": float(row["mean_test_accuracy"]) >= config.minimum_test_accuracy,
            "dense_gap_gate": (
                float(teacher["mean_test_accuracy"]) - float(row["mean_test_accuracy"])
                <= config.maximum_dense_gap
            ),
            "seed_floor_gate": float(row["minimum_test_accuracy"]) >= config.minimum_seed_accuracy,
            "activity_gate": (
                config.minimum_activity <= float(row["mean_test_activity"]) <= config.maximum_activity
            ),
            "state_gate": (
                float(row["mean_state_contribution"]) >= config.minimum_causal_margin
                and int(row["state_contribution_seed_count"]) >= config.minimum_causal_seed_count
            ),
            "time_order_gate": (
                float(row["mean_temporal_order_contribution"]) >= config.minimum_causal_margin
                and int(row["temporal_order_seed_count"]) >= config.minimum_causal_seed_count
            ),
            "ops_gate": float(row["ops_reduction_vs_dense_teacher"]) >= config.minimum_ops_reduction,
        }
        gates_by_arm[str(row["arm"])] = gates
        if all(gates.values()):
            qualified.append(str(row["arm"]))
    return {
        "status": "pass" if qualified else "stop",
        "qualified_arms": qualified,
        "gates_by_arm": gates_by_arm,
        "best_spiking_arm": max(candidates, key=lambda row: float(row["mean_test_accuracy"]))["arm"],
        "next_milestone": "plasticity_continual_learning" if qualified else "evidence_synthesis",
        "energy_claim_authorized": False,
    }


def estimate_gen20_operations(model, timesteps: int, sensor_size: int = 34) -> tuple[int, int]:
    dense = estimate_nmnist_dense_macs(model, timesteps, sensor_size)
    if not isinstance(model, MultiTimescaleResidualPLIF):
        return dense, dense
    analog_modules = {
        module for module in model.spatial_stem.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    }
    analog = 0

    def hook(module, inputs, output):
        nonlocal analog
        value = output[0] if isinstance(output, tuple) else output
        if isinstance(module, nn.Conv2d):
            analog += int(
                value.numel() * math.prod(module.kernel_size)
                * module.in_channels / module.groups
            )
        elif isinstance(module, nn.Linear):
            analog += int(value.numel() * module.in_features)

    handles = [module.register_forward_hook(hook) for module in analog_modules]
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros((1, timesteps, 2, sensor_size, sensor_size), device=device))
    for handle in handles:
        handle.remove()
    model.train(was_training)
    return dense, analog


def bundle_gen20_artifacts(paths: dict[str, str], output_dir: str | pathlib.Path) -> dict[str, str]:
    output = pathlib.Path(output_dir)
    files = [pathlib.Path(path) for path in paths.values() if pathlib.Path(path).is_file()]
    if not files:
        raise FileNotFoundError("no Gen-20 artifacts exist to bundle")
    rows = [{
        "filename": source.name,
        "bytes": source.stat().st_size,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    } for source in files]
    manifest = output / "gen20_spiking_spatiotemporal_manifest.json"
    manifest.write_text(json.dumps({
        "schema": "ammc-gen20-spiking-spatiotemporal-v1",
        "artifacts": rows,
    }, indent=2) + "\n", encoding="utf-8")
    bundle = output / "gen20_spiking_spatiotemporal_bundle.zip"
    temporary = bundle.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for source in files:
            archive.write(source, arcname=source.name)
        archive.write(manifest, arcname=manifest.name)
    temporary.replace(bundle)
    return {"manifest": str(manifest), "bundle": str(bundle)}


def gen20_plot_series(result: Gen20Result) -> dict:
    """Return plot-ready confirmation data, or screen data after an early stop."""
    if result.summary:
        return {
            "stage": "confirmation",
            "labels": [row["arm"].replace("_", "\n") for row in result.summary],
            "accuracy": [
                100.0 * float(row["mean_test_accuracy"]) for row in result.summary
            ],
            "activity": [
                100.0 * float(row["mean_test_activity"]) for row in result.summary
            ],
            "reduction": [
                float(row["ops_reduction_vs_dense_teacher"]) for row in result.summary
            ],
            "accuracy_gate": 100.0 * float(result.config["minimum_test_accuracy"]),
            "accuracy_label": "Test accuracy (%)",
        }

    rows = result.screen_records
    teacher = next(
        (row for row in rows if row["arm"] == "spatiotemporal_cnn"), None
    )
    teacher_ops = float(teacher["dense_macs_per_sample"]) if teacher else 0.0
    proxies = []
    for row in rows:
        dense = float(row["dense_macs_per_sample"])
        analog = float(row["analog_dense_macs_per_sample"])
        activity = float(row["validation_activity"])
        proxy = (
            analog + (dense - analog) * activity
            if row["arm"] in GEN20_NEW_SPIKING_ARMS
            else dense
        )
        proxies.append(proxy)
    return {
        "stage": "screen",
        "labels": [row["arm"].replace("_", "\n") for row in rows],
        "accuracy": [100.0 * float(row["best_validation_accuracy"]) for row in rows],
        "activity": [100.0 * float(row["validation_activity"]) for row in rows],
        "reduction": [teacher_ops / proxy if proxy > 0 else 0.0 for proxy in proxies],
        "accuracy_gate": 100.0 * float(result.config["minimum_screen_accuracy"]),
        "accuracy_label": "Validation accuracy (%)",
    }


def plot_gen20(result: Gen20Result, path: str | pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    series = gen20_plot_series(result)
    labels = series["labels"]
    accuracy = series["accuracy"]
    activity = series["activity"]
    reduction = series["reduction"]
    figure, axes = plt.subplots(3, 1, figsize=(12, 14), constrained_layout=True)
    axes[0].bar(labels, accuracy, color="#35b4f2")
    axes[0].axhline(
        series["accuracy_gate"], color="#d88935", linestyle="--", label="accuracy gate"
    )
    axes[0].set_ylabel(series["accuracy_label"])
    axes[0].legend()
    axes[1].bar(labels, activity, color="#8b6fd6")
    axes[1].axhspan(1.0, 30.0, color="#6ac68d", alpha=0.15)
    axes[1].set_ylabel("Spike activity (%)")
    axes[2].bar(labels, reduction, color="#ffb31a")
    axes[2].axhline(5.0, color="#167d55", linestyle=":", label="proxy gate")
    axes[2].set_ylabel("Ops reduction vs dense teacher")
    axes[2].legend()
    axes[0].set_title(
        "AMMC Gen-20 spiking spatial-temporal translation "
        f"({series['stage']})"
    )
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _train_model(
    model, train_events, train_labels, validation_events, validation_labels, *,
    epochs: int, config: Gen20Config, seed: int, device, teacher=None,
) -> dict:
    seed_everything(seed, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = getattr(device, "type", None) == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if teacher is not None:
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    best_accuracy = -1.0
    best_activity = 0.0
    best_epoch = 0
    best_state = None
    stable = True
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(
            len(train_events), generator=torch.Generator().manual_seed(seed * 10_000 + epoch)
        )
        for start in range(0, len(order), config.batch_size):
            indices = order[start : start + config.batch_size]
            events = train_events[indices].to(device=device, dtype=torch.float32)
            labels = train_labels[indices].to(device=device)
            events = _augment_events(
                events, dropout=config.event_dropout, maximum_shift=config.maximum_shift
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                if teacher is None:
                    logits, _ = model(events)
                    loss = F.cross_entropy(logits, labels)
                else:
                    logits, _, features = model.forward_features(events)
                    with torch.no_grad():
                        teacher_logits, teacher_features = teacher.forward_features(events)
                    temperature = config.distillation_temperature
                    distillation = F.kl_div(
                        F.log_softmax(logits / temperature, dim=1),
                        F.softmax(teacher_logits / temperature, dim=1),
                        reduction="batchmean",
                    ) * (temperature * temperature)
                    feature_loss = F.mse_loss(
                        F.normalize(features.float(), dim=1),
                        F.normalize(teacher_features.float(), dim=1),
                    )
                    loss = (
                        (1.0 - config.distillation_weight) * F.cross_entropy(logits, labels)
                        + config.distillation_weight * distillation
                        + config.feature_distillation_weight * feature_loss
                    )
            if not bool(torch.isfinite(loss).item()):
                stable = False
                break
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            mark_step(device)
        if not stable:
            break
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
    if best_state is None:
        best_state = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
    return {
        "best_epoch": best_epoch,
        "best_validation_accuracy": max(best_accuracy, 0.0),
        "best_validation_activity": best_activity,
        "best_state": best_state,
        "numerically_stable": stable,
        "train_seconds": time.perf_counter() - started,
    }


def _measure(
    model, events, labels, batch_size: int, device, *,
    state_mode: str = "normal", shuffle_seed: int | None = None,
) -> tuple[float, float, float]:
    model.eval()
    correct = 0
    activity_sum = 0.0
    batches = 0
    generator = torch.Generator().manual_seed(shuffle_seed) if shuffle_seed is not None else None
    sync(device)
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(events), batch_size):
            batch = events[start : start + batch_size]
            if generator is not None:
                batch = _shuffle_time(batch, generator)
            batch = batch.to(device=device, dtype=torch.float32)
            target = labels[start : start + batch_size].to(device=device)
            if isinstance(model, MultiTimescaleResidualPLIF):
                logits, activity = model(batch, state_mode=state_mode)
            else:
                logits, activity = model(batch)
            correct += int((logits.argmax(dim=1) == target).sum().item())
            activity_sum += float(activity.detach().item())
            batches += 1
            mark_step(device)
    sync(device)
    seconds = time.perf_counter() - started
    return correct / max(len(events), 1), activity_sum / max(batches, 1), seconds


def _shuffle_time(events, generator):
    return torch.stack([
        sample[torch.randperm(sample.shape[0], generator=generator)] for sample in events
    ])


def _record(
    arm, seed, model, training, accuracy, activity, seconds, sample_count, dense_macs,
    analog_macs, checkpoint, *, state_removed_accuracy=None,
    time_shuffled_accuracy=None,
):
    is_spiking = isinstance(model, MultiTimescaleResidualPLIF)
    proxy = analog_macs + (dense_macs - analog_macs) * activity if is_spiking else dense_macs
    return {
        "arm": arm,
        "seed": int(seed),
        "parameters": _parameter_count(model),
        "dense_macs_per_sample": dense_macs,
        "analog_dense_macs_per_sample": analog_macs,
        "activity_scaled_ops_proxy": proxy,
        "best_epoch": training["best_epoch"],
        "best_validation_accuracy": training["best_validation_accuracy"],
        "test_accuracy": accuracy,
        "test_activity": activity,
        "state_removed_accuracy": state_removed_accuracy,
        "time_shuffled_accuracy": time_shuffled_accuracy,
        "state_contribution": (
            accuracy - state_removed_accuracy if state_removed_accuracy is not None else None
        ),
        "temporal_order_contribution": (
            accuracy - time_shuffled_accuracy if time_shuffled_accuracy is not None else None
        ),
        "test_examples_per_second": sample_count / max(seconds, 1e-12),
        "test_seconds": seconds,
        "train_seconds": training["train_seconds"],
        "checkpoint": checkpoint,
    }
def _has_record(records, arm: str, seed: int) -> bool:
    return any(row["arm"] == arm and int(row["seed"]) == int(seed) for row in records)


def _checkpoint_path(progress_path, stage: str, arm: str, seed: int) -> pathlib.Path:
    if progress_path is None:
        root = pathlib.Path("gen5_outputs/gen20_checkpoints")
    else:
        root = pathlib.Path(progress_path).parent / "gen20_checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{stage}_{arm}_seed{seed}.pt"


def _save_checkpoint(state, progress_path, stage: str, arm: str, seed: int) -> str:
    path = _checkpoint_path(progress_path, stage, arm, seed)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    return str(path)


def _load_teacher(config, progress_path, stage: str, seed: int, device):
    path = _checkpoint_path(progress_path, stage, "spatiotemporal_cnn", seed)
    if not path.exists():
        raise FileNotFoundError(
            f"Gen-20 teacher checkpoint missing: {path}. Restore checkpoints or restart this stage."
        )
    teacher = Gen20SpatiotemporalTeacher(config.classes).to(device)
    teacher.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    teacher.eval()
    return teacher


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
    if path is None or not pathlib.Path(path).exists():
        return {}
    payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if payload.get("config") != json.loads(json.dumps(asdict(config))):
        raise ValueError("Gen-20 progress uses a different configuration")
    return payload


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _validate_config(config: Gen20Config) -> None:
    if config.sensor_size != 34 or config.timesteps != 10 or config.duration_us != 300_000:
        raise ValueError("Gen-20 freezes native 34x34 events into 10 bins over 300 ms")
    if config.screen_seed != 220 or config.confirmation_seeds != (221, 222, 223):
        raise ValueError("Gen-20 screen and confirmation seeds are frozen")
    if config.screen_epochs != 6 or config.confirmation_epochs != 12:
        raise ValueError("Gen-20 epoch budgets are frozen")
    if config.maximum_promoted_arms != 2:
        raise ValueError("Gen-20 promotes at most two new spiking arms")
    if not 0.0 < config.validation_fraction < 0.5 or config.batch_size <= 0:
        raise ValueError("invalid validation fraction or batch size")
    if not 0.0 < config.minimum_activity < config.maximum_activity < 1.0:
        raise ValueError("invalid activity gate")


def _require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise ImportError("Gen-20 requires PyTorch")
