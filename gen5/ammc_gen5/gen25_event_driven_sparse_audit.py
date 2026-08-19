"""Gen-25 event-driven sparse input-operator audit on SSC."""

from __future__ import annotations

import copy
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import pathlib
import statistics
import time
import zipfile

from .event_mnist import nn, torch
from .gen24_compiled_residual_state import _benchmark_callable
from .runtime import device_kind, resolve_device, seed_everything, sync
from .shd_benchmark import SHDConfig
from .shd_state_placement_diagnostic import (
    ResidualTemporalConvStateClassifier,
    _multiscale_features,
    matched_temporal_conv_residual_channels,
)
from .ssc_benchmark import load_ssc_tensors


GEN25_RUNTIMES = ("compiled_dense", "event_sparse_hybrid")


@dataclass(frozen=True)
class Gen25Config:
    seeds: tuple[int, ...] = (631, 632, 633)
    input_neurons: int = 700
    classes: int = 35
    timesteps: int = 64
    duration_seconds: float = 1.0
    data_root: str = "gen5_data/ssc"
    download: bool = True
    test_samples: int = 256
    batch_sizes: tuple[int, ...] = (1, 32, 256)
    density_batch_size: int = 32
    synthetic_densities: tuple[float, ...] = (0.005, 0.01, 0.05, 0.10)
    target_parameters: int = 133_631
    temporal_levels: tuple[int, ...] = (1, 2, 4, 8)
    kernel_size: int = 5
    surrogate_slope: float = 10.0
    compile_mode: str = "reduce-overhead"
    warmup_iterations: int = 3
    measurement_iterations: int = 10
    measurement_repeats: int = 3
    maximum_logit_difference: float = 1e-4
    minimum_prediction_agreement: float = 1.0
    minimum_real_sparse_speed_ratio: float = 1.0
    low_density_ceiling: float = 0.01


@dataclass
class Gen25Result:
    config: dict
    device: str
    dataset: dict
    architecture: dict
    records: list[dict]
    summary: list[dict]
    decision: dict

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "gen25_event_driven_sparse_audit.json"
        records_path = output / "gen25_event_driven_sparse_audit_records.csv"
        summary_path = output / "gen25_event_driven_sparse_audit_summary.csv"
        json_path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        _write_csv(records_path, self.records)
        _write_csv(summary_path, self.summary)
        paths = {
            "json": str(json_path),
            "records_csv": str(records_path),
            "summary_csv": str(summary_path),
        }
        if plot:
            plot_path = output / "gen25_event_driven_sparse_audit.png"
            plot_gen25(self, plot_path)
            paths["plot"] = str(plot_path)
        return paths


class ResidualLIFStateHead(nn.Module):
    """Phase-49 residual-LIF computation after the temporal input operator."""

    def __init__(self, source: ResidualTemporalConvStateClassifier) -> None:
        super().__init__()
        self.channels = source.channels
        self.temporal_levels = source.temporal_levels
        self.surrogate_slope = source.surrogate_slope
        self.leak_logit = nn.Parameter(source.leak_logit.detach().clone(), requires_grad=False)
        self.threshold_raw = nn.Parameter(source.threshold_raw.detach().clone(), requires_grad=False)
        self.classifier = copy.deepcopy(source.classifier)
        for parameter in self.classifier.parameters():
            parameter.requires_grad_(False)

    def forward(self, currents):  # type: ignore[override]
        direct_trace = torch.relu(currents)
        leak = torch.sigmoid(self.leak_logit)
        threshold = torch.nn.functional.softplus(self.threshold_raw).clamp_min(1e-3)
        membrane = currents.new_zeros((currents.shape[0], self.channels))
        states = []
        for step in range(currents.shape[1]):
            pre_reset = leak * membrane + currents[:, step]
            # Inference uses the exact forward value of SurrogateSpike.
            state = (pre_reset >= threshold).to(currents.dtype)
            membrane = pre_reset - state * threshold
            states.append(state)
        state_trace = torch.stack(states, dim=1)
        features = _multiscale_features(direct_trace, self.temporal_levels)
        features.extend(_multiscale_features(state_trace, self.temporal_levels))
        features.append(membrane / threshold)
        return self.classifier(torch.cat(features, dim=1))


class DenseResidualPipeline(nn.Module):
    def __init__(self, source: ResidualTemporalConvStateClassifier) -> None:
        super().__init__()
        self.temporal = copy.deepcopy(source.temporal)
        self.head = ResidualLIFStateHead(source)
        for parameter in self.temporal.parameters():
            parameter.requires_grad_(False)

    def forward(self, events):  # type: ignore[override]
        currents = self.temporal(events.to(torch.float32).transpose(1, 2)).transpose(1, 2)
        return self.head(currents)


class SparseHybridPipeline:
    """Conservative dense-cache-to-COO operator plus compiled state head."""

    def __init__(self, temporal, compiled_head) -> None:
        self.temporal = temporal
        self.compiled_head = compiled_head

    def __call__(self, events):
        return self.compiled_head(sparse_temporal_currents(events, self.temporal))


def sparse_temporal_currents(events, temporal):
    """Compute Conv1d currents by routing only nonzero input events.

    Dense-to-COO discovery is deliberately included in the operator cost. A
    future sensor-native event stream may avoid it, but Gen-25 does not assume
    that optimization.
    """

    if events.ndim != 3:
        raise ValueError("events must have shape [batch, time, input_neurons]")
    batch, timesteps, input_neurons = events.shape
    weight = temporal.weight
    if int(weight.shape[1]) != int(input_neurons):
        raise ValueError("event input width does not match temporal convolution")
    channels, _, kernel_size = weight.shape
    padding = kernel_size // 2
    nonzero = torch.nonzero(events, as_tuple=False)
    currents = events.new_zeros((batch * timesteps, channels), dtype=torch.float32)
    if temporal.bias is not None:
        currents.add_(temporal.bias.to(torch.float32))
    if nonzero.numel() == 0:
        return currents.reshape(batch, timesteps, channels)
    values = events[nonzero[:, 0], nonzero[:, 1], nonzero[:, 2]].to(torch.float32)
    for kernel_index in range(kernel_size):
        output_time = nonzero[:, 1] - kernel_index + padding
        valid = (output_time >= 0) & (output_time < timesteps)
        selected = nonzero[valid]
        rows = selected[:, 0] * timesteps + output_time[valid]
        columns = selected[:, 2]
        sparse_input = torch.sparse_coo_tensor(
            torch.stack((rows, columns)),
            values[valid],
            size=(batch * timesteps, input_neurons),
            device=events.device,
        ).coalesce()
        currents.add_(torch.sparse.mm(sparse_input, weight[:, :, kernel_index].transpose(0, 1)))
    return currents.reshape(batch, timesteps, channels)


def run_gen25(
    config: Gen25Config = Gen25Config(),
    *,
    device="auto",
    dataset=None,
) -> Gen25Result:
    _validate_config(config)
    if torch is None or not hasattr(torch, "compile"):
        raise RuntimeError("Gen-25 requires torch.compile")
    resolved = resolve_device(device)
    if device_kind(resolved) != "cuda":
        raise ValueError("Gen-25 is a CUDA systems benchmark; pass --device cuda")
    test_events, test_labels = dataset if dataset is not None else _load_test_data(config)
    if int(test_events.shape[0]) < max(config.batch_sizes):
        raise ValueError("test_samples must cover the largest batch size")
    channels, parameters = matched_temporal_conv_residual_channels(
        config.input_neurons,
        config.classes,
        config.target_parameters,
        kernel_size=config.kernel_size,
        temporal_levels=config.temporal_levels,
    )
    architecture = {
        "model": "Phase-49 residual LIF",
        "channels": channels,
        "trainable_parameters": parameters,
        "dense_control": "compiled Conv1d plus compiled residual-LIF head",
        "sparse_candidate": "COO event routing plus compiled residual-LIF head",
        "conversion_cost_included": True,
        "weights": "deterministically seeded and frozen; no accuracy claim",
    }
    records = []
    for seed in config.seeds:
        seed_everything(seed, device=resolved)
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()
        source = ResidualTemporalConvStateClassifier(
            _model_config(config),
            channels=channels,
            kernel_size=config.kernel_size,
            temporal_levels=config.temporal_levels,
            dynamics="lif",
            surrogate_slope=config.surrogate_slope,
        ).to(resolved).eval()
        for parameter in source.parameters():
            parameter.requires_grad_(False)
        for batch_size in config.batch_sizes:
            real_batch = test_events[:batch_size].to(resolved)
            compiled_dense, dense_compile_seconds, dense_error = _compile_module(
                DenseResidualPipeline(source).to(resolved).eval(), real_batch, resolved, config.compile_mode
            )
            real_currents = sparse_temporal_currents(real_batch, source.temporal)
            compiled_head, head_compile_seconds, head_error = _compile_module(
                ResidualLIFStateHead(source).to(resolved).eval(), real_currents, resolved, config.compile_mode
            )
            sparse_pipeline = SparseHybridPipeline(source.temporal, compiled_head) if compiled_head else None
            workloads = [("real_ssc", real_batch, float((real_batch != 0).to(torch.float32).mean().item()))]
            if batch_size == config.density_batch_size:
                for density in config.synthetic_densities:
                    workloads.append((
                        f"synthetic_{density:.4f}",
                        _fixed_density_events(batch_size, config, density, seed=seed, device=resolved),
                        float(density),
                    ))
            for workload, batch, density in workloads:
                records.extend(_measure_workload(
                    seed,
                    workload,
                    density,
                    batch,
                    compiled_dense,
                    sparse_pipeline,
                    source.temporal,
                    dense_compile_seconds,
                    head_compile_seconds,
                    dense_error,
                    head_error,
                    config,
                    resolved,
                ))
    summary = summarize_gen25(records)
    return Gen25Result(
        config=asdict(config),
        device=device_kind(resolved),
        dataset={
            "name": "Spiking Speech Commands",
            "test_samples": int(test_events.shape[0]),
            "timesteps": int(test_events.shape[1]),
            "input_neurons": int(test_events.shape[2]),
            "labels_loaded_for_dataset_identity_only": int(test_labels.shape[0]),
        },
        architecture=architecture,
        records=records,
        summary=summary,
        decision=decide_gen25(records, config),
    )


def _compile_module(module, example, device, mode):
    try:
        compiled = torch.compile(module, mode=mode)
        started = time.perf_counter()
        with torch.inference_mode():
            compiled(example)
        sync(device)
        return compiled, float(time.perf_counter() - started), None
    except Exception as error:
        return None, 0.0, f"{type(error).__name__}: {error}"


def _measure_workload(
    seed,
    workload,
    density,
    batch,
    compiled_dense,
    sparse_pipeline,
    temporal,
    dense_compile_seconds,
    head_compile_seconds,
    dense_error,
    head_error,
    config,
    device,
):
    batch_size = int(batch.shape[0])
    active_events = int((batch != 0).sum().item())
    common = {
        "seed": int(seed),
        "workload": workload,
        "batch_size": batch_size,
        "event_density": float(density),
        "active_events": active_events,
    }
    if compiled_dense is None:
        dense_metrics = _failed_metrics(dense_error)
    else:
        dense_metrics = _benchmark_callable(
            compiled_dense,
            batch,
            device,
            warmup_iterations=config.warmup_iterations,
            measurement_iterations=config.measurement_iterations,
            measurement_repeats=config.measurement_repeats,
        )
        dense_metrics["compile_active"] = True
        dense_metrics["compile_seconds"] = dense_compile_seconds
    dense_row = {**common, "runtime": "compiled_dense", **dense_metrics}
    dense_row.update({
        "maximum_logit_difference": 0.0,
        "prediction_agreement": 1.0,
        "speed_ratio_vs_dense": 1.0,
        "sparse_operator_milliseconds": 0.0,
    })

    if sparse_pipeline is None or compiled_dense is None:
        sparse_metrics = _failed_metrics(head_error or dense_error)
        difference = 1e30
        agreement = 0.0
        operator_ms = 0.0
    else:
        with torch.inference_mode():
            dense_logits = compiled_dense(batch)
            sparse_logits = sparse_pipeline(batch)
        sync(device)
        difference = float((dense_logits - sparse_logits).abs().max().item())
        agreement = float(
            (dense_logits.argmax(dim=1) == sparse_logits.argmax(dim=1)).to(torch.float32).mean().item()
        )
        sparse_metrics = _benchmark_callable(
            sparse_pipeline,
            batch,
            device,
            warmup_iterations=config.warmup_iterations,
            measurement_iterations=config.measurement_iterations,
            measurement_repeats=config.measurement_repeats,
        )
        sparse_metrics["compile_active"] = True
        sparse_metrics["compile_seconds"] = head_compile_seconds
        operator_metrics = _benchmark_callable(
            lambda value: sparse_temporal_currents(value, temporal),
            batch,
            device,
            warmup_iterations=1,
            measurement_iterations=max(1, config.measurement_iterations // 2),
            measurement_repeats=config.measurement_repeats,
        )
        operator_ms = float(operator_metrics["milliseconds_per_batch"])
    sparse_row = {**common, "runtime": "event_sparse_hybrid", **sparse_metrics}
    sparse_row.update({
        "maximum_logit_difference": difference,
        "prediction_agreement": agreement,
        "speed_ratio_vs_dense": float(
            sparse_metrics["examples_per_second"] / max(dense_metrics["examples_per_second"], 1e-12)
        ),
        "sparse_operator_milliseconds": operator_ms,
    })
    return [dense_row, sparse_row]


def _failed_metrics(error):
    return {
        "seconds": 0.0,
        "examples_per_second": 0.0,
        "milliseconds_per_batch": 0.0,
        "compile_seconds": 0.0,
        "compile_active": False,
        "compile_error": error,
        "cuda_peak_memory_mb": 0.0,
    }


def summarize_gen25(records):
    summary = []
    keys = sorted({(row["workload"], int(row["batch_size"]), row["runtime"]) for row in records})
    for workload, batch_size, runtime in keys:
        group = [
            row for row in records
            if row["workload"] == workload
            and int(row["batch_size"]) == batch_size
            and row["runtime"] == runtime
        ]
        summary.append({
            "workload": workload,
            "batch_size": batch_size,
            "runtime": runtime,
            "seeds": len(group),
            "mean_event_density": statistics.fmean(float(row["event_density"]) for row in group),
            "mean_examples_per_second": statistics.fmean(float(row["examples_per_second"]) for row in group),
            "std_examples_per_second": _population_std(float(row["examples_per_second"]) for row in group),
            "mean_milliseconds_per_batch": statistics.fmean(float(row["milliseconds_per_batch"]) for row in group),
            "mean_speed_ratio_vs_dense": statistics.fmean(float(row["speed_ratio_vs_dense"]) for row in group),
            "mean_sparse_operator_milliseconds": statistics.fmean(float(row["sparse_operator_milliseconds"]) for row in group),
            "maximum_logit_difference": max(float(row["maximum_logit_difference"]) for row in group),
            "minimum_prediction_agreement": min(float(row["prediction_agreement"]) for row in group),
            "compile_successes": sum(bool(row["compile_active"]) for row in group),
            "maximum_cuda_peak_memory_mb": max(float(row["cuda_peak_memory_mb"]) for row in group),
        })
    return summary


def decide_gen25(records, config):
    sparse = [row for row in records if row["runtime"] == "event_sparse_hybrid"]
    equivalence = bool(sparse) and all(
        row["compile_active"]
        and float(row["maximum_logit_difference"]) <= config.maximum_logit_difference
        and float(row["prediction_agreement"]) >= config.minimum_prediction_agreement
        for row in sparse
    )
    primary = [
        row for row in sparse
        if row["workload"] == "real_ssc" and int(row["batch_size"]) == max(config.batch_sizes)
    ]
    real_ratio = statistics.fmean(float(row["speed_ratio_vs_dense"]) for row in primary) if primary else 0.0
    low_density = [
        row for row in sparse
        if row["workload"].startswith("synthetic_")
        and float(row["event_density"]) <= config.low_density_ceiling
    ]
    best_low_ratio = max((float(row["speed_ratio_vs_dense"]) for row in low_density), default=0.0)
    real_speed = real_ratio >= config.minimum_real_sparse_speed_ratio
    crossover = best_low_ratio >= 1.0
    passed = bool(equivalence and real_speed)
    return {
        "status": "pass" if passed else "stop",
        "numerical_equivalence_passed": bool(equivalence),
        "mean_real_ssc_sparse_speed_ratio": float(real_ratio),
        "best_registered_low_density_speed_ratio": float(best_low_ratio),
        "real_ssc_event_sparse_speed_supported": bool(real_speed),
        "low_density_crossover_supported": bool(crossover),
        "accuracy_claim_changed": False,
        "hardware_energy_claim_authorized": False,
        "next_milestone": (
            "integrate_event_sparse_training" if passed
            else "custom_triton_event_kernel" if equivalence
            else "correct_sparse_operator"
        ),
    }


def _load_test_data(config):
    data_config = _model_config(config)
    data_config = SHDConfig(**{
        **asdict(data_config),
        "train_samples": 1,
        "test_samples": config.test_samples,
        "data_root": config.data_root,
        "download": config.download,
    })
    data = load_ssc_tensors(data_config, validation_samples=1)
    return data[4], data[5]


def _model_config(config):
    return SHDConfig(
        seeds=config.seeds,
        input_neurons=config.input_neurons,
        classes=config.classes,
        timesteps=config.timesteps,
        duration_seconds=config.duration_seconds,
        hidden_neurons=128,
        max_edges=4096,
        epochs=1,
        warmup_epochs=0,
        batch_size=max(config.batch_sizes),
    )


def _fixed_density_events(batch_size, config, density, *, seed, device):
    total = batch_size * config.timesteps * config.input_neurons
    active = max(1, round(total * density))
    generator = torch.Generator(device="cpu").manual_seed(seed + round(density * 1_000_000))
    indices = torch.randperm(total, generator=generator)[:active]
    flat = torch.zeros(total, dtype=torch.float32)
    flat[indices] = 1.0
    return flat.reshape(batch_size, config.timesteps, config.input_neurons).to(device)


def plot_gen25(result, path):
    import matplotlib.pyplot as plt

    sparse = [row for row in result.summary if row["runtime"] == "event_sparse_hybrid"]
    sparse.sort(key=lambda row: (float(row["mean_event_density"]), int(row["batch_size"])))
    labels = [f"{row['workload']}\nB={row['batch_size']}" for row in sparse]
    x = list(range(len(labels)))
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
    axes[0].bar(x, [row["mean_speed_ratio_vs_dense"] for row in sparse], color="#35b4f2")
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("AMMC Gen-25 event-driven sparse operator audit")
    axes[0].set_ylabel("Sparse / compiled-dense throughput")
    axes[1].bar(x, [row["mean_sparse_operator_milliseconds"] for row in sparse], color="#ffb31a")
    axes[1].set_ylabel("Sparse input operator (ms / batch)")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def bundle_gen25_artifacts(paths, output_dir):
    output = pathlib.Path(output_dir)
    files = [pathlib.Path(value) for value in paths.values() if pathlib.Path(value).is_file()]
    manifest = output / "gen25_event_driven_sparse_audit_manifest.json"
    manifest.write_text(json.dumps({
        "files": [{"name": file.name, "sha256": hashlib.sha256(file.read_bytes()).hexdigest()} for file in files]
    }, indent=2) + "\n", encoding="utf-8")
    archive = output / "gen25_event_driven_sparse_audit_bundle.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for file in files + [manifest]:
            bundle.write(file, arcname=file.name)
    return {"manifest": str(manifest), "bundle": str(archive)}


def _population_std(values):
    values = list(values)
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _validate_config(config):
    if config.input_neurons != 700 or config.classes != 35:
        raise ValueError("Gen-25 is frozen for SSC")
    if len(config.seeds) < 3:
        raise ValueError("Gen-25 requires at least three timing seeds")
    if config.density_batch_size not in config.batch_sizes:
        raise ValueError("density_batch_size must be one of batch_sizes")
    if any(not 0.0 < density < 1.0 for density in config.synthetic_densities):
        raise ValueError("synthetic densities must be between zero and one")


def _write_csv(path, rows):
    destination = pathlib.Path(path)
    if not rows:
        destination.write_text("", encoding="utf-8")
        return
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
