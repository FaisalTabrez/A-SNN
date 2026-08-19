"""Gen-24 compiler audit for the supported residual-LIF state computation."""

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

from .event_mnist import torch
from .runtime import device_kind, resolve_device, seed_everything, sync
from .shd_benchmark import SHDConfig
from .shd_state_placement_diagnostic import (
    ResidualTemporalConvStateClassifier,
    matched_temporal_conv_residual_channels,
)
from .ssc_benchmark import load_ssc_tensors
from .ssc_efficiency_baselines import (
    TemporalDilatedTCNClassifier,
    matched_temporal_tcn_channels,
)


GEN24_MODELS = ("matched_tcn", "residual_lif")
GEN24_RUNTIMES = ("eager", "compiled")


@dataclass(frozen=True)
class Gen24Config:
    seeds: tuple[int, ...] = (621, 622, 623)
    input_neurons: int = 700
    classes: int = 35
    timesteps: int = 64
    duration_seconds: float = 1.0
    data_root: str = "gen5_data/ssc"
    download: bool = True
    test_samples: int = 2048
    batch_sizes: tuple[int, ...] = (1, 32, 256)
    target_parameters: int = 133_631
    temporal_levels: tuple[int, ...] = (1, 2, 4, 8)
    input_kernel_size: int = 5
    hidden_kernel_size: int = 3
    tcn_dilation: int = 2
    surrogate_slope: float = 10.0
    compile_mode: str = "reduce-overhead"
    warmup_iterations: int = 10
    measurement_iterations: int = 40
    measurement_repeats: int = 3
    maximum_logit_difference: float = 1e-4
    minimum_prediction_agreement: float = 1.0
    minimum_primary_speedup: float = 1.5
    minimum_tcn_throughput_ratio: float = 0.9


@dataclass
class Gen24Result:
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
        json_path = output / "gen24_compiled_residual_state.json"
        records_path = output / "gen24_compiled_residual_state_records.csv"
        summary_path = output / "gen24_compiled_residual_state_summary.csv"
        json_path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        _write_csv(records_path, self.records)
        _write_csv(summary_path, self.summary)
        paths = {
            "json": str(json_path),
            "records_csv": str(records_path),
            "summary_csv": str(summary_path),
        }
        if plot:
            plot_path = output / "gen24_compiled_residual_state.png"
            plot_gen24(self, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def available_gen24_models() -> tuple[str, ...]:
    return GEN24_MODELS


def run_gen24(
    config: Gen24Config = Gen24Config(),
    *,
    device="auto",
    dataset=None,
) -> Gen24Result:
    """Benchmark fixed eager and compiled graphs without retraining weights.

    Phase 48/49 already established the residual-LIF accuracy result. Gen-24
    isolates systems execution: weights are seeded and frozen because tensor
    shapes and the executed graph, not learned values, determine this test.
    """

    _validate_config(config)
    if torch is None:
        raise ImportError("Gen-24 requires PyTorch")
    if not hasattr(torch, "compile"):
        raise RuntimeError("Gen-24 requires torch.compile")
    resolved = resolve_device(device)
    if device_kind(resolved) != "cuda":
        raise ValueError("Gen-24 is a CUDA compiler benchmark; pass --device cuda")

    test_events, test_labels = dataset if dataset is not None else _load_test_data(config)
    largest_batch = max(config.batch_sizes)
    if int(test_events.shape[0]) < largest_batch:
        raise ValueError("test_samples must cover the largest batch size")

    tcn_channels, tcn_parameters = matched_temporal_tcn_channels(
        config.input_neurons,
        config.classes,
        config.target_parameters,
        input_kernel_size=config.input_kernel_size,
        hidden_kernel_size=config.hidden_kernel_size,
        temporal_levels=config.temporal_levels,
    )
    lif_channels, lif_parameters = matched_temporal_conv_residual_channels(
        config.input_neurons,
        config.classes,
        config.target_parameters,
        kernel_size=config.input_kernel_size,
        temporal_levels=config.temporal_levels,
    )
    architecture = {
        "matched_tcn": {"channels": tcn_channels, "trainable_parameters": tcn_parameters},
        "residual_lif": {"channels": lif_channels, "trainable_parameters": lif_parameters},
        "weights": "deterministically seeded and frozen; no accuracy claim in Gen-24",
        "accuracy_evidence": "Phase 48/49 SSC validation-selected training",
    }

    records: list[dict] = []
    for seed in config.seeds:
        for model_name in GEN24_MODELS:
            seed_everything(seed, device=resolved)
            model = _build_model(model_name, config, architecture).to(resolved).eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            for batch_size in config.batch_sizes:
                batch = test_events[:batch_size].to(resolved)
                eager = _benchmark_callable(
                    model,
                    batch,
                    resolved,
                    warmup_iterations=config.warmup_iterations,
                    measurement_iterations=config.measurement_iterations,
                    measurement_repeats=config.measurement_repeats,
                )
                records.append(_record(seed, model_name, "eager", batch_size, eager))

                compiled = _compile_and_benchmark(model, batch, resolved, config)
                row = _record(seed, model_name, "compiled", batch_size, compiled)
                row["speedup_vs_eager"] = float(
                    row["examples_per_second"] / max(eager["examples_per_second"], 1e-12)
                )
                records.append(row)

    summary = summarize_gen24(records)
    decision = decide_gen24(records, config)
    return Gen24Result(
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
        decision=decision,
    )


def _load_test_data(config):
    data_config = SHDConfig(
        seeds=config.seeds,
        train_samples=1,
        test_samples=config.test_samples,
        input_neurons=config.input_neurons,
        classes=config.classes,
        timesteps=config.timesteps,
        duration_seconds=config.duration_seconds,
        hidden_neurons=128,
        max_edges=4096,
        epochs=1,
        warmup_epochs=0,
        batch_size=max(config.batch_sizes),
        data_root=config.data_root,
        download=config.download,
    )
    data = load_ssc_tensors(data_config, validation_samples=1)
    return data[4], data[5]


def _build_model(model_name, config, architecture):
    if model_name == "matched_tcn":
        return TemporalDilatedTCNClassifier(
            _model_config(config),
            channels=architecture[model_name]["channels"],
            input_kernel_size=config.input_kernel_size,
            hidden_kernel_size=config.hidden_kernel_size,
            dilation=config.tcn_dilation,
            temporal_levels=config.temporal_levels,
        )
    if model_name == "residual_lif":
        return ResidualTemporalConvStateClassifier(
            _model_config(config),
            channels=architecture[model_name]["channels"],
            kernel_size=config.input_kernel_size,
            temporal_levels=config.temporal_levels,
            dynamics="lif",
            surrogate_slope=config.surrogate_slope,
        )
    raise ValueError(f"unknown Gen-24 model: {model_name}")


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


@torch.inference_mode()
def _benchmark_callable(
    model,
    batch,
    device,
    *,
    warmup_iterations: int,
    measurement_iterations: int,
    measurement_repeats: int,
):
    for _ in range(warmup_iterations):
        model(batch)
    sync(device)
    if device_kind(device) == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    durations = []
    for _ in range(measurement_repeats):
        started = time.perf_counter()
        for _ in range(measurement_iterations):
            model(batch)
        sync(device)
        durations.append(time.perf_counter() - started)
    seconds = statistics.median(durations)
    examples = int(batch.shape[0]) * measurement_iterations
    return {
        "seconds": float(seconds),
        "examples_per_second": float(examples / max(seconds, 1e-12)),
        "milliseconds_per_batch": float(1000.0 * seconds / measurement_iterations),
        "compile_seconds": 0.0,
        "compile_active": False,
        "compile_error": None,
        "maximum_logit_difference": 0.0,
        "prediction_agreement": 1.0,
        "cuda_peak_memory_mb": _cuda_peak_memory_mb(device),
    }


def _compile_and_benchmark(model, batch, device, config):
    try:
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()
        compiled_model = torch.compile(copy.deepcopy(model), mode=config.compile_mode)
        with torch.inference_mode():
            eager_logits = model(batch)
            started = time.perf_counter()
            compiled_logits = compiled_model(batch)
            sync(device)
            compile_seconds = time.perf_counter() - started
            maximum_difference = float((eager_logits - compiled_logits).abs().max().item())
            agreement = float(
                (eager_logits.argmax(dim=1) == compiled_logits.argmax(dim=1)).to(torch.float32).mean().item()
            )
        metrics = _benchmark_callable(
            compiled_model,
            batch,
            device,
            warmup_iterations=config.warmup_iterations,
            measurement_iterations=config.measurement_iterations,
            measurement_repeats=config.measurement_repeats,
        )
        metrics.update(
            {
                "compile_seconds": float(compile_seconds),
                "compile_active": True,
                "compile_error": None,
                "maximum_logit_difference": maximum_difference,
                "prediction_agreement": agreement,
            }
        )
        return metrics
    except Exception as error:  # compiler failures are experimental outcomes
        return {
            "seconds": 0.0,
            "examples_per_second": 0.0,
            "milliseconds_per_batch": 0.0,
            "compile_seconds": 0.0,
            "compile_active": False,
            "compile_error": f"{type(error).__name__}: {error}",
            # Keep failure artifacts strict-JSON-compatible while making the
            # numerical-equivalence gate unambiguously fail.
            "maximum_logit_difference": 1e30,
            "prediction_agreement": 0.0,
            "cuda_peak_memory_mb": _cuda_peak_memory_mb(device),
        }


def _record(seed, model_name, runtime, batch_size, metrics):
    return {
        "seed": int(seed),
        "model": model_name,
        "runtime": runtime,
        "batch_size": int(batch_size),
        **metrics,
        "speedup_vs_eager": 1.0,
    }


def summarize_gen24(records):
    summary = []
    for model_name in GEN24_MODELS:
        for runtime in GEN24_RUNTIMES:
            for batch_size in sorted({int(row["batch_size"]) for row in records}):
                group = [
                    row for row in records
                    if row["model"] == model_name
                    and row["runtime"] == runtime
                    and int(row["batch_size"]) == batch_size
                ]
                if not group:
                    continue
                summary.append(
                    {
                        "model": model_name,
                        "runtime": runtime,
                        "batch_size": batch_size,
                        "seeds": len(group),
                        "mean_examples_per_second": statistics.fmean(float(r["examples_per_second"]) for r in group),
                        "std_examples_per_second": _population_std(float(r["examples_per_second"]) for r in group),
                        "mean_milliseconds_per_batch": statistics.fmean(float(r["milliseconds_per_batch"]) for r in group),
                        "mean_speedup_vs_eager": statistics.fmean(float(r["speedup_vs_eager"]) for r in group),
                        "maximum_logit_difference": max(float(r["maximum_logit_difference"]) for r in group),
                        "minimum_prediction_agreement": min(float(r["prediction_agreement"]) for r in group),
                        "mean_compile_seconds": statistics.fmean(float(r["compile_seconds"]) for r in group),
                        "maximum_cuda_peak_memory_mb": max(float(r["cuda_peak_memory_mb"]) for r in group),
                        "compile_successes": sum(bool(r["compile_active"]) for r in group),
                    }
                )
    return summary


def decide_gen24(records, config):
    primary_batch = max(config.batch_sizes)
    compiled_lif = [
        row for row in records
        if row["model"] == "residual_lif"
        and row["runtime"] == "compiled"
        and int(row["batch_size"]) == primary_batch
    ]
    compiled_tcn = [
        row for row in records
        if row["model"] == "matched_tcn"
        and row["runtime"] == "compiled"
        and int(row["batch_size"]) == primary_batch
    ]
    compiler_success = len(compiled_lif) == len(config.seeds) and all(row["compile_active"] for row in compiled_lif)
    equivalence = compiler_success and all(
        float(row["maximum_logit_difference"]) <= config.maximum_logit_difference
        and float(row["prediction_agreement"]) >= config.minimum_prediction_agreement
        for row in compiled_lif
    )
    mean_speedup = statistics.fmean(float(row["speedup_vs_eager"]) for row in compiled_lif) if compiled_lif else 0.0
    lif_throughput = statistics.fmean(float(row["examples_per_second"]) for row in compiled_lif) if compiled_lif else 0.0
    tcn_throughput = statistics.fmean(float(row["examples_per_second"]) for row in compiled_tcn) if compiled_tcn else 0.0
    throughput_ratio = lif_throughput / max(tcn_throughput, 1e-12)
    speed_gate = mean_speedup >= config.minimum_primary_speedup
    compiler_pass = bool(equivalence and speed_gate)
    parity = bool(compiler_pass and throughput_ratio >= config.minimum_tcn_throughput_ratio)
    return {
        "status": "pass" if compiler_pass else "stop",
        "primary_batch_size": primary_batch,
        "compiler_success": bool(compiler_success),
        "numerical_equivalence_passed": bool(equivalence),
        "mean_residual_lif_compile_speedup": float(mean_speedup),
        "compiled_lif_to_tcn_throughput_ratio": float(throughput_ratio),
        "compiled_residual_state_supported": compiler_pass,
        "software_throughput_parity_vs_tcn_supported": parity,
        "accuracy_claim_changed": False,
        "hardware_energy_claim_authorized": False,
        "next_milestone": "event_driven_sparse_kernel_audit" if compiler_pass else "profile_compiler_graph_breaks",
    }


def plot_gen24(result, path):
    import matplotlib.pyplot as plt

    batches = sorted({int(row["batch_size"]) for row in result.summary})
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), constrained_layout=True)
    for model_name, color in (("matched_tcn", "#ff9f1c"), ("residual_lif", "#35b4f2")):
        compiled = [
            next(row for row in result.summary if row["model"] == model_name and row["runtime"] == "compiled" and int(row["batch_size"]) == batch)
            for batch in batches
        ]
        axes[0].plot(batches, [row["mean_examples_per_second"] for row in compiled], marker="o", label=model_name, color=color)
        axes[1].plot(batches, [row["mean_speedup_vs_eager"] for row in compiled], marker="o", label=model_name, color=color)
    axes[0].set_title("AMMC Gen-24 compiled residual-state audit")
    axes[0].set_ylabel("Examples / second")
    axes[1].set_ylabel("Compile speedup vs eager")
    axes[1].set_xlabel("Batch size")
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.grid(alpha=0.25)
        axis.legend()
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def bundle_gen24_artifacts(paths, output_dir):
    output = pathlib.Path(output_dir)
    files = [pathlib.Path(value) for value in paths.values() if pathlib.Path(value).is_file()]
    manifest = output / "gen24_compiled_residual_state_manifest.json"
    manifest.write_text(
        json.dumps(
            {"files": [{"name": file.name, "sha256": hashlib.sha256(file.read_bytes()).hexdigest()} for file in files]},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    archive = output / "gen24_compiled_residual_state_bundle.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for file in files + [manifest]:
            bundle.write(file, arcname=file.name)
    return {"manifest": str(manifest), "bundle": str(archive)}


def _cuda_peak_memory_mb(device):
    if device_kind(device) != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))


def _population_std(values):
    values = list(values)
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _validate_config(config):
    if config.input_neurons != 700 or config.classes != 35:
        raise ValueError("Gen-24 is frozen for SSC")
    if len(config.seeds) < 3:
        raise ValueError("Gen-24 requires at least three timing seeds")
    if not config.batch_sizes or min(config.batch_sizes) < 1:
        raise ValueError("batch sizes must be positive")
    if config.measurement_iterations < 1 or config.measurement_repeats < 1:
        raise ValueError("measurement counts must be positive")


def _write_csv(path, rows):
    destination = pathlib.Path(path)
    if not rows:
        destination.write_text("", encoding="utf-8")
        return
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
