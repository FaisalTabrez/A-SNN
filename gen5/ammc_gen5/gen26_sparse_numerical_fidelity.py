"""Gen-26 numerical-fidelity diagnostic for event-driven SSC input operators."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import pathlib
import statistics
import zipfile

from .event_mnist import torch
from .gen25_event_driven_sparse_audit import (
    ResidualLIFStateHead,
    sparse_temporal_currents,
)
from .runtime import device_kind, resolve_device, seed_everything
from .shd_benchmark import SHDConfig
from .shd_state_placement_diagnostic import (
    ResidualTemporalConvStateClassifier,
    matched_temporal_conv_residual_channels,
)
from .ssc_benchmark import load_ssc_tensors


GEN26_VARIANTS = ("coo_fp32_counts", "coo_fp64_counts", "coo_fp32_binary")


@dataclass(frozen=True)
class Gen26Config:
    seeds: tuple[int, ...] = (641, 642, 643)
    input_neurons: int = 700
    classes: int = 35
    timesteps: int = 64
    duration_seconds: float = 1.0
    data_root: str = "gen5_data/ssc"
    download: bool = True
    test_samples: int = 256
    batch_sizes: tuple[int, ...] = (1, 32, 256)
    target_parameters: int = 133_631
    temporal_levels: tuple[int, ...] = (1, 2, 4, 8)
    kernel_size: int = 5
    surrogate_slope: float = 10.0
    maximum_current_difference: float = 1e-5
    maximum_logit_difference: float = 1e-4
    minimum_prediction_agreement: float = 1.0
    minimum_binary_semantic_agreement: float = 0.999


@dataclass
class Gen26Result:
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
        json_path = output / "gen26_sparse_numerical_fidelity.json"
        records_path = output / "gen26_sparse_numerical_fidelity_records.csv"
        summary_path = output / "gen26_sparse_numerical_fidelity_summary.csv"
        json_path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        _write_csv(records_path, self.records)
        _write_csv(summary_path, self.summary)
        paths = {
            "json": str(json_path),
            "records_csv": str(records_path),
            "summary_csv": str(summary_path),
        }
        if plot:
            plot_path = output / "gen26_sparse_numerical_fidelity.png"
            plot_gen26(self, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def sparse_temporal_currents_fp64(events, temporal):
    """Count-preserving COO accumulation in float64, returned as float32."""

    batch, timesteps, input_neurons = events.shape
    weight = temporal.weight
    channels, _, kernel_size = weight.shape
    padding = kernel_size // 2
    nonzero = torch.nonzero(events, as_tuple=False)
    currents = torch.zeros(
        (batch * timesteps, channels), dtype=torch.float64, device=events.device
    )
    if temporal.bias is not None:
        currents.add_(temporal.bias.to(torch.float64))
    if nonzero.numel() == 0:
        return currents.to(torch.float32).reshape(batch, timesteps, channels)
    values = events[nonzero[:, 0], nonzero[:, 1], nonzero[:, 2]].to(torch.float64)
    weights = weight.to(torch.float64)
    for kernel_index in range(kernel_size):
        output_time = nonzero[:, 1] - kernel_index + padding
        valid = (output_time >= 0) & (output_time < timesteps)
        selected = nonzero[valid]
        rows = selected[:, 0] * timesteps + output_time[valid]
        sparse_input = torch.sparse_coo_tensor(
            torch.stack((rows, selected[:, 2])),
            values[valid],
            size=(batch * timesteps, input_neurons),
            dtype=torch.float64,
            device=events.device,
            check_invariants=False,
        ).coalesce()
        currents.add_(
            torch.sparse.mm(
                sparse_input, weights[:, :, kernel_index].transpose(0, 1)
            )
        )
    return currents.to(torch.float32).reshape(batch, timesteps, channels)


def run_gen26(
    config: Gen26Config = Gen26Config(),
    *,
    device="auto",
    dataset=None,
) -> Gen26Result:
    _validate_config(config)
    resolved = resolve_device(device)
    if device_kind(resolved) != "cuda":
        raise ValueError("Gen-26 is a CUDA numerical diagnostic; pass --device cuda")
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
    records = []
    for seed in config.seeds:
        seed_everything(seed, device=resolved)
        source = ResidualTemporalConvStateClassifier(
            _model_config(config),
            channels=channels,
            kernel_size=config.kernel_size,
            temporal_levels=config.temporal_levels,
            dynamics="lif",
            surrogate_slope=config.surrogate_slope,
        ).to(resolved).eval()
        head = ResidualLIFStateHead(source).to(resolved).eval()
        for parameter in source.parameters():
            parameter.requires_grad_(False)
        for batch_size in config.batch_sizes:
            counts = test_events[:batch_size].to(resolved).to(torch.float32)
            binary = (counts != 0).to(torch.float32)
            with torch.inference_mode():
                dense_count_currents = _dense_currents(counts, source.temporal)
                dense_binary_currents = _dense_currents(binary, source.temporal)
                dense_count_logits = head(dense_count_currents)
                dense_binary_logits = head(dense_binary_currents)
                count_predictions = dense_count_logits.argmax(dim=1)
                binary_predictions = dense_binary_logits.argmax(dim=1)
                semantic_agreement = float(
                    (count_predictions == binary_predictions).to(torch.float32).mean().item()
                )
                for variant in GEN26_VARIANTS:
                    if variant == "coo_fp32_counts":
                        sparse_currents = sparse_temporal_currents(counts, source.temporal)
                        dense_currents = dense_count_currents
                        dense_logits = dense_count_logits
                    elif variant == "coo_fp64_counts":
                        sparse_currents = sparse_temporal_currents_fp64(counts, source.temporal)
                        dense_currents = dense_count_currents
                        dense_logits = dense_count_logits
                    else:
                        sparse_currents = sparse_temporal_currents(binary, source.temporal)
                        dense_currents = dense_binary_currents
                        dense_logits = dense_binary_logits
                    sparse_logits = head(sparse_currents)
                    current_error = (dense_currents - sparse_currents).abs()
                    logit_error = (dense_logits - sparse_logits).abs()
                    agreement = float(
                        (dense_logits.argmax(dim=1) == sparse_logits.argmax(dim=1))
                        .to(torch.float32).mean().item()
                    )
                    max_current = float(current_error.max().item())
                    max_logit = float(logit_error.max().item())
                    records.append({
                        "seed": int(seed),
                        "variant": variant,
                        "batch_size": int(batch_size),
                        "source_event_density": float((counts != 0).to(torch.float32).mean().item()),
                        "source_nonbinary_fraction": float(
                            ((counts != 0) & (counts != 1)).to(torch.float32).mean().item()
                        ),
                        "maximum_current_difference": max_current,
                        "mean_current_difference": float(current_error.mean().item()),
                        "maximum_logit_difference": max_logit,
                        "mean_logit_difference": float(logit_error.mean().item()),
                        "prediction_agreement": agreement,
                        "binary_vs_count_dense_prediction_agreement": semantic_agreement,
                        "state_amplification_ratio": float(max_logit / max(max_current, 1e-30)),
                    })
    summary = summarize_gen26(records)
    return Gen26Result(
        config=asdict(config),
        device=device_kind(resolved),
        dataset={
            "name": "Spiking Speech Commands",
            "test_samples": int(test_events.shape[0]),
            "timesteps": int(test_events.shape[1]),
            "input_neurons": int(test_events.shape[2]),
            "labels_loaded_for_dataset_identity_only": int(test_labels.shape[0]),
        },
        architecture={
            "model": "Phase-49 residual LIF",
            "channels": channels,
            "trainable_parameters": parameters,
            "weights": "deterministically seeded and frozen; no accuracy claim",
            "purpose": "separate sparse arithmetic error from LIF threshold amplification",
        },
        records=records,
        summary=summary,
        decision=decide_gen26(records, config),
    )


def summarize_gen26(records):
    summary = []
    for variant in GEN26_VARIANTS:
        group = [row for row in records if row["variant"] == variant]
        summary.append({
            "variant": variant,
            "records": len(group),
            "maximum_current_difference": max(float(row["maximum_current_difference"]) for row in group),
            "mean_current_difference": statistics.fmean(float(row["mean_current_difference"]) for row in group),
            "maximum_logit_difference": max(float(row["maximum_logit_difference"]) for row in group),
            "mean_logit_difference": statistics.fmean(float(row["mean_logit_difference"]) for row in group),
            "minimum_prediction_agreement": min(float(row["prediction_agreement"]) for row in group),
            "minimum_binary_vs_count_dense_prediction_agreement": min(
                float(row["binary_vs_count_dense_prediction_agreement"]) for row in group
            ),
            "maximum_state_amplification_ratio": max(float(row["state_amplification_ratio"]) for row in group),
            "mean_source_nonbinary_fraction": statistics.fmean(
                float(row["source_nonbinary_fraction"]) for row in group
            ),
        })
    return summary


def decide_gen26(records, config):
    summary = {row["variant"]: row for row in summarize_gen26(records)}
    fp64 = summary["coo_fp64_counts"]
    binary = summary["coo_fp32_binary"]
    count_repair = (
        float(fp64["maximum_current_difference"]) <= config.maximum_current_difference
        and float(fp64["maximum_logit_difference"]) <= config.maximum_logit_difference
        and float(fp64["minimum_prediction_agreement"]) >= config.minimum_prediction_agreement
    )
    binary_exact = (
        float(binary["maximum_current_difference"]) <= config.maximum_current_difference
        and float(binary["maximum_logit_difference"]) <= config.maximum_logit_difference
        and float(binary["minimum_prediction_agreement"]) >= config.minimum_prediction_agreement
    )
    binary_semantics = (
        float(binary["minimum_binary_vs_count_dense_prediction_agreement"])
        >= config.minimum_binary_semantic_agreement
    )
    passed = bool(count_repair or (binary_exact and binary_semantics))
    if count_repair:
        next_milestone = "custom_count_preserving_event_kernel"
        selected = "coo_fp64_counts"
    elif binary_exact and binary_semantics:
        next_milestone = "event_native_binary_kernel"
        selected = "coo_fp32_binary"
    elif binary_exact:
        next_milestone = "train_and_validate_binary_event_encoding"
        selected = None
    else:
        next_milestone = "threshold_robustness_diagnostic"
        selected = None
    return {
        "status": "pass" if passed else "stop",
        "count_preserving_numerical_repair_supported": bool(count_repair),
        "binary_sparse_operator_exact": bool(binary_exact),
        "binary_encoding_semantically_stable": bool(binary_semantics),
        "selected_operator": selected,
        "accuracy_claim_changed": False,
        "hardware_energy_claim_authorized": False,
        "next_milestone": next_milestone,
    }


def _dense_currents(events, temporal):
    return temporal(events.transpose(1, 2)).transpose(1, 2)


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


def plot_gen26(result, path):
    import matplotlib.pyplot as plt

    labels = [row["variant"].replace("_", "\n") for row in result.summary]
    x = list(range(len(labels)))
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    axes[0].bar(x, [row["maximum_current_difference"] for row in result.summary], color="#35b4f2")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Maximum current difference")
    axes[0].set_title("AMMC Gen-26 sparse numerical fidelity")
    axes[1].bar(x, [row["maximum_logit_difference"] for row in result.summary], color="#ffb31a")
    axes[1].set_yscale("symlog", linthresh=1e-10)
    axes[1].set_ylabel("Maximum logit difference")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25)
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def bundle_gen26_artifacts(paths, output_dir):
    output = pathlib.Path(output_dir)
    files = [pathlib.Path(value) for value in paths.values() if pathlib.Path(value).is_file()]
    manifest = output / "gen26_sparse_numerical_fidelity_manifest.json"
    manifest.write_text(json.dumps({
        "files": [{"name": file.name, "sha256": hashlib.sha256(file.read_bytes()).hexdigest()} for file in files]
    }, indent=2) + "\n", encoding="utf-8")
    archive = output / "gen26_sparse_numerical_fidelity_bundle.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for file in files + [manifest]:
            bundle.write(file, arcname=file.name)
    return {"manifest": str(manifest), "bundle": str(archive)}


def _validate_config(config):
    if config.input_neurons != 700 or config.classes != 35:
        raise ValueError("Gen-26 is frozen for SSC")
    if len(config.seeds) < 3:
        raise ValueError("Gen-26 requires at least three seeds")
    if not config.batch_sizes or min(config.batch_sizes) < 1:
        raise ValueError("batch sizes must be positive")


def _write_csv(path, rows):
    destination = pathlib.Path(path)
    if not rows:
        destination.write_text("", encoding="utf-8")
        return
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
