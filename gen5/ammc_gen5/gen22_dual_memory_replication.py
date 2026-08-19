"""Gen-22 direct replication of dual memory under sequential SSC shifts."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import pathlib
import statistics
import time
import zipfile

from .event_mnist import torch
from .gen9_continual_adaptation import apply_sensor_damage
from .gen21_matched_causal_mechanisms import (
    Gen21Config,
    Gen21MechanismReadout,
    _adapt_model,
    _build_backbone,
    _evaluate_model,
    _ssc_config,
)
from .runtime import device_kind, resolve_device, seed_everything
from .shd_validation_checkpoint import _stratified_split, _train_validation_selected
from .ssc_benchmark import load_ssc_tensors


GEN22_ARMS = (
    "static_backbone",
    "single_memory_control",
    "dual_memory",
    "dual_memory_shuffled_consolidation",
)


@dataclass(frozen=True)
class Gen22Config:
    seeds: tuple[int, ...] = (421, 422, 423, 424, 425)
    input_neurons: int = 700
    classes: int = 35
    timesteps: int = 64
    duration_seconds: float = 1.0
    data_root: str = "gen5_data/ssc"
    download: bool = True
    source_train_samples: int = 20_000
    validation_samples: int = 9_000
    test_samples: int = 8_000
    source_epochs: int = 12
    adaptation_epochs_per_shift: int = 5
    batch_size: int = 256
    source_learning_rate: float = 0.003
    adaptation_learning_rate: float = 0.01
    weight_decay: float = 0.0001
    target_parameters: int = 133_631
    temporal_conv_kernel_size: int = 5
    temporal_levels: tuple[int, ...] = (1, 2, 4, 8)
    surrogate_slope: float = 10.0
    sensor_damage_fraction: float = 0.35
    delay_slots: int = 3
    active_slot_fraction: float = 0.35
    stw_decay: float = 0.98
    consolidation_rate: float = 0.02
    minimum_a_retention_gain_vs_single: float = 0.01
    maximum_b_accuracy_cost_vs_single: float = 0.005
    minimum_ltw_causal_margin: float = 0.005
    minimum_consolidation_identity_margin: float = 0.005
    minimum_qualifying_seed_count: int = 3


@dataclass
class Gen22Result:
    config: dict
    device: str
    dataset: dict
    records: list[dict]
    summary: list[dict]
    decision: dict

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "gen22_dual_memory_replication.json"
        records_path = output / "gen22_dual_memory_replication_records.csv"
        summary_path = output / "gen22_dual_memory_replication_summary.csv"
        json_path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        _write_csv(records_path, self.records)
        _write_csv(summary_path, self.summary)
        paths = {"json": str(json_path), "records_csv": str(records_path), "summary_csv": str(summary_path)}
        if plot:
            plot_path = output / "gen22_dual_memory_replication.png"
            plot_gen22(self, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def available_gen22_arms() -> tuple[str, ...]:
    return GEN22_ARMS


def run_gen22(
    config: Gen22Config = Gen22Config(), *, device="auto",
    progress_path: str | pathlib.Path | None = None, dataset=None,
) -> Gen22Result:
    _validate_config(config)
    resolved = resolve_device(device)
    data = dataset if dataset is not None else _load_dataset(config)
    recovered = _load_progress(progress_path, config)
    records = list(recovered)
    completed = {(int(row["seed"]), row["arm"]) for row in records}
    for seed in config.seeds:
        missing = [arm for arm in GEN22_ARMS if (int(seed), arm) not in completed]
        if not missing:
            continue
        seed_everything(seed, device=resolved)
        backbone = _build_backbone(_gen21_config(config), seed=seed, device=resolved)
        training = _train_validation_selected(
            backbone, data[0], data[1], data[2], data[3],
            _ssc_config(_gen21_config(config), seed=seed), seed=seed, device=resolved,
        )
        backbone.load_state_dict(training["best_state"])
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        damage_a, damage_b = disjoint_sensor_damage_indices(
            config.input_neurons, config.sensor_damage_fraction, seed=seed + 22_000
        )
        adapt_a = apply_sensor_damage(data[4], damage_a)
        adapt_b = apply_sensor_damage(data[6], damage_b)
        test_a = apply_sensor_damage(data[8], damage_a)
        test_b = apply_sensor_damage(data[8], damage_b)
        initial_clean = _evaluate_model(backbone, data[8], data[9], config.batch_size, resolved)[0]
        initial_a = _evaluate_model(backbone, test_a, data[9], config.batch_size, resolved)[0]
        initial_b = _evaluate_model(backbone, test_b, data[9], config.batch_size, resolved)[0]
        for arm in missing:
            record = _run_arm(
                arm, backbone, adapt_a, data[5], adapt_b, data[7],
                data[8], data[9], test_a, test_b, initial_clean, initial_a, initial_b,
                training, config, seed, resolved,
            )
            records.append(record)
            _save_progress(progress_path, config, records)
    _attach_paired_comparisons(records)
    _save_progress(progress_path, config, records)
    summary = summarize_gen22(records)
    return Gen22Result(
        config=asdict(config), device=device_kind(resolved),
        dataset={
            "name": "Spiking Speech Commands",
            "source_train_samples": int(data[0].shape[0]),
            "source_validation_samples": int(data[2].shape[0]),
            "adaptation_a_samples": int(data[4].shape[0]),
            "adaptation_b_samples": int(data[6].shape[0]),
            "test_samples": int(data[8].shape[0]),
            "shift_a": f"disjoint_{config.sensor_damage_fraction:.0%}_sensor_lesion_a",
            "shift_b": f"disjoint_{config.sensor_damage_fraction:.0%}_sensor_lesion_b",
        }, records=records, summary=summary,
        decision=decide_gen22(records, config),
    )


def _run_arm(
    arm, backbone, adapt_a, labels_a, adapt_b, labels_b, clean_test, test_labels,
    test_a, test_b, initial_clean, initial_a, initial_b, training, config, seed, device,
):
    started = time.perf_counter()
    mapped_arm = {
        "static_backbone": "static_backbone",
        "single_memory_control": "global_gradient_control",
        "dual_memory": "dual_memory_only",
        "dual_memory_shuffled_consolidation": "dual_memory_only",
    }[arm]
    model = Gen21MechanismReadout(
        backbone, mapped_arm, _gen21_config(config), seed=seed
    ).to(device)
    if arm == "dual_memory_shuffled_consolidation":
        model.consolidation_mode = "shuffled"
    if arm != "static_backbone":
        _adapt_model(model, adapt_a, labels_a, _gen21_config(config), seed=seed, device=device)
    after_a_a = _evaluate_model(model, test_a, test_labels, config.batch_size, device)[0]
    after_a_b = _evaluate_model(model, test_b, test_labels, config.batch_size, device)[0]
    after_a_clean = _evaluate_model(model, clean_test, test_labels, config.batch_size, device)[0]
    if arm != "static_backbone":
        _adapt_model(model, adapt_b, labels_b, _gen21_config(config), seed=seed + 1_000, device=device)
    after_b_b, seconds, activity = _evaluate_model(model, test_b, test_labels, config.batch_size, device)
    after_b_a = _evaluate_model(model, test_a, test_labels, config.batch_size, device)[0]
    after_b_clean = _evaluate_model(model, clean_test, test_labels, config.batch_size, device)[0]
    ltw_causal_a = after_b_a
    if mapped_arm == "dual_memory_only":
        model.causal_mode = "zero_ltw"
        ltw_causal_a = _evaluate_model(model, test_a, test_labels, config.batch_size, device)[0]
        model.causal_mode = "normal"
    return {
        "seed": int(seed), "arm": arm,
        "source_best_epoch": int(training["best_epoch"]),
        "source_validation_accuracy": float(training["best_validation_accuracy"]),
        "initial_clean_accuracy": float(initial_clean),
        "initial_a_accuracy": float(initial_a), "initial_b_accuracy": float(initial_b),
        "after_a_a_accuracy": float(after_a_a), "after_a_b_accuracy": float(after_a_b),
        "after_a_clean_accuracy": float(after_a_clean),
        "after_b_b_accuracy": float(after_b_b), "after_b_a_accuracy": float(after_b_a),
        "after_b_clean_accuracy": float(after_b_clean),
        "a_adaptation_gain": float(after_a_a - initial_a),
        "b_adaptation_gain": float(after_b_b - initial_b),
        "a_forgetting_after_b": float(after_a_a - after_b_a),
        "clean_retention_drop": float(initial_clean - after_b_clean),
        "ltw_removed_a_accuracy": float(ltw_causal_a),
        "ltw_causal_margin": float(after_b_a - ltw_causal_a),
        "stability_plasticity_score": float(0.5 * (after_b_a + after_b_b)),
        "allocated_slots": int(model.allocated_slots), "active_slots": int(model.active_slots),
        "adapter_memory_bytes": int((model.delta.numel() + model.ltw.numel() + model.active_mask.numel()) * 4),
        "mean_activity": float(activity),
        "test_examples_per_second": float(test_labels.numel() / max(seconds, 1e-12)),
        "wall_seconds": float(time.perf_counter() - started),
    }


def _load_dataset(config):
    gen21 = _gen21_config(config)
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_ssc_tensors(
        _ssc_config(gen21, seed=config.seeds[0]), validation_samples=config.validation_samples
    )
    source_x, source_y, remainder_x, remainder_y = _stratified_split(
        valid_x, valid_y, fraction=2.0 / 3.0, seed=config.seeds[0] + 31
    )
    adapt_a_x, adapt_a_y, adapt_b_x, adapt_b_y = _stratified_split(
        remainder_x, remainder_y, fraction=0.50, seed=config.seeds[0] + 32
    )
    return train_x, train_y, source_x, source_y, adapt_a_x, adapt_a_y, adapt_b_x, adapt_b_y, test_x, test_y


def _gen21_config(config: Gen22Config) -> Gen21Config:
    return Gen21Config(
        screen_seed=config.seeds[0], confirmation_seeds=config.seeds[1:],
        input_neurons=config.input_neurons, classes=config.classes,
        timesteps=config.timesteps, duration_seconds=config.duration_seconds,
        data_root=config.data_root, download=config.download,
        source_train_samples=config.source_train_samples,
        validation_samples=config.validation_samples, test_samples=config.test_samples,
        source_epochs=config.source_epochs,
        adaptation_epochs=config.adaptation_epochs_per_shift,
        batch_size=config.batch_size, source_learning_rate=config.source_learning_rate,
        adaptation_learning_rate=config.adaptation_learning_rate,
        weight_decay=config.weight_decay, target_parameters=config.target_parameters,
        temporal_conv_kernel_size=config.temporal_conv_kernel_size,
        temporal_levels=config.temporal_levels, surrogate_slope=config.surrogate_slope,
        sensor_damage_fraction=config.sensor_damage_fraction,
        delay_slots=config.delay_slots, active_slot_fraction=config.active_slot_fraction,
        stw_decay=config.stw_decay, consolidation_rate=config.consolidation_rate,
    )


def disjoint_sensor_damage_indices(input_neurons: int, fraction: float, *, seed: int):
    if torch is None:
        raise ImportError("Gen-22 requires PyTorch")
    count = round(input_neurons * fraction)
    if input_neurons <= 1 or count <= 0 or 2 * count > input_neurons:
        raise ValueError("two disjoint damage banks must fit the sensor population")
    order = torch.randperm(input_neurons, generator=torch.Generator().manual_seed(seed))
    return tuple(int(x) for x in order[:count]), tuple(int(x) for x in order[count : 2 * count])


def _attach_paired_comparisons(records):
    lookup = {(int(row["seed"]), row["arm"]): row for row in records}
    for row in records:
        single = lookup.get((int(row["seed"]), "single_memory_control"))
        shuffled = lookup.get((int(row["seed"]), "dual_memory_shuffled_consolidation"))
        row["a_retention_gain_vs_single"] = float(row["after_b_a_accuracy"] - single["after_b_a_accuracy"]) if single else 0.0
        row["b_accuracy_gain_vs_single"] = float(row["after_b_b_accuracy"] - single["after_b_b_accuracy"]) if single else 0.0
        row["stability_plasticity_gain_vs_single"] = float(row["stability_plasticity_score"] - single["stability_plasticity_score"]) if single else 0.0
        row["consolidation_identity_margin"] = float(row["stability_plasticity_score"] - shuffled["stability_plasticity_score"]) if shuffled and row["arm"] == "dual_memory" else 0.0


def summarize_gen22(records):
    rows = list(records)
    result = []
    for arm in GEN22_ARMS:
        group = [row for row in rows if row["arm"] == arm]
        if not group:
            continue
        result.append({
            "arm": arm, "seeds": len(group),
            "mean_after_b_a_accuracy": statistics.fmean(float(r["after_b_a_accuracy"]) for r in group),
            "mean_after_b_b_accuracy": statistics.fmean(float(r["after_b_b_accuracy"]) for r in group),
            "mean_a_forgetting_after_b": statistics.fmean(float(r["a_forgetting_after_b"]) for r in group),
            "mean_clean_retention_drop": statistics.fmean(float(r["clean_retention_drop"]) for r in group),
            "mean_stability_plasticity_score": statistics.fmean(float(r["stability_plasticity_score"]) for r in group),
            "mean_a_retention_gain_vs_single": statistics.fmean(float(r["a_retention_gain_vs_single"]) for r in group),
            "mean_b_accuracy_gain_vs_single": statistics.fmean(float(r["b_accuracy_gain_vs_single"]) for r in group),
            "mean_stability_plasticity_gain_vs_single": statistics.fmean(float(r["stability_plasticity_gain_vs_single"]) for r in group),
            "mean_ltw_causal_margin": statistics.fmean(float(r["ltw_causal_margin"]) for r in group),
            "mean_consolidation_identity_margin": statistics.fmean(float(r["consolidation_identity_margin"]) for r in group),
            "allocated_slots": int(group[0]["allocated_slots"]), "active_slots": int(group[0]["active_slots"]),
            "adapter_memory_bytes": int(group[0]["adapter_memory_bytes"]),
        })
    return result


def decide_gen22(records, config):
    dual = [r for r in records if r["arm"] == "dual_memory"]
    seed_passes = [
        r for r in dual
        if float(r["a_retention_gain_vs_single"]) >= config.minimum_a_retention_gain_vs_single
        and float(r["b_accuracy_gain_vs_single"]) >= -config.maximum_b_accuracy_cost_vs_single
        and float(r["ltw_causal_margin"]) >= config.minimum_ltw_causal_margin
        and float(r["consolidation_identity_margin"]) >= config.minimum_consolidation_identity_margin
    ]
    mean = summarize_gen22(records)
    dual_summary = next((r for r in mean if r["arm"] == "dual_memory"), None)
    aggregate = bool(dual_summary) and (
        float(dual_summary["mean_a_retention_gain_vs_single"]) >= config.minimum_a_retention_gain_vs_single
        and float(dual_summary["mean_b_accuracy_gain_vs_single"]) >= -config.maximum_b_accuracy_cost_vs_single
        and float(dual_summary["mean_ltw_causal_margin"]) >= config.minimum_ltw_causal_margin
        and float(dual_summary["mean_consolidation_identity_margin"]) >= config.minimum_consolidation_identity_margin
    )
    passed = aggregate and len(seed_passes) >= config.minimum_qualifying_seed_count
    return {
        "status": "pass" if passed else "stop",
        "qualified_seed_count": len(seed_passes),
        "aggregate_gate_passed": bool(aggregate),
        "dual_memory_timescale_claim_supported": bool(passed),
        "whole_network_memory_claim_supported": False,
        "hardware_energy_claim_authorized": False,
        "next_milestone": "replicate_dual_memory_in_backbone_synapses" if passed else "close_or_redesign_dual_memory",
    }


def plot_gen22(result, path):
    import matplotlib.pyplot as plt
    labels = [r["arm"].replace("_", "\n") for r in result.summary]
    x = list(range(len(labels)))
    fig, axes = plt.subplots(3, 1, figsize=(14, 13), constrained_layout=True)
    axes[0].bar(x, [100 * r["mean_after_b_a_accuracy"] for r in result.summary], label="Shift A retention", color="#35b4f2")
    axes[0].bar(x, [100 * r["mean_after_b_b_accuracy"] for r in result.summary], alpha=0.55, label="Shift B accuracy", color="#ffb31a")
    axes[0].set_ylabel("Accuracy (%)"); axes[0].legend(); axes[0].set_title("AMMC Gen-22 dual-memory sequential-shift replication")
    axes[1].bar(x, [100 * r["mean_a_retention_gain_vs_single"] for r in result.summary], color="#167d55")
    axes[1].axhline(100 * result.config["minimum_a_retention_gain_vs_single"], color="#bd3d3a", linestyle="--")
    axes[1].set_ylabel("A retention gain vs single (points)")
    axes[2].bar(x, [100 * r["mean_ltw_causal_margin"] for r in result.summary], color="#8b6fd6")
    axes[2].set_ylabel("LTW-removal margin (points)")
    for axis in axes:
        axis.set_xticks(x, labels); axis.grid(axis="y", alpha=0.25)
    destination = pathlib.Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180); plt.close(fig)


def bundle_gen22_artifacts(paths, output_dir):
    output = pathlib.Path(output_dir)
    files = [pathlib.Path(value) for value in paths.values() if pathlib.Path(value).is_file()]
    manifest = output / "gen22_dual_memory_replication_manifest.json"
    payload = {"files": [{"name": f.name, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()} for f in files]}
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    archive = output / "gen22_dual_memory_replication_bundle.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for file in files + [manifest]: bundle.write(file, arcname=file.name)
    return {"manifest": str(manifest), "bundle": str(archive)}


def _save_progress(path, config, records):
    if path is None: return
    destination = pathlib.Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"config": asdict(config), "records": records}, indent=2) + "\n", encoding="utf-8")


def _load_progress(path, config):
    if path is None or not pathlib.Path(path).exists(): return []
    try: payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): return []
    return list(payload.get("records", [])) if payload.get("config") == json.loads(json.dumps(asdict(config))) else []


def _validate_config(config):
    if config.input_neurons != 700 or config.classes != 35: raise ValueError("Gen-22 is frozen for SSC")
    if len(config.seeds) < 5: raise ValueError("Gen-22 requires at least five seeds")
    disjoint_sensor_damage_indices(config.input_neurons, config.sensor_damage_fraction, seed=0)
    if config.source_epochs <= 0 or config.adaptation_epochs_per_shift <= 0: raise ValueError("epochs must be positive")


def _write_csv(path, rows):
    if not rows: pathlib.Path(path).write_text("", encoding="utf-8"); return
    with pathlib.Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
