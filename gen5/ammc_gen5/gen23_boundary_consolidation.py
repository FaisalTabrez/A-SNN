"""Gen-23 boundary-gated selective dual-memory consolidation on SSC."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import pathlib
import statistics
import time
import zipfile

from .event_mnist import torch
from .gen9_continual_adaptation import apply_sensor_damage
from .gen21_matched_causal_mechanisms import Gen21MechanismReadout, _build_backbone, _evaluate_model, _ssc_config
from .gen22_dual_memory_replication import Gen22Config, _gen21_config, _load_dataset, disjoint_sensor_damage_indices
from .runtime import device_kind, mark_step, resolve_device, seed_everything
from .shd_validation_checkpoint import _train_validation_selected


GEN23_ARMS = (
    "static_backbone",
    "single_memory_control",
    "bounded_stw_no_consolidation",
    "boundary_selective_dual_memory",
    "boundary_shuffled_dual_memory",
)


@dataclass(frozen=True)
class Gen23Config:
    seeds: tuple[int, ...] = (521, 522, 523, 524, 525)
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
    stw_decay: float = 0.995
    stw_maximum: float = 0.50
    protected_fraction: float = 0.50
    minimum_a_retention_gain_vs_single: float = 0.01
    maximum_b_accuracy_cost_vs_single: float = 0.01
    minimum_ltw_causal_margin: float = 0.005
    minimum_selection_identity_margin: float = 0.005
    maximum_boundary_accuracy_change: float = 0.001
    minimum_qualifying_seed_count: int = 3


@dataclass
class Gen23Result:
    config: dict
    device: str
    dataset: dict
    records: list[dict]
    summary: list[dict]
    decision: dict

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir); output.mkdir(parents=True, exist_ok=True)
        json_path = output / "gen23_boundary_consolidation.json"
        records_path = output / "gen23_boundary_consolidation_records.csv"
        summary_path = output / "gen23_boundary_consolidation_summary.csv"
        json_path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        _write_csv(records_path, self.records); _write_csv(summary_path, self.summary)
        paths = {"json": str(json_path), "records_csv": str(records_path), "summary_csv": str(summary_path)}
        if plot:
            plot_path = output / "gen23_boundary_consolidation.png"
            plot_gen23(self, plot_path); paths["plot"] = str(plot_path)
        return paths


def available_gen23_arms() -> tuple[str, ...]:
    return GEN23_ARMS


def run_gen23(config: Gen23Config = Gen23Config(), *, device="auto", progress_path=None, dataset=None) -> Gen23Result:
    _validate_config(config)
    resolved = resolve_device(device)
    data = dataset if dataset is not None else _load_dataset(_gen22_config(config))
    records = _load_progress(progress_path, config)
    completed = {(int(row["seed"]), row["arm"]) for row in records}
    for seed in config.seeds:
        missing = [arm for arm in GEN23_ARMS if (int(seed), arm) not in completed]
        if not missing: continue
        seed_everything(seed, device=resolved)
        gen21 = _gen21_config(_gen22_config(config))
        backbone = _build_backbone(gen21, seed=seed, device=resolved)
        training = _train_validation_selected(
            backbone, data[0], data[1], data[2], data[3],
            _ssc_config(gen21, seed=seed), seed=seed, device=resolved,
        )
        backbone.load_state_dict(training["best_state"])
        for parameter in backbone.parameters(): parameter.requires_grad_(False)
        damage_a, damage_b = disjoint_sensor_damage_indices(
            config.input_neurons, config.sensor_damage_fraction, seed=seed + 23_000
        )
        adapt_a = apply_sensor_damage(data[4], damage_a); adapt_b = apply_sensor_damage(data[6], damage_b)
        test_a = apply_sensor_damage(data[8], damage_a); test_b = apply_sensor_damage(data[8], damage_b)
        initial_clean = _evaluate_model(backbone, data[8], data[9], config.batch_size, resolved)[0]
        initial_a = _evaluate_model(backbone, test_a, data[9], config.batch_size, resolved)[0]
        initial_b = _evaluate_model(backbone, test_b, data[9], config.batch_size, resolved)[0]
        for arm in missing:
            records.append(_run_arm(
                arm, backbone, adapt_a, data[5], adapt_b, data[7], data[8], data[9],
                test_a, test_b, initial_clean, initial_a, initial_b, training, config, seed, resolved,
            ))
            _save_progress(progress_path, config, records)
    _attach_comparisons(records); _save_progress(progress_path, config, records)
    summary = summarize_gen23(records)
    return Gen23Result(
        config=asdict(config), device=device_kind(resolved),
        dataset={
            "name": "Spiking Speech Commands", "source_train_samples": int(data[0].shape[0]),
            "source_validation_samples": int(data[2].shape[0]), "adaptation_a_samples": int(data[4].shape[0]),
            "adaptation_b_samples": int(data[6].shape[0]), "test_samples": int(data[8].shape[0]),
            "shift_a": f"disjoint_{config.sensor_damage_fraction:.0%}_sensor_lesion_a",
            "shift_b": f"disjoint_{config.sensor_damage_fraction:.0%}_sensor_lesion_b",
        }, records=records, summary=summary, decision=decide_gen23(records, config),
    )


def _run_arm(
    arm, backbone, adapt_a, labels_a, adapt_b, labels_b, clean_test, test_labels,
    test_a, test_b, initial_clean, initial_a, initial_b, training, config, seed, device,
):
    started = time.perf_counter()
    mapped = "static_backbone" if arm == "static_backbone" else (
        "global_gradient_control" if arm == "single_memory_control" else "dual_memory_only"
    )
    model = Gen21MechanismReadout(backbone, mapped, _gen21_config(_gen22_config(config)), seed=seed).to(device)
    model.register_buffer("protected_mask", torch.zeros_like(model.active_mask))
    if arm != "static_backbone":
        _adapt_bounded(model, adapt_a, labels_a, config, seed=seed, device=device, bounded=arm != "single_memory_control")
    pre_boundary_a = _evaluate_model(model, test_a, test_labels, config.batch_size, device)[0]
    if arm == "boundary_selective_dual_memory":
        _consolidate_boundary(model, config.protected_fraction, seed=seed, shuffled=False)
    elif arm == "boundary_shuffled_dual_memory":
        _consolidate_boundary(model, config.protected_fraction, seed=seed, shuffled=True)
    post_boundary_a = _evaluate_model(model, test_a, test_labels, config.batch_size, device)[0]
    if arm != "static_backbone":
        _adapt_bounded(model, adapt_b, labels_b, config, seed=seed + 1_000, device=device, bounded=arm != "single_memory_control")
    after_b_b, seconds, activity = _evaluate_model(model, test_b, test_labels, config.batch_size, device)
    after_b_a = _evaluate_model(model, test_a, test_labels, config.batch_size, device)[0]
    after_b_clean = _evaluate_model(model, clean_test, test_labels, config.batch_size, device)[0]
    ltw_removed_a = after_b_a
    if arm.startswith("boundary_"):
        model.causal_mode = "zero_ltw"
        ltw_removed_a = _evaluate_model(model, test_a, test_labels, config.batch_size, device)[0]
        model.causal_mode = "normal"
    return {
        "seed": int(seed), "arm": arm, "source_best_epoch": int(training["best_epoch"]),
        "source_validation_accuracy": float(training["best_validation_accuracy"]),
        "initial_clean_accuracy": float(initial_clean), "initial_a_accuracy": float(initial_a),
        "initial_b_accuracy": float(initial_b), "pre_boundary_a_accuracy": float(pre_boundary_a),
        "post_boundary_a_accuracy": float(post_boundary_a),
        "boundary_accuracy_change": float(post_boundary_a - pre_boundary_a),
        "after_b_a_accuracy": float(after_b_a), "after_b_b_accuracy": float(after_b_b),
        "after_b_clean_accuracy": float(after_b_clean),
        "a_retention_from_boundary": float(after_b_a - post_boundary_a),
        "b_adaptation_gain": float(after_b_b - initial_b),
        "clean_retention_drop": float(initial_clean - after_b_clean),
        "ltw_removed_a_accuracy": float(ltw_removed_a), "ltw_causal_margin": float(after_b_a - ltw_removed_a),
        "stability_plasticity_score": float(0.5 * (after_b_a + after_b_b)),
        "protected_slots": int(model.protected_mask.sum().item()),
        "allocated_slots": int(model.allocated_slots), "active_slots": int(model.active_slots),
        "adapter_memory_bytes": int((model.delta.numel() + model.ltw.numel() + model.active_mask.numel() + model.protected_mask.numel()) * 4),
        "mean_activity": float(activity), "test_examples_per_second": float(test_labels.numel() / max(seconds, 1e-12)),
        "wall_seconds": float(time.perf_counter() - started),
    }


def _adapt_bounded(model, events, labels, config, *, seed, device, bounded):
    optimizer = torch.optim.AdamW([model.delta], lr=config.adaptation_learning_rate, weight_decay=config.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(seed + 93_000)
    for _ in range(config.adaptation_epochs_per_shift):
        order = torch.randperm(events.shape[0], generator=generator)
        model.train()
        for offset in range(0, events.shape[0], config.batch_size):
            index = order[offset : offset + config.batch_size]
            x = events.index_select(0, index).to(device); y = labels.index_select(0, index).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x), y); loss.backward(); optimizer.step()
            if bounded:
                with torch.no_grad():
                    model.delta.mul_(config.stw_decay).clamp_(-config.stw_maximum, config.stw_maximum)
                    model.delta.mul_(1.0 - model.protected_mask)
            mark_step(device)


@torch.no_grad()
def _consolidate_boundary(model, fraction, *, seed, shuffled):
    active = torch.nonzero(model.active_mask.flatten().bool(), as_tuple=False).flatten()
    count = max(1, round(active.numel() * fraction))
    if shuffled:
        order = torch.randperm(active.numel(), generator=torch.Generator().manual_seed(seed + 94_000)).to(active.device)
        selected = active[order[:count]]
    else:
        utility = model.delta.detach().abs().flatten()[active]
        selected = active[torch.topk(utility, count).indices]
    model.ltw.flatten()[selected] += model.delta.flatten()[selected]
    model.delta.flatten()[selected] = 0.0
    model.protected_mask.flatten()[selected] = 1.0


def _attach_comparisons(records):
    lookup = {(int(r["seed"]), r["arm"]): r for r in records}
    for row in records:
        single = lookup.get((int(row["seed"]), "single_memory_control"))
        shuffled = lookup.get((int(row["seed"]), "boundary_shuffled_dual_memory"))
        row["a_retention_gain_vs_single"] = float(row["after_b_a_accuracy"] - single["after_b_a_accuracy"]) if single else 0.0
        row["b_accuracy_gain_vs_single"] = float(row["after_b_b_accuracy"] - single["after_b_b_accuracy"]) if single else 0.0
        row["stability_plasticity_gain_vs_single"] = float(row["stability_plasticity_score"] - single["stability_plasticity_score"]) if single else 0.0
        row["selection_identity_margin"] = float(row["stability_plasticity_score"] - shuffled["stability_plasticity_score"]) if shuffled and row["arm"] == "boundary_selective_dual_memory" else 0.0


def summarize_gen23(records):
    rows = list(records); summary = []
    for arm in GEN23_ARMS:
        group = [r for r in rows if r["arm"] == arm]
        if not group: continue
        summary.append({
            "arm": arm, "seeds": len(group),
            "mean_after_b_a_accuracy": statistics.fmean(float(r["after_b_a_accuracy"]) for r in group),
            "mean_after_b_b_accuracy": statistics.fmean(float(r["after_b_b_accuracy"]) for r in group),
            "mean_clean_retention_drop": statistics.fmean(float(r["clean_retention_drop"]) for r in group),
            "mean_stability_plasticity_score": statistics.fmean(float(r["stability_plasticity_score"]) for r in group),
            "mean_a_retention_gain_vs_single": statistics.fmean(float(r["a_retention_gain_vs_single"]) for r in group),
            "mean_b_accuracy_gain_vs_single": statistics.fmean(float(r["b_accuracy_gain_vs_single"]) for r in group),
            "mean_ltw_causal_margin": statistics.fmean(float(r["ltw_causal_margin"]) for r in group),
            "mean_selection_identity_margin": statistics.fmean(float(r["selection_identity_margin"]) for r in group),
            "maximum_absolute_boundary_accuracy_change": max(abs(float(r["boundary_accuracy_change"])) for r in group),
            "protected_slots": int(group[0]["protected_slots"]), "active_slots": int(group[0]["active_slots"]),
            "allocated_slots": int(group[0]["allocated_slots"]), "adapter_memory_bytes": int(group[0]["adapter_memory_bytes"]),
        })
    return summary


def decide_gen23(records, config):
    dual = [r for r in records if r["arm"] == "boundary_selective_dual_memory"]
    qualifying = [r for r in dual if (
        float(r["a_retention_gain_vs_single"]) >= config.minimum_a_retention_gain_vs_single
        and float(r["b_accuracy_gain_vs_single"]) >= -config.maximum_b_accuracy_cost_vs_single
        and float(r["ltw_causal_margin"]) >= config.minimum_ltw_causal_margin
        and float(r["selection_identity_margin"]) >= config.minimum_selection_identity_margin
        and abs(float(r["boundary_accuracy_change"])) <= config.maximum_boundary_accuracy_change
    )]
    summary = next((r for r in summarize_gen23(records) if r["arm"] == "boundary_selective_dual_memory"), None)
    aggregate = bool(summary) and (
        float(summary["mean_a_retention_gain_vs_single"]) >= config.minimum_a_retention_gain_vs_single
        and float(summary["mean_b_accuracy_gain_vs_single"]) >= -config.maximum_b_accuracy_cost_vs_single
        and float(summary["mean_ltw_causal_margin"]) >= config.minimum_ltw_causal_margin
        and float(summary["mean_selection_identity_margin"]) >= config.minimum_selection_identity_margin
        and float(summary["maximum_absolute_boundary_accuracy_change"]) <= config.maximum_boundary_accuracy_change
    )
    passed = aggregate and len(qualifying) >= config.minimum_qualifying_seed_count
    return {
        "status": "pass" if passed else "stop", "qualified_seed_count": len(qualifying),
        "aggregate_gate_passed": bool(aggregate), "boundary_gated_dual_memory_supported": bool(passed),
        "whole_network_memory_claim_supported": False, "hardware_energy_claim_authorized": False,
        "next_milestone": "backbone_synapse_consolidation" if passed else "close_dual_memory_readout_program",
    }


def _gen22_config(config):
    return Gen22Config(
        seeds=config.seeds, input_neurons=config.input_neurons, classes=config.classes,
        timesteps=config.timesteps, duration_seconds=config.duration_seconds,
        data_root=config.data_root, download=config.download,
        source_train_samples=config.source_train_samples, validation_samples=config.validation_samples,
        test_samples=config.test_samples, source_epochs=config.source_epochs,
        adaptation_epochs_per_shift=config.adaptation_epochs_per_shift, batch_size=config.batch_size,
        source_learning_rate=config.source_learning_rate, adaptation_learning_rate=config.adaptation_learning_rate,
        weight_decay=config.weight_decay, target_parameters=config.target_parameters,
        temporal_conv_kernel_size=config.temporal_conv_kernel_size, temporal_levels=config.temporal_levels,
        surrogate_slope=config.surrogate_slope, sensor_damage_fraction=config.sensor_damage_fraction,
        delay_slots=config.delay_slots, active_slot_fraction=config.active_slot_fraction,
        stw_decay=config.stw_decay, consolidation_rate=0.0,
    )


def plot_gen23(result, path):
    import matplotlib.pyplot as plt
    labels = [r["arm"].replace("_", "\n") for r in result.summary]; x = list(range(len(labels)))
    fig, axes = plt.subplots(3, 1, figsize=(14, 13), constrained_layout=True)
    axes[0].bar(x, [100*r["mean_after_b_a_accuracy"] for r in result.summary], label="A retention", color="#35b4f2")
    axes[0].bar(x, [100*r["mean_after_b_b_accuracy"] for r in result.summary], alpha=.55, label="B accuracy", color="#ffb31a")
    axes[0].set_title("AMMC Gen-23 boundary-gated consolidation"); axes[0].set_ylabel("Accuracy (%)"); axes[0].legend()
    axes[1].bar(x, [100*r["mean_a_retention_gain_vs_single"] for r in result.summary], color="#167d55"); axes[1].set_ylabel("A retention gain vs single")
    axes[2].bar(x, [100*r["mean_selection_identity_margin"] for r in result.summary], color="#8b6fd6"); axes[2].set_ylabel("Selection identity margin")
    for axis in axes: axis.set_xticks(x, labels); axis.grid(axis="y", alpha=.25)
    destination=pathlib.Path(path); destination.parent.mkdir(parents=True, exist_ok=True); fig.savefig(destination,dpi=180); plt.close(fig)


def bundle_gen23_artifacts(paths, output_dir):
    output=pathlib.Path(output_dir); files=[pathlib.Path(v) for v in paths.values() if pathlib.Path(v).is_file()]
    manifest=output/"gen23_boundary_consolidation_manifest.json"
    manifest.write_text(json.dumps({"files":[{"name":f.name,"sha256":hashlib.sha256(f.read_bytes()).hexdigest()} for f in files]},indent=2)+"\n",encoding="utf-8")
    archive=output/"gen23_boundary_consolidation_bundle.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as bundle:
        for file in files+[manifest]: bundle.write(file,arcname=file.name)
    return {"manifest":str(manifest),"bundle":str(archive)}


def _save_progress(path, config, records):
    if path is None:return
    destination=pathlib.Path(path);destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps({"config":asdict(config),"records":records},indent=2)+"\n",encoding="utf-8")


def _load_progress(path, config):
    if path is None or not pathlib.Path(path).exists():return []
    try:payload=json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError):return []
    return list(payload.get("records",[])) if payload.get("config")==json.loads(json.dumps(asdict(config))) else []


def _validate_config(config):
    if config.input_neurons != 700 or config.classes != 35:raise ValueError("Gen-23 is frozen for SSC")
    if len(config.seeds)<5:raise ValueError("Gen-23 requires at least five seeds")
    if not 0<config.protected_fraction<1 or config.stw_maximum<=0:raise ValueError("invalid consolidation bounds")
    disjoint_sensor_damage_indices(config.input_neurons,config.sensor_damage_fraction,seed=0)


def _write_csv(path,rows):
    if not rows:pathlib.Path(path).write_text("",encoding="utf-8");return
    with pathlib.Path(path).open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
