"""Gen-12 frozen robust backbone with context-gated associative memory."""

from __future__ import annotations

import copy
import csv
from dataclasses import asdict, dataclass, replace
import gc
import json
import math
import pathlib
import statistics
import time
from typing import Iterable

from .event_mnist import torch
from .gen11_plastic_adapter import (
    _adaptation_curve,
    _curve_auc,
    _measure_mode,
    _train_dropout_backbone,
)
from .gen9_continual_adaptation import apply_sensor_damage, sensor_damage_indices
from .milestone_a_architecture import _load_progress, _multiscale_features
from .runtime import device_kind, mark_step, resolve_device, sync
from .shd_benchmark import SHDConfig, _measure
from .shd_temporal_pyramid import DEFAULT_TEMPORAL_LEVELS
from .ssc_benchmark import load_ssc_tensors


GEN12_MEMORY_STRATEGIES = (
    "dropout_tcn_static",
    "dropout_tcn_readout",
    "dropout_tcn_full_finetune",
    "dense_prototype_memory",
    "spiking_prototype_memory",
)


def available_gen12_memory_strategies() -> tuple[str, ...]:
    return GEN12_MEMORY_STRATEGIES


def top_fraction_spike_code(features, fraction: float):
    """Encode each feature vector with a fixed sparse rank-order spike code."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("spike fraction must be in (0, 1]")
    count = max(1, int(math.ceil(features.shape[1] * fraction)))
    indices = torch.topk(features, k=count, dim=1).indices
    code = torch.zeros_like(features)
    return code.scatter_(1, indices, 1.0)


@dataclass
class Gen12AssociativeMemoryResult:
    config: SHDConfig
    device: str
    target_parameters: int
    temporal_levels: tuple[int, ...]
    seeds: tuple[int, ...]
    source_mask_fraction: float
    damage_fraction: float
    damage_seed: int
    adaptation_budgets: tuple[int, ...]
    memory_mix: float
    memory_temperature: float
    spike_fraction: float
    records: list[dict]
    summary: list[dict]
    decision: dict

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "gen12_associative_memory.json"
        records_path = output / "gen12_associative_memory_records.csv"
        summary_path = output / "gen12_associative_memory_summary.csv"
        payload = {
            "config": asdict(self.config),
            "device": self.device,
            "target_parameters": self.target_parameters,
            "temporal_levels": list(self.temporal_levels),
            "seeds": list(self.seeds),
            "source_mask_fraction": self.source_mask_fraction,
            "damage_fraction": self.damage_fraction,
            "damage_seed": self.damage_seed,
            "adaptation_budgets": list(self.adaptation_budgets),
            "memory_mix": self.memory_mix,
            "memory_temperature": self.memory_temperature,
            "spike_fraction": self.spike_fraction,
            "records": self.records,
            "summary": self.summary,
            "decision": self.decision,
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        _write_csv(records_path, self.records)
        _write_csv(summary_path, self.summary)
        paths = {
            "json": str(json_path),
            "records_csv": str(records_path),
            "summary_csv": str(summary_path),
        }
        if plot and self.summary:
            plot_path = output / "gen12_associative_memory.png"
            plot_gen12_associative_memory(self.records, self.summary, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def run_gen12_associative_memory(
    config: SHDConfig,
    *,
    seeds: Iterable[int] = (157, 158, 159),
    source_epochs: int = 15,
    source_mask_fraction: float = 0.20,
    damage_fraction: float = 0.35,
    damage_seed: int = 909,
    adaptation_budgets: Iterable[int] = (0, 64, 256, 1024, 4096),
    adaptation_epochs_per_block: int = 3,
    adaptation_learning_rate: float = 0.001,
    memory_mix: float = 0.50,
    memory_temperature: float = 0.10,
    spike_fraction: float = 0.20,
    minimum_shift_drop: float = 0.02,
    minimum_adaptation_gain: float = 0.02,
    auc_margin: float = 0.01,
    final_accuracy_margin: float = 0.01,
    forgetting_margin: float = 0.005,
    causal_margin: float = 0.005,
    minimum_spike_density: float = 0.05,
    maximum_spike_density: float = 0.35,
    target_parameters: int = 133_631,
    device="auto",
    temporal_levels: Iterable[int] = DEFAULT_TEMPORAL_LEVELS,
    input_kernel_size: int = 5,
    hidden_kernel_size: int = 3,
    tcn_dilation: int = 2,
    progress_path: str | pathlib.Path | None = None,
) -> Gen12AssociativeMemoryResult:
    if torch is None:
        raise ImportError("Gen-12 associative memory requires PyTorch")
    seed_values = tuple(int(seed) for seed in seeds)
    budgets = tuple(int(value) for value in adaptation_budgets)
    levels = tuple(int(value) for value in temporal_levels)
    _validate_run(
        config, seed_values, budgets, levels, source_epochs,
        source_mask_fraction, damage_fraction, adaptation_epochs_per_block,
        adaptation_learning_rate, memory_mix, memory_temperature, spike_fraction,
        (minimum_shift_drop, minimum_adaptation_gain, auc_margin,
         final_accuracy_margin, forgetting_margin, causal_margin),
        (minimum_spike_density, maximum_spike_density),
    )
    signature = _run_signature(
        config, seeds=seed_values, source_epochs=source_epochs,
        source_mask_fraction=source_mask_fraction, damage_fraction=damage_fraction,
        damage_seed=damage_seed, budgets=budgets,
        adaptation_epochs_per_block=adaptation_epochs_per_block,
        adaptation_learning_rate=adaptation_learning_rate,
        memory_mix=memory_mix, memory_temperature=memory_temperature,
        spike_fraction=spike_fraction,
        gates=(minimum_shift_drop, minimum_adaptation_gain, auc_margin,
               final_accuracy_margin, forgetting_margin, causal_margin,
               minimum_spike_density, maximum_spike_density),
        target_parameters=target_parameters, levels=levels,
        kernels=(input_kernel_size, hidden_kernel_size), tcn_dilation=tcn_dilation,
    )
    progress = _load_progress(progress_path, signature)
    resolved = resolve_device(device)
    full_config = replace(config, train_samples=0, test_samples=0, epochs=source_epochs)
    if progress.get("stage") == "complete":
        records = list(progress.get("records", []))
        summary = summarize_gen12_associative_memory(records, budgets=budgets)
        decision = progress.get("decision") or decide_gen12_associative_memory(
            summary, minimum_shift_drop=minimum_shift_drop,
            minimum_adaptation_gain=minimum_adaptation_gain, auc_margin=auc_margin,
            final_accuracy_margin=final_accuracy_margin,
            forgetting_margin=forgetting_margin, causal_margin=causal_margin,
            minimum_spike_density=minimum_spike_density,
            maximum_spike_density=maximum_spike_density,
        )
        return _result(
            full_config, resolved, target_parameters, levels, seed_values,
            source_mask_fraction, damage_fraction, damage_seed, budgets,
            memory_mix, memory_temperature, spike_fraction, records, summary, decision,
        )

    train_events, train_labels, validation_events, validation_labels, test_events, test_labels = load_ssc_tensors(
        full_config, validation_samples=0
    )
    fixed_mask = sensor_damage_indices(config.input_neurons, damage_fraction, seed=damage_seed)
    records = _run_all_strategies(
        seed_values, full_config, train_events, train_labels,
        validation_events, validation_labels, test_events, test_labels,
        fixed_mask=fixed_mask, source_mask_fraction=source_mask_fraction,
        budgets=budgets, adaptation_epochs_per_block=adaptation_epochs_per_block,
        adaptation_learning_rate=adaptation_learning_rate,
        memory_mix=memory_mix, memory_temperature=memory_temperature,
        spike_fraction=spike_fraction, target_parameters=target_parameters,
        levels=levels, input_kernel_size=input_kernel_size,
        hidden_kernel_size=hidden_kernel_size, tcn_dilation=tcn_dilation,
        device=resolved, existing_records=progress.get("records", ()),
        progress_callback=lambda rows: _save_progress(
            progress_path, signature, stage="adaptation", records=rows
        ),
    )
    summary = summarize_gen12_associative_memory(records, budgets=budgets)
    decision = decide_gen12_associative_memory(
        summary, minimum_shift_drop=minimum_shift_drop,
        minimum_adaptation_gain=minimum_adaptation_gain, auc_margin=auc_margin,
        final_accuracy_margin=final_accuracy_margin,
        forgetting_margin=forgetting_margin, causal_margin=causal_margin,
        minimum_spike_density=minimum_spike_density,
        maximum_spike_density=maximum_spike_density,
    )
    _save_progress(progress_path, signature, stage="complete", records=records, decision=decision)
    return _result(
        full_config, resolved, target_parameters, levels, seed_values,
        source_mask_fraction, damage_fraction, damage_seed, budgets,
        memory_mix, memory_temperature, spike_fraction, records, summary, decision,
    )


def summarize_gen12_associative_memory(
    records: Iterable[dict], *, budgets: Iterable[int]
) -> list[dict]:
    rows = list(records)
    if not rows:
        return []
    max_budget = max(int(value) for value in budgets)
    summary = []
    for strategy in GEN12_MEMORY_STRATEGIES:
        group = [row for row in rows if row["strategy"] == strategy]
        if not group:
            continue
        per_seed = []
        for seed in sorted({int(row["seed"]) for row in group}):
            curve = sorted(
                (row for row in group if int(row["seed"]) == seed),
                key=lambda row: int(row["adaptation_samples"]),
            )
            initial = next(row for row in curve if int(row["adaptation_samples"]) == 0)
            final = next(row for row in curve if int(row["adaptation_samples"]) == max_budget)
            per_seed.append({
                "source_initial": float(initial["source_accuracy"]),
                "shifted_initial": float(initial["shifted_accuracy"]),
                "source_final": float(final["source_accuracy"]),
                "shifted_final": float(final["shifted_accuracy"]),
                "gain": float(final["shifted_accuracy"]) - float(initial["shifted_accuracy"]),
                "forgetting": float(initial["source_accuracy"]) - float(final["source_accuracy"]),
                "auc": _curve_auc(curve, max_budget),
                "activity": float(final["activity"]),
                "memory_contribution": float(final.get("memory_contribution") or 0.0),
                "association_specificity": float(final.get("association_specificity") or 0.0),
                "throughput": float(final["test_examples_per_second"]),
                "seconds": float(final["cumulative_adaptation_seconds"]),
                "cells": int(final.get("active_memory_cells") or 0),
            })
        summary.append({
            "strategy": strategy,
            "runs": len(per_seed),
            "mean_source_initial_accuracy": statistics.fmean(item["source_initial"] for item in per_seed),
            "mean_shifted_initial_accuracy": statistics.fmean(item["shifted_initial"] for item in per_seed),
            "mean_shift_drop": statistics.fmean(item["source_initial"] - item["shifted_initial"] for item in per_seed),
            "mean_source_final_accuracy": statistics.fmean(item["source_final"] for item in per_seed),
            "mean_shifted_final_accuracy": statistics.fmean(item["shifted_final"] for item in per_seed),
            "mean_adaptation_gain": statistics.fmean(item["gain"] for item in per_seed),
            "two_point_gain_seed_count": sum(item["gain"] >= 0.02 for item in per_seed),
            "mean_forgetting": statistics.fmean(item["forgetting"] for item in per_seed),
            "mean_adaptation_auc": statistics.fmean(item["auc"] for item in per_seed),
            "mean_activity": statistics.fmean(item["activity"] for item in per_seed),
            "mean_memory_contribution": statistics.fmean(item["memory_contribution"] for item in per_seed),
            "memory_contribution_seed_count": sum(item["memory_contribution"] >= 0.005 for item in per_seed),
            "mean_association_specificity": statistics.fmean(item["association_specificity"] for item in per_seed),
            "association_specificity_seed_count": sum(item["association_specificity"] >= 0.005 for item in per_seed),
            "mean_test_examples_per_second": statistics.fmean(item["throughput"] for item in per_seed),
            "mean_cumulative_adaptation_seconds": statistics.fmean(item["seconds"] for item in per_seed),
            "mean_active_memory_cells": statistics.fmean(item["cells"] for item in per_seed),
            "adaptation_trainable_parameters": int(group[0]["adaptation_trainable_parameters"]),
        })
    return sorted(summary, key=lambda row: (-float(row["mean_adaptation_auc"]), str(row["strategy"])))


def decide_gen12_associative_memory(
    summary: Iterable[dict], *, minimum_shift_drop: float,
    minimum_adaptation_gain: float, auc_margin: float,
    final_accuracy_margin: float, forgetting_margin: float,
    causal_margin: float, minimum_spike_density: float,
    maximum_spike_density: float,
) -> dict:
    lookup = {row["strategy"]: row for row in summary}
    required_names = (
        "dropout_tcn_static", "dropout_tcn_readout", "spiking_prototype_memory"
    )
    if any(name not in lookup for name in required_names):
        return {
            "status": "stop", "qualified_arms": [],
            "reason": "required strategy missing",
            "next_milestone": "close_gen12_associative_memory",
        }
    static, readout, memory = (lookup[name] for name in required_names)
    required = 2 if int(memory["runs"]) >= 3 else 1
    passed = (
        float(static["mean_shift_drop"]) >= minimum_shift_drop
        and float(memory["mean_adaptation_gain"]) >= minimum_adaptation_gain
        and int(memory["two_point_gain_seed_count"]) >= required
        and float(memory["mean_adaptation_auc"]) >= float(readout["mean_adaptation_auc"]) - auc_margin
        and float(memory["mean_shifted_final_accuracy"]) >= float(readout["mean_shifted_final_accuracy"]) - final_accuracy_margin
        and float(memory["mean_forgetting"]) <= float(readout["mean_forgetting"]) + forgetting_margin
        and float(memory["mean_memory_contribution"]) >= causal_margin
        and int(memory["memory_contribution_seed_count"]) >= required
        and float(memory["mean_association_specificity"]) >= causal_margin
        and int(memory["association_specificity_seed_count"]) >= required
        and minimum_spike_density <= float(memory["mean_activity"]) <= maximum_spike_density
    )
    return {
        "status": "pass" if passed else "stop",
        "qualified_arms": ["spiking_prototype_memory"] if passed else [],
        "next_milestone": (
            "context_free_memory_and_consolidation"
            if passed else "close_gen12_associative_memory"
        ),
    }


def plot_gen12_associative_memory(records, summary, path) -> None:
    import matplotlib.pyplot as plt

    budgets = sorted({int(row["adaptation_samples"]) for row in records})
    figure, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    for strategy in GEN12_MEMORY_STRATEGIES:
        group = [row for row in records if row["strategy"] == strategy]
        if not group:
            continue
        shifted = [
            statistics.fmean(
                float(row["shifted_accuracy"])
                for row in group if int(row["adaptation_samples"]) == budget
            ) for budget in budgets
        ]
        axes[0, 0].plot(budgets, [100.0 * value for value in shifted], marker="o", label=strategy)
    axes[0, 0].set_title("Damaged-task adaptation")
    axes[0, 0].set_xscale("symlog", linthresh=64)
    axes[0, 0].legend(fontsize=7)
    labels = [row["strategy"].replace("_", "\n") for row in summary]
    axes[0, 1].bar(labels, [100.0 * row["mean_adaptation_auc"] for row in summary])
    axes[0, 1].set_ylabel("Adaptation AUC (%)")
    axes[1, 0].bar(labels, [100.0 * row["mean_memory_contribution"] for row in summary], color="#167d55")
    axes[1, 0].axhline(0.5, color="#bd3d3a", linestyle="--")
    axes[1, 0].set_ylabel("Full - memory removed (points)")
    axes[1, 1].bar(labels, [100.0 * row["mean_association_specificity"] for row in summary], color="#35b4f2")
    axes[1, 1].axhline(0.5, color="#bd3d3a", linestyle="--")
    axes[1, 1].set_ylabel("Full - shuffled associations (points)")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _run_all_strategies(
    seeds, config, train_events, train_labels, validation_events, validation_labels,
    test_events, test_labels, *, fixed_mask, source_mask_fraction, budgets,
    adaptation_epochs_per_block, adaptation_learning_rate, memory_mix,
    memory_temperature, spike_fraction, target_parameters, levels,
    input_kernel_size, hidden_kernel_size, tcn_dilation, device,
    existing_records=(), progress_callback=None,
):
    records = list(existing_records)
    completed = {
        (int(row["seed"]), row["strategy"], int(row["adaptation_samples"]))
        for row in records
    }
    for seed in seeds:
        expected = {
            (seed, strategy, budget)
            for strategy in GEN12_MEMORY_STRATEGIES for budget in budgets
        }
        if expected.issubset(completed):
            continue
        backbone = _train_dropout_backbone(
            seed, config, train_events, train_labels,
            validation_events, validation_labels,
            source_mask_fraction=source_mask_fraction,
            target_parameters=target_parameters, levels=levels,
            input_kernel_size=input_kernel_size,
            hidden_kernel_size=hidden_kernel_size,
            tcn_dilation=tcn_dilation, device=device,
        )
        for strategy in GEN12_MEMORY_STRATEGIES:
            if {(seed, strategy, budget) for budget in budgets}.issubset(completed):
                continue
            if strategy in {
                "dropout_tcn_static", "dropout_tcn_readout",
                "dropout_tcn_full_finetune",
            }:
                model = copy.deepcopy(backbone)
                new_rows = _adaptation_curve(
                    model, strategy, seed, validation_events, validation_labels,
                    test_events, test_labels, fixed_mask=fixed_mask,
                    budgets=budgets, epochs_per_block=adaptation_epochs_per_block,
                    learning_rate=adaptation_learning_rate,
                    batch_size=config.batch_size, weight_decay=config.weight_decay,
                    device=device,
                )
                for row in new_rows:
                    row.update({
                        "memory_contribution": None,
                        "association_specificity": None,
                        "active_memory_cells": 0,
                    })
                del model
            else:
                new_rows = _prototype_curve(
                    backbone, strategy, seed,
                    validation_events, validation_labels,
                    test_events, test_labels, fixed_mask=fixed_mask,
                    budgets=budgets, levels=levels, memory_mix=memory_mix,
                    memory_temperature=memory_temperature,
                    spike_fraction=spike_fraction,
                    batch_size=config.batch_size, device=device,
                )
            records = [
                row for row in records
                if not (int(row["seed"]) == seed and row["strategy"] == strategy)
            ]
            records.extend(new_rows)
            completed.update((seed, strategy, budget) for budget in budgets)
            if progress_callback is not None:
                progress_callback(records)
        del backbone
        gc.collect()
    return records


def _prototype_curve(
    backbone, strategy, seed, adaptation_events, adaptation_labels,
    test_events, test_labels, *, fixed_mask, budgets, levels,
    memory_mix, memory_temperature, spike_fraction, batch_size, device,
):
    spiking = strategy == "spiking_prototype_memory"
    generator = torch.Generator(device="cpu").manual_seed(seed + 113_000)
    order = torch.randperm(adaptation_events.shape[0], generator=generator)
    feature_dim = backbone.channels * sum(levels)
    sums = torch.zeros((backbone.config.classes, feature_dim), device=device)
    counts = torch.zeros((backbone.config.classes,), device=device)
    rows = []
    previous = 0
    cumulative_seconds = 0.0
    source_accuracy, _, _ = _measure(
        backbone, test_events, test_labels, batch_size, device
    )
    for budget in budgets:
        if budget > previous:
            sync(device)
            started = time.perf_counter()
            indices = order[previous:budget]
            for offset in range(0, indices.shape[0], batch_size):
                index = indices[offset: offset + batch_size]
                events = apply_sensor_damage(
                    adaptation_events.index_select(0, index), fixed_mask
                ).to(device)
                labels = adaptation_labels.index_select(0, index).to(device)
                with torch.no_grad():
                    features, _ = _backbone_features(backbone, events, levels)
                    codes = top_fraction_spike_code(features, spike_fraction) if spiking else features
                    sums.index_add_(0, labels, codes)
                    counts.index_add_(0, labels, torch.ones_like(labels, dtype=counts.dtype))
                    mark_step(device)
            sync(device)
            cumulative_seconds += time.perf_counter() - started
        prototypes = sums / counts.clamp_min(1.0).unsqueeze(1)
        shifted_accuracy, elapsed, activity = _measure_memory(
            backbone, test_events, test_labels, fixed_mask=fixed_mask,
            prototypes=prototypes, counts=counts, levels=levels,
            memory_mix=memory_mix, temperature=memory_temperature,
            spike_fraction=spike_fraction, spiking=spiking,
            batch_size=batch_size, device=device, mode="full",
        )
        removed_accuracy = _measure_memory(
            backbone, test_events, test_labels, fixed_mask=fixed_mask,
            prototypes=prototypes, counts=counts, levels=levels,
            memory_mix=memory_mix, temperature=memory_temperature,
            spike_fraction=spike_fraction, spiking=spiking,
            batch_size=batch_size, device=device, mode="removed",
        )[0]
        shuffled_accuracy = _measure_memory(
            backbone, test_events, test_labels, fixed_mask=fixed_mask,
            prototypes=prototypes.roll(shifts=1, dims=0),
            counts=counts.roll(shifts=1, dims=0),
            levels=levels, memory_mix=memory_mix,
            temperature=memory_temperature, spike_fraction=spike_fraction,
            spiking=spiking, batch_size=batch_size, device=device, mode="full",
        )[0]
        active_classes = int((counts > 0).sum().item())
        rows.append({
            "seed": seed,
            "strategy": strategy,
            "adaptation_samples": int(budget),
            "source_accuracy": source_accuracy,
            "shifted_accuracy": shifted_accuracy,
            "activity": activity,
            "state_contribution": None,
            "state_specificity": None,
            "mean_absolute_gate": 0.0,
            "memory_contribution": shifted_accuracy - removed_accuracy,
            "association_specificity": shifted_accuracy - shuffled_accuracy,
            "active_memory_cells": active_classes * feature_dim,
            "test_examples_per_second": test_events.shape[0] / max(elapsed, 1e-12),
            "cumulative_adaptation_seconds": cumulative_seconds,
            "adaptation_trainable_parameters": 0,
        })
        previous = int(budget)
    return rows


def _backbone_features(backbone, events, levels):
    direct = torch.relu(backbone.input_conv(events.to(torch.float32).transpose(1, 2)))
    trace = torch.relu(backbone.hidden_conv(direct) + direct).transpose(1, 2)
    features = torch.cat(_multiscale_features(trace, levels), dim=1)
    return features, backbone.classifier(features)


def _measure_memory(
    backbone, events, labels, *, fixed_mask, prototypes, counts, levels,
    memory_mix, temperature, spike_fraction, spiking, batch_size, device, mode,
):
    backbone.eval()
    correct = total = 0
    activity_sum = 0.0
    sync(device)
    started = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, events.shape[0], batch_size):
            batch_events = apply_sensor_damage(
                events[offset: offset + batch_size], fixed_mask
            ).to(device)
            batch_labels = labels[offset: offset + batch_size].to(device)
            features, base_logits = _backbone_features(backbone, batch_events, levels)
            code = top_fraction_spike_code(features, spike_fraction) if spiking else features
            activity_sum += float((code != 0).to(torch.float32).mean().item()) * int(batch_labels.shape[0])
            logits = base_logits
            if mode == "full" and bool((counts > 0).any().item()):
                query = torch.nn.functional.normalize(code, dim=1)
                keys = torch.nn.functional.normalize(prototypes, dim=1)
                similarity = query @ keys.transpose(0, 1)
                similarity[:, counts <= 0] = -1e9
                memory_probability = torch.softmax(similarity / temperature, dim=1)
                base_probability = torch.softmax(base_logits, dim=1)
                logits = torch.log(
                    ((1.0 - memory_mix) * base_probability + memory_mix * memory_probability).clamp_min(1e-8)
                )
            correct += int((logits.argmax(dim=1) == batch_labels).sum().item())
            total += int(batch_labels.shape[0])
            mark_step(device)
    sync(device)
    return (
        correct / max(total, 1),
        time.perf_counter() - started,
        activity_sum / max(total, 1),
    )


def _result(
    config, device, target_parameters, levels, seeds, source_mask_fraction,
    damage_fraction, damage_seed, budgets, memory_mix, memory_temperature,
    spike_fraction, records, summary, decision,
):
    return Gen12AssociativeMemoryResult(
        config=config, device=device_kind(device), target_parameters=target_parameters,
        temporal_levels=levels, seeds=seeds,
        source_mask_fraction=source_mask_fraction,
        damage_fraction=damage_fraction, damage_seed=damage_seed,
        adaptation_budgets=budgets, memory_mix=memory_mix,
        memory_temperature=memory_temperature, spike_fraction=spike_fraction,
        records=records, summary=summary, decision=decision,
    )


def _validate_run(
    config, seeds, budgets, levels, source_epochs, source_mask_fraction,
    damage_fraction, adaptation_epochs, adaptation_lr, memory_mix,
    memory_temperature, spike_fraction, gates, spike_gate,
):
    if not seeds or not levels or source_epochs <= 0 or adaptation_epochs <= 0:
        raise ValueError("seeds, levels, and epochs must be positive")
    if not budgets or budgets[0] != 0 or tuple(sorted(set(budgets))) != budgets:
        raise ValueError("budgets must be unique, increasing, and start at zero")
    if budgets[-1] > 9981 or adaptation_lr <= 0.0:
        raise ValueError("invalid adaptation budget or learning rate")
    if not 0.0 < source_mask_fraction < 1.0 or not 0.0 < damage_fraction < 1.0:
        raise ValueError("invalid mask fraction")
    if not 0.0 <= memory_mix <= 1.0 or memory_temperature <= 0.0:
        raise ValueError("invalid memory mixture")
    if not 0.0 < spike_fraction <= 1.0:
        raise ValueError("invalid spike fraction")
    if any(not 0.0 <= value <= 1.0 for value in gates):
        raise ValueError("invalid gate")
    if not 0.0 <= spike_gate[0] <= spike_gate[1] <= 1.0:
        raise ValueError("invalid spike-density gate")
    if config.input_neurons <= 0:
        raise ValueError("invalid input size")


def _run_signature(config, **values):
    signature = {
        "version": 1,
        "experiment": "gen12_associative_memory",
        "strategies": list(GEN12_MEMORY_STRATEGIES),
        "input_neurons": config.input_neurons,
        "classes": config.classes,
        "timesteps": config.timesteps,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "data_root": str(config.data_root),
        "data_seed": config.data_seed,
    }
    for key, value in values.items():
        signature[key] = list(value) if isinstance(value, tuple) else value
    return signature


def _save_progress(path, signature, *, stage, records, decision=None):
    if path is None:
        return
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "signature": signature,
        "stage": stage,
        "records": list(records),
        "decision": decision,
    }
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
