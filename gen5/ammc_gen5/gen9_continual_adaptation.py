"""Gen-9 controlled sensor-damage continual-adaptation experiment."""

from __future__ import annotations

import copy
import csv
from dataclasses import asdict, dataclass, replace
import gc
import json
import pathlib
import statistics
import time
from typing import Iterable

from .event_mnist import torch
from .gen7_predictive_state import (
    _build_model,
    _measure_sample_gate,
    _train_predictive_validation_selected,
)
from .milestone_a_architecture import _load_progress, _sample_split
from .runtime import device_kind, mark_step, resolve_device, seed_everything, sync
from .shd_benchmark import SHDConfig, _measure
from .shd_temporal_pyramid import DEFAULT_TEMPORAL_LEVELS
from .ssc_benchmark import load_ssc_tensors


@dataclass(frozen=True)
class Gen9SourceArm:
    name: str
    model_kind: str
    conventional: bool
    causal_state: bool
    dynamics: str | None
    predictive_weight: float
    shuffled_future_targets: bool = False


GEN9_SOURCE_ARMS = (
    Gen9SourceArm("dilated_tcn", "tcn", True, False, None, 0.0),
    Gen9SourceArm(
        "predictive_lif", "predictive_lif", False, True, "lif", 0.20
    ),
)

GEN9_ADAPTATION_STRATEGIES = (
    "tcn_static",
    "tcn_readout",
    "tcn_full_finetune",
    "predictive_lif_static",
    "predictive_lif_readout",
)


def available_gen9_source_arms() -> tuple[str, ...]:
    return tuple(arm.name for arm in GEN9_SOURCE_ARMS)


def available_gen9_adaptation_strategies() -> tuple[str, ...]:
    return GEN9_ADAPTATION_STRATEGIES


def sensor_damage_indices(
    input_neurons: int, fraction: float, *, seed: int
) -> tuple[int, ...]:
    """Return a deterministic fixed sensor mask without touching global RNG state."""

    if torch is None:
        raise ImportError("Gen-9 sensor masking requires PyTorch")
    if input_neurons <= 0 or not 0.0 < fraction < 1.0:
        raise ValueError("input_neurons and damage fraction must be valid")
    count = max(1, min(input_neurons - 1, round(input_neurons * fraction)))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return tuple(
        int(value)
        for value in torch.randperm(input_neurons, generator=generator)[:count].tolist()
    )


def apply_sensor_damage(events, indices: Iterable[int]):
    """Zero the same sensor bank for every sample and timestep."""

    if torch is None:
        raise ImportError("Gen-9 sensor masking requires PyTorch")
    if events.ndim != 3:
        raise ValueError("events must have shape [batch, time, input_neurons]")
    damaged = events.clone()
    selected = tuple(int(index) for index in indices)
    if selected:
        damaged[:, :, list(selected)] = 0
    return damaged


@dataclass
class Gen9ContinualAdaptationResult:
    config: SHDConfig
    device: str
    target_parameters: int
    temporal_levels: tuple[int, ...]
    screen_seed: int
    confirm_seeds: tuple[int, ...]
    damage_fraction: float
    damage_seed: int
    adaptation_budgets: tuple[int, ...]
    screen_records: list[dict]
    promoted_source_arms: tuple[str, ...]
    adaptation_records: list[dict]
    adaptation_summary: list[dict]
    decision: dict

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "gen9_continual_adaptation.json"
        screen_path = output / "gen9_continual_adaptation_screen.csv"
        records_path = output / "gen9_continual_adaptation_records.csv"
        summary_path = output / "gen9_continual_adaptation_summary.csv"
        payload = {
            "config": asdict(self.config),
            "device": self.device,
            "target_parameters": self.target_parameters,
            "temporal_levels": list(self.temporal_levels),
            "screen_seed": self.screen_seed,
            "confirm_seeds": list(self.confirm_seeds),
            "damage_fraction": self.damage_fraction,
            "damage_seed": self.damage_seed,
            "adaptation_budgets": list(self.adaptation_budgets),
            "screen_records": self.screen_records,
            "promoted_source_arms": list(self.promoted_source_arms),
            "adaptation_records": self.adaptation_records,
            "adaptation_summary": self.adaptation_summary,
            "decision": self.decision,
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        _write_csv(screen_path, self.screen_records)
        _write_csv(records_path, self.adaptation_records)
        _write_csv(summary_path, self.adaptation_summary)
        paths = {
            "json": str(json_path),
            "screen_csv": str(screen_path),
            "records_csv": str(records_path),
            "summary_csv": str(summary_path),
        }
        if plot and self.adaptation_summary:
            plot_path = output / "gen9_continual_adaptation.png"
            plot_gen9_continual_adaptation(
                self.adaptation_records, self.adaptation_summary, plot_path
            )
            paths["plot"] = str(plot_path)
        return paths


def run_gen9_continual_adaptation(
    config: SHDConfig,
    *,
    screen_seed: int = 148,
    confirm_seeds: Iterable[int] = (148, 149, 150),
    screen_train_samples: int = 15_000,
    screen_validation_samples: int = 3_000,
    screen_test_samples: int = 3_000,
    screen_epochs: int = 4,
    confirm_epochs: int = 15,
    adaptation_budgets: Iterable[int] = (0, 64, 256, 1024, 4096),
    adaptation_epochs_per_block: int = 3,
    adaptation_learning_rate: float = 0.001,
    damage_fraction: float = 0.35,
    damage_seed: int = 909,
    promotion_margin: float = 0.01,
    minimum_parameter_ratio: float = 0.95,
    maximum_parameter_ratio: float = 1.05,
    minimum_spike_rate: float = 0.01,
    maximum_spike_rate: float = 0.30,
    minimum_shift_drop: float = 0.05,
    minimum_adaptation_gain: float = 0.02,
    minimum_auc_advantage: float = 0.01,
    accuracy_margin: float = 0.01,
    forgetting_margin: float = 0.005,
    target_parameters: int = 133_631,
    device="auto",
    temporal_levels: Iterable[int] = DEFAULT_TEMPORAL_LEVELS,
    input_kernel_size: int = 5,
    hidden_kernel_size: int = 3,
    tcn_dilation: int = 2,
    surrogate_slope: float = 10.0,
    future_horizon: int = 4,
    contrastive_temperature: float = 0.10,
    progress_path: str | pathlib.Path | None = None,
) -> Gen9ContinualAdaptationResult:
    if torch is None:
        raise ImportError("Gen-9 continual adaptation requires PyTorch")
    levels = tuple(int(level) for level in temporal_levels)
    seeds = tuple(int(seed) for seed in confirm_seeds)
    budgets = tuple(int(value) for value in adaptation_budgets)
    _validate_run(
        config,
        levels,
        seeds,
        budgets,
        screen_epochs,
        confirm_epochs,
        adaptation_epochs_per_block,
        adaptation_learning_rate,
        (screen_train_samples, screen_validation_samples, screen_test_samples),
        damage_fraction,
        (minimum_parameter_ratio, maximum_parameter_ratio),
        (minimum_spike_rate, maximum_spike_rate),
        (
            promotion_margin,
            minimum_shift_drop,
            minimum_adaptation_gain,
            minimum_auc_advantage,
            accuracy_margin,
            forgetting_margin,
        ),
    )
    signature = _run_signature(
        config,
        screen_seed=screen_seed,
        confirm_seeds=seeds,
        screen_samples=(screen_train_samples, screen_validation_samples, screen_test_samples),
        epochs=(screen_epochs, confirm_epochs),
        adaptation_budgets=budgets,
        adaptation_epochs_per_block=adaptation_epochs_per_block,
        adaptation_learning_rate=adaptation_learning_rate,
        damage_fraction=damage_fraction,
        damage_seed=damage_seed,
        promotion_margin=promotion_margin,
        parameter_gate=(minimum_parameter_ratio, maximum_parameter_ratio),
        spike_gate=(minimum_spike_rate, maximum_spike_rate),
        minimum_shift_drop=minimum_shift_drop,
        minimum_adaptation_gain=minimum_adaptation_gain,
        minimum_auc_advantage=minimum_auc_advantage,
        accuracy_margin=accuracy_margin,
        forgetting_margin=forgetting_margin,
        target_parameters=target_parameters,
        levels=levels,
        input_kernel_size=input_kernel_size,
        hidden_kernel_size=hidden_kernel_size,
        tcn_dilation=tcn_dilation,
        surrogate_slope=surrogate_slope,
        future_horizon=future_horizon,
        contrastive_temperature=contrastive_temperature,
        device=device,
    )
    progress = _load_progress(progress_path, signature)
    resolved = resolve_device(device)
    full_config = replace(config, train_samples=0, test_samples=0, epochs=confirm_epochs)
    if progress.get("stage") == "complete":
        return _completed_result(
            progress,
            full_config,
            resolved,
            target_parameters,
            levels,
            screen_seed,
            seeds,
            damage_fraction,
            damage_seed,
            budgets,
            minimum_shift_drop,
            minimum_adaptation_gain,
            minimum_auc_advantage,
            accuracy_margin,
            forgetting_margin,
            minimum_spike_rate,
            maximum_spike_rate,
        )

    train_events, train_labels, validation_events, validation_labels, test_events, test_labels = load_ssc_tensors(
        full_config, validation_samples=0
    )
    mask = sensor_damage_indices(
        config.input_neurons, damage_fraction, seed=damage_seed
    )
    screen_records = list(progress.get("screen_records", []))
    adaptation_records = list(progress.get("adaptation_records", []))
    expected = {(int(screen_seed), arm.name) for arm in GEN9_SOURCE_ARMS}
    completed = {(int(row["seed"]), str(row["arm"])) for row in screen_records}
    if not expected.issubset(completed):
        generator = torch.Generator(device="cpu").manual_seed(config.data_seed + 99_000)
        sample_sets = (
            _sample_split(train_events, train_labels, screen_train_samples, generator),
            _sample_split(validation_events, validation_labels, screen_validation_samples, generator),
            _sample_split(test_events, test_labels, screen_test_samples, generator),
        )
        screen_records = _run_source_screen(
            GEN9_SOURCE_ARMS,
            int(screen_seed),
            replace(full_config, epochs=screen_epochs),
            *sample_sets[0],
            *sample_sets[1],
            *sample_sets[2],
            mask=mask,
            target_parameters=target_parameters,
            levels=levels,
            input_kernel_size=input_kernel_size,
            hidden_kernel_size=hidden_kernel_size,
            tcn_dilation=tcn_dilation,
            surrogate_slope=surrogate_slope,
            future_horizon=future_horizon,
            contrastive_temperature=contrastive_temperature,
            device=resolved,
            existing_records=screen_records,
            progress_callback=lambda rows: _save_progress(
                progress_path,
                signature,
                stage="screen",
                screen_records=rows,
                promoted_source_arms=(),
                adaptation_records=adaptation_records,
            ),
        )
        del sample_sets
        gc.collect()

    promoted = select_gen9_promoted_source_arms(
        screen_records,
        promotion_margin=promotion_margin,
        minimum_parameter_ratio=minimum_parameter_ratio,
        maximum_parameter_ratio=maximum_parameter_ratio,
        minimum_spike_rate=minimum_spike_rate,
        maximum_spike_rate=maximum_spike_rate,
    )
    _save_progress(
        progress_path,
        signature,
        stage="adaptation",
        screen_records=screen_records,
        promoted_source_arms=promoted,
        adaptation_records=adaptation_records,
    )
    adaptation_records = _run_confirmation(
        promoted,
        seeds,
        full_config,
        train_events,
        train_labels,
        validation_events,
        validation_labels,
        test_events,
        test_labels,
        mask=mask,
        budgets=budgets,
        adaptation_epochs_per_block=adaptation_epochs_per_block,
        adaptation_learning_rate=adaptation_learning_rate,
        target_parameters=target_parameters,
        levels=levels,
        input_kernel_size=input_kernel_size,
        hidden_kernel_size=hidden_kernel_size,
        tcn_dilation=tcn_dilation,
        surrogate_slope=surrogate_slope,
        future_horizon=future_horizon,
        contrastive_temperature=contrastive_temperature,
        device=resolved,
        existing_records=adaptation_records,
        progress_callback=lambda rows: _save_progress(
            progress_path,
            signature,
            stage="adaptation",
            screen_records=screen_records,
            promoted_source_arms=promoted,
            adaptation_records=rows,
        ),
    )
    summary = summarize_gen9_adaptation(adaptation_records, budgets=budgets)
    decision = decide_gen9_continual_adaptation(
        summary,
        minimum_shift_drop=minimum_shift_drop,
        minimum_adaptation_gain=minimum_adaptation_gain,
        minimum_auc_advantage=minimum_auc_advantage,
        accuracy_margin=accuracy_margin,
        forgetting_margin=forgetting_margin,
        minimum_spike_rate=minimum_spike_rate,
        maximum_spike_rate=maximum_spike_rate,
    )
    _save_progress(
        progress_path,
        signature,
        stage="complete",
        screen_records=screen_records,
        promoted_source_arms=promoted,
        adaptation_records=adaptation_records,
        decision=decision,
    )
    return Gen9ContinualAdaptationResult(
        config=full_config,
        device=device_kind(resolved),
        target_parameters=target_parameters,
        temporal_levels=levels,
        screen_seed=int(screen_seed),
        confirm_seeds=seeds,
        damage_fraction=damage_fraction,
        damage_seed=damage_seed,
        adaptation_budgets=budgets,
        screen_records=screen_records,
        promoted_source_arms=promoted,
        adaptation_records=adaptation_records,
        adaptation_summary=summary,
        decision=decision,
    )


def select_gen9_promoted_source_arms(
    records: Iterable[dict],
    *,
    promotion_margin: float,
    minimum_parameter_ratio: float,
    maximum_parameter_ratio: float,
    minimum_spike_rate: float,
    maximum_spike_rate: float,
) -> tuple[str, ...]:
    rows = list(records)
    baseline = next(row for row in rows if row["arm"] == "dilated_tcn")
    promoted = ["dilated_tcn"]
    lif = next(row for row in rows if row["arm"] == "predictive_lif")
    ratio = float(lif["parameter_ratio_vs_target"])
    if (
        float(lif["best_validation_accuracy"])
        >= float(baseline["best_validation_accuracy"]) - promotion_margin
        and minimum_parameter_ratio <= ratio <= maximum_parameter_ratio
        and minimum_spike_rate
        <= float(lif["checkpoint_activity"])
        <= maximum_spike_rate
    ):
        promoted.append("predictive_lif")
    return tuple(promoted)


def summarize_gen9_adaptation(
    records: Iterable[dict], *, budgets: Iterable[int]
) -> list[dict]:
    rows = list(records)
    budget_values = tuple(int(value) for value in budgets)
    if not rows:
        return []
    max_budget = max(budget_values)
    summary = []
    for strategy in GEN9_ADAPTATION_STRATEGIES:
        group = [row for row in rows if row["strategy"] == strategy]
        if not group:
            continue
        seeds = sorted({int(row["seed"]) for row in group})
        per_seed = []
        for seed in seeds:
            curve = sorted(
                (row for row in group if int(row["seed"]) == seed),
                key=lambda row: int(row["adaptation_samples"]),
            )
            initial = next(row for row in curve if int(row["adaptation_samples"]) == 0)
            final = next(
                row for row in curve if int(row["adaptation_samples"]) == max_budget
            )
            per_seed.append(
                {
                    "seed": seed,
                    "source_initial": float(initial["source_accuracy"]),
                    "shifted_initial": float(initial["shifted_accuracy"]),
                    "source_final": float(final["source_accuracy"]),
                    "shifted_final": float(final["shifted_accuracy"]),
                    "adaptation_gain": float(final["shifted_accuracy"])
                    - float(initial["shifted_accuracy"]),
                    "forgetting": float(initial["source_accuracy"])
                    - float(final["source_accuracy"]),
                    "auc": _curve_auc(curve, max_budget=max_budget),
                    "activity": float(final["activity"]),
                    "throughput": float(final["test_examples_per_second"]),
                    "train_seconds": float(final["cumulative_adaptation_seconds"]),
                }
            )
        summary.append(
            {
                "strategy": strategy,
                "source_model": group[0]["source_model"],
                "adaptation_kind": group[0]["adaptation_kind"],
                "runs": len(per_seed),
                "mean_source_initial_accuracy": statistics.fmean(item["source_initial"] for item in per_seed),
                "mean_shifted_initial_accuracy": statistics.fmean(item["shifted_initial"] for item in per_seed),
                "mean_shift_drop": statistics.fmean(item["source_initial"] - item["shifted_initial"] for item in per_seed),
                "mean_source_final_accuracy": statistics.fmean(item["source_final"] for item in per_seed),
                "mean_shifted_final_accuracy": statistics.fmean(item["shifted_final"] for item in per_seed),
                "mean_adaptation_gain": statistics.fmean(item["adaptation_gain"] for item in per_seed),
                "two_point_gain_seed_count": sum(item["adaptation_gain"] >= 0.02 for item in per_seed),
                "mean_forgetting": statistics.fmean(item["forgetting"] for item in per_seed),
                "mean_adaptation_auc": statistics.fmean(item["auc"] for item in per_seed),
                "mean_activity": statistics.fmean(item["activity"] for item in per_seed),
                "mean_test_examples_per_second": statistics.fmean(item["throughput"] for item in per_seed),
                "mean_cumulative_adaptation_seconds": statistics.fmean(item["train_seconds"] for item in per_seed),
                "adaptation_trainable_parameters": int(group[0]["adaptation_trainable_parameters"]),
            }
        )
    lookup = {row["strategy"]: row for row in summary}
    if "predictive_lif_readout" in lookup and "tcn_readout" in lookup:
        lif_by_seed = _per_seed_auc(rows, "predictive_lif_readout", max_budget)
        tcn_by_seed = _per_seed_auc(rows, "tcn_readout", max_budget)
        count = sum(
            lif_by_seed[seed] - tcn_by_seed[seed] >= 0.01
            for seed in set(lif_by_seed) & set(tcn_by_seed)
        )
        for row in summary:
            row["one_point_auc_advantage_seed_count_vs_tcn_readout"] = (
                count if row["strategy"] == "predictive_lif_readout" else 0
            )
    else:
        for row in summary:
            row["one_point_auc_advantage_seed_count_vs_tcn_readout"] = 0
    return sorted(
        summary,
        key=lambda row: (-float(row["mean_adaptation_auc"]), str(row["strategy"])),
    )


def decide_gen9_continual_adaptation(
    summary: Iterable[dict],
    *,
    minimum_shift_drop: float,
    minimum_adaptation_gain: float,
    minimum_auc_advantage: float,
    accuracy_margin: float,
    forgetting_margin: float,
    minimum_spike_rate: float,
    maximum_spike_rate: float,
) -> dict:
    rows = list(summary)
    lookup = {str(row["strategy"]): row for row in rows}
    required_names = (
        "tcn_static",
        "tcn_readout",
        "predictive_lif_static",
        "predictive_lif_readout",
    )
    if any(name not in lookup for name in required_names):
        return {
            "status": "stop",
            "qualified_arms": [],
            "reason": "required source or adaptation strategy did not reach confirmation",
            "next_milestone": "close_gen9_continual_adaptation",
        }
    tcn_static = lookup["tcn_static"]
    tcn_readout = lookup["tcn_readout"]
    lif_static = lookup["predictive_lif_static"]
    lif_readout = lookup["predictive_lif_readout"]
    required = 2 if int(lif_readout["runs"]) >= 3 else 1
    auc_advantage = float(lif_readout["mean_adaptation_auc"]) - float(
        tcn_readout["mean_adaptation_auc"]
    )
    final_accuracy_gap = float(lif_readout["mean_shifted_final_accuracy"]) - float(
        tcn_readout["mean_shifted_final_accuracy"]
    )
    source_accuracy_gap = float(lif_static["mean_source_initial_accuracy"]) - float(
        tcn_static["mean_source_initial_accuracy"]
    )
    forgetting_difference = float(lif_readout["mean_forgetting"]) - float(
        tcn_readout["mean_forgetting"]
    )
    qualified = []
    if (
        float(tcn_static["mean_shift_drop"]) >= minimum_shift_drop
        and source_accuracy_gap >= -accuracy_margin
        and float(lif_readout["mean_adaptation_gain"]) >= minimum_adaptation_gain
        and int(lif_readout["two_point_gain_seed_count"]) >= required
        and auc_advantage >= minimum_auc_advantage
        and int(lif_readout["one_point_auc_advantage_seed_count_vs_tcn_readout"])
        >= required
        and final_accuracy_gap >= -accuracy_margin
        and forgetting_difference <= forgetting_margin
        and minimum_spike_rate
        <= float(lif_readout["mean_activity"])
        <= maximum_spike_rate
    ):
        qualified.append("predictive_lif_readout")
    return {
        "status": "pass" if qualified else "stop",
        "task_valid": float(tcn_static["mean_shift_drop"]) >= minimum_shift_drop,
        "source_accuracy_gap_vs_tcn": source_accuracy_gap,
        "adaptation_auc_advantage_vs_tcn_readout": auc_advantage,
        "final_shifted_accuracy_gap_vs_tcn_readout": final_accuracy_gap,
        "forgetting_difference_vs_tcn_readout": forgetting_difference,
        "qualified_arms": qualified,
        "next_milestone": (
            "gen9_stw_ltw_memory_preregistration"
            if qualified
            else "close_gen9_continual_adaptation"
        ),
    }


def plot_gen9_continual_adaptation(
    records: Iterable[dict], summary: Iterable[dict], path: str | pathlib.Path
) -> None:
    import matplotlib.pyplot as plt

    rows = list(records)
    summaries = list(summary)
    figure, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    for strategy in GEN9_ADAPTATION_STRATEGIES:
        group = [row for row in rows if row["strategy"] == strategy]
        if not group:
            continue
        budgets = sorted({int(row["adaptation_samples"]) for row in group})
        means = [
            statistics.fmean(
                float(row["shifted_accuracy"])
                for row in group
                if int(row["adaptation_samples"]) == budget
            )
            for budget in budgets
        ]
        axes[0, 0].plot(budgets, [100.0 * value for value in means], marker="o", label=strategy)
        retention = [
            statistics.fmean(
                float(row["source_accuracy"])
                for row in group
                if int(row["adaptation_samples"]) == budget
            )
            for budget in budgets
        ]
        axes[0, 1].plot(budgets, [100.0 * value for value in retention], marker="o", label=strategy)
    axes[0, 0].set_title("Gen-9 sensor-damage adaptation")
    axes[0, 0].set_ylabel("Shifted SSC accuracy (%)")
    axes[0, 1].set_title("Source-task retention")
    axes[0, 1].set_ylabel("Undamaged SSC accuracy (%)")
    for axis in axes[0]:
        axis.set_xlabel("Cumulative adaptation samples")
        axis.set_xscale("symlog", linthresh=64)
        axis.legend(fontsize=8)
        axis.grid(alpha=0.25)
    labels = [row["strategy"].replace("_", "\n") for row in summaries]
    axes[1, 0].bar(labels, [100.0 * float(row["mean_adaptation_auc"]) for row in summaries], color="#35b4f2")
    axes[1, 0].set_ylabel("Adaptation AUC (%)")
    axes[1, 1].bar(labels, [100.0 * float(row["mean_forgetting"]) for row in summaries], color="#bd3d3a")
    axes[1, 1].set_ylabel("Source forgetting (points)")
    for axis in axes[1]:
        axis.grid(axis="y", alpha=0.25)
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _run_source_screen(
    arms,
    seed,
    config,
    train_events,
    train_labels,
    validation_events,
    validation_labels,
    test_events,
    test_labels,
    *,
    mask,
    target_parameters,
    levels,
    input_kernel_size,
    hidden_kernel_size,
    tcn_dilation,
    surrogate_slope,
    future_horizon,
    contrastive_temperature,
    device,
    existing_records=(),
    progress_callback=None,
):
    records = list(existing_records)
    completed = {(int(row["seed"]), str(row["arm"])) for row in records}
    for arm in arms:
        if (seed, arm.name) in completed:
            continue
        model, channels, activity_kind, training = _train_source(
            arm,
            seed,
            config,
            train_events,
            train_labels,
            validation_events,
            validation_labels,
            target_parameters=target_parameters,
            levels=levels,
            input_kernel_size=input_kernel_size,
            hidden_kernel_size=hidden_kernel_size,
            tcn_dilation=tcn_dilation,
            surrogate_slope=surrogate_slope,
            future_horizon=future_horizon,
            contrastive_temperature=contrastive_temperature,
            device=device,
        )
        source_accuracy, source_seconds, activity = _measure(
            model, test_events, test_labels, config.batch_size, device
        )
        shifted_accuracy, _, _ = _measure_shifted(
            model, test_events, test_labels, config.batch_size, device, mask
        )
        parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        gate = _measure_sample_gate(model, test_events, config.batch_size, device)
        records.append(
            {
                "seed": seed,
                "arm": arm.name,
                "model_kind": arm.model_kind,
                "channels": channels,
                "best_epoch": training["best_epoch"],
                "best_validation_accuracy": training["best_validation_accuracy"],
                "source_test_accuracy": source_accuracy,
                "shifted_test_accuracy": shifted_accuracy,
                "shift_drop": source_accuracy - shifted_accuracy,
                "checkpoint_activity": activity,
                "activity_kind": activity_kind,
                "mean_absolute_gate": gate,
                "effective_trainable_parameters": parameters,
                "parameter_ratio_vs_target": parameters / target_parameters,
                "test_examples_per_second": test_events.shape[0] / max(source_seconds, 1e-12),
                "train_seconds": training["train_seconds"],
            }
        )
        completed.add((seed, arm.name))
        if progress_callback is not None:
            progress_callback(records)
    return records


def _run_confirmation(
    promoted,
    seeds,
    config,
    train_events,
    train_labels,
    validation_events,
    validation_labels,
    test_events,
    test_labels,
    *,
    mask,
    budgets,
    adaptation_epochs_per_block,
    adaptation_learning_rate,
    target_parameters,
    levels,
    input_kernel_size,
    hidden_kernel_size,
    tcn_dilation,
    surrogate_slope,
    future_horizon,
    contrastive_temperature,
    device,
    existing_records=(),
    progress_callback=None,
):
    records = list(existing_records)
    completed = {
        (int(row["seed"]), str(row["strategy"]), int(row["adaptation_samples"]))
        for row in records
    }
    source_lookup = {arm.name: arm for arm in GEN9_SOURCE_ARMS}
    strategies_by_source = {
        "dilated_tcn": ("tcn_static", "tcn_readout", "tcn_full_finetune"),
        "predictive_lif": ("predictive_lif_static", "predictive_lif_readout"),
    }
    for seed in seeds:
        for source_name in promoted:
            arm = source_lookup[source_name]
            source_strategies = strategies_by_source[source_name]
            source_expected = {
                (int(seed), strategy, int(budget))
                for strategy in source_strategies
                for budget in budgets
            }
            if source_expected.issubset(completed):
                continue
            model, _, _, _ = _train_source(
                arm,
                seed,
                config,
                train_events,
                train_labels,
                validation_events,
                validation_labels,
                target_parameters=target_parameters,
                levels=levels,
                input_kernel_size=input_kernel_size,
                hidden_kernel_size=hidden_kernel_size,
                tcn_dilation=tcn_dilation,
                surrogate_slope=surrogate_slope,
                future_horizon=future_horizon,
                contrastive_temperature=contrastive_temperature,
                device=device,
            )
            for strategy in source_strategies:
                expected = {(int(seed), strategy, int(budget)) for budget in budgets}
                if expected.issubset(completed):
                    continue
                adapted = copy.deepcopy(model)
                new_records = _adaptation_curve(
                    adapted,
                    strategy,
                    source_name,
                    seed,
                    validation_events,
                    validation_labels,
                    test_events,
                    test_labels,
                    mask=mask,
                    budgets=budgets,
                    epochs_per_block=adaptation_epochs_per_block,
                    learning_rate=adaptation_learning_rate,
                    batch_size=config.batch_size,
                    weight_decay=config.weight_decay,
                    device=device,
                )
                records = [
                    row
                    for row in records
                    if not (int(row["seed"]) == seed and row["strategy"] == strategy)
                ]
                records.extend(new_records)
                completed.update(expected)
                if progress_callback is not None:
                    progress_callback(records)
            del model
            gc.collect()
    return records


def _train_source(
    arm,
    seed,
    config,
    train_events,
    train_labels,
    validation_events,
    validation_labels,
    *,
    target_parameters,
    levels,
    input_kernel_size,
    hidden_kernel_size,
    tcn_dilation,
    surrogate_slope,
    future_horizon,
    contrastive_temperature,
    device,
):
    seed_everything(seed, device=device)
    model, channels, activity_kind = _build_model(
        arm,
        config,
        target_parameters=target_parameters,
        levels=levels,
        input_kernel_size=input_kernel_size,
        hidden_kernel_size=hidden_kernel_size,
        tcn_dilation=tcn_dilation,
        surrogate_slope=surrogate_slope,
        future_horizon=future_horizon,
        contrastive_temperature=contrastive_temperature,
        device=device,
    )
    training = _train_predictive_validation_selected(
        model,
        arm,
        train_events,
        train_labels,
        validation_events,
        validation_labels,
        config,
        seed=seed,
        device=device,
    )
    model.load_state_dict(training["best_state"])
    return model, channels, activity_kind, training


def _adaptation_curve(
    model,
    strategy,
    source_model,
    seed,
    adaptation_events,
    adaptation_labels,
    test_events,
    test_labels,
    *,
    mask,
    budgets,
    epochs_per_block,
    learning_rate,
    batch_size,
    weight_decay,
    device,
):
    if strategy.endswith("_static"):
        adaptation_kind = "static"
    elif strategy.endswith("_readout"):
        adaptation_kind = "readout"
    elif strategy.endswith("_full_finetune"):
        adaptation_kind = "full_finetune"
    else:
        raise ValueError(f"unknown adaptation strategy: {strategy}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if adaptation_kind == "readout":
        for parameter in model.classifier.parameters():
            parameter.requires_grad_(True)
    elif adaptation_kind == "full_finetune":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    elif adaptation_kind != "static":
        raise ValueError(f"unknown adaptation kind: {adaptation_kind}")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = (
        torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
        if trainable
        else None
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 91_000)
    order = torch.randperm(adaptation_events.shape[0], generator=generator)
    records = []
    previous = 0
    cumulative_seconds = 0.0
    for budget in budgets:
        budget = int(budget)
        if budget > previous and optimizer is not None:
            indices = order[previous:budget]
            cumulative_seconds += _adapt_block(
                model,
                optimizer,
                adaptation_events,
                adaptation_labels,
                indices,
                mask=mask,
                epochs=epochs_per_block,
                batch_size=batch_size,
                device=device,
                generator=generator,
            )
        source_accuracy, _, _ = _measure(
            model, test_events, test_labels, batch_size, device
        )
        shifted_accuracy, shifted_seconds, activity = _measure_shifted(
            model, test_events, test_labels, batch_size, device, mask
        )
        records.append(
            {
                "seed": int(seed),
                "strategy": strategy,
                "source_model": source_model,
                "adaptation_kind": adaptation_kind,
                "adaptation_samples": budget,
                "source_accuracy": source_accuracy,
                "shifted_accuracy": shifted_accuracy,
                "activity": activity,
                "test_examples_per_second": test_events.shape[0] / max(shifted_seconds, 1e-12),
                "cumulative_adaptation_seconds": cumulative_seconds,
                "adaptation_trainable_parameters": sum(parameter.numel() for parameter in trainable),
            }
        )
        previous = budget
    return records


def _adapt_block(
    model,
    optimizer,
    events,
    labels,
    indices,
    *,
    mask,
    epochs,
    batch_size,
    device,
    generator,
):
    sync(device)
    start = time.perf_counter()
    for _ in range(epochs):
        model.train()
        order = indices.index_select(
            0, torch.randperm(indices.shape[0], generator=generator)
        )
        for offset in range(0, order.shape[0], batch_size):
            index = order[offset : offset + batch_size]
            batch_events = apply_sensor_damage(events.index_select(0, index), mask).to(device)
            batch_labels = labels.index_select(0, index).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(batch_events), batch_labels)
            loss.backward()
            optimizer.step()
            mark_step(device)
    sync(device)
    return time.perf_counter() - start


def _measure_shifted(model, events, labels, batch_size, device, mask):
    model.eval()
    correct = 0
    total = 0
    weighted_activity = 0.0
    sync(device)
    start = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, events.shape[0], batch_size):
            batch_events = apply_sensor_damage(events[offset : offset + batch_size], mask).to(device)
            batch_labels = labels[offset : offset + batch_size].to(device)
            logits, activity = model(batch_events, return_event_rate=True)
            correct += int((logits.argmax(dim=1) == batch_labels).sum().item())
            total += int(batch_labels.shape[0])
            weighted_activity += float(activity.item()) * int(batch_labels.shape[0])
            mark_step(device)
    sync(device)
    return (
        correct / max(total, 1),
        time.perf_counter() - start,
        weighted_activity / max(total, 1),
    )


def _curve_auc(curve, *, max_budget):
    ordered = sorted(curve, key=lambda row: int(row["adaptation_samples"]))
    if max_budget <= 0:
        return float(ordered[-1]["shifted_accuracy"])
    area = 0.0
    for left, right in zip(ordered, ordered[1:]):
        width = (
            int(right["adaptation_samples"]) - int(left["adaptation_samples"])
        ) / max_budget
        area += width * 0.5 * (
            float(left["shifted_accuracy"]) + float(right["shifted_accuracy"])
        )
    return area


def _per_seed_auc(rows, strategy, max_budget):
    group = [row for row in rows if row["strategy"] == strategy]
    return {
        seed: _curve_auc(
            [row for row in group if int(row["seed"]) == seed],
            max_budget=max_budget,
        )
        for seed in {int(row["seed"]) for row in group}
    }


def _completed_result(
    progress,
    config,
    device,
    target_parameters,
    levels,
    screen_seed,
    seeds,
    damage_fraction,
    damage_seed,
    budgets,
    minimum_shift_drop,
    minimum_adaptation_gain,
    minimum_auc_advantage,
    accuracy_margin,
    forgetting_margin,
    minimum_spike_rate,
    maximum_spike_rate,
):
    records = list(progress.get("adaptation_records", []))
    summary = summarize_gen9_adaptation(records, budgets=budgets)
    decision = progress.get("decision") or decide_gen9_continual_adaptation(
        summary,
        minimum_shift_drop=minimum_shift_drop,
        minimum_adaptation_gain=minimum_adaptation_gain,
        minimum_auc_advantage=minimum_auc_advantage,
        accuracy_margin=accuracy_margin,
        forgetting_margin=forgetting_margin,
        minimum_spike_rate=minimum_spike_rate,
        maximum_spike_rate=maximum_spike_rate,
    )
    return Gen9ContinualAdaptationResult(
        config=config,
        device=device_kind(device),
        target_parameters=target_parameters,
        temporal_levels=levels,
        screen_seed=screen_seed,
        confirm_seeds=seeds,
        damage_fraction=damage_fraction,
        damage_seed=damage_seed,
        adaptation_budgets=budgets,
        screen_records=list(progress.get("screen_records", [])),
        promoted_source_arms=tuple(progress.get("promoted_source_arms", [])),
        adaptation_records=records,
        adaptation_summary=summary,
        decision=decision,
    )


def _validate_run(
    config,
    levels,
    seeds,
    budgets,
    screen_epochs,
    confirm_epochs,
    adaptation_epochs,
    adaptation_lr,
    samples,
    damage_fraction,
    parameter_gate,
    spike_gate,
    gates,
):
    if not levels or any(value <= 0 for value in levels):
        raise ValueError("temporal levels must be positive")
    if not seeds or screen_epochs <= 0 or confirm_epochs <= 0 or adaptation_epochs <= 0:
        raise ValueError("seeds and epoch counts must be positive")
    if not budgets or budgets[0] != 0 or tuple(sorted(set(budgets))) != budgets:
        raise ValueError("adaptation budgets must be unique, increasing, and begin at zero")
    if budgets[-1] <= 0 or adaptation_lr <= 0.0:
        raise ValueError("adaptation budget and learning rate must be positive")
    if min(samples) < 0 or not 0.0 < damage_fraction < 1.0:
        raise ValueError("invalid samples or damage fraction")
    if not 0.0 < parameter_gate[0] <= parameter_gate[1]:
        raise ValueError("invalid parameter gate")
    if not 0.0 <= spike_gate[0] <= spike_gate[1] <= 1.0:
        raise ValueError("invalid spike gate")
    if any(not 0.0 <= value <= 1.0 for value in gates):
        raise ValueError("invalid terminal gate")
    if budgets[-1] > 9_981:
        raise ValueError("adaptation budget exceeds the official SSC validation split")


def _run_signature(config, **values):
    signature = {
        "version": 1,
        "source_arms": list(available_gen9_source_arms()),
        "strategies": list(available_gen9_adaptation_strategies()),
        "input_neurons": int(config.input_neurons),
        "classes": int(config.classes),
        "timesteps": int(config.timesteps),
        "duration_seconds": float(config.duration_seconds),
        "source_learning_rate": float(config.learning_rate),
        "weight_decay": float(config.weight_decay),
        "batch_size": int(config.batch_size),
        "data_root": str(config.data_root),
        "data_seed": int(config.data_seed),
    }
    for key, value in values.items():
        signature[key] = list(value) if isinstance(value, tuple) else value
    return signature


def _save_progress(
    path,
    signature,
    *,
    stage,
    screen_records,
    promoted_source_arms,
    adaptation_records,
    decision=None,
):
    """Atomically persist the Gen-9-specific checkpoint schema."""

    if path is None:
        return
    progress_path = pathlib.Path(path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "signature": signature,
        "stage": stage,
        "screen_records": list(screen_records),
        "promoted_source_arms": list(promoted_source_arms),
        "adaptation_records": list(adaptation_records),
        "decision": decision,
    }
    temporary = progress_path.with_suffix(progress_path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(progress_path)


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
