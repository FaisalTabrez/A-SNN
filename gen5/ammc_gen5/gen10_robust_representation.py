"""Gen-10 masked-sensor robust representation reset."""

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
from .gen6_successor import (
    SharedResidualStateTCNClassifier,
    matched_shared_residual_channels,
)
from .gen9_continual_adaptation import (
    _measure_shifted,
    apply_sensor_damage,
    sensor_damage_indices,
)
from .milestone_a_architecture import _load_progress, _sample_split, _save_progress
from .runtime import device_kind, mark_step, resolve_device, seed_everything, sync
from .shd_benchmark import SHDConfig, _measure
from .shd_residual_state_contribution import RESIDUAL_ABLATION_MODES
from .shd_temporal_pyramid import DEFAULT_TEMPORAL_LEVELS
from .ssc_benchmark import load_ssc_tensors
from .ssc_efficiency_baselines import (
    TemporalDilatedTCNClassifier,
    matched_temporal_tcn_channels,
)


@dataclass(frozen=True)
class Gen10RepresentationArm:
    name: str
    model_kind: str
    conventional: bool
    causal_state: bool
    sensor_dropout: bool
    dynamics: str | None = None


GEN10_REPRESENTATION_ARMS = (
    Gen10RepresentationArm("dilated_tcn", "tcn", True, False, False),
    Gen10RepresentationArm("dropout_tcn", "tcn", True, False, True),
    Gen10RepresentationArm(
        "masked_residual_analog", "shared_residual", False, True, True, "analog"
    ),
    Gen10RepresentationArm(
        "masked_residual_lif", "shared_residual", False, True, True, "lif"
    ),
)


def available_gen10_representation_arms() -> tuple[str, ...]:
    return tuple(arm.name for arm in GEN10_REPRESENTATION_ARMS)


class MaskedSensorResidualClassifier(SharedResidualStateTCNClassifier):
    """Residual state model with a parameter-free clean-target alignment loss."""

    def representation_alignment_loss(self, clean_events, masked_events):
        clean_currents, _ = self._traces(clean_events)
        _, masked_state = self._traces(masked_events)
        target = torch.nn.functional.normalize(
            torch.relu(clean_currents).mean(dim=1).detach(), dim=1
        )
        prediction = torch.nn.functional.normalize(masked_state.mean(dim=1), dim=1)
        return 1.0 - (prediction * target).sum(dim=1).mean()

    def _traces(self, events):
        first = torch.relu(
            self.input_conv(events.to(torch.float32).transpose(1, 2))
        )
        currents = (self.hidden_conv(first) + first).transpose(1, 2)
        state, _ = self._state_trace(currents)
        return currents, state


@dataclass
class Gen10RobustRepresentationResult:
    config: SHDConfig
    device: str
    target_parameters: int
    temporal_levels: tuple[int, ...]
    screen_seed: int
    confirm_seeds: tuple[int, ...]
    training_mask_fraction: float
    damage_fraction: float
    damage_seed: int
    screen_records: list[dict]
    promoted_arms: tuple[str, ...]
    confirmation_records: list[dict]
    confirmation_summary: list[dict]
    decision: dict

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "gen10_robust_representation.json"
        screen_path = output / "gen10_robust_representation_screen.csv"
        records_path = output / "gen10_robust_representation_records.csv"
        summary_path = output / "gen10_robust_representation_summary.csv"
        payload = {
            "config": asdict(self.config),
            "device": self.device,
            "target_parameters": self.target_parameters,
            "temporal_levels": list(self.temporal_levels),
            "screen_seed": self.screen_seed,
            "confirm_seeds": list(self.confirm_seeds),
            "training_mask_fraction": self.training_mask_fraction,
            "damage_fraction": self.damage_fraction,
            "damage_seed": self.damage_seed,
            "screen_records": self.screen_records,
            "promoted_arms": list(self.promoted_arms),
            "confirmation_records": self.confirmation_records,
            "confirmation_summary": self.confirmation_summary,
            "decision": self.decision,
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        _write_csv(screen_path, self.screen_records)
        _write_csv(records_path, self.confirmation_records)
        _write_csv(summary_path, self.confirmation_summary)
        paths = {
            "json": str(json_path),
            "screen_csv": str(screen_path),
            "records_csv": str(records_path),
            "summary_csv": str(summary_path),
        }
        if plot and self.confirmation_summary:
            plot_path = output / "gen10_robust_representation.png"
            plot_gen10_robust_representation(self.confirmation_summary, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def run_gen10_robust_representation(
    config: SHDConfig,
    *,
    screen_seed: int = 151,
    confirm_seeds: Iterable[int] = (151, 152, 153),
    screen_train_samples: int = 15_000,
    screen_validation_samples: int = 3_000,
    screen_test_samples: int = 3_000,
    screen_epochs: int = 5,
    confirm_epochs: int = 15,
    training_mask_fraction: float = 0.20,
    damage_fraction: float = 0.35,
    damage_seed: int = 909,
    alignment_weight: float = 0.10,
    promotion_margin: float = 0.01,
    damaged_promotion_margin: float = 0.02,
    minimum_parameter_ratio: float = 0.95,
    maximum_parameter_ratio: float = 1.05,
    minimum_spike_rate: float = 0.01,
    maximum_spike_rate: float = 0.30,
    accuracy_margin: float = 0.01,
    causal_margin: float = 0.005,
    robustness_margin: float = 0.005,
    target_parameters: int = 133_631,
    device="auto",
    temporal_levels: Iterable[int] = DEFAULT_TEMPORAL_LEVELS,
    input_kernel_size: int = 5,
    hidden_kernel_size: int = 3,
    tcn_dilation: int = 2,
    surrogate_slope: float = 10.0,
    progress_path: str | pathlib.Path | None = None,
) -> Gen10RobustRepresentationResult:
    if torch is None:
        raise ImportError("Gen-10 robust representation requires PyTorch")
    levels = tuple(int(level) for level in temporal_levels)
    seeds = tuple(int(seed) for seed in confirm_seeds)
    _validate_run(
        config,
        levels,
        seeds,
        screen_epochs,
        confirm_epochs,
        (training_mask_fraction, damage_fraction),
        alignment_weight,
        (minimum_parameter_ratio, maximum_parameter_ratio),
        (minimum_spike_rate, maximum_spike_rate),
        (promotion_margin, damaged_promotion_margin, accuracy_margin, causal_margin, robustness_margin),
    )
    signature = _run_signature(
        config,
        screen_seed=screen_seed,
        confirm_seeds=seeds,
        screen_samples=(screen_train_samples, screen_validation_samples, screen_test_samples),
        epochs=(screen_epochs, confirm_epochs),
        training_mask_fraction=training_mask_fraction,
        damage_fraction=damage_fraction,
        damage_seed=damage_seed,
        alignment_weight=alignment_weight,
        promotion_margin=promotion_margin,
        damaged_promotion_margin=damaged_promotion_margin,
        parameter_gate=(minimum_parameter_ratio, maximum_parameter_ratio),
        spike_gate=(minimum_spike_rate, maximum_spike_rate),
        terminal_gates=(accuracy_margin, causal_margin, robustness_margin),
        target_parameters=target_parameters,
        levels=levels,
        kernels=(input_kernel_size, hidden_kernel_size),
        tcn_dilation=tcn_dilation,
        surrogate_slope=surrogate_slope,
    )
    progress = _load_progress(progress_path, signature)
    resolved = resolve_device(device)
    full_config = replace(config, train_samples=0, test_samples=0, epochs=confirm_epochs)
    if progress.get("stage") == "complete":
        return _completed_result(
            progress, full_config, resolved, target_parameters, levels, screen_seed,
            seeds, training_mask_fraction, damage_fraction, damage_seed,
            accuracy_margin, causal_margin, robustness_margin,
            minimum_spike_rate, maximum_spike_rate,
        )
    train_events, train_labels, validation_events, validation_labels, test_events, test_labels = load_ssc_tensors(
        full_config, validation_samples=0
    )
    fixed_mask = sensor_damage_indices(config.input_neurons, damage_fraction, seed=damage_seed)
    screen_records = list(progress.get("screen_records", []))
    confirmation_records = list(progress.get("confirmation_records", []))
    expected = {(int(screen_seed), arm.name) for arm in GEN10_REPRESENTATION_ARMS}
    completed = {(int(row["seed"]), str(row["arm"])) for row in screen_records}
    if not expected.issubset(completed):
        generator = torch.Generator(device="cpu").manual_seed(config.data_seed + 100_000)
        sample_sets = (
            _sample_split(train_events, train_labels, screen_train_samples, generator),
            _sample_split(validation_events, validation_labels, screen_validation_samples, generator),
            _sample_split(test_events, test_labels, screen_test_samples, generator),
        )
        screen_records = _run_stage(
            GEN10_REPRESENTATION_ARMS, (int(screen_seed),),
            replace(full_config, epochs=screen_epochs),
            *sample_sets[0], *sample_sets[1], *sample_sets[2],
            fixed_mask=fixed_mask, training_mask_fraction=training_mask_fraction,
            alignment_weight=alignment_weight, target_parameters=target_parameters,
            levels=levels, input_kernel_size=input_kernel_size,
            hidden_kernel_size=hidden_kernel_size, tcn_dilation=tcn_dilation,
            surrogate_slope=surrogate_slope, device=resolved, ablate=False,
            existing_records=screen_records,
            progress_callback=lambda rows: _save_progress(
                progress_path, signature, stage="screen", screen_records=rows,
                promoted_arms=(), confirmation_records=confirmation_records,
            ),
        )
        del sample_sets
        gc.collect()
    promoted = select_gen10_promoted_arms(
        screen_records, promotion_margin=promotion_margin,
        damaged_promotion_margin=damaged_promotion_margin,
        minimum_parameter_ratio=minimum_parameter_ratio,
        maximum_parameter_ratio=maximum_parameter_ratio,
        minimum_spike_rate=minimum_spike_rate,
        maximum_spike_rate=maximum_spike_rate,
    )
    _save_progress(
        progress_path, signature, stage="confirmation", screen_records=screen_records,
        promoted_arms=promoted, confirmation_records=confirmation_records,
    )
    lookup = {arm.name: arm for arm in GEN10_REPRESENTATION_ARMS}
    confirmation_records = _run_stage(
        tuple(lookup[name] for name in promoted), seeds, full_config,
        train_events, train_labels, validation_events, validation_labels,
        test_events, test_labels, fixed_mask=fixed_mask,
        training_mask_fraction=training_mask_fraction, alignment_weight=alignment_weight,
        target_parameters=target_parameters, levels=levels,
        input_kernel_size=input_kernel_size, hidden_kernel_size=hidden_kernel_size,
        tcn_dilation=tcn_dilation, surrogate_slope=surrogate_slope,
        device=resolved, ablate=True, existing_records=confirmation_records,
        progress_callback=lambda rows: _save_progress(
            progress_path, signature, stage="confirmation", screen_records=screen_records,
            promoted_arms=promoted, confirmation_records=rows,
        ),
    )
    summary = summarize_gen10_confirmation(confirmation_records)
    decision = decide_gen10_robust_representation(
        summary, accuracy_margin=accuracy_margin, causal_margin=causal_margin,
        robustness_margin=robustness_margin, minimum_spike_rate=minimum_spike_rate,
        maximum_spike_rate=maximum_spike_rate,
    )
    _save_progress(
        progress_path, signature, stage="complete", screen_records=screen_records,
        promoted_arms=promoted, confirmation_records=confirmation_records,
        decision=decision,
    )
    return Gen10RobustRepresentationResult(
        config=full_config, device=device_kind(resolved), target_parameters=target_parameters,
        temporal_levels=levels, screen_seed=int(screen_seed), confirm_seeds=seeds,
        training_mask_fraction=training_mask_fraction, damage_fraction=damage_fraction,
        damage_seed=damage_seed, screen_records=screen_records, promoted_arms=promoted,
        confirmation_records=confirmation_records, confirmation_summary=summary,
        decision=decision,
    )


def select_gen10_promoted_arms(
    records: Iterable[dict], *, promotion_margin: float,
    damaged_promotion_margin: float, minimum_parameter_ratio: float,
    maximum_parameter_ratio: float, minimum_spike_rate: float,
    maximum_spike_rate: float,
) -> tuple[str, ...]:
    rows = list(records)
    conventional = [row for row in rows if bool(row["conventional"])]
    best_clean = max(float(row["best_validation_accuracy"]) for row in conventional)
    best_damaged = max(float(row["damaged_validation_accuracy"]) for row in conventional)
    promoted = [str(row["arm"]) for row in conventional]
    for arm in GEN10_REPRESENTATION_ARMS:
        if arm.conventional:
            continue
        row = next(row for row in rows if row["arm"] == arm.name)
        ratio = float(row["parameter_ratio_vs_target"])
        activity_ok = True
        if arm.dynamics == "lif":
            activity_ok = minimum_spike_rate <= float(row["checkpoint_activity"]) <= maximum_spike_rate
        if (
            float(row["best_validation_accuracy"]) >= best_clean - promotion_margin
            and float(row["damaged_validation_accuracy"]) >= best_damaged - damaged_promotion_margin
            and minimum_parameter_ratio <= ratio <= maximum_parameter_ratio
            and activity_ok
        ):
            promoted.append(arm.name)
    return tuple(promoted)


def summarize_gen10_confirmation(records: Iterable[dict]) -> list[dict]:
    rows = list(records)
    if not rows:
        return []
    conventional = [row for row in rows if bool(row["conventional"])]
    best_clean = max(statistics.fmean(float(row["clean_accuracy"]) for row in conventional if row["arm"] == arm) for arm in {row["arm"] for row in conventional})
    best_damaged = max(statistics.fmean(float(row["damaged_accuracy"]) for row in conventional if row["arm"] == arm) for arm in {row["arm"] for row in conventional})
    best_damage_drop = min(statistics.fmean(float(row["clean_accuracy"]) - float(row["damaged_accuracy"]) for row in conventional if row["arm"] == arm) for arm in {row["arm"] for row in conventional})
    summary = []
    for arm in available_gen10_representation_arms():
        group = [row for row in rows if row["arm"] == arm]
        if not group:
            continue
        clean = statistics.fmean(float(row["clean_accuracy"]) for row in group)
        damaged = statistics.fmean(float(row["damaged_accuracy"]) for row in group)
        contributions = [float(row["damaged_state_contribution"]) for row in group if row["damaged_state_contribution"] is not None]
        specificities = [float(row["damaged_state_specificity"]) for row in group if row["damaged_state_specificity"] is not None]
        summary.append({
            "arm": arm,
            "model_kind": group[0]["model_kind"],
            "conventional": bool(group[0]["conventional"]),
            "causal_state": bool(group[0]["causal_state"]),
            "runs": len(group),
            "mean_clean_accuracy": clean,
            "std_clean_accuracy": statistics.pstdev(float(row["clean_accuracy"]) for row in group),
            "mean_damaged_accuracy": damaged,
            "std_damaged_accuracy": statistics.pstdev(float(row["damaged_accuracy"]) for row in group),
            "mean_damage_drop": clean - damaged,
            "mean_damage_drop_gap_vs_best_conventional": clean - damaged - best_damage_drop,
            "mean_clean_gap_vs_best_conventional": clean - best_clean,
            "mean_damaged_gap_vs_best_conventional": damaged - best_damaged,
            "mean_damaged_state_contribution": statistics.fmean(contributions) if contributions else 0.0,
            "causal_seed_count": sum(value >= 0.005 for value in contributions),
            "mean_damaged_state_specificity": statistics.fmean(specificities) if specificities else 0.0,
            "specificity_seed_count": sum(value >= 0.005 for value in specificities),
            "mean_activity": statistics.fmean(float(row["checkpoint_activity"]) for row in group),
            "activity_kind": group[0]["activity_kind"],
            "mean_absolute_gate": statistics.fmean(float(row["mean_absolute_gate"]) for row in group),
            "effective_trainable_parameters": int(group[0]["effective_trainable_parameters"]),
            "mean_test_examples_per_second": statistics.fmean(float(row["test_examples_per_second"]) for row in group),
            "mean_train_seconds": statistics.fmean(float(row["train_seconds"]) for row in group),
        })
    return sorted(summary, key=lambda row: (-float(row["mean_damaged_accuracy"]), str(row["arm"])))


def decide_gen10_robust_representation(
    summary: Iterable[dict], *, accuracy_margin: float, causal_margin: float,
    robustness_margin: float, minimum_spike_rate: float, maximum_spike_rate: float,
) -> dict:
    rows = list(summary)
    candidate = next((row for row in rows if row["arm"] == "masked_residual_lif"), None)
    if candidate is None:
        return {"status": "stop", "qualified_arms": [], "reason": "masked residual LIF did not reach confirmation", "next_milestone": "close_gen10_robust_representation"}
    required = 2 if int(candidate["runs"]) >= 3 else 1
    passed = (
        float(candidate["mean_clean_gap_vs_best_conventional"]) >= -accuracy_margin
        and float(candidate["mean_damaged_gap_vs_best_conventional"]) >= -accuracy_margin
        and float(candidate["mean_damaged_state_contribution"]) >= causal_margin
        and int(candidate["causal_seed_count"]) >= required
        and float(candidate["mean_damaged_state_specificity"]) >= causal_margin
        and int(candidate["specificity_seed_count"]) >= required
        and float(candidate["mean_damage_drop_gap_vs_best_conventional"]) <= robustness_margin
        and minimum_spike_rate <= float(candidate["mean_activity"]) <= maximum_spike_rate
    )
    return {
        "status": "pass" if passed else "stop",
        "qualified_arms": ["masked_residual_lif"] if passed else [],
        "best_arm": str(rows[0]["arm"]) if rows else None,
        "next_milestone": "gen11_continual_adaptation" if passed else "close_gen10_robust_representation",
    }


def plot_gen10_robust_representation(summary: list[dict], path: str | pathlib.Path) -> None:
    import matplotlib.pyplot as plt
    labels = [row["arm"].replace("_", "\n") for row in summary]
    figure, axes = plt.subplots(2, 1, figsize=(13, 11), constrained_layout=True)
    x = range(len(summary))
    axes[0].bar([value - 0.18 for value in x], [100.0 * row["mean_clean_accuracy"] for row in summary], width=0.36, label="clean")
    axes[0].bar([value + 0.18 for value in x], [100.0 * row["mean_damaged_accuracy"] for row in summary], width=0.36, label="damaged")
    axes[0].set_xticks(list(x), labels)
    axes[0].set_ylabel("SSC accuracy (%)")
    axes[0].set_title("Gen-10 masked-sensor representation reset")
    axes[0].legend()
    axes[1].bar(labels, [100.0 * row["mean_damaged_state_specificity"] for row in summary], color="#167d55")
    axes[1].axhline(0.5, color="#bd3d3a", linestyle="--", label="+0.5 point gate")
    axes[1].set_ylabel("Damaged full - shuffled state (points)")
    axes[1].legend()
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _run_stage(
    arms, seeds, config, train_events, train_labels, validation_events,
    validation_labels, test_events, test_labels, *, fixed_mask,
    training_mask_fraction, alignment_weight, target_parameters, levels,
    input_kernel_size, hidden_kernel_size, tcn_dilation, surrogate_slope,
    device, ablate, existing_records=(), progress_callback=None,
):
    records = list(existing_records)
    completed = {(int(row["seed"]), str(row["arm"])) for row in records}
    for seed in seeds:
        for arm in arms:
            if (int(seed), arm.name) in completed:
                continue
            seed_everything(seed, device=device)
            model, channels, activity_kind = _build_model(
                arm, config, target_parameters=target_parameters, levels=levels,
                input_kernel_size=input_kernel_size, hidden_kernel_size=hidden_kernel_size,
                tcn_dilation=tcn_dilation, surrogate_slope=surrogate_slope, device=device,
            )
            training = _train_validation_selected(
                model, arm, train_events, train_labels, validation_events,
                validation_labels, config, seed=seed,
                training_mask_fraction=training_mask_fraction,
                alignment_weight=alignment_weight, device=device,
            )
            model.load_state_dict(training["best_state"])
            model.set_ablation_mode("full") if hasattr(model, "set_ablation_mode") else None
            clean_accuracy, _, _ = _measure(model, test_events, test_labels, config.batch_size, device)
            damaged_validation_accuracy, _, _ = _measure_shifted(model, validation_events, validation_labels, config.batch_size, device, fixed_mask)
            modes = RESIDUAL_ABLATION_MODES if ablate and arm.causal_state else ("full",)
            damaged = {}
            for mode in modes:
                model.set_ablation_mode(mode) if hasattr(model, "set_ablation_mode") else None
                damaged[mode] = _measure_shifted(model, test_events, test_labels, config.batch_size, device, fixed_mask)
            full_accuracy, full_seconds, full_activity = damaged["full"]
            direct = damaged.get("direct_only", (None, None, None))[0]
            shuffled = damaged.get("shuffled_state", (None, None, None))[0]
            parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            gate = model.mean_absolute_gate() if hasattr(model, "mean_absolute_gate") else 0.0
            records.append({
                "seed": int(seed), "arm": arm.name, "model_kind": arm.model_kind,
                "conventional": arm.conventional, "causal_state": arm.causal_state,
                "sensor_dropout": arm.sensor_dropout, "channels": int(channels),
                "best_epoch": int(training["best_epoch"]),
                "best_validation_accuracy": float(training["best_validation_accuracy"]),
                "damaged_validation_accuracy": float(damaged_validation_accuracy),
                "clean_accuracy": float(clean_accuracy), "damaged_accuracy": float(full_accuracy),
                "damage_drop": float(clean_accuracy - full_accuracy),
                "damaged_direct_only_accuracy": direct,
                "damaged_shuffled_state_accuracy": shuffled,
                "damaged_state_contribution": float(full_accuracy - direct) if direct is not None else None,
                "damaged_state_specificity": float(full_accuracy - shuffled) if shuffled is not None else None,
                "checkpoint_activity": float(full_activity), "activity_kind": activity_kind,
                "mean_absolute_gate": float(gate), "effective_trainable_parameters": int(parameters),
                "parameter_ratio_vs_target": float(parameters / target_parameters),
                "test_examples_per_second": float(test_events.shape[0] / max(float(full_seconds), 1e-12)),
                "train_seconds": float(training["train_seconds"]),
            })
            completed.add((int(seed), arm.name))
            if progress_callback is not None:
                progress_callback(records)
            del model
            gc.collect()
    return records


def _build_model(arm, config, *, target_parameters, levels, input_kernel_size,
                 hidden_kernel_size, tcn_dilation, surrogate_slope, device):
    if arm.model_kind == "tcn":
        channels, _ = matched_temporal_tcn_channels(
            config.input_neurons, config.classes, target_parameters,
            input_kernel_size=input_kernel_size, hidden_kernel_size=hidden_kernel_size,
            temporal_levels=levels,
        )
        model = TemporalDilatedTCNClassifier(
            config, channels=channels, input_kernel_size=input_kernel_size,
            hidden_kernel_size=hidden_kernel_size, dilation=tcn_dilation,
            temporal_levels=levels,
        )
        activity_kind = "relu_activation"
    else:
        channels, _ = matched_shared_residual_channels(
            config.input_neurons, config.classes, target_parameters,
            input_kernel_size=input_kernel_size, hidden_kernel_size=hidden_kernel_size,
            temporal_levels=levels,
        )
        model = MaskedSensorResidualClassifier(
            config, channels=channels, input_kernel_size=input_kernel_size,
            hidden_kernel_size=hidden_kernel_size, dilation=tcn_dilation,
            temporal_levels=levels, dynamics=arm.dynamics, surrogate_slope=surrogate_slope,
        )
        activity_kind = "spike_rate" if arm.dynamics == "lif" else "analog_activation"
    return model.to(device), channels, activity_kind


def _train_validation_selected(
    model, arm, train_events, train_labels, validation_events, validation_labels,
    config, *, seed, training_mask_fraction, alignment_weight, device,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(seed + 101_000)
    best_accuracy = float("-inf")
    best_state = None
    best_epoch = 0
    sync(device)
    start = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        order = torch.randperm(train_events.shape[0], generator=generator)
        for offset in range(0, order.shape[0], config.batch_size):
            index = order[offset: offset + config.batch_size]
            clean = train_events.index_select(0, index).to(device)
            labels = train_labels.index_select(0, index).to(device)
            masked = _random_sensor_dropout(clean, training_mask_fraction, generator) if arm.sensor_dropout else clean
            optimizer.zero_grad(set_to_none=True)
            logits = model(masked)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            if arm.causal_state:
                loss = loss + alignment_weight * model.representation_alignment_loss(clean, masked)
            loss.backward()
            optimizer.step()
            mark_step(device)
        validation_accuracy, _, _ = _measure(model, validation_events, validation_labels, config.batch_size, device)
        if validation_accuracy > best_accuracy:
            best_accuracy = float(validation_accuracy)
            best_epoch = epoch + 1
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    sync(device)
    return {"best_state": best_state, "best_epoch": best_epoch, "best_validation_accuracy": best_accuracy, "train_seconds": time.perf_counter() - start}


def _random_sensor_dropout(events, fraction, generator):
    keep = (torch.rand((events.shape[0], events.shape[2]), generator=generator) >= fraction).to(events.device)
    return events * keep[:, None, :]


def _completed_result(progress, config, device, target_parameters, levels,
                      screen_seed, seeds, training_mask_fraction, damage_fraction,
                      damage_seed, accuracy_margin, causal_margin, robustness_margin,
                      minimum_spike_rate, maximum_spike_rate):
    records = list(progress.get("confirmation_records", []))
    summary = summarize_gen10_confirmation(records)
    decision = progress.get("decision") or decide_gen10_robust_representation(
        summary, accuracy_margin=accuracy_margin, causal_margin=causal_margin,
        robustness_margin=robustness_margin, minimum_spike_rate=minimum_spike_rate,
        maximum_spike_rate=maximum_spike_rate,
    )
    return Gen10RobustRepresentationResult(
        config=config, device=device_kind(device), target_parameters=target_parameters,
        temporal_levels=levels, screen_seed=screen_seed, confirm_seeds=seeds,
        training_mask_fraction=training_mask_fraction, damage_fraction=damage_fraction,
        damage_seed=damage_seed, screen_records=list(progress.get("screen_records", [])),
        promoted_arms=tuple(progress.get("promoted_arms", [])),
        confirmation_records=records, confirmation_summary=summary, decision=decision,
    )


def _validate_run(config, levels, seeds, screen_epochs, confirm_epochs,
                  fractions, alignment_weight, parameter_gate, spike_gate, gates):
    if not levels or any(level <= 0 for level in levels) or not seeds:
        raise ValueError("temporal levels and seeds must be non-empty and positive")
    if screen_epochs <= 0 or confirm_epochs <= 0:
        raise ValueError("epoch counts must be positive")
    if any(not 0.0 < value < 1.0 for value in fractions):
        raise ValueError("mask fractions must be between zero and one")
    if alignment_weight <= 0.0:
        raise ValueError("alignment weight must be positive")
    if not 0.0 < parameter_gate[0] <= parameter_gate[1]:
        raise ValueError("invalid parameter gate")
    if not 0.0 <= spike_gate[0] <= spike_gate[1] <= 1.0:
        raise ValueError("invalid spike gate")
    if any(not 0.0 <= value <= 1.0 for value in gates):
        raise ValueError("invalid terminal gate")
    if config.input_neurons <= 0 or config.classes <= 1:
        raise ValueError("invalid task configuration")


def _run_signature(config, **values):
    signature = {
        "version": 1, "experiment": "gen10_robust_representation",
        "arms": list(available_gen10_representation_arms()),
        "input_neurons": int(config.input_neurons), "classes": int(config.classes),
        "timesteps": int(config.timesteps), "learning_rate": float(config.learning_rate),
        "weight_decay": float(config.weight_decay), "batch_size": int(config.batch_size),
        "data_root": str(config.data_root), "data_seed": int(config.data_seed),
    }
    for key, value in values.items():
        signature[key] = list(value) if isinstance(value, tuple) else value
    return signature


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
