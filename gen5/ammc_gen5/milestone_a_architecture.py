"""Unified Milestone A screening and confirmation for SSC architectures."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
import gc
import json
import math
import pathlib
import statistics
from typing import Iterable

from .event_mnist import nn, torch
from .runtime import device_kind, resolve_device, seed_everything
from .shd_benchmark import SHDConfig, _measure
from .shd_calibrated_baselines import TemporalConvClassifier, matched_temporal_conv_channels
from .shd_residual_state_contribution import RESIDUAL_ABLATION_MODES
from .shd_state_placement_diagnostic import (
    ResidualTemporalConvStateClassifier,
    matched_temporal_conv_residual_channels,
)
from .shd_temporal_pyramid import DEFAULT_TEMPORAL_LEVELS
from .shd_validation_checkpoint import _train_validation_selected
from .ssc_benchmark import load_ssc_tensors
from .ssc_efficiency_baselines import (
    TemporalDilatedTCNClassifier,
    matched_temporal_tcn_channels,
)
from .trainable_temporal_mnist import SurrogateSpike


@dataclass(frozen=True)
class MilestoneAArm:
    name: str
    model_kind: str
    conventional: bool
    causal_state: bool


MILESTONE_A_ARMS = (
    MilestoneAArm("temporal_conv1d", "conv1d", True, False),
    MilestoneAArm("dilated_tcn", "tcn", True, False),
    MilestoneAArm("residual_lif", "residual_lif", False, True),
    MilestoneAArm("hierarchical_residual_analog", "hierarchical_analog", False, True),
    MilestoneAArm("hierarchical_residual_lif", "hierarchical_lif", False, True),
)


def available_milestone_a_arms() -> tuple[str, ...]:
    return tuple(arm.name for arm in MILESTONE_A_ARMS)


def hierarchical_residual_parameter_count(
    input_neurons: int,
    channels: int,
    classes: int,
    *,
    input_kernel_size: int,
    hidden_kernel_size: int,
    temporal_levels: Iterable[int],
    spiking: bool,
) -> int:
    state_parameters = channels * (2 if spiking else 1)
    pooled_features = 2 * sum(int(level) for level in temporal_levels) + 1
    return int(
        channels * input_neurons * input_kernel_size
        + channels
        + channels * channels * hidden_kernel_size
        + channels
        + state_parameters
        + channels * pooled_features * classes
        + classes
    )


def matched_hierarchical_residual_channels(
    input_neurons: int,
    classes: int,
    target_parameters: int,
    *,
    input_kernel_size: int,
    hidden_kernel_size: int,
    temporal_levels: Iterable[int],
) -> tuple[int, int]:
    channels = 1
    while hierarchical_residual_parameter_count(
        input_neurons,
        channels + 1,
        classes,
        input_kernel_size=input_kernel_size,
        hidden_kernel_size=hidden_kernel_size,
        temporal_levels=temporal_levels,
        spiking=True,
    ) <= target_parameters:
        channels += 1
    return channels, hierarchical_residual_parameter_count(
        input_neurons,
        channels,
        classes,
        input_kernel_size=input_kernel_size,
        hidden_kernel_size=hidden_kernel_size,
        temporal_levels=temporal_levels,
        spiking=True,
    )


class HierarchicalResidualStateClassifier(nn.Module):
    """Dilated temporal hierarchy with preserved direct and state traces."""

    def __init__(
        self,
        config: SHDConfig,
        *,
        channels: int,
        input_kernel_size: int,
        hidden_kernel_size: int,
        dilation: int,
        temporal_levels: Iterable[int],
        dynamics: str,
        surrogate_slope: float,
        initial_leak: float = 0.90,
        initial_threshold: float = 1.0,
    ) -> None:
        if torch is None:
            raise ImportError("Milestone A requires PyTorch")
        if dynamics not in {"analog", "lif"}:
            raise ValueError("dynamics must be analog or lif")
        super().__init__()
        self.config = config
        self.channels = int(channels)
        self.temporal_levels = tuple(int(level) for level in temporal_levels)
        self.dynamics = dynamics
        self.surrogate_slope = float(surrogate_slope)
        self.ablation_mode = "full"
        self.input_conv = nn.Conv1d(
            config.input_neurons,
            channels,
            input_kernel_size,
            padding=input_kernel_size // 2,
        )
        self.hidden_conv = nn.Conv1d(
            channels,
            channels,
            hidden_kernel_size,
            padding=dilation * (hidden_kernel_size // 2),
            dilation=dilation,
        )
        leak_logit = math.log(initial_leak / (1.0 - initial_leak))
        self.leak_logit = nn.Parameter(torch.full((channels,), leak_logit))
        if dynamics == "lif":
            threshold_raw = math.log(math.expm1(initial_threshold))
            self.threshold_raw = nn.Parameter(torch.full((channels,), threshold_raw))
        else:
            self.register_parameter("threshold_raw", None)
        pooled_features = 2 * sum(self.temporal_levels) + 1
        self.classifier = nn.Linear(channels * pooled_features, config.classes)

    def forward(self, events, *, return_event_rate: bool = False):  # type: ignore[override]
        if events.ndim != 3 or events.shape[2] != self.config.input_neurons:
            raise ValueError("events must have shape [batch, time, input_neurons]")
        first = torch.relu(
            self.input_conv(events.to(torch.float32).transpose(1, 2))
        )
        currents = (self.hidden_conv(first) + first).transpose(1, 2)
        direct_trace = torch.relu(currents)
        leak = torch.sigmoid(self.leak_logit)
        membrane = currents.new_zeros((events.shape[0], self.channels))
        states = []
        activity_sum = currents.new_zeros(())
        threshold = None
        if self.dynamics == "lif":
            threshold = torch.nn.functional.softplus(self.threshold_raw).clamp_min(1e-3)
        for step in range(currents.shape[1]):
            pre_reset = leak * membrane + currents[:, step]
            if self.dynamics == "lif":
                state = SurrogateSpike.apply(pre_reset - threshold, self.surrogate_slope)
                membrane = pre_reset - state * threshold
                activity_sum = activity_sum + state.mean()
            else:
                membrane = pre_reset
                state = torch.tanh(membrane)
                activity_sum = activity_sum + state.abs().mean()
            states.append(state)
        state_trace = torch.stack(states, dim=1)
        final_state = torch.tanh(membrane) if self.dynamics == "analog" else membrane / threshold
        if self.ablation_mode == "direct_only":
            state_trace = torch.zeros_like(state_trace)
            final_state = torch.zeros_like(final_state)
        elif self.ablation_mode == "state_only":
            direct_trace = torch.zeros_like(direct_trace)
        elif self.ablation_mode == "shuffled_state":
            state_trace = torch.roll(state_trace, shifts=1, dims=0)
            final_state = torch.roll(final_state, shifts=1, dims=0)
        features = _multiscale_features(direct_trace, self.temporal_levels)
        features.extend(_multiscale_features(state_trace, self.temporal_levels))
        features.append(final_state)
        logits = self.classifier(torch.cat(features, dim=1))
        if return_event_rate:
            return logits, activity_sum / int(currents.shape[1])
        return logits

    def set_ablation_mode(self, mode: str) -> None:
        if mode not in RESIDUAL_ABLATION_MODES:
            raise ValueError("unsupported hierarchical-state ablation mode")
        self.ablation_mode = mode


@dataclass
class MilestoneAResult:
    config: SHDConfig
    device: str
    target_parameters: int
    temporal_levels: tuple[int, ...]
    screen_seed: int
    confirm_seeds: tuple[int, ...]
    screen_records: list[dict]
    promoted_arms: tuple[str, ...]
    confirmation_records: list[dict]
    confirmation_summary: list[dict]
    decision: dict

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "milestone_a_architecture.json"
        screen_path = output / "milestone_a_screen.csv"
        records_path = output / "milestone_a_confirmation_records.csv"
        summary_path = output / "milestone_a_confirmation_summary.csv"
        payload = {
            "config": asdict(self.config),
            "device": self.device,
            "target_parameters": self.target_parameters,
            "temporal_levels": list(self.temporal_levels),
            "screen_seed": self.screen_seed,
            "confirm_seeds": list(self.confirm_seeds),
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
            plot_path = output / "milestone_a_architecture.png"
            plot_milestone_a(self.confirmation_summary, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def run_milestone_a(
    config: SHDConfig,
    *,
    screen_seed: int = 142,
    confirm_seeds: Iterable[int] = (142, 143, 144),
    screen_train_samples: int = 15_000,
    screen_validation_samples: int = 3_000,
    screen_test_samples: int = 3_000,
    screen_epochs: int = 4,
    confirm_epochs: int = 15,
    promotion_margin: float = 0.02,
    minimum_parameter_ratio: float = 0.95,
    maximum_parameter_ratio: float = 1.05,
    minimum_spike_rate: float = 0.01,
    maximum_spike_rate: float = 0.30,
    causal_margin: float = 0.01,
    target_parameters: int = 133_631,
    device="auto",
    temporal_levels: Iterable[int] = DEFAULT_TEMPORAL_LEVELS,
    input_kernel_size: int = 5,
    hidden_kernel_size: int = 3,
    tcn_dilation: int = 2,
    surrogate_slope: float = 10.0,
    progress_path: str | pathlib.Path | None = None,
) -> MilestoneAResult:
    if torch is None:
        raise ImportError("Milestone A requires PyTorch")
    levels = tuple(int(level) for level in temporal_levels)
    confirm_seed_tuple = tuple(int(seed) for seed in confirm_seeds)
    if not levels or any(level <= 0 for level in levels):
        raise ValueError("temporal_levels must contain positive integers")
    if not confirm_seed_tuple:
        raise ValueError("at least one confirmation seed is required")
    if screen_epochs <= 0 or confirm_epochs <= 0:
        raise ValueError("screen_epochs and confirm_epochs must be positive")
    if min(
        screen_train_samples,
        screen_validation_samples,
        screen_test_samples,
    ) < 0:
        raise ValueError("screen sample limits cannot be negative")
    if target_parameters <= 0:
        raise ValueError("target_parameters must be positive")
    if not 0.0 <= promotion_margin <= 1.0:
        raise ValueError("promotion_margin must be between zero and one")
    if not 0.0 < minimum_parameter_ratio <= maximum_parameter_ratio:
        raise ValueError("invalid parameter-ratio gate")
    if not 0.0 <= minimum_spike_rate <= maximum_spike_rate <= 1.0:
        raise ValueError("invalid spike-rate gate")
    signature = {
        "version": 1,
        "arms": list(available_milestone_a_arms()),
        "screen_seed": int(screen_seed),
        "confirm_seeds": list(confirm_seed_tuple),
        "screen_samples": [
            int(screen_train_samples),
            int(screen_validation_samples),
            int(screen_test_samples),
        ],
        "epochs": [int(screen_epochs), int(confirm_epochs)],
        "target_parameters": int(target_parameters),
        "promotion_margin": float(promotion_margin),
        "parameter_ratio_gate": [
            float(minimum_parameter_ratio),
            float(maximum_parameter_ratio),
        ],
        "spike_rate_gate": [float(minimum_spike_rate), float(maximum_spike_rate)],
        "causal_margin": float(causal_margin),
        "input_neurons": int(config.input_neurons),
        "classes": int(config.classes),
        "timesteps": int(config.timesteps),
        "duration_seconds": float(config.duration_seconds),
        "temporal_levels": list(levels),
        "input_kernel_size": int(input_kernel_size),
        "hidden_kernel_size": int(hidden_kernel_size),
        "tcn_dilation": int(tcn_dilation),
        "surrogate_slope": float(surrogate_slope),
        "learning_rate": float(config.learning_rate),
        "weight_decay": float(config.weight_decay),
        "batch_size": int(config.batch_size),
        "device": str(device),
        "data_root": str(config.data_root),
        "data_seed": int(config.data_seed),
    }
    progress = _load_progress(progress_path, signature)
    existing_screen = list(progress.get("screen_records", []))
    existing_confirmation = list(progress.get("confirmation_records", []))
    resolved = resolve_device(device)
    full_config = replace(
        config,
        train_samples=0,
        test_samples=0,
        epochs=int(confirm_epochs),
    )
    if progress.get("stage") == "complete":
        promoted = tuple(str(name) for name in progress.get("promoted_arms", []))
        confirmation_records = list(progress.get("confirmation_records", []))
        summary = summarize_milestone_a_confirmation(confirmation_records)
        decision = progress.get("decision") or decide_milestone_a(
            summary,
            causal_margin=causal_margin,
            minimum_spike_rate=minimum_spike_rate,
            maximum_spike_rate=maximum_spike_rate,
        )
        return MilestoneAResult(
            config=full_config,
            device=device_kind(resolved),
            target_parameters=int(target_parameters),
            temporal_levels=levels,
            screen_seed=int(screen_seed),
            confirm_seeds=confirm_seed_tuple,
            screen_records=existing_screen,
            promoted_arms=promoted,
            confirmation_records=confirmation_records,
            confirmation_summary=summary,
            decision=decision,
        )
    (
        train_events,
        train_labels,
        validation_events,
        validation_labels,
        test_events,
        test_labels,
    ) = load_ssc_tensors(full_config, validation_samples=0)
    expected_screen = {
        (int(screen_seed), arm.name) for arm in MILESTONE_A_ARMS
    }
    completed_screen = {
        (int(row["seed"]), str(row["arm"])) for row in existing_screen
    }
    screen_records = existing_screen
    if not expected_screen.issubset(completed_screen):
        screen_generator = torch.Generator(device="cpu").manual_seed(
            config.data_seed + 90_000
        )
        screen_train_events, screen_train_labels = _sample_split(
            train_events, train_labels, screen_train_samples, screen_generator
        )
        screen_validation_events, screen_validation_labels = _sample_split(
            validation_events,
            validation_labels,
            screen_validation_samples,
            screen_generator,
        )
        screen_test_events, screen_test_labels = _sample_split(
            test_events, test_labels, screen_test_samples, screen_generator
        )
        screen_config = replace(full_config, epochs=int(screen_epochs))
        screen_records = _run_stage(
            MILESTONE_A_ARMS,
            (int(screen_seed),),
            screen_config,
            screen_train_events,
            screen_train_labels,
            screen_validation_events,
            screen_validation_labels,
            screen_test_events,
            screen_test_labels,
            target_parameters=target_parameters,
            levels=levels,
            input_kernel_size=input_kernel_size,
            hidden_kernel_size=hidden_kernel_size,
            tcn_dilation=tcn_dilation,
            surrogate_slope=surrogate_slope,
            device=resolved,
            ablate=False,
            existing_records=screen_records,
            progress_callback=lambda rows: _save_progress(
                progress_path,
                signature,
                stage="screen",
                screen_records=rows,
                promoted_arms=(),
                confirmation_records=existing_confirmation,
            ),
        )
        del screen_train_events, screen_train_labels
        del screen_validation_events, screen_validation_labels
        del screen_test_events, screen_test_labels
        gc.collect()
    promoted = select_promoted_arms(
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
        stage="confirmation",
        screen_records=screen_records,
        promoted_arms=promoted,
        confirmation_records=existing_confirmation,
    )
    arm_lookup = {arm.name: arm for arm in MILESTONE_A_ARMS}
    confirmation_arms = tuple(arm_lookup[name] for name in promoted)
    confirmation_records = _run_stage(
        confirmation_arms,
        confirm_seed_tuple,
        full_config,
        train_events,
        train_labels,
        validation_events,
        validation_labels,
        test_events,
        test_labels,
        target_parameters=target_parameters,
        levels=levels,
        input_kernel_size=input_kernel_size,
        hidden_kernel_size=hidden_kernel_size,
        tcn_dilation=tcn_dilation,
        surrogate_slope=surrogate_slope,
        device=resolved,
        ablate=True,
        existing_records=existing_confirmation,
        progress_callback=lambda rows: _save_progress(
            progress_path,
            signature,
            stage="confirmation",
            screen_records=screen_records,
            promoted_arms=promoted,
            confirmation_records=rows,
        ),
    )
    summary = summarize_milestone_a_confirmation(confirmation_records)
    decision = decide_milestone_a(
        summary,
        causal_margin=causal_margin,
        minimum_spike_rate=minimum_spike_rate,
        maximum_spike_rate=maximum_spike_rate,
    )
    _save_progress(
        progress_path,
        signature,
        stage="complete",
        screen_records=screen_records,
        promoted_arms=promoted,
        confirmation_records=confirmation_records,
        decision=decision,
    )
    return MilestoneAResult(
        config=full_config,
        device=device_kind(resolved),
        target_parameters=int(target_parameters),
        temporal_levels=levels,
        screen_seed=int(screen_seed),
        confirm_seeds=confirm_seed_tuple,
        screen_records=screen_records,
        promoted_arms=promoted,
        confirmation_records=confirmation_records,
        confirmation_summary=summary,
        decision=decision,
    )


def select_promoted_arms(
    screen_records: Iterable[dict],
    *,
    promotion_margin: float,
    minimum_parameter_ratio: float,
    maximum_parameter_ratio: float,
    minimum_spike_rate: float,
    maximum_spike_rate: float,
) -> tuple[str, ...]:
    rows = list(screen_records)
    if not rows:
        return ()
    best_validation = max(float(row["best_validation_accuracy"]) for row in rows)
    conventional = [row for row in rows if bool(row["conventional"])]
    best_conventional = max(
        conventional, key=lambda row: float(row["best_validation_accuracy"])
    )["arm"]
    promoted = [str(best_conventional)]
    for row in rows:
        if row["arm"] == best_conventional or bool(row["conventional"]):
            continue
        ratio = float(row["parameter_ratio_vs_target"])
        validation_ok = float(row["best_validation_accuracy"]) >= (
            best_validation - promotion_margin
        )
        parameter_ok = minimum_parameter_ratio <= ratio <= maximum_parameter_ratio
        activity_ok = True
        if "lif" in str(row["model_kind"]):
            activity = float(row["checkpoint_activity"])
            activity_ok = minimum_spike_rate <= activity <= maximum_spike_rate
        if validation_ok and parameter_ok and activity_ok:
            promoted.append(str(row["arm"]))
    return tuple(promoted)


def summarize_milestone_a_confirmation(records: Iterable[dict]) -> list[dict]:
    rows = list(records)
    if not rows:
        return []
    conventional_rows = [row for row in rows if bool(row["conventional"])]
    conventional_means = {
        arm: statistics.fmean(
            float(row["full_accuracy"])
            for row in conventional_rows
            if row["arm"] == arm
        )
        for arm in {row["arm"] for row in conventional_rows}
    }
    best_conventional = max(conventional_means, key=conventional_means.get)
    best_conventional_mean = conventional_means[best_conventional]
    summary = []
    for arm in {row["arm"] for row in rows}:
        group = [row for row in rows if row["arm"] == arm]
        contributions = [
            float(row["state_contribution_vs_direct_only"])
            for row in group
            if row["state_contribution_vs_direct_only"] is not None
        ]
        specificities = [
            float(row["state_specificity_vs_shuffled"])
            for row in group
            if row["state_specificity_vs_shuffled"] is not None
        ]
        mean_accuracy = statistics.fmean(float(row["full_accuracy"]) for row in group)
        summary.append(
            {
                "arm": arm,
                "model_kind": group[0]["model_kind"],
                "conventional": bool(group[0]["conventional"]),
                "causal_state": bool(group[0]["causal_state"]),
                "runs": len(group),
                "mean_full_accuracy": mean_accuracy,
                "std_full_accuracy": statistics.pstdev(
                    float(row["full_accuracy"]) for row in group
                ),
                "best_conventional_arm": best_conventional,
                "mean_gain_vs_best_conventional": mean_accuracy - best_conventional_mean,
                "mean_state_contribution_vs_direct_only": (
                    statistics.fmean(contributions) if contributions else 0.0
                ),
                "one_point_seed_count_state_contribution": sum(
                    gain >= 0.01 for gain in contributions
                ),
                "mean_state_specificity_vs_shuffled": (
                    statistics.fmean(specificities) if specificities else 0.0
                ),
                "one_point_seed_count_state_specificity": sum(
                    gain >= 0.01 for gain in specificities
                ),
                "mean_activity": statistics.fmean(
                    float(row["checkpoint_activity"]) for row in group
                ),
                "activity_kind": group[0]["activity_kind"],
                "effective_trainable_parameters": int(
                    group[0]["effective_trainable_parameters"]
                ),
                "mean_test_examples_per_second": statistics.fmean(
                    float(row["test_examples_per_second"]) for row in group
                ),
                "mean_train_seconds": statistics.fmean(
                    float(row["train_seconds"]) for row in group
                ),
            }
        )
    return sorted(
        summary,
        key=lambda row: (-float(row["mean_full_accuracy"]), str(row["arm"])),
    )


def decide_milestone_a(
    summary: Iterable[dict],
    *,
    causal_margin: float,
    minimum_spike_rate: float,
    maximum_spike_rate: float,
) -> dict:
    rows = list(summary)
    if not rows:
        return {"status": "no_confirmation", "qualified_arms": []}
    qualified = []
    required_seed_count = 2 if max(int(row["runs"]) for row in rows) >= 3 else 1
    for row in rows:
        if not bool(row["causal_state"]) or "lif" not in str(row["model_kind"]):
            continue
        accuracy_ok = float(row["mean_gain_vs_best_conventional"]) >= -0.02
        causal_ok = (
            float(row["mean_state_contribution_vs_direct_only"]) >= causal_margin
            and int(row["one_point_seed_count_state_contribution"]) >= required_seed_count
            and float(row["mean_state_specificity_vs_shuffled"]) >= causal_margin
            and int(row["one_point_seed_count_state_specificity"]) >= required_seed_count
        )
        activity_ok = True
        if "lif" in str(row["model_kind"]):
            activity = float(row["mean_activity"])
            activity_ok = minimum_spike_rate <= activity <= maximum_spike_rate
        if accuracy_ok and causal_ok and activity_ok:
            qualified.append(str(row["arm"]))
    return {
        "status": "pass" if qualified else "stop",
        "best_arm": str(rows[0]["arm"]),
        "best_accuracy": float(rows[0]["mean_full_accuracy"]),
        "best_conventional_arm": str(rows[0]["best_conventional_arm"]),
        "qualified_arms": qualified,
        "next_milestone": "hardware_efficiency" if qualified else "close_architecture_branch",
    }


def plot_milestone_a(summary: list[dict], path: str | pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    labels = [row["arm"].replace("_", "\n") for row in summary]
    figure, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)
    axes[0].bar(
        labels,
        [100.0 * float(row["mean_full_accuracy"]) for row in summary],
        yerr=[100.0 * float(row["std_full_accuracy"]) for row in summary],
        capsize=5,
        color="#35b4f2",
    )
    axes[0].set_ylabel("SSC test accuracy (%)")
    axes[0].set_title("AMMC Gen-5 Milestone A: screened architecture confirmation")
    axes[1].bar(
        labels,
        [100.0 * float(row["mean_state_specificity_vs_shuffled"]) for row in summary],
        color="#167d55",
    )
    axes[1].axhline(1.0, color="#bd3d3a", linestyle="--", label="+1 point causal gate")
    axes[1].set_ylabel("Full - shuffled state (points)")
    axes[1].legend()
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _run_stage(
    arms: Iterable[MilestoneAArm],
    seeds: Iterable[int],
    config: SHDConfig,
    train_events,
    train_labels,
    validation_events,
    validation_labels,
    test_events,
    test_labels,
    *,
    target_parameters: int,
    levels: tuple[int, ...],
    input_kernel_size: int,
    hidden_kernel_size: int,
    tcn_dilation: int,
    surrogate_slope: float,
    device,
    ablate: bool,
    existing_records: Iterable[dict] = (),
    progress_callback=None,
) -> list[dict]:
    records = list(existing_records)
    completed = {(int(row["seed"]), str(row["arm"])) for row in records}
    for seed in seeds:
        for arm in arms:
            if (int(seed), arm.name) in completed:
                continue
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
                device=device,
            )
            training = _train_validation_selected(
                model,
                train_events,
                train_labels,
                validation_events,
                validation_labels,
                config,
                seed=seed,
                device=device,
            )
            model.load_state_dict(training["best_state"])
            measurements = {}
            modes = RESIDUAL_ABLATION_MODES if ablate and arm.causal_state else ("full",)
            for mode in modes:
                if hasattr(model, "set_ablation_mode"):
                    model.set_ablation_mode(mode)
                accuracy, seconds, activity = _measure(
                    model, test_events, test_labels, config.batch_size, device
                )
                measurements[mode] = (float(accuracy), float(seconds), float(activity))
            if hasattr(model, "set_ablation_mode"):
                model.set_ablation_mode("full")
            full_accuracy, full_seconds, full_activity = measurements["full"]
            direct_only = measurements.get("direct_only", (None, None, None))[0]
            state_only = measurements.get("state_only", (None, None, None))[0]
            shuffled = measurements.get("shuffled_state", (None, None, None))[0]
            parameters = sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            )
            records.append(
                {
                    "seed": int(seed),
                    "arm": arm.name,
                    "model_kind": arm.model_kind,
                    "conventional": arm.conventional,
                    "causal_state": arm.causal_state,
                    "channels": int(channels),
                    "best_epoch": int(training["best_epoch"]),
                    "best_validation_accuracy": float(training["best_validation_accuracy"]),
                    "full_accuracy": float(full_accuracy),
                    "direct_only_accuracy": direct_only,
                    "state_only_accuracy": state_only,
                    "shuffled_state_accuracy": shuffled,
                    "state_contribution_vs_direct_only": (
                        float(full_accuracy - direct_only) if direct_only is not None else None
                    ),
                    "state_specificity_vs_shuffled": (
                        float(full_accuracy - shuffled) if shuffled is not None else None
                    ),
                    "checkpoint_activity": float(full_activity),
                    "activity_kind": activity_kind,
                    "effective_trainable_parameters": int(parameters),
                    "parameter_ratio_vs_target": float(parameters / target_parameters),
                    "test_examples_per_second": float(
                        test_events.shape[0] / max(full_seconds, 1e-12)
                    ),
                    "train_seconds": float(training["train_seconds"]),
                    "train_samples": int(train_events.shape[0]),
                    "validation_samples": int(validation_events.shape[0]),
                    "test_samples": int(test_events.shape[0]),
                }
            )
            completed.add((int(seed), arm.name))
            if progress_callback is not None:
                progress_callback(records)
    return records


def _build_model(
    arm: MilestoneAArm,
    config: SHDConfig,
    *,
    target_parameters: int,
    levels: tuple[int, ...],
    input_kernel_size: int,
    hidden_kernel_size: int,
    tcn_dilation: int,
    surrogate_slope: float,
    device,
):
    if arm.model_kind == "conv1d":
        channels, _ = matched_temporal_conv_channels(
            config.input_neurons,
            config.classes,
            target_parameters,
            kernel_size=input_kernel_size,
            temporal_levels=levels,
        )
        model = TemporalConvClassifier(
            config,
            channels=channels,
            kernel_size=input_kernel_size,
            temporal_levels=levels,
        )
        activity_kind = "relu_activation"
    elif arm.model_kind == "tcn":
        channels, _ = matched_temporal_tcn_channels(
            config.input_neurons,
            config.classes,
            target_parameters,
            input_kernel_size=input_kernel_size,
            hidden_kernel_size=hidden_kernel_size,
            temporal_levels=levels,
        )
        model = TemporalDilatedTCNClassifier(
            config,
            channels=channels,
            input_kernel_size=input_kernel_size,
            hidden_kernel_size=hidden_kernel_size,
            dilation=tcn_dilation,
            temporal_levels=levels,
        )
        activity_kind = "relu_activation"
    elif arm.model_kind == "residual_lif":
        channels, _ = matched_temporal_conv_residual_channels(
            config.input_neurons,
            config.classes,
            target_parameters,
            kernel_size=input_kernel_size,
            temporal_levels=levels,
        )
        model = ResidualTemporalConvStateClassifier(
            config,
            channels=channels,
            kernel_size=input_kernel_size,
            temporal_levels=levels,
            dynamics="lif",
            surrogate_slope=surrogate_slope,
        )
        activity_kind = "spike_rate"
    else:
        channels, _ = matched_hierarchical_residual_channels(
            config.input_neurons,
            config.classes,
            target_parameters,
            input_kernel_size=input_kernel_size,
            hidden_kernel_size=hidden_kernel_size,
            temporal_levels=levels,
        )
        dynamics = "lif" if arm.model_kind == "hierarchical_lif" else "analog"
        model = HierarchicalResidualStateClassifier(
            config,
            channels=channels,
            input_kernel_size=input_kernel_size,
            hidden_kernel_size=hidden_kernel_size,
            dilation=tcn_dilation,
            temporal_levels=levels,
            dynamics=dynamics,
            surrogate_slope=surrogate_slope,
        )
        activity_kind = "spike_rate" if dynamics == "lif" else "analog_activation"
    return model.to(device), channels, activity_kind


def _sample_split(events, labels, limit: int, generator):
    if limit <= 0 or limit >= events.shape[0]:
        return events, labels
    indices = torch.randperm(events.shape[0], generator=generator)[:limit]
    return events.index_select(0, indices), labels.index_select(0, indices)


def _multiscale_features(trace, levels: Iterable[int]) -> list:
    timesteps = int(trace.shape[1])
    features = []
    for level in levels:
        for window in range(int(level)):
            start = window * timesteps // int(level)
            stop = (window + 1) * timesteps // int(level)
            features.append(trace[:, start:stop].mean(dim=1))
    return features


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_progress(path, signature: dict) -> dict:
    if path is None:
        return {}
    progress_path = pathlib.Path(path)
    if not progress_path.exists():
        return {}
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    if payload.get("signature") != signature:
        raise ValueError(
            f"progress checkpoint does not match this run: {progress_path}"
        )
    return payload


def _save_progress(
    path,
    signature: dict,
    *,
    stage: str,
    screen_records: Iterable[dict],
    promoted_arms: Iterable[str],
    confirmation_records: Iterable[dict],
    decision: dict | None = None,
) -> None:
    if path is None:
        return
    progress_path = pathlib.Path(path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "signature": signature,
        "stage": stage,
        "screen_records": list(screen_records),
        "promoted_arms": list(promoted_arms),
        "confirmation_records": list(confirmation_records),
        "decision": decision,
    }
    temporary = progress_path.with_suffix(progress_path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(progress_path)
