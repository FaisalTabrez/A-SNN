"""Gen-7 predictive, sample-conditioned residual-state experiment."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
import gc
import json
import math
import pathlib
import statistics
import time
from typing import Iterable

from .event_mnist import nn, torch
from .milestone_a_architecture import (
    _load_progress,
    _multiscale_features,
    _sample_split,
    _save_progress,
)
from .runtime import device_kind, mark_step, resolve_device, seed_everything, sync
from .shd_benchmark import SHDConfig, _measure
from .shd_temporal_pyramid import DEFAULT_TEMPORAL_LEVELS
from .ssc_benchmark import load_ssc_tensors
from .ssc_efficiency_baselines import (
    TemporalDilatedTCNClassifier,
    matched_temporal_tcn_channels,
    temporal_tcn_parameter_count,
)
from .trainable_temporal_mnist import SurrogateSpike


PREDICTIVE_ABLATION_MODES = (
    "full",
    "direct_only",
    "state_only",
    "shuffled_state",
    "reversed_state",
)


@dataclass(frozen=True)
class Gen7PredictiveStateArm:
    name: str
    model_kind: str
    conventional: bool
    causal_state: bool
    dynamics: str | None
    predictive_weight: float
    shuffled_future_targets: bool = False


GEN7_PREDICTIVE_STATE_ARMS = (
    Gen7PredictiveStateArm(
        "dilated_tcn", "tcn", True, False, None, 0.0
    ),
    Gen7PredictiveStateArm(
        "lif_no_predictive", "predictive_lif", False, True, "lif", 0.0
    ),
    Gen7PredictiveStateArm(
        "analog_paired_predictive",
        "predictive_analog",
        False,
        True,
        "analog",
        0.20,
    ),
    Gen7PredictiveStateArm(
        "lif_shuffled_predictive",
        "predictive_lif",
        False,
        True,
        "lif",
        0.20,
        True,
    ),
    Gen7PredictiveStateArm(
        "lif_paired_predictive",
        "predictive_lif",
        False,
        True,
        "lif",
        0.20,
    ),
)


def available_gen7_predictive_state_arms() -> tuple[str, ...]:
    return tuple(arm.name for arm in GEN7_PREDICTIVE_STATE_ARMS)


def predictive_state_parameter_count(
    input_neurons: int,
    channels: int,
    classes: int,
    *,
    input_kernel_size: int,
    hidden_kernel_size: int,
    temporal_levels: Iterable[int],
    spiking: bool,
) -> int:
    """Count TCN plus state dynamics, predictor, and conditional gate."""

    return int(
        temporal_tcn_parameter_count(
            input_neurons,
            channels,
            classes,
            input_kernel_size=input_kernel_size,
            hidden_kernel_size=hidden_kernel_size,
            temporal_levels=temporal_levels,
        )
        + channels  # leak
        + (channels if spiking else 0)  # threshold
        + channels * channels  # bias-free future projection
        + channels * classes
        + classes  # sample-conditioned gate projection
    )


def matched_predictive_state_channels(
    input_neurons: int,
    classes: int,
    target_parameters: int,
    *,
    input_kernel_size: int,
    hidden_kernel_size: int,
    temporal_levels: Iterable[int],
) -> tuple[int, int]:
    """Preserve the exact width selected for the matched TCN."""

    channels, _ = matched_temporal_tcn_channels(
        input_neurons,
        classes,
        target_parameters,
        input_kernel_size=input_kernel_size,
        hidden_kernel_size=hidden_kernel_size,
        temporal_levels=temporal_levels,
    )
    return channels, predictive_state_parameter_count(
        input_neurons,
        channels,
        classes,
        input_kernel_size=input_kernel_size,
        hidden_kernel_size=hidden_kernel_size,
        temporal_levels=temporal_levels,
        spiking=True,
    )


class PredictiveStateTCNClassifier(nn.Module):
    """TCN plus predictive state and a zero-initialized conditional gate."""

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
        future_horizon: int = 4,
        contrastive_temperature: float = 0.10,
    ) -> None:
        if torch is None:
            raise ImportError("Gen-7 predictive state requires PyTorch")
        if dynamics not in {"analog", "lif"}:
            raise ValueError("dynamics must be analog or lif")
        if future_horizon <= 0 or future_horizon >= config.timesteps:
            raise ValueError("future_horizon must be between 1 and timesteps - 1")
        if contrastive_temperature <= 0.0:
            raise ValueError("contrastive_temperature must be positive")
        super().__init__()
        self.config = config
        self.channels = int(channels)
        self.temporal_levels = tuple(int(level) for level in temporal_levels)
        self.dynamics = dynamics
        self.surrogate_slope = float(surrogate_slope)
        self.future_horizon = int(future_horizon)
        self.contrastive_temperature = float(contrastive_temperature)
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
        feature_dim = channels * sum(self.temporal_levels)
        self.classifier = nn.Linear(feature_dim, config.classes)
        initial_leaks = (0.50, 0.75, 0.90, 0.97)
        leak_values = [initial_leaks[index % len(initial_leaks)] for index in range(channels)]
        leak_logits = [math.log(value / (1.0 - value)) for value in leak_values]
        self.leak_logit = nn.Parameter(torch.tensor(leak_logits, dtype=torch.float32))
        if dynamics == "lif":
            threshold_raw = math.log(math.expm1(1.0))
            self.threshold_raw = nn.Parameter(
                torch.full((channels,), threshold_raw, dtype=torch.float32)
            )
        else:
            self.register_parameter("threshold_raw", None)
        self.future_projection = nn.Linear(channels, channels, bias=False)
        self.conditional_gate = nn.Linear(channels, config.classes)
        nn.init.zeros_(self.conditional_gate.weight)
        nn.init.zeros_(self.conditional_gate.bias)

    def forward(self, events, *, return_event_rate: bool = False):  # type: ignore[override]
        representations = self._representations(events)
        logits = self._logits(representations)
        if return_event_rate:
            return logits, representations["activity"]
        return logits

    def training_terms(self, events, *, shuffled_future_targets: bool = False):
        representations = self._representations(events)
        logits = self._logits(representations)
        loss, margin = self._predictive_objective(
            representations["state_trace"],
            representations["currents"],
            shuffled_targets=shuffled_future_targets,
        )
        return logits, loss, margin, representations["activity"]

    def future_alignment(self, events):
        representations = self._representations(events)
        _, margin = self._predictive_objective(
            representations["state_trace"],
            representations["currents"],
            shuffled_targets=False,
        )
        return margin

    def sample_gate_activity(self, events):
        representations = self._representations(events)
        interaction = (
            representations["direct_trace"].mean(dim=1)
            * representations["state_trace"].mean(dim=1)
        )
        return torch.tanh(self.conditional_gate(interaction)).abs().mean()

    def _representations(self, events):
        if events.ndim != 3 or events.shape[2] != self.config.input_neurons:
            raise ValueError("events must have shape [batch, time, input_neurons]")
        first = torch.relu(
            self.input_conv(events.to(torch.float32).transpose(1, 2))
        )
        currents = (self.hidden_conv(first) + first).transpose(1, 2)
        direct_trace = torch.relu(currents)
        state_trace, activity = self._state_trace(currents)
        return {
            "currents": currents,
            "direct_trace": direct_trace,
            "state_trace": state_trace,
            "activity": activity,
        }

    def _logits(self, representations):
        direct_trace = representations["direct_trace"]
        state_trace = representations["state_trace"]
        if self.ablation_mode == "shuffled_state":
            state_trace = torch.roll(state_trace, shifts=1, dims=0)
        elif self.ablation_mode == "reversed_state":
            state_trace = torch.flip(state_trace, dims=(1,))
        direct_features = torch.cat(
            _multiscale_features(direct_trace, self.temporal_levels), dim=1
        )
        state_features = torch.cat(
            _multiscale_features(state_trace, self.temporal_levels), dim=1
        )
        direct_logits = self.classifier(direct_features)
        state_logits = torch.nn.functional.linear(
            state_features, self.classifier.weight, bias=None
        )
        interaction = direct_trace.mean(dim=1) * state_trace.mean(dim=1)
        gate = torch.tanh(self.conditional_gate(interaction))
        correction = gate * state_logits
        if self.ablation_mode == "direct_only":
            return direct_logits
        if self.ablation_mode == "state_only":
            return correction + self.classifier.bias
        return direct_logits + correction

    def _state_trace(self, currents):
        leak = torch.sigmoid(self.leak_logit)
        membrane = currents.new_zeros((currents.shape[0], self.channels))
        states = []
        activity_sum = currents.new_zeros(())
        threshold = None
        if self.dynamics == "lif":
            threshold = torch.nn.functional.softplus(self.threshold_raw).clamp_min(1e-3)
        for step in range(currents.shape[1]):
            pre_reset = leak * membrane + currents[:, step]
            if self.dynamics == "lif":
                state = SurrogateSpike.apply(
                    pre_reset - threshold, self.surrogate_slope
                )
                membrane = pre_reset - state * threshold
                activity_sum = activity_sum + state.mean()
            else:
                membrane = pre_reset
                state = torch.tanh(membrane)
                activity_sum = activity_sum + state.abs().mean()
            states.append(state)
        return torch.stack(states, dim=1), activity_sum / int(currents.shape[1])

    def _predictive_objective(self, state_trace, currents, *, shuffled_targets: bool):
        horizon = self.future_horizon
        state_summary = state_trace[:, :-horizon].mean(dim=1)
        future_summary = torch.relu(currents[:, horizon:]).mean(dim=1).detach()
        if shuffled_targets and future_summary.shape[0] > 1:
            future_summary = torch.roll(future_summary, shifts=1, dims=0)
        prediction = torch.nn.functional.normalize(
            self.future_projection(state_summary), dim=1
        )
        target = torch.nn.functional.normalize(future_summary, dim=1)
        similarities = prediction @ target.transpose(0, 1)
        paired = similarities.diagonal().mean()
        if similarities.shape[0] > 1:
            shuffled = torch.roll(target, shifts=1, dims=0)
            shuffled_similarity = (prediction * shuffled).sum(dim=1).mean()
            labels = torch.arange(similarities.shape[0], device=similarities.device)
            scaled = similarities / self.contrastive_temperature
            loss = 0.5 * (
                torch.nn.functional.cross_entropy(scaled, labels)
                + torch.nn.functional.cross_entropy(scaled.transpose(0, 1), labels)
            )
        else:
            shuffled_similarity = paired.detach()
            loss = similarities.sum() * 0.0
        return loss, paired - shuffled_similarity

    def set_ablation_mode(self, mode: str) -> None:
        if mode not in PREDICTIVE_ABLATION_MODES:
            raise ValueError("unsupported predictive-state ablation mode")
        self.ablation_mode = mode

@dataclass
class Gen7PredictiveStateResult:
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
        json_path = output / "gen7_predictive_state.json"
        screen_path = output / "gen7_predictive_state_screen.csv"
        records_path = output / "gen7_predictive_state_confirmation_records.csv"
        summary_path = output / "gen7_predictive_state_confirmation_summary.csv"
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
            plot_path = output / "gen7_predictive_state.png"
            plot_gen7_predictive_state(self.confirmation_summary, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def run_gen7_predictive_state(
    config: SHDConfig,
    *,
    screen_seed: int = 142,
    confirm_seeds: Iterable[int] = (142, 143, 144),
    screen_train_samples: int = 15_000,
    screen_validation_samples: int = 3_000,
    screen_test_samples: int = 3_000,
    screen_epochs: int = 4,
    confirm_epochs: int = 15,
    promotion_margin: float = 0.01,
    minimum_parameter_ratio: float = 0.95,
    maximum_parameter_ratio: float = 1.05,
    minimum_spike_rate: float = 0.01,
    maximum_spike_rate: float = 0.30,
    accuracy_margin: float = 0.01,
    causal_margin: float = 0.005,
    alignment_margin: float = 0.02,
    alignment_control_margin: float = 0.01,
    minimum_gate: float = 0.01,
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
) -> Gen7PredictiveStateResult:
    if torch is None:
        raise ImportError("Gen-7 predictive state requires PyTorch")
    levels = tuple(int(level) for level in temporal_levels)
    seeds = tuple(int(seed) for seed in confirm_seeds)
    _validate_run(
        config,
        levels,
        seeds,
        screen_epochs,
        confirm_epochs,
        (screen_train_samples, screen_validation_samples, screen_test_samples),
        target_parameters,
        (minimum_parameter_ratio, maximum_parameter_ratio),
        (minimum_spike_rate, maximum_spike_rate),
        (promotion_margin, accuracy_margin, causal_margin),
        (alignment_margin, alignment_control_margin),
        minimum_gate,
        future_horizon,
        contrastive_temperature,
    )
    signature = _run_signature(
        config,
        screen_seed=screen_seed,
        confirm_seeds=seeds,
        screen_samples=(screen_train_samples, screen_validation_samples, screen_test_samples),
        epochs=(screen_epochs, confirm_epochs),
        target_parameters=target_parameters,
        promotion_margin=promotion_margin,
        parameter_gate=(minimum_parameter_ratio, maximum_parameter_ratio),
        spike_gate=(minimum_spike_rate, maximum_spike_rate),
        accuracy_margin=accuracy_margin,
        causal_margin=causal_margin,
        alignment_margin=alignment_margin,
        alignment_control_margin=alignment_control_margin,
        minimum_gate=minimum_gate,
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
    existing_screen = list(progress.get("screen_records", []))
    existing_confirmation = list(progress.get("confirmation_records", []))
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
            accuracy_margin,
            causal_margin,
            alignment_margin,
            alignment_control_margin,
            minimum_spike_rate,
            maximum_spike_rate,
            minimum_gate,
        )
    data = load_ssc_tensors(full_config, validation_samples=0)
    train_events, train_labels, validation_events, validation_labels, test_events, test_labels = data
    expected_screen = {(int(screen_seed), arm.name) for arm in GEN7_PREDICTIVE_STATE_ARMS}
    completed_screen = {(int(row["seed"]), str(row["arm"])) for row in existing_screen}
    screen_records = existing_screen
    if not expected_screen.issubset(completed_screen):
        generator = torch.Generator(device="cpu").manual_seed(config.data_seed + 97_000)
        screen_train_events, screen_train_labels = _sample_split(
            train_events, train_labels, screen_train_samples, generator
        )
        screen_validation_events, screen_validation_labels = _sample_split(
            validation_events, validation_labels, screen_validation_samples, generator
        )
        screen_test_events, screen_test_labels = _sample_split(
            test_events, test_labels, screen_test_samples, generator
        )
        screen_records = _run_stage(
            GEN7_PREDICTIVE_STATE_ARMS,
            (int(screen_seed),),
            replace(full_config, epochs=screen_epochs),
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
            future_horizon=future_horizon,
            contrastive_temperature=contrastive_temperature,
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
    promoted = select_gen7_promoted_arms(
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
    lookup = {arm.name: arm for arm in GEN7_PREDICTIVE_STATE_ARMS}
    confirmation_records = _run_stage(
        tuple(lookup[name] for name in promoted),
        seeds,
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
        future_horizon=future_horizon,
        contrastive_temperature=contrastive_temperature,
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
    summary = summarize_gen7_confirmation(confirmation_records)
    decision = decide_gen7_predictive_state(
        summary,
        accuracy_margin=accuracy_margin,
        causal_margin=causal_margin,
        alignment_margin=alignment_margin,
        alignment_control_margin=alignment_control_margin,
        minimum_spike_rate=minimum_spike_rate,
        maximum_spike_rate=maximum_spike_rate,
        minimum_gate=minimum_gate,
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
    return Gen7PredictiveStateResult(
        config=full_config,
        device=device_kind(resolved),
        target_parameters=target_parameters,
        temporal_levels=levels,
        screen_seed=int(screen_seed),
        confirm_seeds=seeds,
        screen_records=screen_records,
        promoted_arms=promoted,
        confirmation_records=confirmation_records,
        confirmation_summary=summary,
        decision=decision,
    )


def select_gen7_promoted_arms(
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
    threshold = float(baseline["best_validation_accuracy"]) - promotion_margin
    promoted = ["dilated_tcn"]
    for arm in GEN7_PREDICTIVE_STATE_ARMS[1:]:
        row = next(row for row in rows if row["arm"] == arm.name)
        ratio = float(row["parameter_ratio_vs_target"])
        activity_ok = True
        if arm.dynamics == "lif":
            activity = float(row["checkpoint_activity"])
            activity_ok = minimum_spike_rate <= activity <= maximum_spike_rate
        if (
            float(row["best_validation_accuracy"]) >= threshold
            and minimum_parameter_ratio <= ratio <= maximum_parameter_ratio
            and activity_ok
        ):
            promoted.append(arm.name)
    if "lif_paired_predictive" in promoted:
        for required_control in ("lif_no_predictive", "lif_shuffled_predictive"):
            if required_control not in promoted:
                promoted.append(required_control)
    order = available_gen7_predictive_state_arms()
    return tuple(name for name in order if name in promoted)


def summarize_gen7_confirmation(records: Iterable[dict]) -> list[dict]:
    rows = list(records)
    if not rows:
        return []
    baseline_rows = [row for row in rows if row["arm"] == "dilated_tcn"]
    baseline_mean = statistics.fmean(float(row["full_accuracy"]) for row in baseline_rows)
    summary = []
    for arm in available_gen7_predictive_state_arms():
        group = [row for row in rows if row["arm"] == arm]
        if not group:
            continue
        accuracy = statistics.fmean(float(row["full_accuracy"]) for row in group)
        contributions = _defined_values(group, "state_contribution_vs_direct_only")
        specificities = _defined_values(group, "state_specificity_vs_shuffled")
        reversal = _defined_values(group, "state_temporal_order_vs_reversed")
        alignment = _defined_values(group, "future_alignment_margin")
        summary.append(
            {
                "arm": arm,
                "model_kind": group[0]["model_kind"],
                "conventional": bool(group[0]["conventional"]),
                "causal_state": bool(group[0]["causal_state"]),
                "runs": len(group),
                "mean_full_accuracy": accuracy,
                "std_full_accuracy": statistics.pstdev(float(row["full_accuracy"]) for row in group),
                "mean_gain_vs_tcn": accuracy - baseline_mean,
                "mean_state_contribution_vs_direct_only": _mean_or_zero(contributions),
                "half_point_seed_count_state_contribution": sum(value >= 0.005 for value in contributions),
                "mean_state_specificity_vs_shuffled": _mean_or_zero(specificities),
                "half_point_seed_count_state_specificity": sum(value >= 0.005 for value in specificities),
                "mean_state_temporal_order_vs_reversed": _mean_or_zero(reversal),
                "half_point_seed_count_temporal_order": sum(value >= 0.005 for value in reversal),
                "mean_future_alignment_margin": _mean_or_zero(alignment),
                "alignment_seed_count": sum(value >= 0.02 for value in alignment),
                "mean_activity": statistics.fmean(float(row["checkpoint_activity"]) for row in group),
                "activity_kind": group[0]["activity_kind"],
                "mean_absolute_gate": statistics.fmean(float(row["mean_absolute_gate"]) for row in group),
                "effective_trainable_parameters": int(group[0]["effective_trainable_parameters"]),
                "mean_test_examples_per_second": statistics.fmean(float(row["test_examples_per_second"]) for row in group),
                "mean_train_seconds": statistics.fmean(float(row["train_seconds"]) for row in group),
            }
        )
    return sorted(summary, key=lambda row: (-float(row["mean_full_accuracy"]), str(row["arm"])))


def decide_gen7_predictive_state(
    summary: Iterable[dict],
    *,
    accuracy_margin: float,
    causal_margin: float,
    alignment_margin: float,
    alignment_control_margin: float,
    minimum_spike_rate: float,
    maximum_spike_rate: float,
    minimum_gate: float,
) -> dict:
    rows = list(summary)
    if not rows:
        return {"status": "no_confirmation", "qualified_arms": []}
    lookup = {str(row["arm"]): row for row in rows}
    candidate = lookup.get("lif_paired_predictive")
    control = lookup.get("lif_shuffled_predictive")
    required = 2 if max(int(row["runs"]) for row in rows) >= 3 else 1
    qualified = []
    alignment_control_gain = None
    if candidate is not None and control is not None:
        alignment_control_gain = (
            float(candidate["mean_future_alignment_margin"])
            - float(control["mean_future_alignment_margin"])
        )
        if (
            float(candidate["mean_gain_vs_tcn"]) >= -accuracy_margin
            and float(candidate["mean_state_contribution_vs_direct_only"]) >= causal_margin
            and int(candidate["half_point_seed_count_state_contribution"]) >= required
            and float(candidate["mean_state_specificity_vs_shuffled"]) >= causal_margin
            and int(candidate["half_point_seed_count_state_specificity"]) >= required
            and float(candidate["mean_state_temporal_order_vs_reversed"]) >= causal_margin
            and int(candidate["half_point_seed_count_temporal_order"]) >= required
            and float(candidate["mean_future_alignment_margin"]) >= alignment_margin
            and int(candidate["alignment_seed_count"]) >= required
            and alignment_control_gain >= alignment_control_margin
            and minimum_spike_rate <= float(candidate["mean_activity"]) <= maximum_spike_rate
            and float(candidate["mean_absolute_gate"]) >= minimum_gate
        ):
            qualified.append("lif_paired_predictive")
    return {
        "status": "pass" if qualified else "stop",
        "best_arm": str(rows[0]["arm"]),
        "best_accuracy": float(rows[0]["mean_full_accuracy"]),
        "alignment_gain_vs_shuffled_training": alignment_control_gain,
        "qualified_arms": qualified,
        "next_milestone": "runtime_efficiency" if qualified else "close_gen7_predictive_state",
    }


def plot_gen7_predictive_state(summary: list[dict], path: str | pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    labels = [row["arm"].replace("_", "\n") for row in summary]
    figure, axes = plt.subplots(3, 1, figsize=(14, 13), constrained_layout=True)
    axes[0].bar(
        labels,
        [100.0 * float(row["mean_full_accuracy"]) for row in summary],
        yerr=[100.0 * float(row["std_full_accuracy"]) for row in summary],
        capsize=5,
        color="#35b4f2",
    )
    axes[0].set_ylabel("SSC test accuracy (%)")
    axes[0].set_title("Gen-7 predictive-state successor")
    axes[1].bar(
        labels,
        [100.0 * float(row["mean_state_specificity_vs_shuffled"]) for row in summary],
        color="#167d55",
    )
    axes[1].axhline(0.5, color="#bd3d3a", linestyle="--", label="+0.5 point gate")
    axes[1].set_ylabel("Full - shuffled state (points)")
    axes[1].legend()
    axes[2].bar(
        labels,
        [float(row["mean_future_alignment_margin"]) for row in summary],
        color="#ffb31a",
    )
    axes[2].axhline(0.02, color="#bd3d3a", linestyle="--", label="0.02 alignment gate")
    axes[2].set_ylabel("Paired - shuffled future cosine")
    axes[2].legend()
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _run_stage(
    arms,
    seeds,
    config,
    train_events,
    train_labels,
    validation_events,
    validation_labels,
    test_events,
    test_labels,
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
    ablate,
    existing_records=(),
    progress_callback=None,
):
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
            measurements = {}
            modes = PREDICTIVE_ABLATION_MODES if ablate and arm.causal_state else ("full",)
            for mode in modes:
                if hasattr(model, "set_ablation_mode"):
                    model.set_ablation_mode(mode)
                measurements[mode] = _measure(model, test_events, test_labels, config.batch_size, device)
            if hasattr(model, "set_ablation_mode"):
                model.set_ablation_mode("full")
            future_alignment = _measure_future_alignment(model, test_events, config.batch_size, device)
            full_accuracy, full_seconds, full_activity = measurements["full"]
            direct = measurements.get("direct_only", (None, None, None))[0]
            state = measurements.get("state_only", (None, None, None))[0]
            shuffled = measurements.get("shuffled_state", (None, None, None))[0]
            reversed_state = measurements.get("reversed_state", (None, None, None))[0]
            parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            gate = _measure_sample_gate(model, test_events, config.batch_size, device)
            records.append(
                {
                    "seed": int(seed),
                    "arm": arm.name,
                    "model_kind": arm.model_kind,
                    "conventional": arm.conventional,
                    "causal_state": arm.causal_state,
                    "predictive_weight": float(arm.predictive_weight),
                    "shuffled_future_targets": bool(arm.shuffled_future_targets),
                    "channels": int(channels),
                    "best_epoch": int(training["best_epoch"]),
                    "best_validation_accuracy": float(training["best_validation_accuracy"]),
                    "full_accuracy": float(full_accuracy),
                    "direct_only_accuracy": direct,
                    "state_only_accuracy": state,
                    "shuffled_state_accuracy": shuffled,
                    "reversed_state_accuracy": reversed_state,
                    "state_contribution_vs_direct_only": float(full_accuracy - direct) if direct is not None else None,
                    "state_specificity_vs_shuffled": float(full_accuracy - shuffled) if shuffled is not None else None,
                    "state_temporal_order_vs_reversed": float(full_accuracy - reversed_state) if reversed_state is not None else None,
                    "future_alignment_margin": future_alignment,
                    "checkpoint_activity": float(full_activity),
                    "activity_kind": activity_kind,
                    "mean_absolute_gate": float(gate),
                    "effective_trainable_parameters": int(parameters),
                    "parameter_ratio_vs_target": float(parameters / target_parameters),
                    "test_examples_per_second": float(test_events.shape[0] / max(float(full_seconds), 1e-12)),
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
    arm,
    config,
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
    channels, _ = matched_temporal_tcn_channels(
        config.input_neurons,
        config.classes,
        target_parameters,
        input_kernel_size=input_kernel_size,
        hidden_kernel_size=hidden_kernel_size,
        temporal_levels=levels,
    )
    if arm.model_kind == "tcn":
        model = TemporalDilatedTCNClassifier(
            config,
            channels=channels,
            input_kernel_size=input_kernel_size,
            hidden_kernel_size=hidden_kernel_size,
            dilation=tcn_dilation,
            temporal_levels=levels,
        )
        activity_kind = "relu_activation"
    else:
        model = PredictiveStateTCNClassifier(
            config,
            channels=channels,
            input_kernel_size=input_kernel_size,
            hidden_kernel_size=hidden_kernel_size,
            dilation=tcn_dilation,
            temporal_levels=levels,
            dynamics=arm.dynamics,
            surrogate_slope=surrogate_slope,
            future_horizon=future_horizon,
            contrastive_temperature=contrastive_temperature,
        )
        activity_kind = "spike_rate" if arm.dynamics == "lif" else "analog_activation"
    return model.to(device), channels, activity_kind


def _train_predictive_validation_selected(
    model,
    arm,
    train_events,
    train_labels,
    validation_events,
    validation_labels,
    config,
    *,
    seed,
    device,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(seed + 70_000)
    best_accuracy = -1.0
    best_epoch = 0
    best_state = None
    sync(device)
    start = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        order = torch.randperm(train_events.shape[0], generator=generator)
        for offset in range(0, train_events.shape[0], config.batch_size):
            index = order[offset : offset + config.batch_size]
            batch_events = train_events.index_select(0, index).to(device)
            batch_labels = train_labels.index_select(0, index).to(device)
            optimizer.zero_grad(set_to_none=True)
            if hasattr(model, "training_terms") and float(arm.predictive_weight) > 0.0:
                logits, predictive_loss, _, _ = model.training_terms(
                    batch_events,
                    shuffled_future_targets=arm.shuffled_future_targets,
                )
                loss = torch.nn.functional.cross_entropy(logits, batch_labels)
                loss = loss + float(arm.predictive_weight) * predictive_loss
            else:
                loss = torch.nn.functional.cross_entropy(model(batch_events), batch_labels)
            loss.backward()
            optimizer.step()
            mark_step(device)
        validation_accuracy, _, _ = _measure(
            model, validation_events, validation_labels, config.batch_size, device
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = float(validation_accuracy)
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    sync(device)
    return {
        "best_state": best_state,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "train_seconds": time.perf_counter() - start,
    }


def _measure_future_alignment(model, events, batch_size, device):
    if not hasattr(model, "future_alignment"):
        return 0.0
    model.eval()
    weighted = 0.0
    total = 0
    with torch.no_grad():
        for offset in range(0, events.shape[0], batch_size):
            batch = events[offset : offset + batch_size].to(device)
            margin = model.future_alignment(batch)
            weighted += float(margin.item()) * int(batch.shape[0])
            total += int(batch.shape[0])
            mark_step(device)
    sync(device)
    return weighted / max(total, 1)


def _measure_sample_gate(model, events, batch_size, device):
    if not hasattr(model, "sample_gate_activity"):
        return 0.0
    model.eval()
    weighted = 0.0
    total = 0
    with torch.no_grad():
        for offset in range(0, events.shape[0], batch_size):
            batch = events[offset : offset + batch_size].to(device)
            gate = model.sample_gate_activity(batch)
            weighted += float(gate.item()) * int(batch.shape[0])
            total += int(batch.shape[0])
            mark_step(device)
    sync(device)
    return weighted / max(total, 1)


def _completed_result(
    progress,
    config,
    device,
    target_parameters,
    levels,
    screen_seed,
    seeds,
    accuracy_margin,
    causal_margin,
    alignment_margin,
    alignment_control_margin,
    minimum_spike_rate,
    maximum_spike_rate,
    minimum_gate,
):
    confirmation = list(progress.get("confirmation_records", []))
    summary = summarize_gen7_confirmation(confirmation)
    decision = progress.get("decision") or decide_gen7_predictive_state(
        summary,
        accuracy_margin=accuracy_margin,
        causal_margin=causal_margin,
        alignment_margin=alignment_margin,
        alignment_control_margin=alignment_control_margin,
        minimum_spike_rate=minimum_spike_rate,
        maximum_spike_rate=maximum_spike_rate,
        minimum_gate=minimum_gate,
    )
    return Gen7PredictiveStateResult(
        config=config,
        device=device_kind(device),
        target_parameters=int(target_parameters),
        temporal_levels=levels,
        screen_seed=int(screen_seed),
        confirm_seeds=seeds,
        screen_records=list(progress.get("screen_records", [])),
        promoted_arms=tuple(progress.get("promoted_arms", [])),
        confirmation_records=confirmation,
        confirmation_summary=summary,
        decision=decision,
    )


def _validate_run(
    config,
    levels,
    seeds,
    screen_epochs,
    confirm_epochs,
    samples,
    target_parameters,
    parameter_gate,
    spike_gate,
    accuracy_gates,
    alignment_gates,
    minimum_gate,
    future_horizon,
    contrastive_temperature,
):
    if not levels or any(level <= 0 for level in levels):
        raise ValueError("temporal_levels must contain positive integers")
    if not seeds or screen_epochs <= 0 or confirm_epochs <= 0:
        raise ValueError("seeds and epochs must be non-empty and positive")
    if min(samples) < 0 or target_parameters <= 0:
        raise ValueError("invalid samples or target parameter budget")
    if not 0.0 < parameter_gate[0] <= parameter_gate[1]:
        raise ValueError("invalid parameter gate")
    if not 0.0 <= spike_gate[0] <= spike_gate[1] <= 1.0:
        raise ValueError("invalid spike gate")
    if any(not 0.0 <= value <= 1.0 for value in accuracy_gates):
        raise ValueError("invalid accuracy or causal gate")
    if any(not 0.0 <= value <= 2.0 for value in alignment_gates):
        raise ValueError("invalid alignment gate")
    if not 0.0 <= minimum_gate <= 1.0:
        raise ValueError("invalid conditional gate threshold")
    if not 0 < future_horizon < config.timesteps:
        raise ValueError("future_horizon must be between 1 and timesteps - 1")
    if contrastive_temperature <= 0.0:
        raise ValueError("contrastive_temperature must be positive")


def _run_signature(config, **values) -> dict:
    signature = {
        "version": 1,
        "arms": list(available_gen7_predictive_state_arms()),
        "input_neurons": int(config.input_neurons),
        "classes": int(config.classes),
        "timesteps": int(config.timesteps),
        "duration_seconds": float(config.duration_seconds),
        "learning_rate": float(config.learning_rate),
        "weight_decay": float(config.weight_decay),
        "batch_size": int(config.batch_size),
        "data_root": str(config.data_root),
        "data_seed": int(config.data_seed),
    }
    for key, value in values.items():
        signature[key] = list(value) if isinstance(value, tuple) else value
    return signature


def _defined_values(rows, key):
    return [float(row[key]) for row in rows if row.get(key) is not None]


def _mean_or_zero(values):
    return statistics.fmean(values) if values else 0.0


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
