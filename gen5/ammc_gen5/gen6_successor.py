"""Gen-6 weight-shared residual-state successor experiment."""

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
from .milestone_a_architecture import (
    _load_progress,
    _multiscale_features,
    _sample_split,
    _save_progress,
)
from .runtime import device_kind, resolve_device, seed_everything
from .shd_benchmark import SHDConfig, _measure
from .shd_residual_state_contribution import RESIDUAL_ABLATION_MODES
from .shd_temporal_pyramid import DEFAULT_TEMPORAL_LEVELS
from .shd_validation_checkpoint import _train_validation_selected
from .ssc_benchmark import load_ssc_tensors
from .ssc_efficiency_baselines import (
    TemporalDilatedTCNClassifier,
    matched_temporal_tcn_channels,
    temporal_tcn_parameter_count,
)
from .trainable_temporal_mnist import SurrogateSpike


@dataclass(frozen=True)
class Gen6SuccessorArm:
    name: str
    model_kind: str
    conventional: bool
    causal_state: bool


GEN6_SUCCESSOR_ARMS = (
    Gen6SuccessorArm("dilated_tcn", "tcn", True, False),
    Gen6SuccessorArm(
        "shared_residual_analog", "shared_residual_analog", False, True
    ),
    Gen6SuccessorArm("shared_residual_lif", "shared_residual_lif", False, True),
)


def available_gen6_successor_arms() -> tuple[str, ...]:
    return tuple(arm.name for arm in GEN6_SUCCESSOR_ARMS)


def shared_residual_parameter_count(
    input_neurons: int,
    channels: int,
    classes: int,
    *,
    input_kernel_size: int,
    hidden_kernel_size: int,
    temporal_levels: Iterable[int],
    spiking: bool,
) -> int:
    return int(
        temporal_tcn_parameter_count(
            input_neurons,
            channels,
            classes,
            input_kernel_size=input_kernel_size,
            hidden_kernel_size=hidden_kernel_size,
            temporal_levels=temporal_levels,
        )
        + channels
        + (channels if spiking else 0)
        + classes
    )


def matched_shared_residual_channels(
    input_neurons: int,
    classes: int,
    target_parameters: int,
    *,
    input_kernel_size: int,
    hidden_kernel_size: int,
    temporal_levels: Iterable[int],
) -> tuple[int, int]:
    channels = 1
    while shared_residual_parameter_count(
        input_neurons,
        channels + 1,
        classes,
        input_kernel_size=input_kernel_size,
        hidden_kernel_size=hidden_kernel_size,
        temporal_levels=temporal_levels,
        spiking=True,
    ) <= target_parameters:
        channels += 1
    return channels, shared_residual_parameter_count(
        input_neurons,
        channels,
        classes,
        input_kernel_size=input_kernel_size,
        hidden_kernel_size=hidden_kernel_size,
        temporal_levels=temporal_levels,
        spiking=True,
    )


class SharedResidualStateTCNClassifier(nn.Module):
    """TCN direct predictor plus zero-initialized weight-shared state logits."""

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
            raise ImportError("Gen-6 successor requires PyTorch")
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
        self.classifier = nn.Linear(
            channels * sum(self.temporal_levels), config.classes
        )
        leak_logit = math.log(initial_leak / (1.0 - initial_leak))
        self.leak_logit = nn.Parameter(torch.full((channels,), leak_logit))
        if dynamics == "lif":
            threshold_raw = math.log(math.expm1(initial_threshold))
            self.threshold_raw = nn.Parameter(torch.full((channels,), threshold_raw))
        else:
            self.register_parameter("threshold_raw", None)
        self.correction_gate = nn.Parameter(torch.zeros(config.classes))

    def forward(self, events, *, return_event_rate: bool = False):  # type: ignore[override]
        if events.ndim != 3 or events.shape[2] != self.config.input_neurons:
            raise ValueError("events must have shape [batch, time, input_neurons]")
        first = torch.relu(
            self.input_conv(events.to(torch.float32).transpose(1, 2))
        )
        currents = (self.hidden_conv(first) + first).transpose(1, 2)
        direct_features = torch.cat(
            _multiscale_features(torch.relu(currents), self.temporal_levels), dim=1
        )
        state_trace, activity = self._state_trace(currents)
        if self.ablation_mode == "shuffled_state":
            state_trace = torch.roll(state_trace, shifts=1, dims=0)
        state_features = torch.cat(
            _multiscale_features(state_trace, self.temporal_levels), dim=1
        )
        direct_logits = self.classifier(direct_features)
        state_logits = torch.nn.functional.linear(
            state_features, self.classifier.weight, bias=None
        )
        correction = torch.tanh(self.correction_gate) * state_logits
        if self.ablation_mode == "direct_only":
            logits = direct_logits
        elif self.ablation_mode == "state_only":
            logits = correction + self.classifier.bias
        else:
            logits = direct_logits + correction
        if return_event_rate:
            return logits, activity
        return logits

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

    def set_ablation_mode(self, mode: str) -> None:
        if mode not in RESIDUAL_ABLATION_MODES:
            raise ValueError("unsupported shared-residual ablation mode")
        self.ablation_mode = mode

    def mean_absolute_gate(self) -> float:
        return float(torch.tanh(self.correction_gate).abs().mean().item())


@dataclass
class Gen6SuccessorResult:
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
        json_path = output / "gen6_successor.json"
        screen_path = output / "gen6_successor_screen.csv"
        records_path = output / "gen6_successor_confirmation_records.csv"
        summary_path = output / "gen6_successor_confirmation_summary.csv"
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
            plot_path = output / "gen6_successor.png"
            plot_gen6_successor(self.confirmation_summary, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def run_gen6_successor(
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
    minimum_gate: float = 0.01,
    target_parameters: int = 133_631,
    device="auto",
    temporal_levels: Iterable[int] = DEFAULT_TEMPORAL_LEVELS,
    input_kernel_size: int = 5,
    hidden_kernel_size: int = 3,
    tcn_dilation: int = 2,
    surrogate_slope: float = 10.0,
    progress_path: str | pathlib.Path | None = None,
) -> Gen6SuccessorResult:
    if torch is None:
        raise ImportError("Gen-6 successor requires PyTorch")
    levels = tuple(int(level) for level in temporal_levels)
    seeds = tuple(int(seed) for seed in confirm_seeds)
    _validate_run(
        levels,
        seeds,
        screen_epochs,
        confirm_epochs,
        screen_train_samples,
        screen_validation_samples,
        screen_test_samples,
        target_parameters,
        promotion_margin,
        minimum_parameter_ratio,
        maximum_parameter_ratio,
        minimum_spike_rate,
        maximum_spike_rate,
        accuracy_margin,
        causal_margin,
        minimum_gate,
    )
    signature = _run_signature(
        config,
        screen_seed=screen_seed,
        confirm_seeds=seeds,
        screen_samples=(
            screen_train_samples,
            screen_validation_samples,
            screen_test_samples,
        ),
        epochs=(screen_epochs, confirm_epochs),
        target_parameters=target_parameters,
        promotion_margin=promotion_margin,
        parameter_gate=(minimum_parameter_ratio, maximum_parameter_ratio),
        spike_gate=(minimum_spike_rate, maximum_spike_rate),
        accuracy_margin=accuracy_margin,
        causal_margin=causal_margin,
        minimum_gate=minimum_gate,
        levels=levels,
        input_kernel_size=input_kernel_size,
        hidden_kernel_size=hidden_kernel_size,
        tcn_dilation=tcn_dilation,
        surrogate_slope=surrogate_slope,
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
            minimum_spike_rate,
            maximum_spike_rate,
            minimum_gate,
        )
    data = load_ssc_tensors(full_config, validation_samples=0)
    train_events, train_labels, validation_events, validation_labels, test_events, test_labels = data
    expected_screen = {(int(screen_seed), arm.name) for arm in GEN6_SUCCESSOR_ARMS}
    completed_screen = {
        (int(row["seed"]), str(row["arm"])) for row in existing_screen
    }
    screen_records = existing_screen
    if not expected_screen.issubset(completed_screen):
        generator = torch.Generator(device="cpu").manual_seed(config.data_seed + 96_000)
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
            GEN6_SUCCESSOR_ARMS,
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
    promoted = select_gen6_promoted_arms(
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
    lookup = {arm.name: arm for arm in GEN6_SUCCESSOR_ARMS}
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
    summary = summarize_gen6_confirmation(confirmation_records)
    decision = decide_gen6_successor(
        summary,
        accuracy_margin=accuracy_margin,
        causal_margin=causal_margin,
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
    return Gen6SuccessorResult(
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


def select_gen6_promoted_arms(
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
    for arm in GEN6_SUCCESSOR_ARMS[1:]:
        row = next(row for row in rows if row["arm"] == arm.name)
        ratio = float(row["parameter_ratio_vs_target"])
        activity_ok = True
        if "lif" in arm.model_kind:
            activity = float(row["checkpoint_activity"])
            activity_ok = minimum_spike_rate <= activity <= maximum_spike_rate
        if (
            float(row["best_validation_accuracy"]) >= threshold
            and minimum_parameter_ratio <= ratio <= maximum_parameter_ratio
            and activity_ok
        ):
            promoted.append(arm.name)
    return tuple(promoted)


def summarize_gen6_confirmation(records: Iterable[dict]) -> list[dict]:
    rows = list(records)
    if not rows:
        return []
    baseline_rows = [row for row in rows if row["arm"] == "dilated_tcn"]
    baseline_mean = statistics.fmean(float(row["full_accuracy"]) for row in baseline_rows)
    summary = []
    for arm in available_gen6_successor_arms():
        group = [row for row in rows if row["arm"] == arm]
        if not group:
            continue
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
        accuracy = statistics.fmean(float(row["full_accuracy"]) for row in group)
        summary.append(
            {
                "arm": arm,
                "model_kind": group[0]["model_kind"],
                "conventional": bool(group[0]["conventional"]),
                "causal_state": bool(group[0]["causal_state"]),
                "runs": len(group),
                "mean_full_accuracy": accuracy,
                "std_full_accuracy": statistics.pstdev(
                    float(row["full_accuracy"]) for row in group
                ),
                "mean_gain_vs_tcn": accuracy - baseline_mean,
                "mean_state_contribution_vs_direct_only": (
                    statistics.fmean(contributions) if contributions else 0.0
                ),
                "half_point_seed_count_state_contribution": sum(
                    value >= 0.005 for value in contributions
                ),
                "mean_state_specificity_vs_shuffled": (
                    statistics.fmean(specificities) if specificities else 0.0
                ),
                "half_point_seed_count_state_specificity": sum(
                    value >= 0.005 for value in specificities
                ),
                "mean_activity": statistics.fmean(
                    float(row["checkpoint_activity"]) for row in group
                ),
                "activity_kind": group[0]["activity_kind"],
                "mean_absolute_gate": statistics.fmean(
                    float(row["mean_absolute_gate"]) for row in group
                ),
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
        summary, key=lambda row: (-float(row["mean_full_accuracy"]), str(row["arm"]))
    )


def decide_gen6_successor(
    summary: Iterable[dict],
    *,
    accuracy_margin: float,
    causal_margin: float,
    minimum_spike_rate: float,
    maximum_spike_rate: float,
    minimum_gate: float,
) -> dict:
    rows = list(summary)
    if not rows:
        return {"status": "no_confirmation", "qualified_arms": []}
    required = 2 if max(int(row["runs"]) for row in rows) >= 3 else 1
    qualified = []
    for row in rows:
        if row["arm"] != "shared_residual_lif":
            continue
        if (
            float(row["mean_gain_vs_tcn"]) >= -accuracy_margin
            and float(row["mean_state_contribution_vs_direct_only"]) >= causal_margin
            and int(row["half_point_seed_count_state_contribution"]) >= required
            and float(row["mean_state_specificity_vs_shuffled"]) >= causal_margin
            and int(row["half_point_seed_count_state_specificity"]) >= required
            and minimum_spike_rate
            <= float(row["mean_activity"])
            <= maximum_spike_rate
            and float(row["mean_absolute_gate"]) >= minimum_gate
        ):
            qualified.append(str(row["arm"]))
    return {
        "status": "pass" if qualified else "stop",
        "best_arm": str(rows[0]["arm"]),
        "best_accuracy": float(rows[0]["mean_full_accuracy"]),
        "qualified_arms": qualified,
        "next_milestone": (
            "hardware_efficiency" if qualified else "close_gen6_successor"
        ),
    }


def plot_gen6_successor(summary: list[dict], path: str | pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    labels = [row["arm"].replace("_", "\n") for row in summary]
    figure, axes = plt.subplots(2, 1, figsize=(13, 10), constrained_layout=True)
    axes[0].bar(
        labels,
        [100.0 * float(row["mean_full_accuracy"]) for row in summary],
        yerr=[100.0 * float(row["std_full_accuracy"]) for row in summary],
        capsize=5,
        color="#35b4f2",
    )
    axes[0].set_ylabel("SSC test accuracy (%)")
    axes[0].set_title("Gen-6 weight-shared residual successor")
    axes[1].bar(
        labels,
        [100.0 * float(row["mean_state_specificity_vs_shuffled"]) for row in summary],
        color="#167d55",
    )
    axes[1].axhline(0.5, color="#bd3d3a", linestyle="--", label="+0.5 point gate")
    axes[1].set_ylabel("Full - shuffled state (points)")
    axes[1].legend()
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _run_stage(
    arms: Iterable[Gen6SuccessorArm],
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
                measurements[mode] = _measure(
                    model, test_events, test_labels, config.batch_size, device
                )
            full_accuracy, full_seconds, full_activity = measurements["full"]
            direct = measurements.get("direct_only", (None, None, None))[0]
            state = measurements.get("state_only", (None, None, None))[0]
            shuffled = measurements.get("shuffled_state", (None, None, None))[0]
            parameters = sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            )
            gate = model.mean_absolute_gate() if hasattr(model, "mean_absolute_gate") else 0.0
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
                    "direct_only_accuracy": direct,
                    "state_only_accuracy": state,
                    "shuffled_state_accuracy": shuffled,
                    "state_contribution_vs_direct_only": (
                        float(full_accuracy - direct) if direct is not None else None
                    ),
                    "state_specificity_vs_shuffled": (
                        float(full_accuracy - shuffled) if shuffled is not None else None
                    ),
                    "checkpoint_activity": float(full_activity),
                    "activity_kind": activity_kind,
                    "mean_absolute_gate": float(gate),
                    "effective_trainable_parameters": int(parameters),
                    "parameter_ratio_vs_target": float(parameters / target_parameters),
                    "test_examples_per_second": float(
                        test_events.shape[0] / max(float(full_seconds), 1e-12)
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
    arm: Gen6SuccessorArm,
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
    if arm.model_kind == "tcn":
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
    else:
        channels, _ = matched_shared_residual_channels(
            config.input_neurons,
            config.classes,
            target_parameters,
            input_kernel_size=input_kernel_size,
            hidden_kernel_size=hidden_kernel_size,
            temporal_levels=levels,
        )
        dynamics = "lif" if "lif" in arm.model_kind else "analog"
        model = SharedResidualStateTCNClassifier(
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
    minimum_spike_rate,
    maximum_spike_rate,
    minimum_gate,
):
    confirmation = list(progress.get("confirmation_records", []))
    summary = summarize_gen6_confirmation(confirmation)
    decision = progress.get("decision") or decide_gen6_successor(
        summary,
        accuracy_margin=accuracy_margin,
        causal_margin=causal_margin,
        minimum_spike_rate=minimum_spike_rate,
        maximum_spike_rate=maximum_spike_rate,
        minimum_gate=minimum_gate,
    )
    return Gen6SuccessorResult(
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
    levels,
    seeds,
    screen_epochs,
    confirm_epochs,
    screen_train_samples,
    screen_validation_samples,
    screen_test_samples,
    target_parameters,
    promotion_margin,
    minimum_parameter_ratio,
    maximum_parameter_ratio,
    minimum_spike_rate,
    maximum_spike_rate,
    accuracy_margin,
    causal_margin,
    minimum_gate,
):
    if not levels or any(level <= 0 for level in levels):
        raise ValueError("temporal_levels must contain positive integers")
    if not seeds:
        raise ValueError("at least one confirmation seed is required")
    if screen_epochs <= 0 or confirm_epochs <= 0:
        raise ValueError("epochs must be positive")
    if min(screen_train_samples, screen_validation_samples, screen_test_samples) < 0:
        raise ValueError("screen sample limits cannot be negative")
    if target_parameters <= 0:
        raise ValueError("target_parameters must be positive")
    if not 0.0 <= promotion_margin <= 1.0 or not 0.0 <= accuracy_margin <= 1.0:
        raise ValueError("invalid accuracy gate")
    if not 0.0 < minimum_parameter_ratio <= maximum_parameter_ratio:
        raise ValueError("invalid parameter gate")
    if not 0.0 <= minimum_spike_rate <= maximum_spike_rate <= 1.0:
        raise ValueError("invalid spike gate")
    if not 0.0 <= causal_margin <= 1.0 or not 0.0 <= minimum_gate <= 1.0:
        raise ValueError("invalid causal or gate threshold")


def _run_signature(config, **values) -> dict:
    signature = {
        "version": 1,
        "arms": list(available_gen6_successor_arms()),
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


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
