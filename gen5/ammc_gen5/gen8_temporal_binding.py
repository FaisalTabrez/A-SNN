"""Gen-8 time-local predictive binding experiment."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
import gc
import json
import pathlib
import statistics
from typing import Iterable

from .event_mnist import nn, torch
from .gen7_predictive_state import (
    PREDICTIVE_ABLATION_MODES,
    PredictiveStateTCNClassifier,
    _measure_future_alignment,
    _measure_sample_gate,
    _train_predictive_validation_selected,
    predictive_state_parameter_count,
)
from .milestone_a_architecture import (
    _load_progress,
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
)


@dataclass(frozen=True)
class Gen8TemporalBindingArm:
    name: str
    model_kind: str
    conventional: bool
    causal_state: bool
    dynamics: str | None
    predictive_weight: float
    shuffled_future_targets: bool = False


GEN8_TEMPORAL_BINDING_ARMS = (
    Gen8TemporalBindingArm("dilated_tcn", "tcn", True, False, None, 0.0),
    Gen8TemporalBindingArm(
        "lif_pooled_predictive", "pooled_predictive", False, True, "lif", 0.20
    ),
    Gen8TemporalBindingArm(
        "analog_time_local_binding", "time_local_binding", False, True, "analog", 0.20
    ),
    Gen8TemporalBindingArm(
        "lif_shuffled_time_local", "time_local_binding", False, True, "lif", 0.20, True
    ),
    Gen8TemporalBindingArm(
        "lif_time_local_binding", "time_local_binding", False, True, "lif", 0.20
    ),
)


def available_gen8_temporal_binding_arms() -> tuple[str, ...]:
    return tuple(arm.name for arm in GEN8_TEMPORAL_BINDING_ARMS)


def temporal_binding_parameter_count(
    input_neurons: int,
    channels: int,
    classes: int,
    *,
    input_kernel_size: int,
    hidden_kernel_size: int,
    temporal_levels: Iterable[int],
    spiking: bool,
) -> int:
    """Count the matched TCN, dynamics, local predictor, and binding projection."""

    return predictive_state_parameter_count(
        input_neurons,
        channels,
        classes,
        input_kernel_size=input_kernel_size,
        hidden_kernel_size=hidden_kernel_size,
        temporal_levels=temporal_levels,
        spiking=spiking,
    )


def matched_temporal_binding_channels(
    input_neurons: int,
    classes: int,
    target_parameters: int,
    *,
    input_kernel_size: int,
    hidden_kernel_size: int,
    temporal_levels: Iterable[int],
) -> tuple[int, int]:
    """Use the exact width of the parameter-matched conventional TCN."""

    channels, _ = matched_temporal_tcn_channels(
        input_neurons,
        classes,
        target_parameters,
        input_kernel_size=input_kernel_size,
        hidden_kernel_size=hidden_kernel_size,
        temporal_levels=temporal_levels,
    )
    actual = temporal_binding_parameter_count(
        input_neurons,
        channels,
        classes,
        input_kernel_size=input_kernel_size,
        hidden_kernel_size=hidden_kernel_size,
        temporal_levels=temporal_levels,
        spiking=True,
    )
    return channels, actual


class TimeLocalBindingTCNClassifier(PredictiveStateTCNClassifier):
    """Predict locally and bind direct/state traces before temporal pooling.

    Gen-7 pooled direct and state traces before computing its correction. This
    successor instead forms a class-valued correction at every aligned time
    step from ``direct[t] * state[t]``. Batch shuffling or time reversal thus
    destroys the interaction before aggregation rather than after it.
    """

    def _logits(self, representations):
        direct_trace = representations["direct_trace"]
        state_trace = representations["state_trace"]
        if self.ablation_mode == "shuffled_state":
            state_trace = torch.roll(state_trace, shifts=1, dims=0)
        elif self.ablation_mode == "reversed_state":
            state_trace = torch.flip(state_trace, dims=(1,))

        from .milestone_a_architecture import _multiscale_features

        direct_features = torch.cat(
            _multiscale_features(direct_trace, self.temporal_levels), dim=1
        )
        direct_logits = self.classifier(direct_features)
        binding_trace = direct_trace * state_trace
        correction = torch.tanh(self.conditional_gate(binding_trace)).mean(dim=1)
        if self.ablation_mode == "direct_only":
            return direct_logits
        if self.ablation_mode == "state_only":
            return correction + self.classifier.bias
        return direct_logits + correction

    def sample_gate_activity(self, events):
        representations = self._representations(events)
        binding_trace = (
            representations["direct_trace"] * representations["state_trace"]
        )
        return torch.tanh(self.conditional_gate(binding_trace)).abs().mean()

    def _predictive_objective(self, state_trace, currents, *, shuffled_targets: bool):
        """Contrast samples independently at every aligned future timestep."""

        horizon = self.future_horizon
        prediction = torch.nn.functional.normalize(
            self.future_projection(state_trace[:, :-horizon]), dim=2
        ).transpose(0, 1)
        target = torch.nn.functional.normalize(
            torch.relu(currents[:, horizon:]).detach(), dim=2
        ).transpose(0, 1)
        if shuffled_targets and target.shape[1] > 1:
            target = torch.roll(target, shifts=1, dims=1)

        paired_scores = (prediction * target).sum(dim=2)
        paired = paired_scores.mean()
        if target.shape[1] <= 1:
            return paired * 0.0, paired.detach() * 0.0

        negative_forward = (
            prediction * torch.roll(target, shifts=1, dims=1)
        ).sum(dim=2)
        negative_backward = (
            prediction * torch.roll(target, shifts=-1, dims=1)
        ).sum(dim=2)
        temperature = self.contrastive_temperature
        loss = 0.5 * (
            torch.nn.functional.softplus(
                (negative_forward - paired_scores) / temperature
            ).mean()
            + torch.nn.functional.softplus(
                (negative_backward - paired_scores) / temperature
            ).mean()
        )
        return loss, paired - negative_forward.mean()


@dataclass
class Gen8TemporalBindingResult:
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
        json_path = output / "gen8_temporal_binding.json"
        screen_path = output / "gen8_temporal_binding_screen.csv"
        records_path = output / "gen8_temporal_binding_confirmation_records.csv"
        summary_path = output / "gen8_temporal_binding_confirmation_summary.csv"
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
            plot_path = output / "gen8_temporal_binding.png"
            plot_gen8_temporal_binding(self.confirmation_summary, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def run_gen8_temporal_binding(
    config: SHDConfig,
    *,
    screen_seed: int = 145,
    confirm_seeds: Iterable[int] = (145, 146, 147),
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
    binding_gain_margin: float = 0.005,
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
) -> Gen8TemporalBindingResult:
    if torch is None:
        raise ImportError("Gen-8 temporal binding requires PyTorch")
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
        (
            promotion_margin,
            accuracy_margin,
            causal_margin,
            alignment_margin,
            alignment_control_margin,
            binding_gain_margin,
        ),
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
        binding_gain_margin=binding_gain_margin,
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
            binding_gain_margin,
            minimum_spike_rate,
            maximum_spike_rate,
            minimum_gate,
        )

    train_events, train_labels, validation_events, validation_labels, test_events, test_labels = load_ssc_tensors(
        full_config, validation_samples=0
    )
    screen_records = list(progress.get("screen_records", []))
    confirmation_records = list(progress.get("confirmation_records", []))
    expected = {(int(screen_seed), arm.name) for arm in GEN8_TEMPORAL_BINDING_ARMS}
    completed = {(int(row["seed"]), str(row["arm"])) for row in screen_records}
    if not expected.issubset(completed):
        generator = torch.Generator(device="cpu").manual_seed(config.data_seed + 98_000)
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
            GEN8_TEMPORAL_BINDING_ARMS,
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
                confirmation_records=confirmation_records,
            ),
        )
        del screen_train_events, screen_train_labels
        del screen_validation_events, screen_validation_labels
        del screen_test_events, screen_test_labels
        gc.collect()

    promoted = select_gen8_promoted_arms(
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
        confirmation_records=confirmation_records,
    )
    lookup = {arm.name: arm for arm in GEN8_TEMPORAL_BINDING_ARMS}
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
        existing_records=confirmation_records,
        progress_callback=lambda rows: _save_progress(
            progress_path,
            signature,
            stage="confirmation",
            screen_records=screen_records,
            promoted_arms=promoted,
            confirmation_records=rows,
        ),
    )
    summary = summarize_gen8_confirmation(confirmation_records)
    decision = decide_gen8_temporal_binding(
        summary,
        accuracy_margin=accuracy_margin,
        causal_margin=causal_margin,
        alignment_margin=alignment_margin,
        alignment_control_margin=alignment_control_margin,
        binding_gain_margin=binding_gain_margin,
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
    return Gen8TemporalBindingResult(
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


def select_gen8_promoted_arms(
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
    for arm in GEN8_TEMPORAL_BINDING_ARMS[1:]:
        row = next(row for row in rows if row["arm"] == arm.name)
        ratio = float(row["parameter_ratio_vs_target"])
        activity_ok = arm.dynamics != "lif" or (
            minimum_spike_rate
            <= float(row["checkpoint_activity"])
            <= maximum_spike_rate
        )
        if (
            float(row["best_validation_accuracy"]) >= threshold
            and minimum_parameter_ratio <= ratio <= maximum_parameter_ratio
            and activity_ok
        ):
            promoted.append(arm.name)
    if "lif_time_local_binding" in promoted:
        for control in (
            "lif_pooled_predictive",
            "analog_time_local_binding",
            "lif_shuffled_time_local",
        ):
            if control not in promoted:
                promoted.append(control)
    order = available_gen8_temporal_binding_arms()
    return tuple(name for name in order if name in promoted)


def summarize_gen8_confirmation(records: Iterable[dict]) -> list[dict]:
    rows = list(records)
    if not rows:
        return []
    baseline = statistics.fmean(
        float(row["full_accuracy"]) for row in rows if row["arm"] == "dilated_tcn"
    )
    summary = []
    for arm in available_gen8_temporal_binding_arms():
        group = [row for row in rows if row["arm"] == arm]
        if not group:
            continue
        accuracy = statistics.fmean(float(row["full_accuracy"]) for row in group)
        contribution = _values(group, "state_contribution_vs_direct_only")
        specificity = _values(group, "state_specificity_vs_shuffled")
        temporal_order = _values(group, "state_temporal_order_vs_reversed")
        alignment = _values(group, "future_alignment_margin")
        summary.append(
            {
                "arm": arm,
                "model_kind": group[0]["model_kind"],
                "conventional": bool(group[0]["conventional"]),
                "causal_state": bool(group[0]["causal_state"]),
                "runs": len(group),
                "mean_full_accuracy": accuracy,
                "std_full_accuracy": statistics.pstdev(float(row["full_accuracy"]) for row in group),
                "mean_gain_vs_tcn": accuracy - baseline,
                "mean_state_contribution_vs_direct_only": _mean(contribution),
                "half_point_seed_count_state_contribution": sum(value >= 0.005 for value in contribution),
                "mean_state_specificity_vs_shuffled": _mean(specificity),
                "half_point_seed_count_state_specificity": sum(value >= 0.005 for value in specificity),
                "mean_state_temporal_order_vs_reversed": _mean(temporal_order),
                "half_point_seed_count_temporal_order": sum(value >= 0.005 for value in temporal_order),
                "mean_future_alignment_margin": _mean(alignment),
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


def decide_gen8_temporal_binding(
    summary: Iterable[dict],
    *,
    accuracy_margin: float,
    causal_margin: float,
    alignment_margin: float,
    alignment_control_margin: float,
    binding_gain_margin: float,
    minimum_spike_rate: float,
    maximum_spike_rate: float,
    minimum_gate: float,
) -> dict:
    rows = list(summary)
    if not rows:
        return {"status": "no_confirmation", "qualified_arms": []}
    lookup = {str(row["arm"]): row for row in rows}
    candidate = lookup.get("lif_time_local_binding")
    shuffled = lookup.get("lif_shuffled_time_local")
    pooled = lookup.get("lif_pooled_predictive")
    required = 2 if max(int(row["runs"]) for row in rows) >= 3 else 1
    alignment_control_gain = None
    specificity_gain = None
    temporal_order_gain = None
    qualified = []
    if candidate is not None and shuffled is not None and pooled is not None:
        alignment_control_gain = float(candidate["mean_future_alignment_margin"]) - float(
            shuffled["mean_future_alignment_margin"]
        )
        specificity_gain = float(candidate["mean_state_specificity_vs_shuffled"]) - float(
            pooled["mean_state_specificity_vs_shuffled"]
        )
        temporal_order_gain = float(candidate["mean_state_temporal_order_vs_reversed"]) - float(
            pooled["mean_state_temporal_order_vs_reversed"]
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
            and specificity_gain >= binding_gain_margin
            and temporal_order_gain >= binding_gain_margin
            and minimum_spike_rate <= float(candidate["mean_activity"]) <= maximum_spike_rate
            and float(candidate["mean_absolute_gate"]) >= minimum_gate
        ):
            qualified.append("lif_time_local_binding")
    return {
        "status": "pass" if qualified else "stop",
        "best_arm": str(rows[0]["arm"]),
        "best_accuracy": float(rows[0]["mean_full_accuracy"]),
        "alignment_gain_vs_shuffled_training": alignment_control_gain,
        "state_specificity_gain_vs_pooled": specificity_gain,
        "temporal_order_gain_vs_pooled": temporal_order_gain,
        "qualified_arms": qualified,
        "next_milestone": (
            "runtime_efficiency_preregistration"
            if qualified
            else "close_gen8_temporal_binding"
        ),
    }


def plot_gen8_temporal_binding(summary: list[dict], path: str | pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    labels = [row["arm"].replace("_", "\n") for row in summary]
    figure, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    measures = (
        ("mean_full_accuracy", "SSC test accuracy (%)", 100.0),
        ("mean_state_specificity_vs_shuffled", "Full - shuffled state (points)", 100.0),
        ("mean_state_temporal_order_vs_reversed", "Full - reversed state (points)", 100.0),
        ("mean_future_alignment_margin", "Local future-alignment margin", 1.0),
    )
    colors = ("#35b4f2", "#167d55", "#8b5cf6", "#ffb31a")
    for axis, (key, ylabel, scale), color in zip(axes.flat, measures, colors):
        axis.bar(labels, [scale * float(row[key]) for row in summary], color=color)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].set_title("Gen-8 time-local predictive binding")
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
                measurements[mode] = _measure(
                    model, test_events, test_labels, config.batch_size, device
                )
            if hasattr(model, "set_ablation_mode"):
                model.set_ablation_mode("full")
            future_alignment = _measure_future_alignment(
                model, test_events, config.batch_size, device
            )
            full_accuracy, full_seconds, full_activity = measurements["full"]
            direct = measurements.get("direct_only", (None, None, None))[0]
            state = measurements.get("state_only", (None, None, None))[0]
            shuffled_state = measurements.get("shuffled_state", (None, None, None))[0]
            reversed_state = measurements.get("reversed_state", (None, None, None))[0]
            parameters = sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            )
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
                    "shuffled_state_accuracy": shuffled_state,
                    "reversed_state_accuracy": reversed_state,
                    "state_contribution_vs_direct_only": float(full_accuracy - direct) if direct is not None else None,
                    "state_specificity_vs_shuffled": float(full_accuracy - shuffled_state) if shuffled_state is not None else None,
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
        model_class = (
            PredictiveStateTCNClassifier
            if arm.model_kind == "pooled_predictive"
            else TimeLocalBindingTCNClassifier
        )
        model = model_class(
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
    binding_gain_margin,
    minimum_spike_rate,
    maximum_spike_rate,
    minimum_gate,
):
    confirmation = list(progress.get("confirmation_records", []))
    summary = summarize_gen8_confirmation(confirmation)
    decision = progress.get("decision") or decide_gen8_temporal_binding(
        summary,
        accuracy_margin=accuracy_margin,
        causal_margin=causal_margin,
        alignment_margin=alignment_margin,
        alignment_control_margin=alignment_control_margin,
        binding_gain_margin=binding_gain_margin,
        minimum_spike_rate=minimum_spike_rate,
        maximum_spike_rate=maximum_spike_rate,
        minimum_gate=minimum_gate,
    )
    return Gen8TemporalBindingResult(
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
    gates,
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
    if any(not 0.0 <= value <= 2.0 for value in gates):
        raise ValueError("invalid promotion, causal, alignment, or binding gate")
    if not 0.0 <= minimum_gate <= 1.0:
        raise ValueError("invalid binding activity gate")
    if not 0 < future_horizon < config.timesteps:
        raise ValueError("future_horizon must be between 1 and timesteps - 1")
    if contrastive_temperature <= 0.0:
        raise ValueError("contrastive_temperature must be positive")


def _run_signature(config, **values) -> dict:
    signature = {
        "version": 1,
        "arms": list(available_gen8_temporal_binding_arms()),
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


def _values(rows, key):
    return [float(row[key]) for row in rows if row.get(key) is not None]


def _mean(values):
    return statistics.fmean(values) if values else 0.0


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
