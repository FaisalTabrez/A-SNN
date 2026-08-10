"""Gen-11 frozen robust sensory backbone with bounded plastic state adapters."""

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

from .event_mnist import nn, torch
from .gen10_robust_representation import _random_sensor_dropout
from .gen9_continual_adaptation import apply_sensor_damage, sensor_damage_indices
from .milestone_a_architecture import _load_progress, _multiscale_features
from .runtime import device_kind, mark_step, resolve_device, seed_everything, sync
from .shd_benchmark import SHDConfig, _measure
from .shd_temporal_pyramid import DEFAULT_TEMPORAL_LEVELS
from .ssc_benchmark import load_ssc_tensors
from .ssc_efficiency_baselines import (
    TemporalDilatedTCNClassifier,
    matched_temporal_tcn_channels,
)
from .trainable_temporal_mnist import SurrogateSpike


GEN11_ADAPTATION_STRATEGIES = (
    "dropout_tcn_static",
    "dropout_tcn_readout",
    "dropout_tcn_full_finetune",
    "analog_state_adapter",
    "lif_state_adapter",
)


def available_gen11_adaptation_strategies() -> tuple[str, ...]:
    return GEN11_ADAPTATION_STRATEGIES


class PlasticStateAdapter(nn.Module):
    """Frozen TCN predictor plus a bounded analog or LIF correction pathway."""

    def __init__(
        self,
        backbone: TemporalDilatedTCNClassifier,
        *,
        dynamics: str,
        surrogate_slope: float = 10.0,
    ) -> None:
        if torch is None:
            raise ImportError("Gen-11 plastic adapter requires PyTorch")
        if dynamics not in {"analog", "lif"}:
            raise ValueError("dynamics must be analog or lif")
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config
        self.channels = backbone.channels
        self.temporal_levels = backbone.temporal_levels
        self.dynamics = dynamics
        self.surrogate_slope = float(surrogate_slope)
        self.ablation_mode = "full"
        initial_leaks = (0.50, 0.75, 0.90, 0.97)
        leak_values = [initial_leaks[index % len(initial_leaks)] for index in range(self.channels)]
        self.leak_logit = nn.Parameter(
            torch.tensor([math.log(value / (1.0 - value)) for value in leak_values], dtype=torch.float32)
        )
        if dynamics == "lif":
            self.threshold_raw = nn.Parameter(
                torch.full((self.channels,), math.log(math.expm1(1.0)), dtype=torch.float32)
            )
        else:
            self.register_parameter("threshold_raw", None)
        feature_dim = self.channels * sum(self.temporal_levels)
        self.adapter_classifier = nn.Linear(feature_dim, self.config.classes, bias=False)
        self.correction_gate = nn.Parameter(torch.zeros(self.config.classes))

    def forward(self, events, *, return_event_rate: bool = False):  # type: ignore[override]
        trace = self._backbone_trace(events)
        direct_features = torch.cat(_multiscale_features(trace, self.temporal_levels), dim=1)
        direct_logits = self.backbone.classifier(direct_features)
        state_trace, activity = self._state_trace(trace)
        if self.ablation_mode == "shuffled_state":
            state_trace = torch.roll(state_trace, shifts=1, dims=0)
        state_features = torch.cat(_multiscale_features(state_trace, self.temporal_levels), dim=1)
        correction = torch.tanh(self.correction_gate) * self.adapter_classifier(state_features)
        logits = direct_logits if self.ablation_mode == "direct_only" else direct_logits + correction
        if return_event_rate:
            return logits, activity
        return logits

    def _backbone_trace(self, events):
        direct = torch.relu(
            self.backbone.input_conv(events.to(torch.float32).transpose(1, 2))
        )
        return torch.relu(self.backbone.hidden_conv(direct) + direct).transpose(1, 2)

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
                state = SurrogateSpike.apply(pre_reset - threshold, self.surrogate_slope)
                membrane = pre_reset - state * threshold
                activity_sum = activity_sum + state.mean()
            else:
                membrane = pre_reset
                state = torch.tanh(membrane)
                activity_sum = activity_sum + state.abs().mean()
            states.append(state)
        return torch.stack(states, dim=1), activity_sum / int(currents.shape[1])

    def set_ablation_mode(self, mode: str) -> None:
        if mode not in {"full", "direct_only", "shuffled_state"}:
            raise ValueError("unsupported Gen-11 ablation mode")
        self.ablation_mode = mode

    def mean_absolute_gate(self) -> float:
        return float(torch.tanh(self.correction_gate).abs().mean().item())


@dataclass
class Gen11PlasticAdapterResult:
    config: SHDConfig
    device: str
    target_parameters: int
    temporal_levels: tuple[int, ...]
    seeds: tuple[int, ...]
    source_mask_fraction: float
    damage_fraction: float
    damage_seed: int
    adaptation_budgets: tuple[int, ...]
    records: list[dict]
    summary: list[dict]
    decision: dict

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "gen11_plastic_adapter.json"
        records_path = output / "gen11_plastic_adapter_records.csv"
        summary_path = output / "gen11_plastic_adapter_summary.csv"
        payload = {
            "config": asdict(self.config), "device": self.device,
            "target_parameters": self.target_parameters,
            "temporal_levels": list(self.temporal_levels), "seeds": list(self.seeds),
            "source_mask_fraction": self.source_mask_fraction,
            "damage_fraction": self.damage_fraction, "damage_seed": self.damage_seed,
            "adaptation_budgets": list(self.adaptation_budgets),
            "records": self.records, "summary": self.summary, "decision": self.decision,
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        _write_csv(records_path, self.records)
        _write_csv(summary_path, self.summary)
        paths = {"json": str(json_path), "records_csv": str(records_path), "summary_csv": str(summary_path)}
        if plot and self.summary:
            plot_path = output / "gen11_plastic_adapter.png"
            plot_gen11_plastic_adapter(self.records, self.summary, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def run_gen11_plastic_adapter(
    config: SHDConfig,
    *,
    seeds: Iterable[int] = (154, 155, 156),
    source_epochs: int = 15,
    source_mask_fraction: float = 0.20,
    damage_fraction: float = 0.35,
    damage_seed: int = 909,
    adaptation_budgets: Iterable[int] = (0, 64, 256, 1024, 4096),
    adaptation_epochs_per_block: int = 3,
    adaptation_learning_rate: float = 0.001,
    minimum_shift_drop: float = 0.02,
    minimum_adaptation_gain: float = 0.02,
    auc_margin: float = 0.01,
    final_accuracy_margin: float = 0.01,
    forgetting_margin: float = 0.005,
    causal_margin: float = 0.005,
    minimum_spike_rate: float = 0.01,
    maximum_spike_rate: float = 0.30,
    target_parameters: int = 133_631,
    device="auto",
    temporal_levels: Iterable[int] = DEFAULT_TEMPORAL_LEVELS,
    input_kernel_size: int = 5,
    hidden_kernel_size: int = 3,
    tcn_dilation: int = 2,
    surrogate_slope: float = 10.0,
    progress_path: str | pathlib.Path | None = None,
) -> Gen11PlasticAdapterResult:
    if torch is None:
        raise ImportError("Gen-11 plastic adapter requires PyTorch")
    seed_values = tuple(int(seed) for seed in seeds)
    budgets = tuple(int(value) for value in adaptation_budgets)
    levels = tuple(int(level) for level in temporal_levels)
    _validate_run(config, seed_values, budgets, levels, source_epochs,
                  source_mask_fraction, damage_fraction, adaptation_epochs_per_block,
                  adaptation_learning_rate, (minimum_shift_drop, minimum_adaptation_gain,
                  auc_margin, final_accuracy_margin, forgetting_margin, causal_margin),
                  (minimum_spike_rate, maximum_spike_rate))
    signature = _run_signature(
        config, seeds=seed_values, source_epochs=source_epochs,
        source_mask_fraction=source_mask_fraction, damage_fraction=damage_fraction,
        damage_seed=damage_seed, budgets=budgets,
        adaptation_epochs_per_block=adaptation_epochs_per_block,
        adaptation_learning_rate=adaptation_learning_rate,
        gates=(minimum_shift_drop, minimum_adaptation_gain, auc_margin,
               final_accuracy_margin, forgetting_margin, causal_margin,
               minimum_spike_rate, maximum_spike_rate),
        target_parameters=target_parameters, levels=levels,
        kernels=(input_kernel_size, hidden_kernel_size), tcn_dilation=tcn_dilation,
        surrogate_slope=surrogate_slope,
    )
    progress = _load_progress(progress_path, signature)
    resolved = resolve_device(device)
    full_config = replace(config, train_samples=0, test_samples=0, epochs=source_epochs)
    existing = list(progress.get("records", []))
    if progress.get("stage") == "complete":
        return _completed_result(
            progress, full_config, resolved, target_parameters, levels, seed_values,
            source_mask_fraction, damage_fraction, damage_seed, budgets,
            minimum_shift_drop, minimum_adaptation_gain, auc_margin,
            final_accuracy_margin, forgetting_margin, causal_margin,
            minimum_spike_rate, maximum_spike_rate,
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
        target_parameters=target_parameters, levels=levels,
        input_kernel_size=input_kernel_size, hidden_kernel_size=hidden_kernel_size,
        tcn_dilation=tcn_dilation, surrogate_slope=surrogate_slope,
        device=resolved, existing_records=existing,
        progress_callback=lambda rows: _save_progress(
            progress_path, signature, stage="adaptation", records=rows
        ),
    )
    summary = summarize_gen11_adaptation(records, budgets=budgets)
    decision = decide_gen11_plastic_adapter(
        summary, minimum_shift_drop=minimum_shift_drop,
        minimum_adaptation_gain=minimum_adaptation_gain, auc_margin=auc_margin,
        final_accuracy_margin=final_accuracy_margin,
        forgetting_margin=forgetting_margin, causal_margin=causal_margin,
        minimum_spike_rate=minimum_spike_rate, maximum_spike_rate=maximum_spike_rate,
    )
    _save_progress(progress_path, signature, stage="complete", records=records, decision=decision)
    return Gen11PlasticAdapterResult(
        config=full_config, device=device_kind(resolved), target_parameters=target_parameters,
        temporal_levels=levels, seeds=seed_values, source_mask_fraction=source_mask_fraction,
        damage_fraction=damage_fraction, damage_seed=damage_seed,
        adaptation_budgets=budgets, records=records, summary=summary, decision=decision,
    )


def summarize_gen11_adaptation(records: Iterable[dict], *, budgets: Iterable[int]) -> list[dict]:
    rows = list(records)
    if not rows:
        return []
    max_budget = max(int(value) for value in budgets)
    summary = []
    for strategy in GEN11_ADAPTATION_STRATEGIES:
        group = [row for row in rows if row["strategy"] == strategy]
        if not group:
            continue
        per_seed = []
        for seed in sorted({int(row["seed"]) for row in group}):
            curve = sorted((row for row in group if int(row["seed"]) == seed), key=lambda row: int(row["adaptation_samples"]))
            initial = next(row for row in curve if int(row["adaptation_samples"]) == 0)
            final = next(row for row in curve if int(row["adaptation_samples"]) == max_budget)
            per_seed.append({
                "seed": seed, "source_initial": float(initial["source_accuracy"]),
                "shifted_initial": float(initial["shifted_accuracy"]),
                "source_final": float(final["source_accuracy"]),
                "shifted_final": float(final["shifted_accuracy"]),
                "gain": float(final["shifted_accuracy"]) - float(initial["shifted_accuracy"]),
                "forgetting": float(initial["source_accuracy"]) - float(final["source_accuracy"]),
                "auc": _curve_auc(curve, max_budget), "activity": float(final["activity"]),
                "state_contribution": float(final["state_contribution"] or 0.0),
                "state_specificity": float(final["state_specificity"] or 0.0),
                "gate": float(final["mean_absolute_gate"]),
                "throughput": float(final["test_examples_per_second"]),
                "seconds": float(final["cumulative_adaptation_seconds"]),
            })
        summary.append({
            "strategy": strategy, "runs": len(per_seed),
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
            "mean_state_contribution": statistics.fmean(item["state_contribution"] for item in per_seed),
            "state_contribution_seed_count": sum(item["state_contribution"] >= 0.005 for item in per_seed),
            "mean_state_specificity": statistics.fmean(item["state_specificity"] for item in per_seed),
            "state_specificity_seed_count": sum(item["state_specificity"] >= 0.005 for item in per_seed),
            "mean_absolute_gate": statistics.fmean(item["gate"] for item in per_seed),
            "mean_test_examples_per_second": statistics.fmean(item["throughput"] for item in per_seed),
            "mean_cumulative_adaptation_seconds": statistics.fmean(item["seconds"] for item in per_seed),
            "adaptation_trainable_parameters": int(group[0]["adaptation_trainable_parameters"]),
        })
    return sorted(summary, key=lambda row: (-float(row["mean_adaptation_auc"]), str(row["strategy"])))


def decide_gen11_plastic_adapter(
    summary: Iterable[dict], *, minimum_shift_drop: float,
    minimum_adaptation_gain: float, auc_margin: float,
    final_accuracy_margin: float, forgetting_margin: float, causal_margin: float,
    minimum_spike_rate: float, maximum_spike_rate: float,
) -> dict:
    lookup = {row["strategy"]: row for row in summary}
    required_names = ("dropout_tcn_static", "dropout_tcn_readout", "lif_state_adapter")
    if any(name not in lookup for name in required_names):
        return {"status": "stop", "qualified_arms": [], "reason": "required strategy missing", "next_milestone": "close_gen11_plastic_adapter"}
    static, readout, lif = (lookup[name] for name in required_names)
    required = 2 if int(lif["runs"]) >= 3 else 1
    passed = (
        float(static["mean_shift_drop"]) >= minimum_shift_drop
        and float(lif["mean_adaptation_gain"]) >= minimum_adaptation_gain
        and int(lif["two_point_gain_seed_count"]) >= required
        and float(lif["mean_adaptation_auc"]) >= float(readout["mean_adaptation_auc"]) - auc_margin
        and float(lif["mean_shifted_final_accuracy"]) >= float(readout["mean_shifted_final_accuracy"]) - final_accuracy_margin
        and float(lif["mean_forgetting"]) <= float(readout["mean_forgetting"]) + forgetting_margin
        and float(lif["mean_state_contribution"]) >= causal_margin
        and int(lif["state_contribution_seed_count"]) >= required
        and float(lif["mean_state_specificity"]) >= causal_margin
        and int(lif["state_specificity_seed_count"]) >= required
        and minimum_spike_rate <= float(lif["mean_activity"]) <= maximum_spike_rate
    )
    return {
        "status": "pass" if passed else "stop",
        "qualified_arms": ["lif_state_adapter"] if passed else [],
        "next_milestone": "stw_ltw_memory" if passed else "close_gen11_plastic_adapter",
    }


def plot_gen11_plastic_adapter(records, summary, path) -> None:
    import matplotlib.pyplot as plt
    budgets = sorted({int(row["adaptation_samples"]) for row in records})
    figure, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    for strategy in GEN11_ADAPTATION_STRATEGIES:
        group = [row for row in records if row["strategy"] == strategy]
        if not group:
            continue
        shifted = [statistics.fmean(float(row["shifted_accuracy"]) for row in group if int(row["adaptation_samples"]) == budget) for budget in budgets]
        source = [statistics.fmean(float(row["source_accuracy"]) for row in group if int(row["adaptation_samples"]) == budget) for budget in budgets]
        axes[0, 0].plot(budgets, [100.0 * value for value in shifted], marker="o", label=strategy)
        axes[0, 1].plot(budgets, [100.0 * value for value in source], marker="o", label=strategy)
    for axis, title in ((axes[0, 0], "Damaged-task adaptation"), (axes[0, 1], "Source retention")):
        axis.set_title(title); axis.set_xscale("symlog", linthresh=64); axis.legend(fontsize=7); axis.grid(alpha=0.25)
    labels = [row["strategy"].replace("_", "\n") for row in summary]
    axes[1, 0].bar(labels, [100.0 * row["mean_adaptation_auc"] for row in summary])
    axes[1, 0].set_ylabel("Adaptation AUC (%)")
    axes[1, 1].bar(labels, [100.0 * row["mean_state_specificity"] for row in summary], color="#167d55")
    axes[1, 1].axhline(0.5, color="#bd3d3a", linestyle="--")
    axes[1, 1].set_ylabel("Full - shuffled state (points)")
    for axis in axes[1]: axis.grid(axis="y", alpha=0.25)
    destination = pathlib.Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180); plt.close(figure)


def _run_all_strategies(
    seeds, config, train_events, train_labels, validation_events, validation_labels,
    test_events, test_labels, *, fixed_mask, source_mask_fraction, budgets,
    adaptation_epochs_per_block, adaptation_learning_rate, target_parameters,
    levels, input_kernel_size, hidden_kernel_size, tcn_dilation, surrogate_slope,
    device, existing_records=(), progress_callback=None,
):
    records = list(existing_records)
    completed = {(int(row["seed"]), row["strategy"], int(row["adaptation_samples"])) for row in records}
    for seed in seeds:
        expected = {(seed, strategy, budget) for strategy in GEN11_ADAPTATION_STRATEGIES for budget in budgets}
        if expected.issubset(completed):
            continue
        backbone = _train_dropout_backbone(
            seed, config, train_events, train_labels, validation_events, validation_labels,
            source_mask_fraction=source_mask_fraction, target_parameters=target_parameters,
            levels=levels, input_kernel_size=input_kernel_size,
            hidden_kernel_size=hidden_kernel_size, tcn_dilation=tcn_dilation, device=device,
        )
        models = {
            "dropout_tcn_static": copy.deepcopy(backbone),
            "dropout_tcn_readout": copy.deepcopy(backbone),
            "dropout_tcn_full_finetune": copy.deepcopy(backbone),
            "analog_state_adapter": PlasticStateAdapter(copy.deepcopy(backbone), dynamics="analog", surrogate_slope=surrogate_slope).to(device),
            "lif_state_adapter": PlasticStateAdapter(copy.deepcopy(backbone), dynamics="lif", surrogate_slope=surrogate_slope).to(device),
        }
        for strategy, model in models.items():
            if {(seed, strategy, budget) for budget in budgets}.issubset(completed):
                continue
            new_rows = _adaptation_curve(
                model, strategy, seed, validation_events, validation_labels,
                test_events, test_labels, fixed_mask=fixed_mask, budgets=budgets,
                epochs_per_block=adaptation_epochs_per_block,
                learning_rate=adaptation_learning_rate, batch_size=config.batch_size,
                weight_decay=config.weight_decay, device=device,
            )
            records = [row for row in records if not (int(row["seed"]) == seed and row["strategy"] == strategy)]
            records.extend(new_rows); completed.update((seed, strategy, budget) for budget in budgets)
            if progress_callback is not None: progress_callback(records)
            del model
        del backbone, models
        gc.collect()
    return records


def _train_dropout_backbone(seed, config, train_events, train_labels,
                            validation_events, validation_labels, *, source_mask_fraction,
                            target_parameters, levels, input_kernel_size,
                            hidden_kernel_size, tcn_dilation, device):
    seed_everything(seed, device=device)
    channels, _ = matched_temporal_tcn_channels(
        config.input_neurons, config.classes, target_parameters,
        input_kernel_size=input_kernel_size, hidden_kernel_size=hidden_kernel_size,
        temporal_levels=levels,
    )
    model = TemporalDilatedTCNClassifier(
        config, channels=channels, input_kernel_size=input_kernel_size,
        hidden_kernel_size=hidden_kernel_size, dilation=tcn_dilation,
        temporal_levels=levels,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(seed + 111_000)
    best_accuracy = float("-inf"); best_state = None
    for _ in range(config.epochs):
        model.train(); order = torch.randperm(train_events.shape[0], generator=generator)
        for offset in range(0, order.shape[0], config.batch_size):
            index = order[offset: offset + config.batch_size]
            events = train_events.index_select(0, index).to(device)
            labels = train_labels.index_select(0, index).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(_random_sensor_dropout(events, source_mask_fraction, generator)), labels)
            loss.backward(); optimizer.step(); mark_step(device)
        accuracy, _, _ = _measure(model, validation_events, validation_labels, config.batch_size, device)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def _adaptation_curve(model, strategy, seed, adaptation_events, adaptation_labels,
                      test_events, test_labels, *, fixed_mask, budgets,
                      epochs_per_block, learning_rate, batch_size, weight_decay, device):
    for parameter in model.parameters(): parameter.requires_grad_(False)
    if strategy == "dropout_tcn_readout":
        for parameter in model.classifier.parameters(): parameter.requires_grad_(True)
    elif strategy == "dropout_tcn_full_finetune":
        for parameter in model.parameters(): parameter.requires_grad_(True)
    elif strategy in {"analog_state_adapter", "lif_state_adapter"}:
        for name, parameter in model.named_parameters():
            if not name.startswith("backbone."): parameter.requires_grad_(True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay) if trainable else None
    generator = torch.Generator(device="cpu").manual_seed(seed + 112_000)
    order = torch.randperm(adaptation_events.shape[0], generator=generator)
    rows = []; previous = 0; cumulative_seconds = 0.0
    for budget in budgets:
        if hasattr(model, "set_ablation_mode"):
            model.set_ablation_mode("full")
        if budget > previous and optimizer is not None:
            indices = order[previous:budget]
            cumulative_seconds += _adapt_block(model, optimizer, adaptation_events, adaptation_labels, indices, fixed_mask, epochs_per_block, batch_size, device, generator)
        source_accuracy, _, _ = _measure(model, test_events, test_labels, batch_size, device)
        shifted_accuracy, shifted_seconds, activity = _measure_mode(model, test_events, test_labels, batch_size, device, fixed_mask, "full")
        direct = shuffled = None
        if isinstance(model, PlasticStateAdapter):
            direct = _measure_mode(model, test_events, test_labels, batch_size, device, fixed_mask, "direct_only")[0]
            shuffled = _measure_mode(model, test_events, test_labels, batch_size, device, fixed_mask, "shuffled_state")[0]
            model.set_ablation_mode("full")
        rows.append({
            "seed": seed, "strategy": strategy, "adaptation_samples": int(budget),
            "source_accuracy": source_accuracy, "shifted_accuracy": shifted_accuracy,
            "activity": activity, "state_contribution": shifted_accuracy - direct if direct is not None else None,
            "state_specificity": shifted_accuracy - shuffled if shuffled is not None else None,
            "mean_absolute_gate": model.mean_absolute_gate() if hasattr(model, "mean_absolute_gate") else 0.0,
            "test_examples_per_second": test_events.shape[0] / max(shifted_seconds, 1e-12),
            "cumulative_adaptation_seconds": cumulative_seconds,
            "adaptation_trainable_parameters": sum(parameter.numel() for parameter in trainable),
        })
        previous = int(budget)
    return rows


def _adapt_block(model, optimizer, events, labels, indices, fixed_mask, epochs, batch_size, device, generator):
    sync(device); start = time.perf_counter()
    for _ in range(epochs):
        model.train(); order = indices.index_select(0, torch.randperm(indices.shape[0], generator=generator))
        for offset in range(0, order.shape[0], batch_size):
            index = order[offset: offset + batch_size]
            batch_events = apply_sensor_damage(events.index_select(0, index), fixed_mask).to(device)
            batch_labels = labels.index_select(0, index).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(batch_events), batch_labels)
            loss.backward(); optimizer.step(); mark_step(device)
    sync(device); return time.perf_counter() - start


def _measure_mode(model, events, labels, batch_size, device, fixed_mask, mode):
    if hasattr(model, "set_ablation_mode"): model.set_ablation_mode(mode)
    model.eval(); correct = total = 0; weighted_activity = 0.0
    sync(device); start = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, events.shape[0], batch_size):
            batch_events = apply_sensor_damage(events[offset: offset + batch_size], fixed_mask).to(device)
            batch_labels = labels[offset: offset + batch_size].to(device)
            logits, activity = model(batch_events, return_event_rate=True)
            correct += int((logits.argmax(dim=1) == batch_labels).sum().item())
            total += int(batch_labels.shape[0]); weighted_activity += float(activity.item()) * int(batch_labels.shape[0]); mark_step(device)
    sync(device)
    return correct / max(total, 1), time.perf_counter() - start, weighted_activity / max(total, 1)


def _curve_auc(curve, max_budget):
    ordered = sorted(curve, key=lambda row: int(row["adaptation_samples"])); area = 0.0
    if max_budget <= 0: return float(ordered[-1]["shifted_accuracy"])
    for left, right in zip(ordered, ordered[1:]):
        width = (int(right["adaptation_samples"]) - int(left["adaptation_samples"])) / max_budget
        area += width * 0.5 * (float(left["shifted_accuracy"]) + float(right["shifted_accuracy"]))
    return area


def _completed_result(progress, config, device, target_parameters, levels, seeds,
                      source_mask_fraction, damage_fraction, damage_seed, budgets,
                      minimum_shift_drop, minimum_adaptation_gain, auc_margin,
                      final_accuracy_margin, forgetting_margin, causal_margin,
                      minimum_spike_rate, maximum_spike_rate):
    records = list(progress.get("records", [])); summary = summarize_gen11_adaptation(records, budgets=budgets)
    decision = progress.get("decision") or decide_gen11_plastic_adapter(
        summary, minimum_shift_drop=minimum_shift_drop,
        minimum_adaptation_gain=minimum_adaptation_gain, auc_margin=auc_margin,
        final_accuracy_margin=final_accuracy_margin, forgetting_margin=forgetting_margin,
        causal_margin=causal_margin, minimum_spike_rate=minimum_spike_rate,
        maximum_spike_rate=maximum_spike_rate,
    )
    return Gen11PlasticAdapterResult(
        config=config, device=device_kind(device), target_parameters=target_parameters,
        temporal_levels=levels, seeds=seeds, source_mask_fraction=source_mask_fraction,
        damage_fraction=damage_fraction, damage_seed=damage_seed,
        adaptation_budgets=budgets, records=records, summary=summary, decision=decision,
    )


def _validate_run(config, seeds, budgets, levels, source_epochs, source_mask_fraction,
                  damage_fraction, adaptation_epochs, adaptation_lr, gates, spike_gate):
    if not seeds or not levels or source_epochs <= 0 or adaptation_epochs <= 0: raise ValueError("seeds, levels, and epochs must be positive")
    if not budgets or budgets[0] != 0 or tuple(sorted(set(budgets))) != budgets: raise ValueError("budgets must be unique, increasing, and start at zero")
    if budgets[-1] > 9981 or adaptation_lr <= 0.0: raise ValueError("invalid adaptation budget or learning rate")
    if not 0.0 < source_mask_fraction < 1.0 or not 0.0 < damage_fraction < 1.0: raise ValueError("invalid mask fraction")
    if any(not 0.0 <= value <= 1.0 for value in gates): raise ValueError("invalid gate")
    if not 0.0 <= spike_gate[0] <= spike_gate[1] <= 1.0: raise ValueError("invalid spike gate")
    if config.input_neurons <= 0: raise ValueError("invalid input size")


def _run_signature(config, **values):
    signature = {"version": 1, "experiment": "gen11_plastic_adapter", "strategies": list(GEN11_ADAPTATION_STRATEGIES), "input_neurons": config.input_neurons, "classes": config.classes, "timesteps": config.timesteps, "learning_rate": config.learning_rate, "weight_decay": config.weight_decay, "batch_size": config.batch_size, "data_root": str(config.data_root), "data_seed": config.data_seed}
    for key, value in values.items(): signature[key] = list(value) if isinstance(value, tuple) else value
    return signature


def _save_progress(path, signature, *, stage, records, decision=None):
    if path is None: return
    destination = pathlib.Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"signature": signature, "stage": stage, "records": list(records), "decision": decision}
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8"); temporary.replace(destination)


def _write_csv(path, rows):
    if not rows: path.write_text("", encoding="utf-8"); return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
