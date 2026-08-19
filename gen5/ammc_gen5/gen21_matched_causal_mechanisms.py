"""Gen-21 matched causal mechanism benchmark on frozen SSC residual-LIF state.

The supported Phase 48 residual-state backbone is trained once per seed and
then frozen.  Every adaptive arm receives the same residual readout tensor,
data, update count, and active-slot budget.  The arms differ only in the
registered mechanism used during post-damage adaptation.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
import json
import pathlib
import statistics
import time
from typing import Iterable
import zipfile

from .event_mnist import torch
from .gen9_continual_adaptation import apply_sensor_damage, sensor_damage_indices
from .runtime import device_kind, mark_step, resolve_device, seed_everything, sync
from .shd_benchmark import SHDConfig
from .shd_state_placement_diagnostic import (
    ResidualTemporalConvStateClassifier,
    matched_temporal_conv_residual_channels,
)
from .shd_temporal_pyramid import DEFAULT_TEMPORAL_LEVELS
from .shd_validation_checkpoint import _stratified_split, _train_validation_selected
from .ssc_benchmark import load_ssc_tensors


GEN21_PRIMARY_ARMS = (
    "topology_only",
    "dual_memory_only",
    "learned_delay_only",
    "local_credit_only",
)
GEN21_CONTROL_ARMS = ("static_backbone", "global_gradient_control")
GEN21_ARMS = GEN21_CONTROL_ARMS + GEN21_PRIMARY_ARMS


@dataclass(frozen=True)
class Gen21Config:
    screen_seed: int = 321
    confirmation_seeds: tuple[int, ...] = (322, 323, 324)
    input_neurons: int = 700
    classes: int = 35
    timesteps: int = 64
    duration_seconds: float = 1.0
    data_root: str = "gen5_data/ssc"
    download: bool = True
    source_train_samples: int = 20_000
    validation_samples: int = 6_000
    test_samples: int = 8_000
    source_epochs: int = 12
    adaptation_epochs: int = 5
    batch_size: int = 256
    source_learning_rate: float = 0.003
    adaptation_learning_rate: float = 0.01
    weight_decay: float = 0.0001
    source_validation_fraction: float = 0.50
    target_parameters: int = 133_631
    temporal_conv_kernel_size: int = 5
    temporal_levels: tuple[int, ...] = DEFAULT_TEMPORAL_LEVELS
    surrogate_slope: float = 10.0
    sensor_damage_fraction: float = 0.35
    delay_slots: int = 3
    active_slot_fraction: float = 0.35
    topology_rewire_fraction: float = 0.05
    stw_decay: float = 0.98
    consolidation_rate: float = 0.02
    local_reward_baseline_decay: float = 0.95
    minimum_adaptation_gain: float = 0.01
    maximum_retention_drop: float = 0.015
    minimum_causal_margin: float = 0.005
    minimum_confirmation_seed_count: int = 2


@dataclass
class Gen21Result:
    config: dict
    device: str
    dataset: dict
    screen_records: list[dict]
    promoted_arms: list[str]
    confirmation_records: list[dict]
    summary: list[dict]
    decision: dict

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "gen21_matched_causal_mechanisms.json"
        screen_path = output / "gen21_matched_causal_mechanisms_screen.csv"
        records_path = output / "gen21_matched_causal_mechanisms_records.csv"
        summary_path = output / "gen21_matched_causal_mechanisms_summary.csv"
        json_path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        _write_csv(screen_path, self.screen_records)
        _write_csv(records_path, self.confirmation_records)
        _write_csv(summary_path, self.summary)
        paths = {
            "json": str(json_path),
            "screen_csv": str(screen_path),
            "records_csv": str(records_path),
            "summary_csv": str(summary_path),
        }
        if plot:
            plot_path = output / "gen21_matched_causal_mechanisms.png"
            plot_gen21(self, plot_path)
            paths["plot"] = str(plot_path)
        return paths


class Gen21MechanismReadout(torch.nn.Module if torch is not None else object):
    """Budget-matched residual readout with mechanism-specific adaptation."""

    def __init__(self, backbone, arm: str, config: Gen21Config, *, seed: int):
        if torch is None:
            raise ImportError("Gen-21 requires PyTorch")
        if arm not in GEN21_ARMS and arm != "combined":
            raise ValueError(f"unknown Gen-21 arm: {arm}")
        super().__init__()
        self.backbone = backbone
        self.arm = arm
        self.config = config
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        feature_channels = 2 * int(backbone.channels)
        shape = (config.classes, feature_channels, config.delay_slots)
        self.delta = torch.nn.Parameter(torch.zeros(shape), requires_grad=arm != "static_backbone")
        self.register_buffer("ltw", torch.zeros(shape))
        generator = torch.Generator(device="cpu").manual_seed(seed + 71_000)
        scores = torch.rand(shape, generator=generator)
        count = max(1, int(round(scores.numel() * config.active_slot_fraction)))
        threshold = torch.topk(scores.flatten(), count).values[-1]
        self.register_buffer("active_mask", scores.ge(threshold).to(torch.float32))
        self.register_buffer("gradient_score", torch.zeros(shape))
        self.register_buffer("reward_baseline", torch.zeros(()))
        self.delay_mode = "normal"
        self.causal_mode = "normal"

    @property
    def allocated_slots(self) -> int:
        return int(self.delta.numel())

    @property
    def active_slots(self) -> int:
        return int(self.active_mask.sum().item())

    def forward(self, events, *, return_event_rate: bool = False):  # type: ignore[override]
        with torch.no_grad():
            base_logits, base_activity = self.backbone(events, return_event_rate=True)
            direct, state, _, _ = self.backbone.encode_trace(events)
            trace = torch.cat((direct, state), dim=2)
        if self.arm == "static_backbone" or self.causal_mode == "removed":
            logits = base_logits
        else:
            delayed = self._delay_features(trace)
            weight = self._effective_weight()
            logits = base_logits + torch.einsum("bfd,cfd->bc", delayed, weight)
        if return_event_rate:
            return logits, base_activity
        return logits

    def _delay_features(self, trace):
        time_weight = torch.linspace(0.25, 1.0, trace.shape[1], device=trace.device)
        time_weight = time_weight / time_weight.sum()
        pooled = []
        use_delays = self.arm in {"learned_delay_only", "combined"}
        for delay in range(self.config.delay_slots):
            effective_delay = delay if use_delays else 0
            shifted = trace
            if effective_delay:
                shifted = torch.zeros_like(trace)
                shifted[:, effective_delay:] = trace[:, :-effective_delay]
            pooled.append((shifted * time_weight.view(1, -1, 1)).sum(dim=1))
        values = torch.stack(pooled, dim=2)
        if self.delay_mode == "shuffled":
            values = values.flip(dims=(2,))
        return values

    def _effective_weight(self):
        mask = self.active_mask
        if self.causal_mode == "shuffled_topology":
            mask = mask.flatten().roll(mask.numel() // 3).view_as(mask)
        weight = self.delta
        if self.arm in {"dual_memory_only", "combined"}:
            weight = weight + (torch.zeros_like(self.ltw) if self.causal_mode == "zero_ltw" else self.ltw)
        if self.arm in {"topology_only", "combined"}:
            # Forward uses only active slots while inactive slots retain a
            # straight-through utility gradient for evidence-based sprouting.
            inactive = weight * (1.0 - mask)
            return weight * mask + inactive - inactive.detach()
        return weight * mask

    @torch.no_grad()
    def after_gradient_step(self) -> None:
        if self.delta.grad is not None:
            self.gradient_score.mul_(0.90).add_(self.delta.grad.abs(), alpha=0.10)
        if self.arm in {"dual_memory_only", "combined"}:
            self.ltw.add_(self.delta, alpha=self.config.consolidation_rate)
            self.delta.mul_(self.config.stw_decay)

    @torch.no_grad()
    def rewire(self) -> None:
        if self.arm not in {"topology_only", "combined"}:
            return
        active = self.active_mask.bool().flatten()
        count = max(1, int(round(active.sum().item() * self.config.topology_rewire_fraction)))
        weight = self.delta.detach().abs().flatten()
        active_index = torch.nonzero(active, as_tuple=False).flatten()
        inactive_index = torch.nonzero(~active, as_tuple=False).flatten()
        if not active_index.numel() or not inactive_index.numel():
            return
        count = min(count, active_index.numel(), inactive_index.numel())
        prune = active_index[torch.topk(weight[active_index], count, largest=False).indices]
        grow_score = self.gradient_score.flatten()[inactive_index]
        grow = inactive_index[torch.topk(grow_score, count).indices]
        flat_mask = self.active_mask.flatten()
        flat_mask[prune] = 0.0
        flat_mask[grow] = 1.0
        self.delta.flatten()[prune] = 0.0
        self.delta.flatten()[grow] = 0.0

    @torch.no_grad()
    def local_update(self, events, labels, *, learning_rate: float, shuffle_reward: bool, generator) -> float:
        base_logits, _ = self.backbone(events, return_event_rate=True)
        direct, state, _, _ = self.backbone.encode_trace(events)
        features = self._delay_features(torch.cat((direct, state), dim=2))
        logits = base_logits + torch.einsum("bfd,cfd->bc", features, self._effective_weight())
        probabilities = torch.softmax(logits, dim=1)
        actions = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
        reward = actions.eq(labels).to(torch.float32)
        if shuffle_reward:
            reward = reward[torch.randperm(reward.numel(), device=reward.device, generator=generator)]
        advantage = reward - self.reward_baseline
        score = torch.nn.functional.one_hot(actions, self.config.classes).to(torch.float32) - probabilities
        update = torch.einsum("b,bc,bfd->cfd", advantage, score, features) / max(events.shape[0], 1)
        self.delta.add_(learning_rate * update * self.active_mask)
        self.reward_baseline.mul_(self.config.local_reward_baseline_decay).add_(
            reward.mean(), alpha=1.0 - self.config.local_reward_baseline_decay
        )
        return float(reward.mean().item())


def available_gen21_arms() -> tuple[str, ...]:
    return GEN21_ARMS


def run_gen21(
    config: Gen21Config = Gen21Config(),
    *,
    device="auto",
    progress_path: str | pathlib.Path | None = None,
    dataset=None,
) -> Gen21Result:
    _validate_config(config)
    resolved = resolve_device(device)
    data = dataset if dataset is not None else _load_dataset(config)
    recovered = _load_progress(progress_path, config)
    recovered_screen = [row for row in recovered if row.get("stage") == "screen"]
    screen_records = _run_stage(
        config, (config.screen_seed,), GEN21_ARMS, data, resolved,
        stage="screen", progress_path=progress_path, prior_records=[],
        existing_records=recovered_screen,
    )
    promoted = select_gen21_promoted_arms(screen_records, config)
    confirmation_arms = ("static_backbone", "global_gradient_control") + tuple(promoted)
    if set(GEN21_PRIMARY_ARMS).issubset(promoted):
        confirmation_arms += ("combined",)
    confirmation_records = _run_stage(
        config, config.confirmation_seeds, confirmation_arms, data, resolved,
        stage="confirmation", progress_path=progress_path, prior_records=screen_records,
        existing_records=[row for row in recovered if row.get("stage") == "confirmation"],
    ) if promoted else []
    summary = summarize_gen21(confirmation_records or screen_records)
    decision = decide_gen21(summary, promoted, config)
    return Gen21Result(
        config=asdict(config), device=device_kind(resolved),
        dataset={
            "name": "Spiking Speech Commands",
            "source_train_samples": int(data[0].shape[0]),
            "source_validation_samples": int(data[2].shape[0]),
            "adaptation_samples": int(data[4].shape[0]),
            "test_samples": int(data[6].shape[0]),
            "shift": f"deterministic_{config.sensor_damage_fraction:.0%}_sensor_damage",
        },
        screen_records=screen_records, promoted_arms=promoted,
        confirmation_records=confirmation_records, summary=summary, decision=decision,
    )


def _load_dataset(config: Gen21Config):
    shd = _ssc_config(config, seed=config.screen_seed)
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_ssc_tensors(
        shd, validation_samples=config.validation_samples
    )
    source_val_x, source_val_y, adapt_x, adapt_y = _stratified_split(
        valid_x, valid_y, fraction=1.0 - config.source_validation_fraction,
        seed=config.screen_seed + 19,
    )
    return train_x, train_y, source_val_x, source_val_y, adapt_x, adapt_y, test_x, test_y


def _ssc_config(config: Gen21Config, *, seed: int) -> SHDConfig:
    return SHDConfig(
        seeds=(seed,), train_samples=config.source_train_samples,
        test_samples=config.test_samples, input_neurons=config.input_neurons,
        classes=config.classes, timesteps=config.timesteps,
        duration_seconds=config.duration_seconds, epochs=config.source_epochs,
        learning_rate=config.source_learning_rate, weight_decay=config.weight_decay,
        batch_size=config.batch_size, data_root=config.data_root,
        download=config.download,
    )


def _build_backbone(config: Gen21Config, *, seed: int, device):
    shd = _ssc_config(config, seed=seed)
    channels, _ = matched_temporal_conv_residual_channels(
        config.input_neurons, config.classes, config.target_parameters,
        kernel_size=config.temporal_conv_kernel_size,
        temporal_levels=config.temporal_levels,
    )
    return ResidualTemporalConvStateClassifier(
        shd, channels=channels, kernel_size=config.temporal_conv_kernel_size,
        temporal_levels=config.temporal_levels, dynamics="lif",
        surrogate_slope=config.surrogate_slope,
    ).to(device)


def _run_stage(
    config, seeds, arms, data, device, *, stage, progress_path, prior_records,
    existing_records,
):
    train_x, train_y, source_val_x, source_val_y, adapt_x, adapt_y, test_x, test_y = data
    requested = {(int(seed), arm) for seed in seeds for arm in arms}
    records = [
        row for row in existing_records
        if (int(row["seed"]), row["arm"]) in requested
    ]
    completed = {(int(row["seed"]), row["arm"]) for row in records}
    for seed in seeds:
        missing_arms = [arm for arm in arms if (int(seed), arm) not in completed]
        if not missing_arms:
            continue
        seed_everything(seed, device=device)
        backbone = _build_backbone(config, seed=seed, device=device)
        training = _train_validation_selected(
            backbone, train_x, train_y, source_val_x, source_val_y,
            _ssc_config(config, seed=seed), seed=seed, device=device,
        )
        backbone.load_state_dict(training["best_state"])
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        damage = sensor_damage_indices(config.input_neurons, config.sensor_damage_fraction, seed=seed + 8_000)
        damaged_adapt = apply_sensor_damage(adapt_x, damage)
        damaged_test = apply_sensor_damage(test_x, damage)
        static_clean = _evaluate_backbone(backbone, test_x, test_y, config.batch_size, device)[0]
        static_shift = _evaluate_backbone(backbone, damaged_test, test_y, config.batch_size, device)[0]
        for arm in missing_arms:
            started = time.perf_counter()
            model = Gen21MechanismReadout(backbone, arm, config, seed=seed).to(device)
            causal_model = None
            if arm != "static_backbone":
                _adapt_model(model, damaged_adapt, adapt_y, config, seed=seed, device=device)
                if arm == "local_credit_only":
                    causal_model = Gen21MechanismReadout(backbone, arm, config, seed=seed).to(device)
                    _adapt_model(causal_model, damaged_adapt, adapt_y, config, seed=seed, device=device, shuffle_reward=True)
            shifted_accuracy, latency, activity = _evaluate_model(model, damaged_test, test_y, config.batch_size, device)
            clean_accuracy, _, _ = _evaluate_model(model, test_x, test_y, config.batch_size, device)
            causal_accuracy = _causal_accuracy(model, causal_model, damaged_test, test_y, config, device)
            record = {
                "stage": stage, "seed": int(seed), "arm": arm,
                "source_best_epoch": int(training["best_epoch"]),
                "source_validation_accuracy": float(training["best_validation_accuracy"]),
                "static_clean_accuracy": float(static_clean),
                "static_shifted_accuracy": float(static_shift),
                "shifted_accuracy": float(shifted_accuracy),
                "adaptation_gain": float(shifted_accuracy - static_shift),
                "clean_retention_accuracy": float(clean_accuracy),
                "retention_drop": float(static_clean - clean_accuracy),
                "causal_control_accuracy": float(causal_accuracy),
                "causal_margin": float(shifted_accuracy - causal_accuracy),
                "allocated_slots": int(model.allocated_slots),
                "active_slots": int(model.active_slots),
                "active_slot_fraction": float(model.active_slots / model.allocated_slots),
                "trainable_adapter_parameters": int(model.delta.numel() if model.delta.requires_grad else 0),
                "active_operations_per_sample": int(model.active_slots * config.timesteps),
                "adapter_memory_bytes": int((model.delta.numel() + model.ltw.numel() + model.active_mask.numel()) * 4),
                "mean_activity": float(activity),
                "test_examples_per_second": float(test_y.numel() / max(latency, 1e-12)),
                "wall_seconds": float(time.perf_counter() - started),
            }
            records.append(record)
            _save_progress(progress_path, config, prior_records + records, stage)
    return records


def _adapt_model(model, events, labels, config, *, seed, device, shuffle_reward=False):
    if model.arm == "static_backbone":
        return
    generator = torch.Generator(device=device.type).manual_seed(seed + 90_000)
    optimizer = None
    if model.arm != "local_credit_only":
        optimizer = torch.optim.AdamW([model.delta], lr=config.adaptation_learning_rate, weight_decay=config.weight_decay)
    order_generator = torch.Generator(device="cpu").manual_seed(seed + 91_000)
    for _ in range(config.adaptation_epochs):
        order = torch.randperm(events.shape[0], generator=order_generator)
        model.train()
        for offset in range(0, events.shape[0], config.batch_size):
            index = order[offset : offset + config.batch_size]
            batch_x = events.index_select(0, index).to(device)
            batch_y = labels.index_select(0, index).to(device)
            if model.arm == "local_credit_only":
                model.local_update(
                    batch_x, batch_y, learning_rate=config.adaptation_learning_rate,
                    shuffle_reward=shuffle_reward, generator=generator,
                )
            else:
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                model.after_gradient_step()
            mark_step(device)
        model.rewire()


def _evaluate_backbone(model, events, labels, batch_size, device):
    return _evaluate_model(model, events, labels, batch_size, device)


def _evaluate_model(model, events, labels, batch_size, device):
    model.eval()
    correct = total = 0
    activity_total = 0.0
    sync(device)
    started = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, events.shape[0], batch_size):
            x = events[offset : offset + batch_size].to(device)
            y = labels[offset : offset + batch_size].to(device)
            logits, activity = model(x, return_event_rate=True)
            correct += int(logits.argmax(1).eq(y).sum().item())
            total += int(y.numel())
            activity_total += float(activity.item()) * y.numel()
            mark_step(device)
    sync(device)
    return correct / max(total, 1), time.perf_counter() - started, activity_total / max(total, 1)


def _causal_accuracy(model, causal_model, events, labels, config, device):
    if model.arm == "static_backbone":
        return _evaluate_model(model, events, labels, config.batch_size, device)[0]
    if causal_model is not None:
        return _evaluate_model(causal_model, events, labels, config.batch_size, device)[0]
    if model.arm == "topology_only":
        model.causal_mode = "shuffled_topology"
    elif model.arm == "dual_memory_only":
        model.causal_mode = "zero_ltw"
    elif model.arm == "learned_delay_only":
        model.delay_mode = "shuffled"
    else:
        model.causal_mode = "removed"
    value = _evaluate_model(model, events, labels, config.batch_size, device)[0]
    model.causal_mode = "normal"
    model.delay_mode = "normal"
    return value


def select_gen21_promoted_arms(records: Iterable[dict], config: Gen21Config) -> list[str]:
    lookup = {row["arm"]: row for row in records if row["arm"] in GEN21_PRIMARY_ARMS}
    return [
        arm for arm in GEN21_PRIMARY_ARMS
        if arm in lookup
        and float(lookup[arm]["adaptation_gain"]) >= config.minimum_adaptation_gain
        and float(lookup[arm]["retention_drop"]) <= config.maximum_retention_drop
        and float(lookup[arm]["causal_margin"]) >= config.minimum_causal_margin
    ]


def summarize_gen21(records: Iterable[dict]) -> list[dict]:
    rows = list(records)
    summary = []
    for arm in dict.fromkeys(row["arm"] for row in rows):
        group = [row for row in rows if row["arm"] == arm]
        summary.append({
            "arm": arm, "seeds": len(group),
            "mean_shifted_accuracy": statistics.fmean(float(row["shifted_accuracy"]) for row in group),
            "std_shifted_accuracy": statistics.pstdev(float(row["shifted_accuracy"]) for row in group),
            "mean_adaptation_gain": statistics.fmean(float(row["adaptation_gain"]) for row in group),
            "mean_clean_retention_accuracy": statistics.fmean(float(row["clean_retention_accuracy"]) for row in group),
            "mean_retention_drop": statistics.fmean(float(row["retention_drop"]) for row in group),
            "mean_causal_margin": statistics.fmean(float(row["causal_margin"]) for row in group),
            "positive_adaptation_seed_count": sum(float(row["adaptation_gain"]) >= 0.01 for row in group),
            "positive_causal_seed_count": sum(float(row["causal_margin"]) >= 0.005 for row in group),
            "active_slots": int(group[0]["active_slots"]),
            "allocated_slots": int(group[0]["allocated_slots"]),
            "active_operations_per_sample": int(group[0]["active_operations_per_sample"]),
            "adapter_memory_bytes": int(group[0]["adapter_memory_bytes"]),
            "mean_activity": statistics.fmean(float(row["mean_activity"]) for row in group),
            "mean_test_examples_per_second": statistics.fmean(float(row["test_examples_per_second"]) for row in group),
        })
    return summary


def decide_gen21(summary: Iterable[dict], promoted: Iterable[str], config: Gen21Config) -> dict:
    promoted = list(promoted)
    qualified = []
    for row in summary:
        if row["arm"] not in promoted:
            continue
        if (
            int(row["positive_adaptation_seed_count"]) >= config.minimum_confirmation_seed_count
            and int(row["positive_causal_seed_count"]) >= config.minimum_confirmation_seed_count
            and float(row["mean_retention_drop"]) <= config.maximum_retention_drop
        ):
            qualified.append(row["arm"])
    return {
        "status": "pass" if qualified else "stop",
        "qualified_mechanisms": qualified,
        "combined_arm_authorized": set(GEN21_PRIMARY_ARMS).issubset(qualified),
        "hardware_energy_claim_authorized": False,
        "next_milestone": "direct_mechanism_replication" if qualified else "close_or_redesign_failed_mechanisms",
        "interpretation": (
            "A mechanism qualifies only with paired adaptation, retention, and causal-control evidence; "
            "allocated slots are not interpreted as biological synapses or direct energy."
        ),
    }


def plot_gen21(result: Gen21Result, path: str | pathlib.Path) -> None:
    import matplotlib.pyplot as plt
    rows = result.summary or summarize_gen21(result.screen_records)
    labels = [row["arm"].replace("_", "\n") for row in rows]
    x = list(range(len(rows)))
    figure, axes = plt.subplots(3, 1, figsize=(14, 13), constrained_layout=True)
    axes[0].bar(x, [100 * float(row["mean_shifted_accuracy"]) for row in rows], color="#35b4f2")
    axes[0].set_ylabel("Shifted accuracy (%)")
    axes[0].set_title("AMMC Gen-21 matched causal mechanism benchmark")
    axes[1].bar(x, [100 * float(row["mean_adaptation_gain"]) for row in rows], color="#ffb31a")
    axes[1].axhline(100 * result.config["minimum_adaptation_gain"], color="#bd3d3a", linestyle="--")
    axes[1].set_ylabel("Adaptation gain (points)")
    axes[2].bar(x, [100 * float(row["mean_causal_margin"]) for row in rows], color="#167d55")
    axes[2].axhline(100 * result.config["minimum_causal_margin"], color="#bd3d3a", linestyle="--")
    axes[2].set_ylabel("Causal margin (points)")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25)
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def bundle_gen21_artifacts(paths: dict[str, str], output_dir: str | pathlib.Path) -> dict[str, str]:
    output = pathlib.Path(output_dir)
    files = [pathlib.Path(value) for value in paths.values() if pathlib.Path(value).is_file()]
    manifest = output / "gen21_matched_causal_mechanisms_manifest.json"
    manifest.write_text(json.dumps({"files": [file.name for file in files]}, indent=2) + "\n", encoding="utf-8")
    archive = output / "gen21_matched_causal_mechanisms_bundle.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for file in files + [manifest]:
            bundle.write(file, arcname=file.name)
    return {"manifest": str(manifest), "bundle": str(archive)}


def _save_progress(path, config, records, stage):
    if path is None:
        return
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"config": asdict(config), "stage": stage, "records": records}, indent=2) + "\n", encoding="utf-8")


def _load_progress(path, config):
    if path is None or not pathlib.Path(path).exists():
        return []
    try:
        payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    normalized_config = json.loads(json.dumps(asdict(config)))
    if payload.get("config") != normalized_config:
        return []
    return list(payload.get("records", []))


def _validate_config(config: Gen21Config) -> None:
    if config.input_neurons != 700 or config.classes != 35:
        raise ValueError("Gen-21 is preregistered for SSC (700 inputs, 35 classes)")
    if config.delay_slots < 2 or not 0 < config.active_slot_fraction <= 1:
        raise ValueError("invalid matched readout slot configuration")
    if not 0 < config.sensor_damage_fraction < 1:
        raise ValueError("sensor damage fraction must lie between zero and one")
    if config.source_epochs <= 0 or config.adaptation_epochs <= 0 or config.batch_size <= 0:
        raise ValueError("epoch and batch settings must be positive")


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
