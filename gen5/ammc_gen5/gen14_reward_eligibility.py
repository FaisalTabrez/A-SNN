"""Gen-14 reward-modulated embodied eligibility-trace experiment.

This program is deliberately separate from the failed supervised Gen-13
branch. Agents receive only local sensor/action traces and scalar environmental
reward; no target action or autograd gradient is exposed to plastic weights.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import pathlib
from typing import Sequence

try:  # pragma: no cover
    import torch
except Exception:  # pragma: no cover
    torch = None

from .runtime import resolve_device, seed_everything, sync
from .tensor_environment import TensorEnvironment2D, TensorEnvironmentConfig


GEN14_STRATEGIES = (
    "static_random",
    "oracle_food_reflex",
    "analog_reward_eligibility",
    "spiking_reward_eligibility",
    "spiking_shuffled_reward",
)


@dataclass(frozen=True)
class Gen14Config:
    seeds: tuple[int, ...] = (163, 164, 165)
    agent_count: int = 10_000
    food_count: int = 128
    toxin_count: int = 128
    baseline_steps: int = 600
    training_steps: int = 3_600
    evaluation_steps: int = 600
    reward_delay_steps: int = 12
    eligibility_decay: float = 0.95
    trace_decay: float = 0.90
    reward_baseline_decay: float = 0.99
    local_learning_rate: float = 0.02
    fast_weight_decay: float = 0.0001
    progress_reward_scale: float = 0.05
    temperature: float = 0.50
    maximum_fast_weight: float = 1.0
    minimum_gain_per_1000_steps: float = 0.10
    minimum_control_margin_per_1000_steps: float = 0.10
    minimum_spike_density: float = 0.05
    maximum_spike_density: float = 0.35


@dataclass
class Gen14RewardEligibilityResult:
    config: dict
    device: str
    records: list[dict]
    summary: list[dict]
    decision: dict

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "gen14_reward_eligibility.json"
        records_path = output / "gen14_reward_eligibility_records.csv"
        summary_path = output / "gen14_reward_eligibility_summary.csv"
        json_path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        _write_csv(records_path, self.records)
        _write_csv(summary_path, self.summary)
        paths = {
            "json": str(json_path),
            "records_csv": str(records_path),
            "summary_csv": str(summary_path),
        }
        if plot:
            plot_path = output / "gen14_reward_eligibility.png"
            plot_gen14_reward_eligibility(self, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def available_gen14_strategies() -> tuple[str, ...]:
    return GEN14_STRATEGIES


def reward_modulated_update(
    fast_weight,
    eligibility,
    presynaptic_trace,
    probabilities,
    action_index,
    reward,
    reward_baseline,
    *,
    eligibility_decay: float,
    reward_baseline_decay: float,
    learning_rate: float,
    weight_decay: float,
    maximum_weight: float,
):
    """Return a local pre × post/action × scalar-reward update."""

    if torch is None:  # pragma: no cover
        raise ImportError("Gen-14 requires PyTorch")
    chosen = torch.nn.functional.one_hot(action_index, num_classes=4).to(probabilities.dtype)
    post_surprise = chosen - probabilities
    eligibility = eligibility * eligibility_decay + (
        post_surprise.unsqueeze(2) * presynaptic_trace.unsqueeze(1)
    )
    advantage = reward - reward_baseline
    fast_weight = (
        fast_weight * (1.0 - weight_decay)
        + learning_rate * advantage.unsqueeze(1).unsqueeze(2) * eligibility
    )
    fast_weight = fast_weight.clamp(-maximum_weight, maximum_weight)
    reward_baseline = (
        reward_baseline * reward_baseline_decay
        + reward * (1.0 - reward_baseline_decay)
    )
    return fast_weight, eligibility, reward_baseline


def run_gen14_reward_eligibility(
    config: Gen14Config | None = None,
    *,
    device: str = "auto",
    progress_path: str | pathlib.Path | None = None,
) -> Gen14RewardEligibilityResult:
    if torch is None:  # pragma: no cover
        raise ImportError("Gen-14 requires PyTorch")
    cfg = config or Gen14Config()
    _validate_config(cfg)
    resolved = resolve_device(device)
    records: list[dict] = []
    for seed in cfg.seeds:
        seed_everything(seed)
        records.extend(_run_seed(cfg, seed, resolved))
        if progress_path is not None:
            destination = pathlib.Path(progress_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps({"config": asdict(cfg), "records": records}, indent=2) + "\n",
                encoding="utf-8",
            )
    summary = summarize_gen14(records)
    decision = decide_gen14(summary, cfg)
    return Gen14RewardEligibilityResult(
        config=asdict(cfg),
        device=str(resolved),
        records=records,
        summary=summary,
        decision=decision,
    )


def _run_seed(cfg: Gen14Config, seed: int, device) -> list[dict]:
    group_size = cfg.agent_count // len(GEN14_STRATEGIES)
    environment = TensorEnvironment2D(
        TensorEnvironmentConfig(
            agent_count=cfg.agent_count,
            food_count=cfg.food_count,
            toxin_count=cfg.toxin_count,
            reward_delay_steps=cfg.reward_delay_steps,
        ),
        device=device,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    shape = (cfg.agent_count, 4, 8)
    base_weight = torch.randn(shape, device=device, generator=generator) * 0.05
    fast_weight = torch.zeros(shape, device=device)
    eligibility = torch.zeros(shape, device=device)
    trace = torch.zeros((cfg.agent_count, 8), device=device)
    reward_baseline = torch.zeros(cfg.agent_count, device=device)
    strategy_index = torch.arange(cfg.agent_count, device=device) // group_size
    action_vectors = torch.tensor(
        ((0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)),
        device=device,
    )
    records: list[dict] = []
    records.extend(_run_phase(
        "baseline", cfg.baseline_steps, False, cfg, environment, generator,
        strategy_index, base_weight, fast_weight, eligibility, trace,
        reward_baseline, action_vectors,
    ))
    records.extend(_run_phase(
        "training", cfg.training_steps, True, cfg, environment, generator,
        strategy_index, base_weight, fast_weight, eligibility, trace,
        reward_baseline, action_vectors,
    ))
    records.extend(_run_phase(
        "evaluation", cfg.evaluation_steps, False, cfg, environment, generator,
        strategy_index, base_weight, fast_weight, eligibility, trace,
        reward_baseline, action_vectors,
    ))
    for row in records:
        row["seed"] = seed
    return records


def _run_phase(
    phase, steps, learning, cfg, environment, generator, strategy_index,
    base_weight, fast_weight, eligibility, trace, reward_baseline, action_vectors,
):
    start_fitness = environment.fitness.detach().clone()
    shaped_totals = torch.zeros(len(GEN14_STRATEGIES), device=environment.device)
    activity_totals = torch.zeros(len(GEN14_STRATEGIES), device=environment.device)
    counts = torch.zeros(len(GEN14_STRATEGIES), device=environment.device)
    for _ in range(steps):
        nearest_before = environment.nearest_objects()
        sensory = environment.sensory_tensor()
        random_values = torch.rand(sensory.shape, device=environment.device, generator=generator)
        spike_input = (random_values < sensory.clamp(0.0, 1.0)).to(sensory.dtype)
        trace.mul_(cfg.trace_decay)
        spiking_rows = (strategy_index == 3) | (strategy_index == 4)
        trace[~spiking_rows] = sensory[~spiking_rows]
        trace[spiking_rows] += spike_input[spiking_rows]
        policy_input = trace / trace.norm(dim=1, keepdim=True).clamp_min(1.0)
        logits = torch.einsum("poi,pi->po", base_weight + fast_weight, policy_input)
        probabilities = torch.softmax(logits / cfg.temperature, dim=1)
        noise = -torch.log(-torch.log(torch.rand(
            probabilities.shape, device=environment.device, generator=generator
        ).clamp_(1e-6, 1.0 - 1e-6)))
        action_index = (torch.log(probabilities.clamp_min(1e-8)) + noise).argmax(dim=1)
        oracle = strategy_index == 1
        action_index[oracle] = sensory[oracle, :4].argmax(dim=1)
        world = environment.step(action_vectors[action_index], generator=generator)
        food_progress = nearest_before["nearest_food_dist"] - world["nearest_food_dist"]
        toxin_progress = world["nearest_toxin_dist"] - nearest_before["nearest_toxin_dist"]
        scalar_reward = (
            world["reward"] - world["punishment"]
            + cfg.progress_reward_scale * (food_progress + toxin_progress)
        )
        if learning:
            for group in (2, 3, 4):
                mask = strategy_index == group
                local_reward = scalar_reward[mask]
                if group == 4:
                    local_reward = local_reward[torch.randperm(
                        local_reward.numel(), device=environment.device, generator=generator
                    )]
                updated_weight, updated_eligibility, updated_baseline = reward_modulated_update(
                    fast_weight[mask], eligibility[mask], policy_input[mask],
                    probabilities[mask], action_index[mask], local_reward,
                    reward_baseline[mask],
                    eligibility_decay=cfg.eligibility_decay,
                    reward_baseline_decay=cfg.reward_baseline_decay,
                    learning_rate=cfg.local_learning_rate,
                    weight_decay=cfg.fast_weight_decay,
                    maximum_weight=cfg.maximum_fast_weight,
                )
                fast_weight[mask] = updated_weight
                eligibility[mask] = updated_eligibility
                reward_baseline[mask] = updated_baseline
        for group in range(len(GEN14_STRATEGIES)):
            mask = strategy_index == group
            shaped_totals[group] += scalar_reward[mask].sum()
            activity_totals[group] += spike_input[mask].mean()
            counts[group] += 1
    sync(environment.device)
    fitness_delta = environment.fitness - start_fitness
    rows = []
    for group, strategy in enumerate(GEN14_STRATEGIES):
        mask = strategy_index == group
        weights = fast_weight[mask]
        rows.append({
            "strategy": strategy,
            "phase": phase,
            "steps": steps,
            "mean_net_fitness_per_1000_steps": float(
                1000.0 * fitness_delta[mask].mean().detach().cpu().item() / steps
            ),
            "mean_shaped_reward_per_1000_steps": float(
                1000.0 * shaped_totals[group].detach().cpu().item() / (mask.sum().item() * steps)
            ),
            "mean_spike_density": float(
                activity_totals[group].detach().cpu().item() / counts[group].clamp_min(1).item()
            ),
            "mean_absolute_fast_weight": float(weights.abs().mean().detach().cpu().item()),
            "fast_weight_saturation": float(
                (weights.abs() >= 0.99 * cfg.maximum_fast_weight).float().mean().detach().cpu().item()
            ),
        })
    return rows


def summarize_gen14(records: Sequence[dict]) -> list[dict]:
    rows = []
    for strategy in GEN14_STRATEGIES:
        selected = [row for row in records if row["strategy"] == strategy]
        baseline = [row for row in selected if row["phase"] == "baseline"]
        evaluation = [row for row in selected if row["phase"] == "evaluation"]
        rows.append({
            "strategy": strategy,
            "runs": len(evaluation),
            "mean_baseline_net_fitness_per_1000_steps": _mean(baseline, "mean_net_fitness_per_1000_steps"),
            "mean_final_net_fitness_per_1000_steps": _mean(evaluation, "mean_net_fitness_per_1000_steps"),
            "mean_fitness_gain_per_1000_steps": _mean(evaluation, "mean_net_fitness_per_1000_steps") - _mean(baseline, "mean_net_fitness_per_1000_steps"),
            "mean_final_shaped_reward_per_1000_steps": _mean(evaluation, "mean_shaped_reward_per_1000_steps"),
            "mean_spike_density": _mean(evaluation, "mean_spike_density"),
            "mean_absolute_fast_weight": _mean(evaluation, "mean_absolute_fast_weight"),
            "mean_fast_weight_saturation": _mean(evaluation, "fast_weight_saturation"),
        })
    return rows


def decide_gen14(summary: Sequence[dict], config: Gen14Config) -> dict:
    by_name = {row["strategy"]: row for row in summary}
    static = by_name["static_random"]
    oracle = by_name["oracle_food_reflex"]
    spiking = by_name["spiking_reward_eligibility"]
    shuffled = by_name["spiking_shuffled_reward"]
    oracle_control = oracle["mean_final_net_fitness_per_1000_steps"] > static["mean_final_net_fitness_per_1000_steps"]
    gain_gate = spiking["mean_fitness_gain_per_1000_steps"] >= config.minimum_gain_per_1000_steps
    static_margin = spiking["mean_final_net_fitness_per_1000_steps"] - static["mean_final_net_fitness_per_1000_steps"]
    shuffled_margin = spiking["mean_final_net_fitness_per_1000_steps"] - shuffled["mean_final_net_fitness_per_1000_steps"]
    specificity_gate = min(static_margin, shuffled_margin) >= config.minimum_control_margin_per_1000_steps
    activity_gate = config.minimum_spike_density <= spiking["mean_spike_density"] <= config.maximum_spike_density
    saturation_gate = spiking["mean_fast_weight_saturation"] <= 0.05
    passed = oracle_control and gain_gate and specificity_gate and activity_gate and saturation_gate
    return {
        "status": "pass" if passed else "stop",
        "oracle_positive_control": oracle_control,
        "spiking_gain_gate": gain_gate,
        "spiking_specificity_gate": specificity_gate,
        "spiking_activity_gate": activity_gate,
        "spiking_saturation_gate": saturation_gate,
        "spiking_margin_vs_static_per_1000_steps": static_margin,
        "spiking_margin_vs_shuffled_per_1000_steps": shuffled_margin,
        "next_milestone": "preregister_reward_eligibility_confirmation" if passed else "close_reward_eligibility_screen",
    }


def plot_gen14_reward_eligibility(result: Gen14RewardEligibilityResult, path: str | pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    labels = [row["strategy"].replace("_", "\n") for row in result.summary]
    baseline = [row["mean_baseline_net_fitness_per_1000_steps"] for row in result.summary]
    final = [row["mean_final_net_fitness_per_1000_steps"] for row in result.summary]
    shaped = [row["mean_final_shaped_reward_per_1000_steps"] for row in result.summary]
    figure, axes = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)
    x = range(len(labels))
    axes[0].bar([value - 0.2 for value in x], baseline, width=0.4, label="baseline")
    axes[0].bar([value + 0.2 for value in x], final, width=0.4, label="post-learning")
    axes[0].set_ylabel("Net collision fitness / 1000 steps")
    axes[0].set_title("Gen-14 reward-modulated embodied eligibility")
    axes[0].set_xticks(list(x), labels)
    axes[0].legend()
    axes[1].bar(labels, shaped, color="#35b4f2")
    axes[1].set_ylabel("Shaped scalar reward / 1000 steps")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _mean(rows: Sequence[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0


def _write_csv(path: pathlib.Path, rows: Sequence[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validate_config(config: Gen14Config) -> None:
    if config.agent_count % len(GEN14_STRATEGIES):
        raise ValueError(f"agent_count must be divisible by {len(GEN14_STRATEGIES)}")
    if min(config.baseline_steps, config.training_steps, config.evaluation_steps) <= 0:
        raise ValueError("all phase step counts must be positive")
    if not 0.0 < config.temperature:
        raise ValueError("temperature must be positive")
    if not 0.0 <= config.minimum_spike_density < config.maximum_spike_density <= 1.0:
        raise ValueError("invalid spike-density gate")
