"""Gen-17 parameter-matched sparse-spiking translation of local reward credit."""

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

from .gen15_reward_baseline import discounted_returns
from .gen16_local_score_credit import (
    LinearRewardPolicy,
    manual_score_gradients,
    score_gradient_parity,
)
from .runtime import resolve_device, seed_everything, sync
from .tensor_environment import TensorEnvironment2D, TensorEnvironmentConfig


GEN17_STRATEGIES = (
    "static_spiking_policy",
    "oracle_food_reflex",
    "manual_analog_score_policy",
    "manual_spiking_score_policy",
    "manual_spiking_shuffled_reward",
)


@dataclass(frozen=True)
class Gen17Config:
    seeds: tuple[int, ...] = (172, 173, 174)
    agent_count: int = 1_000
    food_count: int = 64
    toxin_count: int = 64
    evaluation_steps: int = 300
    training_steps: int = 1_800
    rollout_steps: int = 30
    reward_delay_steps: int = 12
    progress_reward_scale: float = 0.05
    learning_rate: float = 0.02
    weight_decay: float = 0.0001
    discount: float = 0.99
    gradient_clip: float = 1.0
    minimum_gain_per_1000_steps: float = 0.10
    minimum_control_margin_per_1000_steps: float = 0.10
    maximum_analog_gain_gap_per_1000_steps: float = 0.15
    minimum_spike_density: float = 0.05
    maximum_spike_density: float = 0.40
    maximum_gradient_error: float = 1e-5
    reset_tolerance: float = 1e-6


@dataclass
class Gen17SparseSpikingCreditResult:
    config: dict
    device: str
    records: list[dict]
    summary: list[dict]
    decision: dict

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "gen17_sparse_spiking_credit.json"
        records_path = output / "gen17_sparse_spiking_credit_records.csv"
        summary_path = output / "gen17_sparse_spiking_credit_summary.csv"
        json_path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        _write_csv(records_path, self.records)
        _write_csv(summary_path, self.summary)
        paths = {
            "json": str(json_path),
            "records_csv": str(records_path),
            "summary_csv": str(summary_path),
        }
        if plot:
            plot_path = output / "gen17_sparse_spiking_credit.png"
            plot_gen17_sparse_spiking_credit(self, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def available_gen17_strategies() -> tuple[str, ...]:
    return GEN17_STRATEGIES


def bernoulli_spike_code(sensory, generator):
    """Sample a parameter-matched binary event for each sensory channel."""

    if sensory.ndim != 2 or sensory.shape[1] != 8:
        raise ValueError("sensory must have shape [agents, 8]")
    # TensorEnvironment2D already bounds these drives to [0, 1]. Avoid a
    # per-tick host synchronization merely to revalidate accelerator tensors.
    return (
        torch.rand(sensory.shape, device=sensory.device, generator=generator)
        < sensory
    ).to(sensory.dtype)


def run_gen17_sparse_spiking_credit(
    config: Gen17Config | None = None,
    *,
    device: str = "auto",
    progress_path: str | pathlib.Path | None = None,
) -> Gen17SparseSpikingCreditResult:
    if torch is None:  # pragma: no cover
        raise ImportError("Gen-17 requires PyTorch")
    cfg = config or Gen17Config()
    _validate_config(cfg)
    resolved = resolve_device(device)
    records: list[dict] = []
    for seed in cfg.seeds:
        seed_everything(seed)
        initial_state = LinearRewardPolicy().state_dict()
        for strategy in GEN17_STRATEGIES:
            seed_everything(seed)
            policy = LinearRewardPolicy().to(resolved)
            policy.load_state_dict(initial_state)
            baseline = _evaluate(strategy, policy, cfg, seed, resolved)
            baseline.update({"seed": seed, "strategy": strategy, "phase": "baseline"})
            records.append(baseline)
            training = {
                "mean_training_loss": 0.0,
                "mean_training_reward_per_1000_steps": 0.0,
                "training_updates": 0,
                "maximum_score_gradient_error": 0.0,
                "policy_weight_delta_norm": 0.0,
                "mean_training_spike_density": 0.0,
            }
            if strategy in (
                "manual_analog_score_policy",
                "manual_spiking_score_policy",
                "manual_spiking_shuffled_reward",
            ):
                training = _train_manual_policy(
                    policy,
                    cfg,
                    seed,
                    resolved,
                    spiking=strategy != "manual_analog_score_policy",
                    shuffle_reward=strategy == "manual_spiking_shuffled_reward",
                )
            final = _evaluate(strategy, policy, cfg, seed, resolved)
            final.update({
                "seed": seed,
                "strategy": strategy,
                "phase": "evaluation",
                **training,
            })
            records.append(final)
        if progress_path is not None:
            destination = pathlib.Path(progress_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps({"config": asdict(cfg), "records": records}, indent=2) + "\n",
                encoding="utf-8",
            )
    summary = summarize_gen17(records)
    decision = decide_gen17(records, summary, cfg)
    return Gen17SparseSpikingCreditResult(
        config=asdict(cfg),
        device=str(resolved),
        records=records,
        summary=summary,
        decision=decision,
    )


def _evaluate(strategy, policy, cfg, seed, device):
    environment = _environment(cfg, device)
    environment_generator = _generator(device, 10_000 + seed)
    action_generator = _generator(device, 20_000 + seed)
    spike_generator = _generator(device, 60_000 + seed)
    environment.reset(generator=environment_generator)
    start_fitness = environment.fitness.detach().clone()
    shaped_total = torch.zeros((), device=device)
    spike_total = torch.zeros((), device=device)
    entropy = torch.zeros((), device=device)
    spiking = strategy in (
        "static_spiking_policy",
        "manual_spiking_score_policy",
        "manual_spiking_shuffled_reward",
    )
    with torch.no_grad():
        for _ in range(cfg.evaluation_steps):
            sensory = environment.sensory_tensor()
            if strategy == "oracle_food_reflex":
                action_index = sensory[:, :4].argmax(dim=1)
                entropy = sensory.new_zeros(())
            else:
                features = bernoulli_spike_code(sensory, spike_generator) if spiking else sensory
                if spiking:
                    spike_total.add_(features.sum())
                probabilities = torch.softmax(policy(features), dim=1)
                action_index = _sample_actions(probabilities, action_generator)
                entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=1).mean()
            nearest_before = environment.nearest_objects()
            world = environment.step(_action_vectors(device)[action_index], generator=environment_generator)
            scalar_reward = _scalar_reward(world, nearest_before, cfg)
            shaped_total.add_(scalar_reward.sum())
    sync(device)
    fitness_delta = environment.fitness - start_fitness
    return {
        "steps": cfg.evaluation_steps,
        "mean_net_fitness_per_1000_steps": float(
            1000.0 * fitness_delta.mean().detach().cpu().item() / cfg.evaluation_steps
        ),
        "mean_shaped_reward_per_1000_steps": float(
            1000.0 * shaped_total.detach().cpu().item() / (cfg.agent_count * cfg.evaluation_steps)
        ),
        "mean_policy_entropy": float(entropy.detach().cpu().item()),
        "mean_evaluation_spike_density": (
            float(spike_total.detach().cpu().item()) / (cfg.evaluation_steps * cfg.agent_count * 8)
            if spiking else 0.0
        ),
        "mean_training_loss": None,
        "mean_training_reward_per_1000_steps": None,
        "training_updates": 0,
        "maximum_score_gradient_error": None,
        "policy_weight_delta_norm": None,
        "mean_training_spike_density": None,
    }


def _train_manual_policy(policy, cfg, seed, device, *, spiking, shuffle_reward):
    environment = _environment(cfg, device)
    environment_generator = _generator(device, 30_000 + seed)
    action_generator = _generator(device, 40_000 + seed)
    shuffle_generator = _generator(device, 50_000 + seed)
    spike_generator = _generator(device, 70_000 + seed)
    environment.reset(generator=environment_generator)
    initial_weight = policy.linear.weight.detach().clone()
    loss_values = []
    gradient_errors = []
    reward_total = torch.zeros((), device=device)
    spike_total = torch.zeros((), device=device)
    completed_steps = 0
    while completed_steps < cfg.training_steps:
        chunk = min(cfg.rollout_steps, cfg.training_steps - completed_steps)
        feature_rows = []
        action_rows = []
        log_probability_rows = []
        reward_rows = []
        for _ in range(chunk):
            sensory = environment.sensory_tensor()
            features = bernoulli_spike_code(sensory, spike_generator) if spiking else sensory
            if spiking:
                spike_total.add_(features.sum().detach())
            probabilities = torch.softmax(policy(features), dim=1)
            action_index = _sample_actions(probabilities, action_generator)
            log_probability = probabilities.gather(
                1, action_index.unsqueeze(1)
            ).clamp_min(1e-8).log().squeeze(1)
            nearest_before = environment.nearest_objects()
            world = environment.step(
                _action_vectors(device)[action_index], generator=environment_generator
            )
            scalar_reward = _scalar_reward(world, nearest_before, cfg)
            reward_total.add_(scalar_reward.sum().detach())
            if shuffle_reward:
                scalar_reward = scalar_reward[torch.randperm(
                    scalar_reward.numel(), device=device, generator=shuffle_generator
                )]
            feature_rows.append(features.detach())
            action_rows.append(action_index.detach())
            log_probability_rows.append(log_probability)
            reward_rows.append(scalar_reward.detach())
        feature_trace = torch.stack(feature_rows)
        action_trace = torch.stack(action_rows)
        rewards = torch.stack(reward_rows)
        returns = discounted_returns(rewards, cfg.discount)
        returns = (returns - returns.mean()) / returns.std(unbiased=False).clamp_min(1e-6)
        loss = -(torch.stack(log_probability_rows) * returns).mean()
        gradient_errors.append(score_gradient_parity(
            policy, feature_trace, action_trace, returns
        ))
        weight_ascent, bias_ascent = manual_score_gradients(
            policy, feature_trace, action_trace, returns
        )
        scale = _gradient_scale((weight_ascent, bias_ascent), cfg.gradient_clip)
        with torch.no_grad():
            policy.linear.weight.mul_(1.0 - cfg.learning_rate * cfg.weight_decay)
            policy.linear.bias.mul_(1.0 - cfg.learning_rate * cfg.weight_decay)
            policy.linear.weight.add_(weight_ascent, alpha=cfg.learning_rate * scale)
            policy.linear.bias.add_(bias_ascent, alpha=cfg.learning_rate * scale)
        loss_values.append(float(loss.detach().cpu().item()))
        completed_steps += chunk
    sync(device)
    return {
        "mean_training_loss": sum(loss_values) / len(loss_values),
        "mean_training_reward_per_1000_steps": (
            1000.0 * reward_total.detach().cpu().item() / (cfg.agent_count * cfg.training_steps)
        ),
        "training_updates": len(loss_values),
        "maximum_score_gradient_error": max(gradient_errors, default=0.0),
        "policy_weight_delta_norm": float(
            (policy.linear.weight.detach() - initial_weight).norm().cpu().item()
        ),
        "mean_training_spike_density": (
            float(spike_total.detach().cpu().item()) / (cfg.training_steps * cfg.agent_count * 8)
            if spiking else 0.0
        ),
    }


def summarize_gen17(records: Sequence[dict]) -> list[dict]:
    rows = []
    for strategy in GEN17_STRATEGIES:
        selected = [row for row in records if row["strategy"] == strategy]
        baseline = [row for row in selected if row["phase"] == "baseline"]
        final = [row for row in selected if row["phase"] == "evaluation"]
        gains = {
            int(end["seed"]): float(end["mean_net_fitness_per_1000_steps"])
            - float(next(start for start in baseline if start["seed"] == end["seed"])["mean_net_fitness_per_1000_steps"])
            for end in final
        }
        rows.append({
            "strategy": strategy,
            "runs": len(final),
            "mean_baseline_fitness_per_1000_steps": _mean(baseline, "mean_net_fitness_per_1000_steps"),
            "mean_final_fitness_per_1000_steps": _mean(final, "mean_net_fitness_per_1000_steps"),
            "mean_fitness_gain_per_1000_steps": sum(gains.values()) / len(gains),
            "positive_gain_seed_count": sum(value > 0.0 for value in gains.values()),
            "mean_final_shaped_reward_per_1000_steps": _mean(final, "mean_shaped_reward_per_1000_steps"),
            "mean_final_policy_entropy": _mean(final, "mean_policy_entropy"),
            "mean_evaluation_spike_density": _mean(final, "mean_evaluation_spike_density"),
            "mean_training_spike_density": _mean_optional(final, "mean_training_spike_density"),
            "maximum_score_gradient_error": _max_optional(final, "maximum_score_gradient_error"),
            "mean_policy_weight_delta_norm": _mean_optional(final, "policy_weight_delta_norm"),
        })
    return rows


def decide_gen17(records: Sequence[dict], summary: Sequence[dict], config: Gen17Config) -> dict:
    by_name = {row["strategy"]: row for row in summary}
    static = by_name["static_spiking_policy"]
    oracle = by_name["oracle_food_reflex"]
    analog = by_name["manual_analog_score_policy"]
    spiking = by_name["manual_spiking_score_policy"]
    shuffled = by_name["manual_spiking_shuffled_reward"]
    analog_gains = _gain_by_seed(records, "manual_analog_score_policy")
    spiking_gains = _gain_by_seed(records, "manual_spiking_score_policy")
    analog_qualified = sum(
        value >= config.minimum_gain_per_1000_steps for value in analog_gains.values()
    )
    spiking_qualified = sum(
        value >= config.minimum_gain_per_1000_steps for value in spiking_gains.values()
    )
    reset_gate = abs(float(static["mean_fitness_gain_per_1000_steps"])) <= config.reset_tolerance
    oracle_gate = oracle["mean_final_fitness_per_1000_steps"] > static["mean_final_fitness_per_1000_steps"]
    analog_gate = (
        analog["mean_fitness_gain_per_1000_steps"] >= config.minimum_gain_per_1000_steps
        and analog_qualified >= 2
    )
    spiking_gain_gate = (
        spiking["mean_fitness_gain_per_1000_steps"] >= config.minimum_gain_per_1000_steps
        and spiking_qualified >= 2
    )
    analog_gain_gap = float(analog["mean_fitness_gain_per_1000_steps"]) - float(
        spiking["mean_fitness_gain_per_1000_steps"]
    )
    translation_gate = analog_gain_gap <= config.maximum_analog_gain_gap_per_1000_steps
    gradient_gate = float(spiking["maximum_score_gradient_error"] or 0.0) <= config.maximum_gradient_error
    evaluation_density = float(spiking["mean_evaluation_spike_density"])
    training_density = float(spiking["mean_training_spike_density"] or 0.0)
    activity_gate = all(
        config.minimum_spike_density <= value <= config.maximum_spike_density
        for value in (evaluation_density, training_density)
    )
    static_margin = float(spiking["mean_final_fitness_per_1000_steps"]) - float(
        static["mean_final_fitness_per_1000_steps"]
    )
    shuffled_margin = float(spiking["mean_final_fitness_per_1000_steps"]) - float(
        shuffled["mean_final_fitness_per_1000_steps"]
    )
    spiking_final = _final_by_seed(records, "manual_spiking_score_policy")
    shuffled_final = _final_by_seed(records, "manual_spiking_shuffled_reward")
    identity_seed_count = sum(
        spiking_final[seed] - shuffled_final[seed] >= config.minimum_control_margin_per_1000_steps
        for seed in spiking_final
    )
    identity_gate = (
        min(static_margin, shuffled_margin) >= config.minimum_control_margin_per_1000_steps
        and identity_seed_count >= 2
    )
    passed = all((
        reset_gate, oracle_gate, analog_gate, spiking_gain_gate,
        translation_gate, gradient_gate, activity_gate, identity_gate,
    ))
    return {
        "status": "pass" if passed else "stop",
        "identical_reset_gate": reset_gate,
        "oracle_positive_control": oracle_gate,
        "analog_reference_gate": analog_gate,
        "spiking_gain_gate": spiking_gain_gate,
        "spiking_translation_gate": translation_gate,
        "manual_gradient_parity_gate": gradient_gate,
        "spike_activity_gate": activity_gate,
        "reward_identity_gate": identity_gate,
        "analog_qualified_gain_seed_count": analog_qualified,
        "spiking_qualified_gain_seed_count": spiking_qualified,
        "analog_minus_spiking_gain_per_1000_steps": analog_gain_gap,
        "maximum_manual_gradient_error": float(spiking["maximum_score_gradient_error"] or 0.0),
        "mean_evaluation_spike_density": evaluation_density,
        "mean_training_spike_density": training_density,
        "spiking_margin_vs_static_per_1000_steps": static_margin,
        "spiking_margin_vs_shuffled_per_1000_steps": shuffled_margin,
        "reward_identity_seed_count": identity_seed_count,
        "next_milestone": (
            "replicate_sparse_spiking_credit_before_memory"
            if passed else "reject_or_redesign_sparse_spiking_translation"
        ),
    }


def plot_gen17_sparse_spiking_credit(
    result: Gen17SparseSpikingCreditResult, path: str | pathlib.Path
) -> None:
    import matplotlib.pyplot as plt

    labels = [row["strategy"].replace("_", "\n") for row in result.summary]
    baseline = [row["mean_baseline_fitness_per_1000_steps"] for row in result.summary]
    final = [row["mean_final_fitness_per_1000_steps"] for row in result.summary]
    figure, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
    x = list(range(len(labels)))
    axes[0].bar([value - 0.2 for value in x], baseline, width=0.4, label="identical-reset baseline")
    axes[0].bar([value + 0.2 for value in x], final, width=0.4, label="identical-reset final")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Net collision fitness / 1,000 steps")
    axes[0].set_title("Gen-17 parameter-matched sparse-spiking reward credit")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()
    axes[1].bar(
        labels,
        [100.0 * row["mean_evaluation_spike_density"] for row in result.summary],
        color="#bd3d3a",
    )
    axes[1].set_ylabel("Evaluation event density (%)")
    axes[1].set_title("Binary sensory-event activity")
    axes[1].grid(axis="y", alpha=0.25)
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _gradient_scale(gradients, maximum_norm):
    squared = sum(gradient.square().sum() for gradient in gradients)
    norm = squared.sqrt()
    return float(torch.clamp(maximum_norm / norm.clamp_min(1e-12), max=1.0).detach().cpu().item())


def _final_by_seed(records, strategy):
    return {
        int(row["seed"]): float(row["mean_net_fitness_per_1000_steps"])
        for row in records
        if row["strategy"] == strategy and row["phase"] == "evaluation"
    }


def _gain_by_seed(records, strategy):
    baseline = {
        int(row["seed"]): float(row["mean_net_fitness_per_1000_steps"])
        for row in records
        if row["strategy"] == strategy and row["phase"] == "baseline"
    }
    final = _final_by_seed(records, strategy)
    return {seed: final[seed] - baseline[seed] for seed in final}


def _environment(cfg, device):
    return TensorEnvironment2D(
        TensorEnvironmentConfig(
            agent_count=cfg.agent_count,
            food_count=cfg.food_count,
            toxin_count=cfg.toxin_count,
            reward_delay_steps=cfg.reward_delay_steps,
        ),
        device=device,
    )


def _scalar_reward(world, nearest_before, cfg):
    food_progress = nearest_before["nearest_food_dist"] - world["nearest_food_dist"]
    toxin_progress = world["nearest_toxin_dist"] - nearest_before["nearest_toxin_dist"]
    return world["reward"] - world["punishment"] + cfg.progress_reward_scale * (
        food_progress + toxin_progress
    )


def _sample_actions(probabilities, generator):
    noise = -torch.log(-torch.log(torch.rand(
        probabilities.shape, device=probabilities.device, generator=generator
    ).clamp_(1e-6, 1.0 - 1e-6)))
    return (torch.log(probabilities.clamp_min(1e-8)) + noise).argmax(dim=1)


def _action_vectors(device):
    return torch.tensor(
        ((0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)),
        device=device,
    )


def _generator(device, seed):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _mean(rows, key):
    return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0


def _mean_optional(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def _max_optional(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return max(values) if values else None


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validate_config(config):
    if config.agent_count <= 0:
        raise ValueError("agent_count must be positive")
    if config.training_steps <= 0 or config.evaluation_steps <= 0:
        raise ValueError("step counts must be positive")
    if config.rollout_steps <= config.reward_delay_steps:
        raise ValueError("rollout_steps must exceed reward_delay_steps")
    if not 0.0 < config.discount <= 1.0:
        raise ValueError("discount must be in (0, 1]")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if not 0.0 <= config.minimum_spike_density < config.maximum_spike_density <= 1.0:
        raise ValueError("spike-density bounds must satisfy 0 <= minimum < maximum <= 1")
