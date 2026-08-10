"""Gen-15 matched reward-learning baseline and reset-evaluation diagnostic."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import pathlib
from typing import Sequence

try:  # pragma: no cover
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None

from .runtime import resolve_device, seed_everything, sync
from .tensor_environment import TensorEnvironment2D, TensorEnvironmentConfig


GEN15_STRATEGIES = (
    "static_random",
    "oracle_food_reflex",
    "reinforce_shared_policy",
    "reinforce_shuffled_reward",
)


@dataclass(frozen=True)
class Gen15Config:
    seeds: tuple[int, ...] = (166, 167, 168)
    agent_count: int = 1_000
    food_count: int = 64
    toxin_count: int = 64
    evaluation_steps: int = 300
    training_steps: int = 1_800
    rollout_steps: int = 30
    reward_delay_steps: int = 12
    progress_reward_scale: float = 0.05
    hidden_units: int = 32
    learning_rate: float = 0.003
    weight_decay: float = 0.0001
    discount: float = 0.99
    entropy_weight: float = 0.01
    gradient_clip: float = 1.0
    minimum_gain_per_1000_steps: float = 0.10
    minimum_control_margin_per_1000_steps: float = 0.10
    reset_tolerance: float = 1e-6


@dataclass
class Gen15RewardBaselineResult:
    config: dict
    device: str
    records: list[dict]
    summary: list[dict]
    decision: dict

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "gen15_reward_baseline.json"
        records_path = output / "gen15_reward_baseline_records.csv"
        summary_path = output / "gen15_reward_baseline_summary.csv"
        json_path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        _write_csv(records_path, self.records)
        _write_csv(summary_path, self.summary)
        paths = {
            "json": str(json_path),
            "records_csv": str(records_path),
            "summary_csv": str(summary_path),
        }
        if plot:
            plot_path = output / "gen15_reward_baseline.png"
            plot_gen15_reward_baseline(self, plot_path)
            paths["plot"] = str(plot_path)
        return paths


class SharedRewardPolicy(nn.Module):
    def __init__(self, hidden_units: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(8, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, 4),
        )

    def forward(self, sensory):
        return self.network(sensory)


def available_gen15_strategies() -> tuple[str, ...]:
    return GEN15_STRATEGIES


def discounted_returns(reward, discount: float):
    """Return time-major discounted returns for a reward tensor ``[T, A]``."""

    returns = torch.zeros_like(reward)
    running = torch.zeros_like(reward[0])
    for index in range(reward.shape[0] - 1, -1, -1):
        running = reward[index] + discount * running
        returns[index] = running
    return returns


def run_gen15_reward_baseline(
    config: Gen15Config | None = None,
    *,
    device: str = "auto",
    progress_path: str | pathlib.Path | None = None,
) -> Gen15RewardBaselineResult:
    if torch is None:  # pragma: no cover
        raise ImportError("Gen-15 requires PyTorch")
    cfg = config or Gen15Config()
    _validate_config(cfg)
    resolved = resolve_device(device)
    records: list[dict] = []
    for seed in cfg.seeds:
        seed_everything(seed)
        initial_state = SharedRewardPolicy(cfg.hidden_units).state_dict()
        for strategy in GEN15_STRATEGIES:
            seed_everything(seed)
            policy = SharedRewardPolicy(cfg.hidden_units).to(resolved)
            policy.load_state_dict(initial_state)
            baseline = _evaluate(strategy, policy, cfg, seed, resolved)
            baseline.update({"seed": seed, "strategy": strategy, "phase": "baseline"})
            records.append(baseline)
            training = {
                "mean_training_loss": 0.0,
                "mean_training_reward_per_1000_steps": 0.0,
                "training_updates": 0,
            }
            if strategy in ("reinforce_shared_policy", "reinforce_shuffled_reward"):
                training = _train_policy(
                    policy,
                    cfg,
                    seed,
                    resolved,
                    shuffle_reward=strategy == "reinforce_shuffled_reward",
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
    summary = summarize_gen15(records)
    decision = decide_gen15(summary, cfg)
    return Gen15RewardBaselineResult(
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
    environment.reset(generator=environment_generator)
    start_fitness = environment.fitness.detach().clone()
    shaped_total = 0.0
    with torch.no_grad():
        for _ in range(cfg.evaluation_steps):
            sensory = environment.sensory_tensor()
            if strategy == "oracle_food_reflex":
                action_index = sensory[:, :4].argmax(dim=1)
                entropy = sensory.new_zeros(())
            else:
                probabilities = torch.softmax(policy(sensory), dim=1)
                action_index = _sample_actions(probabilities, action_generator)
                entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=1).mean()
            nearest_before = environment.nearest_objects()
            world = environment.step(_action_vectors(device)[action_index], generator=environment_generator)
            scalar_reward = _scalar_reward(world, nearest_before, cfg)
            shaped_total += float(scalar_reward.sum().detach().cpu().item())
    sync(device)
    fitness_delta = environment.fitness - start_fitness
    return {
        "steps": cfg.evaluation_steps,
        "mean_net_fitness_per_1000_steps": float(
            1000.0 * fitness_delta.mean().detach().cpu().item() / cfg.evaluation_steps
        ),
        "mean_shaped_reward_per_1000_steps": float(
            1000.0 * shaped_total / (cfg.agent_count * cfg.evaluation_steps)
        ),
        "mean_policy_entropy": float(entropy.detach().cpu().item()),
        "mean_training_loss": None,
        "mean_training_reward_per_1000_steps": None,
        "training_updates": 0,
    }


def _train_policy(policy, cfg, seed, device, *, shuffle_reward):
    environment = _environment(cfg, device)
    environment_generator = _generator(device, 30_000 + seed)
    action_generator = _generator(device, 40_000 + seed)
    shuffle_generator = _generator(device, 50_000 + seed)
    environment.reset(generator=environment_generator)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    loss_values = []
    reward_total = 0.0
    completed_steps = 0
    while completed_steps < cfg.training_steps:
        chunk = min(cfg.rollout_steps, cfg.training_steps - completed_steps)
        log_prob_rows = []
        entropy_rows = []
        reward_rows = []
        for _ in range(chunk):
            sensory = environment.sensory_tensor()
            probabilities = torch.softmax(policy(sensory), dim=1)
            action_index = _sample_actions(probabilities, action_generator)
            log_probability = probabilities.gather(1, action_index.unsqueeze(1)).clamp_min(1e-8).log().squeeze(1)
            entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=1)
            nearest_before = environment.nearest_objects()
            world = environment.step(_action_vectors(device)[action_index], generator=environment_generator)
            scalar_reward = _scalar_reward(world, nearest_before, cfg)
            reward_total += float(scalar_reward.sum().detach().cpu().item())
            if shuffle_reward:
                scalar_reward = scalar_reward[torch.randperm(
                    scalar_reward.numel(), device=device, generator=shuffle_generator
                )]
            log_prob_rows.append(log_probability)
            entropy_rows.append(entropy)
            reward_rows.append(scalar_reward.detach())
        rewards = torch.stack(reward_rows)
        returns = discounted_returns(rewards, cfg.discount)
        returns = (returns - returns.mean()) / returns.std(unbiased=False).clamp_min(1e-6)
        log_probabilities = torch.stack(log_prob_rows)
        entropies = torch.stack(entropy_rows)
        loss = -(log_probabilities * returns).mean() - cfg.entropy_weight * entropies.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.gradient_clip)
        optimizer.step()
        loss_values.append(float(loss.detach().cpu().item()))
        completed_steps += chunk
    sync(device)
    return {
        "mean_training_loss": sum(loss_values) / len(loss_values),
        "mean_training_reward_per_1000_steps": 1000.0 * reward_total / (cfg.agent_count * cfg.training_steps),
        "training_updates": len(loss_values),
    }


def summarize_gen15(records: Sequence[dict]) -> list[dict]:
    rows = []
    for strategy in GEN15_STRATEGIES:
        selected = [row for row in records if row["strategy"] == strategy]
        baseline = [row for row in selected if row["phase"] == "baseline"]
        final = [row for row in selected if row["phase"] == "evaluation"]
        seed_gains = [
            float(end["mean_net_fitness_per_1000_steps"])
            - float(next(start for start in baseline if start["seed"] == end["seed"])["mean_net_fitness_per_1000_steps"])
            for end in final
        ]
        rows.append({
            "strategy": strategy,
            "runs": len(final),
            "mean_baseline_fitness_per_1000_steps": _mean(baseline, "mean_net_fitness_per_1000_steps"),
            "mean_final_fitness_per_1000_steps": _mean(final, "mean_net_fitness_per_1000_steps"),
            "mean_fitness_gain_per_1000_steps": sum(seed_gains) / len(seed_gains),
            "positive_gain_seed_count": sum(value > 0.0 for value in seed_gains),
            "mean_final_shaped_reward_per_1000_steps": _mean(final, "mean_shaped_reward_per_1000_steps"),
            "mean_final_policy_entropy": _mean(final, "mean_policy_entropy"),
            "mean_training_loss": _mean_optional(final, "mean_training_loss"),
            "mean_training_reward_per_1000_steps": _mean_optional(final, "mean_training_reward_per_1000_steps"),
        })
    return rows


def decide_gen15(summary: Sequence[dict], config: Gen15Config) -> dict:
    by_name = {row["strategy"]: row for row in summary}
    static = by_name["static_random"]
    oracle = by_name["oracle_food_reflex"]
    reinforce = by_name["reinforce_shared_policy"]
    shuffled = by_name["reinforce_shuffled_reward"]
    reset_gate = abs(float(static["mean_fitness_gain_per_1000_steps"])) <= config.reset_tolerance
    oracle_gate = oracle["mean_final_fitness_per_1000_steps"] > static["mean_final_fitness_per_1000_steps"]
    gain_gate = (
        reinforce["mean_fitness_gain_per_1000_steps"] >= config.minimum_gain_per_1000_steps
        and reinforce["positive_gain_seed_count"] >= 2
    )
    static_margin = reinforce["mean_final_fitness_per_1000_steps"] - static["mean_final_fitness_per_1000_steps"]
    shuffled_margin = reinforce["mean_final_fitness_per_1000_steps"] - shuffled["mean_final_fitness_per_1000_steps"]
    identity_gate = min(static_margin, shuffled_margin) >= config.minimum_control_margin_per_1000_steps
    passed = reset_gate and oracle_gate and gain_gate and identity_gate
    return {
        "status": "pass" if passed else "stop",
        "identical_reset_gate": reset_gate,
        "oracle_positive_control": oracle_gate,
        "reinforce_gain_gate": gain_gate,
        "reward_identity_gate": identity_gate,
        "reinforce_margin_vs_static_per_1000_steps": static_margin,
        "reinforce_margin_vs_shuffled_per_1000_steps": shuffled_margin,
        "next_milestone": "derive_local_credit_from_validated_baseline" if passed else "redesign_reward_protocol_before_local_learning",
    }


def plot_gen15_reward_baseline(result: Gen15RewardBaselineResult, path: str | pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    labels = [row["strategy"].replace("_", "\n") for row in result.summary]
    baseline = [row["mean_baseline_fitness_per_1000_steps"] for row in result.summary]
    final = [row["mean_final_fitness_per_1000_steps"] for row in result.summary]
    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    x = list(range(len(labels)))
    axis.bar([value - 0.2 for value in x], baseline, width=0.4, label="identical-reset baseline")
    axis.bar([value + 0.2 for value in x], final, width=0.4, label="identical-reset final")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Net collision fitness / 1,000 steps")
    axis.set_title("Gen-15 matched reward-learning baseline")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


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
    return world["reward"] - world["punishment"] + cfg.progress_reward_scale * (food_progress + toxin_progress)


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
