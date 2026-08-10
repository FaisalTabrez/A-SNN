"""Gen-16 matched autograd versus local score-function reward credit."""

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

from .gen15_reward_baseline import discounted_returns
from .runtime import resolve_device, seed_everything, sync
from .tensor_environment import TensorEnvironment2D, TensorEnvironmentConfig


GEN16_STRATEGIES = (
    "static_linear_policy",
    "oracle_food_reflex",
    "autograd_score_policy",
    "manual_local_score_policy",
    "manual_local_shuffled_reward",
)


@dataclass(frozen=True)
class Gen16Config:
    seeds: tuple[int, ...] = (169, 170, 171)
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
    maximum_autograd_gap_per_1000_steps: float = 0.25
    maximum_gradient_error: float = 1e-5
    reset_tolerance: float = 1e-6


@dataclass
class Gen16LocalScoreCreditResult:
    config: dict
    device: str
    records: list[dict]
    summary: list[dict]
    decision: dict

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "gen16_local_score_credit.json"
        records_path = output / "gen16_local_score_credit_records.csv"
        summary_path = output / "gen16_local_score_credit_summary.csv"
        json_path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        _write_csv(records_path, self.records)
        _write_csv(summary_path, self.summary)
        paths = {
            "json": str(json_path),
            "records_csv": str(records_path),
            "summary_csv": str(summary_path),
        }
        if plot:
            plot_path = output / "gen16_local_score_credit.png"
            plot_gen16_local_score_credit(self, plot_path)
            paths["plot"] = str(plot_path)
        return paths


class LinearRewardPolicy(nn.Module):
    """Eight sensory channels mapped directly to four action logits."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 4)

    def forward(self, sensory):
        return self.linear(sensory)


def available_gen16_strategies() -> tuple[str, ...]:
    return GEN16_STRATEGIES


def manual_score_gradients(policy, sensory, action_index, returns):
    """Return local ascent directions for ``return * grad(log policy)``.

    The action-centred post-synaptic factor ``one_hot(action) - probability``
    is multiplied by the presynaptic sensory value and the global scalar
    return. No target action or autograd gradient enters this calculation.
    """

    probabilities = torch.softmax(policy(sensory), dim=-1)
    chosen = torch.nn.functional.one_hot(action_index, num_classes=4).to(probabilities.dtype)
    post_factor = chosen - probabilities
    modulated = returns.unsqueeze(-1) * post_factor
    normalizer = float(sensory.shape[0] * sensory.shape[1])
    weight_ascent = torch.einsum("tao,tai->oi", modulated, sensory) / normalizer
    bias_ascent = modulated.sum(dim=(0, 1)) / normalizer
    return weight_ascent, bias_ascent


def score_gradient_parity(policy, sensory, action_index, returns) -> float:
    """Measure the maximum error between autograd and the manual local rule."""

    clone = LinearRewardPolicy().to(sensory.device)
    clone.load_state_dict(policy.state_dict())
    probabilities = torch.softmax(clone(sensory), dim=-1)
    log_probability = probabilities.gather(
        2, action_index.unsqueeze(-1)
    ).clamp_min(1e-8).log().squeeze(-1)
    loss = -(log_probability * returns).mean()
    loss.backward()
    weight_ascent, bias_ascent = manual_score_gradients(
        policy, sensory, action_index, returns
    )
    weight_error = (clone.linear.weight.grad + weight_ascent).abs().max()
    bias_error = (clone.linear.bias.grad + bias_ascent).abs().max()
    return float(torch.maximum(weight_error, bias_error).detach().cpu().item())


def run_gen16_local_score_credit(
    config: Gen16Config | None = None,
    *,
    device: str = "auto",
    progress_path: str | pathlib.Path | None = None,
) -> Gen16LocalScoreCreditResult:
    if torch is None:  # pragma: no cover
        raise ImportError("Gen-16 requires PyTorch")
    cfg = config or Gen16Config()
    _validate_config(cfg)
    resolved = resolve_device(device)
    records: list[dict] = []
    for seed in cfg.seeds:
        seed_everything(seed)
        initial_state = LinearRewardPolicy().state_dict()
        for strategy in GEN16_STRATEGIES:
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
            }
            if strategy in (
                "autograd_score_policy",
                "manual_local_score_policy",
                "manual_local_shuffled_reward",
            ):
                training = _train_policy(
                    policy,
                    cfg,
                    seed,
                    resolved,
                    manual=strategy != "autograd_score_policy",
                    shuffle_reward=strategy == "manual_local_shuffled_reward",
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
    summary = summarize_gen16(records)
    decision = decide_gen16(records, summary, cfg)
    return Gen16LocalScoreCreditResult(
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
    entropy = torch.zeros((), device=device)
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
        "maximum_score_gradient_error": None,
        "policy_weight_delta_norm": None,
    }


def _train_policy(policy, cfg, seed, device, *, manual, shuffle_reward):
    environment = _environment(cfg, device)
    environment_generator = _generator(device, 30_000 + seed)
    action_generator = _generator(device, 40_000 + seed)
    shuffle_generator = _generator(device, 50_000 + seed)
    environment.reset(generator=environment_generator)
    initial_weight = policy.linear.weight.detach().clone()
    optimizer = None if manual else torch.optim.SGD(
        policy.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    loss_values = []
    gradient_errors = []
    reward_total = 0.0
    completed_steps = 0
    while completed_steps < cfg.training_steps:
        chunk = min(cfg.rollout_steps, cfg.training_steps - completed_steps)
        sensory_rows = []
        action_rows = []
        log_probability_rows = []
        reward_rows = []
        for _ in range(chunk):
            sensory = environment.sensory_tensor()
            probabilities = torch.softmax(policy(sensory), dim=1)
            action_index = _sample_actions(probabilities, action_generator)
            log_probability = probabilities.gather(
                1, action_index.unsqueeze(1)
            ).clamp_min(1e-8).log().squeeze(1)
            nearest_before = environment.nearest_objects()
            world = environment.step(
                _action_vectors(device)[action_index], generator=environment_generator
            )
            scalar_reward = _scalar_reward(world, nearest_before, cfg)
            reward_total += float(scalar_reward.sum().detach().cpu().item())
            if shuffle_reward:
                scalar_reward = scalar_reward[torch.randperm(
                    scalar_reward.numel(), device=device, generator=shuffle_generator
                )]
            sensory_rows.append(sensory.detach())
            action_rows.append(action_index.detach())
            log_probability_rows.append(log_probability)
            reward_rows.append(scalar_reward.detach())
        sensory_trace = torch.stack(sensory_rows)
        action_trace = torch.stack(action_rows)
        rewards = torch.stack(reward_rows)
        returns = discounted_returns(rewards, cfg.discount)
        returns = (returns - returns.mean()) / returns.std(unbiased=False).clamp_min(1e-6)
        loss = -(torch.stack(log_probability_rows) * returns).mean()
        if manual:
            gradient_errors.append(score_gradient_parity(
                policy, sensory_trace, action_trace, returns
            ))
            weight_ascent, bias_ascent = manual_score_gradients(
                policy, sensory_trace, action_trace, returns
            )
            scale = _gradient_scale((weight_ascent, bias_ascent), cfg.gradient_clip)
            with torch.no_grad():
                policy.linear.weight.mul_(1.0 - cfg.learning_rate * cfg.weight_decay)
                policy.linear.bias.mul_(1.0 - cfg.learning_rate * cfg.weight_decay)
                policy.linear.weight.add_(weight_ascent, alpha=cfg.learning_rate * scale)
                policy.linear.bias.add_(bias_ascent, alpha=cfg.learning_rate * scale)
        else:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.gradient_clip)
            optimizer.step()
        loss_values.append(float(loss.detach().cpu().item()))
        completed_steps += chunk
    sync(device)
    return {
        "mean_training_loss": sum(loss_values) / len(loss_values),
        "mean_training_reward_per_1000_steps": (
            1000.0 * reward_total / (cfg.agent_count * cfg.training_steps)
        ),
        "training_updates": len(loss_values),
        "maximum_score_gradient_error": max(gradient_errors, default=0.0),
        "policy_weight_delta_norm": float(
            (policy.linear.weight.detach() - initial_weight).norm().cpu().item()
        ),
    }


def _gradient_scale(gradients, maximum_norm):
    squared = sum(gradient.square().sum() for gradient in gradients)
    norm = squared.sqrt()
    return float(torch.clamp(maximum_norm / norm.clamp_min(1e-12), max=1.0).detach().cpu().item())


def summarize_gen16(records: Sequence[dict]) -> list[dict]:
    rows = []
    for strategy in GEN16_STRATEGIES:
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
            "mean_training_loss": _mean_optional(final, "mean_training_loss"),
            "mean_training_reward_per_1000_steps": _mean_optional(final, "mean_training_reward_per_1000_steps"),
            "maximum_score_gradient_error": _max_optional(final, "maximum_score_gradient_error"),
            "mean_policy_weight_delta_norm": _mean_optional(final, "policy_weight_delta_norm"),
        })
    return rows


def decide_gen16(records: Sequence[dict], summary: Sequence[dict], config: Gen16Config) -> dict:
    by_name = {row["strategy"]: row for row in summary}
    static = by_name["static_linear_policy"]
    oracle = by_name["oracle_food_reflex"]
    autograd = by_name["autograd_score_policy"]
    local = by_name["manual_local_score_policy"]
    shuffled = by_name["manual_local_shuffled_reward"]
    autograd_gains = _gain_by_seed(records, "autograd_score_policy")
    local_gains = _gain_by_seed(records, "manual_local_score_policy")
    autograd_qualified_seeds = sum(
        value >= config.minimum_gain_per_1000_steps for value in autograd_gains.values()
    )
    local_qualified_seeds = sum(
        value >= config.minimum_gain_per_1000_steps for value in local_gains.values()
    )
    reset_gate = abs(float(static["mean_fitness_gain_per_1000_steps"])) <= config.reset_tolerance
    oracle_gate = oracle["mean_final_fitness_per_1000_steps"] > static["mean_final_fitness_per_1000_steps"]
    autograd_gate = (
        autograd["mean_fitness_gain_per_1000_steps"] >= config.minimum_gain_per_1000_steps
        and autograd_qualified_seeds >= 2
    )
    local_gain_gate = (
        local["mean_fitness_gain_per_1000_steps"] >= config.minimum_gain_per_1000_steps
        and local_qualified_seeds >= 2
    )
    autograd_gap = abs(
        float(local["mean_final_fitness_per_1000_steps"])
        - float(autograd["mean_final_fitness_per_1000_steps"])
    )
    equivalence_gate = autograd_gap <= config.maximum_autograd_gap_per_1000_steps
    gradient_gate = float(local["maximum_score_gradient_error"] or 0.0) <= config.maximum_gradient_error
    static_margin = float(local["mean_final_fitness_per_1000_steps"]) - float(static["mean_final_fitness_per_1000_steps"])
    shuffled_margin = float(local["mean_final_fitness_per_1000_steps"]) - float(shuffled["mean_final_fitness_per_1000_steps"])
    local_final = _final_by_seed(records, "manual_local_score_policy")
    shuffled_final = _final_by_seed(records, "manual_local_shuffled_reward")
    identity_seed_count = sum(
        local_final[seed] - shuffled_final[seed] >= config.minimum_control_margin_per_1000_steps
        for seed in local_final
    )
    identity_gate = (
        min(static_margin, shuffled_margin) >= config.minimum_control_margin_per_1000_steps
        and identity_seed_count >= 2
    )
    passed = all((
        reset_gate, oracle_gate, autograd_gate, local_gain_gate,
        equivalence_gate, gradient_gate, identity_gate,
    ))
    return {
        "status": "pass" if passed else "stop",
        "identical_reset_gate": reset_gate,
        "oracle_positive_control": oracle_gate,
        "autograd_learnability_gate": autograd_gate,
        "local_gain_gate": local_gain_gate,
        "autograd_qualified_gain_seed_count": autograd_qualified_seeds,
        "local_qualified_gain_seed_count": local_qualified_seeds,
        "autograd_equivalence_gate": equivalence_gate,
        "manual_gradient_parity_gate": gradient_gate,
        "reward_identity_gate": identity_gate,
        "local_autograd_final_gap_per_1000_steps": autograd_gap,
        "maximum_manual_gradient_error": float(local["maximum_score_gradient_error"] or 0.0),
        "local_margin_vs_static_per_1000_steps": static_margin,
        "local_margin_vs_shuffled_per_1000_steps": shuffled_margin,
        "reward_identity_seed_count": identity_seed_count,
        "next_milestone": (
            "translate_validated_local_score_rule_to_sparse_spikes"
            if passed else "reject_or_redesign_local_score_credit"
        ),
    }


def plot_gen16_local_score_credit(
    result: Gen16LocalScoreCreditResult, path: str | pathlib.Path
) -> None:
    import matplotlib.pyplot as plt

    labels = [row["strategy"].replace("_", "\n") for row in result.summary]
    baseline = [row["mean_baseline_fitness_per_1000_steps"] for row in result.summary]
    final = [row["mean_final_fitness_per_1000_steps"] for row in result.summary]
    figure, axis = plt.subplots(figsize=(13, 6), constrained_layout=True)
    x = list(range(len(labels)))
    axis.bar([value - 0.2 for value in x], baseline, width=0.4, label="identical-reset baseline")
    axis.bar([value + 0.2 for value in x], final, width=0.4, label="identical-reset final")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Net collision fitness / 1,000 steps")
    axis.set_title("Gen-16 local score-function credit equivalence")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


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
