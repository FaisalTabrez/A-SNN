"""Gen-18 held-out replication of analog local reward credit."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import math
import pathlib
import statistics
from typing import Sequence

try:  # pragma: no cover
    import torch
except Exception:  # pragma: no cover
    torch = None

from .gen16_local_score_credit import LinearRewardPolicy, _evaluate, _train_policy
from .runtime import resolve_device, seed_everything


GEN18_STRATEGIES = (
    "static_linear_policy",
    "oracle_food_reflex",
    "manual_local_score_policy",
    "manual_local_shuffled_reward",
)


@dataclass(frozen=True)
class Gen18Config:
    seeds: tuple[int, ...] = tuple(range(180, 190))
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
    minimum_qualified_seed_count: int = 7
    confidence_z: float = 1.96
    maximum_gradient_error: float = 1e-5
    reset_tolerance: float = 1e-6


@dataclass
class Gen18LocalCreditReplicationResult:
    config: dict
    device: str
    records: list[dict]
    summary: list[dict]
    decision: dict

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "gen18_local_credit_replication.json"
        records_path = output / "gen18_local_credit_replication_records.csv"
        summary_path = output / "gen18_local_credit_replication_summary.csv"
        json_path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        _write_csv(records_path, self.records)
        _write_csv(summary_path, self.summary)
        paths = {
            "json": str(json_path),
            "records_csv": str(records_path),
            "summary_csv": str(summary_path),
        }
        if plot:
            plot_path = output / "gen18_local_credit_replication.png"
            plot_gen18_local_credit_replication(self, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def available_gen18_strategies() -> tuple[str, ...]:
    return GEN18_STRATEGIES


def run_gen18_local_credit_replication(
    config: Gen18Config | None = None,
    *,
    device: str = "auto",
    progress_path: str | pathlib.Path | None = None,
) -> Gen18LocalCreditReplicationResult:
    if torch is None:  # pragma: no cover
        raise ImportError("Gen-18 requires PyTorch")
    cfg = config or Gen18Config()
    _validate_config(cfg)
    resolved = resolve_device(device)
    records: list[dict] = []
    for seed in cfg.seeds:
        seed_everything(seed)
        initial_state = LinearRewardPolicy().state_dict()
        for strategy in GEN18_STRATEGIES:
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
                "manual_local_score_policy",
                "manual_local_shuffled_reward",
            ):
                training = _train_policy(
                    policy,
                    cfg,
                    seed,
                    resolved,
                    manual=True,
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
    summary = summarize_gen18(records, cfg.confidence_z)
    decision = decide_gen18(records, summary, cfg)
    return Gen18LocalCreditReplicationResult(
        config=asdict(cfg),
        device=str(resolved),
        records=records,
        summary=summary,
        decision=decision,
    )


def summarize_gen18(records: Sequence[dict], confidence_z: float = 1.96) -> list[dict]:
    rows = []
    for strategy in GEN18_STRATEGIES:
        selected = [row for row in records if row["strategy"] == strategy]
        baseline = [row for row in selected if row["phase"] == "baseline"]
        final = [row for row in selected if row["phase"] == "evaluation"]
        gains = list(_gain_by_seed(records, strategy).values())
        rows.append({
            "strategy": strategy,
            "runs": len(final),
            "mean_baseline_fitness_per_1000_steps": _mean(baseline, "mean_net_fitness_per_1000_steps"),
            "mean_final_fitness_per_1000_steps": _mean(final, "mean_net_fitness_per_1000_steps"),
            "mean_fitness_gain_per_1000_steps": _mean_values(gains),
            "gain_std_per_1000_steps": _sample_std(gains),
            "gain_ci95_lower_per_1000_steps": confidence_lower_bound(gains, confidence_z),
            "positive_gain_seed_count": sum(value > 0.0 for value in gains),
            "mean_final_shaped_reward_per_1000_steps": _mean(final, "mean_shaped_reward_per_1000_steps"),
            "mean_final_policy_entropy": _mean(final, "mean_policy_entropy"),
            "maximum_score_gradient_error": _max_optional(final, "maximum_score_gradient_error"),
            "mean_policy_weight_delta_norm": _mean_optional(final, "policy_weight_delta_norm"),
        })
    return rows


def decide_gen18(
    records: Sequence[dict], summary: Sequence[dict], config: Gen18Config
) -> dict:
    by_name = {row["strategy"]: row for row in summary}
    static = by_name["static_linear_policy"]
    oracle = by_name["oracle_food_reflex"]
    local = by_name["manual_local_score_policy"]
    local_gains = _gain_by_seed(records, "manual_local_score_policy")
    local_final = _final_by_seed(records, "manual_local_score_policy")
    static_final = _final_by_seed(records, "static_linear_policy")
    shuffled_final = _final_by_seed(records, "manual_local_shuffled_reward")
    static_margins = [local_final[seed] - static_final[seed] for seed in local_final]
    identity_margins = [local_final[seed] - shuffled_final[seed] for seed in local_final]
    qualified_gain_count = sum(
        value >= config.minimum_gain_per_1000_steps for value in local_gains.values()
    )
    qualified_static_count = sum(
        value >= config.minimum_control_margin_per_1000_steps for value in static_margins
    )
    qualified_identity_count = sum(
        value >= config.minimum_control_margin_per_1000_steps for value in identity_margins
    )
    reset_gate = abs(float(static["mean_fitness_gain_per_1000_steps"])) <= config.reset_tolerance
    oracle_gate = (
        float(oracle["mean_final_fitness_per_1000_steps"])
        > float(static["mean_final_fitness_per_1000_steps"])
    )
    gain_ci_lower = confidence_lower_bound(list(local_gains.values()), config.confidence_z)
    gain_gate = (
        float(local["mean_fitness_gain_per_1000_steps"])
        >= config.minimum_gain_per_1000_steps
        and qualified_gain_count >= config.minimum_qualified_seed_count
        and gain_ci_lower > 0.0
    )
    static_margin_mean = _mean_values(static_margins)
    static_margin_lower = confidence_lower_bound(static_margins, config.confidence_z)
    static_margin_gate = (
        static_margin_mean >= config.minimum_control_margin_per_1000_steps
        and qualified_static_count >= config.minimum_qualified_seed_count
        and static_margin_lower > 0.0
    )
    identity_margin_mean = _mean_values(identity_margins)
    identity_margin_lower = confidence_lower_bound(identity_margins, config.confidence_z)
    identity_gate = (
        identity_margin_mean >= config.minimum_control_margin_per_1000_steps
        and qualified_identity_count >= config.minimum_qualified_seed_count
        and identity_margin_lower > 0.0
    )
    gradient_gate = float(local["maximum_score_gradient_error"] or 0.0) <= config.maximum_gradient_error
    passed = all((reset_gate, oracle_gate, gain_gate, static_margin_gate, identity_gate, gradient_gate))
    return {
        "status": "pass" if passed else "stop",
        "identical_reset_gate": reset_gate,
        "oracle_positive_control": oracle_gate,
        "replicated_local_gain_gate": gain_gate,
        "replicated_static_margin_gate": static_margin_gate,
        "replicated_reward_identity_gate": identity_gate,
        "manual_gradient_parity_gate": gradient_gate,
        "qualified_gain_seed_count": qualified_gain_count,
        "qualified_static_margin_seed_count": qualified_static_count,
        "qualified_reward_identity_seed_count": qualified_identity_count,
        "local_gain_mean_per_1000_steps": _mean_values(list(local_gains.values())),
        "local_gain_ci95_lower_per_1000_steps": gain_ci_lower,
        "local_margin_vs_static_mean_per_1000_steps": static_margin_mean,
        "local_margin_vs_static_ci95_lower_per_1000_steps": static_margin_lower,
        "local_margin_vs_shuffled_mean_per_1000_steps": identity_margin_mean,
        "local_margin_vs_shuffled_ci95_lower_per_1000_steps": identity_margin_lower,
        "maximum_manual_gradient_error": float(local["maximum_score_gradient_error"] or 0.0),
        "next_milestone": (
            "derive_temporal_spike_encoding_after_bernoulli_rejection"
            if passed else "close_local_reward_credit_program"
        ),
    }


def confidence_lower_bound(values: Sequence[float], z: float = 1.96) -> float:
    """Return a normal-approximation lower confidence bound for the mean."""

    if not values:
        return 0.0
    mean = _mean_values(values)
    if len(values) < 2:
        return mean
    return mean - z * statistics.stdev(values) / math.sqrt(len(values))


def plot_gen18_local_credit_replication(
    result: Gen18LocalCreditReplicationResult, path: str | pathlib.Path
) -> None:
    import matplotlib.pyplot as plt

    labels = [row["strategy"].replace("_", "\n") for row in result.summary]
    baseline = [row["mean_baseline_fitness_per_1000_steps"] for row in result.summary]
    final = [row["mean_final_fitness_per_1000_steps"] for row in result.summary]
    local_gain = _gain_by_seed(result.records, "manual_local_score_policy")
    local_final = _final_by_seed(result.records, "manual_local_score_policy")
    shuffled_final = _final_by_seed(result.records, "manual_local_shuffled_reward")
    seeds = sorted(local_gain)
    identity = [local_final[seed] - shuffled_final[seed] for seed in seeds]

    figure, axes = plt.subplots(2, 1, figsize=(13, 10), constrained_layout=True)
    x = list(range(len(labels)))
    axes[0].bar([value - 0.2 for value in x], baseline, width=0.4, label="baseline")
    axes[0].bar([value + 0.2 for value in x], final, width=0.4, label="final")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Net collision fitness / 1,000 steps")
    axes[0].set_title("Gen-18 held-out analog local-credit replication")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].plot(seeds, [local_gain[seed] for seed in seeds], marker="o", label="local learning gain")
    axes[1].plot(seeds, identity, marker="s", label="local minus shuffled final")
    axes[1].set_xlabel("Held-out seed")
    axes[1].set_ylabel("Fitness difference / 1,000 steps")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

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


def _mean(rows, key):
    return _mean_values([float(row[key]) for row in rows])


def _mean_values(values):
    return sum(values) / len(values) if values else 0.0


def _sample_std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _mean_optional(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return _mean_values(values) if values else None


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
    if len(config.seeds) < 2:
        raise ValueError("Gen-18 requires at least two held-out seeds")
    if len(set(config.seeds)) != len(config.seeds):
        raise ValueError("seeds must be unique")
    if config.minimum_qualified_seed_count > len(config.seeds):
        raise ValueError("minimum_qualified_seed_count cannot exceed the seed count")
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
    if config.confidence_z <= 0.0:
        raise ValueError("confidence_z must be positive")
