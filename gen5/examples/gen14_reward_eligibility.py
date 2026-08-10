"""Gen-14 reward-modulated embodied eligibility CLI."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5 import (  # noqa: E402
    Gen14Config,
    available_gen14_strategies,
    run_gen14_reward_eligibility,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preregistered Gen-14 reward-eligibility screen."
    )
    parser.add_argument("--list-strategies", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[163, 164, 165])
    parser.add_argument("--agent-count", type=int, default=10_000)
    parser.add_argument("--food-count", type=int, default=128)
    parser.add_argument("--toxin-count", type=int, default=128)
    parser.add_argument("--baseline-steps", type=int, default=600)
    parser.add_argument("--training-steps", type=int, default=3_600)
    parser.add_argument("--evaluation-steps", type=int, default=600)
    parser.add_argument("--reward-delay-steps", type=int, default=12)
    parser.add_argument("--eligibility-decay", type=float, default=0.95)
    parser.add_argument("--trace-decay", type=float, default=0.90)
    parser.add_argument("--reward-baseline-decay", type=float, default=0.99)
    parser.add_argument("--local-learning-rate", type=float, default=0.02)
    parser.add_argument("--fast-weight-decay", type=float, default=0.0001)
    parser.add_argument("--progress-reward-scale", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.50)
    parser.add_argument("--minimum-gain", type=float, default=0.10)
    parser.add_argument("--minimum-control-margin", type=float, default=0.10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="gen5_outputs/gen14_reward_eligibility_cuda")
    parser.add_argument("--progress-path", default=None)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_strategies:
        print("\n".join(available_gen14_strategies()))
        return
    config = Gen14Config(
        seeds=tuple(args.seeds),
        agent_count=args.agent_count,
        food_count=args.food_count,
        toxin_count=args.toxin_count,
        baseline_steps=args.baseline_steps,
        training_steps=args.training_steps,
        evaluation_steps=args.evaluation_steps,
        reward_delay_steps=args.reward_delay_steps,
        eligibility_decay=args.eligibility_decay,
        trace_decay=args.trace_decay,
        reward_baseline_decay=args.reward_baseline_decay,
        local_learning_rate=args.local_learning_rate,
        fast_weight_decay=args.fast_weight_decay,
        progress_reward_scale=args.progress_reward_scale,
        temperature=args.temperature,
        minimum_gain_per_1000_steps=args.minimum_gain,
        minimum_control_margin_per_1000_steps=args.minimum_control_margin,
    )
    progress_path = args.progress_path or str(
        pathlib.Path(args.output_dir) / "gen14_reward_eligibility_progress.json"
    )
    result = run_gen14_reward_eligibility(
        config, device=args.device, progress_path=progress_path
    )
    paths = result.save(args.output_dir, plot=not args.no_plot)
    paths["progress"] = progress_path
    print(json.dumps({
        "paths": paths,
        "device": result.device,
        "decision": result.decision,
        "summary": result.summary,
    }, indent=2))


if __name__ == "__main__":
    main()
