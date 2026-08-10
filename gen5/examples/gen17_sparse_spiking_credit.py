"""Gen-17 sparse-spiking local reward-credit CLI."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5 import (  # noqa: E402
    Gen17Config,
    available_gen17_strategies,
    run_gen17_sparse_spiking_credit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preregistered Gen-17 sparse-spiking credit translation."
    )
    parser.add_argument("--list-strategies", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[172, 173, 174])
    parser.add_argument("--agent-count", type=int, default=1000)
    parser.add_argument("--food-count", type=int, default=64)
    parser.add_argument("--toxin-count", type=int, default=64)
    parser.add_argument("--evaluation-steps", type=int, default=300)
    parser.add_argument("--training-steps", type=int, default=1800)
    parser.add_argument("--rollout-steps", type=int, default=30)
    parser.add_argument("--reward-delay-steps", type=int, default=12)
    parser.add_argument("--progress-reward-scale", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--minimum-gain", type=float, default=0.10)
    parser.add_argument("--minimum-control-margin", type=float, default=0.10)
    parser.add_argument("--maximum-analog-gain-gap", type=float, default=0.15)
    parser.add_argument("--minimum-spike-density", type=float, default=0.05)
    parser.add_argument("--maximum-spike-density", type=float, default=0.40)
    parser.add_argument("--maximum-gradient-error", type=float, default=1e-5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="gen5_outputs/gen17_sparse_spiking_credit_cuda")
    parser.add_argument("--progress-path", default=None)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_strategies:
        print("\n".join(available_gen17_strategies()))
        return
    config = Gen17Config(
        seeds=tuple(args.seeds),
        agent_count=args.agent_count,
        food_count=args.food_count,
        toxin_count=args.toxin_count,
        evaluation_steps=args.evaluation_steps,
        training_steps=args.training_steps,
        rollout_steps=args.rollout_steps,
        reward_delay_steps=args.reward_delay_steps,
        progress_reward_scale=args.progress_reward_scale,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        discount=args.discount,
        gradient_clip=args.gradient_clip,
        minimum_gain_per_1000_steps=args.minimum_gain,
        minimum_control_margin_per_1000_steps=args.minimum_control_margin,
        maximum_analog_gain_gap_per_1000_steps=args.maximum_analog_gain_gap,
        minimum_spike_density=args.minimum_spike_density,
        maximum_spike_density=args.maximum_spike_density,
        maximum_gradient_error=args.maximum_gradient_error,
    )
    progress_path = args.progress_path or str(
        pathlib.Path(args.output_dir) / "gen17_sparse_spiking_credit_progress.json"
    )
    result = run_gen17_sparse_spiking_credit(
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
