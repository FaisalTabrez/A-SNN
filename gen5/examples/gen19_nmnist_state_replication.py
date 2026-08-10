"""Gen-19 N-MNIST residual-state replication CLI."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5 import (  # noqa: E402
    Gen19Config,
    available_gen19_arms,
    run_gen19_nmnist_state_replication,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preregistered Gen-19 N-MNIST residual-state replication."
    )
    parser.add_argument("--list-arms", action="store_true")
    parser.add_argument("--seeds", nargs=3, type=int, default=[190, 191, 192])
    parser.add_argument("--train-samples", type=int, default=0)
    parser.add_argument("--test-samples", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=30)
    parser.add_argument("--spatial-bins", type=int, default=8)
    parser.add_argument("--duration-us", type=int, default=300000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-seed", type=int, default=2026)
    parser.add_argument("--data-root", default="gen5_data/nmnist")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--target-parameters", type=int, default=133631)
    parser.add_argument("--temporal-levels", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--surrogate-slope", type=float, default=10.0)
    parser.add_argument("--minimum-reference-accuracy", type=float, default=0.90)
    parser.add_argument("--maximum-accuracy-gap", type=float, default=0.01)
    parser.add_argument("--minimum-state-effect", type=float, default=0.005)
    parser.add_argument("--minimum-effect-seeds", type=int, default=2)
    parser.add_argument("--minimum-spike-activity", type=float, default=0.01)
    parser.add_argument("--maximum-spike-activity", type=float, default=0.30)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="gen5_outputs/gen19_nmnist_state_replication_cuda")
    parser.add_argument("--progress-path", default=None)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_arms:
        print("\n".join(available_gen19_arms()))
        return
    config = Gen19Config(
        seeds=tuple(args.seeds),
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        timesteps=args.timesteps,
        spatial_bins=args.spatial_bins,
        duration_us=args.duration_us,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        data_seed=args.data_seed,
        data_root=args.data_root,
        download=not args.no_download,
        validation_fraction=args.validation_fraction,
        target_parameters=args.target_parameters,
        temporal_levels=tuple(args.temporal_levels),
        temporal_conv_kernel_size=args.kernel_size,
        surrogate_slope=args.surrogate_slope,
        minimum_reference_accuracy=args.minimum_reference_accuracy,
        maximum_accuracy_gap_vs_conv=args.maximum_accuracy_gap,
        minimum_state_effect=args.minimum_state_effect,
        minimum_effect_seed_count=args.minimum_effect_seeds,
        minimum_spike_activity=args.minimum_spike_activity,
        maximum_spike_activity=args.maximum_spike_activity,
    )
    progress_path = args.progress_path or str(
        pathlib.Path(args.output_dir) / "gen19_nmnist_state_replication_progress.json"
    )
    result = run_gen19_nmnist_state_replication(
        config, device=args.device, progress_path=progress_path
    )
    paths = result.save(args.output_dir, plot=not args.no_plot)
    paths["progress"] = progress_path
    print(json.dumps({
        "paths": paths,
        "device": result.device,
        "dataset": result.dataset,
        "decision": result.decision,
        "summary": result.summary,
    }, indent=2))


if __name__ == "__main__":
    main()
