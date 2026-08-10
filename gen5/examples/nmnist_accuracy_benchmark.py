"""Run the bounded full-resolution N-MNIST accuracy benchmark."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5 import (  # noqa: E402
    NMNISTAccuracyConfig,
    available_nmnist_accuracy_arms,
    bundle_nmnist_accuracy_artifacts,
    run_nmnist_accuracy_benchmark,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preregistered full-resolution N-MNIST accuracy track."
    )
    parser.add_argument("--list-arms", action="store_true")
    parser.add_argument("--screen-seed", type=int, default=210)
    parser.add_argument("--confirmation-seeds", nargs=3, type=int, default=[211, 212, 213])
    parser.add_argument("--timesteps", type=int, default=10)
    parser.add_argument("--duration-us", type=int, default=300000)
    parser.add_argument("--screen-train-samples", type=int, default=20000)
    parser.add_argument("--train-samples", type=int, default=0)
    parser.add_argument("--test-samples", type=int, default=0)
    parser.add_argument("--screen-epochs", type=int, default=4)
    parser.add_argument("--confirmation-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--data-seed", type=int, default=2026)
    parser.add_argument("--data-root", default="gen5_data/nmnist")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--promotion-gap", type=float, default=0.01)
    parser.add_argument("--maximum-promoted-arms", type=int, default=2)
    parser.add_argument("--practical-accuracy", type=float, default=0.99)
    parser.add_argument("--stretch-accuracy", type=float, default=0.994)
    parser.add_argument("--event-dropout", type=float, default=0.02)
    parser.add_argument("--maximum-shift", type=int, default=2)
    parser.add_argument("--surrogate-slope", type=float, default=10.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir", default="gen5_outputs/nmnist_accuracy_benchmark_cuda"
    )
    parser.add_argument("--progress-path", default=None)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_arms:
        print("\n".join(available_nmnist_accuracy_arms()))
        return
    config = NMNISTAccuracyConfig(
        screen_seed=args.screen_seed,
        confirmation_seeds=tuple(args.confirmation_seeds),
        timesteps=args.timesteps,
        duration_us=args.duration_us,
        screen_train_samples=args.screen_train_samples,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        screen_epochs=args.screen_epochs,
        confirmation_epochs=args.confirmation_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_fraction=args.validation_fraction,
        data_seed=args.data_seed,
        data_root=args.data_root,
        download=not args.no_download,
        promotion_gap=args.promotion_gap,
        maximum_promoted_arms=args.maximum_promoted_arms,
        practical_accuracy=args.practical_accuracy,
        stretch_accuracy=args.stretch_accuracy,
        event_dropout=args.event_dropout,
        maximum_shift=args.maximum_shift,
        surrogate_slope=args.surrogate_slope,
    )
    progress = args.progress_path or str(
        pathlib.Path(args.output_dir) / "nmnist_accuracy_benchmark_progress.json"
    )
    result = run_nmnist_accuracy_benchmark(
        config, device=args.device, progress_path=progress
    )
    paths = result.save(args.output_dir, plot=not args.no_plot)
    paths["progress"] = progress
    paths.update(bundle_nmnist_accuracy_artifacts(paths, args.output_dir))
    print(json.dumps({
        "paths": paths,
        "device": result.device,
        "dataset": result.dataset,
        "promoted_arms": result.promoted_arms,
        "decision": result.decision,
        "summary": result.summary,
    }, indent=2))


if __name__ == "__main__":
    main()
