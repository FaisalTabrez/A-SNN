"""Run the preregistered Gen-20 N-MNIST translation experiment."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5 import (  # noqa: E402
    Gen20Config,
    available_gen20_arms,
    bundle_gen20_artifacts,
    run_gen20,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen Gen-20 spiking spatial-temporal translation."
    )
    parser.add_argument("--list-arms", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="gen5_data/nmnist")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--train-samples", type=int, default=0)
    parser.add_argument("--test-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--event-dropout", type=float, default=0.02)
    parser.add_argument("--maximum-shift", type=int, default=2)
    parser.add_argument("--surrogate-slope", type=float, default=10.0)
    parser.add_argument(
        "--output-dir", default="gen5_outputs/gen20_spiking_spatiotemporal_cuda"
    )
    parser.add_argument("--progress-path", default=None)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_arms:
        print("\n".join(available_gen20_arms()))
        return
    config = Gen20Config(
        data_root=args.data_root,
        download=not args.no_download,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        event_dropout=args.event_dropout,
        maximum_shift=args.maximum_shift,
        surrogate_slope=args.surrogate_slope,
    )
    progress = args.progress_path or str(
        pathlib.Path(args.output_dir) / "gen20_spiking_spatiotemporal_progress.json"
    )
    result = run_gen20(config, device=args.device, progress_path=progress)
    paths = result.save(args.output_dir, plot=not args.no_plot)
    paths["progress"] = progress
    paths.update(bundle_gen20_artifacts(paths, args.output_dir))
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
