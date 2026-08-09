"""Sprint 48 CLI: replicate residual-LIF contribution on official SSC."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5 import SHDConfig, run_ssc_residual_lif_replication  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 48 SSC residual-LIF replication")
    parser.add_argument("--readout-seeds", nargs="+", type=int, default=[142, 143, 144])
    parser.add_argument("--target-parameters", type=int, default=133631)
    parser.add_argument("--timesteps", type=int, default=64)
    parser.add_argument("--duration-seconds", type=float, default=1.0)
    parser.add_argument("--temporal-levels", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--temporal-conv-kernel-size", type=int, default=5)
    parser.add_argument("--surrogate-slope", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-seed", type=int, default=2026)
    parser.add_argument("--train-samples", type=int, default=0)
    parser.add_argument("--validation-samples", type=int, default=0)
    parser.add_argument("--test-samples", type=int, default=0)
    parser.add_argument("--data-root", default="/content/drive/MyDrive/A-SNN/gen5_data/ssc")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="gen5_outputs/ssc_residual_lif_replication")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SHDConfig(
        seeds=tuple(args.readout_seeds),
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        input_neurons=700,
        classes=35,
        timesteps=args.timesteps,
        duration_seconds=args.duration_seconds,
        hidden_neurons=128,
        max_edges=4096,
        epochs=args.epochs,
        warmup_epochs=0,
        learning_rate=args.learning_rate,
        reservoir_learning_rate=0.0,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        data_seed=args.data_seed,
        data_root=args.data_root,
        download=not args.no_download,
    )
    result = run_ssc_residual_lif_replication(
        config,
        readout_seeds=args.readout_seeds,
        validation_samples=args.validation_samples,
        target_parameters=args.target_parameters,
        device=args.device,
        temporal_levels=args.temporal_levels,
        temporal_conv_kernel_size=args.temporal_conv_kernel_size,
        surrogate_slope=args.surrogate_slope,
    )
    paths = result.save(args.output_dir, plot=not args.no_plot)
    print(json.dumps({"paths": paths, "device": result.device, "summary": result.summary}, indent=2))


if __name__ == "__main__":
    main()
