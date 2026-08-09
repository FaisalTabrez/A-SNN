"""Sprint 31 CLI: transfer fixed heterogeneous delays to official SHD."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5 import (  # noqa: E402
    SHDConfig,
    available_shd_arms,
    run_shd_benchmark,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 31 fixed-delay transfer on Spiking Heidelberg Digits"
    )
    parser.add_argument("--list-arms", action="store_true")
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--train-samples", type=int, default=0)
    parser.add_argument("--test-samples", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=64)
    parser.add_argument("--duration-seconds", type=float, default=1.4)
    parser.add_argument("--hidden-neurons", type=int, default=128)
    parser.add_argument("--sensor-fanout", type=int, default=1)
    parser.add_argument("--recurrent-fanout", type=int, default=4)
    parser.add_argument("--max-edges", type=int, default=2048)
    parser.add_argument("--reservoir-leak", type=float, default=0.90)
    parser.add_argument("--reservoir-threshold", type=float, default=1.0)
    parser.add_argument("--input-gain", type=float, default=1.0)
    parser.add_argument("--count-hidden-units", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--surrogate-slope", type=float, default=10.0)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--reservoir-learning-rate", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--ltw-minimum", type=float, default=0.0)
    parser.add_argument("--ltw-maximum", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-seed", type=int, default=2026)
    parser.add_argument(
        "--data-root",
        default="/content/drive/MyDrive/A-SNN/gen5_data/shd",
    )
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        default="gen5_outputs/shd_benchmark",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_arms:
        print("\n".join(available_shd_arms()))
        return
    config = SHDConfig(
        seeds=tuple(args.seeds),
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        timesteps=args.timesteps,
        duration_seconds=args.duration_seconds,
        hidden_neurons=args.hidden_neurons,
        sensor_fanout=args.sensor_fanout,
        recurrent_fanout=args.recurrent_fanout,
        max_edges=args.max_edges,
        reservoir_leak=args.reservoir_leak,
        reservoir_threshold=args.reservoir_threshold,
        input_gain=args.input_gain,
        count_hidden_units=args.count_hidden_units,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        learning_rate=args.learning_rate,
        reservoir_learning_rate=args.reservoir_learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        data_seed=args.data_seed,
        data_root=args.data_root,
        download=not args.no_download,
    )
    result = run_shd_benchmark(
        config,
        device=args.device,
        surrogate_slope=args.surrogate_slope,
        arm_names=args.arms,
        ltw_minimum=args.ltw_minimum,
        ltw_maximum=args.ltw_maximum,
    )
    paths = result.save(args.output_dir, plot=not args.no_plot)
    print(json.dumps({"paths": paths, "device": result.device, "summary": result.summary}, indent=2))


if __name__ == "__main__":
    main()
