"""Sprint 43 CLI: validation-selected raw versus sparse SHD audit."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5 import SHDConfig, run_shd_validation_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 43 SHD validation checkpoint audit")
    parser.add_argument("--topology-seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--readout-seeds", nargs="+", type=int, default=[142, 143, 144])
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--target-parameters", type=int, default=133631)
    parser.add_argument("--timesteps", type=int, default=64)
    parser.add_argument("--duration-seconds", type=float, default=1.4)
    parser.add_argument("--sensor-fanout", type=int, default=1)
    parser.add_argument("--recurrent-fanout", type=int, default=4)
    parser.add_argument("--max-edges", type=int, default=4096)
    parser.add_argument("--reservoir-leak", type=float, default=0.90)
    parser.add_argument("--input-gain", type=float, default=1.0)
    parser.add_argument("--projection-dim", type=int, default=32)
    parser.add_argument("--temporal-levels", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--readout-hidden-units", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-seed", type=int, default=2026)
    parser.add_argument("--data-root", default="/content/drive/MyDrive/A-SNN/gen5_data/shd")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="gen5_outputs/shd_validation_checkpoint")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SHDConfig(
        seeds=tuple(args.topology_seeds), timesteps=args.timesteps,
        duration_seconds=args.duration_seconds, hidden_neurons=512,
        sensor_fanout=args.sensor_fanout, recurrent_fanout=args.recurrent_fanout,
        max_edges=args.max_edges, reservoir_leak=args.reservoir_leak,
        input_gain=args.input_gain, epochs=args.epochs, warmup_epochs=0,
        learning_rate=args.learning_rate, reservoir_learning_rate=0.0,
        weight_decay=args.weight_decay, batch_size=args.batch_size,
        data_seed=args.data_seed, data_root=args.data_root, download=not args.no_download,
    )
    result = run_shd_validation_checkpoint(
        config, topology_seeds=args.topology_seeds, readout_seeds=args.readout_seeds,
        validation_fraction=args.validation_fraction, target_parameters=args.target_parameters,
        device=args.device, projection_dim=args.projection_dim,
        temporal_levels=args.temporal_levels, readout_hidden_units=args.readout_hidden_units,
    )
    paths = result.save(args.output_dir, plot=not args.no_plot)
    print(json.dumps({"paths": paths, "device": result.device, "summary": result.summary}, indent=2))


if __name__ == "__main__":
    main()
