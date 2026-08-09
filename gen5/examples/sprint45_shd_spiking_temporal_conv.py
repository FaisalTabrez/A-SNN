"""Sprint 45 CLI: learned analog and spiking temporal convolution on SHD."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5 import SHDConfig, run_shd_spiking_temporal_conv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 45 spiking temporal convolution")
    parser.add_argument("--readout-seeds", nargs="+", type=int, default=[142, 143, 144])
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--target-parameters", type=int, default=133631)
    parser.add_argument("--timesteps", type=int, default=64)
    parser.add_argument("--duration-seconds", type=float, default=1.4)
    parser.add_argument("--temporal-levels", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--projection-dim", type=int, default=32)
    parser.add_argument("--dense-lif-hidden-neurons", type=int, default=128)
    parser.add_argument("--dense-lif-projection-dim", type=int, default=16)
    parser.add_argument("--temporal-conv-kernel-size", type=int, default=5)
    parser.add_argument("--surrogate-slope", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-seed", type=int, default=2026)
    parser.add_argument("--data-root", default="/content/drive/MyDrive/A-SNN/gen5_data/shd")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="gen5_outputs/shd_spiking_temporal_conv")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SHDConfig(
        seeds=tuple(args.readout_seeds), timesteps=args.timesteps,
        duration_seconds=args.duration_seconds, hidden_neurons=args.dense_lif_hidden_neurons,
        max_edges=4096, epochs=args.epochs, warmup_epochs=0,
        learning_rate=args.learning_rate, reservoir_learning_rate=0.0,
        weight_decay=args.weight_decay, batch_size=args.batch_size,
        data_seed=args.data_seed, data_root=args.data_root, download=not args.no_download,
    )
    result = run_shd_spiking_temporal_conv(
        config, readout_seeds=args.readout_seeds,
        validation_fraction=args.validation_fraction,
        target_parameters=args.target_parameters, device=args.device,
        temporal_levels=args.temporal_levels, projection_dim=args.projection_dim,
        dense_lif_hidden_neurons=args.dense_lif_hidden_neurons,
        dense_lif_projection_dim=args.dense_lif_projection_dim,
        temporal_conv_kernel_size=args.temporal_conv_kernel_size,
        surrogate_slope=args.surrogate_slope,
    )
    paths = result.save(args.output_dir, plot=not args.no_plot)
    print(json.dumps({"paths": paths, "device": result.device, "summary": result.summary}, indent=2))


if __name__ == "__main__":
    main()
