"""Gen-10 masked-sensor robust representation CLI."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5 import (  # noqa: E402
    SHDConfig,
    available_gen10_representation_arms,
    run_gen10_robust_representation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preregistered Gen-10 masked-sensor representation reset."
    )
    parser.add_argument("--list-arms", action="store_true")
    parser.add_argument("--screen-seed", type=int, default=151)
    parser.add_argument("--confirm-seeds", nargs="+", type=int, default=[151, 152, 153])
    parser.add_argument("--screen-train-samples", type=int, default=15000)
    parser.add_argument("--screen-validation-samples", type=int, default=3000)
    parser.add_argument("--screen-test-samples", type=int, default=3000)
    parser.add_argument("--screen-epochs", type=int, default=5)
    parser.add_argument("--confirm-epochs", type=int, default=15)
    parser.add_argument("--training-mask-fraction", type=float, default=0.20)
    parser.add_argument("--damage-fraction", type=float, default=0.35)
    parser.add_argument("--damage-seed", type=int, default=909)
    parser.add_argument("--alignment-weight", type=float, default=0.10)
    parser.add_argument("--promotion-margin", type=float, default=0.01)
    parser.add_argument("--damaged-promotion-margin", type=float, default=0.02)
    parser.add_argument("--minimum-parameter-ratio", type=float, default=0.95)
    parser.add_argument("--maximum-parameter-ratio", type=float, default=1.05)
    parser.add_argument("--minimum-spike-rate", type=float, default=0.01)
    parser.add_argument("--maximum-spike-rate", type=float, default=0.30)
    parser.add_argument("--accuracy-margin", type=float, default=0.01)
    parser.add_argument("--causal-margin", type=float, default=0.005)
    parser.add_argument("--robustness-margin", type=float, default=0.005)
    parser.add_argument("--target-parameters", type=int, default=133631)
    parser.add_argument("--timesteps", type=int, default=64)
    parser.add_argument("--duration-seconds", type=float, default=1.0)
    parser.add_argument("--temporal-levels", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--input-kernel-size", type=int, default=5)
    parser.add_argument("--hidden-kernel-size", type=int, default=3)
    parser.add_argument("--tcn-dilation", type=int, default=2)
    parser.add_argument("--surrogate-slope", type=float, default=10.0)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-seed", type=int, default=2026)
    parser.add_argument("--data-root", default="/content/drive/MyDrive/A-SNN/gen5_data/ssc")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="gen5_outputs/gen10_robust_representation")
    parser.add_argument("--progress-path", default=None)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_arms:
        print("\n".join(available_gen10_representation_arms()))
        return
    config = SHDConfig(
        seeds=tuple(args.confirm_seeds), train_samples=0, test_samples=0,
        input_neurons=700, classes=35, timesteps=args.timesteps,
        duration_seconds=args.duration_seconds, hidden_neurons=128,
        max_edges=4096, epochs=args.confirm_epochs, warmup_epochs=0,
        learning_rate=args.learning_rate, reservoir_learning_rate=0.0,
        weight_decay=args.weight_decay, batch_size=args.batch_size,
        data_seed=args.data_seed, data_root=args.data_root,
        download=not args.no_download,
    )
    progress_path = args.progress_path or str(
        pathlib.Path(args.output_dir) / "gen10_robust_representation_progress.json"
    )
    result = run_gen10_robust_representation(
        config, screen_seed=args.screen_seed, confirm_seeds=args.confirm_seeds,
        screen_train_samples=args.screen_train_samples,
        screen_validation_samples=args.screen_validation_samples,
        screen_test_samples=args.screen_test_samples,
        screen_epochs=args.screen_epochs, confirm_epochs=args.confirm_epochs,
        training_mask_fraction=args.training_mask_fraction,
        damage_fraction=args.damage_fraction, damage_seed=args.damage_seed,
        alignment_weight=args.alignment_weight, promotion_margin=args.promotion_margin,
        damaged_promotion_margin=args.damaged_promotion_margin,
        minimum_parameter_ratio=args.minimum_parameter_ratio,
        maximum_parameter_ratio=args.maximum_parameter_ratio,
        minimum_spike_rate=args.minimum_spike_rate,
        maximum_spike_rate=args.maximum_spike_rate,
        accuracy_margin=args.accuracy_margin, causal_margin=args.causal_margin,
        robustness_margin=args.robustness_margin,
        target_parameters=args.target_parameters, device=args.device,
        temporal_levels=args.temporal_levels,
        input_kernel_size=args.input_kernel_size,
        hidden_kernel_size=args.hidden_kernel_size,
        tcn_dilation=args.tcn_dilation, surrogate_slope=args.surrogate_slope,
        progress_path=progress_path,
    )
    paths = result.save(args.output_dir, plot=not args.no_plot)
    paths["progress"] = progress_path
    print(json.dumps({
        "paths": paths, "device": result.device,
        "promoted_arms": result.promoted_arms,
        "decision": result.decision,
        "summary": result.confirmation_summary,
    }, indent=2))


if __name__ == "__main__":
    main()
