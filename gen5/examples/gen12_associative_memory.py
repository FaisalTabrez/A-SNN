"""Gen-12 associative-memory CLI."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5 import (  # noqa: E402
    SHDConfig,
    available_gen12_memory_strategies,
    run_gen12_associative_memory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preregistered Gen-12 associative-memory experiment."
    )
    parser.add_argument("--list-strategies", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[157, 158, 159])
    parser.add_argument("--source-epochs", type=int, default=15)
    parser.add_argument("--source-mask-fraction", type=float, default=0.20)
    parser.add_argument("--damage-fraction", type=float, default=0.35)
    parser.add_argument("--damage-seed", type=int, default=909)
    parser.add_argument("--adaptation-budgets", nargs="+", type=int, default=[0, 64, 256, 1024, 4096])
    parser.add_argument("--adaptation-epochs-per-block", type=int, default=3)
    parser.add_argument("--adaptation-learning-rate", type=float, default=0.001)
    parser.add_argument("--memory-mix", type=float, default=0.50)
    parser.add_argument("--memory-temperature", type=float, default=0.10)
    parser.add_argument("--spike-fraction", type=float, default=0.20)
    parser.add_argument("--minimum-shift-drop", type=float, default=0.02)
    parser.add_argument("--minimum-adaptation-gain", type=float, default=0.02)
    parser.add_argument("--auc-margin", type=float, default=0.01)
    parser.add_argument("--final-accuracy-margin", type=float, default=0.01)
    parser.add_argument("--forgetting-margin", type=float, default=0.005)
    parser.add_argument("--causal-margin", type=float, default=0.005)
    parser.add_argument("--minimum-spike-density", type=float, default=0.05)
    parser.add_argument("--maximum-spike-density", type=float, default=0.35)
    parser.add_argument("--target-parameters", type=int, default=133631)
    parser.add_argument("--timesteps", type=int, default=64)
    parser.add_argument("--duration-seconds", type=float, default=1.0)
    parser.add_argument("--temporal-levels", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--input-kernel-size", type=int, default=5)
    parser.add_argument("--hidden-kernel-size", type=int, default=3)
    parser.add_argument("--tcn-dilation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-seed", type=int, default=2026)
    parser.add_argument("--data-root", default="/content/drive/MyDrive/A-SNN/gen5_data/ssc")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="gen5_outputs/gen12_associative_memory")
    parser.add_argument("--progress-path", default=None)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_strategies:
        print("\n".join(available_gen12_memory_strategies()))
        return
    config = SHDConfig(
        seeds=tuple(args.seeds), train_samples=0, test_samples=0,
        input_neurons=700, classes=35, timesteps=args.timesteps,
        duration_seconds=args.duration_seconds, hidden_neurons=128,
        max_edges=4096, epochs=args.source_epochs, warmup_epochs=0,
        learning_rate=args.learning_rate, reservoir_learning_rate=0.0,
        weight_decay=args.weight_decay, batch_size=args.batch_size,
        data_seed=args.data_seed, data_root=args.data_root,
        download=not args.no_download,
    )
    progress_path = args.progress_path or str(
        pathlib.Path(args.output_dir) / "gen12_associative_memory_progress.json"
    )
    result = run_gen12_associative_memory(
        config, seeds=args.seeds, source_epochs=args.source_epochs,
        source_mask_fraction=args.source_mask_fraction,
        damage_fraction=args.damage_fraction, damage_seed=args.damage_seed,
        adaptation_budgets=args.adaptation_budgets,
        adaptation_epochs_per_block=args.adaptation_epochs_per_block,
        adaptation_learning_rate=args.adaptation_learning_rate,
        memory_mix=args.memory_mix,
        memory_temperature=args.memory_temperature,
        spike_fraction=args.spike_fraction,
        minimum_shift_drop=args.minimum_shift_drop,
        minimum_adaptation_gain=args.minimum_adaptation_gain,
        auc_margin=args.auc_margin,
        final_accuracy_margin=args.final_accuracy_margin,
        forgetting_margin=args.forgetting_margin,
        causal_margin=args.causal_margin,
        minimum_spike_density=args.minimum_spike_density,
        maximum_spike_density=args.maximum_spike_density,
        target_parameters=args.target_parameters, device=args.device,
        temporal_levels=args.temporal_levels,
        input_kernel_size=args.input_kernel_size,
        hidden_kernel_size=args.hidden_kernel_size,
        tcn_dilation=args.tcn_dilation, progress_path=progress_path,
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
