"""Sprint 13: sparse-efficiency ablation.

This runner follows the neuron-scaling result: larger AMMC brains filled more
edge capacity without materially improving foraging fitness. The goal here is
to test whether active-edge pressure, stronger low-LTW pruning, capacity-scaled
sprouting, and protected core pathways can preserve behavior while reducing
structural bloat.

Colab-scale run:

```python
!python gen5/examples/sprint13_sparse_efficiency_ablation.py \
  --device cuda \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --generations 500 \
  --population-size 10000 \
  --epoch-steps 120 \
  --neuron-counts 16 32 64 \
  --max-edges 128 256 512 \
  --output-dir gen5_outputs/sparse_efficiency_cuda
```
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5 import (
    NeuronScalePoint,
    SparseEfficiencyConfig,
    SparseEfficiencyRunner,
    default_sparse_efficiency_groups,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AMMC Gen-5 sparse-efficiency ablation")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    parser.add_argument("--generations", type=int, default=500)
    parser.add_argument("--epoch-steps", type=int, default=120)
    parser.add_argument("--population-size", type=int, default=10_000)
    parser.add_argument("--food-count", type=int, default=128)
    parser.add_argument("--toxin-count", type=int, default=128)
    parser.add_argument("--sensor-radius", type=float, default=0.35)
    parser.add_argument("--friction", type=float, default=0.985)
    parser.add_argument("--action-gain", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=25.0)
    parser.add_argument("--reference-max-edges", type=int, default=128)
    parser.add_argument("--protected-core-edge-count", type=int, default=None)
    parser.add_argument(
        "--groups",
        nargs="+",
        default=None,
        help="Subset of sparse-efficiency groups to run. Use --list-groups to inspect names.",
    )
    parser.add_argument("--list-groups", action="store_true", help="Print available group names and exit")
    parser.add_argument("--checkpoint-every-trials", type=int, default=1)
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--neuron-counts", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--max-edges", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="gen5_outputs/sparse_efficiency")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    available_groups = default_sparse_efficiency_groups()
    if args.list_groups:
        print(json.dumps([group.__dict__ for group in available_groups], indent=2))
        return
    if len(args.neuron_counts) != len(args.max_edges):
        raise SystemExit("--neuron-counts and --max-edges must have the same length")

    groups = select_groups(args.groups, available_groups)
    scale_points = tuple(
        NeuronScalePoint(neuron_count=neurons, max_edges=max_edges)
        for neurons, max_edges in zip(args.neuron_counts, args.max_edges)
    )
    runner = SparseEfficiencyRunner(
        SparseEfficiencyConfig(
            seeds=tuple(args.seeds),
            generations=args.generations,
            epoch_steps=args.epoch_steps,
            population_size=args.population_size,
            food_count=args.food_count,
            toxin_count=args.toxin_count,
            sensor_radius=args.sensor_radius,
            friction=args.friction,
            action_gain=args.action_gain,
            reference_max_edges=args.reference_max_edges,
            adaptation_fitness_threshold=args.threshold,
            device=args.device,
            scale_points=scale_points,
            groups=groups,
            protected_core_edge_count=args.protected_core_edge_count,
        )
    )
    if args.no_checkpoint:
        result = runner.run()
        paths = runner.save_outputs(result, args.output_dir, plot=not args.no_plot)
    else:
        result, paths = runner.run_with_checkpoints(
            args.output_dir,
            plot=not args.no_plot,
            checkpoint_every_trials=args.checkpoint_every_trials,
        )
    print(
        json.dumps(
            {
                "paths": paths,
                "scale_points": [point.__dict__ for point in scale_points],
                "groups": [group.name for group in groups],
                "summary": [row.__dict__ for row in result.summary],
            },
            indent=2,
        )
    )


def select_groups(group_names: list[str] | None, available_groups):
    if not group_names:
        return tuple(available_groups)
    by_name = {group.name: group for group in available_groups}
    missing = [name for name in group_names if name not in by_name]
    if missing:
        raise SystemExit(f"unknown sparse-efficiency group(s): {', '.join(missing)}")
    return tuple(by_name[name] for name in group_names)


if __name__ == "__main__":
    main()
