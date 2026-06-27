"""Sprint 12: neuron / decision-node scaling evaluation.

This runner tests whether larger AMMC brains provide measurable capability
gain, instead of merely adding unused capacity. The default sweep keeps the
8-sensor / 4-motor transducer fixed and increases hidden decision nodes:

- 16 neurons: 4 hidden decision nodes
- 32 neurons: 20 hidden decision nodes
- 64 neurons: 52 hidden decision nodes

Colab-scale run:

```python
!python gen5/examples/sprint12_neuron_scaling.py \
  --device cuda \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --generations 500 \
  --population-size 10000 \
  --epoch-steps 120 \
  --neuron-counts 16 32 64 \
  --max-edges 128 256 512 \
  --output-dir gen5_outputs/neuron_scaling
```
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5 import NeuronScalePoint, NeuronScalingConfig, NeuronScalingRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AMMC Gen-5 neuron scaling evaluation")
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
    parser.add_argument("--neuron-counts", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--max-edges", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="gen5_outputs/neuron_scaling")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.neuron_counts) != len(args.max_edges):
        raise SystemExit("--neuron-counts and --max-edges must have the same length")

    scale_points = tuple(
        NeuronScalePoint(neuron_count=neurons, max_edges=max_edges)
        for neurons, max_edges in zip(args.neuron_counts, args.max_edges)
    )
    runner = NeuronScalingRunner(
        NeuronScalingConfig(
            seeds=tuple(args.seeds),
            generations=args.generations,
            epoch_steps=args.epoch_steps,
            population_size=args.population_size,
            food_count=args.food_count,
            toxin_count=args.toxin_count,
            sensor_radius=args.sensor_radius,
            friction=args.friction,
            action_gain=args.action_gain,
            device=args.device,
            scale_points=scale_points,
            adaptation_fitness_threshold=args.threshold,
        )
    )
    result = runner.run()
    paths = runner.save_outputs(result, args.output_dir, plot=not args.no_plot)
    print(
        json.dumps(
            {
                "paths": paths,
                "scale_points": [point.__dict__ for point in scale_points],
                "summary": [row.__dict__ for row in result.summary],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
