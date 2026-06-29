"""Sprint 15: frozen diversified task benchmark.

This runner freezes the AMMC sparse substrate and evaluates it on synthetic
non-foraging tasks. No evolution, plasticity, optimizer, or topology mutation
is allowed during evaluation.

Colab-scale run:

```python
!python gen5/examples/sprint15_frozen_diversified_tasks.py \
  --device cuda \
  --sample-count 4096 \
  --timesteps 8 \
  --neuron-count 16 \
  --max-edges 128 \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/frozen_diversified_tasks_cuda
```
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5 import FrozenTaskConfig, FrozenTaskRunner, available_frozen_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AMMC Gen-5 frozen diversified task benchmark")
    parser.add_argument("--tasks", nargs="+", default=list(available_frozen_tasks()), choices=available_frozen_tasks())
    parser.add_argument("--list-tasks", action="store_true", help="Print available synthetic tasks and exit")
    parser.add_argument("--sample-count", type=int, default=4096)
    parser.add_argument("--timesteps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neuron-count", type=int, default=16)
    parser.add_argument("--max-edges", type=int, default=128)
    parser.add_argument("--sensor-gain", type=float, default=1.0)
    parser.add_argument("--leak", type=float, default=0.9)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--input-amplitude", type=float, default=0.75)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="gen5_outputs/frozen_diversified_tasks")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_tasks:
        print(json.dumps(list(available_frozen_tasks()), indent=2))
        return

    runner = FrozenTaskRunner(
        FrozenTaskConfig(
            tasks=tuple(args.tasks),
            sample_count=args.sample_count,
            timesteps=args.timesteps,
            seed=args.seed,
            neuron_count=args.neuron_count,
            max_edges=args.max_edges,
            sensor_gain=args.sensor_gain,
            leak=args.leak,
            threshold=args.threshold,
            input_amplitude=args.input_amplitude,
            device=args.device,
        )
    )
    result = runner.run()
    paths = runner.save_outputs(result, args.output_dir, plot=not args.no_plot)
    print(
        json.dumps(
            {
                "paths": paths,
                "summary": [row.__dict__ for row in result.summary],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
