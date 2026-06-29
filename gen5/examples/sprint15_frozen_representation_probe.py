"""Sprint 15: frozen representation probe.

This runner keeps the AMMC recurrent sparse substrate frozen, collects final
membrane/spike-count features, and trains only a tiny linear readout. It tells
us whether failed tasks are failing because the representation is missing or
because the fixed motor readout/transducer is too weak.

Colab-scale run:

```python
!python gen5/examples/sprint15_frozen_representation_probe.py \
  --device cuda \
  --sample-count 4096 \
  --timesteps 8 \
  --neuron-count 16 \
  --max-edges 128 \
  --epochs 200 \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/frozen_representation_probe_cuda
```
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5 import FrozenProbeConfig, FrozenRepresentationProbeRunner, FrozenTaskConfig, available_frozen_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AMMC Gen-5 frozen representation probe")
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
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="gen5_outputs/frozen_representation_probe")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_tasks:
        print(json.dumps(list(available_frozen_tasks()), indent=2))
        return

    task_config = FrozenTaskConfig(
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
    runner = FrozenRepresentationProbeRunner(
        FrozenProbeConfig(
            task_config=task_config,
            train_fraction=args.train_fraction,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            standardize_features=not args.no_standardize,
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
