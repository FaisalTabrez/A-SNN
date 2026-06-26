"""Sprint 11: retention / catastrophic-forgetting ablation.

This run extends the plasticity ablation into three phases:

1. Original environment
2. Perturbed environment with food/toxin sensor channels inverted
3. Original environment again

The output measures whether each plasticity regime can adapt to the
perturbation while retaining or recovering the original skill.

Colab TPU/XLA:

```python
!python gen5/examples/sprint11_retention_ablation.py \
  --device xla \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --original-generations 100 \
  --perturbation-generations 300 \
  --recovery-generations 100 \
  --population-size 10000 \
  --epoch-steps 120 \
  --output-dir gen5_outputs/retention_ablation_xla
```
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5 import RetentionAblationConfig, RetentionAblationRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AMMC Gen-5 retention/forgetting ablation")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    parser.add_argument("--original-generations", type=int, default=100)
    parser.add_argument("--perturbation-generations", type=int, default=300)
    parser.add_argument("--recovery-generations", type=int, default=100)
    parser.add_argument("--epoch-steps", type=int, default=120)
    parser.add_argument("--population-size", type=int, default=10_000)
    parser.add_argument("--food-count", type=int, default=128)
    parser.add_argument("--toxin-count", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="gen5_outputs/retention_ablation")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = RetentionAblationRunner(
        RetentionAblationConfig(
            seeds=tuple(args.seeds),
            original_generations=args.original_generations,
            perturbation_generations=args.perturbation_generations,
            recovery_generations=args.recovery_generations,
            epoch_steps=args.epoch_steps,
            population_size=args.population_size,
            food_count=args.food_count,
            toxin_count=args.toxin_count,
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
