"""Sprint 11: multi-seed statistical convergence evaluation.

Colab-scale run:

```python
!python gen5/examples/sprint11_statistical_evaluation.py \
  --device xla \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --generations 500 \
  --population-size 10000 \
  --epoch-steps 120 \
  --output-dir gen5_outputs/statistical_trials
```
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5 import TrialRunner, TrialRunnerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AMMC Gen-5 multi-seed statistical evaluation")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    parser.add_argument("--generations", type=int, default=500)
    parser.add_argument("--epoch-steps", type=int, default=120)
    parser.add_argument("--population-size", type=int, default=10_000)
    parser.add_argument("--food-count", type=int, default=128)
    parser.add_argument("--toxin-count", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="gen5_outputs/statistical_trials")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = TrialRunner(
        TrialRunnerConfig(
            seeds=tuple(args.seeds),
            generations=args.generations,
            epoch_steps=args.epoch_steps,
            population_size=args.population_size,
            food_count=args.food_count,
            toxin_count=args.toxin_count,
            device=args.device,
        )
    )
    result = runner.run()
    paths = runner.save_outputs(result, args.output_dir, plot=not args.no_plot)
    final = result.aggregate_records[-1]
    print(
        json.dumps(
            {
                "paths": paths,
                "final_generation": final.generation,
                "final_mean_best_fitness": final.mean_best_fitness,
                "final_std_best_fitness": final.std_best_fitness,
                "seeds": args.seeds,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
