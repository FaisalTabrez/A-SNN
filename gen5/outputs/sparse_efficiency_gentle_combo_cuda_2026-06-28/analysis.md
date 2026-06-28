# CUDA sparse-efficiency finalist screen

Date: 2026-06-28

This bundle records the CUDA sparse-efficiency follow-up that compared the
new `gentle_ltw_scheduled` group against its two component mechanisms:
`low_ltw_pruning` and `scheduled_sprouting`.

Run context:

- Device: CUDA / Colab T4 class GPU
- Groups: `gentle_ltw_scheduled`, `low_ltw_pruning`, `scheduled_sprouting`
- Seeds: `42`, `43`, `44`
- Generations: `200`
- Population size: `10000`
- Epoch steps: `120`
- Neuron scale points: `16`, `32`, `64`
- Edge capacities: `128`, `256`, `512`
- Completed trials: `27 / 27`

Archived files:

- `sparse_efficiency.json`
- `sparse_efficiency_records.csv`
- `sparse_efficiency_summary.csv`
- `sparse_efficiency_progress.json`
- `sparse_efficiency_summary.png`
- `colab_log.txt`

## Summary

| Group | Neurons | Final mean best fitness | Active synapses | Utilization | Fitness / active synapse | Hidden-edge fraction | Direct sensor-motor fraction | Threshold success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gentle_ltw_scheduled | 16 | 24.33 | 59.13 | 46.20% | 0.338 | 42.84% | 18.27% | 66.67% |
| gentle_ltw_scheduled | 32 | 23.67 | 77.36 | 30.22% | 0.233 | 82.10% | 8.47% | 0% |
| gentle_ltw_scheduled | 64 | 24.00 | 131.01 | 25.59% | 0.137 | 92.18% | 5.40% | 33.33% |
| low_ltw_pruning | 16 | 24.33 | 49.54 | 38.71% | 0.404 | 42.21% | 18.68% | 66.67% |
| low_ltw_pruning | 32 | 26.00 | 97.52 | 38.09% | 0.191 | 84.12% | 6.14% | 100% |
| low_ltw_pruning | 64 | 23.00 | 193.42 | 37.78% | 0.100 | 95.27% | 2.27% | 0% |
| scheduled_sprouting | 16 | 24.67 | 81.84 | 63.93% | 0.212 | 43.33% | 16.37% | 66.67% |
| scheduled_sprouting | 32 | 24.67 | 111.81 | 43.68% | 0.164 | 84.61% | 5.81% | 66.67% |
| scheduled_sprouting | 64 | 23.67 | 134.96 | 26.36% | 0.138 | 94.52% | 2.86% | 33.33% |

## Interpretation

- `low_ltw_pruning` remains the raw-fitness leader at the most promising scale
  point: `32` neurons reached final mean best fitness `26.00` with `100%`
  threshold success.
- `gentle_ltw_scheduled` did solve the earlier protected-core collapse. Active
  synapses scaled from `59.13` to `77.36` to `131.01`, instead of collapsing to
  the roughly `40-50` active-edge regime seen in the original
  `protected_sparse_core` screen.
- The gentle schedule did not beat `low_ltw_pruning` on raw fitness. At
  `32` neurons it traded down from `26.00` to `23.67`, which is too large a
  survival penalty to treat it as the default rule.
- `scheduled_sprouting` remains a stabilizer rather than a standalone winner.
  It controlled large-capacity edge growth well, but did not dominate either
  raw fitness or efficiency.
- Hidden-edge usage rises sharply with neuron count across all groups. This
  confirms that larger brains are routing through hidden nodes, but the current
  simple foraging world is not yet rewarding that extra representational depth.

## Decision

Promote two sparse-efficiency finalists:

1. `low_ltw_pruning` as the raw-fitness candidate.
2. `gentle_ltw_scheduled` as the balanced sparsity candidate.

Do not run more broad sparse-efficiency matrices yet. The next run should spend
Colab time only on these two finalists across `10` seeds and `500` generations.

Recommended command:

```bash
python gen5/examples/sprint13_sparse_efficiency_ablation.py \
  --device cuda \
  --groups low_ltw_pruning gentle_ltw_scheduled \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --generations 500 \
  --population-size 10000 \
  --epoch-steps 120 \
  --neuron-counts 16 32 64 \
  --max-edges 128 256 512 \
  --output-dir gen5_outputs/sparse_efficiency_finalists_cuda
```

