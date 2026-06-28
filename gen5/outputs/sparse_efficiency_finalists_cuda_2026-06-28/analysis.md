# CUDA sparse-efficiency finalist run

Date: 2026-06-28

This bundle records the full finalist sparse-efficiency run comparing
`low_ltw_pruning` against `gentle_ltw_scheduled`.

Run context:

- Device: CUDA / Colab T4 class GPU
- Groups: `low_ltw_pruning`, `gentle_ltw_scheduled`
- Seeds: `42` through `51`
- Generations: `500`
- Population size: `10000`
- Epoch steps: `120`
- Neuron scale points: `16`, `32`, `64`
- Edge capacities: `128`, `256`, `512`
- Completed trials: `60 / 60`

Archived files:

- `sparse_efficiency.json`
- `sparse_efficiency_records.csv`
- `sparse_efficiency_summary.csv`
- `sparse_efficiency_progress.json`
- `sparse_efficiency_summary.png`
- `colab_log.txt`

## Summary

| Group | Neurons | Final mean best fitness | Std | Active synapses | Utilization | Fitness / active synapse | Hidden-edge fraction | Direct sensor-motor fraction | Threshold success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gentle_ltw_scheduled | 16 | 26.00 | 1.41 | 61.08 | 47.72% | 0.306 | 44.73% | 14.22% | 90% |
| gentle_ltw_scheduled | 32 | 25.70 | 1.34 | 80.48 | 31.44% | 0.222 | 85.61% | 4.37% | 80% |
| gentle_ltw_scheduled | 64 | 24.40 | 0.97 | 131.30 | 25.65% | 0.133 | 95.22% | 2.28% | 50% |
| low_ltw_pruning | 16 | 25.40 | 0.70 | 51.64 | 40.34% | 0.350 | 44.33% | 14.30% | 90% |
| low_ltw_pruning | 32 | 26.00 | 1.33 | 103.54 | 40.45% | 0.177 | 86.05% | 3.90% | 100% |
| low_ltw_pruning | 64 | 25.40 | 1.07 | 206.55 | 40.34% | 0.085 | 96.49% | 1.06% | 80% |

## Interpretation

- `low_ltw_pruning` is the strongest default survival rule. It reached the
  best reliability at `32` neurons: final mean best fitness `26.00`, threshold
  success `100%`, and mean generation-to-threshold `99.8`.
- `gentle_ltw_scheduled` is now a real efficiency candidate, not merely a
  speculative variant. At `32` neurons it scored `25.70` versus `26.00` for
  `low_ltw_pruning`, while using `80.48` active synapses instead of `103.54`.
  That is about `22%` fewer active edges for only `0.30` lower mean best
  fitness.
- At `64` neurons, `gentle_ltw_scheduled` used about `36%` fewer active edges
  than `low_ltw_pruning` (`131.30` vs `206.55`) and had better fitness per
  active synapse (`0.133` vs `0.085`), but raw survival reliability was lower.
- The simple foraging world still does not reward larger brains. Hidden-edge
  fraction rises above `95%` at `64` neurons, but final mean best fitness does
  not improve. The model is using hidden structure, yet the environment is not
  complex enough to make that structure pay rent.
- `16` neurons remain surprisingly competitive on this world. `gentle_ltw_scheduled`
  at `16` neurons tied the best raw mean fitness (`26.00`) with fewer total
  representational degrees of freedom, though `32`-neuron `low_ltw_pruning`
  has the better success reliability.

## Decision

Freeze the sparse-efficiency search for the current simple bot world.

Use two presets going forward:

1. `low_ltw_pruning` at `32` neurons as the default raw-survival baseline.
2. `gentle_ltw_scheduled` at `32` neurons as the sparse-efficiency baseline.

Do not scale neuron counts further on this simple environment. The next major
test should increase world complexity so hidden neurons have a reason to exist:
larger arenas, moving hazards, delayed rewards, partial observability, or
multi-step resource chains.

