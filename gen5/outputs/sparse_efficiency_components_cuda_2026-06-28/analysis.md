# CUDA sparse-efficiency component screen

Date: 2026-06-28

Run context:

- Device: `cuda`
- Groups: `active_edge_penalty`, `low_ltw_pruning`, `scheduled_sprouting`
- Seeds: `42`, `43`, `44`
- Generations: `200`
- Population size: `10,000`
- Epoch steps: `120`
- Scale points:
  - `16` neurons / `128` edge slots
  - `32` neurons / `256` edge slots
  - `64` neurons / `512` edge slots
- Checkpointed trials: `27 / 27`

Raw files:

- `sparse_efficiency.json`
- `sparse_efficiency_records.csv`
- `sparse_efficiency_summary.csv`
- `sparse_efficiency_progress.json`
- `sparse_efficiency_summary.png`
- `colab_log.txt`

## Results

| Group | Neurons | Final mean best fitness | Active synapses | Utilization | Fitness / active synapse | Hidden-edge fraction | Direct sensor-motor fraction | Threshold success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| active_edge_penalty | 16 | 24.667 | 61.140 | 47.77% | 0.273 | 42.61% | 17.31% | 33.33% |
| active_edge_penalty | 32 | 23.667 | 133.011 | 51.96% | 0.123 | 85.05% | 5.46% | 0% |
| active_edge_penalty | 64 | 23.667 | 274.015 | 53.52% | 0.073 | 95.50% | 1.98% | 33.33% |
| low_ltw_pruning | 16 | 23.667 | 49.461 | 38.64% | 0.364 | 42.38% | 18.58% | 0% |
| low_ltw_pruning | 32 | 26.000 | 98.002 | 38.28% | 0.191 | 83.86% | 6.18% | 100% |
| low_ltw_pruning | 64 | 24.667 | 194.107 | 37.91% | 0.098 | 95.44% | 2.17% | 66.67% |
| scheduled_sprouting | 16 | 24.000 | 82.210 | 64.23% | 0.243 | 43.41% | 16.82% | 0% |
| scheduled_sprouting | 32 | 24.667 | 111.313 | 43.48% | 0.162 | 84.37% | 5.88% | 33.33% |
| scheduled_sprouting | 64 | 23.333 | 134.134 | 26.20% | 0.142 | 94.55% | 2.90% | 33.33% |

## Interpretation

- `low_ltw_pruning` is the strongest single mechanism in this screen.
  It achieved the best raw fitness at `32` neurons (`26.0`) while reducing
  active synapses to `98.0`, compared with `163.45` in the prior baseline
  screen.
- `low_ltw_pruning` also performed best at preserving threshold success across
  larger scales: `100%` at `32` neurons and `66.67%` at `64` neurons.
- `active_edge_penalty` reduced active synapses relative to baseline, but it
  reduced selection pressure too bluntly and hurt threshold success.
- `scheduled_sprouting` was useful at larger capacities, especially `64`
  neurons, where it held active synapses to `134.13` and achieved the best
  fitness-per-active-synapse among the component groups at that scale. Its raw
  fitness, however, lagged behind `low_ltw_pruning`.
- Across all groups, hidden-edge fraction still rises sharply with neuron
  count. Larger brains route through hidden nodes, but the current task does
  not yet convert that hidden routing into superior raw fitness.

## Decision

The next combined sparse-efficiency variant should not use the original
`protected_sparse_core` pressure schedule. It should combine the two useful
single mechanisms more gently:

1. Keep low-LTW pruning as the primary efficiency mechanism.
2. Add scheduled sprouting to prevent large edge pools from filling too quickly.
3. Avoid or greatly reduce the active-edge fitness penalty until we calibrate
   its coefficient.
4. Make protected-core behavior softer and capacity-aware.

## Recommended next run

Add or run a gentler combined group:

- low-LTW prune threshold: around `0.08`
- low-LTW prune probability: lower than the aggressive protected core, around
  `0.02-0.03`
- scheduled sprouting: enabled
- active-edge penalty: `0.0` or very small, around `0.005`
- protected core: preserve seeded core topology, but do not force the whole
  organism into a tiny fixed active-edge budget

This should be screened before any 500-generation or 1000-generation full run.
