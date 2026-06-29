# CUDA delay-2 neuron scaling, full artifact archive

Date: 2026-06-29

This archive records the recovered full Google Drive artifacts from the
delay-`2` delayed-reward neuron-scaling run.

This supersedes the earlier console-only archive at
`gen5/outputs/neuron_scaling_delay2_cuda_console_2026-06-29/`.

Archived files:

- `sparse_efficiency.json`
- `sparse_efficiency_records.csv`
- `sparse_efficiency_summary.csv`
- `sparse_efficiency_summary.png`
- `sparse_efficiency_progress.json`

The progress file reports `60 / 60` completed trials.

## Run context

- Device: CUDA / Colab T4 class GPU
- World preset: `delayed_reward`
- Reward delay: `2`
- Seeds: `42` through `51`
- Generations: `500`
- Population size: `10000`
- Epoch steps: `120`
- Groups: `gentle_ltw_scheduled`, `low_ltw_pruning`
- Scale points: `16/128`, `32/256`, `64/512`

## Summary

| Group | Neurons | Hidden | Edges | Mean best fitness | Std | Active synapses | Fitness / active synapse | Hidden-edge fraction | Direct sensor-motor fraction | Success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gentle_ltw_scheduled | 16 | 4 | 128 | 25.40 | 1.26 | 62.10 | 0.274 | 44.25% | 14.78% | 80% |
| gentle_ltw_scheduled | 32 | 20 | 256 | 23.70 | 0.48 | 81.45 | 0.211 | 84.60% | 5.07% | 0% |
| gentle_ltw_scheduled | 64 | 52 | 512 | 23.50 | 1.08 | 131.33 | 0.117 | 94.08% | 3.04% | 20% |
| low_ltw_pruning | 16 | 4 | 128 | 25.00 | 1.41 | 52.58 | 0.331 | 43.51% | 15.12% | 70% |
| low_ltw_pruning | 32 | 20 | 256 | 24.40 | 0.84 | 104.72 | 0.158 | 85.49% | 4.40% | 40% |
| low_ltw_pruning | 64 | 52 | 512 | 24.60 | 0.84 | 207.71 | 0.080 | 96.05% | 1.39% | 60% |

## Interpretation

- The full artifacts confirm the console-only finding: compact `16`-neuron
  brains remain strongest under delay-`2` delayed reward.
- `gentle_ltw_scheduled/16` has the best raw mean fitness (`25.40`) and
  threshold success rate (`80%`).
- `low_ltw_pruning/16` has the best sparse efficiency (`0.331` fitness per
  active synapse) and the lowest active synapse count (`52.58`).
- Larger brains route much more traffic through hidden nodes:
  hidden-edge fraction rises from roughly `44%` at `16` neurons to `85-96%`
  at `32-64` neurons.
- Direct sensor-motor fraction collapses from roughly `15%` at `16` neurons
  to `1-5%` at larger sizes.
- The current delayed-reward task still rewards compact sensor-motor loops
  more than larger hidden-state capacity.

## Decision

Promote `16` neurons / `128` edge slots as the current delayed-reward topology
baseline.

Use two compact baselines going forward:

- Raw fitness: `gentle_ltw_scheduled`, `16` neurons, `128` edge slots.
- Sparse efficiency: `low_ltw_pruning`, `16` neurons, `128` edge slots.

Do not scale hidden-node count further on the current delay-`2` task. The next
architectural change should either:

1. preserve direct sensor-motor cores while selectively adding hidden pathways,
   or
2. move to a harder world where hidden state is actually required, such as
   delay-`3`, moving food, cue switching, or maze-like partial observability.
