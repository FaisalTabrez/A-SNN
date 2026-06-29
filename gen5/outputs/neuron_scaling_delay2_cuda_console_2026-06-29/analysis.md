# CUDA delay-2 neuron scaling, console-only archive

Date: 2026-06-29

This archive records the surviving console output from the delay-`2`
delayed-reward neuron-scaling run. The run wrote persistent Google Drive
artifacts, but only the console log was provided in this turn, so this bundle
is summary-level evidence until the Drive outputs are uploaded.

Surviving evidence:

- `console_log.txt`: pasted Colab output with the final JSON summary.
- `summary.json`: reconstructed from the printed summary.
- `summary.csv`: reconstructed from the printed summary.

Reported Drive outputs:

- `/content/drive/MyDrive/A-SNN/gen5_outputs/neuron_scaling_delay2_cuda/sparse_efficiency.json`
- `/content/drive/MyDrive/A-SNN/gen5_outputs/neuron_scaling_delay2_cuda/sparse_efficiency_records.csv`
- `/content/drive/MyDrive/A-SNN/gen5_outputs/neuron_scaling_delay2_cuda/sparse_efficiency_summary.csv`
- `/content/drive/MyDrive/A-SNN/gen5_outputs/neuron_scaling_delay2_cuda/sparse_efficiency_summary.png`
- `/content/drive/MyDrive/A-SNN/gen5_outputs/neuron_scaling_delay2_cuda/sparse_efficiency_progress.json`

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

- Delay-`2` did not reverse the earlier neuron-scaling result. Bigger brains
  still do not improve performance on this task.
- The best raw point is `gentle_ltw_scheduled` with `16` neurons:
  `25.40` mean best fitness and `80%` threshold success.
- The best sparse-efficiency point is `low_ltw_pruning` with `16` neurons:
  `52.58` active synapses and `0.331` fitness per active synapse.
- Larger brains increasingly route through hidden nodes. Hidden-edge fraction
  rises from roughly `44%` at `16` neurons to `85%+` at `32` and `94%+` at
  `64`, while direct sensor-motor fraction collapses from roughly `15%` to
  `1-5%`.
- This suggests the current task rewards compact sensor-motor loops more than
  deep latent decision chains. Extra hidden capacity may dilute credit
  assignment instead of improving memory.

## Decision

Treat `16` neurons as the current preferred topology for delay-`2` delayed
reward. `32` neurons is no longer the default delayed-reward baseline unless a
harder world demonstrates a real need for more hidden capacity.

Recommended next steps:

1. Upload the persistent Drive outputs if available so this result can be
   upgraded from console-only to full artifact evidence.
2. Run a focused `16`-neuron confirmation with more seeds or harder delay
   variants.
3. Add a hidden-path diagnostic or regularizer that preserves useful direct
   sensor-motor loops while allowing hidden nodes only when they improve
   delayed-reward performance.
4. Move to a harder world that actually demands memory or state, such as
   delay-`3`, moving food, or a cue-switching task.
