# CUDA delayed-reward delay-2 full run, console-only archive

Date: 2026-06-29

This archive records the surviving console output from the full
`reward_delay_steps = 2` delayed-reward benchmark. The Colab session ended
before the downloadable JSON/CSV/PNG artifacts were recovered, so this should
be treated as summary-level evidence rather than a complete artifact bundle.

Surviving evidence:

- `console_log.txt`: pasted Colab output with the full final JSON summary.
- `summary.json`: reconstructed from the printed summary.
- `summary.csv`: reconstructed from the printed summary.

Missing evidence:

- Per-generation records.
- Full `harder_worlds.json`.
- Full `harder_worlds_records.csv`.
- Plots.

## Run context

- Device: CUDA / Colab T4 class GPU
- World: `delayed_reward`
- Reward delay: `2`
- Seeds: `42` through `51`
- Generations: `500`
- Population size: `10000`
- Epoch steps: `120`
- Neuron count: `32`
- Hidden neurons: `20`
- Edge capacity: `256`
- Groups: `low_ltw_pruning`, `gentle_ltw_scheduled`

## Summary

| Group | Final mean best fitness | Std | Selection best fitness | Active synapses | Fitness / active synapse | Threshold success | Mean generation to threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| gentle_ltw_scheduled | 24.50 | 1.08 | 16.30 | 81.40 | 0.200 | 50% | 169.20 |
| low_ltw_pruning | 24.50 | 1.43 | 17.40 | 104.82 | 0.166 | 40% | 201.25 |

Topology diagnostics:

| Group | Hidden-edge fraction | Direct sensor-motor fraction | Active edge utilization |
|---|---:|---:|---:|
| gentle_ltw_scheduled | 84.48% | 5.14% | 31.80% |
| low_ltw_pruning | 85.25% | 4.42% | 40.94% |

## Interpretation

- The full `10`-seed, `500`-generation run preserves the main delay-`2`
  screen result: `gentle_ltw_scheduled` matches `low_ltw_pruning` on raw mean
  best fitness while using substantially fewer active synapses.
- `gentle_ltw_scheduled` used about `22.3%` fewer active synapses than
  `low_ltw_pruning` (`81.40` vs. `104.82`).
- `gentle_ltw_scheduled` also had lower variance, higher threshold success
  (`50%` vs. `40%`), and earlier mean threshold crossing (`169.20` vs.
  `201.25` generations).
- `low_ltw_pruning` still had the higher final mean selection-best fitness
  (`17.40` vs. `16.30`), so it may remain useful as a raw-selection baseline.

## Decision

Treat `gentle_ltw_scheduled` at `reward_delay_steps = 2` as the preferred
delayed-reward sparse-efficiency baseline.

Because this is console-only evidence, do not use it as the final publication
artifact. The next run should either:

1. rerun delay-`2` with persistent Drive output or automatic zipping, or
2. proceed to neuron scaling under delay-`2` while using persistent output from
   the beginning.

Recommended next scientific step:

```bash
python gen5/examples/sprint13_sparse_efficiency_ablation.py \
  --device cuda \
  --world-preset delayed_reward \
  --groups gentle_ltw_scheduled low_ltw_pruning \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --generations 500 \
  --population-size 10000 \
  --epoch-steps 120 \
  --neuron-counts 16 32 64 \
  --max-edges 128 256 512 \
  --reward-delay-steps 2 \
  --output-dir gen5_outputs/neuron_scaling_delay2_cuda
```
