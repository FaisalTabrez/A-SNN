# CUDA delayed-reward delay-2 screen

Date: 2026-06-29

This bundle records a clean Sprint 14 delayed-reward screen using
`reward_delay_steps = 2`.

Run context:

- Device: CUDA / Colab T4 class GPU
- Seeds: `42`, `43`, `44`
- Generations: `200`
- Population size: `10000`
- Epoch steps: `120`
- Neuron count: `32`
- Edge capacity: `256`
- Groups: `low_ltw_pruning`, `gentle_ltw_scheduled`
- Output folder uploaded as `delayed_reward_delay2_screen_cuda`

Archived files:

- `harder_worlds.json`
- `harder_worlds_records.csv`
- `harder_worlds_summary.csv`
- `harder_worlds_progress.json`
- per-world `sparse_efficiency.*` bundle under `delayed_reward/`

## Summary

| Group | Final mean best fitness | Std | Selection best fitness | Active synapses | Fitness / active synapse | Threshold success | Mean generation to threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| gentle_ltw_scheduled | 24.67 | 1.53 | 16.00 | 78.13 | 0.205 | 66.67% | 101 |
| low_ltw_pruning | 24.67 | 1.15 | 16.67 | 98.45 | 0.169 | 33.33% | 70 |

Useful topology diagnostics:

| Group | Hidden-edge fraction | Direct sensor-motor fraction | Active edge utilization |
|---|---:|---:|---:|
| gentle_ltw_scheduled | 81.67% | 8.22% | 30.52% |
| low_ltw_pruning | 83.50% | 6.31% | 38.46% |

## Comparison against nearby delay screens

| Delay | Group | Mean best fitness | Active synapses | Threshold success |
|---:|---|---:|---:|---:|
| 1 | gentle_ltw_scheduled | 23.00 | 77.55 | 0% |
| 1 | low_ltw_pruning | 24.67 | 98.65 | 33.33% |
| 2 | gentle_ltw_scheduled | 24.67 | 78.13 | 66.67% |
| 2 | low_ltw_pruning | 24.67 | 98.45 | 33.33% |
| 3 | gentle_ltw_scheduled | 22.67 | 77.83 | 0% |
| 3 | low_ltw_pruning | 22.33 | 97.97 | 0% |

## Interpretation

- Delay `2` is now the strongest hard-but-not-collapsed delayed-reward
  candidate.
- `gentle_ltw_scheduled` matched `low_ltw_pruning` on raw mean best fitness
  while using about `20.6%` fewer active synapses and achieving the higher
  threshold success rate.
- `low_ltw_pruning` still reaches the threshold faster when it succeeds
  (`70` generations vs. `101`), but it succeeded in fewer seeds.
- Delay `1` favored the raw-survival pruning policy, while delay `3` was still
  too unstable for a short screen. Delay `2` is the first setting where the
  sparse/gentle policy looks behaviorally competitive, not merely efficient.

## Decision

Promote `reward_delay_steps = 2` to the next full statistical benchmark.

Recommended full evaluation:

```bash
python gen5/examples/sprint14_harder_worlds.py \
  --device cuda \
  --worlds delayed_reward \
  --groups low_ltw_pruning gentle_ltw_scheduled \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --generations 500 \
  --population-size 10000 \
  --epoch-steps 120 \
  --reward-delay-steps 2 \
  --output-dir gen5_outputs/delayed_reward_delay2_full_cuda
```

If the full delay-`2` run preserves the short-screen result, rerun neuron
scaling on this delayed-reward setting. Delay `1` remains a fallback if the
full delay-`2` result regresses badly; delay `3` remains the next curriculum
target after delay `2` is stable.
