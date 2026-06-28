# CUDA harder-world benchmark

Date: 2026-06-29

This bundle records the first Sprint 14 harder-world benchmark. It compares
the two frozen sparse-efficiency baselines on four world presets:

- `simple`
- `moving_toxins`
- `delayed_reward`
- `gauntlet`

Run context:

- Device: CUDA / Colab T4 class GPU
- Seeds: `42` through `51`
- Generations: `500`
- Population size: `10000`
- Epoch steps: `120`
- Neuron count: `32`
- Edge capacity: `256`
- Groups: `low_ltw_pruning`, `gentle_ltw_scheduled`
- Completed worlds: `4 / 4`

The uploaded folder was named `harder_words_cuda`; it was normalized in the
repository as `harder_worlds_cuda_2026-06-29`.

Archived files:

- `harder_worlds.json`
- `harder_worlds_records.csv`
- `harder_worlds_summary.csv`
- `harder_worlds_progress.json`
- `colab_log.txt`
- per-world `sparse_efficiency.*` bundles under each world subfolder

## Summary

| World | Group | Final mean best fitness | Std | Active synapses | Fitness / active synapse | Threshold success | Mean generation to threshold |
|---|---|---:|---:|---:|---:|---:|---:|
| simple | gentle_ltw_scheduled | 24.70 | 0.82 | 80.53 | 0.231 | 50% | 245.00 |
| simple | low_ltw_pruning | 26.40 | 2.22 | 103.25 | 0.173 | 80% | 125.38 |
| moving_toxins | gentle_ltw_scheduled | 25.60 | 1.65 | 79.82 | 0.218 | 90% | 191.78 |
| moving_toxins | low_ltw_pruning | 25.50 | 0.97 | 102.97 | 0.168 | 90% | 203.89 |
| delayed_reward | gentle_ltw_scheduled | 20.80 | 1.69 | 81.89 | 0.144 | 10% | 70.00 |
| delayed_reward | low_ltw_pruning | 21.40 | 1.35 | 105.30 | 0.131 | 0% | - |
| gauntlet | gentle_ltw_scheduled | 10.60 | 1.58 | 77.82 | 0.053 | 0% | - |
| gauntlet | low_ltw_pruning | 10.60 | 1.35 | 100.58 | 0.038 | 0% | - |

## Interpretation

- `simple` still favors `low_ltw_pruning` on raw survival: `26.40` mean best
  fitness vs `24.70` for `gentle_ltw_scheduled`. The gentle rule remains more
  sparse, using about `22%` fewer active edges.
- `moving_toxins` did not meaningfully increase task difficulty. Both groups
  reached `90%` threshold success, and `gentle_ltw_scheduled` slightly beat
  low-LTW pruning in raw mean fitness (`25.60` vs `25.50`) while keeping about
  `22.5%` fewer active synapses.
- `delayed_reward` is the first real capability wall. Mean best fitness fell
  to roughly `21`, and threshold success collapsed to `10%` for the gentle
  schedule and `0%` for low-LTW pruning. This is strong evidence that delayed
  credit assignment, not hazard movement, is the next target.
- `gauntlet` is too hard as a direct jump. Both groups collapsed to `10.60`
  mean best fitness with no threshold success. It should become a curriculum
  endpoint, not the next primary benchmark.
- Across all worlds, `gentle_ltw_scheduled` continues to use about `22%` fewer
  active synapses. Its efficiency advantage survives harder environments, but
  it does not solve delayed credit.

## Decision

Promote `delayed_reward` to the next main benchmark axis.

Do not treat `gauntlet` as a pass/fail benchmark yet. The next run should be a
curriculum or sweep that isolates delayed reward difficulty before combining it
with sparse cues and larger arenas.

Recommended next screen:

```bash
python gen5/examples/sprint14_harder_worlds.py \
  --device cuda \
  --worlds simple delayed_reward \
  --groups low_ltw_pruning gentle_ltw_scheduled \
  --seeds 42 43 44 \
  --generations 200 \
  --population-size 10000 \
  --epoch-steps 120 \
  --reward-delay-steps 3 \
  --output-dir gen5_outputs/delayed_reward_delay3_screen_cuda
```

After that, sweep `reward_delay_steps` across `3`, `6`, and `12`, then rerun
neuron scaling only on the delayed-reward setting that is hard but not
collapsed.

