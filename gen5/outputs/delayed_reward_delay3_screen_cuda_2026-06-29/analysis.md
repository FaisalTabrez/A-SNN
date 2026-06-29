# CUDA delayed-reward delay-3 screen

Date: 2026-06-29

This bundle records a short Sprint 14 delayed-reward screen using
`reward_delay_steps = 3`.

Run context:

- Device: CUDA / Colab T4 class GPU
- Seeds: `42`, `43`, `44`
- Generations: `200`
- Population size: `10000`
- Epoch steps: `120`
- Neuron count: `32`
- Edge capacity: `256`
- Groups: `low_ltw_pruning`, `gentle_ltw_scheduled`
- Output folder uploaded as `delayed_reward_delay3_screen_cuda`

Important interpretation note:

The command used a global `--reward-delay-steps 3` override. That means both
the `simple` and `delayed_reward` world labels resolved to the same effective
world configuration:

- `world_size`: `1.0`
- `sensor_radius`: `0.35`
- `moving_toxin_speed`: `0.0`
- `reward_delay_steps`: `3`

So this run should be read as two replicate delay-3 screens, not as a clean
comparison between no-delay and delayed-reward worlds.

Archived files:

- `harder_worlds.json`
- `harder_worlds_records.csv`
- `harder_worlds_summary.csv`
- `harder_worlds_progress.json`
- per-world `sparse_efficiency.*` bundles under `simple/` and
  `delayed_reward/`

## Summary

| World label | Group | Final mean best fitness | Std | Active synapses | Fitness / active synapse | Threshold success |
|---|---|---:|---:|---:|---:|---:|
| simple | gentle_ltw_scheduled | 24.00 | 2.65 | 77.37 | 0.215 | 33.33% |
| simple | low_ltw_pruning | 23.67 | 1.15 | 98.09 | 0.170 | 33.33% |
| delayed_reward | gentle_ltw_scheduled | 22.67 | 0.58 | 77.83 | 0.197 | 0% |
| delayed_reward | low_ltw_pruning | 22.33 | 0.58 | 97.97 | 0.174 | 0% |

Combined across the two identical effective configurations:

| Group | Effective seeds | Mean best fitness | Active synapses | Threshold success |
|---|---:|---:|---:|---:|
| gentle_ltw_scheduled | 6 | 23.33 | 77.60 | 16.67% |
| low_ltw_pruning | 6 | 23.00 | 98.03 | 16.67% |

## Interpretation

- Delay `3` is meaningfully easier than the previous full delay-`12` run, where
  final mean best fitness was roughly `21` and success was near zero.
- Delay `3` is still not solved in a short `200`-generation screen. The
  combined threshold success is only `16.67%`.
- `gentle_ltw_scheduled` slightly outperformed `low_ltw_pruning` on raw mean
  fitness in both replicate labels while using about `21%` fewer active edges.
- Because the two world labels had identical effective configs, any difference
  between `simple` and `delayed_reward` rows should be treated as stochastic
  run variance, not an environment effect.

## Decision

Delay `3` is a plausible hard-but-not-collapsed candidate, but it is still too
uncertain to crown. The next move should be a cleaner delay-length sweep that
does not duplicate identical configs under different world labels.

Recommended next screen:

```bash
python gen5/examples/sprint14_harder_worlds.py \
  --device cuda \
  --worlds delayed_reward \
  --groups low_ltw_pruning gentle_ltw_scheduled \
  --seeds 42 43 44 \
  --generations 200 \
  --population-size 10000 \
  --epoch-steps 120 \
  --reward-delay-steps 1 \
  --output-dir gen5_outputs/delayed_reward_delay1_screen_cuda
```

Repeat for `--reward-delay-steps 2` and `3`. If delay `2` or `3` reaches
non-trivial success without collapsing, run the selected setting at `10` seeds
and `500` generations, then revisit neuron scaling.

