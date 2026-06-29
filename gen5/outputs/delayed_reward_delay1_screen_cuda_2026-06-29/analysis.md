# CUDA delayed-reward delay-1 screen

Date: 2026-06-29

This bundle records a clean delayed-reward screen using only
`--worlds delayed_reward` and `reward_delay_steps = 1`.

Run context:

- Device: CUDA / Colab T4 class GPU
- Seeds: `42`, `43`, `44`
- Generations: `200`
- Population size: `10000`
- Epoch steps: `120`
- Neuron count: `32`
- Edge capacity: `256`
- Groups: `low_ltw_pruning`, `gentle_ltw_scheduled`
- Effective world: `delayed_reward`
- Reward delay: `1` step

Archived files:

- `harder_worlds.json`
- `harder_worlds_records.csv`
- `harder_worlds_summary.csv`
- `harder_worlds_progress.json`
- `delayed_reward/sparse_efficiency.*`

## Summary

| Group | Final mean best fitness | Std | Active synapses | Fitness / active synapse | Threshold success | Mean generation to threshold |
|---|---:|---:|---:|---:|---:|---:|
| gentle_ltw_scheduled | 23.00 | 1.00 | 77.55 | 0.232 | 0% | - |
| low_ltw_pruning | 24.67 | 1.15 | 98.65 | 0.193 | 33.33% | 158.00 |

## Interpretation

- Delay `1` is not collapsed. `low_ltw_pruning` reached `24.67` mean best
  fitness and crossed the threshold in `1 / 3` seeds.
- Delay `1` is still not solved in this short `200`-generation screen.
- Unlike the delay-`3` replicate screen, `low_ltw_pruning` clearly beats
  `gentle_ltw_scheduled` on raw survival at delay `1`.
- `gentle_ltw_scheduled` remains substantially sparser: `77.55` active edges
  versus `98.65`, or about `21%` fewer active synapses.

## Comparison with earlier delay screens

| Delay setting | Group | Mean best fitness | Threshold success | Notes |
|---:|---|---:|---:|---|
| 1 | low_ltw_pruning | 24.67 | 33.33% | Best short-screen survival so far |
| 1 | gentle_ltw_scheduled | 23.00 | 0% | Sparser, but weaker survival |
| 3 | low_ltw_pruning | 23.00 | 16.67% | Combined duplicate-config screen |
| 3 | gentle_ltw_scheduled | 23.33 | 16.67% | Combined duplicate-config screen |
| 12 | low_ltw_pruning | 21.40 | 0% | Full harder-world run |
| 12 | gentle_ltw_scheduled | 20.80 | 10% | Full harder-world run |

## Decision

Delay `1` is the best full-evaluation candidate so far, but delay `2` should
be screened before spending the larger `10`-seed, `500`-generation budget.

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
  --reward-delay-steps 2 \
  --output-dir gen5_outputs/delayed_reward_delay2_screen_cuda
```

If delay `2` underperforms delay `1`, promote delay `1` to the next
`10`-seed, `500`-generation evaluation and then rerun neuron scaling on that
setting.

