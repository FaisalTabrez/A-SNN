# Harder bot-world benchmarks

The simple 2D foraging world is now solved well enough that extra hidden
neurons do not improve fitness. Sprint 14 changes the pressure: keep the
current sparse-efficiency baselines fixed and make the world harder.

## Baselines to carry forward

- Raw-survival baseline: `low_ltw_pruning`, `32` neurons, `256` edge slots.
- Sparse-efficiency baseline: `gentle_ltw_scheduled`, `32` neurons, `256`
  edge slots.

## World presets

| Preset | Purpose |
|---|---|
| `simple` | Original simple foraging world. Use as the regression control. |
| `wide_arena` | Larger arena with unchanged sensor range; search is harder. |
| `sparse_cues` | Smaller sensor radius; partial-observability pressure. |
| `moving_toxins` | Toxins drift and bounce; hazard tracking matters. |
| `delayed_reward` | Food reward is delayed; memory traces matter. |
| `gauntlet` | Combined pressure: larger arena, sparse cues, moving toxins, delayed reward. |

## Recommended Colab run

```python
!python gen5/examples/sprint14_harder_worlds.py \
  --device cuda \
  --worlds simple moving_toxins delayed_reward gauntlet \
  --groups low_ltw_pruning gentle_ltw_scheduled \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --generations 500 \
  --population-size 10000 \
  --epoch-steps 120 \
  --output-dir gen5_outputs/harder_worlds_cuda
```

For a quicker screen:

```python
!python gen5/examples/sprint14_harder_worlds.py \
  --device cuda \
  --worlds moving_toxins gauntlet \
  --groups low_ltw_pruning gentle_ltw_scheduled \
  --seeds 42 43 44 \
  --generations 200 \
  --population-size 10000 \
  --epoch-steps 120 \
  --output-dir gen5_outputs/harder_worlds_screen_cuda
```

## Outputs

The runner writes one checkpointed sparse-efficiency run per world plus a
combined summary:

- `harder_worlds.json`
- `harder_worlds_records.csv`
- `harder_worlds_summary.csv`
- `harder_worlds_progress.json`
- `<world>/sparse_efficiency.*`

## What to look for

The key question is whether hidden decision nodes become useful once the world
requires memory, search, and hazard tracking.

Strong evidence would look like:

- `32`-neuron baselines outperforming smaller brains on hard presets,
- `gentle_ltw_scheduled` retaining most raw fitness while using fewer edges,
- larger future neuron counts improving `gauntlet` or delayed-reward results,
- hidden-edge usage correlating with fitness instead of merely increasing with
  capacity.

## First CUDA result

The first full CUDA run is archived at:

- `gen5/outputs/harder_worlds_cuda_2026-06-29/analysis.md`

Main conclusion:

- `moving_toxins` is not yet a meaningful capability wall.
- `delayed_reward` is the first hard setting: mean best fitness dropped to
  roughly `21` and threshold success collapsed.
- `gauntlet` is too hard as a direct jump and should be treated as a curriculum
  endpoint.

Next recommendation: sweep `reward_delay_steps` across `3`, `6`, and `12`,
then rerun neuron scaling only on the delay setting that is hard but not fully
collapsed.

## Delay-3 screen note

`gen5/outputs/delayed_reward_delay3_screen_cuda_2026-06-29/analysis.md`
records a short `reward_delay_steps = 3` screen.

Important caveat: the CLI override was global, so both requested worlds
resolved to the same effective delay-3 environment. Treat the rows as
replicates, not distinct world conditions.

Result: delay `3` is less collapsed than delay `12`, but still only reached
`16.67%` combined threshold success in the short `3`-seed, `200`-generation
screen. Next sweep delay `1`, `2`, and `3` with only `--worlds delayed_reward`.

## Delay-1 screen note

`gen5/outputs/delayed_reward_delay1_screen_cuda_2026-06-29/analysis.md`
records the clean delay-`1` screen.

Result: `low_ltw_pruning` is the best short-screen survival result so far,
with mean best fitness `24.67` and `33.33%` threshold success. It is not solved,
but it is the first delayed-reward setting that looks close enough for a full
evaluation. Screen delay `2` before committing the larger run.
