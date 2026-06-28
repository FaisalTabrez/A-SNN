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
