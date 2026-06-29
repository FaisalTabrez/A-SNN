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
evaluation. Delay `2` has now superseded it as the next full-evaluation
candidate.

## Delay-2 screen note

`gen5/outputs/delayed_reward_delay2_screen_cuda_2026-06-29/analysis.md`
records the clean delay-`2` screen.

Result: both carried-forward groups reached mean best fitness `24.67`.
`gentle_ltw_scheduled` matched `low_ltw_pruning` while using about `20.6%`
fewer active synapses and crossing the threshold in `66.67%` of seeds.
`low_ltw_pruning` crossed threshold faster when it succeeded, but only in
`33.33%` of seeds.

Decision: promote `reward_delay_steps = 2` to the next `10`-seed,
`500`-generation statistical benchmark. If it holds, rerun neuron scaling on
this delayed-reward setting.

## Delay-2 full-run console note

`gen5/outputs/delayed_reward_delay2_full_cuda_console_2026-06-29/analysis.md`
records the surviving console output from the full delay-`2` benchmark.

Caveat: the Colab session ended before downloadable artifacts were recovered,
so this is summary-only evidence. The final printed JSON summary survived.

Result: both groups reached mean best fitness `24.50`. The sparse
`gentle_ltw_scheduled` policy used about `22.3%` fewer active synapses
(`81.40` vs. `104.82`), had lower variance, crossed threshold in more seeds
(`50%` vs. `40%`), and reached threshold earlier on average.

Decision: treat `gentle_ltw_scheduled` at `reward_delay_steps = 2` as the
preferred delayed-reward sparse-efficiency baseline. For publication-grade
evidence, rerun with persistent output or use persistent output on the next
delay-`2` neuron-scaling run.

## Delay-2 neuron-scaling console note

`gen5/outputs/neuron_scaling_delay2_cuda_console_2026-06-29/analysis.md`
records the surviving console output from the delay-`2` neuron-scaling run.

Caveat resolved: the full Drive artifacts were later recovered and archived at
`gen5/outputs/neuron_scaling_delay2_cuda_2026-06-29/analysis.md`.

Result: compact `16`-neuron brains outperformed larger `32`- and `64`-neuron
brains under delayed reward. `gentle_ltw_scheduled/16` reached the best raw
fitness (`25.40`) and threshold success (`80%`), while `low_ltw_pruning/16`
had the best sparse efficiency (`0.331` fitness per active synapse).

Decision: use `16` neurons as the current delayed-reward topology baseline.
Larger hidden-node pools should not be promoted again until a harder world
shows a measurable benefit from hidden state.
