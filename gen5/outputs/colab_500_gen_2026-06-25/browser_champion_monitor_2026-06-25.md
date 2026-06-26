# Browser Champion Replay Monitor — 2026-06-25

Source bundle:

- `champion_connectome.json`
- `colab_weights.json`
- `champion_sparse_adjacency.json`

Browser target:

- `http://127.0.0.1:4173/`

## Setup

The champion connectome and Colab weight payload were loaded into the Gen-4
browser sandbox.

Observed browser validation:

- Status toast: `GPU memories injected · 88 LTW updated · STW cleared`
- Neurons: `16`
- Synapses: `88`
- Selected inspector neuron: `O1 · N9`
- Inspector mean STW: `0.00`
- Inspector mean LTW: `0.14`
- Plasticity Mode: off during this first replay pass
- Astrocyte overlay: on

## Observation window

Duration: approximately `70` seconds.

Sampling: approximately every `5` seconds.

The run crossed both awake and sleep/offline-replay phases.

## Summary findings

The imported champion bundle was structurally valid and stable in the browser,
but the replayed embodied behavior was not yet an apex food-seeking policy.
During this observation window:

- Food hits: `0`
- Toxin hits: `2` total across the observed day/reset cycles
- Net visible fitness: ended at `-1` in the final observed sleep phase
- Displayed bot velocity: `0.00, 0.00` at every sample
- Synapses formed: `0`
- Synapses pruned: `0`
- Synapse count remained stable at `88`
- Sleep/offline replay was observed at ~`43-48s` and again around `44s` in the
  next cycle

Interpretation:

The champion loaded correctly, but the browser embodiment mapping did not
produce visible motor movement in this replay pass. The toxin collisions appear
to be caused by moving toxin objects reaching a stationary or near-stationary
bot, not by active food-seeking motor control.

This does not invalidate the Colab champion fitness result. It likely points to
a translation gap between the Gen-5 tensor environment and the Gen-4 browser
embodiment:

1. Gen-5 food/toxin channels are mapped onto Gen-4 directional sensors with a
   compatibility layer.
2. The browser motor decoder may require stronger/denser motor neuron spiking
   than the exported recurrent state produces.
3. Gen-5 fitness was measured inside the tensorized environment, while this
   replay uses the Gen-4 visual sandbox physics and sensory transduction.

## Sampled telemetry

| Sample | Sim time / phase | Event | Food | Toxins | Fitness | Velocity | Nearest food | Nearest toxin | Queue | Synapses |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |
| 1 | 30.105s awake | O1 · Toxin · GABA | 0 | 1 | -1 | 0.00, 0.00 | 91 px | 104 px | 51 | 88 |
| 2 | 34.499s awake | O1 · Toxin · GABA | 0 | 1 | -1 | 0.00, 0.00 | 78 px | 96 px | 53 | 88 |
| 3 | 38.914s awake | O1 · Toxin · GABA | 0 | 1 | -1 | 0.00, 0.00 | 66 px | 89 px | 59 | 88 |
| 4 | 42.988s sleep | O1 · Sleep · offline replay | 0 | 1 | -1 | 0.00, 0.00 | 63 px | 87 px | 7 | 88 |
| 5 | 47.377s sleep | O1 · Sleep · offline replay | 0 | 1 | -1 | 0.00, 0.00 | 63 px | 87 px | 6 | 88 |
| 6 | 1.833s awake reset | O1 · Exploring | 0 | 0 | 0 | 0.00, 0.00 | 45 px | 71 px | 0 | 88 |
| 7 | 6.446s awake | O1 · Exploring | 0 | 0 | 0 | 0.00, 0.00 | 40 px | 61 px | 59 | 88 |
| 8 | 11.259s awake | O1 · Exploring | 0 | 0 | 0 | 0.00, 0.00 | 44 px | 45 px | 52 | 88 |
| 9 | 16.069s awake | O1 · Exploring | 0 | 0 | 0 | 0.00, 0.00 | 55 px | 30 px | 47 | 88 |
| 10 | 20.456s awake | O1 · Exploring | 0 | 0 | 0 | 0.00, 0.00 | 45 px | 17 px | 52 | 88 |
| 11 | 25.163s awake | O1 · Toxin · GABA | 0 | 1 | -1 | 0.00, 0.00 | 56 px | 99 px | 51 | 88 |
| 12 | 29.908s awake | O1 · Toxin · GABA | 0 | 1 | -1 | 0.00, 0.00 | 100 px | 112 px | 51 | 88 |
| 13 | 34.632s awake | O1 · Toxin · GABA | 0 | 1 | -1 | 0.00, 0.00 | 90 px | 124 px | 51 | 88 |
| 14 | 39.349s awake | O1 · Toxin · GABA | 0 | 1 | -1 | 0.00, 0.00 | 80 px | 132 px | 53 | 88 |
| 15 | 43.989s sleep | O1 · Sleep · offline replay | 0 | 1 | -1 | 0.00, 0.00 | 79 px | 130 px | 8 | 88 |

## Recommended next test

Run a two-condition replay comparison:

1. Static champion replay, plasticity off, with fixed food/toxin seeds.
2. Champion replay with Plasticity Mode on for at least three day/night cycles.

If movement remains zero in both cases, prioritize the Gen-5 -> Gen-4
transducer compatibility layer before claiming visual transfer of the champion.

