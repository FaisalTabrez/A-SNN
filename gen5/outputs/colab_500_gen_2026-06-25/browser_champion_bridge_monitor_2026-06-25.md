# Browser Champion Bridge Replay Monitor — 2026-06-25

Source bundle:

- `champion_connectome.json`
- `colab_weights.json`
- `champion_sparse_adjacency.json`

Browser target:

- `http://127.0.0.1:4173/`

## Setup

This monitor pass was run after adding the Gen-5 browser transducer bridge:

- separate food/toxin directional sensor channels
- imported Gen-5 bridge metadata/inference
- scoped analog motor-readout assist for Gen-5 imports
- finite guards around motor strength and bot physics

Observed browser validation:

- Status toast: `GPU memories injected · 88 LTW updated · STW cleared`
- Neurons: `16`
- Synapses: `88`
- Inspector mean STW: `0.00`
- Inspector mean LTW: `0.14`
- Browser console warnings/errors after monitor: none

## Observation window

Duration: approximately `70` seconds.

Sampling: approximately every `5` seconds.

The run crossed both awake and sleep/offline-replay phases.

## Summary findings

The bridge fix corrected the previous motor-silent / invalid-physics failure.
Motor events were visible, velocity stayed finite, and the bot moved.

Observed outcome:

- Food hits: `0`
- Toxin hits: `0`
- Net visible fitness: `0`
- Velocity finite in all `15/15` samples
- Maximum sampled speed: `1.05`
- Closest food approach: `17 px`
- Closest toxin approach: `47 px`
- Synapses formed: `0`
- Synapses pruned: `0`
- Synapse count remained stable at `88`

Interpretation:

The Gen-5 -> Gen-4 bridge now produces browser motor actuation without corrupting
the physics state. This is a real improvement over the first replay, where the
bot stayed at `0.00, 0.00` velocity and collected only toxin collisions from
moving environment objects.

However, this still is not a successful food-seeking champion demonstration.
The bot moved and avoided toxins in this short window, but did not collect food.
It came within `17 px` of food shortly before sleep, suggesting the next bridge
task is not basic actuation but stronger/fairer sensor-to-motor calibration and
controlled deterministic replay.

## Sampled telemetry

| Sample | Sim time / phase | Event | Food | Toxins | Fitness | Velocity | Speed | Nearest food | Nearest toxin | Queue | Synapses |
| ---: | --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- | ---: | ---: |
| 1 | 19.942s awake | O1 · Motor ← | 0 | 0 | 0 | -0.00, 0.07 | 0.07 | 94 px | 89 px | 42 | 88 |
| 2 | 24.992s awake | O1 · Motor ← | 0 | 0 | 0 | -0.00, 0.07 | 0.07 | 110 px | 90 px | 71 | 88 |
| 3 | 30.055s awake | O1 · Motor ← | 0 | 0 | 0 | -0.00, 0.07 | 0.07 | 127 px | 94 px | 50 | 88 |
| 4 | 35.088s awake | O1 · Motor ← | 0 | 0 | 0 | 0.00, 0.00 | 0.00 | 133 px | 101 px | 58 | 88 |
| 5 | 40.155s sleep | O1 · Sleep · offline replay | 0 | 0 | 0 | 0.00, 0.00 | 0.00 | 132 px | 111 px | 19 | 88 |
| 6 | 45.168s sleep | O1 · Sleep · offline replay | 0 | 0 | 0 | 0.00, 0.00 | 0.00 | 132 px | 111 px | 18 | 88 |
| 7 | 0.183s awake reset | O1 · Motor → | 0 | 0 | 0 | 1.05, 0.00 | 1.05 | 48 px | 67 px | 0 | 88 |
| 8 | 5.240s awake | O1 · Motor ← | 0 | 0 | 0 | 0.00, -0.00 | 0.00 | 47 px | 47 px | 41 | 88 |
| 9 | 10.280s awake | O1 · Motor ← | 0 | 0 | 0 | -0.00, 0.06 | 0.06 | 70 px | 104 px | 35 | 88 |
| 10 | 15.313s awake | O1 · Motor ← | 0 | 0 | 0 | 0.00, 0.07 | 0.07 | 79 px | 98 px | 7 | 88 |
| 11 | 20.379s awake | O1 · Motor ← | 0 | 0 | 0 | 0.00, 0.07 | 0.07 | 92 px | 95 px | 0 | 88 |
| 12 | 25.413s awake | O1 · Motor ← | 0 | 0 | 0 | 0.00, 0.07 | 0.07 | 107 px | 94 px | 0 | 88 |
| 13 | 30.479s awake | O1 · Motor ← | 0 | 0 | 0 | 0.00, 0.06 | 0.06 | 58 px | 97 px | 0 | 88 |
| 14 | 35.546s awake | O1 · Motor ← | 0 | 0 | 0 | 0.00, 0.02 | 0.02 | 41 px | 101 px | 0 | 88 |
| 15 | 40.569s sleep | O1 · Sleep · offline replay | 0 | 0 | 0 | 0.00, 0.00 | 0.00 | 17 px | 83 px | 5 | 88 |

## Comparison to previous browser replay

| Metric | First replay | Bridge replay |
| --- | ---: | ---: |
| Physics validity | finite but stationary | finite and moving |
| Food hits | 0 | 0 |
| Toxin hits | 2 | 0 |
| Max sampled speed | 0.00 | 1.05 |
| Closest food approach | 45 px | 17 px |
| Closest toxin approach | 17 px | 47 px |
| Motor events | none visible as movement | visible Motor ← / Motor → events |

## Recommended next step

Add deterministic browser replay seeds plus a small calibration panel/constant
for Gen-5 bridge motor gain and sensor gain. The bridge now actuates, so the
next question is whether a controlled gain sweep can turn near misses into food
collection without overfitting or causing toxin crashes.

