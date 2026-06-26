# AMMC-SNN Behavior Observation — Corrected Parser

Started: 2026-06-25T05:53:08.570Z

Ended: 2026-06-25T05:54:58.898Z

Duration: 110s, 19 samples at 5s intervals.

## Key findings

- Topology remained stable at the imported 21-edge evolved connectome during the corrected observation window.
- Mean LTW stayed high and flat, indicating the imported Colab weights persisted as long-term memory.
- STW fluctuated during runtime, showing short-term activity/replay continued on top of the fixed LTW baseline.
- No food/toxin collisions occurred during this 90-second corrected window; reward/punishment outcome remains inconclusive.
- Sleep/replay phase was captured with body motion suppressed and offline replay events visible.
- Awake embodied movement phase was captured after the sleep cycle.
- Motor actuation was observed (O1 · Motor ↓; O1 · Motor ↑), so the GPU-injected network is driving the bot.
- 13/19 samples showed non-zero bot velocity.
- 17/19 samples showed local GABA modulation, likely suppressing hyperactive local spiking.

## Metrics

- Initial phase: DAY · AWAKE · 8.3 S
- Final phase: DAY · AWAKE · 16.6 S
- Initial fitness: 0
- Final fitness: 0
- Food hit delta: 0
- Toxin hit delta: 0
- Synapse counts observed: 21
- Mean STW range: 0.07 to 0.13
- Mean LTW range: 0.86 to 0.86
- Regional spike-load range: 0 to 9
- Closest food approach: 16 px
- Closest toxin approach: 42 px
- Non-zero velocity samples: 13/19
- GABA-modulated samples: 17/19

## Distinct visible events

- O1 · Motor ↓
- O1 · Sleep · offline replay
- O1 · Exploring
- O1 · Motor ↑

## Final visible state

```json
{
  "allTimeHigh": "—",
  "assignedRole": "Sensor · North ↑",
  "chemicalState": "GABA -0.09",
  "cycleRemaining": "26.6 s",
  "event": "O1 · Motor ↓",
  "evolutionLog": [
    "Dawn 11: 0.000 STW consolidated into LTW",
    "Night 11: Offline replay · sensory channels muted",
    "Dawn 12: 0.000 STW consolidated into LTW",
    "Night 12: Offline replay · sensory channels muted",
    "Dawn 13: 0.000 STW consolidated into LTW"
  ],
  "fitness": 0,
  "foodHits": 0,
  "formed": "Formed: 0",
  "generation": 1,
  "growthFactor": 0,
  "meanLtw": 0.86,
  "meanStw": 0.13,
  "motorActivity": 0,
  "nearestAstrocyte": "A5 -1.00",
  "nearestFoodPx": 94,
  "nearestToxinPx": 71,
  "neurons": "Neurons: 9",
  "phase": "DAY · AWAKE · 16.6 S",
  "pruned": "Pruned: 0",
  "queue": "Queue: 37",
  "regionalSpikeLoad": 8.9,
  "sampleIndex": 18,
  "selectedNeuron": "O1 · N1",
  "sensoryDrive": 0.7,
  "simTime": "Time: 23.356 s",
  "simulationState": "Simulation awake",
  "synapses": "Synapses: 21",
  "toast": "Night phase · bodies sleeping · hippocampal replay active",
  "toxinHits": 0,
  "velocity": "0.00, 0.27",
  "wallTimeIso": "2026-06-25T05:54:58.882Z"
}
```
