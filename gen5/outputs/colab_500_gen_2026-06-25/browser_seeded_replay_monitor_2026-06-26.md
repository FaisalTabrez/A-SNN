# Browser seeded replay monitor — champion-001

Date: 2026-06-26  
Window: 2026-06-26T05:48:36Z to 2026-06-26T05:49:52Z  
Duration: 16 samples over ~75 seconds  
Browser URL: `http://127.0.0.1:4173/`

## Setup observed

- Replay seed: `champion-001`
- Connectome loaded: `C:\fakepath\champion_connectome.json`
- Weights loaded: `C:\fakepath\colab_weights.json`
- Calibration:
  - Sensor radius: `0.35`
  - Drag multiplier: `0.985`
  - Spike velocity: `0.05`
- Champion topology:
  - Neurons: `16`
  - Synapses: `88`
  - Mean STW: `0.00`
  - Mean LTW: `0.14`
- Plasticity Mode: `OFF`
- Auto-Evolve: `OFF`
- Console warnings/errors: none observed

## Behavioral findings

The replay was valid and active. The dashboard reported:

- `Seeded replay · champion-001 · topology preserved`
- Food acquired during the observed awake windows.
- No toxin hits were observed.
- Sleep replay triggered cleanly, froze body movement, muted sensory channels, and reset the day cycle after dawn.

Because Plasticity Mode was OFF, this should be interpreted as a clean champion bridge replay rather than an online structural-learning run. Synapses formed/pruned stayed at `0/0`, and STW remained `0.00`.

## Key sample observations

| Time in pass | Circadian state | Fitness | Food | Toxins | Velocity | Event | Notes |
|---:|---|---:|---:|---:|---|---|---|
| 0s | Day · Awake · 3.1s remaining | +0 | 0 | 0 | `0.07, 0.05` | `O1 · Motor ←` | Champion moving near end of day |
| 5s | Night · Sleep · 9.8s remaining | +1 | 1 | 0 | `0.00, 0.00` | `O1 · Sleep · offline replay` | Food acquired before sleep; body frozen |
| 20s | Day · Awake · 38.9s remaining | +0 | 0 | 0 | `0.07, 0.00` | `O1 · Motor →` | Dawn reset after sleep |
| 40s | Day · Awake · 24.7s remaining | +1 | 1 | 0 | `-0.07, -0.29` | `O1 · Motor ←` | Food collected in second sampled day |
| 75s | Night · Sleep · 9.4s remaining | +1 | 1 | 0 | `0.00, 0.00` | `O1 · Sleep · offline replay` | Second sleep phase begins |

## Quantitative notes

- Minimum sampled nearest-food distance: `31 px`
- Minimum sampled nearest-toxin distance: `29 px`
- Maximum sampled speed magnitude: approximately `0.30`
- Motor events observed: `←`, `→`, `↑`, plus sleep replay
- Sleep replay phases observed: 2
- Mean LTW remained stable at `0.14`
- Mean STW remained `0.00`

## Interpretation

This run materially improves on the earlier motor-silent replay. With the Gen-5 calibration constants and deterministic replay seed, the champion:

1. loaded the correct 16-neuron / 88-synapse topology,
2. retained imported Colab LTWs,
3. generated visible motor output,
4. collected food without toxin collisions in the sampled window, and
5. transitioned cleanly through sleep/offline replay.

The next useful experiment is to run the same seed with Plasticity Mode ON, then compare:

- food/toxin rate,
- STW accumulation,
- LTW consolidation after night,
- synapses formed/pruned,
- whether the dopamine event finds eligible synapses for reinforcement.

