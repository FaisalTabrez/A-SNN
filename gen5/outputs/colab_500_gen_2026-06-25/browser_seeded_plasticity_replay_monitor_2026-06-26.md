# Browser seeded plasticity replay monitor - champion-001

Date: 2026-06-26  
Window: 2026-06-26T05:56:25Z to 2026-06-26T05:58:01Z  
Duration: 20 samples over ~96 seconds  
Browser URL: `http://127.0.0.1:4173/`

## Setup observed

- Replay seed: `champion-001`
- Connectome loaded: `C:\fakepath\champion_connectome.json`
- Weights loaded: `C:\fakepath\colab_weights.json`
- Plasticity Mode: `ON`
- Auto-Evolve: `OFF`
- Calibration:
  - Sensor radius: `0.35`
  - Drag multiplier: `0.985`
  - Spike velocity: `0.05`
- Initial plasticity check before this sample window:
  - Neurons: `16`
  - Synapses: `82`
  - Synapses formed: `2`
  - Synapses pruned: `8`
  - Mean STW: `0.00`
  - Mean LTW: `0.14`
- Console warnings/errors: none observed

## Behavioral findings

This was the first controlled seeded replay where the champion was allowed to
change topology while running. The result is mixed but extremely useful:

1. Structural plasticity was active immediately.
2. The champion collected food during the awake phase.
3. Dopamine reward activated an astrocyte dopamine zone.
4. New synapses sprouted during the dopamine-positive window.
5. Offline replay consolidated short-term traces into LTW at dawn.
6. A later toxin collision triggered a GABA/stress event and suppressed plasticity.

In other words, the full loop is now visible: behavior -> reward/stress ->
astrocyte modulation -> structural churn -> sleep consolidation.

## Key sample observations

| Time in pass | State | Fitness | Food | Toxins | Synapses | Formed | Pruned | Velocity | Event | Notes |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|
| 0s | Night / sleep | +0 | 0 | 0 | 82 | 2 | 8 | `0.00, 0.00` | Sleep replay | Started after earlier pruning/sprouting had already occurred |
| 5s | Dawn / awake | +0 | 0 | 0 | 82 | 0 | 0 | `0.10, 0.03` | Motor down | Dawn log: `0.316 STW consolidated into LTW` |
| 15s | Day / awake | +0 | 0 | 0 | 83 | 1 | 0 | `0.01, 0.03` | Motor left | Axonal sprout: `N7 -> N6.D1` |
| 25s | Day / awake | +1 | 1 | 0 | 84 | 2 | 0 | `-0.01, -0.08` | Food / dopamine | Dopamine zone reached `A5 +1.00`; sprout: `N4 -> N3.D3` |
| 30s | Day / awake | +1 | 1 | 0 | 85 | 3 | 0 | `-0.05, -0.17` | Motor left | Additional sprout: `N4 -> N8.D2` |
| 70s | Night / sleep | +1 | 1 | 0 | 84 | 3 | 1 | `0.00, 0.00` | Sleep replay | One synapse pruned during sleep window |
| 75s | Dawn / awake | +0 | 0 | 0 | 84 | 0 | 0 | `0.16, 0.00` | Motor right | Dawn log: `0.489 STW consolidated into LTW` |
| 85s | Day / awake | -1 | 0 | 1 | 84 | 0 | 0 | `0.31, -0.14` | Motor right | Toxin hit; GABA `A5 -0.93`; plasticity suppressed |

## Quantitative notes

- Minimum sampled nearest-food distance: `29 px`
- Minimum sampled nearest-toxin distance: `19 px`
- Maximum sampled speed magnitude: approximately `0.34`
- Sleep samples observed: `4`
- Day/awake samples observed: `16`
- Motor events observed: left, right, up, down
- Food/dopamine event observed: yes
- Toxin/GABA event observed: yes
- Browser console warnings/errors: none

## Memory and topology notes

- Initial no-plasticity baseline was `88` synapses.
- Plasticity replay had already reduced this to `82` synapses before the main
  sample window.
- During the observed day, synapses increased from `82` to `85` through
  sprouting.
- During the next sleep window, one synapse was pruned, leaving `84`.
- Dawn logs confirmed STW-to-LTW consolidation:
  - `Dawn 2: 0.316 STW consolidated into LTW`
  - `Dawn 3: 0.489 STW consolidated into LTW`
- The dashboard's mean STW stayed displayed as `0.00`, likely because STW was
  transient and/or rounded away at the sampled cadence.
- Mean LTW stayed displayed as `0.14`; consolidation may be too small or too
  distributed to move the rounded dashboard mean.

## Comparison against no-plasticity seeded replay

| Mode | Food observed | Toxins observed | Structural churn | Sleep consolidation | Interpretation |
|---|---:|---:|---|---|---|
| Plasticity OFF | Yes | No | None | None visible | Clean champion behavior transfer |
| Plasticity ON | Yes | Yes | Pruning and sprouting active | Yes | Full learning loop works, but online plasticity changes the champion enough to create risk |

## Interpretation

Plasticity is functioning, but it is aggressive. The replay shows the living
learning system doing exactly what we asked biologically: pruning weak edges,
sprouting under activity, modulating with dopamine/GABA, and consolidating
during sleep. The cost is that the champion's previously safe behavior became
less stable after structural churn, producing a toxin hit in the third observed
day.

The next implementation question is not whether plasticity works; it does. The
next question is how to gate plasticity so a trained champion can adapt without
catastrophically perturbing a high-performing topology.

Recommended next tuning targets:

1. Add a "Champion Stability" or "low plasticity" mode that scales pruning and
   sprouting down for imported champions.
2. Make dopamine reinforcement eligibility easier to inspect per synapse, not
   only as a global toast.
3. Record unrounded STW/LTW means internally so small consolidation changes are
   visible in diagnostics.
4. Run a seeded A/B sweep:
   - plasticity off
   - plasticity on, current rates
   - plasticity on, 25% pruning/sprouting rate
   - plasticity on, dopamine-gated sprouting only

