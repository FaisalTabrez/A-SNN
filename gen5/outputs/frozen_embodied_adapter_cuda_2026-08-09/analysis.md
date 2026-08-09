# Sprint 16 frozen embodied adapter analysis

Date received: 2026-08-09  
Backend: Colab CUDA  
Archive SHA-256: `D0DC736A93C5117AA26C43CB3BEB7B65A43DCD7A99C1665A3A4E425CC63DAD3F`

## Evidence integrity

- 3 worlds: `simple`, `moving_toxins`, `gauntlet`
- 3 sensor-noise levels: `0.0`, `0.05`, `0.15`
- 3 policies: fixed decoder, base adapter, augmented adapter
- 5 held-out seeds: `43` through `47`
- 10,000 agents, 480 steps per evaluation
- 135 seed-level records and 27 aggregate rows
- all expected JSON, CSV, and PNG artifacts are present

## Main results

Across all 45 world/noise/seed evaluations per policy:

| Policy | Overall mean fitness | Positive evaluations | Mean survival rate | Cue-action coverage | Oracle agreement |
|---|---:|---:|---:|---:|---:|
| Augmented adapter | 1.857 | 45/45 | 66.52% | 100.00% | 64.78% |
| Base adapter | 1.764 | 45/45 | 64.80% | 100.00% | 63.55% |
| Fixed motor decoder | -2.807 | 14/45 | 45.57% | 5.18% | 39.22% |

Mean-fitness contrasts by condition:

| World | Noise | Augmented | Base | Fixed | Augmented - Fixed |
|---|---:|---:|---:|---:|---:|
| Gauntlet | 0.00 | 1.391 | 1.166 | -0.744 | +2.135 |
| Gauntlet | 0.05 | 1.312 | 1.147 | -0.544 | +1.856 |
| Gauntlet | 0.15 | 1.358 | 1.040 | -0.321 | +1.679 |
| Moving toxins | 0.00 | 1.752 | 1.543 | -9.750 | +11.502 |
| Moving toxins | 0.05 | 2.107 | 1.458 | -10.809 | +12.916 |
| Moving toxins | 0.15 | 2.277 | 2.586 | -7.290 | +9.567 |
| Simple | 0.00 | 2.225 | 2.357 | 1.858 | +0.367 |
| Simple | 0.05 | 2.313 | 2.458 | 1.557 | +0.756 |
| Simple | 0.15 | 1.981 | 2.119 | 0.779 | +1.201 |

The augmented adapter beat the fixed decoder in 41/45 paired seed conditions;
the base adapter did so in 44/45. Both adapters remained positive in every
evaluation, including `noise_std=0.15`.

## What the result establishes

- A small trained readout can turn the frozen AMMC trace into materially better
  closed-loop behavior than the existing sparse motor-spike decoder.
- The benefit is largest in moving-toxin and gauntlet worlds, where the fixed
  decoder is mostly inactive or strongly negative-fitness.
- The adapter advantage is not only more movement: oracle agreement rises from
  39.22% to 63.55-64.78%. The trace therefore contains action-relevant state
  that the original decoder discards.
- The earlier synthetic robustness result transfers into physics in the limited
  sense that both adapters remain fitness-positive under all tested noise
  levels and held-out seeds.

## What it does not establish

- The result does not prove autonomous learning. Adapter labels came from the
  explicit food-attraction/toxin-repulsion oracle.
- It does not yet isolate representation quality from action coverage. Both
  adapters act on 100% of cue-bearing steps with magnitude `1.0`, while the
  fixed decoder acts on only 5.18% with mean magnitude about `0.052`.
- Augmentation is not a consistent winner over clean training. Its overall
  paired advantage is only `+0.094`, with 28/45 wins. It is consistently useful
  in the gauntlet, but loses most simple-world and moving-toxin/noise-0.15 seed
  comparisons.
- Higher noise sometimes increases food and toxin collisions together. Fitness
  changes under noise cannot be read as improved reasoning without inspecting
  collision balance.

## Decision

Run an activity-matched control phase before external-task claims:

1. retain the fixed spiking decoder;
2. add a full-magnitude random cardinal controller;
3. add a full-magnitude analog fixed-AMMC decoder;
4. add the direct sensor oracle as an upper/control policy;
5. compare these with base and augmented adapters on identical seeds.

This will separate three effects: movement opportunity, direct sensor policy,
and additional value contributed by the frozen AMMC representation.
