# Phase 22 LTW optimization diagnostic: analysis and goal sanity check

Date: 2026-08-09

Bundle SHA-256: `E521E32EBE7811A1502AC74029A050292CAA0A96B5D95A74E90F470D68CF5274`

The bundle is complete: eight paired arms, linear and MLP readouts, three seeds,
48 seed-level records, 16 aggregate rows, and a readable diagnostic plot.

## Pre-registered gate result

No intervention passes the Phase 22 gate.

| Best or diagnostic arm | Readout | Gain over frozen | Improved seeds | Practical-gain seeds | Event-rate ratio |
|---|---|---:|---:|---:|---:|
| Warm all, `3e-4`, slope 10 | Linear | +0.087 points | 3/3 | 0/3 | 1.036 |
| Warm all, `3e-4`, slope 5 | Linear | +0.080 points | 3/3 | 0/3 | 1.035 |
| Warm sensor, `3e-4`, slope 10 | Linear | +0.060 points | 2/3 | 0/3 | 1.034 |
| Warm all, `3e-4`, slope 10 | MLP | 0.000 points | 2/3 | 0/3 | 1.005 |
| Joint all, `1e-3`, slope 10 | Linear | -0.213 points | 0/3 | 0/3 | 1.714 |
| Joint all, `1e-3`, slope 10 | MLP | -0.260 points | 1/3 | 0/3 | 1.143 |

The required practical gain was at least `0.5` points on average. Every summary
row reports zero practical-gain seeds.

## What the sweep establishes

1. **Warmup solves stability, not usefulness.** Warm interventions hold event
   ratios near `1.0-1.04` and produce zero boundary saturation.
2. **The Phase 21 joint schedule is too aggressive.** It raises linear event
   activity by `71.4%`, saturates approximately `2.8%` of LTWs low and `4.9%`
   high, and reduces both linear and MLP accuracy.
3. **Learning-rate and surrogate-slope differences are negligible.** The best
   linear settings differ by only `0.007` points between slopes 5 and 10.
4. **Sensor-only updates do not reveal a hidden win.** They provide `+0.06`
   linear points and slightly reduce MLP accuracy.
5. **Recurrent-only updates do not help.** They are approximately neutral or
   negative for both readouts.

The result is not “LTWs cannot learn.” It is narrower: on this static MNIST
representation, stable supervised LTW changes do not add useful information
beyond a trained readout.

## Project-goal sanity check

| Project claim area | Phase 22 evidence | Status |
|---|---|---|
| Stable LTW optimization | Warmup controls activity and saturation | Supported |
| Beneficial LTW learning on static MNIST | Best gain is only 0.087 points | Rejected for tested settings |
| Recurrent plasticity | Recurrent-only update is neutral/negative | Unsupported |
| Structural plasticity next | Weight-learning usefulness gate failed | Deferred |
| Continuous temporal learning | Static image classification cannot test it | Not tested |
| Efficiency/superiority | No accuracy gain and extra training cost | Unsupported |

Static MNIST has reached diminishing returns for plasticity research. Continuing
hyperparameter search would optimize an engineering-validation subset without
testing the project's central temporal/adaptive claims.

## Decision

Freeze supervised LTW tuning on static MNIST. Phase 23 performs a causal
recurrence ablation with identical sensor projections and paired readout
initializations:

- sensor temporal state;
- hidden/full temporal state with only sensor-to-hidden feedforward edges;
- hidden/full temporal state with all recurrent edges enabled;
- raw-pixel reference;
- linear and parameter-budget-matched MLP readouts.

If recurrence provides less than `0.5` paired points, move subsequent work to a
task with genuine temporal dependence instead of using static MNIST to justify
recurrent or structural plasticity.

## Validation boundary

The reserved final-test complement remains untouched. Phase 23 is still an
engineering diagnostic, not a final generalization claim.
