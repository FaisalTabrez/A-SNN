# Phase 24 CUDA row-sequential MNIST analysis

Run date: 2026-08-09

Source archive: `sequential_mnist_cuda.zip`

Archive SHA-256: `82FEF360D60C9AD31D23153FBA261C31D9A78FF7C8A9A5B3D5B17F806965A054`

This is a three-seed engineering-validation result. It establishes a causal
direction for the next experiment; it is not a final accuracy or generalization
claim. The reserved final-test complement remains untouched.

## Results

| Representation | Linear accuracy | MLP accuracy | Active edges |
|---|---:|---:|---:|
| Raw flattened | 85.940% | 94.207% | 0 |
| Last row | 23.160% | 44.133% | 0 |
| Integrated rows | 34.787% | 48.327% | 0 |
| Feedforward final state | 32.180% | 38.307% | 16 |
| Recurrent final state | 43.853% | 55.547% | 272 |

Paired recurrent-minus-feedforward effects:

- linear: `+11.673` percentage points, with per-seed gains of `+12.76`,
  `+12.76`, and `+9.50` points;
- MLP: `+17.240` points, with per-seed gains of `+18.34`, `+17.32`, and
  `+16.06` points.

The recurrent state also beats the orderless integrated-row control by
`+9.067` linear points and `+7.220` MLP points. Mean hidden event rate rises
only from `0.014548` to `0.015313` (about `5.3%`), so the large accuracy gain is
not explained by a comparable global activity increase.

## Goal sanity check

- Supported: hidden recurrence causally preserves task-relevant information
  when input arrives sequentially and only final neural state is observable.
- Supported: the final-state protocol removes cumulative spike-count leakage;
  the feedforward and recurrent pairs share topology generation and readout
  initialization.
- Not supported: competitive MNIST accuracy. The recurrent MLP remains `38.66`
  points below the raw-pixel MLP.
- Not supported yet: beneficial LTW learning, structural plasticity,
  continuous learning, catastrophic-forgetting resistance, or broad
  Transformer/SNN superiority.

## Decision

Phase 24 passes its pre-registered recurrence gate by a wide margin for both
readouts and all seeds. Phase 25 therefore keeps the proven 272-edge topology
fixed and applies the stable Phase 22 warm-start schedule (`10` readout-only
epochs, then LTW learning at `3e-4`, surrogate slope `10`). It compares frozen,
all-edge LTW, and recurrent-edge-only LTW arms under paired initialization.
Structural mutation remains deferred until LTW training produces a mean gain
of at least `0.5` points, improves at least two of three seeds, and maintains a
hidden-event-rate ratio in `[0.5, 2.0]` without boundary saturation.
