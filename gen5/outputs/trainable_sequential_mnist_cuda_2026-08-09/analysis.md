# Phase 25 CUDA sequential LTW-training analysis

Run date: 2026-08-09

Source archive: `trainable_sequential_mnist_cuda.zip`

Archive SHA-256: `5664F8985D4D3D739D5B6507317717DC1ED39DD7526BF4706342004E93566A08`

This is a three-seed engineering-validation result. The reserved final-test
complement remains untouched.

## Results

| Intervention | Linear accuracy | Linear gain | MLP accuracy | MLP gain |
|---|---:|---:|---:|---:|
| Frozen recurrent | 43.853% | 0.000 | 55.533% | 0.000 |
| Warm all LTWs | 45.967% | +2.113 | 56.427% | +0.893 |
| Warm recurrent LTWs | 44.353% | +0.500 | 55.533% | +0.000 |
| Raw ceiling | 85.940% | +42.087 | 94.207% | +38.673 |

The all-edge intervention improves all three seeds for both classifiers. Its
per-seed linear gains are `+2.78`, `+1.66`, and `+1.90` points; MLP gains are
`+1.40`, `+1.02`, and `+0.26` points. The MLP therefore improves every seed and
reaches the `0.5`-point practical threshold in two of three seeds.

The recurrent-only intervention is much weaker. Linear gains are `+0.76`,
`-0.56`, and `+1.30` points, while MLP gains are `-0.06`, `+0.10`, and `-0.04`.
Mean MLP improvement is effectively zero.

## Stability and localization

- Warm-all final/initial event-rate ratio is `1.285` for linear and `1.087` for
  MLP, inside the allowed `[0.5, 2.0]` range.
- Warm-all upper-bound LTW saturation is only `0.613%` linear and `0.245%` MLP;
  lower-bound saturation is zero.
- Sensor-edge movement dominates: mean sensor LTW change is `0.10045` linear
  and `0.04576` MLP, versus recurrent changes of `0.01488` and `0.00415`.
- Recurrent-only learning keeps activity almost unchanged but does not deliver
  a robust MLP benefit.

## Goal sanity check

- Supported: fixed-topology LTW learning can improve a causally useful
  recurrent substrate on a true sequential task.
- Supported: the primary optimization bottleneck is currently the sparse
  sensor-to-hidden projection, not the recurrent core.
- Not supported: competitive MNIST accuracy; the trained recurrent MLP remains
  `37.78` points below the raw MLP.
- Not supported yet: beneficial synaptogenesis, pruning, continuous online
  learning, retention, or resistance to catastrophic forgetting.

## Decision

Phase 25 passes the durable-weight gate through the warm-all intervention.
Phase 26 therefore unlocks a tightly scoped structural experiment: preserve
the original 272-edge recurrent core, sprout additional sensor-to-hidden edges
after the ten-epoch readout warmup, and compare 16-edge and 48-edge sensor
growth against fixed-topology LTW learning. A 64-edge recurrent-sprouting arm
serves as a localization control. Pruning remains disabled so growth has an
unambiguous causal interpretation.
