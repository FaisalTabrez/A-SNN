# Phase 27 analysis: utility-gated structural MNIST

## Result

One-shot absolute-gradient sensor-edge selection failed its paired random-growth
gate. Every guided arm underperformed `random_sensor_48` on every seed for both
readouts.

| Arm | Linear accuracy | MLP accuracy | Linear vs random | MLP vs random |
|---|---:|---:|---:|---:|
| Fixed warm all | 45.967% | 56.393% | -0.767 pt | +0.200 pt |
| Random sensor 48 | 46.733% | 56.193% | control | control |
| Gradient sensor 16 | 45.260% | 55.073% | -1.473 pt | -1.120 pt |
| Gradient sensor 48 | 45.027% | 54.093% | -1.707 pt | -2.100 pt |
| Gradient sensor 48 + prune | 45.360% | 55.220% | -1.373 pt | -0.973 pt |

The paired deficits were negative for all six seed/readout comparisons in each
guided arm. This is stronger evidence than the aggregate means alone.

## Mechanistic diagnosis

- Activity remained healthy: final/initial event-rate ratios ranged from about
  `1.06` to `1.39` and LTW boundary saturation stayed minimal.
- The chosen linear-readout edges ended with mean LTWs around `0.082-0.090`,
  below their `0.1` birth weight. Random linear edges instead rose to about
  `0.127`. Training therefore suppressed many edges ranked highly by the
  one-shot selector.
- MLP gradient scores were much larger than linear scores, yet MLP accuracy was
  worse. Absolute score magnitude was not a reliable utility estimate.
- Peripheral pruning removed exactly 24 of 48 new edges and improved the
  flawed 48-edge guided arm by `+0.333` linear and `+1.127` MLP points. It
  retained the stronger half, but remained below random and fixed controls.

The likely failure is the estimator: absolute gradients at zero weight measure
instantaneous sensitivity, including harmful directions, rather than durable
edge utility. This result rejects this selector and schedule, not every possible
gradient-based rewiring method.

## Goal sanity check and decision

The project goal remains a sparse temporal system whose biological mechanisms
earn their complexity under paired controls. Phase 27 strengthens that process:
we have a reproducible negative result, a working conservative-pruning
mechanism, and evidence that merely adding topology is not the current
bottleneck.

Do not tune or scale this selector. Phase 28 freezes the proven 272-edge graph
and tests fixed adaptive thresholds (ALIF/LSNN-style slow neuron state) against
paired LIF controls. If adaptive state passes, combine it with executable delay
buckets. If it fails, test delays with ordinary LIF neurons.
