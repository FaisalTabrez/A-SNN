# Phase 18 frozen event-coded MNIST: analysis

Date: 2026-08-09

Bundle SHA-256: `36243164048BB5B08604E2DC5BB76550965BAB30B0B521ACBBC6CA2567EA88DB`

The bundle is complete: three reservoir/readout seeds, four model families,
12 seed-level records, the aggregate table, and the rendered comparison plot.
The run used 20,000 training and 5,000 official-test examples per seed.

## Result

| Model | Test accuracy | Trainable parameters | Hidden spike rate |
|---|---:|---:|---:|
| Raw pixel linear | 85.94% +/- 0.12% | 650 | n/a |
| Raw pixel MLP | 95.14% +/- 0.18% | 34,210 | n/a |
| Frozen AMMC linear | 79.31% +/- 0.45% | 2,570 | 2.37% |
| Frozen AMMC MLP | 86.11% +/- 0.66% | 34,186 | 2.37% |

The reported deviations are population standard deviations across seeds, as
defined by the Phase 18 runner.

Paired seed differences:

- frozen AMMC linear minus raw linear: `-6.63` percentage points; per-seed
  differences `[-6.52, -6.02, -7.36]`; exploratory 95% paired t interval
  `[-8.32, -4.95]` points;
- frozen AMMC MLP minus parameter-matched raw MLP: `-9.03` points; per-seed
  differences `[-8.44, -8.52, -10.14]`; exploratory interval
  `[-11.42, -6.65]` points;
- frozen AMMC MLP minus frozen AMMC linear: `+6.80` points, showing that the
  frozen trace retains nonlinear information even though it is inferior to
  the raw representation.

## Interpretation

- Phase 18 fails both predefined success rules. The current frozen sparse
  reservoir does not improve linear separability and does not beat a
  parameter-budget-matched raw-pixel MLP.
- The reservoir is active rather than completely silent, but a mean hidden
  spike rate of `2.37%` is low and varies with topology seed (`2.09%` to
  `2.61%`).
- The result does not identify whether the loss occurs in latency coding,
  collapsing temporal traces into summary features, or recurrent dynamics.
- The AMMC MLP's precomputed-feature inference rate is about `0.70x` the raw
  MLP rate. More importantly, the reservoir feature pass runs at roughly
  75,800 examples/s when the first-seed compilation/warm-up cost is included;
  the existing readout-only inference figure is not an end-to-end throughput
  measure.
- The small train/test gaps show no obvious severe overfitting. The dominant
  issue is representation quality, not readout memorization.

## Decision

Do not increase neuron count or enable plasticity yet. Implement Phase 19 as an
event-representation decomposition using the same frozen reservoir and data
split. Compare raw intensity, flattened latency events, sensor trace, hidden
trace, full trace, and raw-plus-hidden residual features with both linear and
parameter-budget-matched MLP heads.

This decomposition distinguishes:

1. information lost by event quantization;
2. information lost when temporal events are collapsed into sensor summaries;
3. degradation or added information from recurrent hidden dynamics;
4. whether hidden activity is complementary to raw pixels even when it is not
   sufficient on its own.

Only after that result should Phase 20 tune event coding, firing thresholds, or
recurrent gain.

## Statistical boundary

The paired intervals above are exploratory because only three seeds were run.
They support a stable directional result, not a publication-grade uncertainty
claim. Full MNIST and larger multi-seed validation should follow only after the
representation bottleneck is corrected.
