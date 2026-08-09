# Phase 36 analysis: parameter-matched SHD temporal pyramid

Archive SHA-256:
`419F26034B7FA13995025808901C58BFCD23A049414B7D94DEE572B88B76D8C7`

## Registered-gate result

Phase 36 passes both the representation and causal chronology gates.

| Arm | Mean accuracy | Gain vs same-width global | Parameters vs global |
|---|---:|---:|---:|
| 256 global | 52.282% | - | 100.0% |
| 256 ordered pyramid | 76.193% | +23.910 points | 99.72% |
| 512 global | 60.777% | - | 100.0% |
| 512 ordered pyramid | 80.065% | +19.287 points | 99.38% |
| 512 fixed-shuffled pyramid | 73.807% | +13.030 points | 99.38% |

The ordered 512-neuron pyramid exceeds global pooling by `+19.287` points.
All three paired seed gains are large (`+18.772`, `+17.270`, and `+21.820`
points), comfortably clearing the registered `+3`-point mean and two-seed
gates.

The ordered model also exceeds the parameter-identical fixed-shuffle control
by `+6.257` points. All three seed gains clear five points (`+5.080`, `+5.256`,
and `+8.436`), so the causal chronology gate passes.

## Interpretation

The largest verified SHD bottleneck was temporal collapse at the readout, not
hidden capacity or axonal delays. Coarse multi-scale temporal position carries
far more class information than the global hidden-spike mean.

The fixed shuffle still scores `73.807%`, `+13.030` points over global pooling.
This control does not erase temporal information: it applies the same
permutation to every example, so the decoder can still learn position-specific
statistics. It does disrupt natural chronology and local temporal continuity
during recurrent propagation. The ordered-over-shuffled gain therefore
supports a real chronology-sensitive component, while the large shuffled gain
shows that time-resolved readout features are the dominant improvement.

The result is not explained by parameter count, firing inflation, or LTW
saturation:

- the 512 pyramid uses `99.38%` of the global effective parameter count;
- final hidden event rate falls from `14.00%` to `13.16%`;
- mean LTW movement falls from `0.0227` to `0.0095`;
- upper LTW saturation falls from `0.170%` to `0.024%`;
- inference throughput falls about 9%, not the roughly 40% delay penalty.

## Goal sanity check

This is the strongest SHD result in the project and robustly validates
time-aware decoding. It does not yet prove that the recurrent AMMC reservoir is
responsible for the 80.1% result. A trainable temporal readout may extract much
of the same information directly from raw binned events or from a feedforward
sensor projection. Until those controls are run, this is evidence for the
combined AMMC-plus-temporal-readout system, not evidence that recurrence or
structural plasticity is uniquely valuable.

An `80.065%` three-seed result is a substantial internal advance but remains
below strong published SHD systems. State-of-the-art claims remain unsupported.

## Decision

Phase 37 will perform a causal temporal-control decomposition at 512 neurons:
event-count MLP, parameter-matched raw-event temporal pyramid, global AMMC,
feedforward AMMC temporal pyramid with recurrent edges disabled, and recurrent
AMMC temporal pyramid. The phase must separate readout value, sparse sensor
expansion, and recurrent value before further accuracy optimization.
