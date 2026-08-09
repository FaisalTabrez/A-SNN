# Phase 41 analysis: fixed-budget sparse width scaling

Archive SHA-256: `562E5813608B99ADB8EC54BB1C5A9ABDAE66B98F54C4A2DE5CE566B8B6E3A5FF`

## Registered gates

| Gate | Required | Observed | Result |
|---|---:|---:|---|
| 512 vs 128 width | +2 points; two +1 seeds | +14.959; three +1 seeds | Pass |
| 1024 vs 512 | +1 point; two positive seeds | +0.839; three positive seeds | Fail narrowly |
| Best width vs raw temporal | +2 points | +0.736 | Fail |

## Interpretation

Accuracy rises sharply from `62.898%` at width 128 to `74.823%` at 256 and
`77.856%` at 512, then reaches `78.696%` at 1024. All arms remain within 0.75%
of the fixed `133,631` effective-parameter target, so the early gain is a real
representation-width effect rather than additional parameter allocation.

Topology diagnostics explain the shape. Width 128 forces about `5.48` sensors
into each connected hidden node and produces `70.0%` mean analog activation.
At width 1024, about 514 nodes are connected, fan-in falls to `1.36`, and mean
activity falls to `19.2%`. Width reduces destructive sensor collisions, but the
benefit saturates once most sensor channels have distinct representational room.

The absolute result is not robust. Raw temporal reaches `77.959%`, statistically
matching 512 and trailing 1024 by only `0.736` points. The Phase 40 512 result
of `81.140%` is not reproduced. Constructor/readout RNG order differs between
the two phases, identifying initialization as a likely hidden variable.

## Goal sanity check

Fixed-budget width scaling is supported, but a dependable sparse advantage is
not. This remains an analog feature-transform result, not evidence for spiking,
plasticity, recurrence, or neuromorphic efficiency. Phase 42 must establish
initialization robustness before further architectural development.
