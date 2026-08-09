# Phase 29 analysis: executable axonal delays

## Result

Heterogeneous recurrent delays pass decisively and improve every seed for both
readouts.

| Arm | Linear accuracy | MLP accuracy | Linear gain | MLP gain |
|---|---:|---:|---:|---:|
| No delay warm | 45.967% | 56.440% | control | control |
| Uniform delay 1 warm | 48.220% | 54.133% | +2.253 pt | -2.307 pt |
| Hash delays 0-2 warm | 53.967% | 63.907% | +8.000 pt | +7.467 pt |
| Distance delays 0-2 warm | 54.053% | 64.033% | +8.087 pt | +7.593 pt |

Every heterogeneous-delay paired delta was positive across all three seeds and
both readouts. Distance-delay gains ranged from `+6.16` to `+9.76` linear and
`+6.36` to `+8.42` MLP points.

## Mechanistic diagnosis

- Uniform delay one helps the linear readout but hurts the MLP. Slower
  recurrence alone is therefore not the explanation.
- Two independently constructed heterogeneous assignments produce almost the
  same large gain. Both distribute roughly one third of recurrent edges across
  delays 0, 1, and 2.
- Event rates remain approximately `0.98-1.02x` the no-delay control for the
  heterogeneous arms.
- LTW movement remains comparable to the no-delay control and saturation stays
  below `0.8%`.
- The result is not obtained by adding edges or neurons: all recurrent arms
  retain the same 272 active edges and readout dimensions.

The evidence supports a causal interpretation: heterogeneous temporal routing
creates a richer final-state representation from the same sparse topology.
This is the first tested AMMC temporal mechanism to provide a large,
readout-independent, multi-seed gain on the conventional benchmark.

## Goal sanity check and decision

The result materially strengthens the project. It does not establish an SNN
state of the art—the raw MLP ceiling remains `94.21%` versus `64.03%` for the
best sparse recurrent arm—but it validates the project's spatiotemporal-delay
hypothesis under controlled conditions.

Phase 30 will optimize per-edge delay assignment through differentiable delay
gates while keeping topology fixed. This is the final MNIST mechanism phase.
After it, the fixed Phase 29 winner or a passing learned-delay winner moves to
SHD, where precise timing is intrinsic to the data.
