# Phase 31 analysis: full SHD temporal transfer

## Result

The registered fixed-delay transfer gate fails. SHD is learnable under the
current preprocessing, but the fixed 0/1/2 delays do not materially improve the
paired sparse recurrent model.

| Arm | Test accuracy | Standard deviation | Effective parameters |
|---|---:|---:|---:|
| Event-count linear | 47.350% | 1.130 pt | 14,020 |
| Event-count MLP | 51.914% | 1.338 pt | 92,308 |
| Sparse no delay | 36.204% | 2.707 pt | 6,352 |
| Sparse distance 0-2 | 36.425% | 2.530 pt | 6,352 |

The fixed-delay paired gains were `+0.398`, `+0.309`, and `-0.044` points for
seeds 42-44, giving a mean of only `+0.221` points. Two seeds improved, but no
seed reached the registered `+1.0` point practical threshold.

## Mechanistic diagnosis

- The count controls reach 47-52%, proving that download, labels, 700-channel
  binning, and optimizer plumbing preserve substantial class information.
- Sparse no-delay accuracy is `11.15` points below count-linear and `15.71`
  points below count-MLP. The dominant gap occurs before or at the sparse
  representation/readout, not in the dataset pipeline.
- Sparse train accuracy is only about 34-36%, so this is underfitting rather
  than train/test overfitting.
- The delay arm executes about 344 delayed recurrent edges with mean delay
  `0.985`, but final event rate is only `1.002x` no delay. It changes timing
  without changing overall activity or class separation materially.
- Hidden event rates remain high at roughly 36-37%. This is stable but may wash
  out temporal selectivity.
- Mean LTW movement is about `0.056`; lower saturation is `0.1%` and upper
  saturation is `3.0-3.4%`. Optimizer collapse is not the explanation.
- Executable delays increase mean training time from 36.2 to 56.8 seconds and
  lower inference throughput by roughly one third, without a practical gain.

## Goal sanity check and decision

The cross-domain result narrows the claim. Fixed heterogeneous delays were
strong on row-sequential MNIST but do not transfer automatically to SHD under
the present 128-neuron, linear-readout configuration. We still have a compact
model with meaningful 36% accuracy at 6,352 effective parameters, but no basis
for a competitive SHD or universal-delay claim.

Decision: reject fixed delays as the next optimization target on SHD. Phase 32
will decompose the representation bottleneck by testing a nonlinear readout,
increased hidden capacity, and lower-activity dynamics while retaining paired
no-delay/delay controls. This distinguishes decoder, capacity, and firing-rate
bottlenecks before we consider architectural redesign or published baselines.
