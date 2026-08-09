# Phase 49: SSC matched baseline and efficiency audit

Archive SHA-256:
`55D3D7F0F68D525628715AC44129D89893C46640C98FE1FC978A7529C6FF1DC5`

## Result

| Arm | Test accuracy | Parameters | Dense MAC proxy | Throughput |
| --- | ---: | ---: | ---: | ---: |
| Dilated TCN | 59.225% ± 0.541 pt | 131,971 | 7.381M | 53,080/s |
| Residual LIF | 55.973% ± 0.018 pt | 133,087 | 6.527M | 16,682/s |
| Conv1D | 48.948% ± 1.695 pt | 132,893 | 7.409M | 48,318/s |

The dilated TCN beats residual LIF by 3.253 mean points. Its paired gains are
2.630, 3.160, and 3.969 points, so all three seeds exceed the preregistered
two-point threshold. Residual LIF fails the final predictive competitiveness
gate despite exceptionally low three-seed variance.

Residual LIF has an 11.569% lower dense-MAC proxy than TCN, but adds 1,856 state
updates and averages 95.5 spike events per sample. More importantly, the dense
PyTorch implementation is 3.182 times slower than TCN on the T4. The MAC proxy
does not constitute a latency or energy advantage.

## Final sanity check

The cross-dataset causal state result remains valid because Phase 49 does not
repeat or contradict the fixed-checkpoint ablations. What fails is the broader
claim that this residual implementation is the best matched predictor or an
efficient software system.

## Decision

Close empirical tuning for this milestone. Produce a machine-readable claim
ledger and final report distinguishing:

- supported cross-dataset residual-state complementarity;
- rejected standalone-LIF and matched-baseline-superiority claims;
- rejected current software-throughput advantage;
- proxy-only arithmetic reduction; and
- untested hardware energy efficiency.
