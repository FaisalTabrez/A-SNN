# Phase 46: SHD temporal-state placement

Archive SHA-256:
`F90F7D11AE2040E0AFD567F0053C7709C3144DBEDB79E8C19A7D655C0C78D312`

## Result

| Arm | Best-validation test accuracy | Standard deviation | Gain vs Conv1D | Gain vs state-only | Test examples/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Residual LIF | 83.804% | 1.520 pt | +0.942 pt | +8.525 pt | 19,587 |
| Residual analog | 83.142% | 1.007 pt | +0.280 pt | +5.933 pt | 27,371 |
| Conv1D | 82.862% | 0.862 pt | 0.000 pt | — | 51,781 |
| Analog state-only | 77.208% | 1.500 pt | -5.654 pt | 0.000 pt | 30,627 |
| LIF state-only | 75.280% | 0.739 pt | -7.582 pt | 0.000 pt | 18,618 |

Residual analog passes the preregistered recovery gate: it recovers 5.933 mean
points, two seeds clear four points, and all three are within two points of
Conv1D. Residual LIF passes more strongly, recovering 8.525 points with all
three seeds over four points and within two points of Conv1D.

The residual LIF mean is 0.942 points above Conv1D, but paired gains are
`-1.413`, `+4.196`, and `+0.044`. This is viability, not robust superiority.
Its 25.736% spike rate is healthy, while its inference throughput is about 62%
lower than Conv1D.

## Decision

The direct-feature bypass repairs the state-placement failure. It is still
unknown whether analog or spiking state contributes to the learned decision.
Run a fixed-checkpoint component ablation: remove direct features, remove state
features, and shuffle state between samples. The state claim survives only if
full accuracy beats both direct-only and shuffled-state evaluation by one mean
point with the effect replicated on two seeds.
