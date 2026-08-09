# Phase 44: calibrated SHD temporal baselines

Archive SHA-256:
`558F9DAA53050B9A8F2EA6FE43B85B7FA5AC427615820DB96C6862CC4517FBAD`

## Result

| Arm | Best-validation test accuracy | Standard deviation | Parameters | Test examples/s |
| --- | ---: | ---: | ---: | ---: |
| Temporal Conv1D | 82.847% | 0.930 pt | 133,055 | 51,371 |
| Raw temporal pyramid | 80.374% | 1.838 pt | 132,944 | 37,997 |
| Dense recurrent LIF | 75.103% | 2.355 pt | 133,353 | 11,748 |
| GRU | 46.363% | 0.867 pt | 133,420 | 57,082 |

Temporal Conv1D beats raw on all three seeds by `+5.875`, `+0.707`, and
`+0.839` points. Its mean gain is `+2.473` points, although only one seed meets
the preregistered `+2`-point per-seed criterion. Raw therefore fails its
within-two-points competitiveness gate, but strict replicated ANN dominance is
not established under the more demanding per-seed rule.

Validation selection helps raw substantially (`+2.282` mean points) and dense
LIF (`+1.796` points), but barely changes Conv1D (`+0.103` points). Conv1D's
strong result is therefore not an artifact of rescuing an unstable final
epoch. The GRU's low validation score indicates that this configuration is an
unsuitable control rather than evidence against the GRU family.

## Decision

Learned local temporal filtering is now the minimum architectural target. The
next experiment introduces the same learned temporal front end into matched
leaky analog and LIF state models. The LIF redesign must come within two points
of the 82.847% Conv1D result, maintain 1%-30% spike activity, and beat dense LIF
by at least three points before further spiking-mechanism work is justified.
