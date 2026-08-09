# Phase 47: residual-state causal contribution on SHD

Archive SHA-256:
`3971FF33F009DC1242BC51F395CB1DBBF417C3B8E6C476315E3D136350BAF1E5`

## Result

| Model | Full | Conv1D reference | Direct only | State only | Shuffled state |
| --- | ---: | ---: | ---: | ---: | ---: |
| Residual LIF | 83.908% | 82.656% | 77.488% | 22.144% | 79.741% |
| Residual analog | 83.746% | 82.656% | 52.680% | 18.802% | 59.364% |

Residual LIF loses 6.419 mean points when state is removed and 4.167 points
when state is shuffled between samples. Every seed clears the one-point
contribution and specificity thresholds. The full model also exceeds the
separately trained Conv1D reference by 1.251 mean points and has only 0.435
points of three-seed standard deviation.

Residual analog loses 31.066 points without state and 24.382 points under
state shuffling. Both direct-only and state-only performance are weak relative
to the full model, demonstrating strong feature co-adaptation.

## Interpretation

Feature removal from a jointly trained classifier causes distribution shift,
so the direct-only result cannot establish unique information by itself. The
shuffled-state ablation is the stronger observation: it preserves the marginal
state distribution but breaks its correspondence to each sample. The
replicated loss shows that sample-specific state is used by the learned
decision. State-only accuracy remains low, so this is a cooperative hybrid
mechanism rather than a standalone SNN result.

## Decision

The SHD causal gates pass. Replicate the matched Conv1D/residual-LIF comparison
and the same fixed-checkpoint ablations on all official Spiking Speech Commands
splits. Until that replication passes, do not generalize the state-contribution
claim beyond SHD or claim an efficiency advantage.
