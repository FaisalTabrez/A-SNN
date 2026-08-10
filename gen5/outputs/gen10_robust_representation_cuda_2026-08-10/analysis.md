# Gen-10 robust-representation analysis

## Decision

`stop` — only `dilated_tcn` and `dropout_tcn` promoted. Neither residual-state
arm reached confirmation, so no causal state test was run and Gen-11
adaptation did not open under the registered gate.

## Screen

| Arm | Clean validation | Damaged validation | Clean test | Damaged test | Activity | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dilated TCN | 39.567% | 31.800% | 38.667% | 31.067% | 25.483% ReLU | 131,971 |
| Dropout TCN | 47.400% | 41.033% | 44.833% | 40.400% | 32.881% ReLU | 131,971 |
| Masked residual analog | 41.900% | 37.833% | 41.367% | 37.300% | 71.942% analog | 132,038 |
| Masked residual LIF | 38.200% | 34.300% | 37.333% | 33.600% | 11.538% spikes | 132,070 |

Relative to the best conventional screen arm, residual analog missed clean and
damaged validation by 5.500 and 3.200 points. Residual LIF missed by 9.200 and
6.733 points. Both were parameter-matched; LIF activity was healthy. These are
representation-accuracy failures, not capacity or silent-neuron failures.

## Confirmed conventional controls

Across seeds 151–153, sensor dropout improved clean TCN accuracy from 56.918%
to 59.602% (+2.684 points) and damaged accuracy from 46.988% to 55.750%
(+8.763 points). Its damage drop was 3.851 points versus 9.930 for ordinary
TCN, a 6.079-point robustness improvement.

## Goal sanity check

The experiment supports sensor dropout as a strong conventional robustness
intervention. It does not support masked residual analog/LIF state as a
source-competent representation, causal spiking state, continual learning, or
memory consolidation. The original brain-like objective remains open, but the
evidence argues against forcing a spiking state path to relearn the entire
sensory classifier.

The evidence-selected new hypothesis is functional separation: preserve the
source-competent sensor-dropout TCN as a frozen sensory cortex and test a
bounded spiking adapter as a plastic downstream memory/action pathway. This is
not a mask-rate or alignment-loss sweep of Gen-10.
