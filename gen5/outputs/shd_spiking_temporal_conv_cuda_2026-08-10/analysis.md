# Phase 45: learned spiking temporal convolution

Archive SHA-256:
`66D5D92F64F290F7EACE46FCED03EF378F93E8E0B039A889B01D82CE457B30DF`

## Result

| Arm | Best-validation test accuracy | Standard deviation | Gain vs Conv1D | Activity | Test examples/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Temporal Conv1D | 82.921% | 0.998 pt | 0.000 pt | 162.19% ReLU magnitude | 49,966 |
| Raw temporal pyramid | 80.374% | 1.838 pt | -2.547 pt | 9.93% input rate | 39,961 |
| Leaky analog state | 76.472% | 1.180 pt | -6.449 pt | 88.63% analog magnitude | 31,155 |
| Dense recurrent LIF | 75.103% | 2.355 pt | -7.818 pt | 21.66% spike rate | 11,835 |
| Leaky convolutional LIF | 74.308% | 1.307 pt | -8.613 pt | 28.96% spike rate | 19,954 |

The convolutional LIF fails the primary gate: it is not within two mean points
of Conv1D or analog, no seed is within two points of Conv1D, and it trails
analog by 2.164 mean points. It also fails the improvement gate against dense
LIF, trailing by 0.795 points with no seed gaining three points.

Activity is not degenerate. Mean LIF spike rate is 28.963%, and the three seeds
range from 24.800% to 32.758%. The more revealing comparison is analog:
stateful integration without thresholding already loses 6.449 points relative
to direct Conv1D. The dominant problem is therefore where state replaces the
successful local representation; spike thresholding adds a secondary penalty.

## Decision

Do not perform a threshold sweep. Run one state-placement diagnostic that
preserves direct Conv1D features beside analog or LIF state. Residual arms must
recover four mean points over state-only and finish within two points of Conv1D.
Failure closes this SHD spiking redesign; recovery permits one component
ablation to determine whether the state branch contributes beyond its bypass.
