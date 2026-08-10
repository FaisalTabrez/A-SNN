# Gen-9 continual-adaptation analysis

## Decision

`stop` — zero qualified arms. Only `dilated_tcn` passed source screening, so
the preregistered predictive-LIF adaptation comparison could not proceed.

## Source screen

| Model | Validation | Source test | Damaged test | Damage drop | Activity | Parameters | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dilated TCN | 31.567% | 30.167% | 25.633% | 4.533 pt | 27.612% ReLU | 131,971 | 50,238/s |
| Predictive LIF | 25.100% | 24.733% | 18.433% | 6.300 pt | 8.964% spikes | 134,214 | 18,656/s |

Predictive LIF was parameter-matched and non-degenerate, but it missed the
one-point source-competence gate by 6.467 points relative to TCN.

## Confirmed TCN adaptation controls

Across seeds 148–150, static TCN fell from 57.690% source accuracy to 48.325%
under damage, validating a 9.364-point distribution shift.

| Strategy | Final damaged | Adaptation gain | Adaptation AUC | Source forgetting | Trainable parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| Static | 48.325% | 0.000 pt | 48.325% | 0.000 pt | 0 |
| Readout | 53.927% | 5.601 pt | 52.574% | 2.831 pt | 16,835 |
| Full fine-tune | 56.787% | 8.462 pt | 54.844% | 1.138 pt | 131,971 |

Both trainable controls adapt on all three seeds. Full fine-tuning exceeds the
readout by 2.860 final damaged-accuracy points and 2.270 AUC points while
forgetting 1.693 fewer source points.

## Interpretation and boundary

Gen-9 establishes a reproducible continual-learning shift and useful
conventional baselines. It does not establish AMMC continual learning: the
candidate neural representation failed before memory mechanisms were tested.
Adding STW/LTW, replay, neuromodulation, or structural plasticity would now
mask a base-representation problem and violate the frozen gate.

The next authorized work is evidence closeout. Any later continual-learning
program must begin with a separately preregistered representation that first
matches the conventional source model under the same screening discipline.
