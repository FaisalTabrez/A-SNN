# Gen-9 continual-adaptation preregistration

Status (2026-08-10): implemented and frozen before observing Gen-9 results.

## Program boundary

Gen-9 returns to AMMC's original objective: adaptation during an organism's
lifetime. It does not reopen the rejected Gen-6–Gen-8 static-classifier
architecture branch. This first Gen-9 milestone asks whether the predictive
LIF representation already demonstrated in Gen-7 adapts more efficiently than
the matched TCN representation under controlled sensor damage.

STW/LTW consolidation, neuromodulation, sleep replay, and structural
plasticity remain closed until this representation-level adaptation gate
passes. This prevents later biological mechanisms from masking an unsuitable
base representation.

## Shift and data separation

The source task is ordinary 64-bin SSC. The shifted task permanently zeros 35%
of the 700 input sensor channels, selected once by seed 909. Labels and event
timing otherwise remain unchanged.

- Official SSC training data trains the source model.
- Official validation data supplies the adaptation stream.
- Official test data is evaluated in undamaged and damaged forms.
- Test accuracy never selects a source checkpoint or adaptation budget.

Adaptation is sequential at cumulative budgets `0, 64, 256, 1024, 4096`.
Each newly observed block is used for three epochs and is not replayed in later
blocks. This measures adaptation from finite new experience rather than
retraining repeatedly on the entire accumulated dataset.

## Registered source models

- `dilated_tcn`: matched conventional reference.
- `predictive_lif`: the Gen-7 pooled predictive LIF architecture with paired
  four-bin future prediction, fixed auxiliary weight 0.20, and temperature
  0.10.

The screen uses seed 148, 15,000/3,000/3,000 source train/validation/test
examples, and four source epochs. Predictive LIF promotes only if it remains
within one validation point of TCN, stays within 95–105% of the 133,631
parameter target, and maintains a 1–30% spike rate.

## Registered confirmation strategies

- `tcn_static`: no adaptation.
- `tcn_readout`: update only the TCN classifier.
- `tcn_full_finetune`: conventional all-parameter BPTT upper control.
- `predictive_lif_static`: no adaptation.
- `predictive_lif_readout`: update only the identically sized classifier;
  predictive dynamics and the state gate remain frozen.

Confirmation uses complete official splits, 15 source epochs, and seeds
148–150. Source checkpoints are selected only by undamaged validation
accuracy. The fixed adaptation learning rate is 0.001.

## Measurements

At every adaptation budget the runner records:

- damaged-task accuracy;
- undamaged source accuracy;
- event/spike activity;
- inference throughput;
- cumulative adaptation time;
- trainable adaptation parameters.

Adaptation AUC is the trapezoidal damaged-accuracy curve over the linear sample
budget. Forgetting is source accuracy at zero samples minus source accuracy
after 4,096 samples.

## Terminal pass gate

`predictive_lif_readout` passes only if:

- sensor damage reduces static TCN accuracy by at least five points, validating
  that the shift is non-trivial;
- source predictive-LIF accuracy remains within one point of source TCN;
- its final damaged accuracy improves by at least two points over its own
  zero-sample result on at least two of three seeds;
- its mean adaptation AUC exceeds TCN readout by at least one point, replicated
  on at least two seeds;
- its final damaged accuracy remains within one point of TCN readout;
- its source forgetting is no more than 0.5 point worse than TCN readout;
- its final damaged spike rate remains between 1% and 30%.

Passing opens a separately preregistered STW/LTW memory milestone. Failure
closes predictive-LIF continual adaptation and prevents automatic addition of
memory, replay, structural plasticity, or a sensor-mask severity sweep.
