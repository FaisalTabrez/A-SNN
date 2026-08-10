# Gen-13 three-factor local-plasticity preregistration

Status (2026-08-10): completed with terminal decision `stop`.

## Hypothesis

Gen-12 showed that coarse retrieval is not enough, while ordinary readout
adaptation reliably repairs the damaged task. Gen-13 asks whether the useful
credit assignment can be localized to output synapses rather than propagated
through the sensory backbone.

The candidate uses a manual three-factor update:

`Δw = learning_rate × presynaptic_trace × postsynaptic_class_error`

The postsynaptic factor is target minus predicted class probability. The
implementation performs no autograd operation and never updates the frozen
sensor-dropout TCN. This is still supervised class error, not a claim of
reward-only or fully biologically local learning.

## Frozen comparison

Seeds are 160–162. Source training, 20% sensor dropout, fixed 35% sensor mask
at seed 909, official SSC splits, budgets `0, 64, 256, 1024, 4096`, and three
passes over each new block match Gen-12.

Five strategies are registered:

1. `dropout_tcn_static`;
2. `dropout_tcn_readout` using the established AdamW/autograd control;
3. `dropout_tcn_full_finetune`;
4. `analog_three_factor_readout`;
5. `spiking_three_factor_readout`.

Local arms use learning rate 0.50 and weight decay 0.0001. The larger local
rate compensates for unit-normalized traces and minibatch-averaged outer
products; it is frozen before observing results. Analog presynaptic
traces are normalized frozen features. Spiking traces use the frozen Gen-12
rank-order encoder with exactly 20% active units, then unit normalization.
Fast weights start at zero and are active only in the known damaged context.
They are experimental fast state—not STW/LTW with decay or consolidation.

## Causal controls

At each budget the runner measures:

- fast weights removed;
- fast output-class rows cyclically shuffled;
- source accuracy with fast weights context-gated off;
- trace activity, active fast synapses, mean weight magnitude, update time,
  and inference throughput.

## Terminal gate

The spiking local rule passes only if all conditions hold:

- static damage drop is at least 2 points;
- mean gain is at least 2 points and repeats on 2/3 seeds;
- adaptation AUC and final damaged accuracy are each within 1 point of the
  autograd readout control;
- source forgetting is no more than 0.5 point worse than readout adaptation;
- removing fast weights costs at least 0.5 point on average and on 2/3 seeds;
- shuffling output classes costs at least 0.5 point on average and on 2/3 seeds;
- spiking trace density remains between 5% and 35%.

A pass opens a separately preregistered STW/LTW consolidation experiment on
the qualified local fast weights. A stop closes this local-rule branch without
learning-rate, density, epoch, normalization, damage, or budget sweeps. No
best-SNN, autonomous continual-learning, or biological-plausibility claim is
authorized by a pass alone.

## Observed terminal decision

The spiking rule gained 0.410 point, with 0/3 seeds reaching the registered
two-point threshold. Fast-weight removal cost 0.410 point and class shuffling
cost 0.468 point, also below their replicated 0.5-point gates. Trace density
was healthy at exactly 20%, source forgetting was zero, and the fast matrix
was broadly occupied. The stored decision is `stop`; the branch is closed
without rescue sweeps. See `GEN13_LOCAL_PLASTICITY_ANALYSIS.md`.
