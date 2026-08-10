# Gen-8 time-local predictive-binding preregistration

Status (2026-08-10): completed; terminal gate returned `stop`.

The paired time-local LIF candidate failed screening at 7.267% validation
accuracy with a 50.656% spike rate and therefore was not confirmed. The analog
binder showed replicated order sensitivity but no sample-identity specificity.
The branch is closed without a rescue sweep. See the retained
[`analysis.md`](../outputs/gen8_temporal_binding_cuda_2026-08-10/analysis.md).

## Motivation

Gen-7 learned a strongly future-aligned state and slightly exceeded the TCN's
mean accuracy, but its pooled output interaction did not use the correct
sample's state and barely used state order. This separates representation
learning from causal use. Gen-8 tests one new hypothesis: the missing operation
is binding direct evidence to recurrent state at the same time and sample
before temporal aggregation.

This is not a threshold or loss-weight rescue sweep. The predictive objective
and output interaction both change from sequence-global pooling to aligned,
time-local operations. Every threshold below is frozen before execution.

## Hypothesis

A LIF trace trained to predict the same sample at each future timestep, then
fused through `direct[t] * state[t]` before pooling, will make classification
depend beneficially on both state identity and temporal order.

The time-local binding projection is zero-initialized. The successor therefore
starts with logits exactly equal to the matched dilated TCN, while its paired
future objective can train the state path from the first update.

## Registered arms

- `dilated_tcn`: conventional reference.
- `lif_pooled_predictive`: exact Gen-7-style pooled predictive reference.
- `analog_time_local_binding`: non-spiking local-binding control.
- `lif_shuffled_time_local`: identical LIF binding model trained against
  batch-shuffled future targets.
- `lif_time_local_binding`: paired time-local LIF candidate.

All arms use the TCN-selected width. Residual arms remain within 95–105% of the
133,631-parameter target. The paired and shuffled LIF arms differ only in
future-target identity.

## Protocol

Screening uses SSC seed 145, 15,000 training examples, 3,000 validation
examples, 3,000 descriptive test examples, and four epochs. Validation
accuracy, parameter ratio, and LIF activity determine promotion. If the paired
candidate promotes, all mechanistic controls are automatically confirmed.

Confirmation uses complete official SSC splits, 15 epochs, and seeds 145–147.
Validation selects checkpoints. Test values never control promotion or
checkpoint selection.

Every confirmed state arm is evaluated from the same checkpoint in full,
direct-only, state-only, batch-shuffled-state, and time-reversed-state modes.
The local objective compares the paired future against the adjacent samples in
both circular batch directions, avoiding a quadratic batch×batch matrix while
preserving explicit negative identities. Local future alignment is paired
cosine similarity minus one-sample batch-shuffled similarity at matched
timesteps.

## Terminal pass gate

`lif_time_local_binding` passes only if it:

- remains within one mean test point of TCN;
- loses at least 0.5 mean point when state is removed;
- loses at least 0.5 mean point when state identity is shuffled;
- loses at least 0.5 mean point when state time is reversed;
- reproduces each causal loss on at least two of three seeds;
- achieves local future alignment of at least 0.02 on at least two seeds;
- exceeds shuffled-target future alignment by at least 0.01;
- improves both identity specificity and temporal-order sensitivity by at
  least 0.5 point over the pooled Gen-7 reference;
- maintains a 1–30% spike rate;
- learns at least 0.01 mean absolute binding correction.

Passing freezes the model and opens a separately preregistered runtime and
external-replication decision. Failure closes the temporal-binding hypothesis.
No horizon, temperature, gate, threshold, or loss-weight sweep follows.
