# Gen-22 dual-memory sequential-shift replication preregistration

Date frozen: 2026-08-20

## Hypothesis

Dual STW/LTW memory improves the stability-plasticity tradeoff relative to an
allocated-slot and update-matched single-memory learner during two sequential
SSC distribution shifts.

## Protocol

Five seeds (421–425) train the supported residual-LIF SSC backbone. The official
validation set is split into disjoint source-selection, shift-A adaptation,
and shift-B adaptation thirds. Two deterministic, non-overlapping 35% sensor
lesions define A and B. The official test labels are evaluation-only.

Arms are static backbone, single-memory gradient control, dual memory, and
dual memory with class-shuffled consolidation. Every arm allocates the same
readout tensors and active slots. Adaptive arms receive five epochs on A and
then five epochs on B with identical orders and batch counts.

## Frozen primary endpoints

After B adaptation, dual memory must:

1. retain shift A at least 1.0 point better than single memory;
2. lose no more than 0.5 point of shift-B accuracy versus single memory;
3. lose at least 0.5 point of A retention when LTW is removed;
4. exceed shuffled consolidation by at least 0.5 point on the mean of A
   retention and B accuracy.

The aggregate means and at least three of five paired seeds must satisfy all
four gates. These rules are frozen before execution.

## Claim boundary

A pass supports dual timescales only in the frozen SSC residual readout. It
authorizes a later backbone-synapse replication, not a combined-mechanism model
or hardware-energy claim. A stop closes or redesigns the present consolidation
rule without a rescue sweep.
