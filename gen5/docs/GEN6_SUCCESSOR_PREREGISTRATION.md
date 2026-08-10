# Gen-6 successor preregistration

Status (2026-08-10): completed; terminal gate returned `stop`.

The LIF successor matched TCN within 0.065 mean test point and passed its spike
and gate checks. Removing state cost only 0.386 mean point with one of three
seeds clearing the 0.5-point threshold. Shuffling state identity improved
accuracy by 0.657 mean point and no seed cleared the specificity threshold.
The successor therefore has no qualified causal arm, hardware work remains
closed, and no rescue sweep is authorized. See the retained
[`analysis.md`](../outputs/gen6_successor_cuda_2026-08-10/analysis.md).

## Why this is a new generation

Gen-5 established that sample-specific LIF state can complement direct temporal
features, but its residual and hierarchical implementations failed the
Milestone A competitiveness gate. Gen-6 does not tune those rejected models.
It tests a different architectural hypothesis: preserve the strongest TCN
predictor intact and let a parameter-light spiking state act only as a residual
logit correction.

## Hypothesis

A weight-shared, zero-initialized LIF correction can add causal temporal-state
information without destroying the TCN representation that already performs
well.

The successor uses the same two-layer dilated temporal backbone and direct
classifier as the matched TCN. It computes an additional leaky state trace from
the backbone current, pools it using the same temporal pyramid, and projects it
with the direct classifier's existing weight matrix. A trainable per-class gate
scales this correction and begins at exactly zero. Therefore the model begins
as the TCN baseline instead of forcing direct and state features through a new
joint classifier.

## Arms

- `dilated_tcn`: conventional reference.
- `shared_residual_analog`: non-spiking state control with the identical shared
  residual interface.
- `shared_residual_lif`: successor candidate.

All arms target 133,631 trainable parameters. The residual arms add only leak,
threshold, and class-gate parameters; the classifier matrix is shared rather
than duplicated.

## Protocol

The runner first uses seed 142, 15,000 SSC training examples, 3,000 validation
examples, 3,000 descriptive test examples, and four epochs. Promotion uses
validation only. The TCN and any residual candidate within one validation point
of it, within 95–105% of the parameter budget, and—when spiking—within a 1–30%
spike rate advance automatically.

Confirmation uses the complete official SSC train/validation/test splits,
15 epochs, and seeds 142–144. Selected residual checkpoints are evaluated in
full, direct-only, state-only, and batch-shuffled-state modes without
retraining.

## Terminal pass gate

The Gen-6 LIF successor passes only if it:

- remains within one mean test point of TCN;
- loses at least 0.5 mean point when its state correction is removed;
- loses at least 0.5 mean point when state identity is shuffled;
- reproduces each causal loss on at least two of three seeds;
- maintains a 1–30% mean spike rate;
- learns a non-trivial mean absolute class gate of at least 0.01.

Failure closes this successor immediately. No rescue sweep follows. Passing
freezes the model and is the sole condition for reopening hardware-efficiency
work.
