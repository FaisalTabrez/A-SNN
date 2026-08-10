# Gen-7 predictive-state preregistration

Status: protocol frozen before training.

## Motivation

Gen-6 preserved the matched TCN predictor but failed its causal-state gate.
The learned LIF correction was active, yet batch-shuffling state improved
accuracy. This rejects another unstructured residual-state rescue. Gen-7 tests
a different mechanism: state is assigned an explicit temporal job and may
influence logits only through a sample-conditioned interaction.

## Hypothesis

A multi-timescale LIF trace trained to predict the same sample's future encoder
features will become beneficially sample-specific. A sample-conditioned gate
computed from direct/state agreement will add the correction only where that
state agrees with the current example.

The direct two-layer dilated TCN and classifier remain intact. The correction
gate is initialized to exactly zero, so every residual arm begins with the TCN
logits. The LIF channels receive heterogeneous initial leak constants. A small
future projection is trained with symmetric in-batch contrastive prediction
from early state to later encoder features.

## Registered arms

- `dilated_tcn`: conventional reference.
- `lif_no_predictive`: identical LIF architecture with classification loss
  only.
- `analog_paired_predictive`: non-spiking paired-future control.
- `lif_shuffled_predictive`: LIF model whose auxiliary target identities are
  deliberately batch-shuffled during training.
- `lif_paired_predictive`: Gen-7 candidate using correctly paired targets.

All residual arms use the same TCN width and remain within 95–105% of the
133,631-parameter reference budget. The paired/shuffled LIF arms differ only
in target identity, not architecture or optimizer.

## Protocol

Screening uses SSC seed 142, 15,000 training examples, 3,000 validation
examples, 3,000 descriptive test examples, and four epochs. Promotion uses
validation accuracy, parameter budget, and LIF activity only. If the paired
LIF candidate is promoted, its no-predictive and shuffled-predictive controls
are also confirmed even if their screen accuracies are lower, because they are
required to interpret the mechanism.

Confirmation uses the complete official SSC train/validation/test splits,
15 epochs, and seeds 142–144. Checkpoint selection uses validation accuracy.
Test results never control promotion or checkpoint selection.

Every confirmed state arm is evaluated without retraining in:

- full mode;
- direct-only mode;
- state-only mode;
- batch-shuffled-state mode;
- time-reversed-state mode.

Future alignment is the mean paired cosine similarity minus one-step
batch-shuffled cosine similarity. It is measured from the selected checkpoint,
not used for checkpoint selection.

## Terminal pass gate

`lif_paired_predictive` passes only if it:

- remains within one mean test point of TCN;
- loses at least 0.5 mean point when state is removed;
- loses at least 0.5 mean point when state identity is shuffled;
- loses at least 0.5 mean point when the state trace is time-reversed;
- reproduces each causal loss on at least two of three seeds;
- achieves a future-alignment margin of at least 0.02 on at least two seeds;
- exceeds the shuffled-target arm's mean future-alignment margin by at least
  0.01;
- maintains a 1–30% mean spike rate;
- learns a mean absolute sample-conditioned gate of at least 0.01.

Passing freezes the candidate and opens a separately designed runtime
milestone. Failure closes this Gen-7 hypothesis. No threshold, loss-weight,
horizon, or architecture rescue sweep follows this run.
