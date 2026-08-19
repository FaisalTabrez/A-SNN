# Gen-20 spiking spatial-temporal translation preregistration

Date frozen: 2026-08-11
Status: preregistered and implemented; results pending

## Motivation

The bounded full-resolution N-MNIST benchmark established a reproducible dense
upper control at 99.4767% mean test accuracy. Its ConvPLIF candidate screened at
93.07% validation accuracy and was not promoted. Gen-20 tests one new causal
hypothesis: the missing accuracy comes from inadequate separation of spatial
feature extraction and multi-timescale spiking state, rather than from event
data or class separability.

This is not a continuation or retuning of Gen-19. It is a separately theorized
translation of the successful spatial-temporal receptive field into an
inference-time spiking system.

## Frozen representation and split protocol

- Dataset: official N-MNIST train/test split.
- Encoding: native 34x34x2 polarity sensor, 10 binary temporal bins over 300 ms.
- Model selection: training/validation only.
- Official test set: terminal confirmation only.
- Dense reference: the existing spatial-temporal CNN architecture, retrained
  under the same seed and split protocol.
- No structural plasticity, evolutionary mutation, or online adaptation is
  introduced in this experiment.

## Arms

1. `spatiotemporal_cnn`: dense upper control.
2. `conv_plif`: frozen baseline from the accuracy benchmark.
3. `multiscale_residual_plif`: shared 2-D spatial stem applied per temporal bin,
   parallel learnable fast/medium/slow LIF state banks, residual membrane/spike
   fusion, and a population readout.
4. `distilled_multiscale_plif`: the same inference-time spiking architecture,
   trained with label loss plus frozen-teacher logit/feature distillation. The
   teacher is absent at inference and its cost is reported separately.

The analog residual path may exist only in the spatial stem before the first
LIF state. No direct analog path may bypass the spiking temporal banks into the
classifier.

## Bounded screen

- Seed: 220.
- Stratified training subset: 20,000 examples.
- Epochs: 6.
- Selection metric: validation accuracy only.
- The dense upper control and frozen ConvPLIF baseline are always reported.
- At most two new spiking arms are promoted.
- A new arm is eligible only if validation accuracy is at least 97.5%, spike
  density lies between 1% and 30%, and training is numerically stable.
- If neither new spiking arm qualifies, status is `stop` and no rescue sweep is
  authorized.

## Confirmation

- Seeds: 221, 222, 223.
- Full official training split with the frozen validation partition.
- Epochs: 12, with best-validation checkpoint selection.
- The official test set is evaluated once per selected seed/checkpoint.
- The strongest confirmed spiking arm is also evaluated with temporal state
  removed and with input-bin order independently shuffled per sample, without
  retraining.

## Pass gates

All gates are conjunctive:

1. mean test accuracy is at least 99.0%;
2. mean accuracy is within 0.75 percentage points of the paired dense control;
3. every confirmation seed reaches at least 98.7%;
4. spike density remains between 1% and 30%;
5. removing temporal state costs at least one mean accuracy point and passes on
   at least two of three seeds;
6. shuffling temporal order costs at least one mean accuracy point and passes
   on at least two of three seeds;
7. the activity-scaled operation proxy is at least 5x lower than the dense
   spatial-temporal control at inference.

The operation gate remains a proxy. Gen-20 cannot establish wall-plug or
neuromorphic energy efficiency without direct hardware measurement.

## Terminal decisions

- `pass`: the spiking spatial-temporal representation is accurate, causal, and
  sparse enough to justify the next plasticity/continual-learning milestone.
- `stop`: close this translation hypothesis and return to evidence synthesis;
  do not adjust thresholds, promotion margins, or activity bounds after seeing
  the official test results.

Either outcome is one Gen-20 package. No Gen-21 is implied automatically.

## Implementation

The frozen runner is implemented in
`gen5/ammc_gen5/gen20_spiking_spatiotemporal.py`, with the Colab entry point at
`gen5/examples/gen20_spiking_spatiotemporal.py`. It reuses the existing
full-resolution N-MNIST cache, saves validation-selected teacher and student
checkpoints beside the progress file, resumes completed arms and seeds, and
emits JSON/CSV/plot artifacts plus a checksummed ZIP bundle.

## Terminal result (2026-08-20)

Status: `stop` at the preregistered screen. Neither new spiking arm reached the
97.5% validation-accuracy promotion gate, so confirmation and causal controls
were correctly not run. The dense teacher reached 99.1165%, ConvPLIF 96.2160%,
the multiscale residual PLIF 96.3661%, and its distilled counterpart 96.3327%.
The best new arm therefore missed promotion by 1.1339 percentage points;
distillation changed accuracy by -0.0333 points relative to the undistilled
arm. Both new arms had healthy activity (12.69-12.96%) and low operation
proxies, but efficiency cannot compensate for failure of the accuracy gate.

This closes the frozen Gen-20 translation hypothesis without a rescue sweep.
Because no candidate was promoted, Gen-20 supplies no confirmation-set,
state-removal, or temporal-order evidence. The declared next milestone is
program-level evidence synthesis, not an automatic Gen-21.
