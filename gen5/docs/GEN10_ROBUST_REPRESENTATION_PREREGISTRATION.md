# Gen-10 masked-sensor representation reset preregistration

Status (2026-08-10): implemented and frozen before observing Gen-10 results.

## Hypothesis and boundary

Gen-9 failed before continual-learning memory could be tested because pooled
predictive LIF missed source competence by 6.467 validation points. Gen-10 does
not tune that model. It tests a new hypothesis: a bounded residual spiking
state trained to reconstruct the clean direct representation from randomly
masked sensors can remain source-competent and causally useful under fixed
sensor damage.

STW/LTW, replay, neuromodulation, structural plasticity, and adaptation remain
closed. A Gen-10 pass opens only a separately preregistered Gen-11 adaptation
comparison.

## Frozen arms

- `dilated_tcn`: ordinary conventional reference.
- `dropout_tcn`: the same TCN trained with random sensor masking; this controls
  for augmentation without spiking state.
- `masked_residual_analog`: residual analog-state control trained with masking
  and clean-target alignment.
- `masked_residual_lif`: matched residual LIF candidate trained with masking
  and clean-target alignment.

The residual classifier and direct predictor share weights as in Gen-6. The
alignment loss has no trainable projection: it maximizes cosine agreement
between the masked state mean and detached clean direct-feature mean. Training
randomly masks 20% of sensors per sample. Evaluation uses the unchanged Gen-9
fixed 35% mask (seed 909), which is never exposed as a training mask.

## Screen and confirmation

The screen uses seed 151, 15,000/3,000/3,000 official SSC examples, and five
epochs. Both conventional controls always proceed. A state arm promotes only
if clean validation is within one point and damaged validation within two
points of the best conventional control, parameters remain within 95–105% of
133,631, and LIF activity remains between 1% and 30%.

Confirmation uses complete official splits, 15 epochs, and seeds 151–153.
Checkpoints are selected only by clean validation accuracy. Test measurements
include clean and fixed-damage accuracy, throughput, activity, direct-only
damaged accuracy, and batch-shuffled-state damaged accuracy.

## Terminal gate

`masked_residual_lif` passes only if all conditions hold:

- mean clean and damaged test accuracy are each within one point of the best
  conventional control;
- removing state costs at least 0.5 point, replicated on at least two seeds;
- shuffling state identity costs at least 0.5 point, replicated on at least two
  seeds;
- mean fixed-damage loss is no more than 0.5 point worse than the best
  conventional control's damage loss;
- mean spike rate remains between 1% and 30%.

A pass opens Gen-11 continual adaptation for this frozen representation. A
stop closes this masked-sensor representation and forbids automatic mask-rate,
alignment-weight, threshold, leak, or gate sweeps.
