# Gen-19 N-MNIST residual-state replication preregistration

Status (2026-08-10): implemented and frozen before observing Gen-19 results.

## Program boundary

Gen-18 closed the local reward-credit program. Gen-19 does not tune or replace
that learner. It starts the previously identified external event-benchmark
program and tests the strongest supported AMMC result: causal residual LIF
state inside a hybrid temporal model.

N-MNIST is a real event-camera conversion of MNIST and Tonic documents its
sensor as `34×34×2`. Events are binned into 30 fixed time intervals and an
`8×8×2` polarity/spatial grid. This tests cross-modality event-vision transfer.
It is not treated as proof of rich temporal reasoning because fixed acquisition
motion can make N-MNIST timing less discriminative than natural event streams.

References:

- Tonic N-MNIST documentation:
  <https://tonic.readthedocs.io/en/main/getting_started/nmnist.html>
- Tonic dataset API and sensor size:
  <https://tonic.readthedocs.io/en/develop/autoapi/tonic/datasets/nmnist/>
- Temporal-content caution:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC8027306/>

## Frozen protocol

- Official N-MNIST train and test splits; no sample cap by default.
- Seeds: `190, 191, 192`.
- Input: binary occupancy in `30×128` event cells.
- Validation: stratified 10% of the official training split.
- Training: 8 epochs, AdamW `0.003`, weight decay `0.0001`, batch `256`.
- Parameter target: `133,631`, matching the established SHD/SSC comparison.
- Temporal pooling levels: `1, 2, 4`; Conv1D kernel: `5`.
- Arms: matched temporal Conv1D and residual LIF.
- Residual ablations: full, direct-only, state-only, and batch-shuffled state.

Cached event tensors are keyed by every encoding and sampling parameter. Raw
data and caches are not result evidence and must not be committed. The result
progress file is validated against the full configuration and resumes after
each completed seed.

## Pass gates

Every gate must pass:

1. The matched Conv1D reaches at least 90% mean test accuracy.
2. Residual LIF remains within one mean accuracy point of Conv1D.
3. Removing state costs at least 0.5 mean point and this repeats on at least
   two of three seeds.
4. Shuffling state between samples costs at least 0.5 mean point and this
   repeats on at least two seeds.
5. Mean LIF activity remains between 1% and 30%.

## Decision

- `pass`: freeze a cross-modal residual-state mechanism and proceed to a
  publication/hardware milestone without reopening local reward credit.
- `stop`: limit the residual-state contribution claim to event audio and move
  directly to publication closeout or a separately theorized architecture.

Accuracy alone cannot pass Gen-19.

## Colab command

```bash
pip install -q tonic
python gen5/examples/gen19_nmnist_state_replication.py \
  --device cuda \
  --output-dir gen5_outputs/gen19_nmnist_state_replication_cuda
```
