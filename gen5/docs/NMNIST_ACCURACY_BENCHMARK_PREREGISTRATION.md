# Full-resolution N-MNIST accuracy benchmark preregistration

Status (2026-08-10): implemented and frozen before observing benchmark
results.

## Scope

This is a bounded leaderboard-style side track before Gen-20. It does not
alter or rescue Gen-19. Gen-19 remains a stopped causal identity replication;
this benchmark asks only how accurately the codebase can classify N-MNIST
under a standard full-resolution protocol.

## Input protocol

- Official N-MNIST train and test splits through Tonic.
- Native 34x34 spatial resolution and both event polarities.
- Binary occupancy in 10 uniform bins across the 300 ms recording.
- A stratified 10% validation split is taken from training data.
- The official test set is not evaluated during screening or model selection.
- Training augmentation is fixed at 2% event dropout and a shared random
  spatial translation of at most two pixels.

The encoded full training cache is approximately 1.39 GB as unsigned bytes.
It is written once under the configured Drive data root and reused on resume;
the default root also reuses the raw N-MNIST download from Gen-19.

## Frozen candidates

1. `frame_cnn`: temporal occupancy counts collapsed into a full-resolution
   2-D CNN. This is the static-image accuracy ceiling.
2. `spatiotemporal_cnn`: a 3-D convolutional model retaining time, polarity,
   and spatial axes.
3. `conv_plif`: a convolutional SNN with channel-wise learnable membrane decay
   and thresholds, hard forward spikes, and surrogate gradients.

## Screen and promotion

The screen uses seed 210, 20,000 stratified training samples, the full
validation split, and four epochs. Arms are sorted by best validation
accuracy. At most two arms within one percentage point of the best screen arm
are promoted. Test accuracy cannot influence promotion.

## Confirmation

Promoted arms are trained from scratch on the full training split for ten
epochs using seeds 211-213. Each seed selects its checkpoint by validation
accuracy and evaluates the official test split once. The runner records:

- mean, standard deviation, and minimum test accuracy;
- trainable parameter count;
- dense Conv/Linear MAC proxy;
- activity-scaled operation proxy;
- spiking activity;
- training time and test examples per second.

The practical gate is 99.0% mean test accuracy. The stretch target is 99.4%.
These gates apply to the best promoted model; a separate field records whether
the convolutional spiking candidate was confirmed and its accuracy.

## Terminal rule

This track ends after the one screen/confirmation package. Both `pass` and
`stop` return to Gen-20. No result establishes temporal memory, continual
learning, local credit assignment, structural plasticity, or hardware energy
efficiency.

## Colab command

```python
%cd /content
!rm -rf A-SNN
!git clone https://github.com/FaisalTabrez/A-SNN.git
%cd /content/A-SNN
!pip install -q tonic
!python gen5/examples/nmnist_accuracy_benchmark.py \
  --device cuda \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data/nmnist \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/nmnist_accuracy_benchmark_cuda
```

Progress is saved after every completed arm/seed. The final directory includes
individual JSON/CSV/plot artifacts, SHA-256 hashes, and a single downloadable
`nmnist_accuracy_benchmark_bundle.zip`.
