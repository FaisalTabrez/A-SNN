# Phase 25: Fixed-Topology LTW Training on Sequential MNIST

Phase 24 established that recurrence provides a large causal benefit when
MNIST images arrive one row per step and only final hidden state is available.
Phase 25 asks the next gated question: can durable LTWs improve that useful
recurrent substrate without destabilizing its activity?

## Paired interventions

For each seed and classifier, all recurrent arms receive the same 272-edge
topology and identical readout initialization:

- `frozen_recurrent`: train the readout while all LTWs remain fixed;
- `warm_all_3em4`: ten readout-only warmup epochs, then update all 272 LTWs at
  `3e-4` for the remaining five epochs;
- `warm_recurrent_3em4`: the same schedule, but update only the 256 recurrent
  LTWs and preserve the 16 input projection weights;
- `raw`: flattened-pixel linear and parameter-matched MLP ceilings.

Topology is fixed, STW remains zero, and LTWs are clamped to `[0, 1]`.
Surrogate-gradient slope `10` is used only for backward propagation; forward
spikes remain hard thresholds. Readouts receive only final hidden spikes and
membrane state.

## Colab CUDA run

```python
%cd /content/A-SNN-phase18
!git pull

!python gen5/examples/sprint25_trainable_sequential_mnist.py \
  --device cuda \
  --seeds 42 43 44 \
  --train-samples 20000 \
  --test-samples 5000 \
  --epochs 15 \
  --warmup-epochs 10 \
  --surrogate-slope 10 \
  --batch-size 512 \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/trainable_sequential_mnist_cuda
```

## Decision rule

LTW learning passes only if an intervention:

- gains at least `0.5` percentage points over its paired frozen recurrent arm
  on average;
- improves at least two of three seeds;
- keeps final/initial hidden event rate within `[0.5, 2.0]`;
- avoids material lower or upper LTW boundary saturation.

A pass justifies a tightly scoped structural-plasticity experiment on the same
sequential task. A failure means the next intervention should improve temporal
credit assignment or optimization—not add or remove edges.
