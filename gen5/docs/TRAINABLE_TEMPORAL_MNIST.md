# Phase 21: Fixed-Topology LTW Training

Phase 20 showed that AMMC state carries useful temporal information, but its
random frozen dynamics still trail the raw-pixel MLP. Phase 21 asks the next
narrow question: can task-trained long-term synaptic weights improve that
temporal representation without changing the graph topology?

## Experimental boundary

- The sparse source/target edge list is fixed for the entire run.
- STW is disabled and held at zero.
- Only active LTWs and the readout are optimized.
- A hard spike is used in the forward pass and a fast-sigmoid surrogate
  derivative is used during backpropagation.
- LTWs are clamped after each optimizer step and inactive edge slots remain
  zero.

This is not yet a structural-plasticity experiment. Sprouting and pruning are
deferred so that any improvement can be attributed to trainable dynamics.

## Comparison groups

The runner evaluates matched seeds for raw pixels, frozen AMMC temporal state,
and LTW-trained AMMC temporal state, each with linear and MLP readouts. It
records accuracy, active edges, readout and optimizer parameter counts, hidden
event rate, LTW displacement, training time, and inference throughput.

`effective_trainable_parameters` counts active LTWs rather than unused
capacity slots; `optimizer_parameters` reports the actual fixed-capacity
tensors.

## Colab CUDA run

```python
%cd /content/A-SNN-phase18
!git pull
!python gen5/examples/sprint21_trainable_temporal_mnist.py \
  --device cuda \
  --seeds 42 43 44 \
  --train-samples 20000 \
  --test-samples 5000 \
  --epochs 15 \
  --batch-size 512 \
  --reservoir-learning-rate 0.001 \
  --surrogate-slope 10 \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/trainable_temporal_mnist_cuda
```

If the checkout is elsewhere, replace the `%cd` path with the directory that
contains this repository's `gen5` folder.

## Decision rule

Advance to topology plasticity only if the trained-LTW groups improve over
their paired frozen groups while hidden event activity remains stable and the
gain repeats across seeds. If LTWs move but accuracy does not improve, or event
activity collapses, diagnose the dynamics/gradient path before adding topology
mutation.

The three-seed run is engineering validation, not a final statistical claim.
After selecting settings, confirm them with more seeds and an untouched test
split. The raw MLP remains the external performance reference.
