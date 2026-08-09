# Phase 28: Adaptive Neurons on Sequential MNIST

Phase 27 found that one-shot absolute-gradient sprouting was consistently
worse than paired random sprouting. The failure was not caused by dead activity
or LTW saturation, so Phase 28 returns to the proven fixed 272-edge topology
and tests whether slow neuron state improves temporal capacity.

## Paired design

Every recurrent arm uses the same topology, initial LTWs, readout shape, data,
and seed. Adaptive LIF (ALIF) neurons add a fixed activity trace that raises
their threshold after firing:

```text
a[t+1] = 0.95 * a[t] + spike[t]
threshold[t] = 1.0 + 0.5 * a[t]
```

The trace has no trainable parameters. The readout still sees only final spikes
and membrane, so LIF and ALIF have identical readout parameter counts.

The arms are:

- `raw`: flattened-pixel ceiling;
- `lif_frozen`: fixed-LTW LIF control;
- `lif_warm_all`: Phase 25 fixed-topology LTW control;
- `alif50_frozen`: isolate adaptive dynamics without LTW training;
- `alif25_warm_all`, `alif50_warm_all`, `alif100_warm_all`: adaptive-neuron
  dose response with the Phase 25 warm-start LTW schedule.

## Colab CUDA run

```python
%cd /content/A-SNN
!git pull

!python gen5/examples/sprint28_adaptive_sequential_mnist.py \
  --device cuda \
  --seeds 42 43 44 \
  --train-samples 20000 \
  --test-samples 5000 \
  --epochs 15 \
  --warmup-epochs 10 \
  --adaptation-decay 0.95 \
  --adaptation-strength 0.5 \
  --batch-size 512 \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/adaptive_sequential_mnist_cuda
```

## Decision gate

An adaptive arm passes when it:

- gains at least `0.5` percentage points over its paired LIF control;
- improves at least two of three seeds;
- keeps event rate between `0.5x` and `2.0x` the paired LIF rate;
- avoids material LTW boundary saturation.

A pass carries adaptive neurons into the executable-delay experiment. A
failure moves directly to delay buckets with ordinary LIF neurons. This phase
does not reopen structural plasticity: its purpose is to isolate neuron-state
capacity after the Phase 27 selector failure.
