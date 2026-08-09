# Phase 24: Streaming Row-Sequential MNIST

Phase 23 showed that recurrence adds no practically meaningful value when a
readout can inspect the complete temporal trace of a static image. Phase 24
creates an actual memory requirement: an 8x8 MNIST image is presented one row
per neural step, and sparse reservoir groups expose only their final hidden
state after the eighth row.

## Causal comparison

Each seed constructs one sparse topology and evaluates two matched forms:

- **feedforward final state:** 16 row-input-to-hidden edges; recurrent edges are
  disabled;
- **recurrent final state:** the same 16 input edges plus 256 hidden-to-hidden
  edges.

Both variants return 128 features: final hidden spikes and final hidden
membrane state. Cumulative spike counts are used only for the activity metric,
so they cannot act as an external memory bypass. Their linear and matched-MLP
readouts use the same model seed and minibatch order. The benchmark also reports:

- raw flattened pixels, an information-rich non-temporal ceiling;
- the final image row, which cannot recover earlier rows;
- row-integrated pixels, an order-insensitive control.

## Colab CUDA run

```python
%cd /content/A-SNN-phase18
!git pull

!python gen5/examples/sprint24_sequential_mnist.py \
  --device cuda \
  --seeds 42 43 44 \
  --train-samples 20000 \
  --test-samples 5000 \
  --epochs 15 \
  --batch-size 512 \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/sequential_mnist_cuda
```

## Decision rule

Recurrence passes only if recurrent final-state accuracy exceeds its paired
feedforward counterpart by at least `0.5` percentage points on average and
improves at least two of three seeds. Also compare it with the integrated-row
control: recurrence must preserve useful sequential information, not merely
reproduce an orderless intensity summary.

- If the gate passes, Phase 25 trains fixed-topology LTWs on this sequential
  task before considering structural plasticity.
- If the gate fails, redesign reservoir time constants and explicit delay
  buffers; do not resume plasticity work on an uninformative substrate.
- The reserved final-test complement remains untouched during this diagnostic.

## First CUDA result (2026-08-09)

Phase 24 passes the recurrence gate decisively. The recurrent final state beats
its paired feedforward state by `+11.673` percentage points with a linear
readout and `+17.240` points with an MLP, improving every seed. It also beats
the orderless integrated-row control by `+9.067` and `+7.220` points.

This establishes a useful causal role for recurrence on a genuine sequential
task, but not competitive MNIST performance: recurrent MLP accuracy is
`55.547%`, versus `94.207%` for the raw-pixel MLP. Phase 25 therefore trains
LTWs on the fixed recurrent topology before any structural mutation is allowed.

See the retained [Phase 24 analysis](../outputs/sequential_mnist_cuda_2026-08-09/analysis.md).
