# Phase 29: Executable Axonal Delays on Sequential MNIST

Phase 28 found that fixed adaptive thresholds reduced accuracy as adaptive
coverage increased. Phase 29 therefore retains ordinary LIF neurons and tests
the other pre-registered temporal mechanism: causal synaptic delay buckets.

`DynamicSparseLinear` already stores `delay_steps`, but earlier sequential
models did not execute them. This phase routes each edge from an explicit
history buffer. Delay `0` reproduces the current path; delay `1` or `2` reads
source state from one or two additional row steps in the past.

## Paired interventions

- `lif_no_delay_frozen` and `lif_no_delay_warm_all`: exact controls;
- `recurrent_delay1_frozen`: pure fixed-delay dynamics without LTW training;
- `recurrent_delay1_warm_all`: uniform extra one-step recurrent delay;
- `recurrent_hash012_warm_all`: deterministic heterogeneous recurrent delays;
- `recurrent_distance012_warm_all`: delays based on circular hidden-index
  separation as a reproducible spatial proxy;
- `raw`: flattened-pixel ceiling.

Only recurrent edges receive delays. Sensor rows remain contemporaneous. All
arms preserve the same 272 active edges, initial LTWs, LIF parameters, readout
dimensions, and optimizer budget.

## Colab CUDA run

```python
%cd /content/A-SNN
!git pull

!python gen5/examples/sprint29_delayed_sequential_mnist.py \
  --device cuda \
  --seeds 42 43 44 \
  --train-samples 20000 \
  --test-samples 5000 \
  --epochs 15 \
  --warmup-epochs 10 \
  --batch-size 512 \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/delayed_sequential_mnist_cuda
```

## Decision gate

A fixed-delay arm passes if it gains at least `0.5` percentage points over its
paired no-delay LIF control, improves at least two of three seeds, keeps event
rate within `[0.5x, 2.0x]`, and avoids material LTW saturation.

If a fixed pattern passes, Phase 30 may optimize delay assignment without
changing topology. If all fixed patterns fail, row-sequential MNIST has served
its diagnostic purpose and the framework should move to SHD rather than keep
tuning this task.
