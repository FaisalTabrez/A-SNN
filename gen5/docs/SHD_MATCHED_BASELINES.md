# Phase 38: parameter-matched SHD baselines

## Purpose

Phase 37 found that the current random recurrent AMMC graph adds only 0.65
accuracy points over the feedforward sparse model. Phase 38 tests whether the
retained 80.3% system is competitive with conventional temporal architectures
at the same approximate trainable-parameter budget.

## Models

- Event-count MLP lower control.
- Parameter-matched raw-event temporal pyramid.
- Dense 128-neuron recurrent LIF trained end-to-end with surrogate-gradient
  backpropagation through time, Xavier input initialization, and a stable
  half-gain orthogonal recurrent initialization.
- A one-layer GRU whose largest valid hidden width is selected automatically
  under the AMMC parameter budget (58 units at the default configuration).
- Sparse 512-neuron feedforward AMMC temporal pyramid.
- Sparse 512-neuron recurrent AMMC temporal pyramid.

The target is the default recurrent AMMC effective budget of 135,679
parameters. All primary trainable comparators must remain within 10% of it.

## Registered interpretation

Sparse-advantage gate:

- recurrent AMMC beats dense LIF by at least 2 mean accuracy points;
- at least two seeds gain 1 point;
- activity and LTW saturation remain stable.

The GRU is a contextual conventional temporal reference rather than a target
for post-hoc tuning. Throughput is reported with accuracy because the project
ultimately claims sparse systems efficiency, not accuracy alone.

If a matched conventional baseline wins, the next phase must redesign AMMC
dynamics or learning. It must not compensate by increasing the readout or
performing another capacity sweep.

## Colab command

```python
!python gen5/examples/sprint38_shd_matched_baselines.py \
  --device cuda \
  --seeds 42 43 44 \
  --sparse-hidden-neurons 512 \
  --dense-lif-hidden-neurons 128 \
  --timesteps 64 \
  --temporal-levels 1 2 4 8 \
  --epochs 15 \
  --warmup-epochs 5 \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data/shd \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/shd_matched_baselines_cuda
```

The existing cached SHD files are reused. Download or zip the complete
`shd_matched_baselines_cuda` output directory after completion.
