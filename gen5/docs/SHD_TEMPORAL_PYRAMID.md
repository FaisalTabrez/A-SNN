# Phase 36: parameter-matched SHD temporal pyramid

## Why this phase exists

Phase 35 rejected a robust capacity-by-delay interaction. The dependable
512-neuron no-delay model reaches about 60.7% SHD accuracy, but its decoder only
sees the global mean hidden spike count and final membrane. That operation
discards where activity occurred within the 64-bin utterance.

Phase 36 tests whether that temporal collapse is now the main bottleneck.

## Registered arms

- 256-neuron global MLP
- 256-neuron ordered temporal pyramid
- 512-neuron global MLP
- 512-neuron ordered temporal pyramid
- 512-neuron fixed-shuffled temporal pyramid

The pyramid averages hidden spikes over 1, 2, 4, and 8 contiguous windows. A
shared 32-dimensional projection is applied to each window, the projected
windows are concatenated with final membrane, and an MLP performs classification.
Its bottleneck width is calculated so total readout parameters remain within
10% of the paired global MLP.

The fixed-shuffled arm applies one deterministic permutation to the input time
axis and otherwise has the exact same architecture. It preserves event counts
but destroys the natural local ordering used during reservoir propagation.

## Gates

Primary representation gate:

- ordered 512 pyramid gains at least 3 mean accuracy points over global 512;
- at least two seeds gain 2 points;
- event rate remains within 0.5x to 2.0x;
- effective parameters remain within 10%.

Causal timing gate:

- ordered 512 pyramid gains at least 2 mean points over shuffled pyramid;
- at least two seeds gain 2 points.

Passing only the first gate means richer summary statistics help. Passing both
supports a representation that depends on natural temporal order.

## Colab command

Run from the repository root in a CUDA/T4 Colab runtime:

```python
!python gen5/examples/sprint36_shd_temporal_pyramid.py \
  --device cuda \
  --seeds 42 43 44 \
  --timesteps 64 \
  --temporal-levels 1 2 4 8 \
  --projection-dim 32 \
  --epochs 15 \
  --warmup-epochs 5 \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data/shd \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/shd_temporal_pyramid_cuda
```

No upload is required when `gen5_data/shd` already contains the cached official
SHD files from Phases 31-35. After completion, download or zip the entire
`shd_temporal_pyramid_cuda` output directory.
