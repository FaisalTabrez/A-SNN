# Phase 26: Targeted Synaptogenesis on Sequential MNIST

Phase 25 showed that fixed-topology LTW learning improves the sequential
reservoir, with useful change concentrated in its 16 sensor-to-hidden edges.
Phase 26 tests whether adding input projection capacity produces a benefit
beyond the successful fixed-topology LTW intervention.

## Paired structural interventions

Every recurrent arm begins with the same 272-edge graph and identical readout
initialization. The original graph is protected: no existing edge is removed.
After ten readout-only warmup epochs, the structural arms sprout deterministic,
unique connections and all active LTWs train at `3e-4` for five epochs.

- `frozen_recurrent`: frozen graph and frozen LTWs;
- `fixed_warm_all`: Phase 25 warm-all baseline, without topology growth;
- `sensor_sprout_16`: add two new hidden targets for each of eight row sensors;
- `sensor_sprout_48`: add six new hidden targets for each row sensor;
- `recurrent_sprout_64`: add one new outgoing recurrent edge per hidden neuron;
- `raw`: raw-pixel ceilings.

For a given seed, the 16 sensor sprouts are a strict prefix of the 48-sprout
topology, so the growth-dose comparison does not use unrelated random graphs.

New sensor edges are excitatory. New recurrent edges use the existing 80/20
excitatory/inhibitory convention. All are born at LTW `0.1`, and reported
sprouted-edge movement is measured from that birth value rather than zero.

## Colab CUDA run

```python
%cd /content/A-SNN-phase18
!git pull

!python gen5/examples/sprint26_structural_sequential_mnist.py \
  --device cuda \
  --seeds 42 43 44 \
  --train-samples 20000 \
  --test-samples 5000 \
  --epochs 15 \
  --warmup-epochs 10 \
  --surrogate-slope 10 \
  --batch-size 512 \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/structural_sequential_mnist_cuda
```

## Decision rule

Targeted synaptogenesis passes only if a sprouting arm:

- beats its paired `fixed_warm_all` control by at least `0.5` percentage points
  on average;
- improves at least two of three seeds;
- keeps final/initial hidden event rate within `[0.5, 2.0]`;
- avoids material LTW boundary saturation.

If sensor growth passes while recurrent growth does not, the next phase may
introduce reward- or gradient-gated sensor sprouting and conservative pruning.
If no growth arm passes, topology capacity is not the immediate bottleneck and
pruning remains unjustified.
