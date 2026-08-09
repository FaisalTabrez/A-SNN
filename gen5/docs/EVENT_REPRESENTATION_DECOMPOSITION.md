# Phase 19: event representation decomposition

Phase 18 established a stable negative result: the frozen AMMC trace lost
accuracy against both raw-pixel controls. Phase 19 locates that loss before the
project changes topology, firing dynamics, or plasticity.

The dataset split, frozen sparse reservoir, and optimizer settings remain
identical to Phase 18. Only the representation presented to the classifier is
changed. Because Phase 18 results already informed this design, its 5,000-image
test subset is now an engineering validation subset rather than an unbiased
final test set.

## Representations

| Feature | Dimension | Question |
|---|---:|---|
| `raw_intensity` | 64 | Original downsampled information ceiling. |
| `flattened_latency` | 512 | Does one-spike temporal quantization preserve the digit? |
| `sensor_trace` | 128 | Does collapsing sensor events into count/final-membrane summaries lose timing? |
| `hidden_trace` | 128 | What can recurrent hidden dynamics represent alone? |
| `full_trace` | 256 | Phase 18 sensor-plus-hidden representation. |
| `raw_plus_hidden` | 192 | Does the hidden reservoir add complementary information to raw pixels? |

Each representation receives:

- a linear classifier;
- an MLP whose width is chosen to approximate the same 34k trainable-parameter
  budget used by the Phase 18 AMMC MLP.

The runner records accuracy, feature dimensionality, parameter count, hidden
spike rate, feature-generation time, and feature examples per second.

## Colab command

Run from the temporary clone created for Phase 18:

```python
%cd /content/A-SNN-phase18
!git pull

!python gen5/examples/sprint19_event_representation_decomposition.py \
  --device cuda \
  --seeds 42 43 44 \
  --train-samples 20000 \
  --test-samples 5000 \
  --epochs 15 \
  --batch-size 512 \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/event_mnist_decomposition_cuda
```

Expected outputs:

- `event_mnist_decomposition.json`
- `event_mnist_decomposition_records.csv`
- `event_mnist_decomposition_summary.csv`
- `event_mnist_decomposition_summary.png`

## Decision tree

- Latency below raw: improve temporal coding resolution or use rate/population
  coding before changing the reservoir.
- Latency near raw, sensor trace below latency: retain explicit time bins or
  train a temporal readout instead of final-state pooling.
- Hidden below sensor/full: recurrent dynamics are destructive or too weak;
  sweep threshold, gain, sign balance, and recurrent weight scale.
- Raw-plus-hidden above raw: the reservoir contributes complementary features,
  so preserve a residual sensor path in the architecture.
- Raw-plus-hidden does not beat raw: the current frozen random reservoir adds no
  useful MNIST information; Phase 20 should train/evolve the substrate rather
  than merely scaling it.

With only three seeds, this remains an engineering diagnosis. Any winning
configuration must later be rerun with more seeds and confirmed on the unused
5,000-image complement of the official MNIST test split before a final
generalization claim.

## First CUDA result

Flattened latency improved linear accuracy from `85.94%` to `88.11%`, showing
that the event code retains useful class structure. The major loss occurs when
time is collapsed: sensor-summary linear accuracy fell to `72.74%`. Hidden
summary features did not reliably improve that result.

Raw-plus-hidden features reached `86.92%` with a linear head, a consistent
`+0.98` point gain over raw intensity, but their MLP remained below the raw MLP.
The next phase preserves per-timestep neuron state and keeps a residual raw
pathway.

Full analysis:
`gen5/outputs/event_mnist_decomposition_cuda_2026-08-09/analysis.md`.
