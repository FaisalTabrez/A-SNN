# Phase 20: temporal-state preservation on MNIST

Phase 19 showed that explicit latency bins are useful but count-plus-final-state
pooling destroys their linear separability. Phase 20 keeps the frozen reservoir
unchanged and records each neuron's pre-reset state at every timestep.

Pre-reset state is the membrane value after input and recurrent current are
added but before threshold reset. It therefore retains both subthreshold
voltage and threshold-crossing magnitude in one value per neuron and timestep.

## Representations

| Feature | Default dimension |
|---|---:|
| Raw intensity | 64 |
| Flattened latency events | 512 |
| Phase 19 full summary | 256 |
| Sensor temporal state | 512 |
| Hidden temporal state | 512 |
| Full temporal state | 1,024 |
| Raw plus hidden temporal state | 576 |

All representations receive a linear head and an MLP width-adjusted to the
same approximate 34k trainable-parameter budget. The topology, LTWs, event
code, train subset, and engineering-validation subset remain frozen.

## Colab command

```python
%cd /content/A-SNN-phase18
!git pull

!python gen5/examples/sprint20_temporal_state_mnist.py \
  --device cuda \
  --seeds 42 43 44 \
  --train-samples 20000 \
  --test-samples 5000 \
  --epochs 15 \
  --batch-size 512 \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/event_mnist_temporal_cuda
```

Expected outputs:

- `event_mnist_temporal.json`
- `event_mnist_temporal_records.csv`
- `event_mnist_temporal_summary.csv`
- `event_mnist_temporal_summary.png`

## Decision rules

- Sensor temporal near/above flattened latency: final pooling was the sensor
  bottleneck and time-preserving readouts should become the default.
- Hidden temporal above hidden summary: recurrent information existed but was
  erased by final pooling.
- Full temporal above sensor temporal: recurrence contributes useful temporal
  features.
- Raw-plus-hidden temporal above raw for both heads: adopt a residual sensor
  pathway and tune/train recurrent dynamics next.
- Hidden/full temporal remain inferior: stop frozen-reservoir feature
  engineering and move Phase 21 to supervised/evolved sparse dynamics.

The reused 5,000-image subset is an engineering validation set. Do not inspect
the unused 5,000-image official-test complement until the intervention is fixed.

## First CUDA result

Temporal preservation succeeds decisively. Full temporal state reaches
`91.52%` with a linear head, compared with `79.40%` for the final summary and
`85.94%` for raw pixels. The gain is positive for every seed.

The parameter-matched raw MLP remains stronger (`95.14%` versus `92.43%`), and
hidden temporal state alone does not reliably beat sensor temporal state. The
next phase therefore trains LTWs within the same fixed 384-edge topology using
surrogate gradients. Structural mutation remains gated on that result.

Full analysis:
`gen5/outputs/event_mnist_temporal_cuda_2026-08-09/analysis.md`.
