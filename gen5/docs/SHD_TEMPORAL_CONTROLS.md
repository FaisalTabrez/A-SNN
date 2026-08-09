# Phase 37: SHD temporal-control decomposition

## Question

Phase 36 raised 512-neuron SHD accuracy from 60.8% to 80.1% by replacing global
pooling with a parameter-matched temporal pyramid. Phase 37 asks whether the
recurrent AMMC reservoir materially contributes to that result, or whether a
time-aware decoder can obtain similar accuracy directly from raw events.

## Arms

1. Event-count MLP: discards event order.
2. Raw temporal pyramid: operates directly on the 700 input channels with the
   same 1/2/4/8 temporal hierarchy and approximately the same readout budget.
3. Sparse 512 global: reproduces the pre-pyramid AMMC baseline.
4. Sparse 512 feedforward pyramid: preserves sensor-to-hidden projections but
   physically disables all hidden-to-hidden edges.
5. Sparse 512 recurrent pyramid: reproduces the Phase 36 winner.

Every sparse comparison uses the same seed, sensor projections, initial LTWs,
readout initialization, dataset order, and optimizer schedule. The feedforward
arm differs only by recurrent edge deactivation.

## Registered gates

Recurrence gate:

- recurrent pyramid gains at least 3 mean points over feedforward pyramid;
- at least two seeds gain 2 points;
- activity and LTW saturation remain stable.

Reservoir gate:

- recurrent pyramid gains at least 2 mean points over raw temporal;
- at least two seeds gain 1 point.

If the raw temporal control matches or exceeds recurrent AMMC, Phase 36 is
primarily a decoder result. In that case, scaling the existing reservoir is not
scientifically justified; its recurrent learning mechanism must be redesigned.

## Colab command

```python
!python gen5/examples/sprint37_shd_temporal_controls.py \
  --device cuda \
  --seeds 42 43 44 \
  --hidden-neurons 512 \
  --timesteps 64 \
  --temporal-levels 1 2 4 8 \
  --projection-dim 32 \
  --epochs 15 \
  --warmup-epochs 5 \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data/shd \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/shd_temporal_controls_cuda
```

No new dataset upload is required. Download or zip the complete
`shd_temporal_controls_cuda` output directory when the run finishes.
