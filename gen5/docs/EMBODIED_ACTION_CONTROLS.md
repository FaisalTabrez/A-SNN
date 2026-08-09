# Embodied action controls

Sprint 17 tests the main confound exposed by the first frozen embodied adapter
run: trained adapters acted on every cue-bearing step, while the fixed spiking
decoder acted on only about 5.2%.

The experiment keeps the AMMC topology and all weights frozen and compares six
controllers on identical held-out seeds.

| Policy | Role |
|---|---|
| `fixed_motor_spiking` | Original sparse spiking decoder. |
| `fixed_analog_cardinal` | Normalized cardinal action from frozen AMMC motor evidence. |
| `random_cardinal` | Full-activity movement-opportunity control. |
| `direct_sensor_oracle` | Direct food-minus-toxin upper/control policy. |
| `base_adapter` | Clean-trained frozen-trace MLP. |
| `augmented_adapter` | Noise/amplitude-augmented frozen-trace MLP. |

## Colab command

```python
!python gen5/examples/sprint17_embodied_action_controls.py \
  --device cuda \
  --worlds simple moving_toxins gauntlet \
  --eval-seeds 43 44 45 46 47 \
  --population-size 10000 \
  --steps 480 \
  --sensor-noise-stds 0.0 0.05 0.15 \
  --train-samples 8192 \
  --train-window 4 \
  --epochs 200 \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/embodied_action_controls_cuda
```

The runner prints progress for all 270 policy/world/noise/seed evaluations.
Expected outputs are:

- `embodied_action_controls.json`
- `embodied_action_controls_records.csv`
- `embodied_action_controls_summary.csv`
- `embodied_action_controls_summary.png`

## Decision rules

- Adapter > random: the gain is not merely full-time movement.
- Adapter > fixed analog at comparable action magnitude: the trainable readout
  extracts more useful state than the fixed AMMC motor channels.
- Adapter approaches direct oracle: the frozen trace preserves most of the
  immediately actionable sensor policy.
- Fixed analog approaches adapter: the Sprint 16 result was mainly a threshold
  and motor-calibration problem.
- All full-activity policies perform similarly: the current world still rewards
  exploration more than representation quality and must be redesigned before
  MNIST or broader cognitive claims.

## First CUDA result

The complete run contains 270 evaluations. Both adapters beat the
full-activity random controller and the normalized fixed analog decoder in all
45 paired conditions:

- augmented adapter minus random mean fitness: `+2.071` (`45/45` wins);
- base adapter minus random: `+1.932` (`45/45` wins);
- augmented adapter minus fixed analog: `+1.998` (`45/45` wins);
- base adapter minus fixed analog: `+1.859` (`45/45` wins).

This resolves the Phase 16 movement-opportunity confound. The direct sensor
oracle remains the overall ceiling, and augmentation only slightly and
inconsistently improves on the clean adapter. The full analysis is retained in
`gen5/outputs/embodied_action_controls_cuda_2026-08-09/analysis.md`.
