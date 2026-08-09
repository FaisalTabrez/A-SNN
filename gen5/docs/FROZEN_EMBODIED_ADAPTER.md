# Frozen embodied readout benchmark

Sprint 16 moves the successful frozen-readout experiment back into the batched
2D physics worlds. It asks whether the information decoded from synthetic AMMC
traces can drive useful physical behavior under world and sensor distribution
shift.

## What remains frozen

- the recurrent AMMC edge topology,
- LTW/STW values,
- all neuron dynamics,
- sprouting and pruning.

Only the small MLP motor readout is trained. Its supervision comes from an
explicit sensor-space oracle: approach the food vector while subtracting a
weighted toxin vector. Consequently, this benchmark measures representation and
transducer deployment. It does **not** measure autonomous policy discovery.

## Controlled comparison

| Policy | Training | Purpose |
|---|---|---|
| `fixed_motor_argmax` | None | Existing frozen motor-spike decoder baseline. |
| `base_adapter` | Clean nominal sensory traces | Tests direct readout transfer without domain randomization. |
| `augmented_adapter` | Amplitude- and noise-augmented traces | Tests whether the Sprint 15 robustness result transfers into physics. |

All three policies receive identical initial worlds for a given evaluation seed.
They also use the same finite AMMC trace-window reset schedule. Results are
grouped by world, policy, sensor-noise level, and held-out seed.

## Colab CUDA command

```python
!python gen5/examples/sprint16_frozen_embodied_adapter.py \
  --device cuda \
  --worlds simple moving_toxins gauntlet \
  --eval-seeds 43 44 45 46 47 \
  --population-size 10000 \
  --steps 480 \
  --food-count 128 \
  --toxin-count 128 \
  --sensor-noise-stds 0.0 0.05 0.15 \
  --train-samples 8192 \
  --train-window 4 \
  --epochs 200 \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/frozen_embodied_adapter_cuda
```

For a short pipeline check, use `--population-size 256 --steps 24
--train-samples 512 --epochs 5 --worlds simple --eval-seeds 43`.

## Outputs

- `frozen_embodied_adapter.json`: configuration, seed-level records, and summary,
- `frozen_embodied_adapter_records.csv`: one record per policy/world/noise/seed,
- `frozen_embodied_adapter_summary.csv`: mean and population-standard-deviation summaries,
- `frozen_embodied_adapter_summary.png`: mean-fitness comparison plot.

Primary metrics are mean population fitness, food hits, toxin hits, and the
fraction of agents with non-negative final fitness. Diagnostic metrics include
cue-conditioned action coverage, agreement with the sensor oracle, mean action
magnitude, and each adapter's final training loss. These distinguish a useful
policy from a controller that improves fitness only by moving more often. The
strongest evidence for adapter transfer would be:

1. the augmented adapter beats the fixed decoder in hard worlds,
2. its advantage survives `noise_std=0.15`, and
3. it improves food acquisition without merely increasing toxin collisions.

If both trained adapters fail similarly, the likely bottleneck is the mismatch
between the synthetic trace distribution and continuously embodied AMMC state.
If the base adapter fails but the augmented adapter succeeds, the earlier
robustness result has transferred cleanly into physics.
