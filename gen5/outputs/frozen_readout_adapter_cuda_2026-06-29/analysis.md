# Frozen Readout Adapter CUDA Result

Date: 2026-06-29

Source: pasted Colab JSON summary from
`/content/drive/MyDrive/A-SNN/gen5_outputs/frozen_readout_adapter_cuda`.

Configuration:

- Adapter kind: `linear`
- Feature mode: `full_trace`
- Samples: `4096`
- Train/test split: `2867 / 1229`
- Timesteps: `8`
- Neurons: `16`
- Max edges: `128`
- Feature dimension: `32`

## Summary

| Task | Frozen motor | Adapter | Train | Best reflex | Adapter gain vs frozen | Adapter gain vs best reflex |
|---|---:|---:|---:|---:|---:|---:|
| direction_copy | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% |
| anti_toxin | 25.00% | 100.00% | 100.00% | 25.31% | 75.00% | 74.69% |
| cue_switch | 50.42% | 73.39% | 75.65% | 51.18% | 22.98% | 22.21% |
| delayed_recall | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% |
| two_pulse_sum | 25.00% | 41.50% | 46.46% | 24.90% | 16.50% | 16.60% |

## Findings

- `anti_toxin` is confirmed as a readout/transducer failure. A linear adapter
  over the frozen AMMC trace reaches perfect test accuracy while the fixed motor
  readout is inactive.
- `cue_switch` improves materially above frozen and reflex baselines, but the
  adapter underperforms the earlier diagnostic representation probe. This may
  be due to train/test split variation, optimization variance, or a genuine gap
  between probe ceiling and deployed adapter configuration.
- `two_pulse_sum` improves from chance to `41.50%`. This is still far from
  solved, but it weakens the previous interpretation that the substrate has no
  usable compositional signal at all. The frozen substrate contains partial
  sequence information that a readout can exploit.
- `direction_copy` and `delayed_recall` remain solved, but they are still
  explainable by direct/reflexive evidence integration.

## Decision

The next diagnostic should compare three adapter variants on the same tasks and
seed:

1. `linear/full_trace`
2. `linear/motor_trace`
3. `mlp/full_trace`

This will separate three possible bottlenecks:

- fixed motor pathway bottleneck,
- non-linear readout bottleneck,
- missing recurrent temporal/compositional substrate.

If `mlp/full_trace` substantially improves `cue_switch` or `two_pulse_sum`,
then the representation exists but is not linearly separable. If it does not,
the next target should be recurrent substrate training or explicit temporal
state, not more readout-only work.
