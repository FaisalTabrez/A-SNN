# Frozen Readout Adapter Sweep CUDA Result

Date: 2026-06-29

Source: pasted Colab JSON summary from
`/content/drive/MyDrive/A-SNN/gen5_outputs/frozen_readout_adapter_sweep_cuda`.

Configuration:

- Samples: `4096`
- Train/test split: `2867 / 1229`
- Timesteps: `8`
- Neurons: `16`
- Max edges: `128`
- Variants:
  - `linear/full_trace`
  - `linear/motor_trace`
  - `mlp/full_trace`

## Summary

| Task | Linear full trace | Linear motor trace | MLP full trace | Frozen motor |
|---|---:|---:|---:|---:|
| direction_copy | 100.00% | 100.00% | 100.00% | 100.00% |
| anti_toxin | 100.00% | 100.00% | 100.00% | 25.00% |
| cue_switch | 73.39% | 73.39% | 100.00% | 50.42% |
| delayed_recall | 100.00% | 100.00% | 100.00% | 100.00% |
| two_pulse_sum | 41.50% | 34.09% | 100.00% | 25.00% |

## Findings

- `anti_toxin` remains a pure readout/transducer failure. Even the strict
  `linear/motor_trace` adapter reaches `100%`, meaning the motor-neuron trace
  already contains the avoidant direction signal, but the fixed hardcoded motor
  decision rule cannot express it.
- `cue_switch` is nonlinearly decodable. A linear adapter reaches `73.39%`, but
  the `mlp/full_trace` adapter reaches `100%`.
- `two_pulse_sum` is also nonlinearly decodable from the frozen trace. The
  previous chance-level frozen motor readout and weak linear probe were not
  evidence that the substrate lacked all sequence information; rather, the
  information is not linearly separable.
- The gap between `linear/full_trace` (`41.50%`) and `linear/motor_trace`
  (`34.09%`) on `two_pulse_sum` indicates some useful compositional signal lives
  outside the motor channels.

## Updated interpretation

The current AMMC frozen substrate is better described as a nonlinear reservoir
than as a direct controller. It can preserve and mix task-relevant temporal
signals, but a hardcoded argmax over motor spikes throws much of that structure
away.

## Decision

Before moving to harder claims or recurrent substrate training, validate adapter
generalization:

1. Train the readout adapter on one synthetic seed.
2. Evaluate it on held-out seeds with no additional training.
3. Compare split-test accuracy against held-out-seed accuracy.

If `mlp/full_trace` stays high on held-out seeds, the frozen AMMC substrate is a
strong reusable reservoir. If it collapses, the adapter has learned shortcuts in
the current synthetic distribution and needs harder task variation.
