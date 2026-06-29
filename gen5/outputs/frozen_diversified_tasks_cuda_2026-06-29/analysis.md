# CUDA frozen diversified tasks, summary archive

Date: 2026-06-29

This archive records the first Sprint 15 frozen diversified task run. The
Colab command reported Drive-backed outputs, but only the final printed summary
was provided in this turn, so this is currently summary-level evidence.

Reported Drive outputs:

- `/content/drive/MyDrive/A-SNN/gen5_outputs/frozen_diversified_tasks_cuda/frozen_diversified_tasks.json`
- `/content/drive/MyDrive/A-SNN/gen5_outputs/frozen_diversified_tasks_cuda/frozen_diversified_tasks_summary.csv`
- `/content/drive/MyDrive/A-SNN/gen5_outputs/frozen_diversified_tasks_cuda/frozen_diversified_tasks_summary.png`

## Summary

| Task | Frozen AMMC | Random | Instant reflex | Integrated reflex | Inactive output | Margin |
|---|---:|---:|---:|---:|---:|---:|
| direction_copy | 100.00% | 25.29% | 100.00% | 100.00% | 0.00% | 2.000 |
| anti_toxin | 25.00% | 25.27% | 25.00% | 25.00% | 100.00% | 0.000 |
| cue_switch | 50.42% | 24.49% | 50.42% | 50.42% | 0.00% | 0.196 |
| delayed_recall | 100.00% | 26.05% | 25.00% | 100.00% | 0.00% | 0.319 |
| two_pulse_sum | 25.00% | 24.56% | 25.00% | 6.05% | 0.00% | -0.339 |

## Interpretation

- The frozen AMMC solves `direction_copy`, but so do the reflex baselines. This
  is expected from the seeded direct food-direction prior.
- The frozen AMMC also solves `delayed_recall`, but the integrated-reflex
  baseline solves it too. This means the result is temporal evidence
  integration, not yet proof of hidden-state memory.
- `anti_toxin` collapses to chance with `100%` inactive output. The current
  frozen seed prior treats toxin channels as inhibitory/suppressive; it does
  not convert toxin evidence into an active opposite-direction motor command.
- `cue_switch` lands at about `50%`, exactly matching reflex baselines. The
  frozen model is ignoring the context cue and following the direct direction
  half the time.
- `two_pulse_sum` is chance-level for the frozen AMMC and instant reflex, while
  integrated reflex performs even worse because summing two directional pulses
  is not the same as modular composition.

## Decision

The current frozen AMMC substrate is mostly a direct reflex / evidence
integration system. It does not yet show context-dependent routing or symbolic
sequence composition.

Recommended next implementation:

1. Add a frozen representation probe: collect final membrane/spike traces and
   train only a tiny linear readout. If the probe solves `cue_switch` or
   `two_pulse_sum`, the information exists but the motor readout is wrong. If
   not, the frozen substrate itself lacks the representation.
2. Add task-specific transducer baselines, especially an opponent-toxin
   avoidance readout, to separate transducer mismatch from neural failure.
3. Only after those probes should we add plasticity or trainable recurrent
   weights on these tasks.
