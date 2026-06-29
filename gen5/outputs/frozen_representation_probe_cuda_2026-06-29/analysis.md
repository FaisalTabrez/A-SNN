# CUDA frozen representation probe, summary archive

Date: 2026-06-29

This archive records the first Sprint 15 frozen representation probe run. The
Colab command reported Drive-backed outputs, but only the final printed summary
was provided in this turn, so this is currently summary-level evidence.

Reported Drive outputs:

- `/content/drive/MyDrive/A-SNN/gen5_outputs/frozen_representation_probe_cuda/frozen_representation_probe.json`
- `/content/drive/MyDrive/A-SNN/gen5_outputs/frozen_representation_probe_cuda/frozen_representation_probe_summary.csv`
- `/content/drive/MyDrive/A-SNN/gen5_outputs/frozen_representation_probe_cuda/frozen_representation_probe_summary.png`

## Summary

| Task | Frozen motor | Linear probe | Probe train | Best reflex | Gain over frozen | Gain over best reflex | Loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| direction_copy | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% | 0.001 |
| anti_toxin | 25.00% | 100.00% | 100.00% | 24.98% | 75.00% | 75.02% | 0.001 |
| cue_switch | 50.42% | 85.76% | 88.32% | 53.13% | 35.35% | 32.63% | 0.353 |
| delayed_recall | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% | 0.001 |
| two_pulse_sum | 25.00% | 31.81% | 33.00% | 26.69% | 6.81% | 5.13% | 1.384 |

## Interpretation

- `anti_toxin` is not a representation failure. The frozen substrate contains
  enough information for a linear readout to reach `100%`, while the fixed
  motor readout is inactive. This is a transducer/readout mapping problem.
- `cue_switch` is partially represented. The linear probe reaches `85.76%`,
  far above frozen motor readout and reflex baselines, but below perfect. This
  suggests the cue and direction information are present but not fully linearly
  disentangled.
- `direction_copy` and `delayed_recall` remain solved by the frozen substrate
  and simple baselines.
- `two_pulse_sum` remains close to chance even with the linear probe. The
  frozen substrate does not encode modular sequence composition in a linearly
  recoverable way.

## Decision

The next Sprint 15 step should not be full recurrent training yet. It should be
a minimal trainable readout/transducer adapter:

1. freeze the sparse recurrent AMMC substrate,
2. train a small motor readout head,
3. compare against the linear probe ceiling,
4. then decide whether recurrent substrate learning is needed.

Expected near-term effect:

- `anti_toxin` should become solvable with readout training alone.
- `cue_switch` should improve substantially but may not reach `100%` without a
  better context representation.
- `two_pulse_sum` likely requires recurrent/plastic substrate learning or a
  richer temporal representation.
