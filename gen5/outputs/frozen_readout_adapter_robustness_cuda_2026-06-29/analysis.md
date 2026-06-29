# Frozen Readout Adapter Robustness Result

Date: 2026-06-29

Source: pasted Colab JSON summary from
`/content/drive/MyDrive/A-SNN/gen5_outputs/frozen_readout_adapter_robustness_cuda`.

Configuration:

- Adapter kind: `mlp`
- Feature mode: `full_trace`
- Train seed: `42`
- Held-out seeds: `43`, `44`, `45`, `46`, `47`
- Base training distribution:
  - amplitude: `0.75`
  - timesteps: `8`
  - noise: `0.0`
- Robustness perturbations:
  - amplitudes: `0.35`, `0.55`, `0.75`, `1.0`
  - noise stds: `0.0`, `0.05`, `0.15`
  - timesteps: `4`, `8`, `12`

## Aggregate adapter accuracy by condition

| Task | Base | Amp 0.35 | Amp 0.55 | Amp 1.0 | Noise 0.05 | Noise 0.15 | T=4 | T=12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| direction_copy | 100.00% | 100.00% | 25.00% | 100.00% | 18.84% | 23.64% | 100.00% | 100.00% |
| anti_toxin | 100.00% | 100.00% | 100.00% | 100.00% | 27.99% | 25.55% | 100.00% | 100.00% |
| cue_switch | 100.00% | 100.00% | 25.00% | 87.50% | 20.85% | 24.54% | 100.00% | 100.00% |
| delayed_recall | 100.00% | 0.00% | 100.00% | 100.00% | 23.15% | 23.60% | 100.00% | 100.00% |
| two_pulse_sum | 100.00% | 24.91% | 100.00% | 100.00% | 24.19% | 24.72% | 81.45% | 100.00% |

## Findings

- Clean timestep shifts are mostly robust. `timesteps_12` is perfect across all
  tasks and `timesteps_4` is perfect except for `two_pulse_sum`, which remains
  meaningfully above chance at roughly `81%`.
- Additive sensory noise is the dominant failure mode. Even `noise_std=0.05`
  collapses most tasks to chance or below chance.
- Some amplitude shifts are unexpectedly brittle:
  - `direction_copy` and `cue_switch` collapse at amplitude `0.55`.
  - `delayed_recall` collapses at amplitude `0.35`.
  - `two_pulse_sum` collapses at amplitude `0.35`.
- These failures indicate the current MLP adapter is a high-performing decoder
  for the base manifold, but it is not yet a calibrated or robust decoder over
  perturbed reservoir states.

## Interpretation

The reusable-reservoir claim survives clean seed and timestep changes, but not
sensory noise or all amplitude shifts. The likely bottleneck is readout
calibration/data coverage rather than missing reservoir information, because
many of the underlying frozen/reflex baselines remain high under conditions
where the adapter collapses.

## Decision

Extend the robustness runner so the adapter can be trained on augmented
reservoir traces:

- multiple input amplitudes,
- multiple sensory-noise levels,
- multiple timestep lengths.

Then rerun the same robustness suite. If augmented training repairs the failure
modes, the next major phase can return to embodied/harder environments with an
adapter-equipped controller. If it does not, the project needs either explicit
normalization/calibration in the adapter or reservoir-level noise robustness.
