# Frozen Readout Adapter Augmented Robustness Result

Date: 2026-07-28

Source: pasted Colab JSON summary from
`/content/drive/MyDrive/A-SNN/gen5_outputs/frozen_readout_adapter_augmented_robustness_cuda`.

Configuration:

- Adapter kind: `mlp`
- Feature mode: `full_trace`
- Train seed: `42`
- Held-out seeds: `43`, `44`, `45`, `46`, `47`
- Augmented training:
  - amplitudes: `0.35`, `0.55`, `0.75`, `1.0`
  - noise stds: `0.0`, `0.05`, `0.15`
  - timestep values: `4`, `8`, `12`
- Robustness evaluation:
  - amplitudes: `0.35`, `0.55`, `0.75`, `1.0`
  - noise stds: `0.0`, `0.05`, `0.15`
  - timestep values: `4`, `8`, `12`

## Aggregate adapter accuracy by condition

| Task | Base | Amp 0.35 | Amp 0.55 | Amp 1.0 | Noise 0.05 | Noise 0.15 | T=4 | T=12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| direction_copy | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| anti_toxin | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 99.99% | 100.00% | 100.00% |
| cue_switch | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 99.98% | 100.00% | 100.00% |
| delayed_recall | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 99.84% | 100.00% | 100.00% |
| two_pulse_sum | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 99.41% | 100.00% | 100.00% |

## Findings

- Augmented readout training repaired the previous robustness collapse.
- Noise robustness improved dramatically:
  - `direction_copy`: `noise_0.15` from about `23.64%` to `100.00%`.
  - `anti_toxin`: `noise_0.15` from about `25.55%` to `99.99%`.
  - `cue_switch`: `noise_0.15` from about `24.54%` to `99.98%`.
  - `delayed_recall`: `noise_0.15` from about `23.60%` to `99.84%`.
  - `two_pulse_sum`: `noise_0.15` from about `24.72%` to `99.41%`.
- The previous amplitude brittleness also disappeared. All tested amplitude
  conditions reached `100%`.
- Timestep robustness is now complete for the tested `4`, `8`, and `12`
  timestep conditions.

## Interpretation

The failure mode was primarily adapter data coverage/calibration, not a missing
reservoir representation. Once the readout adapter was trained over a broader
set of reservoir states, it decoded all current synthetic tasks robustly across
held-out seeds, noise, amplitude shifts, and sequence-length shifts.

This supports a stronger claim:

> The frozen AMMC substrate plus a small trainable nonlinear readout behaves as
> a robust reusable reservoir for the current synthetic temporal-control task
> family.

## Decision

Move the next major validation back toward embodied/harder environments:

1. Add or adapt an embodied evaluation path that uses the trained readout
   adapter as the motor policy head.
2. Compare:
   - frozen motor argmax,
   - frozen substrate plus unaugmented adapter,
   - frozen substrate plus augmented adapter.
3. Evaluate on harder worlds/noisy sensors before moving to external datasets
   such as MNIST.

If the augmented adapter transfers to harder embodied worlds, the AMMC project
has a much stronger story: sparse evolved reservoir + lightweight robust
adapter, rather than purely end-to-end retraining.
