# Frozen Readout Adapter Held-Out-Seed Generalization Result

Date: 2026-06-29

Source: pasted Colab JSON summary from
`/content/drive/MyDrive/A-SNN/gen5_outputs/frozen_readout_adapter_generalization_cuda`.

Configuration:

- Adapter kind: `mlp`
- Feature mode: `full_trace`
- Hidden units: `32`
- Train seed: `42`
- Held-out seeds: `43`, `44`, `45`, `46`, `47`
- Samples per held-out seed: `4096`
- Timesteps: `8`
- Neurons: `16`
- Max edges: `128`

## Summary

| Task | Train-seed split | Held-out seeds mean | Frozen baseline on held-out seeds |
|---|---:|---:|---:|
| direction_copy | 100.00% | 100.00% | 100.00% |
| anti_toxin | 100.00% | 100.00% | 25.00% |
| cue_switch | 100.00% | 100.00% | ~50.02% |
| delayed_recall | 100.00% | 100.00% | 100.00% |
| two_pulse_sum | 100.00% | 100.00% | 25.00% |

## Finding

The trained `mlp/full_trace` readout adapter generalized perfectly from
`train_seed=42` to held-out seeds `43` through `47` across all five synthetic
tasks.

This materially strengthens the previous result. The adapter is not merely
memorizing a train/test split from one seed. It is learning a stable decoder over
the frozen AMMC trace representation for this synthetic task family.

## Interpretation

The frozen AMMC substrate currently behaves as a reusable nonlinear reservoir:

- the reservoir preserves direct directional evidence,
- it carries avoidant toxin information even when the fixed motor argmax is
  inactive,
- it contains enough cue-context information for nonlinear decoding,
- it contains enough two-pulse temporal information for nonlinear modular-sum
  decoding.

The main bottleneck is no longer "does the frozen substrate contain information?"
for these toy tasks. The bottleneck is whether the reservoir-plus-adapter stays
robust under distribution shift: amplitude changes, sensory noise, variable
timesteps, distractors, channel permutations, and harder world dynamics.

## Decision

Add a robustness runner that trains the adapter on the base synthetic
distribution, then evaluates without retraining under amplitude, noise, and
timestep perturbations.

If robustness remains high, the next major phase should move from synthetic
tasks back to embodied/harder environments. If robustness collapses, task
diversification and augmentation become the next priority.
