---
type: experiment
run_id: "gen17_sparse_spiking_credit_cuda_2026-08-10"
sprint_id: "gen17"
title: "Gen-17 Sparse Spiking Credit"
dataset: "[[Embodied Delayed Reward]]"
architecture: "[[Sparse Spiking Credit]]"
hypothesis: "[[Sparse Spiking Credit]]"
status: "stop"
provenance: "repository-artifact"
source_artifact: "gen5/outputs/gen17_sparse_spiking_credit_cuda_2026-08-10/gen17_sparse_spiking_credit.json"
metrics:
  status: "stop"
  identical_reset_gate: true
  oracle_positive_control: true
  analog_reference_gate: false
  spiking_gain_gate: false
  spiking_translation_gate: false
  manual_gradient_parity_gate: true
  spike_activity_gate: true
  reward_identity_gate: false
  analog_qualified_gain_seed_count: 1
  spiking_qualified_gain_seed_count: 1
  analog_minus_spiking_gain_per_1000_steps: 0.39555559141768354
  maximum_manual_gradient_error: 3.725290298461914e-09
  mean_evaluation_spike_density: 0.12078472222222221
  mean_training_spike_density: 0.06369039351851852
  spiking_margin_vs_static_per_1000_steps: -0.39111114210552655
  spiking_margin_vs_shuffled_per_1000_steps: -1.0522222932842042
  reward_identity_seed_count: 1
tags: [experiment, gen17, gen5]
---

# Gen-17 Sparse Spiking Credit

## Executive Summary
Imported from `gen5\outputs\gen17_sparse_spiking_credit_cuda_2026-08-10\gen17_sparse_spiking_credit.json`. Provenance: **repository-artifact**.

## Context & Graph Connections
- Parent sprint: [[Gen-17]]
- Benchmark: [[Embodied Delayed Reward]]
- Architecture: [[Sparse Spiking Credit]]
- Hypothesis: [[Sparse Spiking Credit]]

## Metrics Summary
- **status**: `stop`
- **identical_reset_gate**: `True`
- **oracle_positive_control**: `True`
- **analog_reference_gate**: `False`
- **spiking_gain_gate**: `False`
- **spiking_translation_gate**: `False`
- **manual_gradient_parity_gate**: `True`
- **spike_activity_gate**: `True`
- **reward_identity_gate**: `False`
- **analog_qualified_gain_seed_count**: `1`
- **spiking_qualified_gain_seed_count**: `1`
- **analog_minus_spiking_gain_per_1000_steps**: `0.39555559141768354`
- **maximum_manual_gradient_error**: `3.725290298461914e-09`
- **mean_evaluation_spike_density**: `0.12078472222222221`
- **mean_training_spike_density**: `0.06369039351851852`
- **spiking_margin_vs_static_per_1000_steps**: `-0.39111114210552655`
- **spiking_margin_vs_shuffled_per_1000_steps**: `-1.0522222932842042`
- **reward_identity_seed_count**: `1`
