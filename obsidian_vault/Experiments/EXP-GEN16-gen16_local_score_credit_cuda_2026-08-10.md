---
type: experiment
run_id: "gen16_local_score_credit_cuda_2026-08-10"
sprint_id: "gen16"
title: "Gen-16 Local Score Credit"
dataset: "[[Embodied Delayed Reward]]"
architecture: "[[Local Score Credit]]"
hypothesis: "[[Local Score Credit]]"
status: "pass"
provenance: "repository-artifact"
source_artifact: "gen5/outputs/gen16_local_score_credit_cuda_2026-08-10/gen16_local_score_credit.json"
metrics:
  status: "pass"
  identical_reset_gate: true
  oracle_positive_control: true
  autograd_learnability_gate: true
  local_gain_gate: true
  autograd_qualified_gain_seed_count: 2
  local_qualified_gain_seed_count: 2
  autograd_equivalence_gate: true
  manual_gradient_parity_gate: true
  reward_identity_gate: true
  local_autograd_final_gap_per_1000_steps: 0.0
  maximum_manual_gradient_error: 2.7939677238464355e-09
  local_margin_vs_static_per_1000_steps: 0.18333334527495837
  local_margin_vs_shuffled_per_1000_steps: 0.2488889012278782
  reward_identity_seed_count: 3
  next_milestone: "translate_validated_local_score_rule_to_sparse_spikes"
tags: [experiment, gen16, gen5]
---

# Gen-16 Local Score Credit

## Executive Summary
Imported from `gen5\outputs\gen16_local_score_credit_cuda_2026-08-10\gen16_local_score_credit.json`. Provenance: **repository-artifact**.

## Context & Graph Connections
- Parent sprint: [[Gen-16]]
- Benchmark: [[Embodied Delayed Reward]]
- Architecture: [[Local Score Credit]]
- Hypothesis: [[Local Score Credit]]

## Metrics Summary
- **status**: `pass`
- **identical_reset_gate**: `True`
- **oracle_positive_control**: `True`
- **autograd_learnability_gate**: `True`
- **local_gain_gate**: `True`
- **autograd_qualified_gain_seed_count**: `2`
- **local_qualified_gain_seed_count**: `2`
- **autograd_equivalence_gate**: `True`
- **manual_gradient_parity_gate**: `True`
- **reward_identity_gate**: `True`
- **local_autograd_final_gap_per_1000_steps**: `0.0`
- **maximum_manual_gradient_error**: `2.7939677238464355e-09`
- **local_margin_vs_static_per_1000_steps**: `0.18333334527495837`
- **local_margin_vs_shuffled_per_1000_steps**: `0.2488889012278782`
- **reward_identity_seed_count**: `3`
- **next_milestone**: `translate_validated_local_score_rule_to_sparse_spikes`
