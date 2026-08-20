---
type: experiment
run_id: "gen18_local_credit_replication_cuda_2026-08-10"
sprint_id: "gen18"
title: "Gen-18 Local Credit Replication"
dataset: "[[Embodied Delayed Reward]]"
architecture: "[[Local Score Credit]]"
hypothesis: "[[Local Score Credit]]"
status: "stop"
provenance: "repository-artifact"
source_artifact: "gen5/outputs/gen18_local_credit_replication_cuda_2026-08-10/gen18_local_credit_replication.json"
metrics:
  status: "stop"
  identical_reset_gate: true
  oracle_positive_control: true
  replicated_local_gain_gate: false
  replicated_static_margin_gate: false
  replicated_reward_identity_gate: false
  manual_gradient_parity_gate: true
  qualified_gain_seed_count: 5
  qualified_static_margin_seed_count: 5
  qualified_reward_identity_seed_count: 6
  local_gain_mean_per_1000_steps: 0.7956667011603713
  local_gain_ci95_lower_per_1000_steps: -0.01628414509997178
  local_margin_vs_static_mean_per_1000_steps: 0.7956667011603713
  local_margin_vs_static_ci95_lower_per_1000_steps: -0.01628414509997178
  local_margin_vs_shuffled_mean_per_1000_steps: 0.5100000280266006
  local_margin_vs_shuffled_ci95_lower_per_1000_steps: -0.013375779535129029
  maximum_manual_gradient_error: 3.725290298461914e-09
  next_milestone: "close_local_reward_credit_program"
tags: [experiment, gen18, gen5]
---

# Gen-18 Local Credit Replication

## Executive Summary
Imported from `gen5\outputs\gen18_local_credit_replication_cuda_2026-08-10\gen18_local_credit_replication.json`. Provenance: **repository-artifact**.

## Context & Graph Connections
- Parent sprint: [[Gen-18]]
- Benchmark: [[Embodied Delayed Reward]]
- Architecture: [[Local Score Credit]]
- Hypothesis: [[Local Score Credit]]

## Metrics Summary
- **status**: `stop`
- **identical_reset_gate**: `True`
- **oracle_positive_control**: `True`
- **replicated_local_gain_gate**: `False`
- **replicated_static_margin_gate**: `False`
- **replicated_reward_identity_gate**: `False`
- **manual_gradient_parity_gate**: `True`
- **qualified_gain_seed_count**: `5`
- **qualified_static_margin_seed_count**: `5`
- **qualified_reward_identity_seed_count**: `6`
- **local_gain_mean_per_1000_steps**: `0.7956667011603713`
- **local_gain_ci95_lower_per_1000_steps**: `-0.01628414509997178`
- **local_margin_vs_static_mean_per_1000_steps**: `0.7956667011603713`
- **local_margin_vs_static_ci95_lower_per_1000_steps**: `-0.01628414509997178`
- **local_margin_vs_shuffled_mean_per_1000_steps**: `0.5100000280266006`
- **local_margin_vs_shuffled_ci95_lower_per_1000_steps**: `-0.013375779535129029`
- **maximum_manual_gradient_error**: `3.725290298461914e-09`
- **next_milestone**: `close_local_reward_credit_program`
