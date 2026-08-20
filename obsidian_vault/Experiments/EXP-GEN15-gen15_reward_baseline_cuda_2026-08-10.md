---
type: experiment
run_id: "gen15_reward_baseline_cuda_2026-08-10"
sprint_id: "gen15"
title: "Gen-15 Reward Baseline"
dataset: "[[Embodied Delayed Reward]]"
architecture: "[[Reward Learning Baseline]]"
hypothesis: "[[Reward Learning Baseline]]"
status: "pass"
provenance: "repository-artifact"
source_artifact: "gen5/outputs/gen15_reward_baseline_cuda_2026-08-10/gen15_reward_baseline.json"
metrics:
  status: "pass"
  identical_reset_gate: true
  oracle_positive_control: true
  reinforce_gain_gate: true
  reward_identity_gate: true
  reinforce_margin_vs_static_per_1000_steps: 0.9922222627533805
  reinforce_margin_vs_shuffled_per_1000_steps: 1.2666667252779005
  next_milestone: "derive_local_credit_from_validated_baseline"
tags: [experiment, gen15, gen5]
---

# Gen-15 Reward Baseline

## Executive Summary
Imported from `gen5\outputs\gen15_reward_baseline_cuda_2026-08-10\gen15_reward_baseline.json`. Provenance: **repository-artifact**.

## Context & Graph Connections
- Parent sprint: [[Gen-15]]
- Benchmark: [[Embodied Delayed Reward]]
- Architecture: [[Reward Learning Baseline]]
- Hypothesis: [[Reward Learning Baseline]]

## Metrics Summary
- **status**: `pass`
- **identical_reset_gate**: `True`
- **oracle_positive_control**: `True`
- **reinforce_gain_gate**: `True`
- **reward_identity_gate**: `True`
- **reinforce_margin_vs_static_per_1000_steps**: `0.9922222627533805`
- **reinforce_margin_vs_shuffled_per_1000_steps**: `1.2666667252779005`
- **next_milestone**: `derive_local_credit_from_validated_baseline`
