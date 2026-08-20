---
type: experiment
run_id: "gen14_reward_eligibility_cuda_2026-08-10"
sprint_id: "gen14"
title: "Gen-14 Reward Eligibility"
dataset: "[[Embodied Delayed Reward]]"
architecture: "[[Reward-modulated Eligibility]]"
hypothesis: "[[Reward-modulated Eligibility]]"
status: "stop"
provenance: "repository-artifact"
source_artifact: "gen5/outputs/gen14_reward_eligibility_cuda_2026-08-10/gen14_reward_eligibility.json"
metrics:
  status: "stop"
  oracle_positive_control: true
  spiking_gain_gate: true
  spiking_specificity_gate: false
  spiking_activity_gate: true
  spiking_saturation_gate: true
  spiking_margin_vs_static_per_1000_steps: -0.7500000287675195
  spiking_margin_vs_shuffled_per_1000_steps: -0.16111113027566007
  next_milestone: "close_reward_eligibility_screen"
tags: [experiment, gen14, gen5]
---

# Gen-14 Reward Eligibility

## Executive Summary
Imported from `gen5\outputs\gen14_reward_eligibility_cuda_2026-08-10\gen14_reward_eligibility.json`. Provenance: **repository-artifact**.

## Context & Graph Connections
- Parent sprint: [[Gen-14]]
- Benchmark: [[Embodied Delayed Reward]]
- Architecture: [[Reward-modulated Eligibility]]
- Hypothesis: [[Reward-modulated Eligibility]]

## Metrics Summary
- **status**: `stop`
- **oracle_positive_control**: `True`
- **spiking_gain_gate**: `True`
- **spiking_specificity_gate**: `False`
- **spiking_activity_gate**: `True`
- **spiking_saturation_gate**: `True`
- **spiking_margin_vs_static_per_1000_steps**: `-0.7500000287675195`
- **spiking_margin_vs_shuffled_per_1000_steps**: `-0.16111113027566007`
- **next_milestone**: `close_reward_eligibility_screen`
