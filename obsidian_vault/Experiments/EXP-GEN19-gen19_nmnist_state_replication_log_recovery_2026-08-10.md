---
type: experiment
run_id: "gen19_nmnist_state_replication_log_recovery_2026-08-10"
sprint_id: "gen19"
title: "N-MNIST State Replication"
dataset: "[[N-MNIST]]"
architecture: "[[Residual LIF State]]"
hypothesis: "[[Residual LIF State]]"
status: "stop"
provenance: "recovered-provenance"
source_artifact: "gen5/outputs/gen19_nmnist_state_replication_log_recovery_2026-08-10/gen19_nmnist_state_replication.json"
metrics:
  runs: 3
  mean_conv_accuracy: 0.9686
  std_conv_accuracy: 0.0016990193249833098
  mean_full_accuracy: 0.9631666666666666
  std_full_accuracy: 0.0006944222218666303
  mean_direct_only_accuracy: 0.8110666666666666
  mean_state_only_accuracy: 0.7681
  mean_shuffled_state_accuracy: 0.9861666666666666
  mean_gain_vs_conv: -0.00543333333333329
  mean_state_contribution_vs_direct_only: 0.15210000000000004
  state_contribution_seed_count: 3
  mean_state_specificity_vs_shuffled: -0.022999999999999982
  state_specificity_seed_count: 0
  mean_direct_contribution_vs_state_only: 0.1950666666666667
  mean_spike_activity: 0.1705244640827179
  mean_conv_test_examples_per_second: 147508.59472768343
  mean_residual_test_examples_per_second: 41045.71488045243
tags: [experiment, gen19, gen5]
---

# N-MNIST State Replication

## Executive Summary
Imported from `gen5\outputs\gen19_nmnist_state_replication_log_recovery_2026-08-10\gen19_nmnist_state_replication.json`. Provenance: **recovered-provenance**.

## Context & Graph Connections
- Parent sprint: [[Gen-19]]
- Benchmark: [[N-MNIST]]
- Architecture: [[Residual LIF State]]
- Hypothesis: [[Residual LIF State]]

## Metrics Summary
- **runs**: `3`
- **mean_conv_accuracy**: `0.9686`
- **std_conv_accuracy**: `0.0016990193249833098`
- **mean_full_accuracy**: `0.9631666666666666`
- **std_full_accuracy**: `0.0006944222218666303`
- **mean_direct_only_accuracy**: `0.8110666666666666`
- **mean_state_only_accuracy**: `0.7681`
- **mean_shuffled_state_accuracy**: `0.9861666666666666`
- **mean_gain_vs_conv**: `-0.00543333333333329`
- **mean_state_contribution_vs_direct_only**: `0.15210000000000004`
- **state_contribution_seed_count**: `3`
- **mean_state_specificity_vs_shuffled**: `-0.022999999999999982`
- **state_specificity_seed_count**: `0`
- **mean_direct_contribution_vs_state_only**: `0.1950666666666667`
- **mean_spike_activity**: `0.1705244640827179`
- **mean_conv_test_examples_per_second**: `147508.59472768343`
- **mean_residual_test_examples_per_second**: `41045.71488045243`
