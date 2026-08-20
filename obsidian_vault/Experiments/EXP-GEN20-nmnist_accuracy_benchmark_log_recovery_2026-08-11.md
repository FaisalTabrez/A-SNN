---
type: experiment
run_id: "nmnist_accuracy_benchmark_log_recovery_2026-08-11"
sprint_id: "gen20"
title: "N-MNIST Accuracy Benchmark"
dataset: "[[N-MNIST]]"
architecture: "[[Spatial-temporal CNN]]"
hypothesis: "[[N-MNIST Accuracy Frontier]]"
status: "pass"
provenance: "recovered-provenance"
source_artifact: "gen5/outputs/nmnist_accuracy_benchmark_log_recovery_2026-08-11/nmnist_accuracy_benchmark.json"
metrics:
  status: "pass"
  best_arm: "spatiotemporal_cnn"
  best_mean_test_accuracy: 0.9947666666666667
  practical_gate_99_0: true
  stretch_gate_99_4: true
  spiking_confirmed: false
  next_milestone: "return_to_gen20"
tags: [experiment, gen20, gen5]
---

# N-MNIST Accuracy Benchmark

## Executive Summary
Imported from `gen5\outputs\nmnist_accuracy_benchmark_log_recovery_2026-08-11\nmnist_accuracy_benchmark.json`. Provenance: **recovered-provenance**.

## Context & Graph Connections
- Parent sprint: [[Gen-20]]
- Benchmark: [[N-MNIST]]
- Architecture: [[Spatial-temporal CNN]]
- Hypothesis: [[N-MNIST Accuracy Frontier]]

## Metrics Summary
- **status**: `pass`
- **best_arm**: `spatiotemporal_cnn`
- **best_mean_test_accuracy**: `0.9947666666666667`
- **practical_gate_99_0**: `True`
- **stretch_gate_99_4**: `True`
- **spiking_confirmed**: `False`
- **next_milestone**: `return_to_gen20`
