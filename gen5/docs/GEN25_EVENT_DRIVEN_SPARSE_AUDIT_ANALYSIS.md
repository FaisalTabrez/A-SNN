# Gen-25 event-driven sparse operator analysis

Date analyzed: 2026-08-20

The supplied artifact manifest passed SHA-256 verification. Gen-25 stopped.
The generic COO hybrid achieved only 6.473% of compiled-dense throughput on
real SSC at batch 256. It reached just 3.955% and 3.857% at registered 0.5% and
1% synthetic density, so no sparse crossover exists in the tested range. COO
construction and sparse multiplication imposed a roughly 4 ms floor and used
more peak memory than the dense control.

Binary synthetic inputs were numerically exact. Real SSC has count-valued bins;
the COO accumulation order introduced small floating-point differences. Hard
LIF thresholds amplified them: maximum logit deviation rose from `4.98e-5` at
batch 1 to `1.82e-2` at batch 256, although predicted classes remained exactly
identical. The frozen all-workload `1e-4` logit gate therefore failed.

This rejects generic PyTorch COO as the production operator. Before authoring a
custom kernel, Gen-26 must determine whether count-preserving FP64 accumulation
repairs fidelity or whether binary event semantics are both exact and
behaviorally interchangeable with the count-valued encoding.
