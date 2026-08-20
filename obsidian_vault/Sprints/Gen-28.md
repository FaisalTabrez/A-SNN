---
type: sprint
sprint_id: "gen28"
title: "Triton Event Kernel"
status: "active-implementation-complete-results-pending"
tags: [sprint, gen5]
---

# GEN28 - Triton Event Kernel

## Graph Connections
[[Spiking Speech Commands]], [[Event-driven Sparse Execution]]

## Evidence
See linked experiment records and the repository's `gen5/docs/` preregistration or analysis for the source protocol.

## Current Research Position
**Status:** `active-implementation-complete-results-pending`

Gen-28 is the current bounded systems experiment. Its Triton event-scatter kernel, behavioral-equivalence gates, and benchmark runner are implemented; terminal results are pending.

**Next action:** Run the frozen Gen-28 package on an NVIDIA L4, verify its manifest, and import the terminal result without a rescue sweep.

**Frozen protocol:** gen5/docs/GEN28_TRITON_EVENT_KERNEL_PREREGISTRATION.md<br>
**Implementation:** gen5/examples/gen28_triton_event_kernel.py and gen5/ammc_gen5/gen28_triton_event_kernel.py
