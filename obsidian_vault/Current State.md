---
type: current-state
current_sprint: "gen28"
status: "active-implementation-complete-results-pending"
updated: "2026-08-20"
tags: [current-state, gen5, gen28]
---

# Current State

## Current Research Position
[[Gen-28]] is the active research package: **Gen-28 Triton Event-Native Kernel Audit**.

Gen-28 is the current bounded systems experiment. Its Triton event-scatter kernel, behavioral-equivalence gates, and benchmark runner are implemented; terminal results are pending.

## Frozen Scope
- Benchmark: [[Spiking Speech Commands]]
- Hypothesis: [[Event-driven Sparse Execution]]
- Protocol: gen5/docs/GEN28_TRITON_EVENT_KERNEL_PREREGISTRATION.md
- Implementation: gen5/examples/gen28_triton_event_kernel.py and gen5/ammc_gen5/gen28_triton_event_kernel.py

## Next Action
Run the frozen Gen-28 package on an NVIDIA L4, verify its manifest, and import the terminal result without a rescue sweep.

## Guardrail
Do not claim hardware energy efficiency or create another sparse-kernel phase until Gen-28 reaches its frozen terminal decision.
