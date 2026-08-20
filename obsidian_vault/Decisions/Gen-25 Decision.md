---
type: decision
sprint_id: "gen25"
status: "stop"
evidence: "gen5/docs/GEN25_EVENT_DRIVEN_SPARSE_AUDIT_ANALYSIS.md"
tags: [decision, gen25, gen5]
---

# Gen-25 Decision - Event-Driven Sparse Audit

## Decision
Generic PyTorch COO was behaviorally stable but far slower than compiled dense execution.

## Key Evidence
15201 vs 234864 examples/s; ratio 0.06473

## Graph Connections
- Phase: [[Gen-25]]
- Source: gen5/docs/GEN25_EVENT_DRIVEN_SPARSE_AUDIT_ANALYSIS.md
