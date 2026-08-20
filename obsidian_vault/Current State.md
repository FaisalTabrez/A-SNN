---
type: current-state
current_sprint: "gen30"
status: "completed-stop"
updated: "2026-08-20"
tags: [current-state, gen5, gen30]
---

# Current State

## Current Research Position
[[Gen-30]] is the current program position: **Gen-30 Dendritic Predictive Credit**.

Gen-30 stopped: DPC showed large causal gains over its three ablations but failed absolute new-context accuracy, old-context retention, retention-drop, and seed-replication gates. The result blocks SSC transfer and separates weak local credit from interference protection.

## Frozen Scope
- Benchmark: [[Delayed Contextual Binding]]
- Hypothesis: [[Dendritic Predictive Credit]]
- Protocol: gen5/docs/GEN30_DENDRITIC_PREDICTIVE_CREDIT_PREREGISTRATION.md
- Implementation: gen5/examples/gen30_dendritic_predictive_credit.py and gen5/ammc_gen5/gen30_dendritic_predictive_credit.py

## Next Action
Design and preregister a new matched-capacity mechanism that separately improves hidden credit strength and protects prior context mappings.

## Guardrail
Do not tune Gen-30 after inspection, revive closed LTW/STW claims, enable structural plasticity, or transfer to SSC from this stop result.
