---
type: "query"
date: "2026-10-14T08:58:55.446362+00:00"
question: "what about instead of LIF we remove the leaky part and see how things happen and has anyone worked in that direction"
contributor: "graphify"
outcome: "useful"
source_nodes: ["shd_state_placement_diagnostic.py", "State-component isolation follow-up — draft, not registered"]
---

# Q: what about instead of LIF we remove the leaky part and see how things happen and has anyone worked in that direction

## Answer

Expanded via graph vocabulary: lif leak leaky membrane reset state memory continual stability. Verified shd_state_placement_diagnostic.py lines 148-159: per-channel sigmoid leak, subtractive reset, membrane initialized to zero on each sequence. Exact IF requires an explicit leak coefficient of one, not initial_leak=1 in its logarithm initialization. Removing leak preserves subthreshold state only under zero current and no spike; reset, bias, distractors and subsequent weight updates remain. It does not itself protect learned functions across tasks. Literature verified: BS4NN arXiv:2007.04039 uses non-leaky IF with time-to-first-spike coding; Rueckauer et al. 2017 doi:10.3389/fnins.2017.00682 studies integrate/fire conversion and subtractive reset; Pellegrini et al. arXiv:2011.06846 compares IF versus LIF in speech and reports LIF superiority. Proposed only: matched learned-LIF, fixed-leak LIF, exact IF, and analog-integrator controls, separate delay retention from A/B weight-interference retention, inspect spike activity and voltage drift. No training, code modification, successor authorization or validated improvement. Evidence-1 remains active.

## Outcome

- Signal: useful

## Source Nodes

- shd_state_placement_diagnostic.py
- State-component isolation follow-up — draft, not registered