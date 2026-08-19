# Program sanity check after Gen-23

The core research question remains appropriate: determine whether adaptive
spiking mechanisms causally improve real event-task performance after matching
parameter count, architecture, seeds, data, and update budget.

Gen-23 reaches a stopping rule, not a project failure. Four proposed adaptive
mechanisms currently lack qualifying evidence: structural topology, delay
learning, local reward credit, and dual-memory consolidation. A fifth rescue
rule would now be post-hoc mechanism shopping, so this branch is closed.

The strongest surviving AMMC-specific finding is recurrent residual LIF state:
Phase 48 showed a large causal state-removal effect on SSC. Phase 49 then found
that the implementation was approximately 3.2 times slower than its matched
TCN despite a lower dense-operation proxy. The next warranted experiment is
therefore a systems falsification: test whether compilation preserves exact
outputs and removes the interpreter overhead. It does not alter any accuracy
claim and cannot authorize a hardware-energy claim.
