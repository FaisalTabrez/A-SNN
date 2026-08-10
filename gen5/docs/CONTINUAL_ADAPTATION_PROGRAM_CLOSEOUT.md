# Continual-adaptation program closeout

Status (2026-08-10): Gen-9–13 branch closed.

## Sanity check against the project goal

The long-term goal remains a brain-inspired SNN that learns temporally,
locally, continuously, and efficiently. This branch established a reliable
damage-and-adaptation protocol, but it did not establish the proposed local
learning mechanism.

What is supported:

- residual LIF state carries causal sample-specific information on SHD and SSC;
- a conventional temporal backbone is robust to sensor dropout;
- a damaged frozen representation can be repaired by a trainable readout;
- source retention can be protected by explicit context gating.

What is not supported:

- the tested predictive LIF representation as a source-competent replacement;
- bounded state adapters as sample-specific continual learners;
- dense or rank-order associative prototypes as useful fast memory;
- the tested supervised three-factor output rule as a substitute for autograd;
- STW/LTW consolidation, replay, autonomous structural plasticity, or a
  hardware-efficiency claim based on these failed mechanisms.

The central positive result is therefore narrower than the original vision:
temporal spiking state can be causally useful inside a hybrid model, but the
current local learning and memory mechanisms do not yet turn that state into
competitive continual adaptation.

## Next major-program choices

1. **Reward-modulated embodiment (recommended).** Return to a small-action
   embodied task and test eligibility traces with delayed scalar reward. This
   is a genuinely different credit signal from the supervised class-error
   rule that failed Gen-13 and aligns best with the biological objective.
2. **External event-benchmark generalization.** Package and replicate the
   validated residual-state mechanism on an additional independent event
   dataset without adding new plasticity claims.
3. **Publication and reproducibility closeout.** Freeze architecture work and
   turn the 15-source ledger, causal controls, and negative results into a
   formal benchmark/report package.

No option starts automatically. The selected program requires its own
preregistration, baseline, causal controls, resource budget, and terminal stop
rule. The failed Gen-13 rule is not eligible for a rescue sweep.

Selection update (2026-08-10): the user authorized option 1. Gen-14 now starts
the separately preregistered reward-modulated embodied-eligibility screen; it
does not reopen or tune the stopped Gen-13 mechanism.
