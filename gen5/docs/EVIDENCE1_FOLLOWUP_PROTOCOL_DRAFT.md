# State-component isolation follow-up — draft, not registered

Status: implemented control primitives, no experimental authorization. Evidence-1
remains active. This document does not supply a human decision lock or reopen
closed structural-plasticity/local-learning mechanisms.

## Entry gate

Before running: mentor reviews and locks the original `causal_audit_only`
decision; a new protocol is signed/hashed before training; a genuinely untouched
holdout and permitted use/license are selected. SHD/SSC historical official-test
exposure must be disclosed, not rebranded as an untouched holdout. Do not choose
the next task or margins using new holdout outcomes.

## Missing controls to isolate

- Global derangement versus within-class derangement, same frozen checkpoint,
  paired training seeds and three fixed shuffle seeds. Singleton classes cause
  an explicit protocol error; no cross-class fallback.
- Zero pooled spike trace while retaining final analog membrane.
- Zero final analog membrane while retaining pooled spike trace.
- Zero both, unchanged from the old ablation, reported as a joint intervention.
- Separately retrained direct-only, analog-only and full hybrid controls with
  capacity, preprocessing, training budget and validation selection matched.
  Fixed-checkpoint interventions and retrained models are separate estimands.
- Any topology/noise/plasticity experiment must match edge capacity, noise,
  pruning and sprouting independently; the old 8-vs-85.95-edge comparison is
  not evidence for a specific plasticity effect.

`evidence1_followup_controls.py` supplies class-preserving partners and separate
state-component interventions. It is deliberately not wired into the existing
confirmation runner. The representation is pooled spike blocks followed by a
final hidden-width membrane block; a different representation requires a new
control implementation and tests.

## Statistical decisions still requiring human preregistration

Dataset and split identity; exact model/capacity recipe for retrained controls;
minimum meaningful effect; familywise correction for the additional comparisons;
seed budget/power; exclusions; and timing/rate perturbation definitions are
unresolved. No numeric choices have been back-filled from the returned test
results. Report paired intervals, all arms, failed gates and missingness.

The dormant authorization helper is a workflow guard, not cryptographic proof
of a reviewer. A self-written JSON object cannot certify human approval.
