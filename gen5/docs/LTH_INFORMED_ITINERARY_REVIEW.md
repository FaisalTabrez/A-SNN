# LTH-informed itinerary review

Decision date: 2026-08-26

Source reviewed: `LTH_Informed_Refinements_AMMC-SNN.md`

## Decision

The proposal is accepted as research input, not as a replacement for the
frozen primary-evidence sequence. Evidence-1 remains the active program. The
LTH-inspired ideas enter the itinerary only after the SHD/SSC residual-state
decision is locked, and only in the gated form below.

## Review matrix

| Proposed experiment | Itinerary decision | Required refinement |
|---|---|---|
| Structured per-dendrite pruning | Conditional systems branch | A compartment mask is not itself a faster operator. Proceed only with a real block/channel-sparse execution plan, a matched unstructured control, end-to-end steady-state timing, and prediction-equivalence checks. |
| Reset-vs-continue evolution | Optional evolutionary diagnostic | LTH resets surviving weights to their original initialization. Resetting to a recent checkpoint is a different intervention. Compare original birth weights, reinitialized weights, continued mutation, and topology-only inheritance under paired seeds. |
| Supermask dendritic gating | First deferred mechanism study | This is the best fit with the active-dendrite research track. Separate explicit oracle context from inferred context, match total score/mask capacity, and compare against shared-mask, random-mask, shuffled-context, replay, and regularization controls. |
| Overparameterize then prune | Exploratory only | Match total search compute as well as final edge count. Do not infer that a supervised-optimization LTH result transfers automatically to evolutionary search. |

## Approved sequence

### E1-E3 - Primary audio evidence

Complete the canonical SHD/SSC paired-seed protocol, causal residual-state
ablations, matched dense baselines, compiled systems check, clean-clone
reproduction, and evidence package described in
`PRIMARY_EVIDENCE_TRACK_ROADMAP.md`. No LTH-inspired mechanism arm may alter
the frozen primary question.

The evidence package should preserve initialization provenance and mask/hash
metadata so later topology and initialization hypotheses can be tested without
retrospective ambiguity.

### R1 - Context-specific dendritic supermasks

After the Evidence-1 decision, preregister a routing experiment on the existing
two-context task. Start with an explicit context identifier to test the routing
mechanism itself. Freeze the backbone, train only bounded per-context scores,
and report both accuracy and total trainable/allocated mask capacity. A later
experiment may replace the explicit context with an inferred or astrocytic
signal; those are separate hypotheses.

The minimum causal controls are:

- shared mask across contexts;
- random matched-sparsity masks;
- shuffled context identities;
- no-context routing;
- matched replay and regularization baselines;
- frozen-backbone and jointly trained-backbone accounting.

Passing requires new-context learning, old-context retention, a positive paired
margin over the strongest matched control, and survival of the shuffled-context
test. Otherwise the context-routing claim closes at this task scale.

### S1 - Executable structured sparsity

Open this branch only if the primary audio architecture survives the evidence
gates and a supported block/channel-sparse operator can skip the pruned work.
Benchmarking a dense operator multiplied by a compartment mask is not evidence
of sparse acceleration. Any speed claim must include warm-up, compilation cost
separation, identical hardware, equal batch shapes, prediction fidelity, peak
memory, and dense plus unstructured baselines. No power or energy claim is
authorized without measurement.

### D1-D2 - Optional evolutionary diagnostics

The birth-initialization mutation test and overparameterize-then-prune study are
kept in the backlog. They are not publication-critical and do not precede R1.
If opened, both require paired seeds and compute-matched controls; D1 must
distinguish initialization values from topology, while D2 must match cumulative
search compute rather than only final model size.

## Interpretation boundary

Gen-19 tested transfer of a residual-state mechanism from audio to event vision;
it did not test transfer of an LTH winning-ticket mask together with its original
initialization. It is therefore relevant negative transfer evidence, but not a
direct falsification of winning-ticket transfer. External writing must preserve
that distinction.

The LTH, supermask, and edge-popup literature mappings in the source document
remain hypotheses until their citations and experimental equivalence are
checked in the later research track.
