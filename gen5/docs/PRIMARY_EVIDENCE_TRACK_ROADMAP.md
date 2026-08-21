# Primary evidence track roadmap

Decision date: 2026-08-21

## Purpose

Consolidate the project's strongest supported result before opening another
mechanism workstream. The target is the sample-specific residual LIF state in
temporal audio, together with the compiled dense execution path and the
shuffle-control methodology used to establish causal contribution.

This track precedes active-dendrite and astrocyte-gating research. It does not
reopen structural plasticity, STW/LTW memory, local three-factor learning,
learned-delay, or event-sparse GPU claims.

## Primary question

Under identical data, preprocessing, parameter accounting, optimization,
validation selection, and seed budgets, does a residual spiking-state audio
model provide a reproducible benefit over its direct-only and shuffled-state
ablations, and does it remain competitive with the strongest matched dense
temporal baseline?

## Stage E1 - Canonical protocol and runner

Create one runner for SHD and SSC with:

- frozen official train/test data and an explicitly constructed validation
  split where the dataset lacks one;
- paired seeds, initializations, minibatch orders, preprocessing, epochs,
  early-stopping rules, and checkpoint selection;
- residual LIF hybrid, direct-only, zero-state, and shuffled-state causal arms;
- strong dense controls: matched TCN and GRU, plus an SSM control only if its
  implementation and training budget can be matched transparently;
- exact trainable and active parameter counts, peak accelerator memory,
  compiled steady-state throughput, and accuracy confidence intervals;
- checkpointed records, SHA-256 manifest, environment metadata, and a clean
  clone execution contract.

No architecture or hyperparameter may be selected using the final test set.

## Stage E2 - Paired confirmation

Run at least ten paired seeds per benchmark. Report means, standard deviations,
paired confidence intervals, and per-seed deltas rather than only the best
seed. The following questions are separate gates:

1. **Causal state gate:** the full model must exceed both direct-only and
   shuffled-state controls with a confidence interval excluding zero on SHD
   and SSC.
2. **Accuracy gate:** the full model must match or exceed the strongest matched
   dense baseline within the preregistered equivalence margin on both
   benchmarks.
3. **Systems gate:** compilation must preserve predictions and improve
   steady-state inference over the identical eager model. This authorizes no
   energy claim.
4. **Replication gate:** a clean clone must reproduce the aggregate decision
   from frozen commands and documented dependencies.

Failure of the causal gate closes the residual-state mechanism claim. Passing
the causal gate but failing the accuracy gate narrows the output to a
causal-audit/methodology contribution. Passing both authorizes a focused audio
paper and reusable library.

## Stage E3 - Research artifact

If the gates support publication:

- prepare a compact pip-installable temporal-audio package;
- ship the causal ablations as first-class evaluation commands;
- write a paper around the matched multi-seed evidence, not biological or
  hardware speculation;
- validate on a deployment-relevant audio dataset only after SHD/SSC freeze;
- seek independent reproduction before any product or novelty superlative.

## Deferred research track

After the primary evidence decision is locked, begin a separate preregistered
active-dendrite context-gating program. Its first experiment must distinguish
routing-based interference protection from the memory-storage mechanisms
already rejected. Astrocyte-generated context is a later ablation, not an
assumed capability.

## Claim boundary

Until this roadmap completes, the defensible statement remains narrow:
sample-specific residual LIF state showed causal value on internal SHD/SSC
experiments, and compiled dense execution removed the eager-loop confound.
Competitive accuracy, independent replication, product value, and measured
energy efficiency are not yet established.
