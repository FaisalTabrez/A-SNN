# Program sanity check after Gen-20

Date: 2026-08-20

## Goal

The long-term goal remains a continuously adapting, structurally sparse,
event-driven neural system whose useful behavior arises from causal local
mechanisms rather than parameter inflation or uncontrolled architecture
changes.

The operational research question is narrower and testable:

> Do dynamic topology, dual short/long-term memory, learned delays, and local
> reward credit independently improve task performance, adaptation, or
> retention after controlling parameters, active operations, seeds,
> optimization budget, and architectural confounds?

## What the evidence supports

- Residual LIF state is beneficially sample-specific on SHD and SSC when
  combined with direct temporal features.
- Conventional temporal representations support robust sensor-dropout and
  readout/full-finetune adaptation controls.
- The embodied task is solvable, and its stationary delayed-reward protocol can
  train a conventional policy.
- The derived analog local score-function update matches autograd exactly.
- Gen-20's proposed PLIF arms maintain healthy activity and a low analytical
  operation proxy.

## What remains rejected or unproven

- Standalone spiking inference is not yet competitive with the strongest
  matched conventional temporal baselines.
- Sample-specific residual-state benefit did not generalize to N-MNIST.
- The Gen-20 multiscale PLIF translation did not pass its accuracy screen, and
  distillation did not improve it.
- Reliable behavioral local credit did not replicate across held-out seeds.
- Dynamic topology, STW/LTW consolidation, replay, and learned-delay benefit
  have not passed controlled real-task tests.
- No direct hardware-energy advantage has been measured.

## Sanity decision

The project is still pursuing its original brain-inspired objective, but it is
not currently justified to combine every biological mechanism into a larger
model. That would obscure causality and recreate the confounds named in the
research question. Gen-19 and Gen-20 also close the current event-vision rescue
branch; more N-MNIST tuning would be exploratory architecture search.

## Next phase

The next experimental phase should be a **Matched Causal Mechanism Benchmark**
on the supported event-audio residual-state backbone. It should be one
factorial package rather than four serial phases:

1. static matched backbone;
2. topology-only adaptation;
3. dual-memory-only adaptation;
4. learned-delay-only adaptation;
5. local-credit-only adaptation;
6. a combined arm only if individual mechanisms pass their screen gates.

All arms must share data splits, seeds, parameter and active-operation budgets,
training updates, model-selection rules, and evaluation checkpoints. Required
outcomes are task accuracy, adaptation gain, retention after a task switch,
causal removal/shuffle controls, active edges, activity, latency, and memory.
Hardware energy remains a separate direct-measurement milestone.

This phase should be preregistered before implementation. It must not reuse the
failed Gen-20 architecture or reinterpret proxy sparsity as causal benefit.
