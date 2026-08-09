# Phase 21 fixed-topology LTW training: analysis and goal sanity check

Date: 2026-08-09

Bundle SHA-256: `4CD20E4320F9C18C4A134DE32BB69167CC6532CDAD0FEE09FAE50ECA02A5C7C6`

The bundle is complete: six groups, three seeds, 18 seed-level records, six
aggregate rows, and a readable plot. All comparisons below use the same
engineering-validation subset as Phases 18-20.

## Aggregate result

| Group | Accuracy | Active edges | Effective trainable parameters |
|---|---:|---:|---:|
| Raw linear | 85.94% | 0 | 650 |
| Raw MLP | 95.14% | 0 | 34,210 |
| Frozen temporal linear | 91.58% | 384 | 10,250 |
| Frozen temporal MLP | 92.11% | 384 | 34,165 |
| Trained-LTW temporal linear | 91.27% | 384 | 10,634 |
| Trained-LTW temporal MLP | 92.24% | 384 | 34,549 |

## Paired findings

- Trained-LTW linear minus frozen linear: `-0.31` points on average. All three
  seed deltas are negative (`-0.38`, `-0.50`, `-0.06`). The exploratory 95%
  paired interval is `[-0.88, +0.25]` points.
- Trained-LTW MLP minus frozen MLP: `+0.13` points on average, with deltas
  `+0.28`, `-0.16`, and `+0.26`. The exploratory interval is
  `[-0.49, +0.74]`, spanning meaningful harm and benefit.
- The raw MLP remains `2.90` points ahead of trained-LTW temporal MLP.

With only three seeds, the intervals are descriptive rather than confirmatory.
Neither trained group meets the pre-registered across-seed improvement gate.

## Learning and activity diagnostics

The failed gain is not explained by a disconnected gradient:

- Mean absolute LTW displacement is `0.108` for linear and `0.063` for MLP.
- Mean LTW rises by `14.2%` for linear and `7.1%` for MLP.
- Hidden event rate rises by `61.2%` for linear and `12.8%` for MLP.

The update is strong enough to alter the dynamics, but not controlled enough
to improve representation reliably. Linear training in particular appears to
over-activate the reservoir.

Cost also moves in the wrong direction:

- training time increases by approximately `74.3%` for linear and `71.3%` for
  MLP;
- linear end-to-end inference throughput falls by `21.0%` in this run;
- the sparse model still trails the raw MLP while requiring temporal rollout.

## Project-goal sanity check

| Project claim area | Phase 21 evidence | Status |
|---|---|---|
| Surrogate gradient reaches LTW | LTWs move materially in every trained group | Supported |
| Fixed sparse topology can be optimized beneficially | No reliable paired gain | Not demonstrated |
| Structural plasticity should be added next | Weight-only gate failed | No |
| Continuous learning / retention | Offline supervised MNIST only | Not tested |
| Sparse efficiency | Extra rollout cost without accuracy advantage | Not demonstrated |
| Best-SNN / dense-model replacement | Raw MLP remains stronger and faster | Unsupported |

The project remains viable as a temporal sparse-learning investigation, but
Phase 21 rejects the assumption that simply enabling LTW gradients improves the
substrate. This useful negative evidence narrows the next experiment.

## Decision

Phase 22 retains the fixed 384-edge topology and runs paired LTW optimization
diagnostics:

- readout warmup before LTW unfreezing;
- lower LTW learning rates (`1e-4`, `3e-4`);
- surrogate slopes `5` and `10`;
- all-edge, sensor-edge-only, and recurrent-edge-only updates;
- identical initial graph and readout within every paired comparison;
- event-rate drift, edge-type LTW displacement, and boundary saturation.

Structural mutation remains deferred. It becomes eligible only after a stable,
repeatable weight-learning intervention improves the frozen substrate.

## Validation boundary

The reused 5,000-image subset remains engineering validation. The reserved
official-test complement must stay untouched until the optimization choice and
hyperparameters are frozen.
