# Phase 20 temporal-state MNIST: analysis and goal sanity check

Date: 2026-08-09

Bundle SHA-256: `DE2A3CEF725D4C3C1660DE171B376DA2B7546885DE1FD35037FD7704D217FB43`

The bundle is complete: seven feature families, two classifier families, three
seeds, 42 seed-level records, 14 aggregate rows, and a readable plot.

## Aggregate result

| Representation | Linear | Parameter-matched MLP |
|---|---:|---:|
| Raw intensity | 85.94% | 95.14% |
| Flattened latency | 88.11% | 91.40% |
| Full final summary | 79.40% | 85.99% |
| Sensor temporal state | 89.79% | 92.22% |
| Hidden temporal state | 89.53% | 92.41% |
| Full temporal state | 91.52% | 92.43% |
| Raw plus hidden temporal | 90.63% | 93.15% |

## Paired findings

- Full temporal minus full summary: `+12.12` linear points and `+6.44` MLP
  points, positive for every seed.
- Sensor temporal minus flattened latency: `+1.69` linear and `+0.82` MLP
  points, positive for every seed.
- Full temporal minus sensor temporal: `+1.73` linear points; only `+0.21` MLP
  points, with an exploratory interval crossing zero.
- Hidden temporal minus sensor temporal: no reliable advantage (`-0.27` linear,
  `+0.19` MLP).
- Raw-plus-hidden temporal minus raw: `+4.69` linear points but `-1.99` MLP
  points.
- Full temporal minus raw: `+5.58` linear points but `-2.71` MLP points.

The result strongly validates the temporal-pooling diagnosis. Pre-reset traces
recover information erased by spike-count/final-membrane summaries.

## Interpretation

1. Time-preserving state should become the default AMMC representation
   interface. Final summary pooling is no longer defensible for sequential
   benchmarks.
2. The frozen sparse dynamics create a useful nonlinear/time-expanded basis:
   a linear head reaches `91.52%`, well above raw linear and latency linear.
3. The recurrent hidden temporal state approximately matches sensor temporal
   state but does not independently outperform it. Combining sensor and hidden
   traces produces a clear linear gain.
4. The raw MLP remains the strongest model at `95.14%`. A frozen random AMMC
   reservoir has not demonstrated superiority to a conventional dense model.
5. The fixed parameter budget narrows MLP heads as feature dimension grows.
   Linear comparisons are the cleanest evidence for representation gain; MLP
   comparisons remain necessary as a practical ceiling.

## Project-goal sanity check

| Project claim area | Current evidence | Status |
|---|---|---|
| Sparse temporal representation | Frozen temporal trace improves linear separability by `5.58` points | Supported at engineering scale |
| Trainable sparse substrate | Sparse LTWs have not been trained on MNIST | Not tested |
| Structural plasticity | MNIST topology remains fixed at 384 edges | Not tested in this branch |
| Continuous learning / retention | MNIST is offline supervised classification | Not tested |
| Astrocyte modulation | Chemical layer is absent from MNIST | Not tested |
| Parameter/memory advantage | Temporal feature expansion and extraction add cost | Not demonstrated |
| Transformer alternative / best SNN | No sequence baseline or state-of-the-art comparison | Unsupported |

The project remains scientifically viable as a sparse adaptive temporal system,
but current results do not justify broad replacement claims. The immediate goal
should be to show that the sparse substrate itself can learn while retaining
its 384-edge topology and temporal advantage.

## Decision

Implement Phase 21 as a fixed-topology LTW-training ablation using surrogate
spike gradients:

- raw linear and raw MLP controls;
- frozen temporal linear and MLP controls;
- LTW-trained temporal linear and MLP models;
- identical seeds, data, topology, parameter budget, and validation subset;
- end-to-end training/inference time, LTW displacement, edge count, and hidden
  event rate.

Do not add structural mutation yet. First establish whether weight learning
improves the sparse substrate. Structural plasticity becomes Phase 22 only if
trained LTWs beat the frozen controls without collapsing temporal activity.

## Validation boundary

The same 5,000-image engineering-validation subset was reused. The unused
5,000-image official-test complement remains reserved until the Phase 21 model
choice and hyperparameters are frozen.
