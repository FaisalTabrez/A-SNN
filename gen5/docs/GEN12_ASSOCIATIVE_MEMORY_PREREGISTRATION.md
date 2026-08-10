# Gen-12 frozen-backbone associative-memory preregistration

Status (2026-08-10): implemented and frozen before observing Gen-12 results.

## Hypothesis

Gen-11 showed that a bounded state adapter learns a generic class correction:
removing state erased its small gain, but shuffling sample identity did not.
Gen-12 tests a distinct mechanism inspired by cortical–hippocampal functional
separation. A stable sensor-dropout TCN acts as the cortical encoder, while a
fast associative prototype memory stores labeled damaged-stream observations
without gradient updates to the backbone.

This is not synaptic STW/LTW. It is an explicit, context-gated fast-memory
test whose purpose is to establish whether sample-keyed associations can adapt
without source forgetting. Passing may open a separately preregistered test of
context discovery and consolidation; failing closes this memory branch.

## Frozen comparison

Seeds are 157–159. Source training, the 20% random sensor dropout, fixed 35%
damage mask at seed 909, SSC splits, cumulative budgets `0, 64, 256, 1024,
4096`, and conventional adaptation optimizer match Gen-11.

Five strategies are registered:

1. `dropout_tcn_static`;
2. `dropout_tcn_readout`;
3. `dropout_tcn_full_finetune`;
4. `dense_prototype_memory`;
5. `spiking_prototype_memory`.

Each memory holds one cumulative prototype per observed class. The dense arm
uses continuous frozen-backbone features. The spiking arm converts each
feature vector into a fixed rank-order code whose top 20% of units fire. Class
prototypes are updated by sums and counts only; they have no trainable
parameters. Query and prototype cosine similarity is converted to class
probability at temperature 0.10 and mixed 50:50 with frozen-backbone
probability. These values are frozen, not sweep variables.

Memory is active only in the known damaged context. Clean-source evaluation
disables it, making zero source forgetting an architectural property and an
explicit limitation—not evidence of autonomous context recognition.

## Causal controls

At every budget, especially 4,096:

- `memory removed` evaluates the frozen backbone alone;
- `association shuffled` cyclically remaps class prototypes without retraining;
- spike density, active memory cells, inference throughput, and storage time
  are recorded.

The shuffle control is the decisive Gen-11 correction: a useful associative
memory must depend on which prototype belongs to which class.

## Terminal gate

The spiking prototype memory passes only if all conditions hold:

- static damage drop is at least 2 points;
- mean adaptation gain is at least 2 points and repeats on at least 2/3 seeds;
- adaptation AUC is within 1 point of readout adaptation;
- final damaged accuracy is within 1 point of readout adaptation;
- source forgetting is no more than 0.5 point worse than readout adaptation;
- memory removal costs at least 0.5 point on average and on at least 2/3 seeds;
- shuffled associations cost at least 0.5 point on average and on at least 2/3 seeds;
- mean spike-code density remains between 5% and 35%.

A pass opens only a new context-free memory/consolidation preregistration. A
stop closes prototype memory without mixture, temperature, density, prototype,
damage, or budget sweeps. In either case, no best-SNN, continual-learning,
synaptic-memory, or hardware-efficiency claim follows automatically.
