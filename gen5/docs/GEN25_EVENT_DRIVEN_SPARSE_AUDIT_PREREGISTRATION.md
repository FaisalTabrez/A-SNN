# Gen-25 event-driven sparse operator audit preregistration

Date frozen: 2026-08-20

Gen-25 replaces only the Phase-49 residual-LIF temporal Conv1d input operator.
The dense control uses compiled Conv1d plus the compiled residual-LIF head. The
candidate discovers nonzero events, routes them through an exact COO sparse
matrix multiplication for each temporal kernel offset, and uses the identical
compiled head and frozen weights.

Dense-to-COO conversion is included, making this a conservative software test.
Three timing seeds use real SSC at batch sizes 1, 32, and 256. Registered
synthetic densities of 0.5%, 1%, 5%, and 10% are tested at batch 32 to locate a
possible crossover without redefining the real-data endpoint.

The sparse operator must preserve predicted classes exactly and keep maximum
logit deviation at or below `1e-4` for every workload. Promotion additionally
requires mean real-SSC throughput at batch 256 to equal or exceed the compiled
dense control. Low-density crossover is reported separately. Failure after
numerical equivalence authorizes a custom Triton/CUDA event-kernel phase; it
does not authorize threshold changes, accuracy claims, or energy claims.
