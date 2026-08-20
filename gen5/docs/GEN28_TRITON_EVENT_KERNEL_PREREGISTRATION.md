# Gen-28 Triton event-kernel preregistration

Date frozen: 2026-08-20

Gen-28 implements a custom Triton kernel that scatters precomputed events into
the Phase-49 temporal currents with atomic accumulation. Compiled dense,
sensor-native Triton (event-list conversion excluded), and dense-cache Triton
(conversion included) share weights and the compiled residual-LIF head.

Real SSC is measured at batches 1, 32, and 256. Registered 0.1%, 0.5%, and 1%
synthetic workloads at batch 32 test a low-density crossover. The custom kernel
must remain within `1e-3` of the validated COO currents and preserve at least
99.9% class identity. Promotion requires sensor-native real-SSC batch-256
throughput at least equal to compiled dense. End-to-end cache conversion and
low-density crossover are separate endpoints. No energy claim is authorized.
