# Gen-26 sparse numerical-fidelity preregistration

Date frozen: 2026-08-20

Gen-26 is a correctness diagnostic, not a throughput or accuracy benchmark.
It freezes the Phase-49 residual-LIF weights and compares three sparse temporal
operators over real SSC batches 1, 32, and 256: FP32 count-valued COO, FP64
count-preserving accumulation cast back to FP32, and FP32 binary occupancy COO.

Each operator is compared with a dense reference receiving the identical input
semantics. Current and logit maximum/mean error, predicted-class agreement, and
the logit/current amplification ratio are recorded. Binary dense predictions
are also compared with count-valued dense predictions to test whether encoding
normalization changes behavior.

A count-preserving repair requires maximum current error <=`1e-5`, maximum
logit error <=`1e-4`, and exact predicted-class identity for FP64 COO. Binary
promotion requires the same exactness plus at least 99.9% binary-versus-count
dense prediction agreement. Passing selects the corresponding custom-kernel
semantics. Binary exactness without semantic stability requires a separately
trained binary-encoding accuracy experiment. No energy claim is authorized.
