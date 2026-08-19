# Gen-24 compiled residual-state preregistration

Date frozen: 2026-08-20

Gen-24 audits the software execution cost of the one replicated supported
mechanism: residual LIF temporal state on SSC. It compares the exact
parameter-matched Phase-49 residual-LIF and dilated-TCN architectures under
PyTorch eager and `torch.compile(mode="reduce-overhead")` execution.

Weights are deterministically seeded and frozen. No training or accuracy claim
is made because compilation changes execution, not model capability. Real SSC
tensors, three timing seeds, and batch sizes 1, 32, and 256 are used. Each
compiled shape receives an isolated compilation, warm-up, and three repeated
measurements. Compile latency and peak CUDA allocation are reported separately.

At the primary batch size of 256, compiled residual LIF must:

1. compile successfully for all three timing seeds;
2. have maximum eager/compiled logit difference no greater than `1e-4`;
3. preserve 100% predicted-class identity; and
4. achieve at least 1.5 times the eager residual-LIF throughput.

Passing supports compiled execution of the residual-state computation. A
separate descriptive gate asks whether compiled residual LIF reaches 90% of
compiled matched-TCN throughput. Neither gate changes the Phase-48/49 accuracy
result or authorizes hardware-energy claims. Failure moves to compiler graph
break profiling rather than changing the thresholds or architecture.
