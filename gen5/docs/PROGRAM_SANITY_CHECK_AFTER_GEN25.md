# Program sanity check after Gen-25

Gen-25 does not weaken the compiled residual-LIF result. It identifies two
systems constraints: generic sparse primitives have prohibitive fixed overhead,
and hard spiking thresholds can amplify tiny operator-order differences.

The project should not claim event-driven efficiency from sparsity proxies.
Likewise, it should not start a Triton kernel until its numerical semantics are
specified. Gen-26 therefore freezes a short fidelity diagnostic comparing
FP32 count COO, FP64 count accumulation, and binary event COO against their
matched dense references. This preserves the causal and matched-control goals
while preventing an implementation artifact from becoming an architectural
claim.
