# Program sanity check after Gen-28

## Original question

Do dynamic topology, dual memory timescales, delay learning, and local reward
credit causally improve spiking-network performance on real event tasks, rather
than appearing useful because of parameter inflation, seed selection, or
architectural confounds?

## Evidence-based answer

Not with the mechanisms tested so far. Matched controls rejected or failed to
replicate structural topology, readout-level dual memory, learned delays, and
local reward credit. The strongest supported neural result is narrower:
sample-specific residual LIF state contributes complementary information on SHD
and SSC event audio. That mechanism is cooperative with direct temporal
features, not a standalone SNN advantage, and it did not replicate on N-MNIST.

The systems work also has a clear boundary. Compilation removed the eager
Python-loop confound and brought residual LIF to 91.006% of matched TCN
throughput. Generic COO and a custom Triton event-scatter kernel both failed to
beat compiled dense execution. No hardware-energy claim is authorized.

## Next action

Close this workstream with one deterministic causal evidence synthesis. It must
state supported, rejected, untested, and proxy-only claims without adding a new
training sweep. A future biological-learning workstream requires a new
mathematical mechanism and preregistered causal microtask before another
real-dataset phase.
