# Gen-28 Triton event-kernel analysis

Date: 2026-08-20

The supplied manifest passed SHA-256 verification for all four declared
artifacts. Gen-28 stopped under its frozen decision rule.

On real SSC at batch 256, compiled dense execution reached 167,179 examples/s.
The sensor-native Triton path reached 54,769 examples/s, a 0.3276 throughput
ratio. Including event discovery reduced the ratio to 0.2900. The event-native
path therefore missed the preregistered parity gate by a wide margin.

There was no low-density rescue. At registered synthetic densities of 0.1%,
0.5%, and 1%, the best sensor-native ratio was 0.5208. This rules out the
predeclared density-gated hybrid continuation for the tested implementation.

Current differences versus COO were small (at most 2.98e-6), but prediction
agreement at real batch 256 fell to 99.609%, below the frozen 99.9% behavioral
gate. Smaller real batches and all synthetic workloads retained exact
prediction identity. The numerical contract therefore failed even though the
error magnitude was small.

The correct decision is to close the current software event-sparse path.
Generic COO and the custom atomic-scatter Triton kernel are both slower than
compiled dense execution, and neither hardware energy nor neuromorphic
efficiency is established. Gen-28 does not weaken the replicated residual-state
mechanism; it limits the production claim to compiled dense accelerator
execution.
