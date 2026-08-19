# Gen-24 compiled residual-state analysis

Date analyzed: 2026-08-20

The supplied manifest passed SHA-256 verification. Gen-24 passed every frozen
gate on an NVIDIA L4. Compiled residual LIF preserved 100% predicted-class
identity, with maximum logit deviation below `9e-8`, and compiled successfully
for all seeds and batch sizes.

At the primary batch size of 256, residual LIF improved from 26,962 to 243,381
examples per second, a 9.027x speedup. The matched compiled TCN reached 267,435
examples per second, leaving residual LIF at 91.006% throughput parity. At
batch sizes 1 and 32, residual-LIF compilation yielded 67.56x and 62.45x
speedups respectively. Compile latency was substantial (about 13.1 seconds at
batch 256), so these figures describe warmed steady-state inference.

This reverses the Phase-49 software interpretation: the previous residual-LIF
deficit was primarily eager Python-loop overhead. It does not change accuracy
evidence and does not establish hardware-energy efficiency. The next systems
question is whether sparse events can avoid dense input work while preserving
the now-supported compiled state computation.
