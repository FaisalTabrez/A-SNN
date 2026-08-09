# Phase 37 analysis: SHD temporal-control decomposition

Archive SHA-256:
`0D075AD9404F7E0769454189DDCCD6F6022014FD025C7BB2DCC621001B017E7E`

## Result

| Arm | Accuracy | Parameters | Throughput |
|---|---:|---:|---:|
| Event-count MLP | 51.914% | 92,308 | 63,970 examples/s |
| Raw temporal pyramid | 77.959% | 132,944 | 39,242 examples/s |
| Sparse 512 global | 60.998% | 136,528 | 7,210 examples/s |
| Sparse 512 feedforward pyramid | 79.623% | 133,631 | 7,140 examples/s |
| Sparse 512 recurrent pyramid | 80.271% | 135,679 | 6,860 examples/s |

The registered reservoir gate passes narrowly. Recurrent AMMC exceeds the
parameter-matched raw temporal model by `+2.312` mean points. All three seeds
improve, and two seeds gain at least one point. The gains are uneven:
`+2.164`, `+0.265`, and `+4.505` points.

The registered recurrence gate fails decisively. Recurrent AMMC exceeds the
feedforward sparse model by only `+0.648` mean points. Two seeds improve, one
declines, and no seed gains two points (`-0.133`, `+1.193`, `+0.883`).

## Interpretation

The Phase 36 advance belongs primarily to the temporal decoder. A sparse
feedforward LIF expansion adds a smaller `+1.664` points over raw temporal
features, while the current random recurrent graph contributes less than one
point.

Recurrence raises the hidden event rate from `8.34%` to `13.16%`—about 58%
more spiking activity—for a `+0.648`-point accuracy change. Under the current
implementation, it also lowers throughput from `7,140` to `6,860`
examples/second. Random recurrence therefore has not earned its activity and
systems cost.

The raw temporal model is about 5.7 times faster than recurrent AMMC and uses
98.0% as many effective parameters. The sparse model's `+2.312`-point accuracy
gain is scientifically interesting but not yet an efficiency advantage.

## Goal sanity check

The project now has an internally reproducible 80.3% SHD system and modest
evidence that sparse LIF transformation improves over a matched raw temporal
decoder. It does not yet have evidence that random recurrence, structural
plasticity, axonal delays, or the current sparse tensor implementation offers a
compelling advantage over conventional temporal models.

Claims should therefore be limited to:

- multi-scale temporal decoding is a verified major improvement;
- sparse feedforward LIF features show a small positive contribution;
- the tested recurrent topology fails its causal gate.

State-of-the-art, neuromorphic-efficiency, and Transformer-alternative claims
remain unsupported.

## Decision

Do not tune AMMC recurrence in isolation yet. Phase 38 will place the result
against parameter-matched conventional temporal baselines: a standard dense
recurrent LIF trained with BPTT and a GRU, alongside raw temporal, sparse
feedforward, and sparse recurrent controls. This determines whether the 80.3%
system is competitive at its parameter budget before recurrence redesign.
