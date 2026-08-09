# Phase 42 analysis: SHD initialization robustness

Archive SHA-256: `70AF5E4E662B195862063DFD68FC98893D975376BAF59C9230DAB6C5817A0394`

## Registered gates

| Gate | Required | Observed | Result |
|---|---:|---:|---|
| Sparse 512 vs paired raw | +2 mean points; 6/9 positive | -0.299; 3/9 positive | Fail |
| Sparse 1024 vs paired raw | +2 mean points; 6/9 positive | -0.977; 3/9 positive | Fail |
| Sparse 1024 vs paired 512 | +1 mean point; 6/9 positive | -0.677; 4/9 positive | Fail |

## Interpretation

Raw temporal decoding averages `78.357%`. Sparse 512 averages `78.058%`, and
sparse 1024 averages `77.380%`. Neither sparse width provides a reliable paired
advantage once final readout initialization and optimizer order are independent
of topology construction.

Within-topology readout variance exceeds between-topology variance. At 512
nodes the respective standard deviations are `1.863` and `1.025` points; at
1024 they are `1.268` and `0.854`. Individual high scores such as `81.449%`
are therefore compatible with favorable initialization selection.

The Phase 40 result was scientifically useful as a hypothesis generator but is
not a robust estimate of expected performance. Increasing width does not solve
the problem and 1024 nodes are slower and less accurate on average.

## Goal sanity check

The reproducible contribution on SHD is the temporal pyramid decoder. Evidence
does not presently support sparse expansion, spiking, LTW learning, recurrence,
structural plasticity, or neuromorphic efficiency. Phase 43 is a final audit of
whether validation-selected checkpoints stabilize sparse generalization. A
failure should close this branch rather than trigger more hyperparameter tuning.
