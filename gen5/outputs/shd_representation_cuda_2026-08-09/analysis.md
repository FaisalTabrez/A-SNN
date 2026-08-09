# Phase 33 analysis: SHD representation diagnostic

## Result

The diagnostic identifies decoder capacity and hidden-neuron capacity as the
dominant bottlenecks. Fixed delays remain non-contributory on SHD.

| Arm | Test accuracy | Gain vs relevant control | Effective parameters |
|---|---:|---:|---:|
| Event-count MLP | 51.914% | reference | 92,308 |
| Sparse 128, linear, no delay | 36.278% | baseline | 6,352 |
| Sparse 128, MLP, no delay | 42.535% | +6.257 pt | 36,688 |
| Sparse 128, MLP, distance delays | 42.609% | +0.074 pt | 36,688 |
| Sparse 256, MLP, distance delays | 54.711% | +12.102 pt | 69,968 |
| Sparse 128, MLP, delays, threshold 1.5 | 44.405% | +1.796 pt | 36,688 |

## Registered gates

- **Nonlinear decoder: pass.** MLP decoding improves all three seeds by
  `4.15-9.32` points; all clear the `+3` point practical gate.
- **Delays under MLP: fail.** Mean gain is only `+0.074` points. Two seeds
  improve, one declines, and only one seed clears `+1` point.
- **Hidden capacity: pass decisively.** Moving from 128 to 256 hidden neurons
  improves every seed by `10.25-13.52` points, averaging `+12.10`.
- **Activity control: directional but below gate.** Threshold 1.5 lowers event
  rate from `35.46%` to `26.28%` and improves all seeds, but the mean gain
  `+1.80` misses the `+2` point gate and is dominated by the 256-neuron arm.

## Mechanistic diagnosis

- The 128-neuron graph discarded information that a linear readout could not
  recover. An MLP recovers 6.26 points without changing topology or LTWs.
- Wider hidden state is the larger effect. The 256-neuron arm reaches `54.71%`,
  beating the event-count MLP by `2.80` points despite using about 24% fewer
  effective parameters (`69,968` versus `92,308`).
- The wider graph naturally lowers mean event rate to `23.59%`, suggesting that
  distributing input over more neurons improves selectivity as well as capacity.
- LTW changes remain stable (`~0.022`) and upper saturation is below `0.4%` for
  the 256-neuron arm.
- Fixed delays do not explain the capacity result: at 128 neurons they add only
  `0.074` points while adding substantial runtime overhead.

## Goal sanity check and decision

This is the first SHD configuration to outperform the task's count-based MLP
control while remaining smaller in effective parameter count. It is meaningful
evidence that sparse recurrent spike state can encode useful temporal
information when given sufficient representational width and a competent
decoder. It is not a state-of-the-art SHD result and remains far below published
specialized systems.

Decision: retire delays from the next SHD optimization phase. Phase 34 will run
a no-delay hidden-capacity scaling study at 128, 192, 256, 384, and 512 neurons,
with a single 256-neuron delayed comparator. The objective is to confirm the
capacity effect, locate its efficiency knee, and test whether gains continue or
plateau before adding architectural mechanisms.
