# Phase 35 analysis: SHD capacity-delay interaction

Archive SHA-256:
`8452829B4AB6AF10DB51A3A49998C156914A5767756FC0ED117D799D4F1761FB`

## Result

The registered universal capacity-delay hypothesis is rejected.

At 256 hidden neurons, all three delayed variants improve over the paired
no-delay mean:

| Timing pattern | Accuracy | Gain | Seeds gaining >=1 point |
|---|---:|---:|---:|
| no delay | 52.135% | - | - |
| uniform delay 1 | 54.667% | +2.532 points | 1/3 |
| hash delays 0-2 | 54.711% | +2.577 points | 2/3 |
| distance delays 0-2 | 54.608% | +2.473 points | 2/3 |

The two heterogeneous arms pass the pre-registered single-width gate. However,
the uniform-delay control produces nearly the same mean gain, and most of the
gain is concentrated in seed 43. This is evidence for a width-specific timing
or slowing effect, not yet evidence for heterogeneous polychronization.

At 512 hidden neurons, the effect disappears:

| Timing pattern | Accuracy | Gain |
|---|---:|---:|
| no delay | 60.704% | - |
| uniform delay 1 | 60.807% | +0.103 points |
| hash delays 0-2 | 61.028% | +0.324 points |
| distance delays 0-2 | 61.131% | +0.427 points |

No 512-neuron delayed arm has a seed gaining one point, so none passes the
registered gate. The heterogeneous effect therefore does not replicate across
capacity levels.

## Mechanistic and systems interpretation

- Hidden event rates and LTW movement remain stable, ruling out firing collapse
  or weight saturation as explanations.
- The 512-neuron no-delay result (`60.704%`) reproduces the Phase 34 result
  (`60.615%`) to within `0.089` points.
- Delays are expensive in the present tensor implementation. At 512 neurons,
  distance delays reduce mean inference throughput from roughly `8,309` to
  `4,973` examples/second, about a 40% loss.
- The strongest absolute Phase 35 score is `61.131%`, but its paired advantage
  is only `0.427` points and is not practically robust.

## Goal sanity check

AMMC has reproducible sparse-capacity scaling and a stable SHD baseline, but it
does not yet have evidence that heterogeneous axonal delays are a general source
of accuracy. A roughly 61% SHD result remains far from competitive published
systems, so no state-of-the-art or Transformer-alternative claim is justified.

The next bottleneck is temporal representation at the output. The current MLP
receives only the global mean hidden spike count and final membrane, discarding
where within the 64-bin utterance most activity occurred.

## Decision

Retain the 512-neuron no-delay model as the reliable baseline. Stop delay tuning.
Phase 36 will test a parameter-matched multi-scale temporal-pyramid readout at
256 and 512 neurons. A fixed time-shuffle arm at 512 neurons will distinguish a
true temporal-order benefit from a generic increase in readout features.
