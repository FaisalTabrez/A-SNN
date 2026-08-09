# Phase 34 analysis: SHD hidden-capacity scaling

## Result

No-delay SHD accuracy scales strongly through 512 hidden neurons, but parameter
efficiency declines with every increase in width.

| Hidden neurons | Accuracy | Gain vs 128 | Effective parameters | Accuracy / 1k params |
|---:|---:|---:|---:|---:|
| 128 | 42.624% | baseline | 36,688 | 0.01162 |
| 192 | 48.837% | +6.213 pt | 53,328 | 0.00916 |
| 256 | 51.929% | +9.305 pt | 69,968 | 0.00742 |
| 384 | 57.759% | +15.135 pt | 103,248 | 0.00559 |
| 512 | 60.615% | +17.992 pt | 136,528 | 0.00444 |

The event-count MLP reaches `51.914%` with 92,308 parameters. The 256-neuron
no-delay model essentially matches it with 24% fewer parameters; 384 and 512
neurons exceed it by `5.85` and `8.70` points respectively.

## Registered gates

- **256 versus 128 capacity confirmation: pass.** Mean gain is `+9.305`
  points; all seeds improve and two of three clear the `+8` point primary gate.
- **Scaling beyond 256: pass.** The 384-neuron model gains `+5.830` points and
  the 512-neuron model gains `+8.687`; all seeds improve and the 512 arm clears
  the `+2` point practical threshold for every seed.
- **Efficiency: declining.** Accuracy per thousand effective parameters falls
  monotonically. Width buys accuracy, not better parameter efficiency.

## Dynamics and delay interaction

- Mean hidden event rate falls smoothly from `35.07%` at 128 neurons to
  `14.00%` at 512. Wider state distributes activity more selectively.
- LTW movement remains approximately `0.022-0.027` and upper saturation falls
  below `0.2%` at the widest scales.
- Training time remains roughly 34 seconds across no-delay scales in this T4
  run; 512-neuron inference remains about 6,495 examples/s.
- The 256-neuron distance-delay comparator reaches `54.682%`, a surprising
  `+2.753` point gain over paired no delay. All seeds improve, but gains are
  highly uneven: `+1.634`, `+6.581`, and `+0.044` points.
- Delays add about 55% training time and reduce inference throughput from about
  8,168 to 5,425 examples/s at 256 neurons.

## Goal sanity check and decision

The project now has reproducible evidence that sparse recurrent SHD performance
is capacity-limited and can exceed count-based controls. However, the declining
parameter efficiency and 60.6% ceiling show that width alone will not close the
large gap to specialized SHD systems.

The width-dependent delay signal cannot be claimed yet. It conflicts with the
near-zero 128-neuron result and is dominated by one seed. Phase 35 will test a
capacity-by-delay interaction at 256 and 512 neurons using no delay, uniform
delay one, heterogeneous hash 0-2, and heterogeneous distance 0-2. This will
distinguish generic slowing, heterogeneous timing, and seed noise before any
new temporal architecture is introduced.
