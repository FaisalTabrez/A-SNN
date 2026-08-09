# Phase 28 analysis: adaptive neurons on sequential MNIST

## Result

No adaptive-threshold arm passed the paired LIF gate. Accuracy generally fell
as the adaptive fraction increased.

| Arm | Linear accuracy | MLP accuracy | Linear vs LIF | MLP vs LIF |
|---|---:|---:|---:|---:|
| LIF frozen | 43.853% | 55.567% | control | control |
| ALIF 50% frozen | 43.533% | 55.400% | -0.320 pt | -0.167 pt |
| LIF warm all | 45.967% | 56.373% | control | control |
| ALIF 25% warm all | 45.360% | 56.067% | -0.607 pt | -0.307 pt |
| ALIF 50% warm all | 44.587% | 55.980% | -1.380 pt | -0.393 pt |
| ALIF 100% warm all | 43.467% | 55.280% | -2.500 pt | -1.093 pt |

No warm adaptive arm achieved a practical `+0.5`-point paired gain on any
seed. The 100% adaptive arm lost on every seed for both readouts.

## Mechanistic diagnosis

- Mean effective adaptive thresholds reached only about `1.04-1.06`, but event
  rates still fell monotonically with adaptive coverage.
- At 25%, 50%, and 100% adaptive coverage, linear event rates were about
  `0.943x`, `0.852x`, and `0.729x` paired LIF. MLP ratios were `0.953x`,
  `0.874x`, and `0.770x`.
- LTW changes remained well behaved and boundary saturation stayed below
  `0.7%`. This is not optimizer divergence.
- The frozen 50% arm also lost slightly, showing that the harm begins in the
  dynamics rather than only in LTW co-adaptation.

The fixed threshold-adaptation rule suppresses useful events in this short
eight-step row task and does not create more decodable memory. This rejects the
tested ALIF settings; it does not establish that adaptation is universally
unhelpful on longer speech or continual-learning sequences.

## Goal sanity check and decision

The core project goal remains a sparse temporal learner whose mechanisms must
earn their complexity under controls. Phase 28 is another useful falsification:
the current sequential task benefits from recurrence and LTW optimization, but
not from this adaptive-threshold mechanism.

Do not tune adaptation strength or coverage on this task. Phase 29 retains
ordinary LIF neurons and makes stored axonal delays executable through fixed
history buckets. If fixed delays also fail, stop optimizing row-sequential
MNIST and move to SHD, where timing is intrinsic rather than artificially
constructed from image rows.
