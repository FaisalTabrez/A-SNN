# Phase 31 screen analysis: SHD temporal transfer plumbing

## Scope

This is the explicitly registered plumbing screen, not the Phase 31 result. It
uses one seed, 1,000 training examples, 500 test examples, two epochs, and only
the paired sparse arms. Chance accuracy on the 20-class task is 5%.

## Result

| Arm | Train accuracy | Test accuracy | Active edges | Delayed edges |
|---|---:|---:|---:|---:|
| Sparse no delay | 6.0% | 6.0% | 1,212 | 0 |
| Sparse distance 0-2 | 5.9% | 6.0% | 1,212 | 348 |

The screen produces no accuracy separation. Both test results are only one
point above chance, and the zero-point paired difference cannot be interpreted
as evidence for or against delay transfer under this deliberately tiny budget.

## Systems and dynamics checks

- The official 700-channel event tensors loaded and trained on CUDA.
- The fixed-distance arm executed 348 delayed recurrent edges with mean delay
  `1.002`, confirming that the 0/1/2 routing path is active.
- Initial/final hidden event rates were approximately `0.364/0.364` without
  delays and `0.369/0.370` with delays. There is no dead-network collapse.
- LTW movement was `~0.00060`, expected after only one post-warmup epoch.
- Neither arm showed lower or upper LTW saturation.
- The delayed arm's final event rate was `1.015x` the paired no-delay rate, well
  inside the registered stability band.
- CUDA throughput was about 7,430 test examples/s without delays and 5,023/s
  with delays. Executable delay buckets therefore cost about 32% throughput in
  this small screen.

## Goal sanity check and decision

The screen validates the implementation contract: data acquisition, temporal
binning, CUDA training, sparse gradients, executable delays, paired metrics,
and serialization all work. It does not validate learning or cross-domain
transfer. The event-count controls were intentionally omitted, so it also does
not yet prove that labels are learnable under the current preprocessing.

Decision: do not create or tune Phase 32 from this screen. Run the already
registered full Phase 31 matrix using all 8,156 training samples, all 2,264 test
samples, three seeds, 15 epochs, and both event-count controls. Only that result
can apply the `+1.0` point delay-transfer gate. If the full sparse arms remain
near chance while the count controls learn, the next phase will diagnose the
sparse temporal representation; if the delay arm passes, the next phase will
add published SHD baselines and capacity-matched comparisons.
