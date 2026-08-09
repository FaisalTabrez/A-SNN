# Phase 30 analysis: trainable recurrent delays

## Result

Trainable delay assignment does not pass the registered improvement gate. The
fixed heterogeneous distance assignment remains the retained AMMC baseline.

| Arm | Linear accuracy | MLP accuracy | Linear gain vs fixed | MLP gain vs fixed |
|---|---:|---:|---:|---:|
| Fixed distance 0-2 | 54.053% | 64.073% | control | control |
| Learned soft, distance init | 53.467% | 64.013% | -0.587 pt | -0.060 pt |
| Learned straight-through, distance init | 54.307% | 64.173% | +0.253 pt | +0.100 pt |
| Learned soft, flat init | 50.280% | 57.993% | -3.773 pt | -6.080 pt |

The straight-through arm improved two of three seeds for each readout, but only
one linear seed cleared a practical `+0.5` point gain and the mean gain remained
below the pre-registered threshold. Delay learning added 768 trainable logits.

## Mechanistic diagnosis

- Distance-initialized straight-through gates changed only about five recurrent
  assignments for the linear readout and one for the MLP. Their marginal gains
  do not justify the added optimizer state or model complexity.
- Distance-initialized soft gates retained low entropy (`~0.23`) but slightly
  degraded both mean accuracies. Mixing multiple histories is not equivalent to
  routing an edge through one discrete axonal delay.
- Flat soft initialization changed 29-42 assignments, retained high entropy
  (`~1.06`), and collapsed its selected mean delay toward zero (`~0.20-0.25`).
  The optimizer did not rediscover the balanced 0/1/2 structure.
- LTW movement, event rates, and saturation remained stable. The failure is an
  assignment-optimization result, not an activity or numerical failure.

## Goal sanity check and decision

Phase 29 remains the strongest causal conventional-task result: fixed,
heterogeneous delays improve the same sparse graph by about 7.6-8.1 points
without adding neurons or edges. Phase 30 shows that a naive differentiable
delay parameterization is not automatically better.

The project has evidence for a useful temporal-routing mechanism, but not a
state-of-the-art MNIST model: the retained sparse arm reaches `64.07%` MLP
accuracy versus `94.21%` for the raw-pixel MLP. MNIST mechanism tuning ends
here. Phase 31 transfers the fixed distance 0/1/2 winner to Spiking Heidelberg
Digits, where event timing is intrinsic rather than imposed by row scanning.
