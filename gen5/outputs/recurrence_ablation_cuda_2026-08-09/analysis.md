# Phase 23 CUDA recurrence-ablation analysis

Run date: 2026-08-09

Source archive: `recurrence_ablation_cuda.zip`

Archive SHA-256: `6FB78FCC779C57EFEEA83D7E0CC7DA70B4D460FCA3A77B49A5F6DD20914E3B55`

This is a three-seed engineering-validation result, not a final generalization
claim. The reserved final-test complement was not used.

## Results

| Representation | Linear accuracy | MLP accuracy | Active edges |
|---|---:|---:|---:|
| Raw intensity | 85.940% | 95.140% | 0 |
| Sensor temporal | 89.787% | 92.213% | 0 |
| Hidden feedforward temporal | 89.393% | 92.387% | 128 |
| Full feedforward temporal | 91.407% | 92.613% | 128 |
| Hidden recurrent temporal | 89.527% | 92.513% | 384 |
| Full recurrent temporal | 91.513% | 92.560% | 384 |

Paired recurrence effects:

- hidden-only linear: `+0.133` percentage points, positive in 2/3 seeds;
- hidden-only MLP: `+0.127` points, positive in 1/3 seeds;
- full-state linear: `+0.107` points, positive in 3/3 seeds;
- full-state MLP: `-0.053` points, negative in 3/3 seeds.

The full feedforward representation already gained `+1.620` points over the
sensor-only linear readout. Adding 256 recurrent edges increased mean hidden
event rate from `0.021194` to `0.023677` (about `11.7%`) but did not produce a
practically meaningful accuracy gain.

## Decision

The pre-registered recurrence gate required a mean gain of at least `0.5`
points and improvement in at least two of three seeds. Phase 23 fails the
effect-size requirement for both readouts. The evidence supports sparse random
feedforward expansion as a useful representation, but it does not support
recurrence, LTW learning, or structural plasticity on static MNIST.

Static MNIST is closed as a recurrence/plasticity justification. Phase 24
presents one image row per step and exposes only final hidden state, forcing
the graph to preserve information across time before recurrence is evaluated
again.
