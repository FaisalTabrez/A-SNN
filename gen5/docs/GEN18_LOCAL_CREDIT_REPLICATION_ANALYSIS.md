# Gen-18 held-out local-credit replication analysis

Status (2026-08-10): `stop`; the local reward-credit program is closed.

## Result

The exact manual gradient and environmental controls remained valid, and the
aggregate behavior was directionally positive. The effect did not meet the
frozen replication or uncertainty gates.

- Static baseline/final fitness: `-1.275667` / `-1.275667`.
- Oracle fitness: `+9.357667`.
- Correct-reward local baseline/final fitness: `-1.275667` / `-0.480000`.
- Mean local learning gain: `+0.795667`.
- Gain standard deviation: `1.310007`.
- Lower normal-approximation 95% gain bound: `-0.016284`.
- Gain-qualified seeds: `5/10`, below the frozen `7/10` requirement.
- Correct minus shuffled final margin: `+0.510000`.
- Lower 95% reward-identity margin: `-0.013376`.
- Reward-identity-qualified seeds: `6/10`, below `7/10`.
- Maximum manual-gradient error: `3.73e-09`.

Five seeds lost fitness after correct-reward learning. Four seeds showed a
negative correct-versus-shuffled margin. The positive mean was driven by a
small subset of large gains, particularly seeds 181, 182, 185, and 188.

## Sanity check against the project goal

The experiment supports exact local score-gradient computation and suggests
that local reward credit can occasionally find useful behavior. It does not
support a reliable learning algorithm: the lower confidence bounds include
zero, replication counts fail, and shuffled reward also improves on average.

According to the preregistration, the following are now closed:

- further seeds, learning-rate sweeps, or reward-shaping rescue of this rule;
- the one-sample Bernoulli sparse translation;
- STW/LTW, replay, and structural plasticity built on this unstable learner.

The broader brain-inspired project remains open. Its strongest supported
mechanism is causal residual LIF state on SHD and SSC. Gen-19 begins a distinct
external-generalization program by testing that frozen mechanism on real
N-MNIST event-vision data.

## Provenance

- Raw archive SHA-256:
  `F96182405A2D051AA91D6569789F8D00C0A08480A89C1ACA367C39F265761834`.
- Extracted evidence:
  `gen5/outputs/gen18_local_credit_replication_cuda_2026-08-10/`.
