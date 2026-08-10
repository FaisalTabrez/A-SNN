# Gen-17 sparse-spiking local-credit analysis

Status (2026-08-10): `stop`.

## Result

Gen-17 rejected the one-sample Bernoulli translation of the Gen-16 local
score-function rule. Sparse event generation was active and the manual
gradient remained numerically correct, but the translated learner did not
improve behavior and did not preserve reward identity.

- Static spiking baseline/final fitness: `-0.847778` / `-0.847778`.
- Oracle fitness: `+8.320000`.
- Analog local baseline/final fitness: `-0.095556` / `-0.091111`.
- Analog local mean gain: `+0.004444`; only `1/3` seeds met the registered
  `+0.10` gain threshold.
- Correct-reward spiking local final fitness: `-1.238889`.
- Correct-reward spiking mean gain: `-0.391111`.
- Shuffled-reward spiking final fitness: `-0.186667`.
- Correct minus shuffled final margin: `-1.052222`.
- Training/evaluation spike density: `6.369%` / `12.078%`.
- Maximum manual-gradient error: `3.73e-09`.

The gradient and spike-activity controls passed. The analog reference,
spiking-gain, translation, and reward-identity gates failed. This means the
failure cannot be explained by a silent encoder or an algebraic gradient bug.

## Interpretation

Two conclusions are supported:

1. A single Bernoulli sample per channel and decision step is not a valid
   translation of the analog policy under this training budget.
2. The Gen-16 analog gain is not yet a robust foundation: it failed its
   three-seed reference check on fresh seeds.

The next experiment therefore does not add another spike encoder. Gen-18
first performs a ten-seed held-out replication of the analog local-credit
mechanism. Only a statistically positive, reward-specific replication can
authorize a theory-derived temporal spike encoding.

## Provenance

- Raw archive SHA-256:
  `6FDA07EE4FAF5B99FB90E2C68029E38A5C49AE19D4D4951DD87D680F5E95E8D3`.
- Extracted evidence:
  `gen5/outputs/gen17_sparse_spiking_credit_cuda_2026-08-10/`.
