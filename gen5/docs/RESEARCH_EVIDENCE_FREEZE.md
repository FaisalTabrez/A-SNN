# AMMC research evidence freeze after Gen-14

Status (2026-08-10): empirical mechanism expansion paused.

## Sanity check

The project goal remains a temporally capable, locally learning,
brain-inspired SNN. The evidence presently supports only part of that goal:

- residual LIF state is causally informative inside a hybrid temporal model
  on SHD and SSC;
- robust conventional temporal features and readout adaptation work;
- the embodied world exposes a strong, interpretable sensor-action optimum.

It does not yet support:

- competitive standalone spiking inference;
- sample-specific state adapters or associative fast memory;
- supervised or reward-modulated local plasticity;
- continuous learning, STW/LTW consolidation, replay, structural plasticity,
  or a neuromorphic hardware-efficiency advantage.

Four successive adaptation mechanisms failed their frozen causal gates. More
unstructured mechanism phases would now be parameter search disguised as
hypothesis testing.

## Next work package

The next phase is an evidence and theory reset, not Gen-15 model training:

1. freeze the 16-source evidence ledger and publication-grade figures;
2. specify a reset evaluation protocol that replays identical initial world
   states before and after learning;
3. add a matched conventional reward-learning baseline to prove that the
   scalar reward protocol itself is learnable;
4. derive the next local-credit rule from that diagnostic and relevant
   literature before implementing it;
5. require a preregistered causal identity control and compute budget.

This keeps the original ambition intact while preventing negative results from
being hidden by additional phases or hyperparameter rescue attempts.
