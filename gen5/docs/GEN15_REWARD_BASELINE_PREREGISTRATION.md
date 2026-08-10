# Gen-15 matched reward-baseline preregistration

Status (2026-08-10): implemented and frozen before observing Gen-15 results.

## Question

Can a conventional shared-policy REINFORCE learner extract useful credit from
the exact delayed scalar-reward protocol that failed Gen-14?

Gen-15 is a diagnostic, not a new biological mechanism. It addresses two
ambiguities exposed by Gen-14: the cold-start and final evaluations were not
drawn from identical world states, and no conventional reward-trained control
proved that the scalar reward was learnable.

## Frozen protocol

Seeds are 166–168. Each strategy receives an independent 1,000-agent tensor
world with identical seeded initial states for baseline and final evaluation.
The unchanged static policy must reproduce its baseline exactly. Four arms are
registered:

1. static random policy;
2. oracle food reflex;
3. shared 8→32→4 REINFORCE policy with correct reward;
4. the same REINFORCE policy with reward shuffled between agents.

Training uses 1,800 steps, 30-step rollouts, AdamW at 0.003, discount 0.99,
entropy coefficient 0.01, 12-step collision delay, and the same frozen scalar
progress term as Gen-14. Baseline and final evaluation each use 300 steps and
the same environment/action random seeds.

## Terminal gate

The diagnostic passes only if:

- static baseline and final fitness match within 1e-6;
- the oracle beats static;
- correct-reward REINFORCE gains at least 0.10 fitness per 1,000 steps on at
  least two of three seeds;
- it beats both static and shuffled-reward REINFORCE by at least 0.10.

A pass establishes only that the reward protocol supports identity-specific
conventional learning. It permits theoretical derivation of a new local rule;
it does not validate Gen-14 or open STW/LTW. A stop requires redesigning the
reward/evaluation protocol before any new local-learning experiment. No
learning-rate, architecture, rollout, shaping, delay, or world sweep is
authorized within Gen-15.
