# Gen-17 sparse-spiking local-credit preregistration

Status (2026-08-10): implemented and frozen before observing Gen-17 results.

## Question

Does the validated Gen-16 local score-function rule retain reward-specific
learning when each analog sensory channel is replaced by a sparse binary event?

## Frozen translation

Each of the eight bounded sensory drives is interpreted as a Bernoulli firing
probability and produces one binary event per environment tick. The policy
remains an 8-to-4 linear map, so analog and spiking arms have identical
parameter counts. The local update remains:

`normalized_return * presynaptic_event * (chosen_action - action_probability)`

No target action, label, surrogate gradient, or autograd update reaches the
policy. Seeded spike generators make baseline/final evaluation reproducible.

## Arms

1. frozen random spiking policy;
2. oracle food-direction reflex;
3. manual analog score-function reference;
4. manual sparse-spiking score-function policy;
5. the same spiking policy with reward shuffled between agents.

## Frozen budget

- seeds 172, 173, and 174;
- 1,000 agents per independent arm;
- 300 baseline, 1,800 training, and 300 final steps;
- 30-step normalized returns and a 12-step reward delay;
- SGD-equivalent local updates at 0.02, weight decay 0.0001;
- discount 0.99 and gradient norm cap 1.0;
- eight input channels and four outputs in both analog and spiking policies.

## Decision rule

A pass requires:

- exact static reset and a positive oracle control;
- analog and spiking gains of at least +0.10 in the mean and on at least two
  of three seeds;
- spiking mean gain no more than 0.15 below analog mean gain;
- analytic local-gradient error below 1e-5;
- training and evaluation event density between 5% and 40%;
- spiking final fitness at least +0.10 above static and shuffled reward in the
  mean, and above shuffled reward by +0.10 on at least two seeds.

A pass opens only a larger-seed replication of sparse-spiking credit before
any memory mechanism. A stop rejects or redesigns this translation without a
spike-rate, learning-rate, reward, budget, or seed sweep.
