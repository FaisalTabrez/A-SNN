# Gen-16 local score-function credit preregistration

Status (2026-08-10): implemented and frozen before observing Gen-16 results.

## Question

Can an explicit local three-factor score-function rule reproduce matched
autograd reward learning when both update the same linear 8-to-4 policy?

Gen-15 proved that the reward protocol carries usable identity-specific credit,
but its effect was weak and seed-sensitive. Gen-16 therefore isolates the
missing credit term before introducing spiking, hidden-state, or structural
plasticity.

## Frozen mechanism

At each sensory-to-action synapse, the manual ascent direction is:

`normalized_return * presynaptic_sensor * (chosen_action - action_probability)`

The chosen-minus-probability term is the exact local derivative of the sampled
softmax log policy. Thirty-step normalized discounted returns span the fixed
12-step reward delay. The rule receives no target action, label, or autograd
gradient. A manual bias trace uses the same post-synaptic and reward factors.

## Arms and controls

Five independent, identically seeded worlds are used:

1. frozen random linear policy;
2. oracle food-direction reflex;
3. matched linear policy trained by autograd REINFORCE and SGD;
4. the manual local score-function rule;
5. the same manual rule with reward shuffled between agents.

Autograd and manual arms share initialization, optimizer-equivalent SGD,
rollout length, return normalization, weight decay, gradient clipping, action
randomness, and environment randomness. Baseline and final evaluations rebuild
the same seeded world and action stream.

## Frozen budget

- seeds 169, 170, and 171;
- 1,000 agents per independent arm;
- 300 baseline and 300 final evaluation steps;
- 1,800 training steps in 30-step rollouts;
- 64 food and 64 toxin objects;
- discount 0.99, learning rate 0.02, weight decay 0.0001;
- gradient norm cap 1.0 and no entropy bonus.

## Decision rule

A pass requires all of the following:

- exact static reset within 1e-6;
- oracle fitness above static fitness;
- autograd and manual policies each gain at least 0.10 fitness per 1,000 steps
  on at least two of three seeds and in the mean;
- the manual final mean lies within 0.25 fitness per 1,000 steps of autograd;
- the manual analytic gradient matches autograd within 1e-5;
- the manual policy beats static and shuffled reward by 0.10 in the mean, and
  beats shuffled reward by 0.10 on at least two of three seeds.

A pass opens only translation of this validated rule into sparse spiking
eligibility. A stop rejects or redesigns this local credit formulation. It does
not authorize learning-rate, budget, reward-shaping, seed, or threshold sweeps.
