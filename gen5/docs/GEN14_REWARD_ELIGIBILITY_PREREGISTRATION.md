# Gen-14 reward-modulated embodied eligibility preregistration

Status (2026-08-10): implemented and frozen before observing Gen-14 results.

## New hypothesis

Gen-13 showed that supervised three-factor class-error updates do not reproduce
autograd output credit. Gen-14 starts a distinct program: local eligibility
traces are reinforced by delayed scalar reward from an embodied foraging
environment. No target action, class label, or autograd gradient reaches the
plastic weights.

Each agent maintains a sensor trace, an action-surprise eligibility trace, a
reward baseline, and fast sensor-to-motor weights. The local update is:

`Δw = learning_rate × eligibility × (reward - reward_baseline)`

Food/toxin collisions provide delayed reward and punishment. A small frozen
distance-progress term improves screen sensitivity while remaining a scalar
environmental signal.

## Frozen screen

Seeds are 163–165. Ten thousand agents are split equally between:

1. static random policy;
2. oracle food-direction reflex, as an environment positive control;
3. analog reward eligibility;
4. spiking reward eligibility;
5. spiking eligibility with reward shuffled between agents.

The run uses 600 baseline steps, 3,600 learning steps, 600 frozen evaluation
steps, 12-step collision-reward delay, eligibility decay 0.95, trace decay
0.90, learning rate 0.02, and 20 ms-scale fast-weight decay 0.0001 per step.

## Terminal screen gate

The spiking arm passes only if:

- the oracle beats the static policy;
- post-learning collision fitness improves by at least 0.10 per 1,000 steps;
- it beats both static and shuffled-reward controls by at least 0.10 per 1,000
  steps;
- mean spike density remains between 5% and 35%;
- no more than 5% of fast weights saturate.

A pass opens a separately preregistered confirmation with replicated
trajectory controls, fast-weight removal, reward-delay sweeps chosen before
execution, and source-skill retention. A stop closes this eligibility rule
without learning-rate, shaping, trace-decay, temperature, or world sweeps.
Neither outcome directly establishes STW/LTW, structural plasticity, or
biological equivalence.
