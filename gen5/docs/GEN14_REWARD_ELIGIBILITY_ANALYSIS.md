# Gen-14 reward-eligibility analysis

Status (2026-08-10): completed with terminal decision `stop`.

## Result

The oracle food reflex reached 8.381 net collision fitness per 1,000 steps,
compared with 0.641 for the static policy. This is a strong positive control:
the environment, directional sensors, motor mapping, and reward accounting can
support substantially better behavior.

The spiking eligibility arm rose from -2.907 during the cold-start baseline
to -0.109 during final evaluation, apparently clearing its within-arm gain
gate. That comparison is not evidence of learning because the unchanged
static arm rose even more, from -3.342 to +0.641. The phase transition includes
agent acceleration, environment evolution, and reward-buffer warm-up.

The causal between-arm comparisons reject the learning mechanism:

| Final measurement | Fitness / 1,000 steps |
| --- | ---: |
| Static random | +0.641 |
| Oracle food reflex | +8.381 |
| Analog eligibility | -0.572 |
| Spiking eligibility | -0.109 |
| Spiking shuffled reward | +0.052 |

Correctly rewarded spiking eligibility finished 0.750 below static and 0.161
below shuffled reward. All three seeds showed the same broad pattern of an
oracle advantage without reward-specific local learning.

Spiking density was healthy at 20.04%, mean absolute fast weight reached
0.0722, and no weights saturated. The result is not attributable to silent
spikes or clipping. The update changed weights, but those changes did not
encode useful agent-specific reward credit.

## Decision

Accept the environment, sensor, motor, oracle, trace-activity, and weight-range
controls. Reject the interpretation of within-phase improvement as learning,
reject reward-specific eligibility, and accept the stored `status=stop`.

Do not sweep learning rate, shaping strength, trace decay, reward delay,
temperature, population size, or world configuration. No causal confirmation,
STW/LTW consolidation, replay, neuromodulation, or structural-plasticity phase
opens from this result.
