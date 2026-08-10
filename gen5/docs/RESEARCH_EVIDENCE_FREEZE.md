# AMMC research evidence freeze after Gen-14

Status (2026-08-10): analog local-credit gate passed; sparse translation is open.

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

Selection update (2026-08-10): the user authorized the diagnostic portion of
this work package as Gen-15. It implements identical-reset evaluation and a
matched conventional REINFORCE baseline. It is not a new local mechanism and
does not reopen Gen-14.

Result update (2026-08-10): Gen-15 passed. Correct-reward REINFORCE improved
by +0.992 fitness per 1,000 steps and beat agent-shuffled reward by +1.267;
the static reset was exact. Because final fitness remained -0.271 and the
gain was dominated by seed 168, this is protocol validation rather than a
strong-learning claim.

Gen-16 is the single authorized theory-derived mechanism test. It implements
the exact local score-function factor from REINFORCE on a matched linear
policy and requires analytic-gradient equivalence, behavioral equivalence,
and replicated reward-identity effects. Spiking, STW/LTW, replay, and
structural plasticity remain closed pending that result.

Gen-16 result update (2026-08-10): the exact manual score-function gradient
matched autograd within 2.79e-9, their final behavior was identical, and the
manual rule passed reward identity on all three seeds. Its mean fitness gain
was only +0.183, so this establishes a linear analog proof of mechanism rather
than strong learning.

Gen-17 is the sole open translation. It substitutes parameter-matched binary
sensory events and requires preserved learning, reward identity, and healthy
event density. STW/LTW, replay, structural plasticity, and hardware claims
remain closed until sparse credit passes and replicates.

Gen-17 result update (2026-08-10): the translation returned `stop`. Event
activity and gradient parity were healthy, but correct-reward spiking credit
lost 0.391 fitness per 1,000 steps and finished 1.052 below shuffled reward.
The analog reference gained only 0.004 on the fresh seeds, so the Gen-16 gain
is no longer treated as replicated.

Gen-18 is the sole open experiment. It is a ten-seed held-out replication of
the unchanged analog local-credit rule. A pass requires at least 7/10 seeds
and positive lower 95% confidence bounds for both learning gain and reward
identity. A stop closes the local reward-credit program; no parameter sweep,
new spike encoder, STW/LTW, replay, or structural plasticity is authorized.
