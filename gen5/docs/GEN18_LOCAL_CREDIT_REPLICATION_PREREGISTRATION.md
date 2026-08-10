# Gen-18 held-out local-credit replication preregistration

Status (2026-08-10): implemented and frozen before observing Gen-18 results.

## Question

Does the exact analog local score-function rule found in Gen-16 produce a
repeatable, reward-specific behavioral gain on ten untouched random seeds?

Gen-17 showed that the analog reference did not reproduce on seeds 172-174.
That result makes robustness the blocking question. Gen-18 adds no new
learning mechanism, representation, memory, spike encoder, or structural
plasticity.

## Frozen protocol

- Seeds: `180..189`.
- Agents per seed: `1,000`.
- Food/toxin objects: `64/64`.
- Evaluation/training steps: `300/1,800`.
- Rollout/reward delay: `30/12` steps.
- Learning rate/weight decay: `0.02/0.0001`.
- Discount/gradient clip: `0.99/1.0`.
- Policy: the Gen-16 dense `8 -> 4` linear stochastic policy.
- Identical-reset arms:
  `static_linear_policy`, `oracle_food_reflex`,
  `manual_local_score_policy`, and `manual_local_shuffled_reward`.

The fresh seed range must not be changed after results are inspected.

## Pass gates

Every gate must pass:

1. Static reset drift is at most `1e-6`.
2. The oracle outperforms the static policy.
3. Correct-reward local learning gains at least `+0.10` fitness per 1,000
   steps on average, reaches that threshold on at least `7/10` seeds, and
   has a positive normal-approximation 95% lower confidence bound.
4. Correct-reward local learning beats the static final policy by at least
   `+0.10` on average and on at least `7/10` seeds, with a positive lower
   95% confidence bound.
5. Correct-reward local learning beats the shuffled-reward final policy by
   at least `+0.10` on average and on at least `7/10` seeds, with a positive
   lower 95% confidence bound.
6. Maximum manual-gradient error is at most `1e-5`.

## Decision

- `pass`: derive one new temporal/rate spike encoding while retaining the
  validated local-credit rule.
- `stop`: close the local reward-credit program. Do not rescue it with a
  parameter sweep or more seeds.

Run from the repository root:

```bash
python gen5/examples/gen18_local_credit_replication.py \
  --device cuda \
  --output-dir gen5_outputs/gen18_local_credit_replication_cuda
```
