# Gen-15 matched reward-baseline analysis

Status (2026-08-10): **pass**, with a narrow and seed-sensitive learning effect.

## Archived evidence

- Archive: `gen15_reward_baseline_cuda-20260810T153912Z-1-001.zip`
- SHA-256: `2311348DF5C2B87CACD3D75F2EDC3F267A74BC7FB53D86A6194A6FB30F8748F3`
- Extracted results: `gen5/outputs/gen15_reward_baseline_cuda_2026-08-10/`

## Registered gates

The static arm reproduced exactly under the identical seeded reset. The oracle
reached +10.661 net fitness per 1,000 steps, versus -1.263 for static behavior.
Correct-reward REINFORCE improved by +0.992 and finished +0.992 above static
and +1.267 above agent-shuffled reward. All four registered gates therefore
passed.

The pass is not evidence for a strong policy. Final correct-reward fitness was
still -0.271, and the mean gain was dominated by seed 168 (+3.057). Seed 166
regressed by -0.087 and seed 167 improved by only +0.007. Reward identity was
nevertheless informative: correct reward beat shuffled reward on all three
seeds.

## Interpretation

Gen-15 resolves the Gen-14 ambiguity. The delayed scalar reward, sensor/action
interface, and agent-identity assignment can support conventional learning
under stationary evaluation. It does not rescue Gen-14's eligibility rule and
does not validate STW/LTW, replay, structural plasticity, or neuromorphic
efficiency.

The next experiment must isolate credit assignment. It may derive a local rule
from the successful score-function gradient, but must first demonstrate
mathematical and behavioral equivalence to a matched autograd policy. Spiking
and topology changes remain closed until that equivalence passes.
