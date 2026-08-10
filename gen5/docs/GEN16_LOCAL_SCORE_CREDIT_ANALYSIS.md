# Gen-16 local score-credit analysis

Status (2026-08-10): **pass**, with exact algorithmic equivalence and a small
behavioral effect.

## Archived evidence

- Archive: `gen16_local_score_credit_cuda-20260810T160150Z-1-001.zip`
- SHA-256: `9962AFCAD961DF889281B4CA8D1D4607BD7356B6D2E729F78BE7B56C0E8979F4`
- Extracted results: `gen5/outputs/gen16_local_score_credit_cuda_2026-08-10/`

## Result

Static reset was exact and the oracle reached +10.176 fitness per 1,000 steps.
Autograd and manual score-function policies were behaviorally identical: both
gained +0.183 and finished at +0.300. Their final gap was exactly zero. The
manual gradient differed from autograd by at most 2.79e-9.

The manual rule gained on all three seeds, with gains of +0.030, +0.170, and
+0.350; two cleared the frozen +0.10 threshold. Correct reward finished +0.183
above static and +0.249 above shuffled reward, and beat shuffled reward on all
three seeds.

## Interpretation

This is the first AMMC experiment in the reward program to establish an exact
local-credit implementation rather than merely healthy activity. The result is
still a linear-policy proof of mechanism, not evidence for a capable standalone
SNN or continuous structural learning. The weight change was small and the
behavioral advantage remains far below the oracle.

Decision: accept the local score-function rule and open only a
parameter-matched sparse sensory-spike translation. STW/LTW, consolidation,
replay, structural plasticity, and hardware-efficiency claims remain closed.
