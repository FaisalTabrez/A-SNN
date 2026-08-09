# Phase 43 analysis: validation-selected SHD checkpointing

Archive SHA-256: `663F7AC8BFD01A549981FDA669CA70FFCB4122DF2794A3875F46C65AC5192877`

## Registered gates

| Gate | Required | Observed | Result |
|---|---:|---:|---|
| Sparse checkpoint vs paired raw | +2 points; 6/9 positive | -2.351; 2/9 positive | Fail |
| Sparse variance reduction | At least 25% | Standard deviation rises 21% | Fail |
| Sparse mean retention | No worse than -0.5 points | -0.172 | Pass, but no benefit |

## Interpretation

Validation checkpointing raises raw temporal accuracy from `78.092%` to
`80.374%`. It slightly lowers sparse accuracy from `78.195%` to `78.023%`.
Sparse trails paired raw in seven of nine runs and its validation accuracy is
substantially lower (`83.606%` versus `91.830%`).

The sparse instability is not fixed by ordinary checkpoint selection. Best
sparse epochs average 14.56 of 15, checkpointing increases variance, and several
validation-selected states generalize worse than the final state. The transform
has no reproducible advantage under the registered evaluation protocol.

## Goal sanity check

This closes the current sparse SHD branch. The evidence supports a useful
temporal decoder but not the project's proposed spiking, recurrence, structural
plasticity, LTW/STW, or sparse-efficiency mechanisms on SHD. Phase 44 establishes
validation-calibrated temporal CNN, GRU, and dense-LIF baselines. A future AMMC
redesign must beat those baselines rather than isolated historical champions.
