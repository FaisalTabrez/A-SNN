# Gen-21 matched causal mechanism benchmark analysis

Date: 2026-08-20

Status: `pass` for one bounded mechanism; analysis reconstructed from the
terminal Colab log. The result bundle remains required for canonical archival
and verification of all screen records.

## Result

Gen-21 used 20,000 SSC source-training examples, disjoint source-validation
and adaptation subsets of approximately 3,000 examples each, 8,000 test
examples, and a deterministic 35% sensor lesion. Only `dual_memory_only`
advanced from screening.

Across three confirmation seeds:

- static shifted accuracy was 38.0667%;
- the matched global-gradient control reached 47.1708%, a +9.1042-point gain;
- dual memory also reached 47.1708%, with the same +9.1042-point gain;
- clean retention fell by 1.2208 points for both adaptive arms;
- removing LTW from dual memory cost 4.3333 points;
- all three seeds passed the registered adaptation and causal thresholds.

The backbone remained active at 8.09%. Every readout allocated 6,090 slots,
activated 2,132, and reported the same 136,448 active-slot operations per
sample and 73,080 adapter bytes. No direct energy claim is authorized.

## Interpretation

This is the program's first preregistered positive causal result for a proposed
adaptive mechanism on a real event dataset: consolidated LTW carries useful
post-lesion information, and the effect repeats across the three confirmation
seeds. It also preserves clean performance within the preregistered limit.

The stronger claim—two memory timescales outperform one—does **not** yet follow.
Dual memory and ordinary global-gradient adaptation had identical final shifted
and clean accuracies. LTW removal tests whether consolidated weights matter, but
does not compare the plasticity-stability tradeoff against a matched
single-memory learner. The experiment also covers only a residual readout, one
sensor-lesion distribution, and a modest 47% source-task operating point.

## Decision

Proceed to one direct dual-memory replication. It must use two sequential,
disjoint sensor lesions and compare dual memory against a slot- and
update-matched single-memory control. It must measure learning of shift B,
retention of shift A, clean retention, LTW removal, and shuffled consolidation.
Do not combine topology, delay, or local-credit mechanisms yet.
