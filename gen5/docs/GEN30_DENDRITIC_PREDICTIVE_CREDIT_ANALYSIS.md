# Gen-30 dendritic predictive-credit analysis

Date: 2026-08-20

## Decision

**Stop.** The proposed dendritic predictive-credit (DPC) rule failed the
preregistered absolute accuracy, retention, retention-drop, and seed-
replication gates. Fixed-topology SSC transfer is not authorized.

This analysis is based on the complete ten-seed console summary supplied from
the Colab L4 run. The generated bundle is still required for manifest and
record-level verification.

## Gate outcome

| Gate | Result | Outcome |
|---|---:|---|
| Context-B accuracy | 52.8125% | Fail (minimum 80%) |
| Context-A retention | 40.6348% | Fail (minimum 75%) |
| Context-A retention drop | 14.8730 points | Fail (maximum 5 points) |
| Joint parity with e-prop | +0.5127 point | Pass |
| Margin over shuffled apical | +22.7148 points | Pass |
| Margin over no eligibility | +21.5771 points | Pass |
| Margin over shuffled modulator | +23.8916 points | Pass |
| Mean hidden spike activity | 3.6517% | Pass (1-30%) |
| Qualified seeds | 0/10 | Fail (minimum 8/10) |

## Interpretation

The three causal-control margins are large. Temporally extended eligibility
and correctly aligned apical/modulatory identity are therefore necessary for
the observed above-chance learning. This is a bounded component-level result;
it is not evidence that the full DPC mechanism solves delayed local credit.

DPC and e-prop were essentially tied (46.7236% versus 46.2109% joint
accuracy). The local predictive residual did not demonstrate a meaningful
incremental benefit over the simpler fixed broadcast rule. Absolute learning
also remained weak: both local rules were far below the 80% new-context gate.

The BPTT arm reached 81.3477% on Context B and 98.4082% on Context A before the
switch, showing that the task and model are learnable. Its final Context-A
accuracy collapsed to 27.1094%, however, and activity rose to 39.3860%. The
experiment therefore exposes two independent deficiencies:

1. local hidden-layer credit is too weak relative to BPTT; and
2. neither local nor global training protects the earlier context mapping.

## Claim boundary

Gen-30 supports only the statement that aligned teaching identity and temporal
eligibility causally improve this fixed-topology synthetic task relative to
their registered ablations. It does not support DPC as a sufficient learning
rule, continuous-learning adaptation, structural plasticity, dual-memory,
real-dataset transfer, or hardware-energy claims.

## Next design requirement

Do not tune Gen-30 after seeing these results. A successor must introduce and
preregister a genuinely new mechanism that separately addresses credit
strength and interference protection under matched capacity. It should retain
the same task first, add a direct no-prediction control, and require causal
benefit over both e-prop and an explicit retention baseline before any SSC
transfer.
