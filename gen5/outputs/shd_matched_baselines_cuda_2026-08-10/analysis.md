# Phase 38 analysis: matched SHD baselines

Archive SHA-256: `D2AD248E20CD0EB551A4B0BB089B969CC1C47E2E0DDE0B0D345AC2D789162928`

## Result

The registered sparse-advantage gate passes. Sparse recurrent AMMC reaches
`79.873%`, compared with `73.763%` for the parameter-matched dense recurrent
LIF: a paired `+6.110` point gain. All three seeds improve and all exceed the
one-point practical threshold. Sparse feedforward AMMC also reaches `79.417%`,
or `+5.654` points over dense LIF.

The recurrence mechanism is not supported. Recurrent sparse AMMC improves over
its feedforward counterpart by only `+0.456` points; one seed declines and none
gains two points. The sparse feedforward transformation, rather than the tested
random recurrence, explains the matched-LIF advantage.

The raw temporal pyramid reaches `77.577%`, beating dense LIF by `+3.813`
points and trailing sparse recurrent AMMC by only `2.297` points. It is about
`5.18x` faster at inference. Sparse recurrence has lower activity than dense
LIF (`13.15%` versus `20.07%`) but is not yet a systems-efficiency win.

The GRU result (`44.464%`) is not a credible superiority comparison. Training
accuracy is high while test accuracy is highly variable, indicating severe
overfitting or an under-tuned baseline. It is retained as a failed reference,
not evidence that GRUs are intrinsically weak.

## Goal sanity check

- Supported: a 512-node sparse feedforward LIF representation with temporal
  decoding beats this matched dense recurrent LIF under the registered setup.
- Not supported: recurrence, polychronization, structural plasticity, hardware
  efficiency, state-of-the-art SHD performance, or general ANN superiority.
- Remaining ambiguity: the sparse gain may come from hard spikes, learned LTWs,
  or merely the wider sparse expansion.

## Decision

Freeze recurrence tuning. Phase 39 uses a paired causal ablation of hard LIF
spikes versus analog leaky dynamics and frozen versus trainable LTWs, with the
raw temporal model retained. The result must identify a mechanism before the
project advances to structural plasticity or wider claims.
