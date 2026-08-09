# Phase 17 embodied action controls: analysis

Date: 2026-08-09

Bundle SHA-256: `FCC50C9B5580BABB26B10A34950C3C4748A2CFBF0CC3442FDB04106290BDBDC3`

The bundle is internally complete: 54 condition summaries and 270
seed-level records covering three worlds, three sensor-noise levels, five
held-out seeds, and six policies.

## Main result

The action-coverage confound identified in Phase 16 is resolved. Both trained
frozen-trace adapters beat the full-activity random controller and the
normalized fixed analog AMMC decoder in every paired condition (`45/45`).

| Policy | Mean fitness | Positive runs | Cue-action coverage | Oracle agreement |
|---|---:|---:|---:|---:|
| Augmented adapter | 1.860 | 45/45 | 100.0% | 64.8% |
| Base adapter | 1.721 | 45/45 | 100.0% | 63.5% |
| Direct sensor oracle | 3.083 | 45/45 | 100.0% | 100.0% |
| Fixed analog cardinal | -0.139 | 19/45 | 73.3% | 26.0% |
| Fixed motor spiking | -2.799 | 14/45 | 5.1% | 39.3% |
| Random cardinal | -0.211 | 27/45 | 100.0% | 25.0% |

Paired mean-fitness differences:

- augmented adapter minus random: `+2.071`, wins `45/45`;
- base adapter minus random: `+1.932`, wins `45/45`;
- augmented adapter minus fixed analog: `+1.998`, wins `45/45`;
- base adapter minus fixed analog: `+1.859`, wins `45/45`;
- augmented adapter minus base adapter: `+0.139`, wins `29/45`.

## Interpretation

- Movement opportunity alone does not explain the adapter result: the random
  controller acts at full magnitude on every cue-bearing step and still loses
  every paired comparison.
- A simple analog calibration of the frozen motor channels is insufficient:
  both adapters beat that decoder in all paired conditions.
- The frozen AMMC trace therefore contains action-relevant information that a
  small trainable readout can use in closed-loop control.
- The direct sensor oracle remains the clearest ceiling, especially in the
  simple world. Occasional adapter wins in `moving_toxins` reflect closed-loop
  trajectory dynamics, not a generally superior sensor policy.
- Augmentation provides only a small and inconsistent gain over clean adapter
  training. It should remain an optional robustness treatment, not a headline
  architectural claim.

## Decision

Close the first bot-world validation cycle. The next phase is an external,
event-coded MNIST benchmark with the sparse AMMC substrate frozen and with
raw-pixel linear and MLP baselines. This tests whether AMMC traces provide a
useful representation outside the foraging task without conflating the result
with recurrent training or structural plasticity.

## Scientific boundary

Phase 17 validates representation-to-action decoding from a frozen sparse
substrate. It does not demonstrate autonomous policy discovery, universal
transfer, or superiority to trained dense neural networks.
