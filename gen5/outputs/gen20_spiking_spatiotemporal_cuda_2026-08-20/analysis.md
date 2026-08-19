# Gen-20 spiking spatial-temporal translation analysis

Date analyzed: 2026-08-20

Execution backend: CUDA on Colab Pro L4

Protocol status: frozen preregistration

Terminal decision: `stop` (`no_new_spiking_arm_passed_screen`)

## Provenance and integrity

The original downloaded result directory was copied into this evidence folder
without modifying its canonical artifacts. Every artifact hash in
`gen20_spiking_spatiotemporal_manifest.json` matches the copied file. The
original bundle SHA-256 is
`097D79D15E6EE2AD471E1A0409527C7730BC52ACF17AC5586ACF037F22F844CB`.

The dataset metadata also matches the previously cached native N-MNIST data:
60,000 total training examples, 10,000 official test examples, 10 temporal
bins, two polarities, 34 x 34 pixels, train density 0.08133554, and test density
0.08149972. The screen used the declared seed 220 and 20,000 training examples.

## Screen results

| Arm | Validation accuracy | Gap to teacher | Gap to 97.5% gate | Parameters | Activity | Dense MACs | Activity-scaled proxy reduction vs teacher |
|---|---:|---:|---:|---:|---:|---:|---:|
| Spatial-temporal CNN | 99.1165% | 0.0000 pp | passed | 1,551,402 | n/a | 966,173,696 | 1.00x |
| ConvPLIF | 96.2160% | 2.9005 pp | -1.2840 pp | 471,114 | 15.1313% | 99,274,240 | 9.73x (dense proxy) |
| Multiscale residual PLIF | 96.3661% | 2.7505 pp | -1.1339 pp | 435,306 | 12.6924% | 35,999,200 | 74.37x |
| Distilled multiscale PLIF | 96.3327% | 2.7838 pp | -1.1673 pp | 435,306 | 12.9617% | 35,999,200 | 73.97x |

The multiscale residual arm improved on ConvPLIF by only 0.1500 percentage
points. Distillation changed the corresponding result by -0.0333 points, so it
did not close the representation gap. Both proposed arms were numerically
stable and had healthy activity inside the 1-30% gate, while their approximate
operation proxies were far below the dense teacher. Those strengths do not
override the conjunctive accuracy requirement.

## Interpretation and limits

Neither new spiking arm reached the frozen 97.5% validation gate. The empty
promotion, confirmation, and summary tables are therefore expected outputs,
not evidence of interruption. The official test set was not evaluated for
these arms, and no state-removal or temporal-order controls ran. Consequently:

- Gen-20 rejects this particular multiscale residual PLIF translation as the
  bridge to the successful dense spatial-temporal N-MNIST representation.
- Gen-20 does not show that temporal state or temporal ordering caused task
  performance, because the candidate never qualified for those controls.
- It does not evaluate structural plasticity, dual-memory consolidation,
  learned delays, local reward credit, or hardware energy.
- The operation values remain analytical proxies; they are not energy or
  latency measurements.
- No rescue sweep is authorized under the preregistration.

## Artifact note

The original PNG is visually blank because the plotting function expected a
confirmation summary even after a valid early stop. The canonical numerical
artifacts are unaffected. `gen20_spiking_spatiotemporal_screen_replot.png` is a
locally derived visualization of the exact screen records, and the plotting
code now falls back to screen data for future early-stop runs.

## Program decision

The project should not auto-create Gen-21. The next declared milestone is an
evidence synthesis that separates supported findings from rejected mechanisms
and unanswered causal questions. The most defensible subsequent experiment
should be chosen from that ledger under matched parameter, compute, seed,
optimization, and architecture controls.
