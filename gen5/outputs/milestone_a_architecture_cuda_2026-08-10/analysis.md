# Milestone A: architecture screen and confirmation

Archive SHA-256:
`FDF13E7CC2CF3A600389041D0B044E9564BD26E7C4DC736A777A779D304591C3`

## Screen

All arms used seed 142, 15,000 training examples, 3,000 validation examples,
3,000 test examples, four epochs, and approximately 133,631 trainable
parameters. Promotion was determined from validation accuracy, parameter
budget, and LIF activity—not screen test accuracy.

| Arm | Validation | Gap from TCN | Test | Activity | Promoted |
| --- | ---: | ---: | ---: | ---: | --- |
| Dilated TCN | 50.533% | 0.000 pt | 48.000% | 42.778% ReLU | yes |
| Residual LIF | 44.333% | -6.200 pt | 41.000% | 10.547% spikes | no |
| Hierarchical residual analog | 41.967% | -8.567 pt | 40.400% | 63.005% analog | no |
| Hierarchical residual LIF | 37.067% | -13.467 pt | 35.200% | 8.812% spikes | no |
| Conv1D | 31.433% | -19.100 pt | 30.967% | 17.793% ReLU | no |

Both LIF candidates had valid parameter ratios and non-degenerate spike rates.
They failed solely on the preregistered two-point validation promotion gate.
The analog hierarchy also failed its accuracy gate. This rules out blaming the
result on dead spiking activity or a gross parameter mismatch.

## Confirmation

Only the dilated TCN was eligible for full official SSC confirmation. Across
seeds 142–144 it reached `59.170% +/- 0.230` points on 20,382 test examples,
with `56,392` examples/s mean T4 throughput. The result closely reproduces the
Phase 49 TCN (`59.225% +/- 0.541`), supporting pipeline consistency.

No causal model was promoted, so there are intentionally no new direct-only,
state-only, or shuffled-state confirmation measurements. The empty causal
panel in the plot is the visible consequence of the stop gate, not missing
data.

## Sanity check

Milestone A returns:

```text
status: stop
qualified_arms: []
next_milestone: close_architecture_branch
```

The earlier causal mechanism finding remains supported: residual LIF state
carried sample-specific complementary information on SHD and SSC. Milestone A
answers a different question and fails it: none of the tested residual-state
architectures was accurate enough in the preregistered screen to justify a
full competitive confirmation. The current AMMC Gen-5 architecture is
therefore not a credible best-SNN or Transformer-alternative candidate.

## Decision

Honor the terminal gate. Do not run Milestone B, do not perform a rescue
threshold sweep, and do not reinterpret the screen as a hardware-efficiency
result. Close the architecture branch, update the final claim ledger, and
retain any successor architecture as a separately preregistered generation.
