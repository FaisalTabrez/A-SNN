# Gen-13 local-plasticity analysis

Status (2026-08-10): completed with terminal decision `stop`.

## Result

The fixed sensor failure reduced frozen-backbone accuracy from 60.068% to
56.579%, a 3.488-point shift. The conventional controls verified that the
task remained adaptable: full fine-tuning recovered 3.269 points on all three
seeds, and autograd readout adaptation recovered 2.049 points on two of three
seeds.

The manual analog and spiking three-factor rules recovered only 0.420 and
0.410 point. Neither achieved a two-point gain on any seed. They also missed
the registered causal thresholds:

| Measurement | Analog | Spiking | Required |
| --- | ---: | ---: | ---: |
| Adaptation gain | 0.420 pt | 0.410 pt | at least 2.0 pt on 2/3 seeds |
| Fast-weight removal cost | 0.420 pt | 0.410 pt | at least 0.5 pt on 2/3 seeds |
| Class-shuffle cost | 0.463 pt | 0.468 pt | at least 0.5 pt on 2/3 seeds |
| Trace density | 64.906% | 20.000% | spiking: 5–35% |
| Source forgetting | 0.000 pt | 0.000 pt | no more than 0.5 pt worse than readout |

The spiking rule used about 16,768 active fast synapses with mean absolute
weight 0.00317. Its healthy 20% activity, broad fast-synapse occupancy, and
zero source forgetting rule out silent traces, missing storage capacity, and
catastrophic forgetting as explanations. The learned update was simply too
small and insufficiently class-specific to reproduce the useful output-layer
credit assignment.

## Decision

Accept the source, distribution-shift, conventional adaptation, activity,
capacity, and retention controls. Reject useful local output plasticity and
causally class-specific spiking fast weights under the frozen protocol. Honor
the stored `status=stop` and do not sweep learning rate, density, epochs,
normalization, damage, or budgets.

STW/LTW consolidation, replay, neuromodulation, and structural plasticity do
not open. This completes the Gen-9–13 supervised continual-adaptation branch.
