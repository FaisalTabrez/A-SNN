# Gen-21 matched causal mechanism benchmark preregistration

Date frozen: 2026-08-20

## Question

Do dynamic topology, dual short/long-term memory, learned delays, or local
reward credit improve post-shift adaptation on a supported real event-audio
representation when data, backbone, allocated parameter slots, active slot
budget, seeds, update count, and model selection are held fixed?

## Supported boundary

The benchmark uses Spiking Speech Commands and the residual Conv1D plus LIF
state backbone supported by Phases 47 and 48. The backbone is trained once per
seed, selected only by a disjoint source-validation split, and frozen before
adaptation. Gen-21 therefore tests mechanisms in a matched residual readout; it
does not claim that a result transfers to every synapse in the backbone.

The distribution shift is a deterministic 35% sensor-bank lesion. Official
SSC validation examples are partitioned into disjoint source-validation and
adaptation sets. The official test set is never used for training or selection.

## Arms and confound controls

- `static_backbone`: no adaptation;
- `global_gradient_control`: ordinary gradient adaptation upper control;
- `topology_only`: fixed-capacity active mask with utility pruning and
  gradient-supported sprouting;
- `dual_memory_only`: rapidly decaying STW plus consolidated LTW;
- `learned_delay_only`: matched 0/1/2-step connection slots;
- `local_credit_only`: three-factor sampled-action eligibility update using
  scalar correctness reward and no autograd through the readout;
- `combined`: may run only when all four individual mechanisms clear screening.

All adaptive arms allocate the same `[class, residual feature, delay slot]`
tensor and begin from zero residual adaptation. Their active mask cardinality,
batch order, examples, epochs, and nominal update count are matched. Dense MACs
are not relabeled as synaptic operations; both allocated slots and active-slot
operation proxies are reported.

## Frozen gates

An individual arm advances from seed 321 only if all three conditions hold:

1. shifted-test gain over the paired static backbone is at least 1.0 point;
2. clean-test retention drop is no more than 1.5 points;
3. the paired mechanism causal control costs at least 0.5 point.

Promoted arms run on seeds 322–324. Confirmation requires adaptation and causal
margins on at least two of three seeds while preserving the retention gate.
Topology is tested by mask shuffling, dual memory by LTW removal, delays by
delay-slot reversal, and local credit by training with sample-shuffled reward.

## Interpretation limits

Passing supports only a causal contribution in this bounded frozen-readout SSC
adaptation setting. It does not establish whole-network structural plasticity,
biological fidelity, direct energy savings, neuromorphic superiority, or
general intelligence. Failure closes or redesigns the corresponding mechanism
instead of being rescued by an unregistered hyperparameter sweep.
