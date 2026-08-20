# Gen-30 dendritic predictive-credit preregistration

Date frozen: 2026-08-20

## Question

Can a fixed-topology, compartmental local-learning rule assign delayed credit
and retain a conflicting association without global backpropagation through
time?

## Microtask

The Delayed Contextual Binding task emits one of four sensory cues, a brief
context cue, distractor spikes, a blank delay, and a final query pulse. Context
A maps cues directly to four actions. Context B uses the fixed conflicting
permutation `[1, 0, 3, 2]`. Models learn A first, then B, and are evaluated on
both contexts after each stage. Inputs, mappings, train/test examples, initial
weights, minibatch order, neurons, synapses, and update counts are paired by
seed.

## Fixed arms

- `static`: no learning;
- `bptt`: surrogate-gradient upper control;
- `eprop_broadcast`: eligibility traces with a fixed broadcast teaching
  signal;
- `dendritic_predictive_credit`: eligibility traces multiplied by the
  residual between a fixed apical teaching signal and each neuron's local error
  prediction;
- `dpc_shuffled_apical`: hidden-neuron identities shuffled after the apical
  teaching projection;
- `dpc_no_eligibility`: only instantaneous pre/post coincidence is retained;
- `dpc_shuffled_modulator`: sample identities in the output teaching signal
  shuffled before the apical projection.

Structural plasticity, STW/LTW separation, trainable delays, replay, and
astrocyte modulation are disabled. Gen-30 tests the local credit rule only.

## Frozen implementation

- 64 recurrent LIF neurons, 11 input channels, four fixed output classes;
- one fixed topology with 4,800 trainable input/recurrent synapses and a fixed
  normalized decoder;
- symmetric fixed decoder feedback for both e-prop and DPC so feedback
  alignment is not an uncontrolled variable;
- 24 steps, cue at step 0, context at step 4, distractors at steps 7-16, and
  query at step 23;
- 2,048 training and 1,024 held-out examples per context;
- batch size 256 and ten passes over Context A followed by ten over Context B;
- seeds 42-51, paired initial weights, examples, minibatch order, and update
  counts;
- checkpointing after every completed arm/seed transaction.

These quantities and all learning rates are serialized in the result JSON and
covered by its progress signature. Resuming with a changed configuration is an
error.

## Gates

Across seeds 42-51, DPC must:

1. reach at least 80% context-B accuracy after B learning;
2. retain at least 75% context-A accuracy with no more than a five-point drop;
3. finish within five points of e-prop on the mean A/B joint score;
4. exceed shuffled apical, no eligibility, and shuffled modulator controls by
   at least ten joint-accuracy points each;
5. maintain hidden spike activity between 1% and 30%;
6. satisfy the accuracy and retention gates on at least 8/10 seeds.

Passing authorizes a fixed-topology SSC transfer. Failure stops before real
data and requires a new local-credit theory. No topology, memory, hardware, or
energy claim follows from either outcome.
