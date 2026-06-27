# CUDA neuron-scaling evaluation

Date: 2026-06-27

Run context:

- Device: `cuda`
- Seeds: `42` through `51`
- Generations: `500`
- Population size: `10,000`
- Epoch steps: `120`
- Environment: `128` food objects, `128` toxin objects
- Sensor radius: `0.35`
- Friction: `0.985`
- Action gain: `0.05`
- Fitness threshold: `25.0`

Raw files:

- `neuron_scaling.json`
- `neuron_scaling_records.csv`
- `neuron_scaling_summary.csv`
- `neuron_scaling_summary.png`

## Results

| Neurons | Hidden decision nodes | Edge slots | Final mean best fitness | Std | Mean active synapses | Edge utilization | Fitness / active synapse | Threshold success | Mean generation to threshold |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 4 | 128 | 25.900 | 1.101 | 85.924 | 67.13% | 0.301 | 100% | 193.0 |
| 32 | 20 | 256 | 26.100 | 2.331 | 171.617 | 67.04% | 0.152 | 80% | 199.5 |
| 64 | 52 | 512 | 25.500 | 1.354 | 343.621 | 67.11% | 0.074 | 70% | 182.4 |

## Interpretation

- Larger neuron counts did not produce a meaningful fitness gain in this task.
  The 32-neuron condition improved final mean best fitness by only `+0.2`
  over the 16-neuron baseline, while using roughly `2.0x` active synapses.
- The 64-neuron condition underperformed the 16-neuron baseline by `-0.4`
  fitness while using roughly `4.0x` active synapses.
- Edge utilization stayed almost perfectly constant at about `67%` across all
  capacities. This suggests the current mutation/sprouting dynamics fill a
  fixed fraction of the available pool rather than discovering a compact
  task-dependent optimum.
- Fitness per active synapse dropped sharply as capacity increased:
  - `0.301` at 16 neurons,
  - `0.152` at 32 neurons,
  - `0.074` at 64 neurons.
- The threshold generation for 64 neurons is conditional on successful seeds
  only. Because only `70%` of 64-neuron seeds crossed threshold, the lower
  mean generation-to-threshold should not be interpreted as better overall
  sample efficiency.

## Decision

The current Gen-5 foraging task is not neuron-capacity limited. It is
structure-efficiency limited.

For the next phase, prioritize active-edge pressure and useful-routing
mechanisms before increasing neuron count further.

## Recommended next experiment

Run a sparse-efficiency ablation across the same `16/32/64` neuron scale points:

1. Add a fitness penalty per active edge.
2. Lower sprout probability for larger edge pools.
3. Increase pruning pressure for low-LTW edges.
4. Track final fitness, active synapses, utilization, and
   fitness-per-active-synapse.

Success criterion: a larger brain should only be considered better if it
improves final fitness without collapsing fitness-per-active-synapse.
