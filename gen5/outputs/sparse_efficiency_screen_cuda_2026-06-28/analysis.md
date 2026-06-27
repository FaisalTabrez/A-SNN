# CUDA sparse-efficiency screen

Date: 2026-06-28

Run context:

- Device: `cuda`
- Groups: `baseline_capacity_fill`, `protected_sparse_core`
- Seeds: `42`, `43`, `44`
- Generations: `200`
- Population size: `10,000`
- Epoch steps: `120`
- Scale points:
  - `16` neurons / `128` edge slots
  - `32` neurons / `256` edge slots
  - `64` neurons / `512` edge slots
- Checkpointed trials: `18 / 18`

Raw files:

- `sparse_efficiency.json`
- `sparse_efficiency_records.csv`
- `sparse_efficiency_summary.csv`
- `sparse_efficiency_progress.json`
- `sparse_efficiency_summary.png`
- `colab_log.txt`

## Results

| Group | Neurons | Final mean best fitness | Active synapses | Utilization | Fitness / active synapse | Hidden-edge fraction | Direct sensor-motor fraction | Threshold success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_capacity_fill | 16 | 25.667 | 81.703 | 63.83% | 0.233 | 43.58% | 16.49% | 100% |
| baseline_capacity_fill | 32 | 25.000 | 163.450 | 63.85% | 0.108 | 85.11% | 4.93% | 66.67% |
| baseline_capacity_fill | 64 | 24.000 | 326.658 | 63.80% | 0.057 | 95.89% | 1.62% | 33.33% |
| protected_sparse_core | 16 | 24.333 | 41.807 | 32.66% | 0.462 | 36.11% | 30.00% | 66.67% |
| protected_sparse_core | 32 | 18.667 | 46.431 | 18.14% | 0.344 | 71.31% | 20.16% | 0% |
| protected_sparse_core | 64 | 15.000 | 44.000 | 8.59% | 0.220 | 78.86% | 19.11% | 0% |

## Interpretation

- The baseline repeated the neuron-scaling finding: active synapses scaled with
  edge capacity, while raw fitness did not improve.
- `protected_sparse_core` dramatically reduced active synapses:
  - about `49%` fewer active synapses at `16` neurons,
  - about `72%` fewer at `32` neurons,
  - about `87%` fewer at `64` neurons.
- At `16` neurons, protected sparse core was close to baseline raw fitness
  while nearly doubling fitness per active synapse:
  `0.462` versus `0.233`.
- At `32` and `64` neurons, protected sparse core underfit badly. It kept the
  active synapse count near `42-46` regardless of available decision capacity,
  which appears too restrictive for larger topologies.
- Hidden-edge fraction rose with neuron count in both groups. In the baseline,
  this mostly reflects topology filling: larger brains become dominated by
  hidden-touching edges without gaining fitness. In the protected group,
  hidden-edge use is still high, but total active edge count is too low for the
  larger brains to preserve behavior.

## Decision

Sparse pressure is useful, but the current protected sparse core is too strong
for larger neuron counts.

The project should not abandon sparse efficiency. Instead, tune it:

1. Test component ablations individually:
   - `active_edge_penalty`
   - `low_ltw_pruning`
   - `scheduled_sprouting`
2. Add a gentler protected-core variant:
   - lower active-edge penalty,
   - lower weak-edge prune probability,
   - capacity-aware minimum active edge target.
3. Treat `16` neurons as the immediate sparse-efficiency winner for the simple
   foraging task.
4. Do not run a full `10` seed x `500` generation sparse matrix until a gentler
   pressure schedule beats this screen.

## Recommended next run

Run the three individual mechanisms before another combined protected-core run:

```python
!python gen5/examples/sprint13_sparse_efficiency_ablation.py \
  --device cuda \
  --groups active_edge_penalty low_ltw_pruning scheduled_sprouting \
  --seeds 42 43 44 \
  --generations 200 \
  --population-size 10000 \
  --epoch-steps 120 \
  --neuron-counts 16 32 64 \
  --max-edges 128 256 512 \
  --output-dir gen5_outputs/sparse_efficiency_components_cuda
```
