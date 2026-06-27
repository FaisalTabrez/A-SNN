# CUDA champion-topology compiled hotpath throughput

Date: 2026-06-27

Run context:

- Device: `cuda`
- Topology preset: `champion`
- Adjacency JSON: `gen5_outputs/champion/champion_sparse_adjacency.json`
- Seeded active edges: `55`
- Edge pool capacity during run: `128`
- Active edge utilization: `42.97%`
- Steps: `240`
- `torch.compile`: requested and active
- Tick mode: `tensor_hot_path_no_epoch_control`

Raw files:

- `throughput_results.json`
- `throughput_results.csv`
- `throughput_scaling.png`

Supersession note:

- This run predates `adjacency_sha256` output and seeded `55` active edges.
- The current publication-grade compiled champion artifact is:
  `gen5/outputs/throughput_cuda_champion_compile_hotpath_fingerprinted_2026-06-27/analysis.md`.
  It records `86` active edges and adjacency SHA-256
  `de4cdb8f715389f8206e025435856cd2b4a55d8a7688b28b9cc3eabd5f3d904a`.

## Results

| Population | Ticks/sec | Agent-steps/sec | Mean active synapses | CUDA max memory MB |
|---:|---:|---:|---:|---:|
| 1,000 | 1,400.453 | 1,400,453.327 | 55.000 | 4.879 |
| 10,000 | 778.297 | 7,782,966.959 | 55.000 | 49.200 |
| 50,000 | 604.552 | 30,227,606.471 | 55.000 | 243.596 |
| 100,000 | 372.509 | 37,250,874.070 | 55.000 | 488.195 |

Peak observed compiled champion throughput: `37.25M` agent-steps/sec at
`100k` agents.

## Comparison against saturated compiled hotpath

Saturated reference:

- `gen5/outputs/throughput_cuda_saturated_compile_hotpath_2026-06-27/throughput_results.json`
- Device: `cuda`
- Topology: synthetic saturated `86`-edge prior
- Edge pool capacity: `128`
- `torch.compile`: active
- Tick mode: `tensor_hot_path_no_epoch_control`

| Population | Champion agent-steps/sec | Saturated agent-steps/sec | Champion / saturated | Champion max MB | Saturated max MB | Memory ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 1,400,453.327 | 2,196,606.636 | 0.638 | 4.879 | 8.848 | 0.551 |
| 10,000 | 7,782,966.959 | 22,097,942.371 | 0.352 | 49.200 | 53.016 | 0.928 |
| 50,000 | 30,227,606.471 | 38,456,327.383 | 0.786 | 243.596 | 246.852 | 0.987 |
| 100,000 | 37,250,874.070 | 39,145,695.626 | 0.952 | 488.195 | 488.195 | 1.000 |

## Edge-work perspective

| Population | Champion edge-steps/sec | Saturated edge-steps/sec | Champion / saturated edge-steps |
|---:|---:|---:|---:|
| 1,000 | 77,024,932.971 | 188,908,170.722 | 0.408 |
| 10,000 | 428,063,182.762 | 1,900,423,044.643 | 0.225 |
| 50,000 | 1,662,518,355.902 | 3,307,244,154.908 | 0.503 |
| 100,000 | 2,048,798,073.852 | 3,366,529,823.808 | 0.609 |

`edge-steps/sec` is computed as `agent_steps_per_second * seeded_active_edges`.
It is useful as a biological-work metric, but it is not yet a perfect hardware
work metric because the current `TensorEvolver` uses a fixed-capacity edge pool.

## Interpretation

- The current champion topology is computationally viable at scale:
  `37.25M` agent-steps/sec at `100k` agents.
- This champion export has `55` active edges, not the older archived
  `88`-synapse champion. Treat it as the current/fresh champion package from
  `gen5_outputs/champion`.
- Despite using only `55` active edges versus saturated's `86`, the champion is
  not proportionally faster. At `100k`, it reaches `95.2%` of saturated
  agent-throughput.
- This is expected under the current fixed-capacity tensor design:
  `TensorEvolver` stores each organism as `[population, max_edges]`, and the
  benchmark used `max_edges=128`. Inactive slots are masked out biologically,
  but the tensor path still gathers/multiplies/scatters over the fixed edge
  pool capacity.
- The result proves the fixed-capacity XLA/CUDA-compatible design scales, but
  it also shows the remaining gap between biological sparsity and true
  hardware sparsity. The future dynamic sparse backend or compacted edge-bucket
  mode should make active edge count more directly translate into lower compute.

## Next benchmark actions

1. Run champion eager hotpath without `--compile` for a clean exact-topology
   compiler comparison.
2. Run foraging 8-edge hotpath with and without `--compile`.
3. Add a max-edge-capacity sweep for champion:
   - `--max-edges 64`
   - `--max-edges 96`
   - `--max-edges 128`
   to quantify fixed-pool overhead.
4. Add active-edge-normalized and capacity-normalized metrics to future
   published summaries.
