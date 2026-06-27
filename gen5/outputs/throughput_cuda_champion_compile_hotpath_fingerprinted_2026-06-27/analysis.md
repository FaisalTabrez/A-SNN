# CUDA champion-topology compiled hotpath throughput, fingerprinted

Date: 2026-06-27

Run context:

- Device: `cuda`
- Topology preset: `champion`
- Adjacency JSON: `gen5_outputs/champion/champion_sparse_adjacency.json`
- Resolved adjacency JSON: `gen5_outputs/champion/champion_sparse_adjacency.json`
- Adjacency SHA-256:
  `de4cdb8f715389f8206e025435856cd2b4a55d8a7688b28b9cc3eabd5f3d904a`
- Seeded active edges: `86`
- Edge pool capacity: `128`
- Active edge utilization: `67.19%`
- Steps: `240`
- `torch.compile`: requested and active
- Tick mode: `tensor_hot_path_no_epoch_control`

Raw files:

- `throughput_results.json`
- `throughput_results.csv`
- `throughput_scaling.png`

## Results

| Population | Ticks/sec | Agent-steps/sec | Mean active synapses | CUDA max memory MB |
|---:|---:|---:|---:|---:|
| 1,000 | 1,663.407 | 1,663,407.163 | 86.000 | 8.848 |
| 10,000 | 2,128.179 | 21,281,787.331 | 86.000 | 53.016 |
| 50,000 | 691.317 | 34,565,832.932 | 86.000 | 246.852 |
| 100,000 | 353.486 | 35,348,599.664 | 86.000 | 488.195 |

Peak observed fingerprinted compiled champion throughput: `35.35M`
agent-steps/sec at `100k` agents.

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
| 1,000 | 1,663,407.163 | 2,196,606.636 | 0.757 | 8.848 | 8.848 | 1.000 |
| 10,000 | 21,281,787.331 | 22,097,942.371 | 0.963 | 53.016 | 53.016 | 1.000 |
| 50,000 | 34,565,832.932 | 38,456,327.383 | 0.899 | 246.852 | 246.852 | 1.000 |
| 100,000 | 35,348,599.664 | 39,145,695.626 | 0.903 | 488.195 | 488.195 | 1.000 |

## Interpretation

- This is the first champion compiled hotpath artifact with a recorded topology
  fingerprint.
- The fingerprinted 86-edge champion reaches `35.35M` agent-steps/sec at
  `100k` agents, or `90.3%` of the synthetic saturated 86-edge compiled
  hotpath.
- CUDA max memory is identical to the saturated 86-edge compiled run at every
  population size. This reinforces that fixed edge-pool capacity, not active
  edge identity alone, dominates memory footprint in the current backend.
- Throughput is lower than the synthetic saturated topology despite matching
  active edge count. Exact source/target distribution and scatter behavior
  matter.
- This result supersedes the earlier non-fingerprinted `55`-edge compiled
  champion run for publication-grade champion throughput evidence.

## Caveats

- The available champion eager run seeded `83` active edges and predates
  `adjacency_sha256` output, so it is not a valid eager pair for this
  fingerprinted 86-edge compiled run.
- A clean champion compiler speedup still requires rerunning eager hotpath
  against the same adjacency SHA-256:
  `de4cdb8f715389f8206e025435856cd2b4a55d8a7688b28b9cc3eabd5f3d904a`.

## Next benchmark actions

1. Rerun champion eager hotpath using the same adjacency and confirm the same
   `adjacency_sha256`.
2. Run champion capacity sweeps with this same SHA:
   - `--max-edges 96`
   - `--max-edges 128`
   - optionally `--max-edges 160`
3. Run foraging 8-edge hotpath with and without `--compile`.
