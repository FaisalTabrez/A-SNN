# CUDA champion-topology eager hotpath throughput

Date: 2026-06-27

Run context:

- Device: `cuda`
- Topology preset: `champion`
- Adjacency JSON: `gen5_outputs/champion/champion_sparse_adjacency.json`
- Seeded active edges: `83`
- Edge pool capacity: `128`
- Active edge utilization: `64.84%`
- Steps: `240`
- `torch.compile`: not requested
- Tick mode: `tensor_hot_path_no_epoch_control`

Raw files:

- `throughput_results.json`
- `throughput_results.csv`
- `throughput_scaling.png`

## Results

| Population | Ticks/sec | Agent-steps/sec | Mean active synapses | CUDA max memory MB |
|---:|---:|---:|---:|---:|
| 1,000 | 491.532 | 491,531.557 | 83.000 | 8.522 |
| 10,000 | 493.944 | 4,939,441.277 | 83.000 | 86.594 |
| 50,000 | 112.804 | 5,640,190.208 | 83.000 | 425.916 |
| 100,000 | 56.873 | 5,687,259.652 | 83.000 | 853.889 |

Peak observed eager champion throughput: `5.69M` agent-steps/sec at `100k`
agents.

## Comparison against saturated eager hotpath

Saturated eager reference:

- `gen5/outputs/throughput_cuda_saturated_eager_hotpath_2026-06-27/throughput_results.json`
- Device: `cuda`
- Topology: synthetic saturated `86`-edge prior
- Edge pool capacity: `128`
- `torch.compile`: not requested
- Tick mode: `tensor_hot_path_no_epoch_control`

| Population | Champion eager agent-steps/sec | Saturated eager agent-steps/sec | Champion / saturated | Champion max MB | Saturated max MB | Memory ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 491,531.557 | 434,755.009 | 1.131 | 8.522 | 8.823 | 0.966 |
| 10,000 | 4,939,441.277 | 4,202,243.296 | 1.175 | 86.594 | 89.590 | 0.967 |
| 50,000 | 5,640,190.208 | 4,666,166.963 | 1.209 | 425.916 | 440.950 | 0.966 |
| 100,000 | 5,687,259.652 | 4,705,267.105 | 1.209 | 853.889 | 883.836 | 0.966 |

## Non-comparable compiled champion context

The previous compiled champion hotpath run reported the same display path,
`gen5_outputs/champion/champion_sparse_adjacency.json`, but seeded only `55`
active edges. This eager run seeded `83` active edges. That means the champion
file changed between runs or a different file was present at the same runtime
path.

Because the topology is not identical, the previous compiled champion result
must not be used as a clean compiled/eager speedup pair for this eager run.

Future benchmark outputs now include:

- `resolved_adjacency_json`
- `adjacency_sha256`

These fields make exact-topology comparisons auditable even when Colab runtime
folders are reused across runs.

## Interpretation

- The current/fresh `83`-edge champion eager hotpath reaches `5.69M`
  agent-steps/sec at `100k` agents.
- It is roughly `1.21x` faster than the synthetic saturated 86-edge eager
  hotpath at `100k`, with about `3.4%` lower CUDA max memory.
- This reinforces that exact topology matters. Even under the same fixed
  128-slot edge pool, source/target distribution and active-edge layout can
  change throughput.
- The compiled/eager champion comparison still needs to be rerun on a single
  fingerprinted adjacency file.

## Next benchmark actions

1. Rerun champion compiled hotpath using this same `83`-edge adjacency after
   the benchmark SHA-256 fields are available.
2. Rerun champion eager hotpath, if needed, and confirm the
   `adjacency_sha256` matches the compiled run.
3. Run a champion capacity sweep with the same adjacency:
   - `--max-edges 96`
   - `--max-edges 128`
   - optionally `--max-edges 160`
4. Run foraging 8-edge hotpath with and without `--compile`.
