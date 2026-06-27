# CUDA champion-topology eager hotpath throughput, fingerprinted

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
- `torch.compile`: not requested
- Tick mode: `tensor_hot_path_no_epoch_control`

Raw files:

- `throughput_results.json`
- `throughput_results.csv`
- `throughput_scaling.png`

## Results

| Population | Ticks/sec | Agent-steps/sec | Mean active synapses | CUDA max memory MB |
|---:|---:|---:|---:|---:|
| 1,000 | 339.219 | 339,219.142 | 86.000 | 8.522 |
| 10,000 | 367.500 | 3,675,002.299 | 86.000 | 86.594 |
| 50,000 | 111.565 | 5,578,244.953 | 86.000 | 425.916 |
| 100,000 | 56.013 | 5,601,273.103 | 86.000 | 853.889 |

Peak observed fingerprinted eager champion throughput: `5.60M`
agent-steps/sec at `100k` agents.

## Clean champion compiler comparison

Compiled champion reference:

- `gen5/outputs/throughput_cuda_champion_compile_hotpath_fingerprinted_2026-06-27/throughput_results.json`
- Same adjacency SHA-256:
  `de4cdb8f715389f8206e025435856cd2b4a55d8a7688b28b9cc3eabd5f3d904a`
- Device: `cuda`
- `torch.compile`: active
- Tick mode: `tensor_hot_path_no_epoch_control`

| Population | Eager agent-steps/sec | Compiled agent-steps/sec | Compiled / eager | Eager max MB | Compiled max MB | Memory ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 339,219.142 | 1,663,407.163 | 4.904 | 8.522 | 8.848 | 1.038 |
| 10,000 | 3,675,002.299 | 21,281,787.331 | 5.791 | 86.594 | 53.016 | 0.612 |
| 50,000 | 5,578,244.953 | 34,565,832.932 | 6.197 | 425.916 | 246.852 | 0.580 |
| 100,000 | 5,601,273.103 | 35,348,599.664 | 6.311 | 853.889 | 488.195 | 0.572 |

## Comparison against saturated eager hotpath

Saturated eager reference:

- `gen5/outputs/throughput_cuda_saturated_eager_hotpath_2026-06-27/throughput_results.json`
- Device: `cuda`
- Topology: synthetic saturated `86`-edge prior
- Edge pool capacity: `128`
- `torch.compile`: not requested
- Tick mode: `tensor_hot_path_no_epoch_control`

| Population | Champion eager agent-steps/sec | Saturated eager agent-steps/sec | Champion / saturated eager |
|---:|---:|---:|---:|
| 1,000 | 339,219.142 | 434,755.009 | 0.780 |
| 10,000 | 3,675,002.299 | 4,202,243.296 | 0.875 |
| 50,000 | 5,578,244.953 | 4,666,166.963 | 1.195 |
| 100,000 | 5,601,273.103 | 4,705,267.105 | 1.190 |

## Interpretation

- This is the first clean champion eager-vs-compiled comparison with a matching
  topology fingerprint.
- At `100k` agents, `torch.compile` gives the champion a `6.31x` speedup:
  `35.35M` compiled agent-steps/sec versus `5.60M` eager.
- At large population sizes, compiled execution also reduces peak CUDA memory:
  `488.19 MB` compiled versus `853.89 MB` eager at `100k`.
- The champion eager run is slower than saturated eager at small populations
  but faster at `50k` and `100k`, showing that exact edge layout can interact
  with population scale.
- This pair is now the publication-grade champion compiler evidence.

## Next benchmark actions

1. Run foraging 8-edge hotpath with and without `--compile`.
2. Run champion capacity sweeps with this same SHA:
   - `--max-edges 96`
   - `--max-edges 128`
   - optionally `--max-edges 160`
3. Report both active-edge-normalized and capacity-normalized throughput.
