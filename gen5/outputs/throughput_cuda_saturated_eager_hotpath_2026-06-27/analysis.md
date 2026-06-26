# CUDA saturated-topology eager hotpath throughput

Date: 2026-06-27

Run context:

- Device: `cuda`
- Topology preset: `saturated`
- Requested active edges: `86`
- Seeded active edges: `86`
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
| 1,000 | 434.755 | 434,755.009 | 86.000 | 8.823 |
| 10,000 | 420.224 | 4,202,243.296 | 86.000 | 89.590 |
| 50,000 | 93.323 | 4,666,166.963 | 86.000 | 440.950 |
| 100,000 | 47.053 | 4,705,267.105 | 86.000 | 883.836 |

Peak observed eager hotpath saturated throughput: `4.71M` agent-steps/sec at
`100k` agents.

## Clean hotpath compiler comparison

Compiled hotpath reference:

- `gen5/outputs/throughput_cuda_saturated_compile_hotpath_2026-06-27/throughput_results.json`
- Device: `cuda`
- Topology: synthetic saturated `86`-edge prior
- `torch.compile`: active
- Tick mode: `tensor_hot_path_no_epoch_control`

| Population | Eager hotpath agent-steps/sec | Compiled hotpath agent-steps/sec | Compiled / eager | Eager max MB | Compiled max MB | Memory ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 434,755.009 | 2,196,606.636 | 5.053 | 8.823 | 8.848 | 1.003 |
| 10,000 | 4,202,243.296 | 22,097,942.371 | 5.259 | 89.590 | 53.016 | 0.592 |
| 50,000 | 4,666,166.963 | 38,456,327.383 | 8.242 | 440.950 | 246.852 | 0.560 |
| 100,000 | 4,705,267.105 | 39,145,695.626 | 8.320 | 883.836 | 488.195 | 0.552 |

## Comparison against previous full-step eager benchmark

Previous full-step eager reference:

- `gen5/outputs/throughput_cuda_saturated_2026-06-27/throughput_results.json`
- Device: `cuda`
- Topology: synthetic saturated `86`-edge prior
- `torch.compile`: not requested
- Tick mode: full `EvolvingHeadlessAMMCLoop.step()`

| Population | Full-step eager agent-steps/sec | Eager hotpath agent-steps/sec | Eager hotpath / full-step eager |
|---:|---:|---:|---:|
| 1,000 | 434,260.848 | 434,755.009 | 1.001 |
| 10,000 | 4,081,741.472 | 4,202,243.296 | 1.030 |
| 50,000 | 4,608,966.681 | 4,666,166.963 | 1.012 |
| 100,000 | 4,643,764.924 | 4,705,267.105 | 1.013 |

## Interpretation

- This is the missing apples-to-apples control for the compiled hotpath run.
- On the same `tensor_hot_path_no_epoch_control` workload, `torch.compile`
  delivers a `5.05x` to `8.32x` throughput improvement for the saturated
  86-edge topology.
- Eager hotpath is only `1.001x` to `1.030x` faster than the earlier full-step
  eager benchmark. Therefore, the massive compiled hotpath gain is not merely
  caused by skipping host epoch control or diagnostic telemetry; it reflects
  compiler fusion/optimization of the tensor workload.
- Compiled memory is materially lower at larger population sizes:
  `0.552x` eager max memory at `100k`. This suggests the compiler reduces
  intermediate allocation pressure in broadcast/scatter-heavy sections of the
  loop.
- The current best publishable CUDA throughput statement is:
  saturated 86-edge AMMC reaches `39.15M` agent-steps/sec at `100k` agents with
  `torch.compile`, versus `4.71M` eager on the same hotpath.

## Next benchmark actions

1. Run exact `champion_sparse_adjacency.json` hotpath with and without
   `--compile`.
2. Run `foraging` hotpath with and without `--compile` to compare edge-load
   sensitivity under the same benchmark mode.
3. Repeat `foraging`, `saturated`, and `champion` presets on TPU/XLA once the
   Colab PyTorch/XLA runtime is ABI-clean.
4. Add active-edge-normalized throughput summaries:
   agent-steps/sec, edge-steps/sec, and MB per 10k agents.
