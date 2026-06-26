# CUDA saturated-topology compiled hotpath throughput

Date: 2026-06-27

Run context:

- Device: `cuda`
- Topology preset: `saturated`
- Requested active edges: `86`
- Seeded active edges: `86`
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
| 1,000 | 2,196.607 | 2,196,606.636 | 86.000 | 8.848 |
| 10,000 | 2,209.794 | 22,097,942.371 | 86.000 | 53.016 |
| 50,000 | 769.127 | 38,456,327.383 | 86.000 | 246.852 |
| 100,000 | 391.457 | 39,145,695.626 | 86.000 | 488.195 |

Peak observed compiled hotpath saturated throughput: `39.15M`
agent-steps/sec at `100k` agents.

## Comparison against prior full-step saturated CUDA compile

Prior reference:

- `gen5/outputs/throughput_cuda_saturated_compile_2026-06-27/throughput_results.json`
- Device: `cuda`
- Topology: synthetic saturated `86`-edge prior
- `torch.compile`: active
- Tick mode: full `EvolvingHeadlessAMMCLoop.step()` with host epoch control
  and diagnostics payload

| Population | Full-step compiled agent-steps/sec | Hotpath compiled agent-steps/sec | Hotpath / full-step | Full-step max MB | Hotpath max MB | Memory ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 439,827.118 | 2,196,606.636 | 4.994 | 10.728 | 8.848 | 0.825 |
| 10,000 | 4,055,814.161 | 22,097,942.371 | 5.448 | 89.590 | 53.016 | 0.592 |
| 50,000 | 4,590,713.206 | 38,456,327.383 | 8.377 | 440.950 | 246.852 | 0.560 |
| 100,000 | 4,618,652.250 | 39,145,695.626 | 8.476 | 883.836 | 488.195 | 0.552 |

## Comparison against saturated eager full-step CUDA

Prior reference:

- `gen5/outputs/throughput_cuda_saturated_2026-06-27/throughput_results.json`
- Device: `cuda`
- Topology: synthetic saturated `86`-edge prior
- `torch.compile`: not requested
- Tick mode: full `EvolvingHeadlessAMMCLoop.step()`

| Population | Full-step eager agent-steps/sec | Hotpath compiled agent-steps/sec | Hotpath / full-step eager |
|---:|---:|---:|---:|
| 1,000 | 434,260.848 | 2,196,606.636 | 5.058 |
| 10,000 | 4,081,741.472 | 22,097,942.371 | 5.414 |
| 50,000 | 4,608,966.681 | 38,456,327.383 | 8.344 |
| 100,000 | 4,643,764.924 | 39,145,695.626 | 8.430 |

## Interpretation

- The recompile-safe tensor hotpath unlocked the expected compiler behavior.
  The previous saturated `--compile` run was not compute-limited by the sparse
  brain alone; it was dominated by host control, diagnostics payload assembly,
  and compiler guards around Python epoch state.
- With host epoch control removed, the 86-edge saturated topology reaches
  `39.15M` agent-steps/sec at `100k` agents on the CUDA runtime.
- CUDA max memory at `100k` falls from `883.84 MB` in the full-step compiled
  diagnostic run to `488.19 MB` in the compiled hotpath run. This strongly
  suggests telemetry/diagnostic tensors and unused return payloads were a major
  allocation source in the prior benchmark path.
- The hotpath result is now the best throughput evidence for the pure
  vectorized AMMC compute loop. The full-step benchmark remains useful for
  training-loop overhead measurement.

## Caveats

- This run measures the compiled tensor hotpath, not full epoch evolution,
  champion snapshotting, telemetry logging, or JSON/CSV output.
- The clean eager-vs-compiled hotpath comparison has now been run and archived:
  `gen5/outputs/throughput_cuda_saturated_eager_hotpath_2026-06-27/analysis.md`.
  At `100k`, compiled hotpath is `8.320x` faster than eager hotpath on the same
  saturated 86-edge workload.

## Follow-up implementation

After this run, `TensorEnvironment2D.step()` was extended with
`collect_telemetry=False`, and `benchmark_tick()` now uses that mode. This makes
the benchmark/inference hotpath skip nearest-object diagnostics and cloned
payload tensors explicitly instead of relying on `torch.compile` to remove
unused outputs.

Full training/evaluation calls keep the default `collect_telemetry=True`
behavior.

## Next benchmark actions

1. Run exact `champion_sparse_adjacency.json` hotpath with and without
   `--compile`.
2. Run `foraging` hotpath with and without `--compile`.
3. Repeat `foraging`, `saturated`, and `champion` presets on TPU/XLA once the
   Colab PyTorch/XLA runtime is ABI-clean.
