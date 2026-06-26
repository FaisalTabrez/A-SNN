# CUDA saturated-topology throughput with `torch.compile`

Date: 2026-06-27

Run context:

- Device: `cuda`
- Topology preset: `saturated`
- Requested active edges: `86`
- Seeded active edges: `86`
- Steps: `240`
- `torch.compile`: requested and active

Raw files:

- `throughput_results.json`
- `throughput_results.csv`
- `throughput_scaling.png`

## Results

| Population | Ticks/sec | Agent-steps/sec | Mean active synapses | CUDA max memory MB |
|---:|---:|---:|---:|---:|
| 1,000 | 439.827 | 439,827.118 | 86.000 | 10.728 |
| 10,000 | 405.581 | 4,055,814.161 | 86.000 | 89.590 |
| 50,000 | 91.814 | 4,590,713.206 | 86.000 | 440.950 |
| 100,000 | 46.187 | 4,618,652.250 | 86.000 | 883.836 |

Peak observed saturated compiled throughput: `4.62M` agent-steps/sec at
`100k` agents.

## Comparison against saturated eager CUDA run

Prior reference:

- `gen5/outputs/throughput_cuda_saturated_2026-06-27/throughput_results.json`
- Device: `cuda`
- Topology: synthetic saturated `86`-edge prior
- `torch.compile`: not requested

| Population | Eager agent-steps/sec | Compiled agent-steps/sec | Compiled / eager | Eager max MB | Compiled max MB | Memory ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 434,260.848 | 439,827.118 | 1.013 | 8.823 | 10.728 | 1.216 |
| 10,000 | 4,081,741.472 | 4,055,814.161 | 0.994 | 89.590 | 89.590 | 1.000 |
| 50,000 | 4,608,966.681 | 4,590,713.206 | 0.996 | 440.950 | 440.950 | 1.000 |
| 100,000 | 4,643,764.924 | 4,618,652.250 | 0.995 | 883.836 | 883.836 | 1.000 |

## Interpretation

- `torch.compile` did not materially improve the saturated CUDA benchmark in
  this run. At large population sizes it was effectively flat to slightly
  slower than eager execution.
- The run emitted a Torch Dynamo recompile-limit warning from
  `EvolvingHeadlessAMMCLoop.step()` because the full training step mutates the
  Python integer `_epoch_step_host`. Dynamo treats integer attributes on
  `nn.Module` as static guards, so each host-counter value can trigger a new
  specialization.
- This means the uploaded compiled results are useful as diagnostic evidence,
  but they are not a clean compiler benchmark for publication.

## Action taken

The benchmark path has been updated to use
`EvolvingHeadlessAMMCLoop.benchmark_tick()`, a control-free tensor hot path that
keeps environment sensing, recurrent sparse brain compute, motor decoding,
physics, and tensor epoch counting while skipping host epoch control,
evolution, champion snapshots, logger writes, and diagnostics dictionary
assembly.

Future benchmark JSON/CSV rows include:

- `tick_mode: tensor_hot_path_no_epoch_control`

This should prevent `torch.compile` from specializing on `_epoch_step_host` and
separate compiler-hot-path measurements from full evolutionary training
semantics.

## Next benchmark actions

1. Rerun saturated CUDA with `--compile` after this patch and confirm the Dynamo
   recompile warning disappears.
2. Rerun saturated CUDA without `--compile` on the same `benchmark_tick()` path
   so eager-vs-compiled comparisons share the same measured workload.
3. Run the exact `champion_sparse_adjacency.json` topology using the same
   benchmark path.
4. Repeat `foraging`, `saturated`, and `champion` presets on TPU/XLA once the
   Colab PyTorch/XLA runtime is ABI-clean.
