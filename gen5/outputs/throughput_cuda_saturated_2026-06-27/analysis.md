# CUDA saturated-topology throughput analysis

Date: 2026-06-27

Run context:

- Device: `cuda`
- Topology preset: `saturated`
- Requested active edges: `86`
- Seeded active edges: `86`
- Steps: `240`
- `torch.compile`: not requested

Raw files:

- `throughput_results.json`
- `throughput_results.csv`
- `throughput_scaling.png`

## Results

| Population | Ticks/sec | Agent-steps/sec | Mean active synapses | CUDA max memory MB |
|---:|---:|---:|---:|---:|
| 1,000 | 434.261 | 434,260.848 | 86.000 | 8.823 |
| 10,000 | 408.174 | 4,081,741.472 | 86.000 | 89.590 |
| 50,000 | 92.179 | 4,608,966.681 | 86.000 | 440.950 |
| 100,000 | 46.438 | 4,643,764.924 | 86.000 | 883.836 |

Peak observed saturated throughput: `4.64M` agent-steps/sec at `100k` agents.

## Comparison against prior 8-edge CUDA benchmark

Prior reference:

- `gen5/outputs/phase11_remaining_outputs_2026-06-26/gen5_outputs/throughput/throughput_results.json`
- Device: `cuda`
- Topology: original foraging prior
- Mean active synapses: `8.0`
- `torch.compile`: active

| Population | 8-edge agent-steps/sec | 86-edge agent-steps/sec | 86-edge / 8-edge throughput | 8-edge max MB | 86-edge max MB | Memory ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 599,599.824 | 434,260.848 | 0.724 | 8.891 | 8.823 | 0.992 |
| 10,000 | 5,950,716.818 | 4,081,741.472 | 0.686 | 53.471 | 89.590 | 1.675 |
| 50,000 | 22,475,376.876 | 4,608,966.681 | 0.205 | 249.138 | 440.950 | 1.770 |
| 100,000 | 29,288,333.833 | 4,643,764.924 | 0.159 | 486.647 | 883.836 | 1.816 |

## Interpretation

- The saturated 86-edge topology remains viable at `100k` agents on CUDA:
  throughput is still roughly `4.64M` agent-steps/sec.
- The saturated topology is much more expensive than the 8-edge prior at large
  population sizes. At `100k`, saturated throughput is about `15.9%` of the
  prior 8-edge throughput.
- Memory remains under `1GB` at `100k`, but grows to about `1.82x` the 8-edge
  benchmark at the same population.
- The throughput curve plateaus from `50k` to `100k`, suggesting the saturated
  path becomes memory/scatter-bandwidth bound.
- Caveat: the prior 8-edge run used `torch.compile`, while this saturated run
  did not. A fairer comparison should rerun saturated with `--compile` on CUDA
  and also run both topology presets on TPU/XLA once the XLA runtime is clean.

## Compile follow-up

A saturated CUDA rerun with `--compile` was uploaded on 2026-06-27:

- `gen5/outputs/throughput_cuda_saturated_compile_2026-06-27/throughput_results.json`
- Peak: `4.62M` agent-steps/sec at `100k`
- Compiled / eager throughput at `100k`: `0.995x`

That run emitted a Torch Dynamo recompile-limit warning from the full
`EvolvingHeadlessAMMCLoop.step()` path because `_epoch_step_host` is a mutable
Python integer on an `nn.Module`. The benchmark code has since been updated to
use a control-free `benchmark_tick()` tensor hot path and to record
`tick_mode: tensor_hot_path_no_epoch_control` in future outputs.

## Next benchmark actions

1. Rerun saturated CUDA eager and compiled on the new `benchmark_tick()` path.
2. Run exact `champion_sparse_adjacency.json` topology, not only synthetic
   saturated topology.
3. Run `foraging`, `saturated`, and `champion` presets on TPU/XLA after the
   PyTorch/XLA environment is fixed.
4. Add active-edge-normalized throughput metrics to future benchmark summaries.
