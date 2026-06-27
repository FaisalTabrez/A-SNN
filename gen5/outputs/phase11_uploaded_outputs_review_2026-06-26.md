# Phase 11 uploaded outputs review

Date: 2026-06-26  
Reviewer: Codex  
Scope: `1st run` output bundle, uploaded plot images, and raw Sprint 11 files
found in `C:\Users\FAISAL TABREZ\Downloads`.

## Artifact inventory

### Raw files available in workspace

Folder: `1st run/`

- `evolution_telemetry.json`
- `evolution_telemetry.csv`
- `evolution_telemetry.png`
- `champion_connectome.json`
- `champion_sparse_adjacency.json`
- `colab_weights.json`

Folder: `gen5/outputs/phase11_colab_2026-06-26/`

- `multi_seed_trials.json`
- `multi_seed_trials.csv`
- `multi_seed_aggregate.csv`
- `multi_seed_best_fitness_mean_std.png`
- `plasticity_ablation.json`
- `plasticity_ablation_records.csv`
- `plasticity_ablation_summary.csv`
- `plasticity_ablation_best_fitness.png`

### Uploaded images available as attachments

- `image-1.png`: AMMC Gen-5 Multi-Seed Convergence
- `image-2.png`: AMMC Gen-5 Plasticity Ablation
- `image-3.png`: AMMC Gen-5 Evolution Telemetry

### Raw Sprint 11 files still not found

The following expected files were not present in the workspace or attachment
folder at review time:

- `throughput_results.json`
- `throughput_results.csv`
- `baseline_comparison.json`
- `baseline_comparison.csv`

This means exact statistical claims for convergence and plasticity ablation are
now supported by raw data, but hardware throughput and external baseline claims
remain blocked until their raw JSON/CSV outputs are uploaded.

## Validated champion-run evidence

The raw `1st run` champion package validates cleanly.

| Metric | Value |
|---|---:|
| Telemetry records | 500 |
| First generation | 1 |
| Last generation | 500 |
| Best max fitness | 24.0 |
| Best generations | 236, 450 |
| First 25 generations mean max fitness | 15.48 |
| First 100 generations mean max fitness | 17.77 |
| Final 100 generations mean max fitness | 18.42 |
| Overall mean max fitness | 18.408 |
| Overall std max fitness | 1.834 |
| Final generation max fitness | 19.0 |
| Final generation mean population fitness | -0.0151 |
| First generation mean active synapses | 9.156 |
| Final mean active synapses | 86.033 |
| Max mean active synapses | 86.321 |
| First generation where mean active synapses >= 80 | 175 |
| First generation where mean active synapses >= 85 | 287 |
| Champion fitness in adjacency export | 24.0 |
| Champion neurons | 16 |
| Champion active edges | 88 |
| Connectome synapses | 88 |
| Sparse adjacency rows | 88 |
| Colab weight sparse edges | 88 |

Integrity checks:

- Sparse adjacency row count matches browser connectome synapse count.
- Colab weight sparse edge count matches sparse adjacency row count.
- Champion bundle is internally consistent and suitable for browser replay.

## Verified multi-seed statistical convergence

Raw files:

- `gen5/outputs/phase11_colab_2026-06-26/multi_seed_trials.json`
- `gen5/outputs/phase11_colab_2026-06-26/multi_seed_trials.csv`
- `gen5/outputs/phase11_colab_2026-06-26/multi_seed_aggregate.csv`

Run shape:

- Seeds: `42` through `51`
- Trial records: `5,000`
- Aggregate generations: `500`

Final generation aggregate:

| Metric | Value |
|---|---:|
| Generation | 500 |
| Mean all-time best fitness | 26.0 |
| Std all-time best fitness | 0.667 |
| Min all-time best fitness | 25.0 |
| Max all-time best fitness | 27.0 |
| Mean population fitness | 0.0367 |
| Mean active synapses | 85.888 |

Final all-time best fitness by seed:

| Seed | Final all-time best fitness |
|---:|---:|
| 42 | 26.0 |
| 43 | 26.0 |
| 44 | 27.0 |
| 45 | 25.0 |
| 46 | 26.0 |
| 47 | 26.0 |
| 48 | 26.0 |
| 49 | 25.0 |
| 50 | 27.0 |
| 51 | 26.0 |

Mean-best threshold crossing:

| Mean best fitness threshold | First generation reached |
|---:|---:|
| 15 | 8 |
| 20 | 20 |
| 22 | 33 |
| 24 | 91 |
| 25 | 178 |
| 26 | 446 |

Interpretation:

- The genetic loop converged reliably across all 10 seeds.
- No seed collapsed or remained trapped in low-fitness behavior.
- Final spread is tight: all seeds ended between `25` and `27`.
- Topology again saturates near the upper edge-pool range, reinforcing the need
  for active-edge pressure in future runs.

## Verified plasticity ablation

Raw files:

- `gen5/outputs/phase11_colab_2026-06-26/plasticity_ablation.json`
- `gen5/outputs/phase11_colab_2026-06-26/plasticity_ablation_records.csv`
- `gen5/outputs/phase11_colab_2026-06-26/plasticity_ablation_summary.csv`

Final summary:

| Group | Seeds | Final mean best fitness | Std | Final mean active synapses |
|---|---:|---:|---:|---:|
| `static_snn` | 10 | 13.6 | 0.843 | 8.000 |
| `full_plasticity_infant` | 10 | 25.9 | 0.994 | 85.951 |
| `gated_plasticity_adult` | 10 | 24.6 | 1.075 | 66.995 |

Final all-time best fitness by seed:

| Group | Seed-level final best fitness |
|---|---|
| `static_snn` | 13, 14, 13, 12, 14, 15, 14, 14, 13, 14 |
| `full_plasticity_infant` | 27, 28, 25, 25, 25, 26, 26, 25, 26, 26 |
| `gated_plasticity_adult` | 24, 23, 24, 24, 25, 25, 24, 25, 27, 25 |

Threshold success:

| Group | Seeds >=20 | Seeds >=24 | Seeds >=25 | Mean generation to >=24 | Mean generation to >=25 |
|---|---:|---:|---:|---:|---:|
| `static_snn` | 0/10 | 0/10 | 0/10 | n/a | n/a |
| `full_plasticity_infant` | 10/10 | 10/10 | 10/10 | 90.4 | 176.9 |
| `gated_plasticity_adult` | 10/10 | 9/10 | 5/10 | 170.3 | 281.0 |

Parameter-efficiency note:

| Group | Fitness / mean active synapse |
|---|---:|
| `full_plasticity_infant` | 0.301 |
| `gated_plasticity_adult` | 0.367 |

Interpretation:

- Structural plasticity clearly improves adaptation under the perturbed
  food/toxin sensor condition.
- Static topology fails to cross meaningful high-fitness thresholds.
- Full/aggressive plasticity reaches the highest final fitness and fastest
  threshold crossing.
- Gated/adult plasticity sacrifices about `1.3` final fitness points compared
  with full plasticity, but uses about `22%` fewer active synapses and has
  better fitness per active synapse.
- The current ablation supports "plasticity beats static" strongly.
- It does not prove "gated adult plasticity is best" on raw fitness alone.

Follow-up:

The initial ablation measured perturbed-task adaptation but did not fully
measure catastrophic forgetting. The uploaded retention bundle now closes that
gap with an original -> perturbed -> original protocol.

## Verified retention ablation

Raw files:

- `gen5/outputs/phase11_remaining_outputs_2026-06-26/gen5_outputs/retention_ablation/retention_ablation.json`
- `gen5/outputs/phase11_remaining_outputs_2026-06-26/gen5_outputs/retention_ablation/retention_ablation_records.csv`
- `gen5/outputs/phase11_remaining_outputs_2026-06-26/gen5_outputs/retention_ablation/retention_ablation_summary.csv`

Records: `15,000`

Summary:

| Group | Seeds | Original final epoch best | Perturbation peak best | Recovery final epoch best | Recovery retention ratio | Forgetting delta | Final mean active synapses |
|---|---:|---:|---:|---:|---:|---:|---:|
| `static_snn` | 10 | 8.3 | 13.3 | 8.7 | 1.048 | -0.400 | 8.000 |
| `full_plasticity_infant` | 10 | 19.6 | 25.3 | 18.9 | 0.964 | 0.700 | 85.904 |
| `gated_plasticity_adult` | 10 | 18.4 | 24.2 | 17.0 | 0.924 | 1.400 | 67.544 |

Interpretation:

- Both plastic groups adapted far beyond static during the perturbation phase.
- Full plasticity achieved the strongest raw retention in this run:
  `18.9` recovery final epoch best and `0.964` retention ratio.
- Gated/adult plasticity remained more compact but did not preserve more
  original-task fitness under the current gate settings.
- Static SNN has a superficially high retention ratio because its original
  fitness was low; it did not learn the perturbed task to a competitive level.
- The adult/gated thesis is therefore not validated yet. The mechanism is
  promising for sparsity, but the current gate is too restrictive or is gating
  the wrong plasticity events.

## Verified throughput benchmark

Raw files:

- `gen5/outputs/phase11_remaining_outputs_2026-06-26/gen5_outputs/throughput/throughput_results.json`
- `gen5/outputs/phase11_remaining_outputs_2026-06-26/gen5_outputs/throughput/throughput_results.csv`

Run context:

- Device: `cuda`
- Dtype: `torch.float32`
- `torch.compile`: requested and active for all runs
- Steps per population size: `240`
- Mean active synapses during throughput benchmark: `8.0`

Results:

| Population | Ticks/sec | Agent-steps/sec | CUDA max memory MB |
|---:|---:|---:|---:|
| 1,000 | 599.600 | 599,599.824 | 8.891 |
| 10,000 | 595.072 | 5,950,716.818 | 53.471 |
| 50,000 | 449.508 | 22,475,376.876 | 249.138 |
| 100,000 | 292.883 | 29,288,333.833 | 486.647 |

Interpretation:

- Gen-5 ran at nearly `29.3M` agent-steps/sec at `100k` agents on CUDA.
- Memory growth is close to linear with population size and remains below
  `0.5GB` at `100k` agents for the benchmark configuration.
- Tick rate drops as population increases, but aggregate agent-throughput keeps
  rising, which supports the vectorized swarm direction.
- Caveat resolved in follow-up: this benchmark used `8` mean active synapses,
  not the saturated champion topology near `86` active synapses. The
  `2026-06-27` saturated CUDA follow-up is summarized below.

## Verified saturated-topology CUDA throughput follow-up

Raw files:

- `gen5/outputs/throughput_cuda_saturated_2026-06-27/throughput_results.json`
- `gen5/outputs/throughput_cuda_saturated_2026-06-27/throughput_results.csv`
- `gen5/outputs/throughput_cuda_saturated_2026-06-27/throughput_scaling.png`

Run context:

- Device: `cuda`
- Topology preset: `saturated`
- Active edges: `86`
- `torch.compile`: not requested

Results:

| Population | Ticks/sec | Agent-steps/sec | CUDA max memory MB |
|---:|---:|---:|---:|
| 1,000 | 434.261 | 434,260.848 | 8.823 |
| 10,000 | 408.174 | 4,081,741.472 | 89.590 |
| 50,000 | 92.179 | 4,608,966.681 | 440.950 |
| 100,000 | 46.438 | 4,643,764.924 | 883.836 |

Comparison to prior 8-edge CUDA benchmark:

| Population | Saturated / 8-edge throughput | Saturated / 8-edge max memory |
|---:|---:|---:|
| 1,000 | 0.724 | 0.992 |
| 10,000 | 0.686 | 1.675 |
| 50,000 | 0.205 | 1.770 |
| 100,000 | 0.159 | 1.816 |

Interpretation:

- Saturated topology remains computationally viable at `100k` agents:
  `4.64M` agent-steps/sec and under `1GB` CUDA max memory.
- Active-edge load matters enormously. At `100k`, the `86`-edge saturated run
  achieves about `15.9%` of the original `8`-edge throughput.
- The saturated curve plateaus from `50k` to `100k`, suggesting memory/scatter
  bandwidth pressure.
- Caveat: the prior 8-edge run used `torch.compile`; this saturated run did
  not. A fair follow-up should rerun saturated CUDA with `--compile`.

## Verified saturated-topology CUDA compile follow-up

Raw files:

- `gen5/outputs/throughput_cuda_saturated_compile_2026-06-27/throughput_results.json`
- `gen5/outputs/throughput_cuda_saturated_compile_2026-06-27/throughput_results.csv`
- `gen5/outputs/throughput_cuda_saturated_compile_2026-06-27/throughput_scaling.png`

Run context:

- Device: `cuda`
- Topology preset: `saturated`
- Active edges: `86`
- `torch.compile`: requested and active

Results:

| Population | Ticks/sec | Agent-steps/sec | CUDA max memory MB |
|---:|---:|---:|---:|
| 1,000 | 439.827 | 439,827.118 | 10.728 |
| 10,000 | 405.581 | 4,055,814.161 | 89.590 |
| 50,000 | 91.814 | 4,590,713.206 | 440.950 |
| 100,000 | 46.187 | 4,618,652.250 | 883.836 |

Comparison to saturated eager CUDA run:

| Population | Compiled / eager throughput | Compiled / eager max memory |
|---:|---:|---:|
| 1,000 | 1.013 | 1.216 |
| 10,000 | 0.994 | 1.000 |
| 50,000 | 0.996 | 1.000 |
| 100,000 | 0.995 | 1.000 |

Interpretation:

- `torch.compile` did not produce a useful speedup for this saturated run; it
  was flat to slightly slower at larger population sizes.
- The run emitted a Torch Dynamo recompile-limit warning from
  `EvolvingHeadlessAMMCLoop.step()` because the full training step mutates the
  Python-side `_epoch_step_host` counter.
- The benchmark code now targets a separate `benchmark_tick()` tensor hot path
  and records `tick_mode: tensor_hot_path_no_epoch_control` for future runs.
  This keeps full evolutionary training semantics intact while preventing
  compiler measurements from specializing on host epoch control.
- These uploaded compiled results should be treated as diagnostic evidence, not
  the final compiler benchmark.

## Verified saturated-topology CUDA compiled hotpath follow-up

Raw files:

- `gen5/outputs/throughput_cuda_saturated_compile_hotpath_2026-06-27/throughput_results.json`
- `gen5/outputs/throughput_cuda_saturated_compile_hotpath_2026-06-27/throughput_results.csv`
- `gen5/outputs/throughput_cuda_saturated_compile_hotpath_2026-06-27/throughput_scaling.png`

Run context:

- Device: `cuda`
- Topology preset: `saturated`
- Active edges: `86`
- `torch.compile`: requested and active
- Tick mode: `tensor_hot_path_no_epoch_control`

Results:

| Population | Ticks/sec | Agent-steps/sec | CUDA max memory MB |
|---:|---:|---:|---:|
| 1,000 | 2,196.607 | 2,196,606.636 | 8.848 |
| 10,000 | 2,209.794 | 22,097,942.371 | 53.016 |
| 50,000 | 769.127 | 38,456,327.383 | 246.852 |
| 100,000 | 391.457 | 39,145,695.626 | 488.195 |

Comparison to prior full-step saturated CUDA compile:

| Population | Hotpath / full-step compiled throughput | Hotpath / full-step compiled max memory |
|---:|---:|---:|
| 1,000 | 4.994 | 0.825 |
| 10,000 | 5.448 | 0.592 |
| 50,000 | 8.377 | 0.560 |
| 100,000 | 8.476 | 0.552 |

Interpretation:

- The compiler-friendly benchmark path resolves the earlier Dynamo host-counter
  issue and exposes the pure tensor throughput story.
- Saturated 86-edge AMMC compute now reaches `39.15M` agent-steps/sec at
  `100k` agents on CUDA.
- Memory at `100k` falls to `488.19 MB`, close to the earlier 8-edge benchmark,
  indicating full-step diagnostics and unused return payloads were a major
  allocation source.
- This hotpath run is the current best throughput evidence for AMMC's vectorized
  compute loop. Full-step runs should still be kept as training-loop overhead
  measurements.
- After this run, `TensorEnvironment2D.step()` gained
  `collect_telemetry=False`, and `benchmark_tick()` now uses that mode so
  diagnostics are skipped explicitly during benchmark/inference hotpaths.

## Verified saturated-topology CUDA eager hotpath follow-up

Raw files:

- `gen5/outputs/throughput_cuda_saturated_eager_hotpath_2026-06-27/throughput_results.json`
- `gen5/outputs/throughput_cuda_saturated_eager_hotpath_2026-06-27/throughput_results.csv`
- `gen5/outputs/throughput_cuda_saturated_eager_hotpath_2026-06-27/throughput_scaling.png`

Run context:

- Device: `cuda`
- Topology preset: `saturated`
- Active edges: `86`
- `torch.compile`: not requested
- Tick mode: `tensor_hot_path_no_epoch_control`

Results:

| Population | Ticks/sec | Agent-steps/sec | CUDA max memory MB |
|---:|---:|---:|---:|
| 1,000 | 434.755 | 434,755.009 | 8.823 |
| 10,000 | 420.224 | 4,202,243.296 | 89.590 |
| 50,000 | 93.323 | 4,666,166.963 | 440.950 |
| 100,000 | 47.053 | 4,705,267.105 | 883.836 |

Clean hotpath compiler comparison:

| Population | Compiled / eager hotpath throughput | Compiled / eager hotpath max memory |
|---:|---:|---:|
| 1,000 | 5.053 | 1.003 |
| 10,000 | 5.259 | 0.592 |
| 50,000 | 8.242 | 0.560 |
| 100,000 | 8.320 | 0.552 |

Interpretation:

- This control confirms that the compiled hotpath result is a real compiler
  throughput win, not merely an artifact of removing telemetry.
- Eager hotpath is only `1.001x` to `1.030x` faster than the earlier full-step
  eager run, while compiled hotpath is `5.05x` to `8.32x` faster than eager
  hotpath.
- The publishable CUDA statement for saturated 86-edge AMMC is now:
  `39.15M` compiled agent-steps/sec versus `4.71M` eager agent-steps/sec at
  `100k` agents on the same tensor hotpath.

## Verified champion-topology CUDA compiled hotpath follow-up

Raw files:

- `gen5/outputs/throughput_cuda_champion_compile_hotpath_2026-06-27/throughput_results.json`
- `gen5/outputs/throughput_cuda_champion_compile_hotpath_2026-06-27/throughput_results.csv`
- `gen5/outputs/throughput_cuda_champion_compile_hotpath_2026-06-27/throughput_scaling.png`

Run context:

- Device: `cuda`
- Topology preset: `champion`
- Adjacency JSON: `gen5_outputs/champion/champion_sparse_adjacency.json`
- Active edges: `55`
- Edge pool capacity: `128`
- Active edge utilization: `42.97%`
- `torch.compile`: requested and active
- Tick mode: `tensor_hot_path_no_epoch_control`

Results:

| Population | Ticks/sec | Agent-steps/sec | CUDA max memory MB |
|---:|---:|---:|---:|
| 1,000 | 1,400.453 | 1,400,453.327 | 4.879 |
| 10,000 | 778.297 | 7,782,966.959 | 49.200 |
| 50,000 | 604.552 | 30,227,606.471 | 243.596 |
| 100,000 | 372.509 | 37,250,874.070 | 488.195 |

Comparison to saturated 86-edge compiled hotpath:

| Population | Champion / saturated throughput | Champion / saturated max memory |
|---:|---:|---:|
| 1,000 | 0.638 | 0.551 |
| 10,000 | 0.352 | 0.928 |
| 50,000 | 0.786 | 0.987 |
| 100,000 | 0.952 | 1.000 |

Interpretation:

- The current/fresh `55`-edge champion reaches `37.25M` agent-steps/sec at
  `100k` agents on CUDA.
- This is `95.2%` of the saturated 86-edge compiled hotpath throughput at the
  same population size.
- Fewer active edges do not yet translate into proportionally lower compute,
  because `TensorEvolver` currently uses fixed-capacity `[population,
  max_edges]` tensors. In this run, biological active-edge utilization is
  `55 / 128 = 42.97%`, but the hardware tensor path is still shaped by the
  128-slot edge pool.
- This strengthens the case for reporting both active edges and edge-pool
  capacity, and for a future compact/dynamic edge backend where active topology
  maps more directly to hardware work.

## Verified champion-topology CUDA eager hotpath follow-up

Raw files:

- `gen5/outputs/throughput_cuda_champion_eager_hotpath_2026-06-27/throughput_results.json`
- `gen5/outputs/throughput_cuda_champion_eager_hotpath_2026-06-27/throughput_results.csv`
- `gen5/outputs/throughput_cuda_champion_eager_hotpath_2026-06-27/throughput_scaling.png`

Run context:

- Device: `cuda`
- Topology preset: `champion`
- Adjacency JSON: `gen5_outputs/champion/champion_sparse_adjacency.json`
- Active edges: `83`
- Edge pool capacity: `128`
- Active edge utilization: `64.84%`
- `torch.compile`: not requested
- Tick mode: `tensor_hot_path_no_epoch_control`

Results:

| Population | Ticks/sec | Agent-steps/sec | CUDA max memory MB |
|---:|---:|---:|---:|
| 1,000 | 491.532 | 491,531.557 | 8.522 |
| 10,000 | 493.944 | 4,939,441.277 | 86.594 |
| 50,000 | 112.804 | 5,640,190.208 | 425.916 |
| 100,000 | 56.873 | 5,687,259.652 | 853.889 |

Comparison to saturated 86-edge eager hotpath:

| Population | Champion / saturated throughput | Champion / saturated max memory |
|---:|---:|---:|
| 1,000 | 1.131 | 0.966 |
| 10,000 | 1.175 | 0.967 |
| 50,000 | 1.209 | 0.966 |
| 100,000 | 1.209 | 0.966 |

Interpretation:

- The current/fresh `83`-edge champion eager hotpath reaches `5.69M`
  agent-steps/sec at `100k` agents.
- It is about `1.21x` faster than the synthetic saturated 86-edge eager
  hotpath at the same population.
- This run is not directly comparable to the previous compiled champion
  hotpath run because that run seeded `55` active edges from the same displayed
  runtime path. The champion file changed between runs or a different file was
  present at the same path.
- Future benchmark outputs now include `resolved_adjacency_json` and
  `adjacency_sha256` so exact-topology comparisons can be audited.

## Verified champion-topology CUDA compiled hotpath, fingerprinted

Raw files:

- `gen5/outputs/throughput_cuda_champion_compile_hotpath_fingerprinted_2026-06-27/throughput_results.json`
- `gen5/outputs/throughput_cuda_champion_compile_hotpath_fingerprinted_2026-06-27/throughput_results.csv`
- `gen5/outputs/throughput_cuda_champion_compile_hotpath_fingerprinted_2026-06-27/throughput_scaling.png`

Run context:

- Device: `cuda`
- Topology preset: `champion`
- Adjacency JSON: `gen5_outputs/champion/champion_sparse_adjacency.json`
- Adjacency SHA-256:
  `de4cdb8f715389f8206e025435856cd2b4a55d8a7688b28b9cc3eabd5f3d904a`
- Active edges: `86`
- Edge pool capacity: `128`
- Active edge utilization: `67.19%`
- `torch.compile`: requested and active
- Tick mode: `tensor_hot_path_no_epoch_control`

Results:

| Population | Ticks/sec | Agent-steps/sec | CUDA max memory MB |
|---:|---:|---:|---:|
| 1,000 | 1,663.407 | 1,663,407.163 | 8.848 |
| 10,000 | 2,128.179 | 21,281,787.331 | 53.016 |
| 50,000 | 691.317 | 34,565,832.932 | 246.852 |
| 100,000 | 353.486 | 35,348,599.664 | 488.195 |

Comparison to saturated 86-edge compiled hotpath:

| Population | Champion / saturated throughput | Champion / saturated max memory |
|---:|---:|---:|
| 1,000 | 0.757 | 1.000 |
| 10,000 | 0.963 | 1.000 |
| 50,000 | 0.899 | 1.000 |
| 100,000 | 0.903 | 1.000 |

Interpretation:

- The fingerprinted 86-edge champion reaches `35.35M` agent-steps/sec at
  `100k` agents on CUDA.
- This is `90.3%` of synthetic saturated 86-edge compiled throughput at the
  same population size.
- Memory is identical to the saturated compiled hotpath, reinforcing that the
  current backend's fixed edge-pool capacity dominates memory footprint.
- This supersedes the earlier non-fingerprinted `55`-edge compiled champion
  run as the publication-grade champion compiled throughput artifact.
- A clean champion eager-vs-compiled speedup still requires rerunning eager
  hotpath with the same `adjacency_sha256`.

## Verified champion-topology CUDA eager hotpath, fingerprinted

Raw files:

- `gen5/outputs/throughput_cuda_champion_eager_hotpath_fingerprinted_2026-06-27/throughput_results.json`
- `gen5/outputs/throughput_cuda_champion_eager_hotpath_fingerprinted_2026-06-27/throughput_results.csv`
- `gen5/outputs/throughput_cuda_champion_eager_hotpath_fingerprinted_2026-06-27/throughput_scaling.png`

Run context:

- Device: `cuda`
- Topology preset: `champion`
- Adjacency JSON: `gen5_outputs/champion/champion_sparse_adjacency.json`
- Adjacency SHA-256:
  `de4cdb8f715389f8206e025435856cd2b4a55d8a7688b28b9cc3eabd5f3d904a`
- Active edges: `86`
- Edge pool capacity: `128`
- Active edge utilization: `67.19%`
- `torch.compile`: not requested
- Tick mode: `tensor_hot_path_no_epoch_control`

Results:

| Population | Ticks/sec | Agent-steps/sec | CUDA max memory MB |
|---:|---:|---:|---:|
| 1,000 | 339.219 | 339,219.142 | 8.522 |
| 10,000 | 367.500 | 3,675,002.299 | 86.594 |
| 50,000 | 111.565 | 5,578,244.953 | 425.916 |
| 100,000 | 56.013 | 5,601,273.103 | 853.889 |

Matched-SHA champion compiler comparison:

| Population | Compiled / eager throughput | Compiled / eager max memory |
|---:|---:|---:|
| 1,000 | 4.904 | 1.038 |
| 10,000 | 5.791 | 0.612 |
| 50,000 | 6.197 | 0.580 |
| 100,000 | 6.311 | 0.572 |

Interpretation:

- This is the first clean champion eager-vs-compiled comparison with a matching
  topology fingerprint.
- At `100k` agents, `torch.compile` gives the champion a `6.31x` speedup:
  `35.35M` compiled agent-steps/sec versus `5.60M` eager.
- Compiled peak memory at `100k` is `57.2%` of eager memory.
- This pair is now the publication-grade champion compiler evidence.

## Verified baseline comparison

Raw files:

- `gen5/outputs/phase11_remaining_outputs_2026-06-26/gen5_outputs/baselines/baseline_comparison.json`
- `gen5/outputs/phase11_remaining_outputs_2026-06-26/gen5_outputs/baselines/baseline_comparison.csv`

Results:

| Baseline | Status | Agent-steps/sec | Active params | Total params | Parameter memory MB | Max fitness | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| `ammc_sparse_evolver` | ok | 3,696,163 | 80,000 | 1,280,000 | 9.766 | 20 | Sparse AMMC policy counts active edges as active parameters |
| `dense_lif_snn` | ok | 2,421,955 | 4,932 | 4,932 | 0.019 | 4 | snnTorch available; internal LIF surrogate used unless snnTorch training is installed |
| `dense_mlp_policy` | ok | 2,901,829 | 4,866 | 4,866 | 0.019 | 3 | Dense MLP inference scaffold; PPO training hook is dependency-gated |
| `ppo_mlp_policy` | skipped | n/a | n/a | n/a | n/a | n/a | `stable-baselines3` unavailable |

Interpretation:

- AMMC sparse evolution is faster than the dense LIF and dense MLP scaffolds in
  this benchmark bundle.
- AMMC also produced a much higher short-run max fitness (`20`) than the
  untrained dense baselines (`4` and `3`).
- This is a useful systems baseline, but it is not yet a fair learned-policy
  comparison against trained BPTT SNN or trained PPO because the PPO dependency
  was unavailable and the dense policies are scaffold baselines.

## Image-level single-run telemetry read

`image-3.png` shows a single 500-generation evolution run:

- Max fitness rapidly improves early, then fluctuates on a plateau around
  `18-20`, with occasional spikes above `25`.
- Mean population fitness remains noisy around zero.
- Mean active synapses rises quickly and saturates near the high 80s.

This aligns with the validated raw champion package and reinforces the earlier
finding that topology saturation is a real pressure.

## Current evidence status

| Requirement | Evidence status | Notes |
|---|---|---|
| Single champion export integrity | Proven | Raw files validate 16 neurons / 88 synapses / matched weights |
| Multi-seed convergence | Proven | 10 seeds, final mean best `26.0 +/- 0.667`, all seeds `25-27` |
| Plasticity benefit over static network | Proven | Full `25.9`, gated `24.6`, static `13.6` |
| Gated adult plasticity best overall | Not proven | Full plasticity has higher raw fitness and better retention; gated is more synapse-efficient |
| Catastrophic forgetting measured | Proven | Retention ablation now uploaded and analyzed |
| Throughput scaling 1k/10k/50k/100k | Proven | 8-edge CUDA `29.3M` agent-steps/sec at 100k; 86-edge saturated CUDA `4.64M` at 100k |
| Baseline comparison against dense LIF / PPO | Partial | Dense LIF/MLP scaffold compared; PPO skipped due missing `stable-baselines3` |

## Recommended next actions

1. Add topology pressure to future evolution runs because both the single-run
   telemetry and prior champion run show active-edge saturation.
2. Rerun saturated-topology CUDA throughput with `--compile`, and run exact
   champion adjacency throughput.
3. Tune the adult/gated plasticity rule. Current settings improve compactness
   but do not beat full plasticity on retention.
4. Run trained baselines:
   - BPTT-trained static LIF SNN
   - PPO-trained MLP after installing `stable-baselines3`
5. Track active-parameter-adjusted fitness and watts/memory-normalized
   throughput in the next comparison report.
