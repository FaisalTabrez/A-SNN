# AMMC Gen-5

Production-grade neuromorphic framework scaffold for the AMMC-SNN line.

Gen-4 proved the biological mechanics in a JavaScript sandbox: structural
plasticity, astrocyte modulation, embodied feedback, sleep consolidation,
serialization, PyTorch export, and Colab weight import. Gen-5 moves those ideas
into a mathematical Python backend with TPU/XLA as the near-term accelerator
target, plus future CUDA/C++ and neuromorphic hardware backends.

## Design stance

Gen-5 is not a visual CAD simulator. It is a hardware-accelerated learning
runtime for continuous agents.

The first scaffold uses PyTorch for integration and testing, but it keeps the
hard parts behind replaceable backend boundaries:

- `DynamicSparseLinear` models a sparse edge-list allocator with active slots,
  pruning, sprouting, STW/LTW memory, and an autograd-compatible forward pass.
- `LTWSTWMemory` keeps volatile short-term weight dynamics separate from
  durable long-term weight dynamics.
- `DualTensorManager` models the slow astrocyte chemical field as a low-rate
  tensor overlay that modulates the fast electrical graph.

This prototype intentionally uses fixed-capacity edge buffers instead of
pretending standard PyTorch can freely resize trainable parameters during an
active backward pass. The XLA path benefits from those static capacities now;
the CUDA/C++ implementation can later replace the allocator without changing
the high-level AMMC API.

## Accelerator runtime stance

Gen-5 is now TPU/XLA-first for Colab-scale work:

- `--device auto` tries XLA first when PyTorch/XLA can acquire a device, then
  CUDA, then CPU.
- `--device xla` / `--device tpu` explicitly targets a Colab TPU runtime.
- CUDA/T4 remains supported through ordinary PyTorch, but custom CUDA kernels
  are deferred until the XLA architecture and statistical claims settle.
- XLA step flushing and synchronization live in `ammc_gen5.runtime` so model
  code does not scatter direct `torch_xla` imports everywhere.
- Static-shape mutation/respawn patterns are preferred over dynamic allocation
  inside hot loops.

## Sprint map

1. Dynamic sparse edge-list backend
   - PyTorch prototype in `ammc_gen5.dynamic_sparse`
   - Later: C++/CUDA allocator that physically frees/reuses VRAM edge slots

2. Dual-frequency tensor manager
   - Fast sparse electrical graph
   - Slow dense chemical astrocyte grid

3. LTW/STW optimizer compatibility
   - STW for short-lived working memory
   - LTW for durable consolidated memory
   - optimizer parameter groups can treat them differently

## Quick smoke test

From the repo root:

```powershell
python -m compileall gen5
python gen5/examples/sprint1_smoke.py
python gen5/examples/sprint4_5_vectorized_loop.py
python gen5/examples/sprint6_7_tensor_evolver.py
python gen5/examples/sprint8_evolving_headless_loop.py
```

The smoke example requires PyTorch. Syntax checks do not.

## Sprint 4/5 additions

The Gen-5 package now includes the first headless embodiment path:

- `TensorEnvironment2D`: batched 2D physics for large swarms, with tensorized
  velocity updates, nearest food/toxin sensing, collision detection, respawn,
  and per-agent fitness.
- `VectorizedTransducer`: converts environment sensory tensors into neural
  channels and decodes motor neurons back into x/y actions.
- `HeadlessAMMCLoop`: couples `TensorEnvironment2D` directly to
  `DynamicSparseLinear` without rendering.

The API is designed for 10,000-agent runs on XLA/CUDA, while remaining small
enough to unit test on CPU.

## Sprint 6/7 additions

The first tensorized genetic algorithm lives in `TensorEvolver`:

- ranks population fitness with `torch.argsort`
- broadcasts top genomes into bottom performers without CPU organism loops
- applies LTW Gaussian noise to children
- prunes active child edges with a mutation mask
- sprouts inactive child edge slots with random source/target/sign/delay tensors
- can apply each organism's own sparse genome to its own neural state using a
  batched scatter-add forward pass

This is the stepping stone from a shared-brain vectorized loop to true
per-agent connectomes in VRAM.

## Sprint 8 addition

`EvolvingHeadlessAMMCLoop` is the first complete Gen-5 cycle:

```text
TensorEnvironment2D physics
-> VectorizedTransducer sensors
-> TensorEvolver per-agent recurrent brains
-> VectorizedTransducer motors
-> TensorEnvironment2D actions
```

At `epoch_steps`, it reads `environment.fitness`, calls
`TensorEvolver.evolve(...)`, resets environment positions/scores, clears neural
membrane state, and advances the generation counter.

Attach `EvolutionTelemetryLogger` to the loop to save headless evolution
metrics:

```python
logger = EvolutionTelemetryLogger()
loop = EvolvingHeadlessAMMCLoop(env, evolver, logger=logger)
loop.run(steps=epoch_steps * 500)
logger.save_json("gen5_outputs/evolution_telemetry.json")
logger.save_csv("gen5_outputs/evolution_telemetry.csv")
logger.plot("gen5_outputs/evolution_telemetry.png")
```

The plot tracks max fitness, mean population fitness, and mean active synapses
over generations.

## Champion export

After a long Colab run, export the all-time best genome:

```python
from ammc_gen5 import ChampionExporter

exporter = ChampionExporter()
result = exporter.export_from_snapshot(
    loop.best_genome_snapshot,
    "gen5_outputs/champion",
    neuron_count=evolver.neuron_count,
    organism_id="Gen5Champion",
    fitness=loop.best_fitness,
)
print(result)
```

This writes:

- `champion_connectome.json` — load first in `index.html`
- `colab_weights.json` — import second with "Import PyTorch Weights"
- `champion_sparse_adjacency.json` — sparse adjacency/LTW analysis file

Telemetry files alone cannot reconstruct a champion brain; the live Colab
kernel must still hold `loop.best_genome_snapshot`, or you must save a checkpoint
containing that snapshot during training.

## Sprint 11 evaluation and benchmarks

Sprint 11 moves Gen-5 from "the mechanism works" to statistical proof.

### Multi-seed convergence trials

Run 10 independent seeds and plot mean +/- standard deviation of all-time best
fitness:

```powershell
python gen5/examples/sprint11_statistical_evaluation.py `
  --device xla `
  --seeds 42 43 44 45 46 47 48 49 50 51 `
  --generations 500 `
  --population-size 10000 `
  --epoch-steps 120 `
  --output-dir gen5_outputs/statistical_trials
```

Outputs:

- `multi_seed_trials.json`
- `multi_seed_trials.csv`
- `multi_seed_aggregate.csv`
- `multi_seed_best_fitness_mean_std.png`

### Plasticity ablation

Run the static/full/gated plasticity comparison under an inverted food/toxin
sensor perturbation:

```powershell
python gen5/examples/sprint11_plasticity_ablation.py `
  --device xla `
  --seeds 42 43 44 45 46 47 48 49 50 51 `
  --generations 500 `
  --population-size 10000 `
  --epoch-steps 120 `
  --output-dir gen5_outputs/plasticity_ablation
```

Groups:

- `static_snn`: topology and weights locked.
- `full_plasticity_infant`: aggressive sprouting/pruning/noise.
- `gated_plasticity_adult`: pruning and LTW noise gated behind positive reward.

### Retention / forgetting ablation

The first plasticity ablation measures adaptation under perturbation. To measure
catastrophic forgetting, run the three-phase retention protocol:

```powershell
python gen5/examples/sprint11_retention_ablation.py `
  --device xla `
  --seeds 42 43 44 45 46 47 48 49 50 51 `
  --original-generations 100 `
  --perturbation-generations 300 `
  --recovery-generations 100 `
  --population-size 10000 `
  --epoch-steps 120 `
  --output-dir gen5_outputs/retention_ablation
```

Phases:

1. original environment
2. inverted food/toxin sensor perturbation
3. original environment again

Outputs:

- `retention_ablation.json`
- `retention_ablation_records.csv`
- `retention_ablation_summary.csv`
- `retention_ablation_phase_fitness.png`

### Throughput benchmark

Measure raw tensor-loop scaling on Colab TPU/XLA:

```powershell
python gen5/benchmarks/benchmark_throughput.py `
  --device xla `
  --topology-preset foraging `
  --population-sizes 1000 10000 50000 100000 `
  --steps 240 `
  --warmup 30 `
  --output-dir gen5_outputs/throughput_xla
```

Run the champion-like saturated-topology comparison:

```powershell
python gen5/benchmarks/benchmark_throughput.py `
  --device xla `
  --topology-preset saturated `
  --active-edges 86 `
  --population-sizes 1000 10000 50000 100000 `
  --steps 240 `
  --warmup 30 `
  --output-dir gen5_outputs/throughput_xla_saturated
```

If a champion sparse adjacency has been exported, benchmark that exact topology:

```powershell
Get-ChildItem -Recurse -Filter champion_sparse_adjacency.json
```

```powershell
python gen5/benchmarks/benchmark_throughput.py `
  --device xla `
  --topology-preset champion `
  --adjacency-json gen5/outputs/colab_500_gen_2026-06-25/champion_sparse_adjacency.json `
  --population-sizes 1000 10000 50000 100000 `
  --steps 240 `
  --warmup 30 `
  --output-dir gen5_outputs/throughput_xla_champion
```

In Colab, use `find /content -name champion_sparse_adjacency.json -print` and
pass the printed path to `--adjacency-json` if your champion export lives in a
runtime output folder rather than the repository archive.

For a Colab T4/L4 CUDA fallback:

```powershell
python gen5/benchmarks/benchmark_throughput.py `
  --device cuda `
  --topology-preset foraging `
  --population-sizes 1000 10000 50000 100000 `
  --steps 240 `
  --warmup 30 `
  --compile `
  --output-dir gen5_outputs/throughput
```

The benchmark reports ticks/sec and agent-steps/sec. CUDA memory is reported
when available; XLA memory is left as null because Colab TPU memory counters do
not share CUDA's allocation semantics.

### Baseline comparison scaffold

Compare inference footprint and speed against dense LIF-style and dense MLP
policies:

```powershell
python gen5/benchmarks/comparison_baselines.py `
  --device xla `
  --population-size 10000 `
  --steps 240 `
  --output-dir gen5_outputs/baselines
```

This scaffold records whether `snntorch` and `stable-baselines3` are available.
Full BPTT/PPO training should be run in Colab after those dependencies are
installed.
