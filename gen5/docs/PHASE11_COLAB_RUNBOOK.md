# Phase 11 Colab Runbook

Use this after the verified Sprint 11 multi-seed and plasticity ablation runs.

Current verified evidence:

- Multi-seed convergence:
  - final mean all-time best fitness `26.0 +/- 0.667`
  - all seeds finished between `25` and `27`
- Plasticity ablation:
  - static SNN: `13.6 +/- 0.843`
  - full plasticity: `25.9 +/- 0.994`
  - gated adult plasticity: `24.6 +/- 1.075`
  - gated adult uses fewer active synapses, but full plasticity wins raw
    perturbed fitness

Phase 11 evidence is now complete for the first benchmark pass. This runbook is
kept as the repeatable Colab protocol and has been updated for the TPU/XLA-first
runtime.

Device policy:

- Use `--device xla` on a Colab TPU runtime.
- Use `--device cuda` on a Colab T4/L4 GPU runtime.
- Use `--device auto` only when you are comfortable with Gen-5 choosing XLA
  first, then CUDA, then CPU.

Quick TPU check:

```python
import sys, torch
print("Python:", sys.version)
print("Torch:", torch.__version__)
try:
    import torch_xla
    print("XLA device:", torch_xla.device() if hasattr(torch_xla, "device") else "legacy torch_xla API")
except Exception as exc:
    print("PyTorch/XLA unavailable:", exc)
```

If `torch_xla` is unavailable, install a PyTorch/XLA build compatible with the
current Colab Python/PyTorch runtime, then restart the runtime. For current
Colab Python 3.12 TPU runtimes, start with:

```python
!pip install -q --pre torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
```

If the error mentions `_XLAC` or `undefined symbol`, `torch_xla` is installed
but binary-incompatible with the active `torch` wheel. In that case, use a fresh
TPU runtime and install matching nightly CPU PyTorch + PyTorch/XLA wheels
together:

```python
!pip uninstall -y torch torch_xla torchvision torchaudio
!pip install -q --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
!pip install -q --pre torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
```

Then restart the runtime and rerun the TPU check above. If the notebook is a
T4/L4 GPU runtime rather than a TPU runtime, use `--device cuda` instead of
`--device xla`.

## 1. Retention / forgetting ablation

This tests original -> perturbed -> original behavior.

```python
!python gen5/examples/sprint11_retention_ablation.py \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --original-generations 100 \
  --perturbation-generations 300 \
  --recovery-generations 100 \
  --population-size 10000 \
  --epoch-steps 120 \
  --device xla \
  --output-dir gen5_outputs/retention_ablation
```

Expected outputs:

- `gen5_outputs/retention_ablation/retention_ablation.json`
- `gen5_outputs/retention_ablation/retention_ablation_records.csv`
- `gen5_outputs/retention_ablation/retention_ablation_summary.csv`
- `gen5_outputs/retention_ablation/retention_ablation_phase_fitness.png`

Main metric:

- `recovery_retention_ratio`
- `forgetting_delta`
- `perturbation_gain_over_original`

## 2. Throughput scaling

Run this first on Colab TPU/XLA.

```python
!python gen5/benchmarks/benchmark_throughput.py \
  --device xla \
  --topology-preset foraging \
  --population-sizes 1000 10000 50000 100000 \
  --steps 240 \
  --warmup 30 \
  --output-dir gen5_outputs/throughput_xla
```

Run the saturated-topology comparison after the foraging-prior run:

```python
!python gen5/benchmarks/benchmark_throughput.py \
  --device xla \
  --topology-preset saturated \
  --active-edges 86 \
  --population-sizes 1000 10000 50000 100000 \
  --steps 240 \
  --warmup 30 \
  --output-dir gen5_outputs/throughput_xla_saturated
```

If the champion adjacency file is present in Colab, run the exact champion
topology:

```python
!find /content -name champion_sparse_adjacency.json -print
```

```python
!python gen5/benchmarks/benchmark_throughput.py \
  --device xla \
  --topology-preset champion \
  --adjacency-json gen5/outputs/colab_500_gen_2026-06-25/champion_sparse_adjacency.json \
  --population-sizes 1000 10000 50000 100000 \
  --steps 240 \
  --warmup 30 \
  --output-dir gen5_outputs/throughput_xla_champion
```

Use the path printed by `find` if your champion package lives somewhere else,
for example under `gen5_outputs/champion/` from a fresh exporter run.

Then optionally run the T4/L4 CUDA fallback for continuity with prior Phase 11
numbers:

```python
!python gen5/benchmarks/benchmark_throughput.py \
  --device cuda \
  --topology-preset foraging \
  --population-sizes 1000 10000 50000 100000 \
  --steps 240 \
  --warmup 30 \
  --compile \
  --output-dir gen5_outputs/throughput_cuda
```

For the exact champion topology on CUDA/T4, replace `ADJ_PATH` with the path
printed by the `find` command above:

```python
ADJ_PATH = "gen5/outputs/colab_500_gen_2026-06-25/champion_sparse_adjacency.json"
!python gen5/benchmarks/benchmark_throughput.py \
  --device cuda \
  --topology-preset champion \
  --adjacency-json "$ADJ_PATH" \
  --population-sizes 1000 10000 50000 100000 \
  --steps 240 \
  --warmup 30 \
  --compile \
  --output-dir gen5_outputs/throughput_cuda_champion_compile_hotpath
```

Expected outputs:

- `throughput_results.json`
- `throughput_results.csv`
- `throughput_scaling.png`

Main metrics:

- `ticks_per_second`
- `agent_steps_per_second`
- `topology_preset`
- `seeded_active_edges`
- `edge_pool_capacity`
- `active_edge_utilization`
- CUDA memory allocated / max allocated when running CUDA
- `accelerator_backend`, which should read `xla` on TPU runs

## 3. Baseline comparison

This records AMMC sparse, dense LIF-style SNN, dense MLP, and PPO dependency
availability on the same foraging task.

```python
!python gen5/benchmarks/comparison_baselines.py \
  --device xla \
  --population-size 10000 \
  --steps 240 \
  --output-dir gen5_outputs/baselines_xla
```

Optional baseline dependencies:

```python
!pip install snntorch stable-baselines3 gymnasium
```

Expected outputs:

- `baseline_comparison.json`
- `baseline_comparison.csv`

Main metrics:

- active parameters
- total parameters
- parameter memory
- ticks/sec
- agent-steps/sec
- mean/max fitness

## 4. Zip and download all remaining outputs

```python
!zip -r phase11_remaining_outputs.zip \
  gen5_outputs/retention_ablation \
  gen5_outputs/throughput_xla \
  gen5_outputs/throughput_xla_saturated \
  gen5_outputs/throughput_xla_champion \
  gen5_outputs/throughput_cuda \
  gen5_outputs/baselines_xla
```

```python
from google.colab import files
files.download("phase11_remaining_outputs.zip")
```

Upload `phase11_remaining_outputs.zip` back into Codex for analysis.

## 5. Verify package completeness

After copying/downloading outputs, run:

```python
!python gen5/tools/verify_phase11_outputs.py --roots . gen5_outputs
```

On Windows/Codex local review, the equivalent command is:

```powershell
python gen5/tools/verify_phase11_outputs.py `
  --roots . "C:\Users\FAISAL TABREZ\Downloads"
```

The verifier reports which groups are complete:

- `champion`
- `multi_seed`
- `plasticity_ablation`
- `retention_ablation`
- `throughput`
- `baselines`
