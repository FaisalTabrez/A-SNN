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
import torch
try:
    import torch_xla
    print("XLA device:", torch_xla.device() if hasattr(torch_xla, "device") else "legacy torch_xla API")
except Exception as exc:
    print("PyTorch/XLA unavailable:", exc)
```

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
  --population-sizes 1000 10000 50000 100000 \
  --steps 240 \
  --warmup 30 \
  --output-dir gen5_outputs/throughput_xla
```

Then optionally run the T4/L4 CUDA fallback for continuity with prior Phase 11
numbers:

```python
!python gen5/benchmarks/benchmark_throughput.py \
  --device cuda \
  --population-sizes 1000 10000 50000 100000 \
  --steps 240 \
  --warmup 30 \
  --compile \
  --output-dir gen5_outputs/throughput_cuda
```

Expected outputs:

- `throughput_results.json`
- `throughput_results.csv`
- `throughput_scaling.png`

Main metrics:

- `ticks_per_second`
- `agent_steps_per_second`
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
