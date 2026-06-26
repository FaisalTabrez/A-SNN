# Gen-5 TPU/XLA migration note

Date: 2026-06-26

## Decision

Gen-5 is moving to a TPU/XLA-first execution architecture for the near-term
Colab workflow. CUDA remains supported through standard PyTorch on T4/L4 GPUs,
but custom CUDA kernels and allocator work are deferred.

This is not a rejection of CUDA. It is a sequencing decision: the current
hardware path should match the accelerator the project can actually use now.

## Runtime contract

The backend contract is centralized in `ammc_gen5.runtime`:

- `resolve_device("auto")`
  - XLA first if PyTorch/XLA can acquire a device
  - CUDA second
  - CPU last
- `resolve_device("xla")` or `resolve_device("tpu")`
  - explicit TPU/XLA target
- `mark_step(device)`
  - flushes one logical XLA step without requiring model code to know the
    PyTorch/XLA API details
- `sync(device)`
  - synchronizes CUDA or XLA for benchmark timing
- `seed_everything(...)` and `make_generator(...)`
  - keep RNG handling backend-aware

## XLA-friendly coding rules

Prefer:

- fixed-capacity tensors,
- fixed-shape random tensors plus masks,
- host-side counters for Python control flow,
- sparse slots marked inactive instead of dynamically resizing tensors,
- explicit XLA step markers in long loops,
- benchmark synchronization only at warmup/timing boundaries.

Avoid in hot loops:

- `.item()` for control flow,
- `bool(tensor.any())`,
- dynamically sized tensors from `mask.sum()`,
- Python loops over organisms,
- backend-specific imports scattered across model files.

## Current code changes

- `TensorEnvironment2D._respawn(...)` now uses fixed-shape masked respawns.
- `TensorEvolver.mutate_children(...)` now generates fixed-shape sprout
  candidates and applies them with masks.
- `EvolvingHeadlessAMMCLoop` uses host epoch/generation counters to avoid
  per-step `.item()` synchronization.
- Headless loops call `mark_step(...)` so TPU lazy execution sees logical tick
  boundaries.
- Throughput and baseline scripts accept `--device xla`.
- `torch.compile` is skipped on XLA because PyTorch/XLA already compiles lazily
  into XLA graphs.

## Colab TPU commands

On a TPU runtime:

```python
!python gen5/benchmarks/benchmark_throughput.py \
  --device xla \
  --population-sizes 1000 10000 50000 100000 \
  --steps 240 \
  --warmup 30 \
  --output-dir gen5_outputs/throughput_xla
```

```python
!python gen5/examples/sprint11_statistical_evaluation.py \
  --device xla \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --generations 500 \
  --population-size 10000 \
  --epoch-steps 120 \
  --output-dir gen5_outputs/statistical_trials_xla
```

```python
!python gen5/examples/sprint11_retention_ablation.py \
  --device xla \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --original-generations 100 \
  --perturbation-generations 300 \
  --recovery-generations 100 \
  --population-size 10000 \
  --epoch-steps 120 \
  --output-dir gen5_outputs/retention_ablation_xla
```

## Migration caveats

- XLA support is currently a compatibility layer over PyTorch tensor ops, not a
  hand-written TPU kernel.
- XLA memory numbers are not directly comparable to CUDA allocation counters;
  current benchmark fields return null for XLA memory.
- Some sparse/mutation operations may compile slowly on first use. Always
  separate warmup timing from measured timing.
- Dynamic topology is still represented as fixed-capacity masked slots. This is
  intentionally XLA-friendly and should remain the default until a later custom
  allocator exists.

## Next validation target

Run the Phase 11 throughput, multi-seed, plasticity, and retention scripts on a
Colab TPU runtime with `--device xla`, then compare:

- convergence curve vs CUDA/T4,
- retention summary vs CUDA/T4,
- throughput at 1k/10k/50k/100k,
- compile/warmup cost,
- any unsupported XLA operations.
