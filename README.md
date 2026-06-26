# A-SNN / AMMC Gen-5

Adaptive Spiking Neural Network research workspace for the AMMC line:
browser-proven biological mechanics, Gen-5 headless tensor evolution, and
accelerator-oriented benchmarking.

## What is in this repo

- `index.html` — Gen-4 browser sandbox for visual neuromorphic inspection,
  connectome import/export, PyTorch weight import, seeded replay, and embodied
  foraging behavior.
- `gen5/` — Python Gen-5 framework scaffold:
  - dynamic sparse edge pools,
  - STW/LTW memory separation,
  - vectorized 2D embodiment,
  - tensorized swarm evolution,
  - champion export,
  - statistical evaluation and benchmark harnesses.
- `research.md` — living research log. We update this whenever project
  decisions or findings change.
- `design/` — project concept and UI reference assets.
- `gen5/outputs/` — retained experiment outputs and Phase 11 evidence bundles.

## Current architecture direction

Gen-5 is TPU/XLA-first for near-term Colab-scale work. CUDA/T4 remains
supported through ordinary PyTorch, while custom CUDA kernels are deferred
until the XLA-compatible algorithm surface stabilizes.

Useful references:

- `gen5/README.md`
- `gen5/docs/GEN5_ARCHITECTURE.md`
- `gen5/docs/TPU_XLA_MIGRATION.md`
- `gen5/docs/PHASE11_COLAB_RUNBOOK.md`

## Quick validation

From the repository root:

```powershell
python -m compileall gen5
python -m unittest discover -s gen5\tests -v
```

Torch-dependent tests require a Python environment with PyTorch installed. The
local bundled runtime used by Codex may skip those tests if PyTorch is absent.

## Colab TPU/XLA smoke benchmark

On a Colab TPU runtime:

```python
!python gen5/benchmarks/benchmark_throughput.py \
  --device xla \
  --population-sizes 1000 10000 50000 100000 \
  --steps 240 \
  --warmup 30 \
  --output-dir gen5_outputs/throughput_xla
```

## Research discipline

The project uses `research.md` as the source of truth for:

- core findings,
- project decisions,
- evidence status,
- open questions,
- next recommended steps.

When a meaningful implementation or research decision is made, update
`research.md` in the same change set.
