# A-SNN / AMMC

Adaptive spiking-neural-network research workspace for the AMMC line: a
browser-proven biological sandbox plus a headless Gen-5 tensor framework for
large-scale embodied evolution.

The short version:

- `index.html` is the Gen-4 browser sandbox for visual inspection.
- `gen5/` is the Python framework for headless tensor evolution and benchmarks.
- `research.md` is the living project memory and decision log.
- `gen5/outputs/` contains retained experiment evidence.

## Start here

If you are new to the repo, read these in order:

1. [Repository structure](docs/PROJECT_STRUCTURE.md)
2. [Gen-5 architecture](gen5/docs/GEN5_ARCHITECTURE.md)
3. [Current research log](research.md)
4. [Gen-5 README](gen5/README.md)
5. [Phase 11 Colab runbook](gen5/docs/PHASE11_COLAB_RUNBOOK.md)

## Repository map

```text
.
|-- index.html                 # Gen-4 browser sandbox / visual connectome lab
|-- research.md                # living findings, decisions, and next steps
|-- docs/                      # newcomer navigation and repo-level docs
|-- assets/design/             # concept and UI reference images
`-- gen5/
    |-- ammc_gen5/             # Python package: sparse brains, environment, evolution
    |-- examples/              # runnable sprint/evaluation scripts
    |-- benchmarks/            # throughput and baseline comparisons
    |-- tests/                 # unit/contract tests
    |-- tools/                 # verification utilities
    |-- docs/                  # Gen-5 technical runbooks and architecture notes
    `-- outputs/               # retained experiment bundles and analyses
```

## Current research baseline

For the simple 2D bot world, sparse-efficiency tuning is now frozen:

- Default raw-survival baseline: `low_ltw_pruning`, `32` neurons.
- Sparse-efficiency baseline: `gentle_ltw_scheduled`, `32` neurons.
- Next scientific step: harder bot-world variants that reward hidden-state
  computation before expanding neuron count further.

See [research.md](research.md) for the evidence trail.

## Browser sandbox quick start

From the repository root, serve the static sandbox:

```powershell
python -m http.server 4173
```

Then open:

```text
http://127.0.0.1:4173/
```

The root `index.html` remains intentionally stable because it is the visual QA
entry point for connectome import/export and Gen-5 champion replay.

## Gen-5 quick validation

From the repository root:

```powershell
python -m compileall gen5
python -m unittest discover -s gen5\tests -v
```

Torch-dependent tests require a Python environment with PyTorch installed.

## Common Gen-5 commands

List sparse-efficiency groups:

```powershell
python gen5/examples/sprint13_sparse_efficiency_ablation.py --list-groups
```

Run a small local smoke test:

```powershell
python gen5/examples/sprint1_smoke.py
python gen5/examples/sprint4_5_vectorized_loop.py
```

Run the Colab/XLA throughput benchmark on a TPU runtime:

```python
!python gen5/benchmarks/benchmark_throughput.py \
  --device xla \
  --population-sizes 1000 10000 50000 100000 \
  --steps 240 \
  --warmup 30 \
  --output-dir gen5_outputs/throughput_xla
```

## Evidence discipline

Experiment outputs kept in the repository should live under `gen5/outputs/`
and include an `analysis.md` whenever possible. New research decisions should
update [research.md](research.md) in the same change set.

The repo is deliberately part lab notebook, part framework. The code tells us
what can run; the evidence folders tell us what we have actually observed.
