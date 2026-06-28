# Repository structure

This repo has two connected layers:

1. A browser sandbox that makes the biology visible.
2. A Python Gen-5 framework that makes the same ideas scalable and measurable.

## Top-level files

| Path | Purpose |
|---|---|
| `index.html` | Gen-4 browser sandbox. Keep this at the root so local visual QA stays simple. |
| `research.md` | Living lab notebook: core findings, project decisions, evidence status, and next steps. |
| `README.md` | Main newcomer landing page. |
| `.gitignore` | Python/cache ignore rules. |

## Top-level directories

| Path | Purpose |
|---|---|
| `docs/` | Repository-level documentation and navigation. |
| `assets/design/` | Design/concept images used for communication and UI reference. |
| `gen5/` | Headless Python framework, evaluation scripts, tests, benchmarks, and retained outputs. |

## Gen-5 layout

| Path | Purpose |
|---|---|
| `gen5/ammc_gen5/` | Importable Python package. Core runtime, sparse edge pools, environment, transducer, evolver, telemetry, and export tools live here. |
| `gen5/examples/` | Reproducible sprint/evaluation scripts. These are the scripts normally run in Colab. |
| `gen5/benchmarks/` | Throughput and comparison-baseline benchmarks. |
| `gen5/tests/` | Unit and contract tests. |
| `gen5/tools/` | Output verification and utility scripts. |
| `gen5/docs/` | Gen-5 technical architecture and runbooks. |
| `gen5/outputs/` | Retained experiment bundles. These are evidence, not scratch space. |

## Output bundle convention

Experiment folders under `gen5/outputs/` should use descriptive names:

```text
<topic>_<backend-or-context>_<YYYY-MM-DD>/
```

Good examples:

- `sparse_efficiency_finalists_cuda_2026-06-28/`
- `throughput_cuda_champion_compile_hotpath_fingerprinted_2026-06-27/`
- `colab_500_gen_2026-06-25/`

Each important bundle should include:

- raw JSON/CSV outputs,
- plots when generated,
- `analysis.md` summarizing the result,
- enough metadata to reproduce the command.

## Current working baseline

For the simple bot world:

- use `low_ltw_pruning` with `32` neurons as the default raw-fitness baseline,
- use `gentle_ltw_scheduled` with `32` neurons as the sparse-efficiency
  baseline,
- move next toward harder bot worlds instead of scaling neuron count further.
- use `gen5/examples/sprint14_harder_worlds.py` to test whether harder
  environments make hidden decision nodes useful.

The detailed evidence lives in:

- `gen5/outputs/sparse_efficiency_finalists_cuda_2026-06-28/analysis.md`
- `research.md`
