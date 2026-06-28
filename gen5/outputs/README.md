# Gen-5 outputs

This folder stores retained experiment evidence for AMMC Gen-5.

It is intentionally not treated as disposable scratch space. Important runs are
archived here so later decisions can point to exact raw files and analysis.

## Bundle convention

Use:

```text
<topic>_<backend-or-context>_<YYYY-MM-DD>/
```

Examples:

- `colab_500_gen_2026-06-25/`
- `neuron_scaling_cuda_2026-06-27/`
- `sparse_efficiency_finalists_cuda_2026-06-28/`
- `throughput_cuda_champion_compile_hotpath_fingerprinted_2026-06-27/`

## What each major bundle should include

- raw `.json` and/or `.csv` outputs,
- plots such as `.png` when available,
- `analysis.md` for interpretation,
- any Colab log or command transcript needed to reproduce the run.

## Legacy uploads

`legacy_first_run_upload_2026-06-25/` preserves the original top-level
`1st run/` upload bundle after the repository reorganization. The interpreted
and annotated version of that run lives in `colab_500_gen_2026-06-25/`.
