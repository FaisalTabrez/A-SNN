# Contributing notes

This project is both a framework and a research log. Please keep changes easy
to audit.

## Before changing code

- Read `README.md` and `docs/PROJECT_STRUCTURE.md`.
- Check `research.md` for the current baseline and next recommended step.
- Prefer small, evidence-backed changes over broad rewrites.

## When adding experiment results

- Put retained evidence under `gen5/outputs/<descriptive_run_name>/`.
- Include raw JSON/CSV outputs and plots when available.
- Add an `analysis.md` explaining what the run proves or fails to prove.
- Update `research.md` if the result changes a decision, baseline, or roadmap.

## When changing Gen-5 code

Run at least:

```powershell
python -m compileall gen5
python -m unittest discover -s gen5\tests -v
```

If PyTorch is unavailable locally, record that limitation in the handoff.

## Current baseline discipline

For the simple bot world:

- raw-survival baseline: `low_ltw_pruning`, `32` neurons,
- sparse-efficiency baseline: `gentle_ltw_scheduled`, `32` neurons.

Do not expand neuron count on the simple world without first adding a harder
environment that can reward hidden-state computation.
