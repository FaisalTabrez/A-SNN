# Graphify knowledge workflow

Status: active repository knowledge layer as of 2026-08-26.

Graphify replaces the custom Obsidian generator, retriever, and sync utilities.
Markdown remains the evidence authority; Graphify is a derived navigation and
query layer, never the source of a research claim.

## Canonical corpus

Graphify should index:

- `research.md` for findings, decisions, claim boundaries, and next actions;
- `gen5/docs/` for preregistrations, analyses, architecture, and runbooks;
- `docs/` and `README.md` for repository navigation;
- `gen5/ammc_gen5/`, `gen5/examples/`, `gen5/benchmarks/`, `gen5/tools/`, and
  `gen5/tests/` for executable implementation and contracts;
- Markdown `analysis.md` records under `gen5/outputs/`.

`.graphifyignore` excludes raw datasets, ZIP bundles, plots, local result
imports, caches, Graphify's own output, and the archived Obsidian vault. This
keeps the graph focused on reviewable evidence rather than duplicating hundreds
of megabytes of raw artifacts.

## Installation

On Windows PowerShell from the repository root:

```powershell
uv tool install graphifyy==0.9.50
graphify install --project --platform codex
```

The project-scoped installation adds the Graphify skill and `AGENTS.md`
guidance. The committed hook uses the portable `graphify` command, so each
checkout must install the CLI on `PATH`.

## Building the graph

Code-only extraction is fully local and creates the initial implementation
graph without sending repository contents to a model provider:

```powershell
graphify extract . --code-only --no-cluster
graphify cluster-only . --no-label
graphify update . --force
```

The tested `0.9.50` update path also records Markdown headings and explicit
references as `EXTRACTED` nodes and edges, which is enough for the repository's
research-source coverage gate. Optional richer semantic relationships require
a model-assisted update. Run that from Codex with `/graphify . --update`, or
choose an explicit backend for headless extraction:

```powershell
graphify extract . --backend ollama --force
```

Replace `ollama` only after making an explicit data-residency decision. Code is
parsed locally; prose supplied to a cloud backend leaves the machine for that
semantic pass. Never rely on backend auto-detection for private research.

Graphify writes the shared graph to `graphify-out/`. Commit `graph.json`,
`GRAPH_REPORT.md`, the interactive HTML, and the portable manifest. Keep the
cache, local interpreter/root sidecars, and cost ledger untracked.

## Updating

After code changes:

```powershell
graphify update .
```

After research or documentation changes, run `graphify update . --force` and
the research contract below. If the shared graph intentionally includes richer
`INFERRED` prose relationships, repeat semantic extraction with the same
explicitly selected backend. Do not silently change providers between graph
revisions.

## Acceptance contract

Run:

```powershell
python gen5/tools/verify_graphify_contract.py
python gen5/tools/verify_graphify_contract.py --require-research
```

The first command checks implementation coverage. The second is the migration
gate and must find `research.md`, the primary evidence roadmap, and the
LTH-informed itinerary in the graph. It proves source coverage, not the
correctness of model-inferred relationships.

The following queries form the human acceptance test:

```powershell
graphify query "What is the current active evidence track?"
graphify query "Which AMMC adaptive mechanisms have failed their causal gates?"
graphify query "What evidence supports residual LIF state on SHD and SSC?"
graphify query "Why is active-dendrite research deferred?"
graphify query "What did the LTH itinerary review approve and defer?"
```

Expected anchors:

- Evidence-1 is active before the mechanism research track.
- Residual LIF state is an internal, sample-specific positive result awaiting
  the canonical paired-seed evidence decision.
- Structural plasticity, dual memory, learned delays, and local reward credit
  do not regain authorization merely because Graphify links them.
- Context-specific dendritic supermasks are deferred until Evidence-1 closes.
- Graph-derived `INFERRED` edges are navigation hypotheses; source documents
  and experiment manifests decide claims.

## Historical Obsidian vault

`obsidian_vault/` is retained as a read-only historical snapshot so prior work
is not destroyed. It is excluded from Graphify, is no longer regenerated, and
must not be treated as current project state. Remove it only in a separate
cleanup after the research acceptance contract passes.
