# Graphify knowledge workflow

Status: sole repository knowledge and recall layer as of 2026-10-13.

Graphify replaces the custom Obsidian generator, retriever, and sync utilities.
The vault and its local settings have been removed from the repository.
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
imports, caches, and Graphify's own output. This
keeps the graph focused on reviewable evidence rather than duplicating hundreds
of megabytes of raw artifacts.

## Installation

On Windows PowerShell from the repository root:

```powershell
uv tool install graphifyy==0.9.53
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

The local update path also records Markdown headings and explicit
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
python gen5/tools/verify_graphify_contract.py --require-research --require-corpus --refresh-source-receipt
```

The first command checks implementation coverage. The second is the migration
gate and must find `research.md`, the primary evidence roadmap, and the
LTH-informed itinerary in the graph. It proves source coverage, not the
correctness of model-inferred relationships.

The stronger `--require-corpus` check requires every canonical source file to
have real node provenance, verifies edge endpoints, and binds graph/source
hashes in `graphify-out/source_contract.json`. Refresh that receipt only after
the local graph update, not as a substitute for rebuilding changed sources.
It certifies structural coverage/receipt freshness, not semantic correctness.
Schema 2 normalizes CRLF to LF for canonical text hashes so clean Git checkouts
are portable. Experimental source/cache/ZIP byte checks remain exact and are
not replaced by this text contract.
Some Markdown reviews are classified as papers by Graphify's detector and
skipped by its normal AST watch path. The local Graphify interpreter can run
`gen5/tools/refresh_graphify_markdown.py --force` after `graphify update . --force`
to refresh all canonical Markdown using Graphify's extraction/merge/export APIs,
including changed documents already represented. Without `--force`, only missing
Markdown sources are supplemented; that is not a changed-document refresh.
No API backend or inferred findings are added by that supplement.

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

## Recall across sessions

For project history, current state, research constraints, and code questions,
start with `graphify query "<question>"`. Use `graphify path "<A>" "<B>"` to
trace relationships and `graphify explain "<concept>"` for a focused concept.
Read the returned source locations before treating an answer as research
evidence. If a query is incomplete, refine it using the graph's vocabulary and
then inspect the cited sources. If the graph is unavailable, rebuild it using
the local commands above and report any unresolved coverage gaps.

`AGENTS.md` makes this the persistent agent workflow. The Graphify-aware
`gen5/tools/codex_harness.py` remains the sprint execution wrapper. The former
vault, Obsidian sync/retrieval utilities, and generator are removed; their
committed history remains recoverable through Git. Graphify's optional Obsidian
export is not part of this project's workflow.
