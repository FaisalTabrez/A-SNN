## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## A-SNN evidence rules

- `research.md`, `gen5/docs/`, retained `analysis.md` files, and experiment
  manifests are authoritative. Graphify is a derived retrieval layer.
- Treat `INFERRED` and `AMBIGUOUS` graph edges as leads to verify against the
  cited source, never as validated research findings.
- The active program remains Evidence-1 until its registered decision is
  locked. Graph proximity does not reopen a closed mechanism claim.
- Use the corpus and validation contract in
  `docs/GRAPHIFY_KNOWLEDGE_WORKFLOW.md`.
- After documentation changes, do not claim that the graph is current until
  `graphify update . --force` and
  `verify_graphify_contract.py --require-research` pass. Optional semantic
  enrichment requires an explicitly approved backend.

## A-SNN evidence rules

- `research.md`, `gen5/docs/`, retained `analysis.md` files, and experiment
  manifests are authoritative. Graphify is a derived retrieval layer.
- Treat `INFERRED` and `AMBIGUOUS` graph edges as leads to verify against the
  cited source, never as validated research findings.
- The active program remains Evidence-1 until its registered decision is
  locked. Graph proximity does not reopen a closed mechanism claim.
- Use the corpus and validation contract in
  `docs/GRAPHIFY_KNOWLEDGE_WORKFLOW.md`.
- After documentation changes, do not claim that the graph is current until a
  semantic update with an explicitly approved backend passes
  `verify_graphify_contract.py --require-research`.
