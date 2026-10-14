#!/usr/bin/env python3
"""Verify that the shared Graphify graph covers A-SNN's canonical sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = REPOSITORY_ROOT / "graphify-out" / "graph.json"

IMPLEMENTATION_ANCHORS = (
    "gen5/ammc_gen5/",
    "gen5/examples/",
    "gen5/tests/",
)

RESEARCH_ANCHORS = (
    "research.md",
    "gen5/docs/primary_evidence_track_roadmap.md",
    "gen5/docs/lth_informed_itinerary_review.md",
)

REMOVED_KNOWLEDGE_SOURCES = (
    "obsidian_vault/",
    "gen5/tools/obsidian_sync.py",
    "gen5/tools/obsidian_retriever.py",
    "gen5/tools/build_obsidian_graph.py",
    "gen5/tests/test_obsidian_tools_contract.py",
)


def _nodes(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("graph.json must contain a JSON object")
    candidates = payload.get("nodes")
    if candidates is None and isinstance(payload.get("graph"), dict):
        candidates = payload["graph"].get("nodes")
    if not isinstance(candidates, list):
        raise ValueError("graph.json does not contain a node list")
    return [node for node in candidates if isinstance(node, dict)]


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _strings(nested)


def graph_strings(payload: Any) -> set[str]:
    return {
        text.replace("\\", "/").lower()
        for node in _nodes(payload)
        for text in _strings(node)
        if text
    }


def missing_anchors(strings: set[str], anchors: Iterable[str]) -> list[str]:
    return [
        anchor
        for anchor in anchors
        if not any(anchor.lower() in candidate for candidate in strings)
    ]


def verify_graph(graph_path: Path, *, require_research: bool = False) -> dict[str, Any]:
    payload = json.loads(graph_path.read_text(encoding="utf-8-sig"))
    strings = graph_strings(payload)
    missing_implementation = missing_anchors(strings, IMPLEMENTATION_ANCHORS)
    missing_research = missing_anchors(strings, RESEARCH_ANCHORS) if require_research else []
    stale_sources = sorted({
        source.replace("\\", "/")
        for node in _nodes(payload)
        for field in ("source_file", "path")
        if isinstance(source := node.get(field), str)
        if any(source.replace("\\", "/").lower().startswith(removed)
               for removed in REMOVED_KNOWLEDGE_SOURCES)
    })
    return {
        "graph": str(graph_path),
        "node_count": len(_nodes(payload)),
        "require_research": require_research,
        "missing_implementation": missing_implementation,
        "missing_research": missing_research,
        "stale_knowledge_sources": stale_sources,
        "passed": not missing_implementation and not missing_research and not stale_sources,
    }


def canonical_files(root: Path) -> list[Path]:
    files = [root / name for name in ['research.md', 'README.md'] if (root / name).is_file()]
    for folder in ['docs', 'gen5/docs']:
        files.extend((root / folder).rglob('*.md'))
    files.extend((root / 'gen5/outputs').rglob('analysis.md'))
    for folder in ['gen5/ammc_gen5', 'gen5/examples', 'gen5/tools', 'gen5/tests', 'gen5/benchmarks']:
        files.extend((root / folder).rglob('*.py'))
    return sorted(set(files))


def verify_corpus(graph_path: Path, root: Path, *, refresh_receipt=False) -> dict[str, Any]:
    """Structural provenance/freshness only, never inferred-claim validation."""
    root = root.resolve()
    payload = json.loads(graph_path.read_text(encoding='utf-8-sig'))
    represented = set()
    for node in _nodes(payload):
        if not isinstance(node.get('source_file'), str): continue
        path = Path(node['source_file'].replace('\\', '/'))
        if path.is_absolute():
            try: path = path.relative_to(root)
            except ValueError: continue
        represented.add(path.as_posix().lower())
    files = canonical_files(root)
    # Git checkouts can normalize CRLF without changing source text. This receipt
    # is a canonical-text contract, not a replacement for experiment byte hashes.
    hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in files}
    missing = [name for name in hashes if name.lower() not in represented]
    ids = {n.get('id') for n in _nodes(payload)}
    dangling = [str(e.get('source')) + ' -> ' + str(e.get('target')) for e in payload.get('links', payload.get('edges', []))
                if e.get('source') not in ids or e.get('target') not in ids]
    receipt_path = graph_path.parent / 'source_contract.json'
    current = {'schema': 2, 'hash_encoding': 'UTF-8 source bytes with CRLF normalized to LF',
               'claim_boundary': 'structural coverage and receipt freshness; semantics unverified',
               'graph_sha256': hashlib.sha256(graph_path.read_bytes().replace(b'\r\n', b'\n')).hexdigest(), 'source_sha256': hashes}
    if refresh_receipt and not missing and not dangling:
        receipt_path.write_text(json.dumps(current, indent=2) + '\n', encoding='utf-8')
    saved = json.loads(receipt_path.read_text(encoding='utf-8')) if receipt_path.is_file() else None
    return {'canonical_file_count': len(files), 'missing_corpus_sources': missing,
            'dangling_edge_count': len(dangling), 'receipt_matches_current_graph_and_sources': saved == current,
            'semantic_freshness_verified': False,
            'passed': not missing and not dangling and saved == current}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--require-research", action="store_true")
    parser.add_argument('--require-corpus', action='store_true')
    parser.add_argument('--refresh-source-receipt', action='store_true')
    args = parser.parse_args()

    if not args.graph.exists():
        print(json.dumps({"passed": False, "error": f"Graph not found: {args.graph}"}, indent=2))
        return 2
    try:
        result = verify_graph(args.graph, require_research=args.require_research)
        if args.require_corpus or args.refresh_source_receipt:
            result['corpus'] = verify_corpus(args.graph, REPOSITORY_ROOT, refresh_receipt=args.refresh_source_receipt)
            result['passed'] = result['passed'] and result['corpus']['passed']
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
