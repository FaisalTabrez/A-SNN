#!/usr/bin/env python3
"""Verify that the shared Graphify graph covers A-SNN's canonical sources."""

from __future__ import annotations

import argparse
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
    return {
        "graph": str(graph_path),
        "node_count": len(_nodes(payload)),
        "require_research": require_research,
        "missing_implementation": missing_implementation,
        "missing_research": missing_research,
        "passed": not missing_implementation and not missing_research,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--require-research", action="store_true")
    args = parser.parse_args()

    if not args.graph.exists():
        print(json.dumps({"passed": False, "error": f"Graph not found: {args.graph}"}, indent=2))
        return 2
    try:
        result = verify_graph(args.graph, require_research=args.require_research)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
