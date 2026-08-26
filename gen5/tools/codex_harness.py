#!/usr/bin/env python3
"""Run one sprint with Graphify context and refresh the local code graph."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = REPOSITORY_ROOT / "graphify-out" / "graph.json"


def query_graph(question: str, graph_path: Path) -> None:
    if not graph_path.exists():
        print(f"Graphify graph not found at {graph_path}; continuing without prior graph context.")
        return
    if shutil.which("graphify") is None:
        print("Graphify is not on PATH; continuing without prior graph context.")
        return
    subprocess.run(
        [
            "graphify",
            "query",
            question,
            "--graph",
            str(graph_path),
            "--budget",
            "1500",
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
    )


def run_harness(args: argparse.Namespace) -> int:
    script_path = args.script_path.resolve()
    graph_path = args.graph_path.resolve()
    print("=== Step 1: Querying Graphify Knowledge Graph ===")
    query_graph(
        args.context_query or f"What prior evidence and decisions constrain {args.sprint_id}?",
        graph_path,
    )
    print("\n=== Step 2: Formulating Execution Context ===")
    print(f"Historical context retrieved; executing {script_path.relative_to(REPOSITORY_ROOT)}.")
    print(f"\n=== Step 3: Executing Sprint Script ({script_path}) ===")
    result = subprocess.run([sys.executable, str(script_path), *args.script_args], cwd=REPOSITORY_ROOT)
    if result.returncode:
        print(f"Execution failed with return code {result.returncode}; metrics were not synced.", file=sys.stderr)
        return result.returncode
    metrics_path = args.metrics_json or REPOSITORY_ROOT / "gen5" / "outputs" / args.run_id / "metrics.json"
    metrics_path = metrics_path.resolve()
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8-sig")) if metrics_path.exists() else {}
    except json.JSONDecodeError as exc:
        print(f"Metrics file is invalid JSON: {exc}", file=sys.stderr)
        return 2
    if not isinstance(metrics, dict):
        print("Metrics file must contain a JSON object.", file=sys.stderr)
        return 2
    print("\n=== Step 4: Reporting Metrics ===")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if graph_path.exists() and not args.skip_graph_update and shutil.which("graphify"):
        print("\n=== Step 5: Refreshing Graphify Code Graph ===")
        update = subprocess.run(["graphify", "update", "."], cwd=REPOSITORY_ROOT)
        if update.returncode:
            print(
                "Sprint completed, but Graphify update failed. Refresh it manually before relying on the graph.",
                file=sys.stderr,
            )
            return update.returncode
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute a Gen-5 sprint with Graphify context.")
    parser.add_argument("--sprint-id", required=True)
    parser.add_argument("--script-path", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--metrics-json", type=Path)
    parser.add_argument("--graph-path", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--context-query")
    parser.add_argument("--skip-graph-update", action="store_true")
    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments passed through to the sprint script (prefix with --).")
    return run_harness(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
