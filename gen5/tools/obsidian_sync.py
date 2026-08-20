#!/usr/bin/env python3
"""Write a Gen-5 experiment artifact into the repository Obsidian vault."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def get_git_commit_hash(repository_root: Path = REPOSITORY_ROOT) -> str:
    """Return the current short revision without making the logger depend on git."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repository_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _yaml_scalar(value: Any) -> str:
    """Serialize JSON-compatible values safely for simple YAML frontmatter."""
    return json.dumps(value, ensure_ascii=False)


def _sprint_note_name(sprint_id: str) -> str:
    suffix = sprint_id.removeprefix("sprint").removeprefix("Sprint")
    return f"Sprint-{suffix}" if suffix else sprint_id


def create_experiment_note(
    vault_path: Path,
    run_id: str,
    sprint_id: str,
    script_name: str,
    dataset: str,
    architecture: str,
    hypothesis: str,
    metrics: dict[str, Any],
    summary: str = "",
) -> Path:
    """Create or replace the deterministic note for one run and return its path."""
    experiments_dir = vault_path / "Experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    note_file = experiments_dir / f"EXP-{sprint_id.upper()}-{run_id}.md"
    metrics_yaml = "\n".join(f"  {key}: {_yaml_scalar(value)}" for key, value in metrics.items())
    if not metrics_yaml:
        metrics_yaml = "  {}"
    sprint_note = _sprint_note_name(sprint_id)
    content = f"""---
type: experiment
run_id: {_yaml_scalar(run_id)}
sprint_id: {_yaml_scalar(sprint_id)}
script: {_yaml_scalar(script_name)}
dataset: "[[{dataset}]]"
architecture: "[[{architecture}]]"
hypothesis: "[[{hypothesis}]]"
status: completed
date: {date.today().isoformat()}
metrics:
{metrics_yaml}
commit_hash: {_yaml_scalar(get_git_commit_hash())}
tags: [experiment, {sprint_id}, gen5]
---

# Experiment Log: {sprint_id.upper()} — Run {run_id}

## Executive Summary
{summary or "Automated experiment run logged by the Codex harness."}

## Context & Graph Connections
- Parent Sprint: [[{sprint_note}]]
- Network Topology: [[{architecture}]]
- Benchmark Target: [[{dataset}]]
- Tested Hypothesis: [[{hypothesis}]]

## Metrics Summary
"""
    content += "\n".join(f"- **{key}**: `{value}`" for key, value in metrics.items()) or "- No metrics artifact was found."
    note_file.write_text(content + "\n", encoding="utf-8")
    return note_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync sprint metrics to the Obsidian vault.")
    parser.add_argument("--vault-path", type=Path, default=REPOSITORY_ROOT / "obsidian_vault")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sprint-id", required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--script-name", required=True)
    parser.add_argument("--dataset", default="Spiking-Heidelberg-Digits")
    parser.add_argument("--architecture", default="Analog-Leaky-Topology")
    parser.add_argument("--hypothesis", default="Sparse-Width-Scaling")
    parser.add_argument("--summary", default="")
    args = parser.parse_args()
    try:
        metrics = json.loads(args.metrics_json.read_text(encoding="utf-8-sig")) if args.metrics_json.exists() else {}
    except json.JSONDecodeError as exc:
        parser.error(f"metrics JSON is invalid: {exc}")
    if not isinstance(metrics, dict):
        parser.error("metrics JSON must contain an object")
    note = create_experiment_note(
        vault_path=args.vault_path,
        run_id=args.run_id,
        sprint_id=args.sprint_id,
        script_name=args.script_name,
        dataset=args.dataset,
        architecture=args.architecture,
        hypothesis=args.hypothesis,
        metrics=metrics,
        summary=args.summary,
    )
    print(f"[obsidian_sync] Logged experiment note: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
