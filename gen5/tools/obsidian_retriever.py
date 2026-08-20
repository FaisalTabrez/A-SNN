#!/usr/bin/env python3
"""Retrieve compact, prompt-ready historical context from the Obsidian vault."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.DOTALL)


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        return [item.strip().strip('"') for item in value[1:-1].split(",") if item.strip()]
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value.strip('"')


def parse_frontmatter(file_path: Path) -> tuple[dict[str, Any], str]:
    content = file_path.read_text(encoding="utf-8")
    match = FRONTMATTER.match(content)
    if not match:
        return {}, content
    metadata: dict[str, Any] = {}
    active_map: dict[str, Any] | None = None
    for line in match.group(1).splitlines():
        if line.startswith("  ") and active_map is not None and ":" in line:
            key, value = line.strip().split(":", 1)
            active_map[key] = _parse_scalar(value)
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if not value:
            active_map = {}
            metadata[key] = active_map
        else:
            metadata[key] = _parse_scalar(value)
            active_map = None
    return metadata, match.group(2)


def _summary(body: str) -> str:
    match = re.search(r"## Executive Summary\s*\n(.*?)(?=\n## |\Z)", body, re.DOTALL)
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""


def query_vault_context(vault_path: Path, sprint_query: str | None = None, tags: list[str] | None = None) -> str:
    experiments_dir = vault_path / "Experiments"
    if not experiments_dir.exists():
        return "No prior experiment context found in Obsidian vault."
    matches: list[tuple[Path, dict[str, Any], str]] = []
    requested_tags = set(tags or [])
    for note in sorted(experiments_dir.glob("*.md"), key=lambda path: path.stat().st_mtime, reverse=True):
        metadata, body = parse_frontmatter(note)
        if sprint_query and sprint_query.casefold() not in str(metadata.get("sprint_id", "")).casefold():
            continue
        note_tags = set(metadata.get("tags", [])) if isinstance(metadata.get("tags"), list) else set()
        if requested_tags and not requested_tags.intersection(note_tags):
            continue
        matches.append((note, metadata, body))
    if not matches:
        return f"No past experiments found matching query: sprint={sprint_query!r}, tags={tags!r}"
    lines = ["### Obsidian Knowledge Base Context", f"Found {len(matches)} relevant historical experiment(s):"]
    for note, metadata, body in matches[:5]:
        lines.extend([
            f"#### Note: [[{note.stem}]]",
            f"- Sprint: {metadata.get('sprint_id', 'N/A')}",
            f"- Architecture: {metadata.get('architecture', 'N/A')}",
            f"- Hypothesis: {metadata.get('hypothesis', 'N/A')}",
            f"- Key Metrics: {metadata.get('metrics', {})}",
        ])
        if summary := _summary(body):
            lines.append(f"- Summary: {summary}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieve context from the Obsidian vault.")
    parser.add_argument("--vault-path", type=Path, default=REPOSITORY_ROOT / "obsidian_vault")
    parser.add_argument("--sprint")
    parser.add_argument("--tags", nargs="+")
    args = parser.parse_args()
    print(query_vault_context(args.vault_path, args.sprint, args.tags))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
