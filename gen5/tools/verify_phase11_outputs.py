r"""Verify Phase 11 evidence packages.

This utility scans one or more folders for the expected Phase 11 artifacts and
prints a compact JSON audit. It intentionally uses only the Python standard
library so it can run locally, in Colab, or inside minimal CI.

Example:

```powershell
python gen5/tools/verify_phase11_outputs.py `
  --roots . "C:\Users\FAISAL TABREZ\Downloads"
```
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ARTIFACT_GROUPS = {
    "champion": [
        "evolution_telemetry.json",
        "champion_connectome.json",
        "champion_sparse_adjacency.json",
        "colab_weights.json",
    ],
    "multi_seed": [
        "multi_seed_trials.json",
        "multi_seed_trials.csv",
        "multi_seed_aggregate.csv",
    ],
    "plasticity_ablation": [
        "plasticity_ablation.json",
        "plasticity_ablation_records.csv",
        "plasticity_ablation_summary.csv",
    ],
    "retention_ablation": [
        "retention_ablation.json",
        "retention_ablation_records.csv",
        "retention_ablation_summary.csv",
    ],
    "throughput": [
        "throughput_results.json",
        "throughput_results.csv",
    ],
    "baselines": [
        "baseline_comparison.json",
        "baseline_comparison.csv",
    ],
}


def main() -> None:
    args = parse_args()
    roots = [Path(root).expanduser().resolve() for root in args.roots]
    index = build_file_index(roots)
    audit = {
        "roots": [str(root) for root in roots],
        "groups": audit_groups(index),
        "summaries": build_summaries(index),
    }
    print(json.dumps(audit, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify AMMC Gen-5 Phase 11 output artifacts")
    parser.add_argument("--roots", nargs="+", default=["."], help="Folders to scan recursively")
    return parser.parse_args()


def build_file_index(roots: list[Path]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            index.setdefault(path.name, []).append(str(path))
    return {name: sorted(paths) for name, paths in sorted(index.items())}


def audit_groups(index: dict[str, list[str]]) -> dict[str, dict[str, Any]]:
    groups = {}
    for group, filenames in ARTIFACT_GROUPS.items():
        present = {name: index.get(name, []) for name in filenames if name in index}
        missing = [name for name in filenames if name not in index]
        groups[group] = {
            "complete": not missing,
            "present": present,
            "missing": missing,
        }
    return groups


def build_summaries(index: dict[str, list[str]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    if "multi_seed_aggregate.csv" in index:
        summaries["multi_seed"] = summarize_multi_seed(Path(index["multi_seed_aggregate.csv"][0]))
    if "plasticity_ablation_summary.csv" in index:
        summaries["plasticity_ablation"] = read_csv_rows(Path(index["plasticity_ablation_summary.csv"][0]))
    if "retention_ablation_summary.csv" in index:
        summaries["retention_ablation"] = read_csv_rows(Path(index["retention_ablation_summary.csv"][0]))
    if "throughput_results.csv" in index:
        summaries["throughput"] = summarize_throughput(Path(index["throughput_results.csv"][0]))
    if "baseline_comparison.csv" in index:
        summaries["baselines"] = summarize_baselines(Path(index["baseline_comparison.csv"][0]))
    if "champion_sparse_adjacency.json" in index and "champion_connectome.json" in index:
        summaries["champion"] = summarize_champion(
            Path(index["champion_sparse_adjacency.json"][0]),
            Path(index["champion_connectome.json"][0]),
        )
    return summaries


def summarize_multi_seed(path: Path) -> dict[str, Any]:
    rows = read_csv_rows(path)
    if not rows:
        return {"records": 0}
    final = rows[-1]
    return {
        "records": len(rows),
        "final_generation": int_float(final.get("generation")),
        "final_mean_best_fitness": int_float(final.get("mean_best_fitness")),
        "final_std_best_fitness": int_float(final.get("std_best_fitness")),
        "final_min_best_fitness": int_float(final.get("min_best_fitness")),
        "final_max_best_fitness": int_float(final.get("max_best_fitness")),
        "final_mean_active_synapses": int_float(final.get("mean_active_synapses")),
    }


def summarize_throughput(path: Path) -> list[dict[str, Any]]:
    rows = read_csv_rows(path)
    return [
        {
            "population_size": int_float(row.get("population_size")),
            "ticks_per_second": int_float(row.get("ticks_per_second")),
            "agent_steps_per_second": int_float(row.get("agent_steps_per_second")),
            "cuda_max_memory_allocated_mb": int_float(row.get("cuda_max_memory_allocated_mb")),
            "torch_compile_active": row.get("torch_compile_active"),
            "compile_error": row.get("compile_error"),
        }
        for row in rows
    ]


def summarize_baselines(path: Path) -> list[dict[str, Any]]:
    rows = read_csv_rows(path)
    return [
        {
            "name": row.get("name"),
            "status": row.get("status"),
            "active_parameters": int_float(row.get("active_parameters")),
            "total_parameters": int_float(row.get("total_parameters")),
            "parameter_memory_mb": int_float(row.get("parameter_memory_mb")),
            "agent_steps_per_second": int_float(row.get("agent_steps_per_second")),
            "max_fitness": int_float(row.get("max_fitness")),
            "notes": row.get("notes"),
        }
        for row in rows
    ]


def summarize_champion(adjacency_path: Path, connectome_path: Path) -> dict[str, Any]:
    adjacency = json.loads(adjacency_path.read_text(encoding="utf-8"))
    connectome = json.loads(connectome_path.read_text(encoding="utf-8"))
    return {
        "fitness": adjacency.get("fitness"),
        "neuron_count": adjacency.get("neuron_count"),
        "active_edges": adjacency.get("active_edges"),
        "adjacency_rows": len(adjacency.get("rows", [])),
        "connectome_neurons": len(connectome.get("neurons", [])),
        "connectome_synapses": len(connectome.get("synapses", [])),
        "edge_count_consistent": len(adjacency.get("rows", [])) == len(connectome.get("synapses", [])),
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def int_float(value: Any) -> int | float | None:
    if value in (None, ""):
        return None
    number = float(value)
    return int(number) if number.is_integer() else number


if __name__ == "__main__":
    main()
