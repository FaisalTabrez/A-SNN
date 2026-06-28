"""Sprint 14: harder bot-world benchmark.

The simple foraging world no longer rewards extra hidden decision nodes. This
runner keeps the sparse-efficiency finalists fixed and varies the world instead:
larger arenas, sparse cues, moving toxins, delayed reward, and a combined
gauntlet.

Colab-scale run:

```python
!python gen5/examples/sprint14_harder_worlds.py \
  --device cuda \
  --worlds simple moving_toxins delayed_reward gauntlet \
  --groups low_ltw_pruning gentle_ltw_scheduled \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --generations 500 \
  --population-size 10000 \
  --epoch-steps 120 \
  --output-dir gen5_outputs/harder_worlds_cuda
```
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from dataclasses import asdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5 import (  # noqa: E402
    NeuronScalePoint,
    SparseEfficiencyConfig,
    SparseEfficiencyRunner,
    available_world_presets,
    default_sparse_efficiency_groups,
    world_preset_config,
    world_preset_names,
)


DEFAULT_WORLDS = ("simple", "moving_toxins", "delayed_reward", "gauntlet")
DEFAULT_GROUPS = ("low_ltw_pruning", "gentle_ltw_scheduled")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AMMC Gen-5 harder-world benchmark")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    parser.add_argument("--generations", type=int, default=500)
    parser.add_argument("--epoch-steps", type=int, default=120)
    parser.add_argument("--population-size", type=int, default=10_000)
    parser.add_argument("--worlds", nargs="+", default=list(DEFAULT_WORLDS), choices=world_preset_names())
    parser.add_argument("--list-worlds", action="store_true", help="Print available world presets and exit")
    parser.add_argument("--food-count", type=int, default=128)
    parser.add_argument("--toxin-count", type=int, default=128)
    parser.add_argument("--sensor-radius", type=float, default=None)
    parser.add_argument("--friction", type=float, default=None)
    parser.add_argument("--action-gain", type=float, default=None)
    parser.add_argument("--world-size", type=float, default=None)
    parser.add_argument("--max-speed", type=float, default=None)
    parser.add_argument("--collision-radius", type=float, default=None)
    parser.add_argument("--respawn-margin", type=float, default=None)
    parser.add_argument("--food-reward", type=float, default=None)
    parser.add_argument("--toxin-penalty", type=float, default=None)
    parser.add_argument("--reward-delay-steps", type=int, default=None)
    parser.add_argument("--punishment-delay-steps", type=int, default=None)
    parser.add_argument("--moving-food-speed", type=float, default=None)
    parser.add_argument("--moving-toxin-speed", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=25.0)
    parser.add_argument("--reference-max-edges", type=int, default=128)
    parser.add_argument("--protected-core-edge-count", type=int, default=None)
    parser.add_argument(
        "--groups",
        nargs="+",
        default=list(DEFAULT_GROUPS),
        help="Sparse-efficiency finalist groups to compare.",
    )
    parser.add_argument("--list-groups", action="store_true", help="Print available group names and exit")
    parser.add_argument("--checkpoint-every-trials", type=int, default=1)
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--neuron-counts", nargs="+", type=int, default=[32])
    parser.add_argument("--max-edges", nargs="+", type=int, default=[256])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="gen5_outputs/harder_worlds")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    available_groups = default_sparse_efficiency_groups()
    if args.list_groups:
        print(json.dumps([group.__dict__ for group in available_groups], indent=2))
        return
    if args.list_worlds:
        print(json.dumps([asdict(preset) for preset in available_world_presets()], indent=2))
        return
    if len(args.neuron_counts) != len(args.max_edges):
        raise SystemExit("--neuron-counts and --max-edges must have the same length")

    groups = select_groups(args.groups, available_groups)
    scale_points = tuple(
        NeuronScalePoint(neuron_count=neurons, max_edges=max_edges)
        for neurons, max_edges in zip(args.neuron_counts, args.max_edges)
    )
    output_root = pathlib.Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    combined_records: list[dict] = []
    combined_summary: list[dict] = []
    world_paths: dict[str, dict[str, str]] = {}
    world_configs: dict[str, dict] = {}

    for world in args.worlds:
        env_config = build_env_config_for_report(args, world)
        world_configs[world] = asdict(env_config)
        print(f"world={world} config={world_configs[world]}", flush=True)

        runner = SparseEfficiencyRunner(
            SparseEfficiencyConfig(
                seeds=tuple(args.seeds),
                generations=args.generations,
                epoch_steps=args.epoch_steps,
                population_size=args.population_size,
                world_preset=world,
                food_count=args.food_count,
                toxin_count=args.toxin_count,
                sensor_radius=args.sensor_radius,
                friction=args.friction,
                action_gain=args.action_gain,
                world_size=args.world_size,
                max_speed=args.max_speed,
                collision_radius=args.collision_radius,
                respawn_margin=args.respawn_margin,
                food_reward=args.food_reward,
                toxin_penalty=args.toxin_penalty,
                reward_delay_steps=args.reward_delay_steps,
                punishment_delay_steps=args.punishment_delay_steps,
                moving_food_speed=args.moving_food_speed,
                moving_toxin_speed=args.moving_toxin_speed,
                reference_max_edges=args.reference_max_edges,
                adaptation_fitness_threshold=args.threshold,
                device=args.device,
                scale_points=scale_points,
                groups=groups,
                protected_core_edge_count=args.protected_core_edge_count,
            )
        )
        world_output = output_root / world
        if args.no_checkpoint:
            result = runner.run()
            paths = runner.save_outputs(result, world_output, plot=not args.no_plot)
        else:
            result, paths = runner.run_with_checkpoints(
                world_output,
                plot=not args.no_plot,
                checkpoint_every_trials=args.checkpoint_every_trials,
            )
        world_paths[world] = paths
        combined_records.extend({"world": world, **asdict(row)} for row in result.records)
        combined_summary.extend({"world": world, **asdict(row)} for row in result.summary)
        save_combined_outputs(
            output_root,
            args=args,
            scale_points=scale_points,
            groups=groups,
            world_configs=world_configs,
            world_paths=world_paths,
            records=combined_records,
            summary=combined_summary,
        )

    paths = save_combined_outputs(
        output_root,
        args=args,
        scale_points=scale_points,
        groups=groups,
        world_configs=world_configs,
        world_paths=world_paths,
        records=combined_records,
        summary=combined_summary,
    )
    print(json.dumps({"paths": paths, "summary": combined_summary}, indent=2))


def build_env_config_for_report(args: argparse.Namespace, world: str):
    return world_preset_config(
        world,
        agent_count=args.population_size,
        food_count=args.food_count,
        toxin_count=args.toxin_count,
        sensor_radius=args.sensor_radius,
        friction=args.friction,
        action_gain=args.action_gain,
        world_size=args.world_size,
        max_speed=args.max_speed,
        collision_radius=args.collision_radius,
        respawn_margin=args.respawn_margin,
        food_reward=args.food_reward,
        toxin_penalty=args.toxin_penalty,
        reward_delay_steps=args.reward_delay_steps,
        punishment_delay_steps=args.punishment_delay_steps,
        moving_food_speed=args.moving_food_speed,
        moving_toxin_speed=args.moving_toxin_speed,
    )


def save_combined_outputs(
    output_root: pathlib.Path,
    *,
    args: argparse.Namespace,
    scale_points,
    groups,
    world_configs: dict[str, dict],
    world_paths: dict[str, dict[str, str]],
    records: list[dict],
    summary: list[dict],
) -> dict[str, str]:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "harder_worlds.json"
    records_csv = output_root / "harder_worlds_records.csv"
    summary_csv = output_root / "harder_worlds_summary.csv"
    progress_path = output_root / "harder_worlds_progress.json"

    payload = {
        "config": {
            "seeds": args.seeds,
            "generations": args.generations,
            "epoch_steps": args.epoch_steps,
            "population_size": args.population_size,
            "worlds": args.worlds,
            "groups": [group.name for group in groups],
            "scale_points": [point.__dict__ for point in scale_points],
            "threshold": args.threshold,
            "device": args.device,
        },
        "world_configs": world_configs,
        "world_paths": world_paths,
        "records": records,
        "summary": summary,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_csv(records_csv, records)
    write_csv(summary_csv, summary)
    progress_path.write_text(
        json.dumps(
            {
                "completed_worlds": list(world_paths),
                "total_worlds": len(args.worlds),
                "paths": {
                    "json": str(json_path),
                    "records_csv": str(records_csv),
                    "summary_csv": str(summary_csv),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "json": str(json_path),
        "records_csv": str(records_csv),
        "summary_csv": str(summary_csv),
        "progress": str(progress_path),
    }


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_groups(group_names: list[str] | None, available_groups):
    if not group_names:
        return tuple(available_groups)
    by_name = {group.name: group for group in available_groups}
    missing = [name for name in group_names if name not in by_name]
    if missing:
        raise SystemExit(f"unknown sparse-efficiency group(s): {', '.join(missing)}")
    return tuple(by_name[name] for name in group_names)


if __name__ == "__main__":
    main()
