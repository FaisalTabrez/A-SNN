"""Sprint 15: frozen readout adapter variant sweep.

Runs multiple readout/transducer adapter variants on the same frozen AMMC tasks
and seed so the results are directly comparable.

Default variants:

- `linear/full_trace`
- `linear/motor_trace`
- `mlp/full_trace`

Colab-scale run:

```python
!python gen5/examples/sprint15_frozen_readout_adapter_sweep.py \
  --device cuda \
  --sample-count 4096 \
  --timesteps 8 \
  --neuron-count 16 \
  --max-edges 128 \
  --epochs 200 \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/frozen_readout_adapter_sweep_cuda
```
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from dataclasses import asdict
from typing import Iterable

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5 import (  # noqa: E402
    FrozenReadoutAdapterConfig,
    FrozenReadoutAdapterRunner,
    FrozenTaskConfig,
    available_frozen_tasks,
)


DEFAULT_VARIANTS = ("linear:full_trace", "linear:motor_trace", "mlp:full_trace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AMMC Gen-5 frozen readout adapter variant sweep")
    parser.add_argument("--tasks", nargs="+", default=list(available_frozen_tasks()), choices=available_frozen_tasks())
    parser.add_argument("--list-tasks", action="store_true", help="Print available synthetic tasks and exit")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_VARIANTS),
        help="Adapter variants as adapter_kind:feature_mode, e.g. linear:full_trace mlp:full_trace",
    )
    parser.add_argument("--sample-count", type=int, default=4096)
    parser.add_argument("--timesteps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neuron-count", type=int, default=16)
    parser.add_argument("--max-edges", type=int, default=128)
    parser.add_argument("--sensor-gain", type=float, default=1.0)
    parser.add_argument("--leak", type=float, default=0.9)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--input-amplitude", type=float, default=0.75)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--hidden-units", type=int, default=32)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="gen5_outputs/frozen_readout_adapter_sweep")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_tasks:
        print(json.dumps(list(available_frozen_tasks()), indent=2))
        return

    variants = [_parse_variant(raw) for raw in args.variants]
    task_config = FrozenTaskConfig(
        tasks=tuple(args.tasks),
        sample_count=args.sample_count,
        timesteps=args.timesteps,
        seed=args.seed,
        neuron_count=args.neuron_count,
        max_edges=args.max_edges,
        sensor_gain=args.sensor_gain,
        leak=args.leak,
        threshold=args.threshold,
        input_amplitude=args.input_amplitude,
        device=args.device,
    )

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variant_results = []
    rows = []
    for adapter_kind, feature_mode in variants:
        variant_name = f"{adapter_kind}_{feature_mode}"
        runner = FrozenReadoutAdapterRunner(
            FrozenReadoutAdapterConfig(
                task_config=task_config,
                train_fraction=args.train_fraction,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                adapter_kind=adapter_kind,
                hidden_units=args.hidden_units,
                feature_mode=feature_mode,
                standardize_features=not args.no_standardize,
            )
        )
        result = runner.run()
        paths = runner.save_outputs(result, output_dir / variant_name, plot=not args.no_plot)
        variant_results.append(
            {
                "variant": variant_name,
                "adapter_kind": adapter_kind,
                "feature_mode": feature_mode,
                "paths": paths,
                "summary": [asdict(row) for row in result.summary],
            }
        )
        for row in result.summary:
            record = asdict(row)
            record["variant"] = variant_name
            rows.append(record)

    json_path = output_dir / "frozen_readout_adapter_sweep.json"
    json_path.write_text(
        json.dumps(
            {
                "config": {
                    "tasks": list(task_config.tasks),
                    "sample_count": task_config.sample_count,
                    "timesteps": task_config.timesteps,
                    "seed": task_config.seed,
                    "neuron_count": task_config.neuron_count,
                    "max_edges": task_config.max_edges,
                    "variants": list(args.variants),
                    "epochs": args.epochs,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "hidden_units": args.hidden_units,
                    "standardize_features": not args.no_standardize,
                    "device": args.device,
                },
                "results": variant_results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    summary_csv = output_dir / "frozen_readout_adapter_sweep_summary.csv"
    _write_csv(summary_csv, rows)

    paths = {
        "json": str(json_path),
        "summary_csv": str(summary_csv),
    }
    if not args.no_plot:
        try:
            plot_path = output_dir / "frozen_readout_adapter_sweep_summary.png"
            _plot_sweep(rows, plot_path)
            paths["plot"] = str(plot_path)
        except Exception as exc:  # pragma: no cover - optional plotting
            paths["plot"] = f"skipped: {exc}"

    print(json.dumps({"paths": paths, "summary": rows}, indent=2))


def _parse_variant(raw: str) -> tuple[str, str]:
    try:
        adapter_kind, feature_mode = raw.split(":", 1)
    except ValueError as exc:
        raise ValueError(f"variant must be adapter_kind:feature_mode, got {raw!r}") from exc
    if adapter_kind not in {"linear", "mlp"}:
        raise ValueError(f"unknown adapter_kind in variant {raw!r}")
    if feature_mode not in {"full_trace", "motor_trace"}:
        raise ValueError(f"unknown feature_mode in variant {raw!r}")
    return adapter_kind, feature_mode


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = ["variant"] + [key for key in rows[0].keys() if key != "variant"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_sweep(rows: Iterable[dict], output_path: pathlib.Path) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    rows = list(rows)
    tasks = list(dict.fromkeys(row["task"] for row in rows))
    variants = list(dict.fromkeys(row["variant"] for row in rows))
    x = list(range(len(tasks)))
    width = min(0.8 / max(1, len(variants)), 0.25)

    fig, ax = plt.subplots(figsize=(max(9, len(tasks) * 1.7), 5))
    for idx, variant in enumerate(variants):
        values = []
        for task in tasks:
            match = next(row for row in rows if row["task"] == task and row["variant"] == variant)
            values.append(match["adapter_accuracy"])
        offset = (idx - (len(variants) - 1) / 2.0) * width
        ax.bar([i + offset for i in x], values, width, label=variant)

    ax.axhline(0.25, color="gray", linestyle="--", linewidth=1, label="Random 4-way chance")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Adapter accuracy")
    ax.set_title("AMMC Gen-5 frozen readout adapter sweep")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
