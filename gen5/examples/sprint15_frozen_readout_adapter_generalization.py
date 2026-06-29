"""Sprint 15: frozen readout adapter held-out-seed generalization.

The adapter sweep showed that an MLP readout can solve all current synthetic
tasks from frozen AMMC traces. This runner asks the next harder question:

    Does the trained readout generalize to new synthetic seeds without retraining?

It trains one adapter per task on `--train-seed`, evaluates the held-out split
from that seed, then evaluates fresh full batches from `--test-seeds`.

Colab-scale run:

```python
!python gen5/examples/sprint15_frozen_readout_adapter_generalization.py \
  --device cuda \
  --adapter-kind mlp \
  --feature-mode full_trace \
  --sample-count 4096 \
  --timesteps 8 \
  --neuron-count 16 \
  --max-edges 128 \
  --epochs 200 \
  --test-seeds 43 44 45 46 47 \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/frozen_readout_adapter_generalization_cuda
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
    make_generator,
    mark_step,
    resolve_device,
    seed_everything,
    sync,
)

try:  # pragma: no cover - exercised in accelerator runtimes
    import torch
except Exception:  # pragma: no cover
    torch = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate frozen readout adapter generalization on held-out seeds")
    parser.add_argument("--tasks", nargs="+", default=list(available_frozen_tasks()), choices=available_frozen_tasks())
    parser.add_argument("--list-tasks", action="store_true", help="Print available synthetic tasks and exit")
    parser.add_argument("--sample-count", type=int, default=4096)
    parser.add_argument("--timesteps", type=int, default=8)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--test-seeds", nargs="+", type=int, default=[43, 44, 45, 46, 47])
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
    parser.add_argument("--adapter-kind", choices=("linear", "mlp"), default="mlp")
    parser.add_argument("--feature-mode", choices=("full_trace", "motor_trace"), default="full_trace")
    parser.add_argument("--hidden-units", type=int, default=32)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="gen5_outputs/frozen_readout_adapter_generalization")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_tasks:
        print(json.dumps(list(available_frozen_tasks()), indent=2))
        return
    if torch is None:
        raise ImportError("frozen readout adapter generalization requires PyTorch")

    task_config = FrozenTaskConfig(
        tasks=tuple(args.tasks),
        sample_count=args.sample_count,
        timesteps=args.timesteps,
        seed=args.train_seed,
        neuron_count=args.neuron_count,
        max_edges=args.max_edges,
        sensor_gain=args.sensor_gain,
        leak=args.leak,
        threshold=args.threshold,
        input_amplitude=args.input_amplitude,
        device=args.device,
    )
    adapter_config = FrozenReadoutAdapterConfig(
        task_config=task_config,
        train_fraction=args.train_fraction,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adapter_kind=args.adapter_kind,
        feature_mode=args.feature_mode,
        hidden_units=args.hidden_units,
        standardize_features=not args.no_standardize,
    )
    runner = FrozenReadoutAdapterRunner(adapter_config)
    device = resolve_device(args.device)
    seed_everything(args.train_seed, device=device)
    split_generator = make_generator(args.train_seed + 40_000, device=device)

    rows: list[dict] = []
    for offset, task_name in enumerate(task_config.tasks):
        train_generator = make_generator(args.train_seed + offset + 1, device=device)
        train_batch = runner.task_runner._make_task(task_name, train_generator, device)
        train_trace = runner.task_runner._frozen_ammc_trace(train_batch.inputs, device)
        train_features = runner._select_features(train_trace).detach()
        train_targets = train_batch.targets

        split = _split_features(
            train_features,
            train_targets,
            args.train_fraction,
            split_generator,
            device,
        )
        adapter, final_loss = _train_adapter(
            runner,
            split["x_train"],
            split["y_train"],
            device,
            args.epochs,
            args.learning_rate,
            args.weight_decay,
        )

        rows.append(
            _evaluate_batch(
                adapter=adapter,
                runner=runner,
                task_name=task_name,
                batch=train_batch,
                trace=train_trace,
                features=split["x_test_raw"],
                targets=split["y_test"],
                feature_mean=split["mean"],
                feature_std=split["std"],
                eval_scope="train_seed_split",
                train_seed=args.train_seed,
                eval_seed=args.train_seed,
                final_train_loss=final_loss,
                baseline_indices=split["test_idx"],
            )
        )

        for test_seed in args.test_seeds:
            test_generator = make_generator(test_seed + offset + 1, device=device)
            test_batch = runner.task_runner._make_task(task_name, test_generator, device)
            test_trace = runner.task_runner._frozen_ammc_trace(test_batch.inputs, device)
            test_features = runner._select_features(test_trace).detach()
            rows.append(
                _evaluate_batch(
                    adapter=adapter,
                    runner=runner,
                    task_name=task_name,
                    batch=test_batch,
                    trace=test_trace,
                    features=test_features,
                    targets=test_batch.targets,
                    feature_mean=split["mean"],
                    feature_std=split["std"],
                    eval_scope="heldout_seed",
                    train_seed=args.train_seed,
                    eval_seed=test_seed,
                    final_train_loss=final_loss,
                    baseline_indices=None,
                )
            )
            sync(device)

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "frozen_readout_adapter_generalization.json"
    summary_csv = output_dir / "frozen_readout_adapter_generalization_summary.csv"
    json_path.write_text(
        json.dumps(
            {
                "config": {
                    "tasks": list(task_config.tasks),
                    "sample_count": task_config.sample_count,
                    "timesteps": task_config.timesteps,
                    "train_seed": args.train_seed,
                    "test_seeds": args.test_seeds,
                    "neuron_count": task_config.neuron_count,
                    "max_edges": task_config.max_edges,
                    "adapter_kind": args.adapter_kind,
                    "feature_mode": args.feature_mode,
                    "hidden_units": args.hidden_units,
                    "epochs": args.epochs,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "standardize_features": not args.no_standardize,
                    "device": args.device,
                    "seed_edges": [asdict(edge) for edge in task_config.seed_edges],
                },
                "summary": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(summary_csv, rows)

    paths = {
        "json": str(json_path),
        "summary_csv": str(summary_csv),
    }
    if not args.no_plot:
        try:
            plot_path = output_dir / "frozen_readout_adapter_generalization_summary.png"
            _plot_generalization(rows, plot_path)
            paths["plot"] = str(plot_path)
        except Exception as exc:  # pragma: no cover - optional plotting
            paths["plot"] = f"skipped: {exc}"

    print(json.dumps({"paths": paths, "summary": rows}, indent=2))


def _split_features(features, targets, train_fraction: float, generator, device) -> dict:
    order = features.new_empty((targets.numel(),), dtype=torch.long)
    order.copy_(torch.randperm(targets.numel(), device=device, generator=generator))
    train_count = max(1, int(round(targets.numel() * train_fraction)))
    train_count = min(train_count, targets.numel() - 1)
    train_idx = order[:train_count]
    test_idx = order[train_count:]
    x_train_raw = features.index_select(0, train_idx)
    y_train = targets.index_select(0, train_idx)
    x_test_raw = features.index_select(0, test_idx)
    y_test = targets.index_select(0, test_idx)
    mean = x_train_raw.mean(dim=0, keepdim=True)
    std = x_train_raw.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return {
        "x_train": (x_train_raw - mean) / std,
        "y_train": y_train,
        "x_test_raw": x_test_raw,
        "y_test": y_test,
        "test_idx": test_idx,
        "mean": mean,
        "std": std,
    }


def _train_adapter(runner, x_train, y_train, device, epochs: int, learning_rate: float, weight_decay: float):
    adapter = runner._build_adapter(int(x_train.shape[1]), device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=learning_rate, weight_decay=weight_decay)
    final_loss = 0.0
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = adapter(x_train)
        loss = torch.nn.functional.cross_entropy(logits, y_train)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().item())
        mark_step(device)
    return adapter, final_loss


def _evaluate_batch(
    *,
    adapter,
    runner,
    task_name: str,
    batch,
    trace: dict,
    features,
    targets,
    feature_mean,
    feature_std,
    eval_scope: str,
    train_seed: int,
    eval_seed: int,
    final_train_loss: float,
    baseline_indices,
) -> dict:
    task_cfg = runner.config.task_config
    x = (features - feature_mean) / feature_std if runner.config.standardize_features else features
    with torch.no_grad():
        adapter_predictions = adapter(x).argmax(dim=1)

    if baseline_indices is None:
        evidence = trace["evidence"].detach()
        baseline_inputs = batch.inputs
        baseline_targets = targets
    else:
        evidence = trace["evidence"].detach().index_select(0, baseline_indices)
        baseline_inputs = batch.inputs.index_select(0, baseline_indices)
        baseline_targets = batch.targets.index_select(0, baseline_indices)

    frozen_predictions = evidence.argmax(dim=1)
    instant_reflex = _instant_reflex_predictions(baseline_inputs, task_cfg.motor_channels)
    integrated_reflex = _integrated_reflex_predictions(baseline_inputs, task_cfg.motor_channels)
    inactive_rate = float((evidence.max(dim=1).values <= 1e-8).to(torch.float32).mean().item())
    return {
        "task": task_name,
        "eval_scope": eval_scope,
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "samples": int(targets.numel()),
        "timesteps": task_cfg.timesteps,
        "adapter_kind": runner.config.adapter_kind,
        "feature_mode": runner.config.feature_mode,
        "feature_dim": int(features.shape[1]),
        "hidden_units": int(runner.config.hidden_units),
        "adapter_accuracy": _accuracy(adapter_predictions, targets),
        "frozen_ammc_accuracy": _accuracy(frozen_predictions, baseline_targets),
        "instant_reflex_accuracy": _accuracy(instant_reflex, baseline_targets),
        "integrated_reflex_accuracy": _accuracy(integrated_reflex, baseline_targets),
        "inactive_output_rate": inactive_rate,
        "final_train_loss": final_train_loss,
    }


def _accuracy(predictions, targets) -> float:
    return float((predictions == targets).to(torch.float32).mean().item())


def _instant_reflex_predictions(inputs, motor_channels: int):
    final_food = inputs[:, -1, :motor_channels]
    return final_food.argmax(dim=1)


def _integrated_reflex_predictions(inputs, motor_channels: int):
    integrated_food = inputs[:, :, :motor_channels].sum(dim=1)
    return integrated_food.argmax(dim=1)


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_generalization(rows: Iterable[dict], output_path: pathlib.Path) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    rows = list(rows)
    tasks = list(dict.fromkeys(row["task"] for row in rows))
    split_values = []
    heldout_values = []
    for task in tasks:
        task_rows = [row for row in rows if row["task"] == task]
        split_values.append(next(row["adapter_accuracy"] for row in task_rows if row["eval_scope"] == "train_seed_split"))
        heldout = [row["adapter_accuracy"] for row in task_rows if row["eval_scope"] == "heldout_seed"]
        heldout_values.append(sum(heldout) / len(heldout) if heldout else 0.0)

    x = list(range(len(tasks)))
    width = 0.32
    fig, ax = plt.subplots(figsize=(max(9, len(tasks) * 1.6), 5))
    ax.bar([i - width / 2 for i in x], split_values, width, label="Train-seed held-out split")
    ax.bar([i + width / 2 for i in x], heldout_values, width, label="Mean held-out seeds")
    ax.axhline(0.25, color="gray", linestyle="--", linewidth=1, label="Random 4-way chance")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Adapter accuracy")
    ax.set_title("AMMC Gen-5 frozen readout adapter generalization")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
