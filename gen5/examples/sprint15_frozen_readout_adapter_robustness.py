"""Sprint 15: frozen readout adapter robustness checks.

Held-out seed generalization passed perfectly for the current synthetic task
family. This runner makes the test less cozy: train the adapter once on the
base distribution, then evaluate without retraining under one-axis perturbations
such as amplitude shifts, sensory noise, and changed sequence length.

Colab-scale run:

```python
!python gen5/examples/sprint15_frozen_readout_adapter_robustness.py \
  --device cuda \
  --adapter-kind mlp \
  --feature-mode full_trace \
  --sample-count 4096 \
  --timesteps 8 \
  --neuron-count 16 \
  --max-edges 128 \
  --epochs 200 \
  --test-seeds 43 44 45 46 47 \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/frozen_readout_adapter_robustness_cuda
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
    FrozenTaskRunner,
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
    parser = argparse.ArgumentParser(description="Evaluate frozen readout adapter robustness under task perturbations")
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
    parser.add_argument("--amplitudes", nargs="+", type=float, default=[0.35, 0.55, 0.75, 1.0])
    parser.add_argument("--noise-stds", nargs="+", type=float, default=[0.0, 0.05, 0.15])
    parser.add_argument("--timestep-values", nargs="+", type=int, default=[4, 8, 12])
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--adapter-kind", choices=("linear", "mlp"), default="mlp")
    parser.add_argument("--feature-mode", choices=("full_trace", "motor_trace"), default="full_trace")
    parser.add_argument("--hidden-units", type=int, default=32)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="gen5_outputs/frozen_readout_adapter_robustness")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_tasks:
        print(json.dumps(list(available_frozen_tasks()), indent=2))
        return
    if torch is None:
        raise ImportError("frozen readout adapter robustness requires PyTorch")

    device = resolve_device(args.device)
    seed_everything(args.train_seed, device=device)
    train_task_config = _task_config(args, seed=args.train_seed, timesteps=args.timesteps, amplitude=args.input_amplitude)
    adapter_config = FrozenReadoutAdapterConfig(
        task_config=train_task_config,
        train_fraction=args.train_fraction,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adapter_kind=args.adapter_kind,
        feature_mode=args.feature_mode,
        hidden_units=args.hidden_units,
        standardize_features=not args.no_standardize,
    )
    train_runner = FrozenReadoutAdapterRunner(adapter_config)
    split_generator = make_generator(args.train_seed + 50_000, device=device)

    conditions = _conditions(args)
    rows: list[dict] = []
    for offset, task_name in enumerate(train_task_config.tasks):
        train_generator = make_generator(args.train_seed + offset + 1, device=device)
        train_batch = train_runner.task_runner._make_task(task_name, train_generator, device)
        train_trace = train_runner.task_runner._frozen_ammc_trace(train_batch.inputs, device)
        train_features = train_runner._select_features(train_trace).detach()
        split = _split_features(
            train_features,
            train_batch.targets,
            args.train_fraction,
            split_generator,
            device,
            standardize=not args.no_standardize,
        )
        adapter, final_loss = _train_adapter(
            train_runner,
            split["x_train"],
            split["y_train"],
            device,
            args.epochs,
            args.learning_rate,
            args.weight_decay,
        )

        for condition in conditions:
            for test_seed in args.test_seeds:
                test_task_config = _task_config(
                    args,
                    seed=test_seed,
                    timesteps=condition["timesteps"],
                    amplitude=condition["amplitude"],
                )
                test_runner = FrozenTaskRunner(test_task_config)
                test_generator = make_generator(test_seed + offset + 1, device=device)
                test_batch = test_runner._make_task(task_name, test_generator, device)
                eval_inputs = _apply_noise(test_batch.inputs, condition["noise_std"], test_seed, offset, device)
                trace = test_runner._frozen_ammc_trace(eval_inputs, device)
                features = train_runner._select_features(trace).detach()
                rows.append(
                    _evaluate(
                        adapter=adapter,
                        runner=train_runner,
                        task_name=task_name,
                        condition=condition,
                        eval_seed=test_seed,
                        inputs=eval_inputs,
                        targets=test_batch.targets,
                        trace=trace,
                        features=features,
                        feature_mean=split["mean"],
                        feature_std=split["std"],
                        final_train_loss=final_loss,
                    )
                )
                sync(device)

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "frozen_readout_adapter_robustness.json"
    summary_csv = output_dir / "frozen_readout_adapter_robustness_summary.csv"
    json_path.write_text(
        json.dumps(
            {
                "config": {
                    "tasks": list(train_task_config.tasks),
                    "sample_count": train_task_config.sample_count,
                    "train_seed": args.train_seed,
                    "test_seeds": args.test_seeds,
                    "base_timesteps": args.timesteps,
                    "base_input_amplitude": args.input_amplitude,
                    "amplitudes": args.amplitudes,
                    "noise_stds": args.noise_stds,
                    "timestep_values": args.timestep_values,
                    "neuron_count": args.neuron_count,
                    "max_edges": args.max_edges,
                    "adapter_kind": args.adapter_kind,
                    "feature_mode": args.feature_mode,
                    "hidden_units": args.hidden_units,
                    "epochs": args.epochs,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "standardize_features": not args.no_standardize,
                    "device": args.device,
                    "seed_edges": [asdict(edge) for edge in train_task_config.seed_edges],
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
            plot_path = output_dir / "frozen_readout_adapter_robustness_summary.png"
            _plot_robustness(rows, plot_path)
            paths["plot"] = str(plot_path)
        except Exception as exc:  # pragma: no cover - optional plotting
            paths["plot"] = f"skipped: {exc}"

    print(json.dumps({"paths": paths, "summary": rows}, indent=2))


def _task_config(args, *, seed: int, timesteps: int, amplitude: float) -> FrozenTaskConfig:
    return FrozenTaskConfig(
        tasks=tuple(args.tasks),
        sample_count=args.sample_count,
        timesteps=timesteps,
        seed=seed,
        neuron_count=args.neuron_count,
        max_edges=args.max_edges,
        sensor_gain=args.sensor_gain,
        leak=args.leak,
        threshold=args.threshold,
        input_amplitude=amplitude,
        device=args.device,
    )


def _conditions(args) -> list[dict]:
    seen: set[tuple[str, float, int, float]] = set()
    conditions: list[dict] = []

    def add(name: str, *, amplitude: float, timesteps: int, noise_std: float) -> None:
        key = (name, float(amplitude), int(timesteps), float(noise_std))
        if key in seen:
            return
        seen.add(key)
        conditions.append(
            {
                "condition": name,
                "amplitude": float(amplitude),
                "timesteps": int(timesteps),
                "noise_std": float(noise_std),
            }
        )

    add("base", amplitude=args.input_amplitude, timesteps=args.timesteps, noise_std=0.0)
    for amplitude in args.amplitudes:
        add(f"amplitude_{amplitude:g}", amplitude=amplitude, timesteps=args.timesteps, noise_std=0.0)
    for noise_std in args.noise_stds:
        add(f"noise_{noise_std:g}", amplitude=args.input_amplitude, timesteps=args.timesteps, noise_std=noise_std)
    for timesteps in args.timestep_values:
        add(f"timesteps_{timesteps}", amplitude=args.input_amplitude, timesteps=timesteps, noise_std=0.0)
    return conditions


def _split_features(features, targets, train_fraction: float, generator, device, *, standardize: bool) -> dict:
    order = torch.randperm(targets.numel(), device=device, generator=generator)
    train_count = max(1, int(round(targets.numel() * train_fraction)))
    train_count = min(train_count, targets.numel() - 1)
    train_idx = order[:train_count]
    x_train_raw = features.index_select(0, train_idx)
    y_train = targets.index_select(0, train_idx)
    mean = x_train_raw.mean(dim=0, keepdim=True)
    std = x_train_raw.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    x_train = (x_train_raw - mean) / std if standardize else x_train_raw
    return {
        "x_train": x_train,
        "y_train": y_train,
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


def _apply_noise(inputs, noise_std: float, seed: int, offset: int, device):
    if noise_std <= 0:
        return inputs
    generator = make_generator(seed + 60_000 + offset, device=device)
    noise = torch.randn(inputs.shape, generator=generator, device=inputs.device, dtype=inputs.dtype) * noise_std
    return torch.clamp(inputs + noise, min=0.0)


def _evaluate(
    *,
    adapter,
    runner,
    task_name: str,
    condition: dict,
    eval_seed: int,
    inputs,
    targets,
    trace: dict,
    features,
    feature_mean,
    feature_std,
    final_train_loss: float,
) -> dict:
    task_cfg = runner.config.task_config
    x = (features - feature_mean) / feature_std if runner.config.standardize_features else features
    with torch.no_grad():
        adapter_predictions = adapter(x).argmax(dim=1)

    evidence = trace["evidence"].detach()
    frozen_predictions = evidence.argmax(dim=1)
    instant_reflex = _instant_reflex_predictions(inputs, task_cfg.motor_channels)
    integrated_reflex = _integrated_reflex_predictions(inputs, task_cfg.motor_channels)
    inactive_rate = float((evidence.max(dim=1).values <= 1e-8).to(torch.float32).mean().item())
    return {
        "task": task_name,
        "condition": condition["condition"],
        "eval_seed": eval_seed,
        "samples": int(targets.numel()),
        "timesteps": int(condition["timesteps"]),
        "amplitude": float(condition["amplitude"]),
        "noise_std": float(condition["noise_std"]),
        "adapter_kind": runner.config.adapter_kind,
        "feature_mode": runner.config.feature_mode,
        "feature_dim": int(features.shape[1]),
        "hidden_units": int(runner.config.hidden_units),
        "adapter_accuracy": _accuracy(adapter_predictions, targets),
        "frozen_ammc_accuracy": _accuracy(frozen_predictions, targets),
        "instant_reflex_accuracy": _accuracy(instant_reflex, targets),
        "integrated_reflex_accuracy": _accuracy(integrated_reflex, targets),
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


def _plot_robustness(rows: Iterable[dict], output_path: pathlib.Path) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    rows = list(rows)
    tasks = list(dict.fromkeys(row["task"] for row in rows))
    conditions = list(dict.fromkeys(row["condition"] for row in rows))
    fig, axes = plt.subplots(len(tasks), 1, figsize=(max(10, len(conditions) * 0.8), max(4, len(tasks) * 2.2)), sharex=True)
    if len(tasks) == 1:
        axes = [axes]
    for ax, task in zip(axes, tasks):
        values = []
        for condition in conditions:
            subset = [row["adapter_accuracy"] for row in rows if row["task"] == task and row["condition"] == condition]
            values.append(sum(subset) / len(subset) if subset else 0.0)
        ax.bar(range(len(conditions)), values)
        ax.axhline(0.25, color="gray", linestyle="--", linewidth=1)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel(task)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(range(len(conditions)))
    axes[-1].set_xticklabels(conditions, rotation=30, ha="right")
    fig.suptitle("AMMC Gen-5 frozen readout adapter robustness")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
