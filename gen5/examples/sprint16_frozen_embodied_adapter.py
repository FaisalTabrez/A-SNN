"""Sprint 16: deploy frozen AMMC readouts in the embodied bot worlds.

This is deliberately a *readout* experiment.  The recurrent AMMC edge pool is
frozen for every policy.  A hand-defined sensor-space navigation oracle supplies
supervision for the two adapter policies; it chooses a food direction while
subtracting a configurable toxin-avoidance vector.  That isolates whether a
small trainable transducer can turn the frozen AMMC state into useful physical
actions.

The three policies are:

* ``fixed_motor_argmax``: existing fixed motor-spike decoder (no training);
* ``base_adapter``: MLP readout trained on clean, nominal sensory traces;
* ``augmented_adapter``: same MLP trained with amplitude and sensory-noise
  augmentation.

Example Colab run:

```python
!python gen5/examples/sprint16_frozen_embodied_adapter.py \
  --device cuda \
  --worlds simple moving_toxins gauntlet \
  --eval-seeds 43 44 45 46 47 \
  --population-size 10000 \
  --steps 480 \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/frozen_embodied_adapter_cuda
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
    FrozenReadoutAdapterConfig,
    FrozenReadoutAdapterRunner,
    FrozenTaskConfig,
    TensorEnvironment2D,
    TensorEvolver,
    TensorEvolverConfig,
    TransducerConfig,
    VectorizedTransducer,
    make_generator,
    resolve_device,
    seed_everything,
    sync,
    world_preset_config,
    world_preset_names,
)
from ammc_gen5.evaluation import default_foraging_seed_edges  # noqa: E402

try:  # pragma: no cover - accelerator dependent
    import torch
except Exception:  # pragma: no cover
    torch = None


POLICIES = ("fixed_motor_argmax", "base_adapter", "augmented_adapter")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare frozen AMMC motor readouts in embodied worlds")
    parser.add_argument("--worlds", nargs="+", choices=world_preset_names(), default=["simple", "moving_toxins", "gauntlet"])
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[43, 44, 45, 46, 47])
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--population-size", type=int, default=10_000)
    parser.add_argument("--steps", type=int, default=480)
    parser.add_argument("--food-count", type=int, default=128)
    parser.add_argument("--toxin-count", type=int, default=128)
    parser.add_argument("--neuron-count", type=int, default=16)
    parser.add_argument("--max-edges", type=int, default=128)
    parser.add_argument("--sensor-noise-stds", nargs="+", type=float, default=[0.0, 0.05, 0.15])
    parser.add_argument("--train-samples", type=int, default=8192)
    parser.add_argument("--train-window", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--hidden-units", type=int, default=32)
    parser.add_argument("--toxin-avoidance", type=float, default=1.25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="gen5_outputs/frozen_embodied_adapter")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if torch is None:
        raise ImportError("Sprint 16 requires PyTorch")
    if args.steps <= 0 or args.population_size < 2 or args.train_samples < 2:
        raise SystemExit("--steps must be positive; --population-size and --train-samples must be at least 2")
    if args.train_window <= 0 or args.epochs <= 0:
        raise SystemExit("--train-window and --epochs must be positive")
    if args.neuron_count < 12:
        raise SystemExit("--neuron-count must be at least 12 for the 8-sensor/4-motor convention")
    if args.max_edges < len(default_foraging_seed_edges()):
        raise SystemExit("--max-edges is too small for the frozen foraging seed")
    if any(value < 0 for value in args.sensor_noise_stds):
        raise SystemExit("--sensor-noise-stds cannot contain negative values")

    device = resolve_device(args.device)
    seed_everything(args.train_seed, device=device)
    adapters = _train_adapters(args, device)
    rows: list[dict] = []
    for world in args.worlds:
        for noise_std in args.sensor_noise_stds:
            for seed in args.eval_seeds:
                for policy in args.policies:
                    rows.append(_evaluate_policy(args, device, adapters, world, float(noise_std), int(seed), policy))
                    sync(device)

    summary = _summarize(rows)
    output = pathlib.Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "frozen_embodied_adapter.json"
    records_path = output / "frozen_embodied_adapter_records.csv"
    summary_path = output / "frozen_embodied_adapter_summary.csv"
    json_path.write_text(
        json.dumps({"config": _config_dict(args, device, adapters), "records": rows, "summary": summary}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    _write_csv(records_path, rows)
    _write_csv(summary_path, summary)
    paths = {"json": str(json_path), "records_csv": str(records_path), "summary_csv": str(summary_path)}
    if not args.no_plot:
        try:
            plot_path = output / "frozen_embodied_adapter_summary.png"
            _plot(summary, plot_path)
            paths["plot"] = str(plot_path)
        except Exception as exc:  # pragma: no cover - plotting optional
            paths["plot"] = f"skipped: {exc}"
    print(json.dumps({"paths": paths, "summary": summary}, indent=2))


def _train_adapters(args, device):
    runner = FrozenReadoutAdapterRunner(
        FrozenReadoutAdapterConfig(
            task_config=FrozenTaskConfig(neuron_count=args.neuron_count, max_edges=args.max_edges, device=str(device)),
            adapter_kind="mlp",
            hidden_units=args.hidden_units,
            feature_mode="full_trace",
        )
    )
    adapters = {}
    conditions = {
        "base_adapter": [(1.0, 0.0)],
        "augmented_adapter": [(0.6, 0.0), (1.0, 0.0), (1.4, 0.0), (0.6, 0.05), (1.0, 0.05), (1.4, 0.15)],
    }
    for name, variants in conditions.items():
        if name not in args.policies:
            continue
        features, targets = _adapter_training_data(args, device, variants)
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        adapter = runner._build_adapter(int(features.shape[1]), device)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        for _ in range(args.epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(adapter((features - mean) / std), targets)
            loss.backward()
            optimizer.step()
        adapters[name] = {"model": adapter.eval(), "mean": mean, "std": std, "final_loss": float(loss.detach().item())}
    return adapters


def _adapter_training_data(args, device, variants):
    feature_parts, target_parts = [], []
    per_variant = max(2, args.train_samples // len(variants))
    for offset, (amplitude, noise_std) in enumerate(variants):
        generator = make_generator(args.train_seed + 10_000 + offset, device=device)
        sensory = _synthetic_sensory(per_variant, device, generator) * amplitude
        if noise_std:
            sensory = torch.clamp(sensory + torch.randn(sensory.shape, device=device, generator=generator) * noise_std, min=0.0)
        targets = _oracle_targets(sensory, args.toxin_avoidance)
        for steps in range(1, args.train_window + 1):
            trace = _frozen_trace(sensory.unsqueeze(1).expand(-1, steps, -1), args, device)
            feature_parts.append(trace["features"])
            target_parts.append(targets)
    return torch.cat(feature_parts, dim=0).detach(), torch.cat(target_parts, dim=0).detach()


def _synthetic_sensory(count, device, generator):
    angles = torch.rand((count, 2), device=device, generator=generator) * (2.0 * torch.pi)
    closeness = 0.1 + 0.9 * torch.rand((count, 2), device=device, generator=generator)
    dx, dy = torch.cos(angles) * closeness, torch.sin(angles) * closeness
    def channels(x, y):
        return torch.stack([torch.clamp(-y, min=0.0), torch.clamp(x, min=0.0), torch.clamp(y, min=0.0), torch.clamp(-x, min=0.0)], dim=1)
    return torch.cat([channels(dx[:, 0], dy[:, 0]), channels(dx[:, 1], dy[:, 1])], dim=1)


def _oracle_targets(sensory, toxin_avoidance):
    # Convert north/east/south/west channels into vectors, subtract the toxin
    # vector, then convert back. This makes a toxin to the north produce a
    # genuine southward preference even when the food cue is weak.
    food, toxin = sensory[:, :4], sensory[:, 4:8]
    desired_x = (food[:, 1] - food[:, 3]) - toxin_avoidance * (toxin[:, 1] - toxin[:, 3])
    desired_y = (food[:, 2] - food[:, 0]) - toxin_avoidance * (toxin[:, 2] - toxin[:, 0])
    scores = torch.stack([-desired_y, desired_x, desired_y, -desired_x], dim=1)
    return scores.argmax(dim=1)


def _frozen_trace(inputs, args, device):
    batch, steps, _ = inputs.shape
    brain = TensorEvolver(TensorEvolverConfig(population_size=batch, neuron_count=args.neuron_count, max_edges=args.max_edges, ltw_noise_std=0.0, sprout_probability=0.0, prune_probability=0.0), device=device, dtype=inputs.dtype)
    brain.seed_from_edges(default_foraging_seed_edges())
    transducer = VectorizedTransducer(TransducerConfig(neuron_count=args.neuron_count))
    membrane = inputs.new_zeros((batch, args.neuron_count))
    counts = inputs.new_zeros((batch, args.neuron_count))
    for step in range(steps):
        current = transducer.encode_sensors(inputs[:, step]) + brain(membrane)
        spikes, membrane = transducer.lif_step(current, membrane)
        counts.add_(spikes)
    return {"features": torch.cat([membrane, counts], dim=1), "membrane": membrane, "spike_counts": counts}


def _evaluate_policy(args, device, adapters, world, noise_std, seed, policy):
    generator = make_generator(seed, device=device)
    env_config = world_preset_config(world, agent_count=args.population_size, food_count=args.food_count, toxin_count=args.toxin_count)
    environment = TensorEnvironment2D(env_config, device=device)
    environment.reset(generator=generator)
    brain = TensorEvolver(TensorEvolverConfig(population_size=args.population_size, neuron_count=args.neuron_count, max_edges=args.max_edges, ltw_noise_std=0.0, sprout_probability=0.0, prune_probability=0.0), device=device)
    brain.seed_from_edges(default_foraging_seed_edges())
    transducer = VectorizedTransducer(TransducerConfig(neuron_count=args.neuron_count))
    membrane = torch.zeros((args.population_size, args.neuron_count), device=device)
    counts = torch.zeros_like(membrane)
    food_before, toxin_before = environment.food_hits.clone(), environment.toxin_hits.clone()
    active_cue_count = torch.zeros((), device=device)
    active_action_count = torch.zeros((), device=device)
    oracle_agreement_count = torch.zeros((), device=device)
    action_magnitude_sum = torch.zeros((), device=device)
    with torch.no_grad():
        for step in range(args.steps):
            if step % args.train_window == 0:
                # Controlled finite trace windows prevent cumulative spike
                # counts from drifting outside either adapter's train domain.
                # Every policy gets the same reset schedule.
                membrane.zero_()
                counts.zero_()
            sensory = environment.sensory_tensor()
            if noise_std:
                sensory = torch.clamp(
                    sensory + torch.randn(sensory.shape, device=device, generator=generator) * noise_std,
                    min=0.0,
                )
            spikes, membrane = transducer.lif_step(
                transducer.encode_sensors(sensory) + brain(membrane),
                membrane,
            )
            counts.add_(spikes)
            if policy == "fixed_motor_argmax":
                action = transducer.decode_motors(spikes)
            else:
                state = adapters[policy]
                features = torch.cat([membrane, counts], dim=1)
                direction = state["model"]((features - state["mean"]) / state["std"]).argmax(dim=1)
                action = _directions_to_actions(direction, features.dtype)
            cue_active = sensory.amax(dim=1) > 1e-8
            action_active = action.abs().sum(dim=1) > 1e-8
            scored = cue_active & action_active
            action_direction = _action_directions(action)
            oracle_direction = _oracle_targets(sensory, args.toxin_avoidance)
            active_cue_count.add_(cue_active.sum())
            active_action_count.add_(scored.sum())
            oracle_agreement_count.add_(((action_direction == oracle_direction) & scored).sum())
            action_magnitude_sum.add_(torch.linalg.vector_norm(action, dim=1).sum())
            environment.step(action, generator=generator, collect_telemetry=False)
    food = (environment.food_hits - food_before).float()
    toxin = (environment.toxin_hits - toxin_before).float()
    cue_count = max(1.0, float(active_cue_count.item()))
    action_count = max(1.0, float(active_action_count.item()))
    return {
        "world": world, "policy": policy, "eval_seed": seed, "sensor_noise_std": noise_std,
        "steps": args.steps, "population_size": args.population_size,
        "mean_fitness": float(environment.fitness.float().mean().item()),
        "best_fitness": float(environment.fitness.max().item()),
        "mean_food_hits": float(food.mean().item()), "mean_toxin_hits": float(toxin.mean().item()),
        "survival_rate": float((environment.fitness >= 0).float().mean().item()),
        "cue_action_coverage": float(active_action_count.item()) / cue_count,
        "oracle_action_agreement": float(oracle_agreement_count.item()) / action_count,
        "mean_action_magnitude": float(action_magnitude_sum.item()) / (args.steps * args.population_size),
    }


def _directions_to_actions(direction, dtype):
    table = torch.tensor([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], device=direction.device, dtype=dtype)
    return table.index_select(0, direction)


def _action_directions(action):
    scores = torch.stack([-action[:, 1], action[:, 0], action[:, 1], -action[:, 0]], dim=1)
    return scores.argmax(dim=1)


def _summarize(rows):
    groups = {}
    for row in rows:
        key = (row["world"], row["policy"], row["sensor_noise_std"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (world, policy, noise), values in sorted(groups.items()):
        def mean(name): return sum(float(value[name]) for value in values) / len(values)
        def std(name):
            avg = mean(name)
            return (sum((float(value[name]) - avg) ** 2 for value in values) / len(values)) ** 0.5
        summary.append(
            {
                "world": world,
                "policy": policy,
                "sensor_noise_std": noise,
                "seeds": len(values),
                "mean_fitness": mean("mean_fitness"),
                "std_mean_fitness": std("mean_fitness"),
                "mean_food_hits": mean("mean_food_hits"),
                "mean_toxin_hits": mean("mean_toxin_hits"),
                "mean_survival_rate": mean("survival_rate"),
                "mean_cue_action_coverage": mean("cue_action_coverage"),
                "mean_oracle_action_agreement": mean("oracle_action_agreement"),
                "mean_action_magnitude": mean("mean_action_magnitude"),
            }
        )
    return summary


def _config_dict(args, device, adapters):
    result = vars(args).copy()
    result["device_resolved"] = str(device)
    result["seed_edges"] = [asdict(edge) for edge in default_foraging_seed_edges()]
    result["policy_note"] = "Adapters are supervised by a sensor-space food-minus-toxin oracle; AMMC recurrent weights stay frozen."
    result["trace_reset_note"] = "Membrane and spike-count features reset every train_window steps for all policies."
    result["adapter_final_train_loss"] = {
        name: float(state["final_loss"])
        for name, state in adapters.items()
    }
    return result


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(summary, path):
    import matplotlib.pyplot as plt
    labels = [f"{row['world']}\n{row['policy']}\nnoise={row['sensor_noise_std']:g}" for row in summary]
    values = [row["mean_fitness"] for row in summary]
    errors = [row["std_mean_fitness"] for row in summary]
    fig, ax = plt.subplots(figsize=(max(10, len(summary) * 0.7), 5))
    ax.bar(range(len(summary)), values, yerr=errors, capsize=3)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(summary)), labels, rotation=45, ha="right")
    ax.set_ylabel("Mean population fitness")
    ax.set_title("Frozen AMMC embodied readout comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
