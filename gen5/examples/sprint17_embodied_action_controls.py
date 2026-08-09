"""Sprint 17: activity-matched controls for frozen embodied readouts.

Sprint 16 showed a large adapter advantage, but the adapters acted at full
magnitude on every cue-bearing step while the fixed spiking decoder was active
only about five percent of the time. This runner separates movement opportunity
from representation quality with six policies evaluated on identical seeds:

* the original sparse motor-spike decoder;
* a normalized cardinal decoder of the frozen AMMC analog motor evidence;
* a full-activity random cardinal controller;
* the direct food-minus-toxin sensor oracle;
* clean-trained and augmented frozen-trace adapters.

The AMMC topology and weights remain frozen throughout.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from dataclasses import asdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXAMPLES))

from ammc_gen5 import (  # noqa: E402
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
import sprint16_frozen_embodied_adapter as sprint16  # noqa: E402

try:  # pragma: no cover - accelerator dependent
    import torch
except Exception:  # pragma: no cover
    torch = None


POLICIES = (
    "fixed_motor_spiking",
    "fixed_analog_cardinal",
    "random_cardinal",
    "direct_sensor_oracle",
    "base_adapter",
    "augmented_adapter",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run activity-matched frozen-AMMC controller controls")
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
    parser.add_argument("--output-dir", default="gen5_outputs/embodied_action_controls")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate(args)
    device = resolve_device(args.device)
    seed_everything(args.train_seed, device=device)
    print("Training requested frozen-trace adapters...", flush=True)
    adapters = sprint16._train_adapters(args, device)
    if adapters:
        losses = ", ".join(f"{name}={state['final_loss']:.6f}" for name, state in adapters.items())
        print(f"Adapter training complete: {losses}", flush=True)
    else:
        print("No trainable adapter policy requested; skipping adapter training.", flush=True)

    total = len(args.worlds) * len(args.sensor_noise_stds) * len(args.eval_seeds) * len(args.policies)
    rows: list[dict] = []
    completed = 0
    for world in args.worlds:
        for noise_std in args.sensor_noise_stds:
            for seed in args.eval_seeds:
                for policy in args.policies:
                    completed += 1
                    print(
                        f"[{completed}/{total}] world={world} noise={float(noise_std):g} "
                        f"seed={int(seed)} policy={policy}",
                        flush=True,
                    )
                    rows.append(_evaluate_policy(args, device, adapters, world, float(noise_std), int(seed), policy))
                    sync(device)

    summary = _summarize(rows)
    output = pathlib.Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "embodied_action_controls.json"
    records_path = output / "embodied_action_controls_records.csv"
    summary_path = output / "embodied_action_controls_summary.csv"
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
            plot_path = output / "embodied_action_controls_summary.png"
            _plot(summary, plot_path)
            paths["plot"] = str(plot_path)
        except Exception as exc:  # pragma: no cover
            paths["plot"] = f"skipped: {exc}"
    print(json.dumps({"paths": paths, "summary": summary}, indent=2))


def _validate(args) -> None:
    if torch is None:
        raise ImportError("Sprint 17 requires PyTorch")
    if args.steps <= 0 or args.population_size < 2 or args.train_samples < 2:
        raise SystemExit("--steps must be positive; --population-size and --train-samples must be at least 2")
    if args.train_window <= 0 or args.epochs <= 0:
        raise SystemExit("--train-window and --epochs must be positive")
    if args.neuron_count < 12:
        raise SystemExit("--neuron-count must be at least 12")
    if args.max_edges < len(default_foraging_seed_edges()):
        raise SystemExit("--max-edges is too small for the frozen foraging seed")
    if any(value < 0 for value in args.sensor_noise_stds):
        raise SystemExit("--sensor-noise-stds cannot contain negative values")


def _evaluate_policy(args, device, adapters, world, noise_std, seed, policy):
    env_generator = make_generator(seed, device=device)
    noise_generator = make_generator(seed + 100_000, device=device)
    policy_generator = make_generator(seed + 200_000, device=device)
    env_config = world_preset_config(
        world,
        agent_count=args.population_size,
        food_count=args.food_count,
        toxin_count=args.toxin_count,
    )
    environment = TensorEnvironment2D(env_config, device=device)
    environment.reset(generator=env_generator)
    brain = TensorEvolver(
        TensorEvolverConfig(
            population_size=args.population_size,
            neuron_count=args.neuron_count,
            max_edges=args.max_edges,
            ltw_noise_std=0.0,
            sprout_probability=0.0,
            prune_probability=0.0,
        ),
        device=device,
    )
    brain.seed_from_edges(default_foraging_seed_edges())
    transducer = VectorizedTransducer(TransducerConfig(neuron_count=args.neuron_count))
    membrane = torch.zeros((args.population_size, args.neuron_count), device=device)
    counts = torch.zeros_like(membrane)
    active_cue_count = torch.zeros((), device=device)
    active_action_count = torch.zeros((), device=device)
    oracle_agreement_count = torch.zeros((), device=device)
    action_magnitude_sum = torch.zeros((), device=device)

    with torch.no_grad():
        for step in range(args.steps):
            if step % args.train_window == 0:
                membrane.zero_()
                counts.zero_()
            sensory = environment.sensory_tensor()
            if noise_std:
                sensory = torch.clamp(
                    sensory
                    + torch.randn(sensory.shape, device=device, generator=noise_generator) * noise_std,
                    min=0.0,
                )
            spikes, membrane = transducer.lif_step(
                transducer.encode_sensors(sensory) + brain(membrane),
                membrane,
            )
            counts.add_(spikes)
            oracle_direction = sprint16._oracle_targets(sensory, args.toxin_avoidance)
            action = _policy_action(
                policy,
                args=args,
                adapters=adapters,
                transducer=transducer,
                sensory=sensory,
                spikes=spikes,
                membrane=membrane,
                counts=counts,
                oracle_direction=oracle_direction,
                generator=policy_generator,
            )
            cue_active = sensory.amax(dim=1) > 1e-8
            action_active = action.abs().sum(dim=1) > 1e-8
            scored = cue_active & action_active
            direction = sprint16._action_directions(action)
            active_cue_count.add_(cue_active.sum())
            active_action_count.add_(scored.sum())
            oracle_agreement_count.add_(((direction == oracle_direction) & scored).sum())
            action_magnitude_sum.add_(torch.linalg.vector_norm(action, dim=1).sum())
            environment.step(action, generator=env_generator, collect_telemetry=False)

    cue_count = max(1.0, float(active_cue_count.item()))
    action_count = max(1.0, float(active_action_count.item()))
    mean_action_magnitude = float(action_magnitude_sum.item()) / (args.steps * args.population_size)
    mean_fitness = float(environment.fitness.float().mean().item())
    return {
        "world": world,
        "policy": policy,
        "eval_seed": seed,
        "sensor_noise_std": noise_std,
        "steps": args.steps,
        "population_size": args.population_size,
        "mean_fitness": mean_fitness,
        "best_fitness": float(environment.fitness.max().item()),
        "mean_food_hits": float(environment.food_hits.float().mean().item()),
        "mean_toxin_hits": float(environment.toxin_hits.float().mean().item()),
        "survival_rate": float((environment.fitness >= 0).float().mean().item()),
        "cue_action_coverage": float(active_action_count.item()) / cue_count,
        "oracle_action_agreement": float(oracle_agreement_count.item()) / action_count,
        "mean_action_magnitude": mean_action_magnitude,
        "fitness_per_unit_action": mean_fitness / max(mean_action_magnitude, 1e-8),
    }


def _policy_action(
    policy,
    *,
    args,
    adapters,
    transducer,
    sensory,
    spikes,
    membrane,
    counts,
    oracle_direction,
    generator,
):
    if policy == "fixed_motor_spiking":
        return transducer.decode_motors(spikes)
    if policy == "fixed_analog_cardinal":
        start = transducer.config.sensor_channels
        end = start + transducer.config.motor_channels
        evidence = counts[:, start:end] + torch.clamp(membrane[:, start:end], min=0.0)
        direction = evidence.argmax(dim=1)
        action = sprint16._directions_to_actions(direction, sensory.dtype)
        return action * (evidence.max(dim=1).values > 1e-8).unsqueeze(1)
    if policy == "random_cardinal":
        direction = torch.randint(0, 4, (sensory.shape[0],), device=sensory.device, generator=generator)
        return sprint16._directions_to_actions(direction, sensory.dtype)
    if policy == "direct_sensor_oracle":
        return sprint16._directions_to_actions(oracle_direction, sensory.dtype)
    if policy in {"base_adapter", "augmented_adapter"}:
        state = adapters[policy]
        features = torch.cat([membrane, counts], dim=1)
        direction = state["model"]((features - state["mean"]) / state["std"]).argmax(dim=1)
        return sprint16._directions_to_actions(direction, sensory.dtype)
    raise ValueError(f"unknown policy: {policy}")


def _config_dict(args, device, adapters):
    result = vars(args).copy()
    result["device_resolved"] = str(device)
    result["seed_edges"] = [asdict(edge) for edge in default_foraging_seed_edges()]
    result["adapter_final_train_loss"] = {
        name: float(state["final_loss"])
        for name, state in adapters.items()
    }
    result["control_design"] = {
        "random_cardinal": "full-magnitude movement-opportunity control",
        "fixed_analog_cardinal": "full-magnitude cardinal decode when frozen AMMC analog motor evidence is active",
        "direct_sensor_oracle": "upper/control policy used to supervise the adapters",
    }
    result["trace_reset_note"] = "AMMC membrane and spike counts reset every train_window steps for all policies."
    return result


def _summarize(rows):
    groups = {}
    for row in rows:
        key = (row["world"], row["policy"], row["sensor_noise_std"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (world, policy, noise), values in sorted(groups.items()):
        def mean(name):
            return sum(float(value[name]) for value in values) / len(values)

        def std(name):
            average = mean(name)
            return (sum((float(value[name]) - average) ** 2 for value in values) / len(values)) ** 0.5

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
                "mean_fitness_per_unit_action": mean("fitness_per_unit_action"),
            }
        )
    return summary


def _write_csv(path, rows) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(summary, path) -> None:
    import matplotlib.pyplot as plt

    worlds = list(dict.fromkeys(row["world"] for row in summary))
    policies = list(dict.fromkeys(row["policy"] for row in summary))
    noises = sorted({float(row["sensor_noise_std"]) for row in summary})
    lookup = {
        (row["world"], row["policy"], float(row["sensor_noise_std"])): row
        for row in summary
    }
    fig, axes = plt.subplots(1, len(worlds), figsize=(6 * len(worlds), 5), sharey=True)
    if len(worlds) == 1:
        axes = [axes]
    width = 0.8 / max(1, len(policies))
    for axis, world in zip(axes, worlds):
        for index, policy in enumerate(policies):
            x = [position - 0.4 + width / 2 + index * width for position in range(len(noises))]
            values = [lookup[(world, policy, noise)]["mean_fitness"] for noise in noises]
            errors = [lookup[(world, policy, noise)]["std_mean_fitness"] for noise in noises]
            axis.bar(x, values, width, yerr=errors, capsize=2, label=policy)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(world)
        axis.set_xticks(range(len(noises)), [f"noise={value:g}" for value in noises])
        axis.set_xlabel("Sensor noise")
    axes[0].set_ylabel("Mean population fitness")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("AMMC Gen-5 activity-matched embodied controls")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
