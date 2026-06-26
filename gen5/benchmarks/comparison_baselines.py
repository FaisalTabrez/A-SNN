"""Baseline footprint and inference-speed comparisons for the 2D foraging task.

This script establishes the shared measurement surface for later "best SNN" or
"Transformer alternative" claims. It does not pretend to train PPO/BPTT if the
required libraries are absent; instead it records dependency availability and
always emits comparable parameter/memory/inference metrics for the AMMC sparse
policy, a dense LIF-style SNN policy, and a dense MLP policy.

Colab TPU/XLA:

```python
!python gen5/benchmarks/comparison_baselines.py \
  --device xla \
  --population-size 10000 \
  --steps 240 \
  --output-dir gen5_outputs/baselines_xla
```
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time
from dataclasses import asdict, dataclass

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"PyTorch is required for baseline comparisons: {exc}") from exc

from ammc_gen5 import (
    EvolvingHeadlessAMMCLoop,
    EvolvingLoopConfig,
    TensorEnvironment2D,
    TensorEnvironmentConfig,
    TensorEvolver,
    TensorEvolverConfig,
    VectorizedTransducer,
    mark_step,
    resolve_device,
    seed_everything,
    sync,
)
from ammc_gen5.evaluation import default_foraging_seed_edges


@dataclass
class BaselineResult:
    name: str
    status: str
    population_size: int
    steps: int
    seconds: float | None
    ticks_per_second: float | None
    agent_steps_per_second: float | None
    active_parameters: int | None
    total_parameters: int | None
    parameter_memory_mb: float | None
    mean_fitness: float | None
    max_fitness: float | None
    notes: str


class DenseMLPPolicy(nn.Module):
    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2),
            nn.Tanh(),
        )

    def reset(self, batch_size: int, device) -> None:
        return None

    def forward(self, sensory):  # type: ignore[override]
        return self.net(sensory)


class DenseLIFPolicy(nn.Module):
    """Small dense LIF-style baseline with fixed topology.

    This internal implementation is used for inference-footprint measurements.
    The training script can later swap this for snnTorch modules when running
    BPTT in Colab.
    """

    def __init__(self, hidden_size: int = 64, beta: float = 0.9, threshold: float = 1.0) -> None:
        super().__init__()
        self.input = nn.Linear(8, hidden_size)
        self.recurrent = nn.Linear(hidden_size, hidden_size, bias=False)
        self.motor = nn.Linear(hidden_size, 4)
        self.beta = beta
        self.threshold = threshold
        self._membrane = None
        self._spikes = None

    def reset(self, batch_size: int, device) -> None:
        self._membrane = torch.zeros((batch_size, self.input.out_features), device=device)
        self._spikes = torch.zeros_like(self._membrane)

    def forward(self, sensory):  # type: ignore[override]
        if self._membrane is None or self._spikes is None or self._membrane.shape[0] != sensory.shape[0]:
            self.reset(sensory.shape[0], sensory.device)
        current = self.input(sensory) + self.recurrent(self._spikes)
        self._membrane = self._membrane * self.beta + current
        self._spikes = (self._membrane >= self.threshold).to(sensory.dtype)
        self._membrane = torch.where(self._spikes.bool(), torch.zeros_like(self._membrane), self._membrane)
        motor = torch.clamp(self.motor(self._spikes), min=0.0)
        north, east, south, west = motor[:, 0], motor[:, 1], motor[:, 2], motor[:, 3]
        return torch.clamp(torch.stack([east - west, south - north], dim=1), -1.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AMMC Gen-5 baseline comparison scaffold")
    parser.add_argument("--population-size", type=int, default=10_000)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="gen5_outputs/baselines")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    seed_everything(args.seed, device=device)

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = [
        benchmark_ammc(args.population_size, args.steps, args.warmup, device),
        benchmark_policy(
            "dense_lif_snn",
            DenseLIFPolicy(args.hidden_size).to(device),
            args.population_size,
            args.steps,
            args.warmup,
            device,
            notes=_dependency_note("snntorch", "Internal LIF surrogate used unless snnTorch training is installed."),
        ),
        benchmark_policy(
            "dense_mlp_policy",
            DenseMLPPolicy(args.hidden_size).to(device),
            args.population_size,
            args.steps,
            args.warmup,
            device,
            notes="Dense MLP inference scaffold; PPO training hook is dependency-gated.",
        ),
        ppo_availability_result(args.population_size, args.steps),
    ]

    json_path = output_dir / "baseline_comparison.json"
    json_path.write_text(json.dumps([asdict(row) for row in results], indent=2) + "\n", encoding="utf-8")

    csv_path = output_dir / "baseline_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in results)

    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "results": [asdict(r) for r in results]}, indent=2))


def benchmark_ammc(population_size: int, steps: int, warmup: int, device) -> BaselineResult:
    environment = TensorEnvironment2D(TensorEnvironmentConfig(agent_count=population_size), device=device)
    evolver = TensorEvolver(TensorEvolverConfig(population_size=population_size, neuron_count=16, max_edges=128), device=device)
    evolver.seed_from_edges(default_foraging_seed_edges())
    loop = EvolvingHeadlessAMMCLoop(
        environment,
        evolver,
        VectorizedTransducer(),
        EvolvingLoopConfig(epoch_steps=10_000_000),
    ).to(device)
    for _ in range(warmup):
        loop.step()
    sync(device)
    start = time.perf_counter()
    for _ in range(steps):
        loop.step()
    sync(device)
    seconds = time.perf_counter() - start
    active_edges = int(evolver.active_mask.sum().item())
    active_parameters = active_edges
    total_parameters = int(evolver.long_term_weight.numel())
    return BaselineResult(
        name="ammc_sparse_evolver",
        status="ok",
        population_size=population_size,
        steps=steps,
        seconds=seconds,
        ticks_per_second=steps / seconds,
        agent_steps_per_second=(steps / seconds) * population_size,
        active_parameters=active_parameters,
        total_parameters=total_parameters,
        parameter_memory_mb=_tensor_memory_mb([evolver.long_term_weight, evolver.short_term_weight]),
        mean_fitness=float(environment.fitness.float().mean().item()),
        max_fitness=float(environment.fitness.max().item()),
        notes="Sparse AMMC policy counts active edges as active parameters.",
    )


def benchmark_policy(
    name: str,
    policy: nn.Module,
    population_size: int,
    steps: int,
    warmup: int,
    device,
    *,
    notes: str,
) -> BaselineResult:
    environment = TensorEnvironment2D(TensorEnvironmentConfig(agent_count=population_size), device=device)
    policy.reset(population_size, device)
    for _ in range(warmup):
        action = policy(environment.sensory_tensor())
        environment.step(action)
        mark_step(device)
    sync(device)
    start = time.perf_counter()
    for _ in range(steps):
        action = policy(environment.sensory_tensor())
        environment.step(action)
        mark_step(device)
    sync(device)
    seconds = time.perf_counter() - start
    total_parameters = sum(parameter.numel() for parameter in policy.parameters())
    return BaselineResult(
        name=name,
        status="ok",
        population_size=population_size,
        steps=steps,
        seconds=seconds,
        ticks_per_second=steps / seconds,
        agent_steps_per_second=(steps / seconds) * population_size,
        active_parameters=total_parameters,
        total_parameters=total_parameters,
        parameter_memory_mb=_tensor_memory_mb(list(policy.parameters())),
        mean_fitness=float(environment.fitness.float().mean().item()),
        max_fitness=float(environment.fitness.max().item()),
        notes=notes,
    )


def ppo_availability_result(population_size: int, steps: int) -> BaselineResult:
    try:
        import stable_baselines3  # noqa: F401

        status = "available"
        notes = "stable-baselines3 is installed; wrap TensorEnvironment2D in a Gymnasium adapter before PPO training."
    except Exception as exc:
        status = "skipped"
        notes = f"stable-baselines3 unavailable: {exc}"
    return BaselineResult(
        name="ppo_mlp_policy",
        status=status,
        population_size=population_size,
        steps=steps,
        seconds=None,
        ticks_per_second=None,
        agent_steps_per_second=None,
        active_parameters=None,
        total_parameters=None,
        parameter_memory_mb=None,
        mean_fitness=None,
        max_fitness=None,
        notes=notes,
    )


def _dependency_note(module_name: str, fallback: str) -> str:
    try:
        __import__(module_name)
        return f"{module_name} available; {fallback}"
    except Exception as exc:
        return f"{module_name} unavailable ({exc}); {fallback}"


def _tensor_memory_mb(tensors) -> float:
    total = 0
    for tensor in tensors:
        total += tensor.numel() * tensor.element_size()
    return total / (1024 * 1024)


if __name__ == "__main__":
    main()
