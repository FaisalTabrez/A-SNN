"""Benchmark AMMC Gen-5 headless throughput on CPU/CUDA/XLA.

Colab TPU/XLA target:

```python
!pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
import torch_xla  # verify this succeeds before running --device xla
!python gen5/benchmarks/benchmark_throughput.py \
  --device xla \
  --topology-preset saturated \
  --active-edges 86 \
  --population-sizes 1000 10000 50000 100000 \
  --steps 240 \
  --warmup 30 \
  --output-dir gen5_outputs/throughput_xla_saturated
```

Colab T4/CUDA fallback:

```python
!python gen5/benchmarks/benchmark_throughput.py \
  --device cuda \
  --topology-preset foraging \
  --population-sizes 1000 10000 50000 100000 \
  --steps 240 \
  --warmup 30 \
  --compile \
  --output-dir gen5_outputs/throughput
```

The script reports both raw simulation ticks/sec and agent-steps/sec. The
second number is usually the better scaling metric because one tick advances an
entire population tensor.

Use `--topology-preset foraging` for the original 8-edge prior,
`--topology-preset saturated --active-edges 86` for a champion-like active
edge load, or `--topology-preset champion --adjacency-json ...` to benchmark an
exported champion sparse adjacency exactly.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
import sys
import time
from dataclasses import asdict, dataclass

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"PyTorch is required for throughput benchmarks: {exc}") from exc

from ammc_gen5 import (
    EvolvingHeadlessAMMCLoop,
    EvolvingLoopConfig,
    TensorEnvironment2D,
    TensorEnvironmentConfig,
    TensorEvolver,
    TensorEvolverConfig,
    VectorizedTransducer,
    clear_memory_stats,
    device_kind,
    memory_namespace,
    resolve_device,
    seed_everything,
    sync,
)
from ammc_gen5.evaluation import default_foraging_seed_edges


@dataclass
class ThroughputResult:
    topology_preset: str
    requested_active_edges: int | None
    seeded_active_edges: int
    adjacency_json: str | None
    population_size: int
    steps: int
    seconds: float
    ticks_per_second: float
    agent_steps_per_second: float
    mean_active_synapses: float
    device: str
    dtype: str
    torch_compile_requested: bool
    torch_compile_active: bool
    compile_error: str | None
    cuda_memory_allocated_mb: float | None
    cuda_max_memory_allocated_mb: float | None
    accelerator_backend: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AMMC Gen-5 throughput benchmark")
    parser.add_argument("--population-sizes", nargs="+", type=int, default=[1_000, 10_000, 50_000, 100_000])
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--food-count", type=int, default=128)
    parser.add_argument("--toxin-count", type=int, default=128)
    parser.add_argument("--epoch-steps", type=int, default=10_000_000)
    parser.add_argument("--max-edges", type=int, default=128)
    parser.add_argument(
        "--topology-preset",
        choices=["foraging", "saturated", "champion"],
        default="foraging",
        help="Seed genome topology: 8-edge foraging prior, synthetic saturated pool, or exported champion adjacency",
    )
    parser.add_argument(
        "--active-edges",
        type=int,
        default=None,
        help="Synthetic active edge count for --topology-preset saturated; defaults to champion-like min(86, max_edges)",
    )
    parser.add_argument(
        "--adjacency-json",
        default=None,
        help="Path to champion_sparse_adjacency.json for --topology-preset champion",
    )
    parser.add_argument("--device", default="auto", help="'auto', 'xla'/'tpu', 'cpu', 'cuda', or a torch device string")
    parser.add_argument("--compile", action="store_true", help="Attempt torch.compile around the loop tick")
    parser.add_argument("--output-dir", default="gen5_outputs/throughput")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_edges = build_seed_edges(
        args.topology_preset,
        max_edges=args.max_edges,
        active_edges=args.active_edges,
        adjacency_json=args.adjacency_json,
        seed=args.seed,
    )

    results = [
        run_one_size(
            population_size=population,
            device=device,
            steps=args.steps,
            warmup=args.warmup,
            food_count=args.food_count,
            toxin_count=args.toxin_count,
            epoch_steps=args.epoch_steps,
            max_edges=args.max_edges,
            topology_preset=args.topology_preset,
            requested_active_edges=args.active_edges,
            adjacency_json=args.adjacency_json,
            seed_edges=seed_edges,
            compile_requested=args.compile,
            seed=args.seed,
        )
        for population in args.population_sizes
    ]

    json_path = output_dir / "throughput_results.json"
    json_path.write_text(json.dumps([asdict(row) for row in results], indent=2) + "\n", encoding="utf-8")

    csv_path = output_dir / "throughput_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in results)

    plot_path = None
    try:
        plot_path = output_dir / "throughput_scaling.png"
        plot_results(results, plot_path)
    except Exception as exc:  # pragma: no cover
        plot_path = f"plot skipped: {exc}"

    print(
        json.dumps(
            {
                "device": str(device),
                "topology_preset": args.topology_preset,
                "seeded_active_edges": len(seed_edges),
                "json": str(json_path),
                "csv": str(csv_path),
                "plot": str(plot_path),
                "results": [asdict(row) for row in results],
            },
            indent=2,
        )
    )


def run_one_size(
    *,
    population_size: int,
    device,
    steps: int,
    warmup: int,
    food_count: int,
    toxin_count: int,
    epoch_steps: int,
    max_edges: int,
    topology_preset: str,
    requested_active_edges: int | None,
    adjacency_json: str | None,
    seed_edges,
    compile_requested: bool,
    seed: int,
) -> ThroughputResult:
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    seed_everything(seed, device=device)
    clear_memory_stats(device)

    environment = TensorEnvironment2D(
        TensorEnvironmentConfig(
            agent_count=population_size,
            food_count=food_count,
            toxin_count=toxin_count,
        ),
        device=device,
    )
    evolver = TensorEvolver(
        TensorEvolverConfig(population_size=population_size, neuron_count=16, max_edges=max_edges),
        device=device,
    )
    evolver.seed_from_edges(seed_edges)
    loop = EvolvingHeadlessAMMCLoop(
        environment,
        evolver,
        VectorizedTransducer(),
        EvolvingLoopConfig(epoch_steps=epoch_steps),
    ).to(device)

    tick = loop.step
    compile_active = False
    compile_error = None
    if compile_requested:
        if device_kind(device) == "xla":
            compile_error = "torch.compile skipped for XLA; PyTorch/XLA performs lazy XLA compilation"
        elif hasattr(torch, "compile"):
            try:
                tick = torch.compile(loop.step)  # type: ignore[attr-defined]
                compile_active = True
            except Exception as exc:  # pragma: no cover
                compile_error = str(exc)
        else:
            compile_error = "torch.compile is unavailable in this PyTorch build"

    try:
        for _ in range(warmup):
            tick()
    except Exception as exc:
        if not compile_active:
            raise
        compile_error = f"torch.compile runtime fallback: {exc}"
        compile_active = False
        tick = loop.step
        for _ in range(warmup):
            tick()
    sync(device)
    start = time.perf_counter()
    try:
        for _ in range(steps):
            tick()
    except Exception as exc:
        if not compile_active:
            raise
        compile_error = f"torch.compile timed-loop fallback: {exc}"
        compile_active = False
        tick = loop.step
        sync(device)
        start = time.perf_counter()
        for _ in range(steps):
            tick()
    sync(device)
    seconds = time.perf_counter() - start

    ticks_per_second = steps / seconds if seconds > 0 else float("inf")
    agent_steps = ticks_per_second * population_size
    mem = memory_namespace(device)
    return ThroughputResult(
        topology_preset=topology_preset,
        requested_active_edges=requested_active_edges,
        seeded_active_edges=len(seed_edges),
        adjacency_json=adjacency_json,
        population_size=population_size,
        steps=steps,
        seconds=seconds,
        ticks_per_second=ticks_per_second,
        agent_steps_per_second=agent_steps,
        mean_active_synapses=float(evolver.active_edge_counts().float().mean().item()),
        device=str(device),
        dtype=str(environment.agent_pos.dtype),
        torch_compile_requested=compile_requested,
        torch_compile_active=compile_active,
        compile_error=compile_error,
        cuda_memory_allocated_mb=mem.allocated_mb,
        cuda_max_memory_allocated_mb=mem.max_allocated_mb,
        accelerator_backend=mem.backend,
    )


def build_seed_edges(
    topology_preset: str,
    *,
    max_edges: int,
    active_edges: int | None,
    adjacency_json: str | None,
    seed: int,
):
    if topology_preset == "foraging":
        return default_foraging_seed_edges()
    if topology_preset == "saturated":
        count = active_edges if active_edges is not None else min(86, max_edges)
        return synthetic_saturated_edges(count, max_edges=max_edges, seed=seed)
    if topology_preset == "champion":
        path = resolve_adjacency_path(adjacency_json)
        return load_adjacency_edges(path, max_edges=max_edges)
    raise ValueError(f"unsupported topology_preset: {topology_preset}")


def synthetic_saturated_edges(active_edges: int, *, max_edges: int, seed: int):
    if active_edges <= 0:
        raise ValueError("active_edges must be positive")
    if active_edges > max_edges:
        raise ValueError(f"active_edges {active_edges} exceeds max_edges {max_edges}")
    rng = random.Random(seed)
    edges = []
    for slot in range(active_edges):
        source = rng.randrange(16)
        target_offset = rng.randrange(1, 16)
        target = (source + target_offset) % 16
        ltw = rng.uniform(0.02, 0.35)
        sign = -1.0 if rng.random() < 0.2 else 1.0
        delay = rng.randrange(0, 65)
        edges.append(
            # Match the champion export convention: mostly LTW, no volatile STW.
            # Slot identity itself is not required by TensorEvolver.seed_from_edges.
            _edge_record(source, target, short_term_weight=0.0, long_term_weight=ltw, sign=sign, delay_steps=delay)
        )
    return tuple(edges)


def resolve_adjacency_path(adjacency_json: str | None) -> pathlib.Path:
    if adjacency_json:
        path = pathlib.Path(adjacency_json)
    else:
        path = ROOT / "outputs" / "colab_500_gen_2026-06-25" / "champion_sparse_adjacency.json"
    if not path.exists():
        raise FileNotFoundError(
            f"champion adjacency file not found: {path}. "
            "Pass --adjacency-json path/to/champion_sparse_adjacency.json."
        )
    return path


def load_adjacency_edges(path: pathlib.Path, *, max_edges: int):
    payload = json.loads(path.read_text(encoding="utf-8"))
    columns = payload.get("columns", [])
    rows = payload.get("rows", [])
    required = ["source_index", "target_index", "ltw", "stw", "sign", "delay_steps"]
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError(f"{path} is missing sparse adjacency columns: {missing}")
    if len(rows) > max_edges:
        raise ValueError(f"{path} contains {len(rows)} active edges, exceeding max_edges {max_edges}")

    index = {name: columns.index(name) for name in required}
    edges = []
    for row in rows:
        edges.append(
            _edge_record(
                int(row[index["source_index"]]),
                int(row[index["target_index"]]),
                short_term_weight=float(row[index["stw"]]),
                long_term_weight=float(row[index["ltw"]]),
                sign=float(row[index["sign"]]),
                delay_steps=int(row[index["delay_steps"]]),
            )
        )
    return tuple(edges)


def _edge_record(source, target, *, short_term_weight, long_term_weight, sign, delay_steps):
    from ammc_gen5.dynamic_sparse import EdgeRecord

    return EdgeRecord(
        int(source),
        int(target),
        short_term_weight=float(short_term_weight),
        long_term_weight=float(long_term_weight),
        sign=float(sign),
        delay_steps=int(delay_steps),
    )


def plot_results(results: list[ThroughputResult], path: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    x = [row.population_size for row in results]
    y = [row.agent_steps_per_second for row in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, marker="o", linewidth=2, color="#38bdf8")
    ax.set_xscale("log")
    ax.set_yscale("log")
    topology = results[0].topology_preset if results else "unknown"
    active = results[0].seeded_active_edges if results else "?"
    ax.set_title(f"AMMC Gen-5 Throughput Scaling ({topology}, {active} edges)")
    ax.set_xlabel("Population size")
    ax.set_ylabel("Agent-steps / second")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


if __name__ == "__main__":
    main()
