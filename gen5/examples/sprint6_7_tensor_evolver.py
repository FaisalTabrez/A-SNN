"""Sprint 6/7 smoke test: tensorized culling, broadcast, and mutation."""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"PyTorch is required for this smoke test: {exc}") from exc

from ammc_gen5 import TensorEvolver, TensorEvolverConfig, resolve_device, sync
from ammc_gen5.dynamic_sparse import EdgeRecord


def main() -> None:
    device = resolve_device("auto")
    population = 10_000
    evolver = TensorEvolver(
        TensorEvolverConfig(
            population_size=population,
            neuron_count=16,
            max_edges=128,
            ltw_noise_std=0.02,
            sprout_probability=0.02,
            prune_probability=0.01,
        ),
        device=device,
    )
    evolver.seed_from_edges(
        [
            EdgeRecord(0, 8, long_term_weight=0.8),
            EdgeRecord(1, 9, long_term_weight=0.8),
            EdgeRecord(2, 10, long_term_weight=0.8),
            EdgeRecord(3, 11, long_term_weight=0.8),
            EdgeRecord(4, 8, long_term_weight=0.6, sign=-1),
            EdgeRecord(5, 9, long_term_weight=0.6, sign=-1),
            EdgeRecord(6, 10, long_term_weight=0.6, sign=-1),
            EdgeRecord(7, 11, long_term_weight=0.6, sign=-1),
        ]
    )

    # Mock environment fitness: in production this is TensorEnvironment2D.fitness.
    fitness = torch.randn(population, device=device)
    report = evolver.evolve_epoch(fitness)

    neural_state = torch.rand((population, 16), device=device)
    recurrent_current = evolver(neural_state)
    sync(device)

    print(
        {
            "device": str(device),
            "population": population,
            "epoch": report["epoch"],
            "best_fitness": round(report["best_fitness"], 6),
            "mean_fitness": round(report["mean_fitness"], 6),
            "sprouts": report["sprout_count"],
            "prunes": report["prune_count"],
            "active_edges_mean": round(float(evolver.active_edge_counts().float().mean().item()), 6),
            "recurrent_current_shape": tuple(recurrent_current.shape),
        }
    )


if __name__ == "__main__":
    main()
