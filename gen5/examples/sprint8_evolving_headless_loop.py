"""Sprint 8 smoke test: full tensorized evolutionary AMMC loop."""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"PyTorch is required for this smoke test: {exc}") from exc

from ammc_gen5 import (
    ChampionExporter,
    EvolvingHeadlessAMMCLoop,
    EvolvingLoopConfig,
    EvolutionTelemetryLogger,
    TensorEnvironment2D,
    TensorEnvironmentConfig,
    TensorEvolver,
    TensorEvolverConfig,
    VectorizedTransducer,
    resolve_device,
    sync,
)
from ammc_gen5.dynamic_sparse import EdgeRecord


def main() -> None:
    device = resolve_device("auto")
    population = 10_000

    environment = TensorEnvironment2D(
        TensorEnvironmentConfig(agent_count=population, food_count=128, toxin_count=128),
        device=device,
    )
    evolver = TensorEvolver(
        TensorEvolverConfig(population_size=population, neuron_count=16, max_edges=128),
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

    logger = EvolutionTelemetryLogger()
    loop = EvolvingHeadlessAMMCLoop(
        environment,
        evolver,
        VectorizedTransducer(),
        EvolvingLoopConfig(epoch_steps=120),
        logger=logger,
    ).to(device)

    telemetry = loop.run(steps=240)
    sync(device)
    report = loop.last_epoch_report or {}
    output_dir = pathlib.Path("gen5_outputs")
    json_path = logger.save_json(output_dir / "evolution_telemetry.json")
    csv_path = logger.save_csv(output_dir / "evolution_telemetry.csv")
    plot_path = None
    try:
        logger.plot(output_dir / "evolution_telemetry.png")
        plot_path = str(output_dir / "evolution_telemetry.png")
    except Exception as exc:  # matplotlib may be unavailable in minimal runtimes
        plot_path = f"plot skipped: {exc}"

    champion_export = None
    if loop.best_genome_snapshot is not None:
        champion_export = ChampionExporter().export_from_snapshot(
            loop.best_genome_snapshot,
            output_dir / "champion",
            neuron_count=evolver.neuron_count,
            organism_id="Gen5Champion",
            fitness=loop.best_fitness,
        )

    print(
        {
            "device": str(device),
            "population": population,
            "generation": int(loop.generation.item()),
            "epoch_step": int(loop.epoch_step.item()),
            "last_best_fitness": round(float(report.get("best_fitness", 0.0)), 6),
            "last_mean_fitness": round(float(report.get("mean_fitness", 0.0)), 6),
            "sprouts": int(report.get("sprout_count", 0)),
            "prunes": int(report.get("prune_count", 0)),
            "active_edges_mean": round(float(evolver.active_edge_counts().float().mean().item()), 6),
            "final_action_shape": tuple(telemetry["action"].shape),
            "telemetry_json": str(json_path),
            "telemetry_csv": str(csv_path),
            "telemetry_plot": plot_path,
            "champion_connectome": str(champion_export.connectome_path) if champion_export else None,
            "champion_weights": str(champion_export.weights_path) if champion_export else None,
            "champion_adjacency": str(champion_export.adjacency_path) if champion_export else None,
        }
    )


if __name__ == "__main__":
    main()
