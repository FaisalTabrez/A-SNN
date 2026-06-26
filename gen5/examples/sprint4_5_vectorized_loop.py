"""Sprint 4/5 smoke test: vectorized environment + AMMC sparse brain loop."""

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
    DynamicSparseLinear,
    HeadlessAMMCLoop,
    TensorEnvironment2D,
    TensorEnvironmentConfig,
    VectorizedTransducer,
    resolve_device,
    sync,
)
from ammc_gen5.dynamic_sparse import EdgeRecord


def main() -> None:
    device = resolve_device("auto")
    agent_count = 10_000

    environment = TensorEnvironment2D(
        TensorEnvironmentConfig(agent_count=agent_count, food_count=128, toxin_count=128),
        device=device,
    )
    brain = DynamicSparseLinear(16, 16, max_edges=64, device=device)
    brain.load_edges(
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

    loop = HeadlessAMMCLoop(environment, brain, VectorizedTransducer()).to(device)
    for _ in range(120):
        telemetry = loop.step()
    sync(device)

    print(
        {
            "device": str(device),
            "agents": agent_count,
            "mean_fitness": round(float(telemetry["fitness"].float().mean().item()), 6),
            "food_hits": int(telemetry["food_hits"].sum().item()),
            "toxin_hits": int(telemetry["toxin_hits"].sum().item()),
            "mean_speed": round(float(environment.agent_vel.norm(dim=1).mean().item()), 6),
            "active_edges": brain.active_edge_count,
        }
    )


if __name__ == "__main__":
    main()
