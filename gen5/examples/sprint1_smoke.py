"""Sprint 1 smoke test for the AMMC Gen-5 sparse backend."""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"PyTorch is required for this smoke test: {exc}") from exc

from ammc_gen5 import DynamicSparseLinear
from ammc_gen5.dynamic_sparse import EdgeRecord


def main() -> None:
    layer = DynamicSparseLinear(4, 3, max_edges=8)
    layer.load_edges(
        [
            EdgeRecord(0, 1, short_term_weight=0.1, long_term_weight=0.4),
            EdgeRecord(2, 0, short_term_weight=0.0, long_term_weight=0.8),
            EdgeRecord(3, 2, short_term_weight=0.2, long_term_weight=0.2, sign=-1),
        ]
    )

    x = torch.tensor([[1.0, 0.0, 0.5, 0.25]], requires_grad=True)
    y = layer(x)
    loss = y.square().sum()
    loss.backward()

    new_slot = layer.sprout(1, 2, short_term_weight=0.05, long_term_weight=0.01)
    consolidated = layer.consolidate(0.5)
    pruned = layer.prune_below(0.02)

    print(
        {
            "output": y.detach().tolist(),
            "active_edges": layer.active_edge_count,
            "new_slot": new_slot,
            "consolidated": round(consolidated, 6),
            "pruned": pruned,
            "grad_input": x.grad.detach().tolist(),
        }
    )


if __name__ == "__main__":
    main()

