from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from ammc_gen5.dynamic_sparse import DynamicSparseLinear, EdgeRecord


@unittest.skipIf(torch is None, "PyTorch is not installed")
class DynamicSparseContractTest(unittest.TestCase):
    def test_forward_backward_and_structural_ops(self) -> None:
        layer = DynamicSparseLinear(3, 2, max_edges=4)
        layer.load_edges(
            [
                EdgeRecord(0, 1, short_term_weight=0.1, long_term_weight=0.4),
                EdgeRecord(2, 0, short_term_weight=0.0, long_term_weight=0.8),
            ]
        )

        x = torch.tensor([[1.0, 0.5, 0.25]], requires_grad=True)
        y = layer(x)
        self.assertEqual(tuple(y.shape), (1, 2))

        y.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(layer.short_term_weight.grad.shape[0], 4)
        self.assertEqual(layer.long_term_weight.grad.shape[0], 4)

        slot = layer.sprout(1, 0, short_term_weight=0.05, long_term_weight=0.0)
        self.assertEqual(slot, 2)
        self.assertEqual(layer.active_edge_count, 3)

        pruned = layer.prune_below(0.01)
        self.assertEqual(pruned, [2])
        self.assertEqual(layer.active_edge_count, 2)


if __name__ == "__main__":
    unittest.main()

