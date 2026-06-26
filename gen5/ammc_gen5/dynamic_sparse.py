"""Dynamic sparse electrical graph prototype.

This module is the Sprint 1 Python contract for the future C++/CUDA backend.

The prototype deliberately uses fixed-capacity edge slots plus an active mask.
That keeps PyTorch optimizers and autograd stable while still exposing the
operations Gen-5 needs: sprouting, pruning, STW/LTW memory, and sparse current
accumulation. A CUDA extension can later replace the storage implementation
without changing the public module API.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterable, Sequence

try:  # pragma: no cover - exercised by environments with torch installed
    import torch
    import torch.nn as nn
    from torch.autograd import Function
except Exception:  # pragma: no cover - lets syntax checks run without torch
    torch = None

    class _MissingModule:
        pass

    nn = SimpleNamespace(Module=_MissingModule, Parameter=lambda value: value)
    Function = object


def _require_torch() -> None:
    if torch is None:
        raise ImportError("AMMC Gen-5 dynamic sparse backend requires PyTorch")


@dataclass(frozen=True)
class EdgeRecord:
    """Serializable description of one directed sparse edge."""

    source: int
    target: int
    short_term_weight: float = 0.0
    long_term_weight: float = 0.1
    sign: float = 1.0
    delay_steps: int = 0


class DynamicSparseLinearFunction(Function):
    """Sparse linear propagation with separate STW/LTW gradients.

    Input shape: ``[batch, in_features]``
    Output shape: ``[batch, out_features]``

    Only active edge slots participate. Gradients flow to both STW and LTW
    because the effective edge magnitude is ``STW + LTW``.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        input_tensor,
        sources,
        targets,
        short_term_weight,
        long_term_weight,
        active_mask,
        signs,
        out_features: int,
    ):
        _require_torch()
        active = active_mask.to(dtype=input_tensor.dtype)
        signed_weight = (short_term_weight + long_term_weight) * signs * active

        gathered = input_tensor.index_select(1, sources)
        edge_current = gathered * signed_weight.unsqueeze(0)
        output = input_tensor.new_zeros((input_tensor.shape[0], out_features))
        output.index_add_(1, targets, edge_current)

        ctx.save_for_backward(input_tensor, sources, targets, signed_weight, signs, active)
        ctx.in_features = input_tensor.shape[1]
        return output

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        _require_torch()
        input_tensor, sources, targets, signed_weight, signs, active = ctx.saved_tensors

        target_grad = grad_output.index_select(1, targets)
        source_input = input_tensor.index_select(1, sources)

        grad_input_edges = target_grad * signed_weight.unsqueeze(0)
        grad_input = input_tensor.new_zeros(input_tensor.shape)
        grad_input.index_add_(1, sources, grad_input_edges)

        grad_effective = (target_grad * source_input).sum(dim=0)
        grad_weight = grad_effective * signs * active

        return (
            grad_input,
            None,
            None,
            grad_weight,
            grad_weight,
            None,
            None,
            None,
        )


class DynamicSparseLinear(nn.Module):
    """Sparse, structurally plastic linear operator.

    The public API models true structural plasticity while the prototype keeps a
    stable maximum-capacity tensor for optimizer compatibility.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_edges: int,
        *,
        device=None,
        dtype=None,
    ) -> None:
        _require_torch()
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if max_edges <= 0:
            raise ValueError("max_edges must be positive")

        factory_kwargs = {"device": device}
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.max_edges = int(max_edges)

        self.register_buffer("sources", torch.zeros(max_edges, dtype=torch.long, **factory_kwargs))
        self.register_buffer("targets", torch.zeros(max_edges, dtype=torch.long, **factory_kwargs))
        self.register_buffer("active_mask", torch.zeros(max_edges, dtype=torch.bool, **factory_kwargs))
        self.register_buffer("signs", torch.ones(max_edges, dtype=dtype or torch.float32, **factory_kwargs))
        self.register_buffer("delay_steps", torch.zeros(max_edges, dtype=torch.long, **factory_kwargs))

        self.short_term_weight = nn.Parameter(torch.zeros(max_edges, dtype=dtype or torch.float32, **factory_kwargs))
        self.long_term_weight = nn.Parameter(torch.zeros(max_edges, dtype=dtype or torch.float32, **factory_kwargs))

    @property
    def active_edge_count(self) -> int:
        return int(self.active_mask.sum().item())

    @property
    def effective_weight(self):
        _require_torch()
        return (self.short_term_weight + self.long_term_weight) * self.signs * self.active_mask.to(self.long_term_weight.dtype)

    def forward(self, input_tensor):  # type: ignore[override]
        return DynamicSparseLinearFunction.apply(
            input_tensor,
            self.sources,
            self.targets,
            self.short_term_weight,
            self.long_term_weight,
            self.active_mask,
            self.signs,
            self.out_features,
        )

    def load_edges(self, edges: Iterable[EdgeRecord | Sequence[float]]) -> None:
        """Replace active edges with the provided edge records."""

        _require_torch()
        records = [self._coerce_edge(edge) for edge in edges]
        if len(records) > self.max_edges:
            raise ValueError(f"received {len(records)} edges, capacity is {self.max_edges}")

        with torch.no_grad():
            self.active_mask.zero_()
            self.short_term_weight.zero_()
            self.long_term_weight.zero_()
            self.signs.fill_(1)
            self.delay_steps.zero_()

            for slot, edge in enumerate(records):
                self._write_edge(slot, edge)

    def sprout(
        self,
        source: int,
        target: int,
        *,
        short_term_weight: float = 0.05,
        long_term_weight: float = 0.0,
        sign: float = 1.0,
        delay_steps: int = 0,
    ) -> int:
        """Allocate a new edge in the first free slot and return its slot id."""

        _require_torch()
        free_slots = (~self.active_mask).nonzero(as_tuple=False).flatten()
        if free_slots.numel() == 0:
            raise RuntimeError("dynamic sparse edge pool is full")
        slot = int(free_slots[0].item())
        edge = EdgeRecord(source, target, short_term_weight, long_term_weight, sign, delay_steps)
        with torch.no_grad():
            self._write_edge(slot, edge)
        return slot

    def prune_below(self, minimum_long_term_weight: float = 1e-6) -> list[int]:
        """Deactivate edges whose LTW is at or below the threshold."""

        _require_torch()
        with torch.no_grad():
            doomed = (self.active_mask & (self.long_term_weight <= minimum_long_term_weight)).nonzero(as_tuple=False).flatten()
            slots = [int(slot.item()) for slot in doomed]
            if doomed.numel():
                self.active_mask[doomed] = False
                self.short_term_weight[doomed] = 0
                self.long_term_weight[doomed] = 0
                self.delay_steps[doomed] = 0
                self.signs[doomed] = 1
            return slots

    def consolidate(self, fraction: float = 0.18) -> float:
        """Transfer a fraction of STW into LTW for active edges."""

        _require_torch()
        if not 0 <= fraction <= 1:
            raise ValueError("fraction must be in [0, 1]")
        with torch.no_grad():
            active = self.active_mask
            transferable = torch.minimum(
                self.short_term_weight * fraction,
                torch.clamp(1.0 - self.long_term_weight, min=0.0),
            )
            transferable = torch.where(active, transferable, torch.zeros_like(transferable))
            self.short_term_weight.sub_(transferable)
            self.long_term_weight.add_(transferable)
            return float(transferable.sum().item())

    def decay_short_term(self, amount: float) -> None:
        """Apply fast STW decay to active slots."""

        _require_torch()
        if amount < 0:
            raise ValueError("amount must be non-negative")
        with torch.no_grad():
            decayed = torch.clamp(self.short_term_weight - amount, min=0.0)
            self.short_term_weight.copy_(torch.where(self.active_mask, decayed, torch.zeros_like(decayed)))

    def edge_records(self) -> list[EdgeRecord]:
        """Return active edges as serializable records."""

        _require_torch()
        slots = self.active_mask.nonzero(as_tuple=False).flatten().tolist()
        return [
            EdgeRecord(
                source=int(self.sources[slot].item()),
                target=int(self.targets[slot].item()),
                short_term_weight=float(self.short_term_weight[slot].item()),
                long_term_weight=float(self.long_term_weight[slot].item()),
                sign=float(self.signs[slot].item()),
                delay_steps=int(self.delay_steps[slot].item()),
            )
            for slot in slots
        ]

    def _write_edge(self, slot: int, edge: EdgeRecord) -> None:
        if not 0 <= edge.source < self.in_features:
            raise ValueError(f"source {edge.source} is outside input range")
        if not 0 <= edge.target < self.out_features:
            raise ValueError(f"target {edge.target} is outside output range")
        self.sources[slot] = edge.source
        self.targets[slot] = edge.target
        self.short_term_weight[slot] = edge.short_term_weight
        self.long_term_weight[slot] = edge.long_term_weight
        self.signs[slot] = -1.0 if edge.sign < 0 else 1.0
        self.delay_steps[slot] = max(0, int(edge.delay_steps))
        self.active_mask[slot] = True

    @staticmethod
    def _coerce_edge(edge: EdgeRecord | Sequence[float]) -> EdgeRecord:
        if isinstance(edge, EdgeRecord):
            return edge
        if len(edge) < 4:
            raise ValueError("edge sequences must contain at least source, target, STW, LTW")
        source, target, stw, ltw, *rest = edge
        sign = rest[0] if len(rest) >= 1 else 1.0
        delay = rest[1] if len(rest) >= 2 else 0
        return EdgeRecord(int(source), int(target), float(stw), float(ltw), float(sign), int(delay))

