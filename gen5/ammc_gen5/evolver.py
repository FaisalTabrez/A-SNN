"""Tensorized genetic evolution for AMMC Gen-5.

Sprint 6/7 target: perform culling, genome broadcast, and topology mutation
with batched tensor operations instead of CPU-side organism loops.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace
from typing import Iterable, Sequence

try:  # pragma: no cover
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None

    class _MissingModule:
        pass

    nn = SimpleNamespace(Module=_MissingModule)

from .dynamic_sparse import EdgeRecord


def _require_torch() -> None:
    if torch is None:
        raise ImportError("AMMC Gen-5 tensor evolver requires PyTorch")


@dataclass(frozen=True)
class TensorEvolverConfig:
    """Configuration for batched genetic evolution."""

    population_size: int = 10_000
    neuron_count: int = 16
    max_edges: int = 128
    survivor_fraction: float = 0.5
    ltw_noise_std: float = 0.02
    sprout_probability: float = 0.02
    prune_probability: float = 0.01
    initial_sprout_ltw: float = 0.05
    initial_sprout_stw: float = 0.0
    inhibitory_probability: float = 0.2
    max_delay_steps: int = 64
    plasticity_reward_threshold: float = 0.0
    gate_ltw_noise_by_positive_fitness: bool = False
    gate_pruning_by_positive_fitness: bool = False
    gate_sprouting_by_positive_fitness: bool = False


class TensorEvolver(nn.Module):
    """Batched sparse genome pool for a whole swarm.

    Tensors are shaped ``[population, max_edges]``. Each row is one organism's
    sparse edge pool. Culling and reproduction are implemented with `argsort`
    plus indexed tensor assignment, so the population can be evolved in VRAM.
    """

    def __init__(self, config: TensorEvolverConfig | None = None, *, device=None, dtype=None) -> None:
        _require_torch()
        super().__init__()
        self.config = config or TensorEvolverConfig()
        if self.config.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if self.config.neuron_count < 2:
            raise ValueError("neuron_count must be at least 2")
        if self.config.max_edges <= 0:
            raise ValueError("max_edges must be positive")
        if not 0 < self.config.survivor_fraction < 1:
            raise ValueError("survivor_fraction must be in (0, 1)")

        factory = {"device": device}
        float_dtype = dtype or torch.float32
        shape = (self.config.population_size, self.config.max_edges)
        self.register_buffer("sources", torch.zeros(shape, dtype=torch.long, **factory))
        self.register_buffer("targets", torch.ones(shape, dtype=torch.long, **factory))
        self.register_buffer("active_mask", torch.zeros(shape, dtype=torch.bool, **factory))
        self.register_buffer("signs", torch.ones(shape, dtype=float_dtype, **factory))
        self.register_buffer("delay_steps", torch.zeros(shape, dtype=torch.long, **factory))
        self.register_buffer("short_term_weight", torch.zeros(shape, dtype=float_dtype, **factory))
        self.register_buffer("long_term_weight", torch.zeros(shape, dtype=float_dtype, **factory))
        self.register_buffer("epoch", torch.zeros((), dtype=torch.long, **factory))

    @property
    def population_size(self) -> int:
        return self.config.population_size

    @property
    def neuron_count(self) -> int:
        return self.config.neuron_count

    @property
    def max_edges(self) -> int:
        return self.config.max_edges

    @property
    def effective_weight(self):
        _require_torch()
        active = self.active_mask.to(self.long_term_weight.dtype)
        return (self.short_term_weight + self.long_term_weight) * self.signs * active

    def forward(self, neural_state):  # type: ignore[override]
        """Apply each organism's sparse genome to its own neural state.

        Args:
            neural_state: ``[population, neuron_count]``

        Returns:
            Sparse recurrent current, also ``[population, neuron_count]``.
        """

        if neural_state.shape != (self.population_size, self.neuron_count):
            raise ValueError(
                f"neural_state shape {tuple(neural_state.shape)} must be "
                f"({self.population_size}, {self.neuron_count})"
            )
        source_current = torch.gather(neural_state, 1, self.sources)
        edge_current = source_current * self.effective_weight
        output = neural_state.new_zeros(neural_state.shape)
        output.scatter_add_(1, self.targets, edge_current)
        return output

    def seed_from_edges(self, edges: Iterable[EdgeRecord | Sequence[float]]) -> None:
        """Broadcast one seed genome to every organism."""

        records = [self._coerce_edge(edge) for edge in edges]
        if len(records) > self.max_edges:
            raise ValueError(f"received {len(records)} edges, capacity is {self.max_edges}")

        with torch.no_grad():
            self.sources.zero_()
            self.targets.fill_(1)
            self.active_mask.zero_()
            self.signs.fill_(1)
            self.delay_steps.zero_()
            self.short_term_weight.zero_()
            self.long_term_weight.zero_()

            if not records:
                return

            source = torch.tensor([edge.source for edge in records], dtype=torch.long, device=self.sources.device)
            target = torch.tensor([edge.target for edge in records], dtype=torch.long, device=self.targets.device)
            sign = torch.tensor([edge.sign for edge in records], dtype=self.signs.dtype, device=self.signs.device)
            delay = torch.tensor([edge.delay_steps for edge in records], dtype=torch.long, device=self.delay_steps.device)
            stw = torch.tensor(
                [edge.short_term_weight for edge in records],
                dtype=self.short_term_weight.dtype,
                device=self.short_term_weight.device,
            )
            ltw = torch.tensor(
                [edge.long_term_weight for edge in records],
                dtype=self.long_term_weight.dtype,
                device=self.long_term_weight.device,
            )

            count = len(records)
            self.sources[:, :count] = source.unsqueeze(0)
            self.targets[:, :count] = target.unsqueeze(0)
            self.signs[:, :count] = torch.where(sign < 0, -torch.ones_like(sign), torch.ones_like(sign)).unsqueeze(0)
            self.delay_steps[:, :count] = delay.unsqueeze(0)
            self.short_term_weight[:, :count] = stw.unsqueeze(0)
            self.long_term_weight[:, :count] = ltw.unsqueeze(0)
            self.active_mask[:, :count] = True

    def evolve_epoch(self, fitness, *, generator=None) -> dict:
        """Cull low performers, broadcast winners, then mutate children."""

        replacement = self.cull_and_broadcast(fitness)
        parent_fitness = fitness.index_select(0, replacement["parent_indices"].to(fitness.device))
        mutation = self.mutate_children(
            replacement["culled_indices"],
            parent_fitness=parent_fitness,
            generator=generator,
        )
        with torch.no_grad():
            self.epoch.add_(1)
        return {**replacement, **mutation, "epoch": int(self.epoch.item())}

    def evolve(self, fitness, *, generator=None) -> dict:
        """Alias used by runtime loops at epoch boundaries."""

        return self.evolve_epoch(fitness, generator=generator)

    def cull_and_broadcast(self, fitness) -> dict:
        """Replace bottom organisms with copied top organisms.

        Uses `torch.argsort` to rank the population. The top half survives; the
        bottom half receives broadcast copies from the top half in rank order.
        """

        if fitness.shape != (self.population_size,):
            raise ValueError(f"fitness shape {tuple(fitness.shape)} must be ({self.population_size},)")

        order = torch.argsort(fitness, descending=True)
        survivor_count = max(1, int(math.ceil(self.population_size * self.config.survivor_fraction)))
        survivor_count = min(survivor_count, self.population_size - 1)
        survivors = order[:survivor_count]
        culled = order[survivor_count:]
        parent_slots = torch.arange(culled.numel(), device=fitness.device) % survivor_count
        parents = survivors.index_select(0, parent_slots)

        with torch.no_grad():
            for tensor in self._genome_tensors():
                tensor[culled] = tensor[parents]

        return {
            "survivor_indices": survivors,
            "culled_indices": culled,
            "parent_indices": parents,
            "best_fitness": float(fitness[survivors[0]].item()),
            "mean_fitness": float(fitness.float().mean().item()),
        }

    def mutate_children(self, child_indices, *, parent_fitness=None, generator=None) -> dict:
        """Apply LTW noise, random pruning, and random sprouting to children."""

        if child_indices.numel() == 0:
            return {"ltw_mutation_count": 0, "sprout_count": 0, "prune_count": 0}

        cfg = self.config
        child_indices = child_indices.to(self.sources.device)
        child_active = self.active_mask.index_select(0, child_indices)
        child_sources = self.sources.index_select(0, child_indices)
        child_targets = self.targets.index_select(0, child_indices)
        child_signs = self.signs.index_select(0, child_indices)
        child_delay = self.delay_steps.index_select(0, child_indices)
        child_stw = self.short_term_weight.index_select(0, child_indices)
        child_ltw = self.long_term_weight.index_select(0, child_indices)

        plasticity_gate = self._plasticity_gate(child_indices, parent_fitness)

        noise = self._randn(child_ltw.shape, generator=generator) * cfg.ltw_noise_std
        ltw_mutation_mask = child_active
        if cfg.gate_ltw_noise_by_positive_fitness:
            ltw_mutation_mask = ltw_mutation_mask & plasticity_gate
        child_ltw = torch.where(ltw_mutation_mask, torch.clamp(child_ltw + noise, 0.0, 1.0), child_ltw)

        prune_mask = child_active & (self._rand(child_active.shape, generator=generator) < cfg.prune_probability)
        if cfg.gate_pruning_by_positive_fitness:
            prune_mask = prune_mask & plasticity_gate
        child_active = child_active & ~prune_mask
        child_stw = torch.where(prune_mask, torch.zeros_like(child_stw), child_stw)
        child_ltw = torch.where(prune_mask, torch.zeros_like(child_ltw), child_ltw)

        inactive = ~child_active
        sprout_mask = inactive & (self._rand(child_active.shape, generator=generator) < cfg.sprout_probability)
        if cfg.gate_sprouting_by_positive_fitness:
            sprout_mask = sprout_mask & plasticity_gate
        new_sources = torch.randint(
            0,
            self.neuron_count,
            child_sources.shape,
            device=child_sources.device,
            generator=generator,
        )
        target_offset = torch.randint(
            1,
            self.neuron_count,
            child_targets.shape,
            device=child_targets.device,
            generator=generator,
        )
        new_targets = (new_sources + target_offset) % self.neuron_count
        new_signs = torch.where(
            self._rand(child_signs.shape, generator=generator) < cfg.inhibitory_probability,
            child_signs.new_full(child_signs.shape, -1.0),
            child_signs.new_ones(child_signs.shape),
        )
        new_delays = torch.randint(
            0,
            cfg.max_delay_steps + 1,
            child_delay.shape,
            device=child_delay.device,
            generator=generator,
        )
        child_sources = torch.where(sprout_mask, new_sources, child_sources)
        child_targets = torch.where(sprout_mask, new_targets, child_targets)
        child_signs = torch.where(sprout_mask, new_signs, child_signs)
        child_delay = torch.where(sprout_mask, new_delays, child_delay)
        child_stw = torch.where(
            sprout_mask,
            child_stw.new_full(child_stw.shape, cfg.initial_sprout_stw),
            child_stw,
        )
        child_ltw = torch.where(
            sprout_mask,
            child_ltw.new_full(child_ltw.shape, cfg.initial_sprout_ltw),
            child_ltw,
        )
        child_active = child_active | sprout_mask

        with torch.no_grad():
            self.active_mask[child_indices] = child_active
            self.sources[child_indices] = child_sources
            self.targets[child_indices] = child_targets
            self.signs[child_indices] = child_signs
            self.delay_steps[child_indices] = child_delay
            self.short_term_weight[child_indices] = child_stw
            self.long_term_weight[child_indices] = child_ltw

        return {
            "ltw_mutation_count": int(ltw_mutation_mask.sum().item()),
            "sprout_count": int(sprout_mask.sum().item()),
            "prune_count": int(prune_mask.sum().item()),
        }

    def active_edge_counts(self):
        return self.active_mask.sum(dim=1)

    def snapshot_genome(self, index: int, *, to_cpu: bool = True) -> dict:
        """Clone one organism genome row for checkpointing/export."""

        if index < 0 or index >= self.population_size:
            raise IndexError(f"genome index {index} is outside population")

        snapshot = {
            "sources": self.sources[index].detach().clone(),
            "targets": self.targets[index].detach().clone(),
            "active_mask": self.active_mask[index].detach().clone(),
            "signs": self.signs[index].detach().clone(),
            "delay_steps": self.delay_steps[index].detach().clone(),
            "short_term_weight": self.short_term_weight[index].detach().clone(),
            "long_term_weight": self.long_term_weight[index].detach().clone(),
        }
        if to_cpu:
            snapshot = {
                key: value.cpu() if hasattr(value, "cpu") else value
                for key, value in snapshot.items()
            }
        return snapshot

    def _rand(self, shape, *, generator=None):
        return torch.rand(shape, device=self.long_term_weight.device, dtype=self.long_term_weight.dtype, generator=generator)

    def _randn(self, shape, *, generator=None):
        return torch.randn(shape, device=self.long_term_weight.device, dtype=self.long_term_weight.dtype, generator=generator)

    def _genome_tensors(self):
        return (
            self.sources,
            self.targets,
            self.active_mask,
            self.signs,
            self.delay_steps,
            self.short_term_weight,
            self.long_term_weight,
        )

    def _plasticity_gate(self, child_indices, parent_fitness=None):
        """Return a per-child/per-edge dopamine gate for structural mutation."""

        if parent_fitness is None:
            return torch.ones(
                (child_indices.numel(), self.max_edges),
                dtype=torch.bool,
                device=self.active_mask.device,
            )
        reward = parent_fitness.to(device=self.active_mask.device, dtype=self.long_term_weight.dtype)
        positive = reward > self.config.plasticity_reward_threshold
        return positive.unsqueeze(1).expand(child_indices.numel(), self.max_edges)

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
