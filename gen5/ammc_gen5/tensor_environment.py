"""Vectorized 2D embodiment environment for AMMC Gen-5.

Sprint 4 target: simulate thousands of embodied agents with pure tensor
operations. The environment keeps agent, food, and toxin state in batched
PyTorch tensors so the same code can run on CPU, CUDA, or future accelerator
backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

try:  # pragma: no cover
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None

    class _MissingModule:
        pass

    nn = SimpleNamespace(Module=_MissingModule)


def _require_torch() -> None:
    if torch is None:
        raise ImportError("AMMC Gen-5 tensor environment requires PyTorch")


@dataclass(frozen=True)
class TensorEnvironmentConfig:
    """Configuration for a vectorized 2D world."""

    agent_count: int = 10_000
    food_count: int = 64
    toxin_count: int = 64
    world_size: float = 1.0
    dt: float = 1.0 / 60.0
    friction: float = 0.985
    max_speed: float = 0.75
    action_gain: float = 0.05
    collision_radius: float = 0.025
    sensor_radius: float = 0.35
    respawn_margin: float = 0.05


class TensorEnvironment2D(nn.Module):
    """Batched 2D physics for large AMMC swarms.

    State tensors:

    - ``agent_pos``: ``[A, 2]``
    - ``agent_vel``: ``[A, 2]``
    - ``food_pos``: ``[F, 2]``
    - ``toxin_pos``: ``[T, 2]``

    The implementation intentionally avoids Python loops over agents. Collision
    and nearest-object queries are computed with broadcast tensor math.
    """

    def __init__(self, config: TensorEnvironmentConfig | None = None, *, device=None, dtype=None) -> None:
        _require_torch()
        super().__init__()
        self.config = config or TensorEnvironmentConfig()
        if self.config.agent_count <= 0:
            raise ValueError("agent_count must be positive")
        if self.config.food_count <= 0 or self.config.toxin_count <= 0:
            raise ValueError("food_count and toxin_count must be positive")

        self._factory = {"device": device, "dtype": dtype or torch.float32}
        self.register_buffer("agent_pos", torch.empty((self.config.agent_count, 2), **self._factory))
        self.register_buffer("agent_vel", torch.zeros((self.config.agent_count, 2), **self._factory))
        self.register_buffer("food_pos", torch.empty((self.config.food_count, 2), **self._factory))
        self.register_buffer("toxin_pos", torch.empty((self.config.toxin_count, 2), **self._factory))
        self.register_buffer("food_hits", torch.zeros(self.config.agent_count, dtype=torch.long, device=device))
        self.register_buffer("toxin_hits", torch.zeros(self.config.agent_count, dtype=torch.long, device=device))
        self.register_buffer("fitness", torch.zeros(self.config.agent_count, **self._factory))
        self.reset()

    @property
    def device(self):
        return self.agent_pos.device

    def reset(self, generator=None) -> None:
        """Reset agents and objects in vectorized form."""

        size = self.config.world_size
        margin = self.config.respawn_margin
        with torch.no_grad():
            self.agent_pos.uniform_(0.4 * size, 0.6 * size, generator=generator)
            self.agent_vel.zero_()
            self.food_pos.uniform_(margin, size - margin, generator=generator)
            self.toxin_pos.uniform_(margin, size - margin, generator=generator)
            self.food_hits.zero_()
            self.toxin_hits.zero_()
            self.fitness.zero_()

    def nearest_objects(self):
        """Return nearest food/toxin vectors and distances for each agent."""

        food_delta = self.food_pos.unsqueeze(0) - self.agent_pos.unsqueeze(1)
        toxin_delta = self.toxin_pos.unsqueeze(0) - self.agent_pos.unsqueeze(1)
        food_dist_sq = food_delta.square().sum(dim=-1)
        toxin_dist_sq = toxin_delta.square().sum(dim=-1)

        nearest_food_dist_sq, nearest_food_idx = food_dist_sq.min(dim=1)
        nearest_toxin_dist_sq, nearest_toxin_idx = toxin_dist_sq.min(dim=1)

        agent_index = torch.arange(self.config.agent_count, device=self.device)
        nearest_food_vec = food_delta[agent_index, nearest_food_idx]
        nearest_toxin_vec = toxin_delta[agent_index, nearest_toxin_idx]
        return {
            "nearest_food_vec": nearest_food_vec,
            "nearest_toxin_vec": nearest_toxin_vec,
            "nearest_food_dist": torch.sqrt(nearest_food_dist_sq.clamp_min(1e-12)),
            "nearest_toxin_dist": torch.sqrt(nearest_toxin_dist_sq.clamp_min(1e-12)),
            "nearest_food_idx": nearest_food_idx,
            "nearest_toxin_idx": nearest_toxin_idx,
        }

    def sensory_tensor(self):
        """Build vectorized food/toxin direction sensor channels.

        Returns ``[A, 8]``:

        - food north, east, south, west
        - toxin north, east, south, west
        """

        nearest = self.nearest_objects()
        food = self._directional_drive(nearest["nearest_food_vec"], nearest["nearest_food_dist"])
        toxin = self._directional_drive(nearest["nearest_toxin_vec"], nearest["nearest_toxin_dist"])
        return torch.cat([food, toxin], dim=1)

    def step(self, action, generator=None):
        """Advance physics by one batched tick.

        Args:
            action: Tensor shaped ``[A, 2]`` in normalized x/y acceleration.

        Returns a dictionary of reward, punishment, fitness, and nearest-object
        tensors for downstream astrocyte and evolutionary logic.
        """

        if action.shape != self.agent_pos.shape:
            raise ValueError(f"action shape {tuple(action.shape)} must match {tuple(self.agent_pos.shape)}")

        cfg = self.config
        action = torch.clamp(action, -1.0, 1.0)
        self.agent_vel.add_(action * cfg.action_gain)
        self.agent_vel.mul_(cfg.friction)

        speed = torch.linalg.vector_norm(self.agent_vel, dim=1, keepdim=True).clamp_min(1e-12)
        speed_scale = torch.clamp(cfg.max_speed / speed, max=1.0)
        self.agent_vel.mul_(speed_scale)

        self.agent_pos.add_(self.agent_vel * cfg.dt)
        self._bounce_world_bounds()

        collisions = self._collide_and_respawn(generator=generator)
        nearest = self.nearest_objects()
        reward = collisions["food_collision"].to(self.fitness.dtype)
        punishment = collisions["toxin_collision"].to(self.fitness.dtype)
        self.fitness.add_(reward - punishment)
        self.food_hits.add_(collisions["food_collision"].to(torch.long))
        self.toxin_hits.add_(collisions["toxin_collision"].to(torch.long))

        return {
            "reward": reward,
            "punishment": punishment,
            "fitness": self.fitness.clone(),
            "food_hits": self.food_hits.clone(),
            "toxin_hits": self.toxin_hits.clone(),
            **nearest,
        }

    def _directional_drive(self, vectors, distances):
        cfg = self.config
        closeness = torch.clamp(1.0 - distances / cfg.sensor_radius, min=0.0, max=1.0)
        unit = vectors / distances.unsqueeze(1).clamp_min(1e-12)
        east = torch.clamp(unit[:, 0], min=0.0) * closeness
        west = torch.clamp(-unit[:, 0], min=0.0) * closeness
        south = torch.clamp(unit[:, 1], min=0.0) * closeness
        north = torch.clamp(-unit[:, 1], min=0.0) * closeness
        return torch.stack([north, east, south, west], dim=1)

    def _bounce_world_bounds(self) -> None:
        size = self.config.world_size
        below = self.agent_pos < 0
        above = self.agent_pos > size
        hit = below | above
        self.agent_pos.clamp_(0, size)
        bounce = torch.where(
            hit,
            self.agent_vel.new_full(hit.shape, -0.55),
            self.agent_vel.new_ones(hit.shape),
        )
        self.agent_vel.mul_(bounce)

    def _collide_and_respawn(self, generator=None):
        cfg = self.config
        food_delta = self.food_pos.unsqueeze(0) - self.agent_pos.unsqueeze(1)
        toxin_delta = self.toxin_pos.unsqueeze(0) - self.agent_pos.unsqueeze(1)
        radius_sq = cfg.collision_radius * cfg.collision_radius
        food_collision_matrix = food_delta.square().sum(dim=-1) <= radius_sq
        toxin_collision_matrix = toxin_delta.square().sum(dim=-1) <= radius_sq
        food_collision = food_collision_matrix.any(dim=1)
        toxin_collision = toxin_collision_matrix.any(dim=1)

        food_touched = food_collision_matrix.any(dim=0)
        toxin_touched = toxin_collision_matrix.any(dim=0)
        self._respawn(self.food_pos, food_touched, generator=generator)
        self._respawn(self.toxin_pos, toxin_touched, generator=generator)
        return {"food_collision": food_collision, "toxin_collision": toxin_collision}

    def _respawn(self, positions, mask, generator=None) -> None:
        cfg = self.config
        fresh = torch.empty_like(positions)
        fresh.uniform_(cfg.respawn_margin, cfg.world_size - cfg.respawn_margin, generator=generator)
        positions.copy_(torch.where(mask.unsqueeze(1), fresh, positions))
