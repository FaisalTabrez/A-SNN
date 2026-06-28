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
    food_reward: float = 1.0
    toxin_penalty: float = 1.0
    reward_delay_steps: int = 0
    punishment_delay_steps: int = 0
    moving_food_speed: float = 0.0
    moving_toxin_speed: float = 0.0


@dataclass(frozen=True)
class WorldPreset:
    """Named world-difficulty preset.

    The preset stores only values that differ from the caller's base
    configuration. Population size, object counts, and explicit CLI overrides
    can still be supplied separately.
    """

    name: str
    description: str
    overrides: dict[str, float | int]


WORLD_PRESETS: tuple[WorldPreset, ...] = (
    WorldPreset(
        name="simple",
        description="Original simple foraging world used for current baselines.",
        overrides={},
    ),
    WorldPreset(
        name="wide_arena",
        description="Larger arena with unchanged sensor radius, making search and memory more important.",
        overrides={"world_size": 2.0, "max_speed": 0.9},
    ),
    WorldPreset(
        name="sparse_cues",
        description="Shorter sensor radius for partial-observability pressure.",
        overrides={"sensor_radius": 0.22},
    ),
    WorldPreset(
        name="moving_toxins",
        description="Toxins drift and bounce, forcing continual hazard tracking.",
        overrides={"moving_toxin_speed": 0.35},
    ),
    WorldPreset(
        name="delayed_reward",
        description="Food collisions pay out after a short delay, rewarding memory traces.",
        overrides={"reward_delay_steps": 12},
    ),
    WorldPreset(
        name="gauntlet",
        description="Combined hard world: larger arena, sparse cues, moving toxins, and delayed food reward.",
        overrides={
            "world_size": 2.0,
            "sensor_radius": 0.25,
            "moving_toxin_speed": 0.4,
            "reward_delay_steps": 12,
            "max_speed": 0.9,
        },
    ),
)


def available_world_presets() -> tuple[WorldPreset, ...]:
    """Return supported named world presets."""

    return WORLD_PRESETS


def world_preset_names() -> tuple[str, ...]:
    """Return just the supported preset names."""

    return tuple(preset.name for preset in WORLD_PRESETS)


def world_preset_config(name: str, **overrides) -> TensorEnvironmentConfig:
    """Build a ``TensorEnvironmentConfig`` from a named preset plus overrides."""

    by_name = {preset.name: preset for preset in WORLD_PRESETS}
    if name not in by_name:
        allowed = ", ".join(sorted(by_name))
        raise ValueError(f"unknown world preset {name!r}; expected one of: {allowed}")
    values = dict(by_name[name].overrides)
    values.update({key: value for key, value in overrides.items() if value is not None})
    return TensorEnvironmentConfig(**values)


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
        if self.config.world_size <= 0:
            raise ValueError("world_size must be positive")
        if self.config.dt <= 0:
            raise ValueError("dt must be positive")
        if self.config.reward_delay_steps < 0 or self.config.punishment_delay_steps < 0:
            raise ValueError("reward_delay_steps and punishment_delay_steps cannot be negative")
        if self.config.moving_food_speed < 0 or self.config.moving_toxin_speed < 0:
            raise ValueError("moving_food_speed and moving_toxin_speed cannot be negative")

        self._factory = {"device": device, "dtype": dtype or torch.float32}
        self.register_buffer("agent_pos", torch.empty((self.config.agent_count, 2), **self._factory))
        self.register_buffer("agent_vel", torch.zeros((self.config.agent_count, 2), **self._factory))
        self.register_buffer("food_pos", torch.empty((self.config.food_count, 2), **self._factory))
        self.register_buffer("toxin_pos", torch.empty((self.config.toxin_count, 2), **self._factory))
        self.register_buffer("food_vel", torch.zeros((self.config.food_count, 2), **self._factory))
        self.register_buffer("toxin_vel", torch.zeros((self.config.toxin_count, 2), **self._factory))
        self.register_buffer(
            "reward_delay_buffer",
            torch.zeros((max(1, self.config.reward_delay_steps), self.config.agent_count), **self._factory),
        )
        self.register_buffer(
            "punishment_delay_buffer",
            torch.zeros((max(1, self.config.punishment_delay_steps), self.config.agent_count), **self._factory),
        )
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
            self._reset_object_velocity(self.food_vel, self.config.moving_food_speed, generator=generator)
            self._reset_object_velocity(self.toxin_vel, self.config.moving_toxin_speed, generator=generator)
            self.reward_delay_buffer.zero_()
            self.punishment_delay_buffer.zero_()
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

    def step(self, action, generator=None, *, collect_telemetry: bool = True):
        """Advance physics by one batched tick.

        Args:
            action: Tensor shaped ``[A, 2]`` in normalized x/y acceleration.
            collect_telemetry: When ``False``, skip nearest-object telemetry
                and cloned diagnostic payloads. This is intended for benchmark
                and inference hotpaths where callers only need environment
                state mutation and fitness accumulation.

        Returns a dictionary of reward, punishment, fitness, and nearest-object
        tensors for downstream astrocyte and evolutionary logic when
        ``collect_telemetry`` is true; otherwise returns ``None``.
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
        self._move_objects()

        collisions = self._collide_and_respawn(generator=generator)
        raw_reward = collisions["food_collision"].to(self.fitness.dtype) * cfg.food_reward
        raw_punishment = collisions["toxin_collision"].to(self.fitness.dtype) * cfg.toxin_penalty
        reward = self._apply_delay(raw_reward, self.reward_delay_buffer, cfg.reward_delay_steps)
        punishment = self._apply_delay(raw_punishment, self.punishment_delay_buffer, cfg.punishment_delay_steps)
        self.fitness.add_(reward - punishment)
        self.food_hits.add_(collisions["food_collision"].to(torch.long))
        self.toxin_hits.add_(collisions["toxin_collision"].to(torch.long))

        if not collect_telemetry:
            return None

        nearest = self.nearest_objects()
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

    def _move_objects(self) -> None:
        if self.config.moving_food_speed > 0:
            self._move_and_bounce(self.food_pos, self.food_vel)
        if self.config.moving_toxin_speed > 0:
            self._move_and_bounce(self.toxin_pos, self.toxin_vel)

    def _move_and_bounce(self, positions, velocities) -> None:
        size = self.config.world_size
        positions.add_(velocities * self.config.dt)
        below = positions < 0
        above = positions > size
        hit = below | above
        positions.clamp_(0, size)
        bounce = torch.where(
            hit,
            velocities.new_full(hit.shape, -1.0),
            velocities.new_ones(hit.shape),
        )
        velocities.mul_(bounce)

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
        self._reset_touched_velocities(self.food_vel, food_touched, cfg.moving_food_speed, generator=generator)
        self._reset_touched_velocities(self.toxin_vel, toxin_touched, cfg.moving_toxin_speed, generator=generator)
        return {"food_collision": food_collision, "toxin_collision": toxin_collision}

    def _respawn(self, positions, mask, generator=None) -> None:
        cfg = self.config
        fresh = torch.empty_like(positions)
        fresh.uniform_(cfg.respawn_margin, cfg.world_size - cfg.respawn_margin, generator=generator)
        positions.copy_(torch.where(mask.unsqueeze(1), fresh, positions))

    def _reset_object_velocity(self, velocities, speed: float, generator=None) -> None:
        if speed <= 0:
            velocities.zero_()
            return
        angles = torch.empty((velocities.shape[0],), device=velocities.device, dtype=velocities.dtype)
        angles.uniform_(0.0, 2.0 * torch.pi, generator=generator)
        velocities[:, 0] = torch.cos(angles) * speed
        velocities[:, 1] = torch.sin(angles) * speed

    def _reset_touched_velocities(self, velocities, mask, speed: float, generator=None) -> None:
        if speed <= 0:
            return
        fresh = torch.empty_like(velocities)
        self._reset_object_velocity(fresh, speed, generator=generator)
        velocities.copy_(torch.where(mask.unsqueeze(1), fresh, velocities))

    def _apply_delay(self, value, buffer, delay_steps: int):
        if delay_steps <= 0:
            return value
        due = buffer[0].clone()
        if buffer.shape[0] > 1:
            buffer[:-1].copy_(buffer[1:].clone())
        buffer[-1].copy_(value)
        return due
