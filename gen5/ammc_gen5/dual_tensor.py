"""Dual-frequency electrical/chemical tensor manager."""

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
        raise ImportError("AMMC Gen-5 dual tensor manager requires PyTorch")


@dataclass(frozen=True)
class AstrocyteGridConfig:
    """Low-frequency chemical grid configuration."""

    depth: int = 8
    height: int = 32
    width: int = 32
    slow_tick_ms: float = 100.0
    smoothing: float = 0.12
    hyperactivity_threshold: float = 0.75
    gaba_gain: float = 0.85
    dopamine_gain: float = 1.0


class DualTensorManager(nn.Module):
    """Synchronizes fast sparse electrical activity with slow astrocyte fields."""

    def __init__(self, config: AstrocyteGridConfig | None = None, *, device=None, dtype=None) -> None:
        _require_torch()
        super().__init__()
        self.config = config or AstrocyteGridConfig()
        grid_shape = (self.config.depth, self.config.height, self.config.width)
        self.register_buffer("chemical_state", torch.zeros(grid_shape, device=device, dtype=dtype or torch.float32))
        self.register_buffer("activity_accumulator", torch.zeros(grid_shape, device=device, dtype=dtype or torch.float32))
        self.elapsed_ms = 0.0

    def accumulate_activity(self, activity_grid) -> None:
        """Accumulate high-frequency spike/activity density into the slow grid."""

        if activity_grid.shape != self.chemical_state.shape:
            raise ValueError(f"activity_grid shape {tuple(activity_grid.shape)} does not match chemical grid")
        self.activity_accumulator.add_(activity_grid.detach())

    def step(self, dt_ms: float, *, reward: float = 0.0, punishment: float = 0.0) -> bool:
        """Advance the chemical grid if the slow tick has elapsed.

        Returns ``True`` when a low-frequency update occurred.
        """

        self.elapsed_ms += float(dt_ms)
        if self.elapsed_ms < self.config.slow_tick_ms:
            return False

        ticks = max(1, int(self.elapsed_ms // self.config.slow_tick_ms))
        self.elapsed_ms -= ticks * self.config.slow_tick_ms
        mean_activity = self.activity_accumulator / ticks
        self.activity_accumulator.zero_()

        hyperactivity = torch.relu(mean_activity - self.config.hyperactivity_threshold)
        target = (
            reward * self.config.dopamine_gain
            - punishment * self.config.gaba_gain
            - hyperactivity * self.config.gaba_gain
        )
        target = torch.clamp(target, -1.0, 1.0)
        self.chemical_state.lerp_(target, self.config.smoothing)
        return True

    def modulation(self):
        """Return electrical and plasticity modulation tensors.

        Values are shaped like the chemical grid:

        - negative state increases leak and threshold
        - positive state lowers threshold and increases plasticity gain
        """

        dopamine = torch.clamp(self.chemical_state, min=0.0)
        gaba = torch.clamp(-self.chemical_state, min=0.0)
        return {
            "leak_multiplier": torch.clamp(1.0 + gaba * 1.8 - dopamine * 0.55, 0.45, 3.0),
            "threshold_shift": gaba * 8.0 - dopamine * 7.0,
            "plasticity_multiplier": torch.clamp(1.0 + dopamine * 1.5 - gaba * 0.85, 0.0, 3.0),
            "sprouting_allowed": gaba < 0.35,
        }

    def sample_nearest(self, normalized_positions):
        """Nearest-neighbor sample chemical state at normalized xyz positions.

        ``normalized_positions`` is expected to be ``[..., 3]`` in ``0..1``.
        This is intentionally simple; the CUDA backend should provide
        trilinear sampling.
        """

        positions = torch.clamp(normalized_positions, 0.0, 1.0)
        scale = torch.tensor(
            [self.config.depth - 1, self.config.height - 1, self.config.width - 1],
            device=positions.device,
            dtype=positions.dtype,
        )
        indices = torch.round(positions * scale).to(torch.long)
        return self.chemical_state[indices[..., 0], indices[..., 1], indices[..., 2]]

