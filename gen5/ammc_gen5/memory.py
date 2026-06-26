"""Short-term / long-term synaptic memory utilities."""

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
        raise ImportError("AMMC Gen-5 memory utilities require PyTorch")


@dataclass(frozen=True)
class LTWSTWConfig:
    """Memory dynamics constants."""

    short_term_decay_per_tick: float = 1.8e-5
    long_term_decay_per_tick: float = 4.0e-9
    consolidation_fraction: float = 0.18
    max_effective_weight: float = 1.2


class LTWSTWMemory(nn.Module):
    """Applies AMMC's two-tier memory rules to tensor weights."""

    def __init__(self, config: LTWSTWConfig | None = None) -> None:
        _require_torch()
        super().__init__()
        self.config = config or LTWSTWConfig()

    def effective(self, short_term_weight, long_term_weight):
        return short_term_weight + long_term_weight

    def reinforce_short_term(self, short_term_weight, long_term_weight, amount):
        capacity = torch.clamp(self.config.max_effective_weight - long_term_weight, min=0.0)
        return torch.minimum(short_term_weight + amount, capacity)

    def decay(self, short_term_weight, long_term_weight, *, plasticity_enabled: bool = True):
        stw = torch.clamp(short_term_weight - self.config.short_term_decay_per_tick, min=0.0)
        if plasticity_enabled:
            ltw = torch.clamp(long_term_weight - self.config.long_term_decay_per_tick, min=0.0)
        else:
            ltw = long_term_weight
        return stw, ltw

    def consolidate(self, short_term_weight, long_term_weight):
        transferable = torch.minimum(
            short_term_weight * self.config.consolidation_fraction,
            torch.clamp(1.0 - long_term_weight, min=0.0),
        )
        return short_term_weight - transferable, long_term_weight + transferable, transferable

    @staticmethod
    def optimizer_groups(module, *, ltw_lr: float, stw_lr: float, **shared):
        """Return optimizer groups that can treat STW and LTW differently."""

        _require_torch()
        return [
            {"params": [module.long_term_weight], "lr": ltw_lr, **shared},
            {"params": [module.short_term_weight], "lr": stw_lr, **shared},
        ]

