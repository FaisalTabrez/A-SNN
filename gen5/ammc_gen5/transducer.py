"""Vectorized environment-to-brain transduction for AMMC Gen-5."""

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

from .dynamic_sparse import DynamicSparseLinear
from .runtime import mark_step
from .tensor_environment import TensorEnvironment2D


def _require_torch() -> None:
    if torch is None:
        raise ImportError("AMMC Gen-5 transducer requires PyTorch")


@dataclass(frozen=True)
class TransducerConfig:
    """Sensor/motor channel mapping."""

    neuron_count: int = 16
    sensor_channels: int = 8
    motor_channels: int = 4
    sensor_gain: float = 1.0
    motor_gain: float = 1.0
    leak: float = 0.9
    threshold: float = 1.0


class VectorizedTransducer(nn.Module):
    """Maps batched world state into neural inputs and motor outputs.

    Default convention:

    - neurons ``0:8`` receive food/toxin directional sensors
    - neurons ``8:12`` are north/east/south/west motor neurons
    """

    def __init__(self, config: TransducerConfig | None = None) -> None:
        _require_torch()
        super().__init__()
        self.config = config or TransducerConfig()
        if self.config.neuron_count < self.config.sensor_channels + self.config.motor_channels:
            raise ValueError("neuron_count must fit sensor and motor channels")

    def encode_sensors(self, sensory_tensor):
        """Place environment sensory channels into a neural input tensor."""

        batch = sensory_tensor.shape[0]
        output = sensory_tensor.new_zeros((batch, self.config.neuron_count))
        output[:, : self.config.sensor_channels] = sensory_tensor * self.config.sensor_gain
        return output

    def decode_motors(self, neural_state):
        """Decode motor channels into x/y action accelerations."""

        start = self.config.sensor_channels
        motor = torch.clamp(neural_state[:, start : start + self.config.motor_channels], min=0.0)
        if motor.shape[1] < 4:
            raise ValueError("at least four motor channels are required")
        north, east, south, west = motor[:, 0], motor[:, 1], motor[:, 2], motor[:, 3]
        dx = east - west
        dy = south - north
        action = torch.stack([dx, dy], dim=1) * self.config.motor_gain
        return torch.clamp(action, -1.0, 1.0)

    def lif_step(self, current, membrane):
        """Simple vectorized LIF surrogate used by the headless loop."""

        membrane = membrane * self.config.leak + current
        spikes = (membrane >= self.config.threshold).to(current.dtype)
        membrane = torch.where(spikes.bool(), torch.zeros_like(membrane), membrane)
        return spikes, membrane


class HeadlessAMMCLoop(nn.Module):
    """Couple TensorEnvironment2D to DynamicSparseLinear with no rendering path."""

    def __init__(
        self,
        environment: TensorEnvironment2D,
        brain: DynamicSparseLinear,
        transducer: VectorizedTransducer | None = None,
    ) -> None:
        _require_torch()
        super().__init__()
        if brain.in_features != brain.out_features:
            raise ValueError("HeadlessAMMCLoop expects a recurrent square brain: in_features == out_features")
        self.environment = environment
        self.brain = brain
        self.transducer = transducer or VectorizedTransducer(
            TransducerConfig(neuron_count=brain.in_features)
        )
        self.register_buffer(
            "membrane",
            torch.zeros(
                (environment.config.agent_count, brain.in_features),
                device=environment.agent_pos.device,
                dtype=environment.agent_pos.dtype,
            ),
        )

    def step(self, generator=None):
        """Run one environment->brain->environment tick."""

        sensory = self.environment.sensory_tensor()
        neural_input = self.transducer.encode_sensors(sensory)
        recurrent_current = self.brain(self.membrane)
        spikes, self.membrane = self.transducer.lif_step(neural_input + recurrent_current, self.membrane)
        action = self.transducer.decode_motors(spikes)
        world = self.environment.step(action, generator=generator)
        mark_step(self.environment.agent_pos.device)
        return {
            "sensory": sensory,
            "spikes": spikes,
            "action": action,
            **world,
        }
