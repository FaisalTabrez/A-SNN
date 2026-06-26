"""Central evolutionary runtime loop for AMMC Gen-5.

Sprint 8 binds:

Environment Physics -> Sensors -> TensorEvolver brains -> Motors -> Environment

At fixed epoch boundaries the loop ranks environment fitness, evolves the
batched genome pool, clears neural state, and resets the environment for the
next generation.
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

from .evolver import TensorEvolver
from .runtime import mark_step
from .tensor_environment import TensorEnvironment2D
from .telemetry import EvolutionTelemetryLogger
from .transducer import TransducerConfig, VectorizedTransducer


def _require_torch() -> None:
    if torch is None:
        raise ImportError("AMMC Gen-5 evolving loop requires PyTorch")


@dataclass(frozen=True)
class EvolvingLoopConfig:
    """Configuration for the tensorized evolutionary loop."""

    epoch_steps: int = 3_600
    reset_membrane_on_epoch: bool = True


class EvolvingHeadlessAMMCLoop(nn.Module):
    """Full headless embodied evolutionary cycle.

    This loop uses `TensorEvolver` as the per-agent sparse recurrent brain.
    Unlike `HeadlessAMMCLoop`, each agent has its own independently mutable
    genome row in the evolver's batched edge pool.
    """

    def __init__(
        self,
        environment: TensorEnvironment2D,
        evolver: TensorEvolver,
        transducer: VectorizedTransducer | None = None,
        config: EvolvingLoopConfig | None = None,
        logger: EvolutionTelemetryLogger | None = None,
    ) -> None:
        _require_torch()
        super().__init__()
        if environment.config.agent_count != evolver.population_size:
            raise ValueError("environment agent_count must equal evolver population_size")
        self.environment = environment
        self.evolver = evolver
        self.transducer = transducer or VectorizedTransducer(
            TransducerConfig(neuron_count=evolver.neuron_count)
        )
        if self.transducer.config.neuron_count != evolver.neuron_count:
            raise ValueError("transducer neuron_count must equal evolver neuron_count")
        self.config = config or EvolvingLoopConfig()
        if self.config.epoch_steps <= 0:
            raise ValueError("epoch_steps must be positive")

        self.register_buffer(
            "membrane",
            torch.zeros(
                (environment.config.agent_count, evolver.neuron_count),
                device=environment.agent_pos.device,
                dtype=environment.agent_pos.dtype,
            ),
        )
        self.register_buffer(
            "epoch_step",
            torch.zeros((), dtype=torch.long, device=environment.agent_pos.device),
        )
        self.register_buffer(
            "generation",
            torch.ones((), dtype=torch.long, device=environment.agent_pos.device),
        )
        self.logger = logger
        self.last_epoch_report: dict | None = None
        self.best_fitness: float = float("-inf")
        self.best_generation: int | None = None
        self.best_genome_snapshot: dict | None = None
        self._epoch_step_host = 0
        self._generation_host = 1

    def step(self, generator=None) -> dict:
        """Run one vectorized embodied neural tick."""

        sensory = self.environment.sensory_tensor()
        neural_input = self.transducer.encode_sensors(sensory)
        recurrent_current = self.evolver(self.membrane)
        spikes, membrane = self.transducer.lif_step(neural_input + recurrent_current, self.membrane)
        self.membrane.copy_(membrane)

        action = self.transducer.decode_motors(spikes)
        world = self.environment.step(action, generator=generator)
        self.epoch_step.add_(1)
        self._epoch_step_host += 1

        epoch_report = None
        if self._epoch_step_host >= self.config.epoch_steps:
            epoch_report = self.finish_epoch(generator=generator)
        mark_step(self.environment.agent_pos.device)

        return {
            "sensory": sensory,
            "spikes": spikes,
            "action": action,
            "epoch_step": self._epoch_step_host,
            "generation": self._generation_host,
            "epoch_report": epoch_report,
            **world,
        }

    def benchmark_tick(self, generator=None):
        """Run one control-free tensor tick for throughput benchmarks.

        ``step()`` intentionally mixes the tensor hot path with Python-side
        epoch bookkeeping and a diagnostics dictionary. That is useful for
        training telemetry, but it makes ``torch.compile`` specialize on host
        counters such as ``_epoch_step_host`` and can trigger repeated
        recompilation. This method keeps only the per-tick tensor work used by
        the throughput benchmark: environment sensing, sparse recurrent brain
        update, motor decoding, physics, and a tensor epoch counter increment.

        Epoch evolution, champion snapshots, logger writes, and host integer
        counters are deliberately skipped.
        """

        with torch.no_grad():
            sensory = self.environment.sensory_tensor()
            neural_input = self.transducer.encode_sensors(sensory)
            recurrent_current = self.evolver(self.membrane)
            spikes, membrane = self.transducer.lif_step(neural_input + recurrent_current, self.membrane)
            self.membrane.copy_(membrane)

            action = self.transducer.decode_motors(spikes)
            self.environment.step(action, generator=generator)
            self.epoch_step.add_(1)
            mark_step(self.environment.agent_pos.device)
            return action

    def finish_epoch(self, generator=None) -> dict:
        """Evolve the population from environment fitness and reset positions."""

        fitness = self.environment.fitness.detach().clone()
        champion = self._update_all_time_champion(fitness)
        report = self.evolver.evolve(fitness, generator=generator)
        self.environment.reset(generator=generator)
        if self.config.reset_membrane_on_epoch:
            self.membrane.zero_()
        self.epoch_step.zero_()
        self.generation.add_(1)
        completed_generation = self._generation_host
        self._epoch_step_host = 0
        self._generation_host += 1
        report = {
            **report,
            "completed_generation": completed_generation,
            "next_generation": self._generation_host,
            "all_time_best_fitness": self.best_fitness,
            "all_time_best_generation": self.best_generation,
            "epoch_champion_index": champion["index"],
            "epoch_champion_fitness": champion["fitness"],
        }
        if self.logger is not None:
            record = self.logger.log_epoch(report, self.evolver)
            report["telemetry_record"] = record
        self.last_epoch_report = report
        return report

    def run(self, steps: int, *, generator=None) -> dict:
        """Run multiple steps and return the final telemetry."""

        if steps <= 0:
            raise ValueError("steps must be positive")
        telemetry = {}
        for _ in range(steps):
            telemetry = self.step(generator=generator)
        return telemetry

    def _update_all_time_champion(self, fitness) -> dict:
        best_index_tensor = fitness.argmax()
        best_index = int(best_index_tensor.item())
        best_fitness = float(fitness[best_index].detach().cpu().item())
        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            self.best_generation = self._generation_host
            self.best_genome_snapshot = self.evolver.snapshot_genome(best_index, to_cpu=True)
        return {"index": best_index, "fitness": best_fitness}
