"""Statistical evaluation and ablation runners for AMMC Gen-5.

This module marks the transition from qualitative sandbox observation to
repeatable evidence. It deliberately keeps the runtime headless and
Colab-friendly: JSON/CSV work with the standard library, while plotting imports
matplotlib lazily only when requested.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

try:  # pragma: no cover
    import torch
except Exception:  # pragma: no cover
    torch = None

from .dynamic_sparse import EdgeRecord
from .evolving_loop import EvolvingHeadlessAMMCLoop, EvolvingLoopConfig
from .evolver import TensorEvolver, TensorEvolverConfig
from .runtime import make_generator, resolve_device, seed_everything
from .telemetry import EvolutionTelemetryLogger
from .tensor_environment import TensorEnvironment2D, TensorEnvironmentConfig
from .transducer import TransducerConfig, VectorizedTransducer


def _require_torch() -> None:
    if torch is None:
        raise ImportError("AMMC Gen-5 evaluation runners require PyTorch")


def default_foraging_seed_edges() -> tuple[EdgeRecord, ...]:
    """Return the hand-coded prior used by the early 10k-agent Colab runs."""

    return (
        EdgeRecord(0, 8, long_term_weight=0.8),
        EdgeRecord(1, 9, long_term_weight=0.8),
        EdgeRecord(2, 10, long_term_weight=0.8),
        EdgeRecord(3, 11, long_term_weight=0.8),
        EdgeRecord(4, 8, long_term_weight=0.6, sign=-1),
        EdgeRecord(5, 9, long_term_weight=0.6, sign=-1),
        EdgeRecord(6, 10, long_term_weight=0.6, sign=-1),
        EdgeRecord(7, 11, long_term_weight=0.6, sign=-1),
    )


@dataclass(frozen=True)
class TrialRunnerConfig:
    """Configuration for repeated independent evolution trials."""

    seeds: tuple[int, ...] = tuple(range(42, 52))
    generations: int = 500
    epoch_steps: int = 120
    population_size: int = 10_000
    neuron_count: int = 16
    max_edges: int = 128
    food_count: int = 128
    toxin_count: int = 128
    sensor_radius: float = 0.35
    friction: float = 0.985
    action_gain: float = 0.05
    survivor_fraction: float = 0.5
    ltw_noise_std: float = 0.02
    sprout_probability: float = 0.02
    prune_probability: float = 0.01
    device: str = "auto"
    seed_edges: tuple[EdgeRecord, ...] = field(default_factory=default_foraging_seed_edges)


@dataclass(frozen=True)
class TrialGenerationRecord:
    """One generation-level row from one trial."""

    seed: int
    generation: int
    epoch_best_fitness: float
    all_time_best_fitness: float
    mean_population_fitness: float
    mean_active_synapses: float
    sprout_count: int
    prune_count: int
    ltw_mutation_count: int


@dataclass(frozen=True)
class AggregateGenerationRecord:
    """Mean/std curve point across multiple seeds."""

    generation: int
    mean_best_fitness: float
    std_best_fitness: float
    min_best_fitness: float
    max_best_fitness: float
    mean_population_fitness: float
    mean_active_synapses: float


@dataclass(frozen=True)
class StatisticalTrialResult:
    """Complete multi-seed evaluation output."""

    config: dict
    trial_records: list[TrialGenerationRecord]
    aggregate_records: list[AggregateGenerationRecord]

    def to_json_dict(self) -> dict:
        return {
            "config": self.config,
            "trial_records": [asdict(row) for row in self.trial_records],
            "aggregate_records": [asdict(row) for row in self.aggregate_records],
        }


class TrialRunner:
    """Run independent evolutionary trials and aggregate convergence curves."""

    def __init__(self, config: TrialRunnerConfig | None = None) -> None:
        self.config = config or TrialRunnerConfig()
        if not self.config.seeds:
            raise ValueError("at least one seed is required")
        if self.config.generations <= 0:
            raise ValueError("generations must be positive")
        if self.config.epoch_steps <= 0:
            raise ValueError("epoch_steps must be positive")

    def run(self) -> StatisticalTrialResult:
        _require_torch()
        records: list[TrialGenerationRecord] = []
        for seed in self.config.seeds:
            records.extend(self.run_trial(seed))
        aggregate = aggregate_trial_records(records)
        return StatisticalTrialResult(
            config=_jsonable_config(self.config),
            trial_records=records,
            aggregate_records=aggregate,
        )

    def run_trial(self, seed: int) -> list[TrialGenerationRecord]:
        _require_torch()
        loop, generator = self._build_loop(seed)
        rows: list[TrialGenerationRecord] = []
        for _ in range(self.config.generations):
            loop.run(self.config.epoch_steps, generator=generator)
            report = loop.last_epoch_report
            if report is None:
                raise RuntimeError("loop did not emit an epoch report")
            rows.append(
                TrialGenerationRecord(
                    seed=int(seed),
                    generation=int(report["completed_generation"]),
                    epoch_best_fitness=_float(report.get("best_fitness", 0.0)),
                    all_time_best_fitness=_float(report.get("all_time_best_fitness", 0.0)),
                    mean_population_fitness=_float(report.get("mean_fitness", 0.0)),
                    mean_active_synapses=_float(loop.evolver.active_edge_counts().float().mean()),
                    sprout_count=int(report.get("sprout_count", 0)),
                    prune_count=int(report.get("prune_count", 0)),
                    ltw_mutation_count=int(report.get("ltw_mutation_count", 0)),
                )
            )
        return rows

    def save_outputs(
        self,
        result: StatisticalTrialResult,
        output_dir: str | Path,
        *,
        plot: bool = True,
    ) -> dict[str, str]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        json_path = output / "multi_seed_trials.json"
        json_path.write_text(json.dumps(result.to_json_dict(), indent=2) + "\n", encoding="utf-8")

        trial_csv = output / "multi_seed_trials.csv"
        _write_csv(trial_csv, [asdict(row) for row in result.trial_records])

        aggregate_csv = output / "multi_seed_aggregate.csv"
        _write_csv(aggregate_csv, [asdict(row) for row in result.aggregate_records])

        paths = {
            "json": str(json_path),
            "trial_csv": str(trial_csv),
            "aggregate_csv": str(aggregate_csv),
        }
        if plot:
            try:
                plot_path = output / "multi_seed_best_fitness_mean_std.png"
                plot_multi_seed_result(result, plot_path)
                paths["plot"] = str(plot_path)
            except Exception as exc:  # pragma: no cover - optional plotting
                paths["plot"] = f"skipped: {exc}"
        return paths

    def _build_loop(self, seed: int):
        device = _resolve_device(self.config.device)
        generator = _make_generator(seed, device)
        environment = TensorEnvironment2D(
            TensorEnvironmentConfig(
                agent_count=self.config.population_size,
                food_count=self.config.food_count,
                toxin_count=self.config.toxin_count,
                sensor_radius=self.config.sensor_radius,
                friction=self.config.friction,
                action_gain=self.config.action_gain,
            ),
            device=device,
        )
        environment.reset(generator=generator)
        evolver = TensorEvolver(
            TensorEvolverConfig(
                population_size=self.config.population_size,
                neuron_count=self.config.neuron_count,
                max_edges=self.config.max_edges,
                survivor_fraction=self.config.survivor_fraction,
                ltw_noise_std=self.config.ltw_noise_std,
                sprout_probability=self.config.sprout_probability,
                prune_probability=self.config.prune_probability,
            ),
            device=device,
        )
        evolver.seed_from_edges(self.config.seed_edges)
        logger = EvolutionTelemetryLogger()
        loop = EvolvingHeadlessAMMCLoop(
            environment,
            evolver,
            VectorizedTransducer(TransducerConfig(neuron_count=self.config.neuron_count)),
            EvolvingLoopConfig(epoch_steps=self.config.epoch_steps),
            logger=logger,
        ).to(device)
        return loop, generator


@dataclass(frozen=True)
class AblationGroupConfig:
    """One seeded plasticity ablation group."""

    name: str
    description: str
    sprout_probability: float
    prune_probability: float
    ltw_noise_std: float
    invert_food_toxin_sensors: bool = True
    gate_ltw_noise_by_positive_fitness: bool = False
    gate_pruning_by_positive_fitness: bool = False
    gate_sprouting_by_positive_fitness: bool = False
    plasticity_reward_threshold: float = 0.0


def default_ablation_groups() -> tuple[AblationGroupConfig, ...]:
    """Return the static/full/adult groups from the Phase 11 objective."""

    return (
        AblationGroupConfig(
            name="static_snn",
            description="Topology and weights locked after import.",
            sprout_probability=0.0,
            prune_probability=0.0,
            ltw_noise_std=0.0,
        ),
        AblationGroupConfig(
            name="full_plasticity_infant",
            description="Aggressive mutation/plasticity; adapts quickly but risks forgetting.",
            sprout_probability=0.02,
            prune_probability=0.01,
            ltw_noise_std=0.02,
        ),
        AblationGroupConfig(
            name="gated_plasticity_adult",
            description="Sprouting allowed; pruning/LTW noise gated behind positive reward.",
            sprout_probability=0.01,
            prune_probability=0.01,
            ltw_noise_std=0.01,
            gate_ltw_noise_by_positive_fitness=True,
            gate_pruning_by_positive_fitness=True,
            plasticity_reward_threshold=0.0,
        ),
    )


@dataclass(frozen=True)
class PlasticityAblationConfig:
    """Configuration for static/full/gated plasticity A/B trials."""

    seeds: tuple[int, ...] = tuple(range(42, 52))
    generations: int = 500
    epoch_steps: int = 120
    population_size: int = 10_000
    neuron_count: int = 16
    max_edges: int = 128
    food_count: int = 128
    toxin_count: int = 128
    sensor_radius: float = 0.35
    friction: float = 0.985
    action_gain: float = 0.05
    survivor_fraction: float = 0.5
    adaptation_fitness_threshold: float = 1.0
    device: str = "auto"
    seed_edges: tuple[EdgeRecord, ...] = field(default_factory=default_foraging_seed_edges)
    groups: tuple[AblationGroupConfig, ...] = field(default_factory=default_ablation_groups)


@dataclass(frozen=True)
class AblationGenerationRecord:
    group: str
    seed: int
    generation: int
    epoch_best_fitness: float
    all_time_best_fitness: float
    mean_population_fitness: float
    mean_active_synapses: float
    sprout_count: int
    prune_count: int
    ltw_mutation_count: int


@dataclass(frozen=True)
class AblationSummaryRecord:
    group: str
    seeds: int
    final_mean_best_fitness: float
    final_std_best_fitness: float
    final_mean_active_synapses: float
    adaptation_success_rate: float
    mean_adaptation_generation: float | None


@dataclass(frozen=True)
class PlasticityAblationResult:
    config: dict
    records: list[AblationGenerationRecord]
    summary: list[AblationSummaryRecord]

    def to_json_dict(self) -> dict:
        return {
            "config": self.config,
            "records": [asdict(row) for row in self.records],
            "summary": [asdict(row) for row in self.summary],
        }


class SensorPerturbationTransducer(VectorizedTransducer):
    """Transducer that can invert food/toxin channels for adaptation tests."""

    def __init__(self, config: TransducerConfig | None = None, *, invert_food_toxin: bool = False) -> None:
        super().__init__(config)
        self.invert_food_toxin = invert_food_toxin

    def encode_sensors(self, sensory_tensor):
        if self.invert_food_toxin:
            sensory_tensor = torch.cat([sensory_tensor[:, 4:8], sensory_tensor[:, :4]], dim=1)
        return super().encode_sensors(sensory_tensor)


class PlasticityAblationRunner:
    """Run the static/full/gated plasticity ablation study."""

    def __init__(self, config: PlasticityAblationConfig | None = None) -> None:
        self.config = config or PlasticityAblationConfig()
        if not self.config.seeds:
            raise ValueError("at least one seed is required")
        if not self.config.groups:
            raise ValueError("at least one ablation group is required")

    def run(self) -> PlasticityAblationResult:
        _require_torch()
        records: list[AblationGenerationRecord] = []
        for group in self.config.groups:
            for seed in self.config.seeds:
                records.extend(self.run_group_trial(group, seed))
        summary = summarize_ablation_records(
            records,
            groups=[group.name for group in self.config.groups],
            threshold=self.config.adaptation_fitness_threshold,
        )
        return PlasticityAblationResult(
            config=_jsonable_config(self.config),
            records=records,
            summary=summary,
        )

    def run_group_trial(self, group: AblationGroupConfig, seed: int) -> list[AblationGenerationRecord]:
        loop, generator = self._build_loop(group, seed)
        rows: list[AblationGenerationRecord] = []
        for _ in range(self.config.generations):
            loop.run(self.config.epoch_steps, generator=generator)
            report = loop.last_epoch_report
            if report is None:
                raise RuntimeError("loop did not emit an epoch report")
            rows.append(
                AblationGenerationRecord(
                    group=group.name,
                    seed=int(seed),
                    generation=int(report["completed_generation"]),
                    epoch_best_fitness=_float(report.get("best_fitness", 0.0)),
                    all_time_best_fitness=_float(report.get("all_time_best_fitness", 0.0)),
                    mean_population_fitness=_float(report.get("mean_fitness", 0.0)),
                    mean_active_synapses=_float(loop.evolver.active_edge_counts().float().mean()),
                    sprout_count=int(report.get("sprout_count", 0)),
                    prune_count=int(report.get("prune_count", 0)),
                    ltw_mutation_count=int(report.get("ltw_mutation_count", 0)),
                )
            )
        return rows

    def save_outputs(
        self,
        result: PlasticityAblationResult,
        output_dir: str | Path,
        *,
        plot: bool = True,
    ) -> dict[str, str]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        json_path = output / "plasticity_ablation.json"
        json_path.write_text(json.dumps(result.to_json_dict(), indent=2) + "\n", encoding="utf-8")

        records_csv = output / "plasticity_ablation_records.csv"
        _write_csv(records_csv, [asdict(row) for row in result.records])

        summary_csv = output / "plasticity_ablation_summary.csv"
        _write_csv(summary_csv, [asdict(row) for row in result.summary])

        paths = {"json": str(json_path), "records_csv": str(records_csv), "summary_csv": str(summary_csv)}
        if plot:
            try:
                plot_path = output / "plasticity_ablation_best_fitness.png"
                plot_ablation_result(result, plot_path)
                paths["plot"] = str(plot_path)
            except Exception as exc:  # pragma: no cover
                paths["plot"] = f"skipped: {exc}"
        return paths

    def _build_loop(self, group: AblationGroupConfig, seed: int):
        device = _resolve_device(self.config.device)
        generator = _make_generator(seed, device)
        environment = TensorEnvironment2D(
            TensorEnvironmentConfig(
                agent_count=self.config.population_size,
                food_count=self.config.food_count,
                toxin_count=self.config.toxin_count,
                sensor_radius=self.config.sensor_radius,
                friction=self.config.friction,
                action_gain=self.config.action_gain,
            ),
            device=device,
        )
        environment.reset(generator=generator)
        evolver = TensorEvolver(
            TensorEvolverConfig(
                population_size=self.config.population_size,
                neuron_count=self.config.neuron_count,
                max_edges=self.config.max_edges,
                survivor_fraction=self.config.survivor_fraction,
                ltw_noise_std=group.ltw_noise_std,
                sprout_probability=group.sprout_probability,
                prune_probability=group.prune_probability,
                gate_ltw_noise_by_positive_fitness=group.gate_ltw_noise_by_positive_fitness,
                gate_pruning_by_positive_fitness=group.gate_pruning_by_positive_fitness,
                gate_sprouting_by_positive_fitness=group.gate_sprouting_by_positive_fitness,
                plasticity_reward_threshold=group.plasticity_reward_threshold,
            ),
            device=device,
        )
        evolver.seed_from_edges(self.config.seed_edges)
        loop = EvolvingHeadlessAMMCLoop(
            environment,
            evolver,
            SensorPerturbationTransducer(
                TransducerConfig(neuron_count=self.config.neuron_count),
                invert_food_toxin=group.invert_food_toxin_sensors,
            ),
            EvolvingLoopConfig(epoch_steps=self.config.epoch_steps),
            logger=EvolutionTelemetryLogger(),
        ).to(device)
        return loop, generator


@dataclass(frozen=True)
class RetentionAblationConfig:
    """Configuration for original -> perturbed -> original retention trials."""

    seeds: tuple[int, ...] = tuple(range(42, 52))
    original_generations: int = 100
    perturbation_generations: int = 300
    recovery_generations: int = 100
    epoch_steps: int = 120
    population_size: int = 10_000
    neuron_count: int = 16
    max_edges: int = 128
    food_count: int = 128
    toxin_count: int = 128
    sensor_radius: float = 0.35
    friction: float = 0.985
    action_gain: float = 0.05
    survivor_fraction: float = 0.5
    device: str = "auto"
    seed_edges: tuple[EdgeRecord, ...] = field(default_factory=default_foraging_seed_edges)
    groups: tuple[AblationGroupConfig, ...] = field(default_factory=default_ablation_groups)


@dataclass(frozen=True)
class RetentionGenerationRecord:
    group: str
    seed: int
    phase: str
    phase_generation: int
    global_generation: int
    epoch_best_fitness: float
    phase_best_fitness: float
    mean_population_fitness: float
    mean_active_synapses: float
    sprout_count: int
    prune_count: int
    ltw_mutation_count: int


@dataclass(frozen=True)
class RetentionSummaryRecord:
    group: str
    seeds: int
    original_final_epoch_best: float
    perturbation_peak_best: float
    recovery_final_epoch_best: float
    recovery_retention_ratio: float | None
    forgetting_delta: float
    perturbation_gain_over_original: float
    final_mean_active_synapses: float


@dataclass(frozen=True)
class RetentionAblationResult:
    config: dict
    records: list[RetentionGenerationRecord]
    summary: list[RetentionSummaryRecord]

    def to_json_dict(self) -> dict:
        return {
            "config": self.config,
            "records": [asdict(row) for row in self.records],
            "summary": [asdict(row) for row in self.summary],
        }


class RetentionAblationRunner:
    """Run original -> perturbed -> original trials to measure forgetting."""

    phases = (
        ("original", False, "original_generations"),
        ("perturbed", True, "perturbation_generations"),
        ("recovery", False, "recovery_generations"),
    )

    def __init__(self, config: RetentionAblationConfig | None = None) -> None:
        self.config = config or RetentionAblationConfig()
        if not self.config.seeds:
            raise ValueError("at least one seed is required")
        if not self.config.groups:
            raise ValueError("at least one ablation group is required")

    def run(self) -> RetentionAblationResult:
        _require_torch()
        records: list[RetentionGenerationRecord] = []
        for group in self.config.groups:
            for seed in self.config.seeds:
                records.extend(self.run_group_trial(group, seed))
        summary = summarize_retention_records(records, groups=[group.name for group in self.config.groups])
        return RetentionAblationResult(
            config=_jsonable_config(self.config),
            records=records,
            summary=summary,
        )

    def run_group_trial(self, group: AblationGroupConfig, seed: int) -> list[RetentionGenerationRecord]:
        loop, generator = self._build_loop(group, seed)
        rows: list[RetentionGenerationRecord] = []
        global_generation = 0
        for phase, inverted, generation_attr in self.phases:
            loop.transducer.invert_food_toxin = inverted
            phase_best = float("-inf")
            phase_generations = int(getattr(self.config, generation_attr))
            for phase_generation in range(1, phase_generations + 1):
                loop.run(self.config.epoch_steps, generator=generator)
                report = loop.last_epoch_report
                if report is None:
                    raise RuntimeError("loop did not emit an epoch report")
                global_generation += 1
                epoch_best = _float(report.get("best_fitness", 0.0))
                phase_best = max(phase_best, epoch_best)
                rows.append(
                    RetentionGenerationRecord(
                        group=group.name,
                        seed=int(seed),
                        phase=phase,
                        phase_generation=phase_generation,
                        global_generation=global_generation,
                        epoch_best_fitness=epoch_best,
                        phase_best_fitness=phase_best,
                        mean_population_fitness=_float(report.get("mean_fitness", 0.0)),
                        mean_active_synapses=_float(loop.evolver.active_edge_counts().float().mean()),
                        sprout_count=int(report.get("sprout_count", 0)),
                        prune_count=int(report.get("prune_count", 0)),
                        ltw_mutation_count=int(report.get("ltw_mutation_count", 0)),
                    )
                )
        return rows

    def save_outputs(
        self,
        result: RetentionAblationResult,
        output_dir: str | Path,
        *,
        plot: bool = True,
    ) -> dict[str, str]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        json_path = output / "retention_ablation.json"
        json_path.write_text(json.dumps(result.to_json_dict(), indent=2) + "\n", encoding="utf-8")

        records_csv = output / "retention_ablation_records.csv"
        _write_csv(records_csv, [asdict(row) for row in result.records])

        summary_csv = output / "retention_ablation_summary.csv"
        _write_csv(summary_csv, [asdict(row) for row in result.summary])

        paths = {"json": str(json_path), "records_csv": str(records_csv), "summary_csv": str(summary_csv)}
        if plot:
            try:
                plot_path = output / "retention_ablation_phase_fitness.png"
                plot_retention_result(result, plot_path)
                paths["plot"] = str(plot_path)
            except Exception as exc:  # pragma: no cover
                paths["plot"] = f"skipped: {exc}"
        return paths

    def _build_loop(self, group: AblationGroupConfig, seed: int):
        device = _resolve_device(self.config.device)
        generator = _make_generator(seed, device)
        environment = TensorEnvironment2D(
            TensorEnvironmentConfig(
                agent_count=self.config.population_size,
                food_count=self.config.food_count,
                toxin_count=self.config.toxin_count,
                sensor_radius=self.config.sensor_radius,
                friction=self.config.friction,
                action_gain=self.config.action_gain,
            ),
            device=device,
        )
        environment.reset(generator=generator)
        evolver = TensorEvolver(
            TensorEvolverConfig(
                population_size=self.config.population_size,
                neuron_count=self.config.neuron_count,
                max_edges=self.config.max_edges,
                survivor_fraction=self.config.survivor_fraction,
                ltw_noise_std=group.ltw_noise_std,
                sprout_probability=group.sprout_probability,
                prune_probability=group.prune_probability,
                gate_ltw_noise_by_positive_fitness=group.gate_ltw_noise_by_positive_fitness,
                gate_pruning_by_positive_fitness=group.gate_pruning_by_positive_fitness,
                gate_sprouting_by_positive_fitness=group.gate_sprouting_by_positive_fitness,
                plasticity_reward_threshold=group.plasticity_reward_threshold,
            ),
            device=device,
        )
        evolver.seed_from_edges(self.config.seed_edges)
        loop = EvolvingHeadlessAMMCLoop(
            environment,
            evolver,
            SensorPerturbationTransducer(
                TransducerConfig(neuron_count=self.config.neuron_count),
                invert_food_toxin=False,
            ),
            EvolvingLoopConfig(epoch_steps=self.config.epoch_steps),
            logger=EvolutionTelemetryLogger(),
        ).to(device)
        return loop, generator


def aggregate_trial_records(records: Iterable[TrialGenerationRecord]) -> list[AggregateGenerationRecord]:
    grouped: dict[int, list[TrialGenerationRecord]] = {}
    for row in records:
        grouped.setdefault(row.generation, []).append(row)

    output: list[AggregateGenerationRecord] = []
    for generation in sorted(grouped):
        rows = grouped[generation]
        best = [row.all_time_best_fitness for row in rows]
        mean_fitness = [row.mean_population_fitness for row in rows]
        active = [row.mean_active_synapses for row in rows]
        output.append(
            AggregateGenerationRecord(
                generation=generation,
                mean_best_fitness=_mean(best),
                std_best_fitness=_std(best),
                min_best_fitness=min(best),
                max_best_fitness=max(best),
                mean_population_fitness=_mean(mean_fitness),
                mean_active_synapses=_mean(active),
            )
        )
    return output


def summarize_ablation_records(
    records: list[AblationGenerationRecord],
    *,
    groups: Iterable[str],
    threshold: float,
) -> list[AblationSummaryRecord]:
    summaries: list[AblationSummaryRecord] = []
    for group in groups:
        group_rows = [row for row in records if row.group == group]
        seeds = sorted({row.seed for row in group_rows})
        final_rows = [
            max((row for row in group_rows if row.seed == seed), key=lambda row: row.generation)
            for seed in seeds
        ]
        adaptation_generations: list[int] = []
        for seed in seeds:
            seed_rows = sorted((row for row in group_rows if row.seed == seed), key=lambda row: row.generation)
            reached = next((row.generation for row in seed_rows if row.all_time_best_fitness >= threshold), None)
            if reached is not None:
                adaptation_generations.append(reached)

        final_best = [row.all_time_best_fitness for row in final_rows]
        final_active = [row.mean_active_synapses for row in final_rows]
        summaries.append(
            AblationSummaryRecord(
                group=group,
                seeds=len(seeds),
                final_mean_best_fitness=_mean(final_best) if final_best else 0.0,
                final_std_best_fitness=_std(final_best) if final_best else 0.0,
                final_mean_active_synapses=_mean(final_active) if final_active else 0.0,
                adaptation_success_rate=(len(adaptation_generations) / len(seeds)) if seeds else 0.0,
                mean_adaptation_generation=_mean(adaptation_generations) if adaptation_generations else None,
            )
        )
    return summaries


def summarize_retention_records(
    records: list[RetentionGenerationRecord],
    *,
    groups: Iterable[str],
) -> list[RetentionSummaryRecord]:
    summaries: list[RetentionSummaryRecord] = []
    for group in groups:
        group_rows = [row for row in records if row.group == group]
        seeds = sorted({row.seed for row in group_rows})
        original_final: list[float] = []
        perturbed_peak: list[float] = []
        recovery_final: list[float] = []
        final_active: list[float] = []
        for seed in seeds:
            seed_rows = [row for row in group_rows if row.seed == seed]
            original_rows = [row for row in seed_rows if row.phase == "original"]
            perturbed_rows = [row for row in seed_rows if row.phase == "perturbed"]
            recovery_rows = [row for row in seed_rows if row.phase == "recovery"]
            if original_rows:
                original_final.append(max(original_rows, key=lambda row: row.phase_generation).epoch_best_fitness)
            if perturbed_rows:
                perturbed_peak.append(max(row.epoch_best_fitness for row in perturbed_rows))
            if recovery_rows:
                final = max(recovery_rows, key=lambda row: row.phase_generation)
                recovery_final.append(final.epoch_best_fitness)
                final_active.append(final.mean_active_synapses)

        original_mean = _mean(original_final)
        perturbed_mean = _mean(perturbed_peak)
        recovery_mean = _mean(recovery_final)
        summaries.append(
            RetentionSummaryRecord(
                group=group,
                seeds=len(seeds),
                original_final_epoch_best=original_mean,
                perturbation_peak_best=perturbed_mean,
                recovery_final_epoch_best=recovery_mean,
                recovery_retention_ratio=(recovery_mean / original_mean) if original_mean else None,
                forgetting_delta=original_mean - recovery_mean,
                perturbation_gain_over_original=perturbed_mean - original_mean,
                final_mean_active_synapses=_mean(final_active),
            )
        )
    return summaries


def plot_multi_seed_result(result: StatisticalTrialResult, path: str | Path | None = None, *, show: bool = False):
    if not result.aggregate_records:
        raise ValueError("no aggregate records to plot")
    import matplotlib.pyplot as plt

    x = [row.generation for row in result.aggregate_records]
    mean = [row.mean_best_fitness for row in result.aggregate_records]
    std = [row.std_best_fitness for row in result.aggregate_records]
    lo = [m - s for m, s in zip(mean, std)]
    hi = [m + s for m, s in zip(mean, std)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, mean, color="#38bdf8", linewidth=2, label="Mean all-time best fitness")
    ax.fill_between(x, lo, hi, color="#38bdf8", alpha=0.22, label="±1 std")
    ax.set_title("AMMC Gen-5 Multi-Seed Convergence")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if path is not None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160)
    if show:
        plt.show()
    return fig


def plot_ablation_result(result: PlasticityAblationResult, path: str | Path | None = None, *, show: bool = False):
    if not result.records:
        raise ValueError("no ablation records to plot")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    groups = sorted({row.group for row in result.records})
    for group in groups:
        trial_records = [
            TrialGenerationRecord(
                seed=row.seed,
                generation=row.generation,
                epoch_best_fitness=row.epoch_best_fitness,
                all_time_best_fitness=row.all_time_best_fitness,
                mean_population_fitness=row.mean_population_fitness,
                mean_active_synapses=row.mean_active_synapses,
                sprout_count=row.sprout_count,
                prune_count=row.prune_count,
                ltw_mutation_count=row.ltw_mutation_count,
            )
            for row in result.records
            if row.group == group
        ]
        aggregate = aggregate_trial_records(trial_records)
        ax.plot(
            [row.generation for row in aggregate],
            [row.mean_best_fitness for row in aggregate],
            linewidth=2,
            label=group,
        )
    ax.set_title("AMMC Gen-5 Plasticity Ablation")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean all-time best fitness")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if path is not None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160)
    if show:
        plt.show()
    return fig


def plot_retention_result(result: RetentionAblationResult, path: str | Path | None = None, *, show: bool = False):
    if not result.records:
        raise ValueError("no retention records to plot")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    groups = sorted({row.group for row in result.records})
    phase_lengths = {
        phase: max((row.phase_generation for row in result.records if row.phase == phase), default=0)
        for phase in ("original", "perturbed", "recovery")
    }
    offsets = {
        "original": 0,
        "perturbed": phase_lengths["original"],
        "recovery": phase_lengths["original"] + phase_lengths["perturbed"],
    }

    max_x = 0
    for group in groups:
        group_rows = [row for row in result.records if row.group == group]
        points: dict[int, list[float]] = {}
        for row in group_rows:
            x = offsets[row.phase] + row.phase_generation
            max_x = max(max_x, x)
            points.setdefault(x, []).append(row.epoch_best_fitness)
        xs = sorted(points)
        ys = [_mean(points[x]) for x in xs]
        ax.plot(xs, ys, linewidth=2, label=group)

    first_boundary = phase_lengths["original"]
    second_boundary = phase_lengths["original"] + phase_lengths["perturbed"]
    if first_boundary:
        ax.axvline(first_boundary, color="#64748b", linestyle="--", linewidth=1)
    if second_boundary:
        ax.axvline(second_boundary, color="#64748b", linestyle="--", linewidth=1)
    top = ax.get_ylim()[1]
    if first_boundary:
        ax.text(first_boundary / 2, top, "original", ha="center", va="top")
    if second_boundary:
        ax.text((first_boundary + second_boundary) / 2, top, "perturbed", ha="center", va="top")
        ax.text((second_boundary + max_x) / 2, top, "recovery", ha="center", va="top")
    ax.set_title("AMMC Gen-5 Retention / Forgetting Ablation")
    ax.set_xlabel("Global generation")
    ax.set_ylabel("Mean epoch best fitness")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if path is not None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160)
    if show:
        plt.show()
    return fig


def _resolve_device(device: str):
    return resolve_device(device)


def _make_generator(seed: int, device):
    seed_everything(seed, device=device)
    return make_generator(seed, device=device)


def _float(value) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _jsonable_config(config) -> dict:
    def convert(value):
        if isinstance(value, EdgeRecord):
            return asdict(value)
        if hasattr(value, "__dataclass_fields__"):
            return {key: convert(val) for key, val in asdict(value).items()}
        if isinstance(value, (tuple, list)):
            return [convert(item) for item in value]
        if isinstance(value, dict):
            return {key: convert(val) for key, val in value.items()}
        return value

    return convert(config)
