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


def hidden_neuron_count(neuron_count: int) -> int:
    """Return non-sensor/non-motor decision nodes for the default transducer."""

    return max(0, int(neuron_count) - 12)


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
class NeuronScalePoint:
    """One topology capacity point for neuron scaling experiments."""

    neuron_count: int
    max_edges: int


def default_neuron_scale_points() -> tuple[NeuronScalePoint, ...]:
    """Return conservative Colab-scale topology growth points."""

    return (
        NeuronScalePoint(neuron_count=16, max_edges=128),
        NeuronScalePoint(neuron_count=32, max_edges=256),
        NeuronScalePoint(neuron_count=64, max_edges=512),
    )


@dataclass(frozen=True)
class NeuronScalingConfig:
    """Configuration for decision-node scaling sweeps."""

    seeds: tuple[int, ...] = tuple(range(42, 52))
    generations: int = 500
    epoch_steps: int = 120
    population_size: int = 10_000
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
    scale_points: tuple[NeuronScalePoint, ...] = field(default_factory=default_neuron_scale_points)
    adaptation_fitness_threshold: float = 25.0


@dataclass(frozen=True)
class NeuronScalingGenerationRecord:
    """One generation-level row for one seed and one neuron scale point."""

    neuron_count: int
    hidden_neurons: int
    max_edges: int
    seed: int
    generation: int
    epoch_best_fitness: float
    all_time_best_fitness: float
    mean_population_fitness: float
    mean_active_synapses: float
    active_edge_utilization: float
    sprout_count: int
    prune_count: int
    ltw_mutation_count: int


@dataclass(frozen=True)
class NeuronScalingSummaryRecord:
    """Final scaling-law summary for one neuron scale point."""

    neuron_count: int
    hidden_neurons: int
    max_edges: int
    seeds: int
    final_mean_best_fitness: float
    final_std_best_fitness: float
    final_mean_active_synapses: float
    final_active_edge_utilization: float
    final_fitness_per_active_synapse: float
    threshold_success_rate: float
    mean_generation_to_threshold: float | None


@dataclass(frozen=True)
class NeuronScalingResult:
    """Complete neuron-scaling evaluation output."""

    config: dict
    records: list[NeuronScalingGenerationRecord]
    summary: list[NeuronScalingSummaryRecord]

    def to_json_dict(self) -> dict:
        return {
            "config": self.config,
            "records": [asdict(row) for row in self.records],
            "summary": [asdict(row) for row in self.summary],
        }


class NeuronScalingRunner:
    """Run multi-seed experiments across increasing decision-node capacity."""

    def __init__(self, config: NeuronScalingConfig | None = None) -> None:
        self.config = config or NeuronScalingConfig()
        if not self.config.seeds:
            raise ValueError("at least one seed is required")
        if not self.config.scale_points:
            raise ValueError("at least one neuron scale point is required")
        if self.config.generations <= 0:
            raise ValueError("generations must be positive")
        if self.config.epoch_steps <= 0:
            raise ValueError("epoch_steps must be positive")
        for point in self.config.scale_points:
            if point.neuron_count < 12:
                raise ValueError("neuron_count must be at least 12 for the default 8-sensor/4-motor transducer")
            if point.max_edges < len(self.config.seed_edges):
                raise ValueError("max_edges must be at least the seeded edge count")

    def run(self) -> NeuronScalingResult:
        _require_torch()
        records: list[NeuronScalingGenerationRecord] = []
        for point in self.config.scale_points:
            trial_runner = TrialRunner(self._trial_config_for(point))
            for seed in self.config.seeds:
                trial_records = trial_runner.run_trial(seed)
                records.extend(self._convert_records(point, trial_records))

        summary = summarize_neuron_scaling_records(
            records,
            threshold=self.config.adaptation_fitness_threshold,
        )
        return NeuronScalingResult(
            config=_jsonable_config(self.config),
            records=records,
            summary=summary,
        )

    def save_outputs(
        self,
        result: NeuronScalingResult,
        output_dir: str | Path,
        *,
        plot: bool = True,
    ) -> dict[str, str]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        json_path = output / "neuron_scaling.json"
        json_path.write_text(json.dumps(result.to_json_dict(), indent=2) + "\n", encoding="utf-8")

        records_csv = output / "neuron_scaling_records.csv"
        _write_csv(records_csv, [asdict(row) for row in result.records])

        summary_csv = output / "neuron_scaling_summary.csv"
        _write_csv(summary_csv, [asdict(row) for row in result.summary])

        paths = {
            "json": str(json_path),
            "records_csv": str(records_csv),
            "summary_csv": str(summary_csv),
        }
        if plot:
            try:
                plot_path = output / "neuron_scaling_summary.png"
                plot_neuron_scaling_result(result, plot_path)
                paths["plot"] = str(plot_path)
            except Exception as exc:  # pragma: no cover - optional plotting
                paths["plot"] = f"skipped: {exc}"
        return paths

    def _trial_config_for(self, point: NeuronScalePoint) -> TrialRunnerConfig:
        return TrialRunnerConfig(
            seeds=self.config.seeds,
            generations=self.config.generations,
            epoch_steps=self.config.epoch_steps,
            population_size=self.config.population_size,
            neuron_count=point.neuron_count,
            max_edges=point.max_edges,
            food_count=self.config.food_count,
            toxin_count=self.config.toxin_count,
            sensor_radius=self.config.sensor_radius,
            friction=self.config.friction,
            action_gain=self.config.action_gain,
            survivor_fraction=self.config.survivor_fraction,
            ltw_noise_std=self.config.ltw_noise_std,
            sprout_probability=self.config.sprout_probability,
            prune_probability=self.config.prune_probability,
            device=self.config.device,
            seed_edges=self.config.seed_edges,
        )

    def _convert_records(
        self,
        point: NeuronScalePoint,
        rows: list[TrialGenerationRecord],
    ) -> list[NeuronScalingGenerationRecord]:
        hidden = hidden_neuron_count(point.neuron_count)
        return [
            NeuronScalingGenerationRecord(
                neuron_count=point.neuron_count,
                hidden_neurons=hidden,
                max_edges=point.max_edges,
                seed=row.seed,
                generation=row.generation,
                epoch_best_fitness=row.epoch_best_fitness,
                all_time_best_fitness=row.all_time_best_fitness,
                mean_population_fitness=row.mean_population_fitness,
                mean_active_synapses=row.mean_active_synapses,
                active_edge_utilization=(row.mean_active_synapses / point.max_edges) if point.max_edges else 0.0,
                sprout_count=row.sprout_count,
                prune_count=row.prune_count,
                ltw_mutation_count=row.ltw_mutation_count,
            )
            for row in rows
        ]


@dataclass(frozen=True)
class SparseEfficiencyGroupConfig:
    """One sparse-efficiency ablation group."""

    name: str
    description: str
    active_edge_fitness_penalty: float = 0.0
    sprout_probability: float = 0.02
    prune_probability: float = 0.01
    ltw_noise_std: float = 0.02
    low_ltw_prune_threshold: float = 0.0
    low_ltw_prune_probability: float = 0.0
    sprout_scale_by_capacity: bool = False
    protect_core: bool = False
    protect_core_topology: bool = True
    protect_core_weights: bool = True


def default_sparse_efficiency_groups() -> tuple[SparseEfficiencyGroupConfig, ...]:
    """Return sparse-efficiency groups that isolate each proposed optimization."""

    return (
        SparseEfficiencyGroupConfig(
            name="baseline_capacity_fill",
            description="Current mutation schedule; no explicit pressure against extra edges.",
        ),
        SparseEfficiencyGroupConfig(
            name="active_edge_penalty",
            description="Rank organisms by raw fitness minus active-edge metabolic cost.",
            active_edge_fitness_penalty=0.015,
        ),
        SparseEfficiencyGroupConfig(
            name="low_ltw_pruning",
            description="Add stronger pruning pressure for weak long-term edges.",
            low_ltw_prune_threshold=0.08,
            low_ltw_prune_probability=0.05,
        ),
        SparseEfficiencyGroupConfig(
            name="scheduled_sprouting",
            description="Reduce sprout probability as edge-pool capacity grows.",
            sprout_scale_by_capacity=True,
        ),
        SparseEfficiencyGroupConfig(
            name="protected_sparse_core",
            description="Protect seeded core pathways while applying edge cost, weak-edge pruning, and scheduled sprouting.",
            active_edge_fitness_penalty=0.015,
            low_ltw_prune_threshold=0.08,
            low_ltw_prune_probability=0.05,
            sprout_scale_by_capacity=True,
            protect_core=True,
        ),
    )


@dataclass(frozen=True)
class SparseEfficiencyConfig:
    """Configuration for sparse-efficiency ablations across neuron scale points."""

    seeds: tuple[int, ...] = tuple(range(42, 52))
    generations: int = 500
    epoch_steps: int = 120
    population_size: int = 10_000
    food_count: int = 128
    toxin_count: int = 128
    sensor_radius: float = 0.35
    friction: float = 0.985
    action_gain: float = 0.05
    survivor_fraction: float = 0.5
    reference_max_edges: int = 128
    adaptation_fitness_threshold: float = 25.0
    device: str = "auto"
    seed_edges: tuple[EdgeRecord, ...] = field(default_factory=default_foraging_seed_edges)
    scale_points: tuple[NeuronScalePoint, ...] = field(default_factory=default_neuron_scale_points)
    groups: tuple[SparseEfficiencyGroupConfig, ...] = field(default_factory=default_sparse_efficiency_groups)
    protected_core_edge_count: int | None = None


@dataclass(frozen=True)
class SparseEfficiencyGenerationRecord:
    """One generation-level row for sparse-efficiency evaluation."""

    group: str
    neuron_count: int
    hidden_neurons: int
    max_edges: int
    seed: int
    generation: int
    epoch_best_fitness: float
    all_time_best_fitness: float
    selection_best_fitness: float
    mean_population_fitness: float
    selection_mean_fitness: float
    mean_active_synapses: float
    active_edge_utilization: float
    fitness_per_active_synapse: float
    mean_hidden_edges: float
    mean_hidden_edge_fraction: float
    mean_direct_sensor_motor_fraction: float
    sprout_count: int
    prune_count: int
    low_ltw_prune_count: int
    ltw_mutation_count: int


@dataclass(frozen=True)
class SparseEfficiencySummaryRecord:
    """Final sparse-efficiency summary for one group and scale point."""

    group: str
    neuron_count: int
    hidden_neurons: int
    max_edges: int
    seeds: int
    final_mean_best_fitness: float
    final_std_best_fitness: float
    final_mean_selection_best_fitness: float
    final_mean_active_synapses: float
    final_active_edge_utilization: float
    final_fitness_per_active_synapse: float
    final_mean_hidden_edge_fraction: float
    final_mean_direct_sensor_motor_fraction: float
    threshold_success_rate: float
    mean_generation_to_threshold: float | None


@dataclass(frozen=True)
class SparseEfficiencyResult:
    """Complete sparse-efficiency evaluation output."""

    config: dict
    records: list[SparseEfficiencyGenerationRecord]
    summary: list[SparseEfficiencySummaryRecord]

    def to_json_dict(self) -> dict:
        return {
            "config": self.config,
            "records": [asdict(row) for row in self.records],
            "summary": [asdict(row) for row in self.summary],
        }


class SparseEfficiencyRunner:
    """Evaluate active-edge pressure, pruning pressure, sprouting schedules, and protected cores."""

    def __init__(self, config: SparseEfficiencyConfig | None = None) -> None:
        self.config = config or SparseEfficiencyConfig()
        if not self.config.seeds:
            raise ValueError("at least one seed is required")
        if not self.config.scale_points:
            raise ValueError("at least one neuron scale point is required")
        if not self.config.groups:
            raise ValueError("at least one sparse-efficiency group is required")
        if self.config.reference_max_edges <= 0:
            raise ValueError("reference_max_edges must be positive")
        for point in self.config.scale_points:
            if point.neuron_count < 12:
                raise ValueError("neuron_count must be at least 12 for the default 8-sensor/4-motor transducer")
            if point.max_edges < len(self.config.seed_edges):
                raise ValueError("max_edges must be at least the seeded edge count")

    def run(self) -> SparseEfficiencyResult:
        _require_torch()
        records: list[SparseEfficiencyGenerationRecord] = []
        for group in self.config.groups:
            for point in self.config.scale_points:
                for seed in self.config.seeds:
                    records.extend(self.run_group_trial(group, point, seed))
        summary = summarize_sparse_efficiency_records(
            records,
            threshold=self.config.adaptation_fitness_threshold,
        )
        return SparseEfficiencyResult(
            config=_jsonable_config(self.config),
            records=records,
            summary=summary,
        )

    def run_group_trial(
        self,
        group: SparseEfficiencyGroupConfig,
        point: NeuronScalePoint,
        seed: int,
    ) -> list[SparseEfficiencyGenerationRecord]:
        loop, generator = self._build_loop(group, point, seed)
        rows: list[SparseEfficiencyGenerationRecord] = []
        for _ in range(self.config.generations):
            loop.run(self.config.epoch_steps, generator=generator)
            report = loop.last_epoch_report
            if report is None:
                raise RuntimeError("loop did not emit an epoch report")
            usage = loop.evolver.edge_usage_stats()
            mean_active = _float(usage["mean_active_synapses"])
            epoch_best = _float(report.get("raw_best_fitness", report.get("best_fitness", 0.0)))
            rows.append(
                SparseEfficiencyGenerationRecord(
                    group=group.name,
                    neuron_count=point.neuron_count,
                    hidden_neurons=hidden_neuron_count(point.neuron_count),
                    max_edges=point.max_edges,
                    seed=int(seed),
                    generation=int(report["completed_generation"]),
                    epoch_best_fitness=epoch_best,
                    all_time_best_fitness=_float(report.get("all_time_best_fitness", 0.0)),
                    selection_best_fitness=_float(report.get("selection_best_fitness", report.get("best_fitness", 0.0))),
                    mean_population_fitness=_float(report.get("raw_mean_fitness", report.get("mean_fitness", 0.0))),
                    selection_mean_fitness=_float(report.get("selection_mean_fitness", report.get("mean_fitness", 0.0))),
                    mean_active_synapses=mean_active,
                    active_edge_utilization=(mean_active / point.max_edges) if point.max_edges else 0.0,
                    fitness_per_active_synapse=(epoch_best / mean_active) if mean_active else 0.0,
                    mean_hidden_edges=_float(usage["mean_hidden_edges"]),
                    mean_hidden_edge_fraction=_float(usage["mean_hidden_edge_fraction"]),
                    mean_direct_sensor_motor_fraction=_float(usage["mean_direct_sensor_motor_fraction"]),
                    sprout_count=int(report.get("sprout_count", 0)),
                    prune_count=int(report.get("prune_count", 0)),
                    low_ltw_prune_count=int(report.get("low_ltw_prune_count", 0)),
                    ltw_mutation_count=int(report.get("ltw_mutation_count", 0)),
                )
            )
        return rows

    def save_outputs(
        self,
        result: SparseEfficiencyResult,
        output_dir: str | Path,
        *,
        plot: bool = True,
    ) -> dict[str, str]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        json_path = output / "sparse_efficiency.json"
        json_path.write_text(json.dumps(result.to_json_dict(), indent=2) + "\n", encoding="utf-8")

        records_csv = output / "sparse_efficiency_records.csv"
        _write_csv(records_csv, [asdict(row) for row in result.records])

        summary_csv = output / "sparse_efficiency_summary.csv"
        _write_csv(summary_csv, [asdict(row) for row in result.summary])

        paths = {
            "json": str(json_path),
            "records_csv": str(records_csv),
            "summary_csv": str(summary_csv),
        }
        if plot:
            try:
                plot_path = output / "sparse_efficiency_summary.png"
                plot_sparse_efficiency_result(result, plot_path)
                paths["plot"] = str(plot_path)
            except Exception as exc:  # pragma: no cover - optional plotting
                paths["plot"] = f"skipped: {exc}"
        return paths

    def _build_loop(self, group: SparseEfficiencyGroupConfig, point: NeuronScalePoint, seed: int):
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
        protected_count = 0
        if group.protect_core:
            protected_count = (
                self.config.protected_core_edge_count
                if self.config.protected_core_edge_count is not None
                else len(self.config.seed_edges)
            )
        evolver = TensorEvolver(
            TensorEvolverConfig(
                population_size=self.config.population_size,
                neuron_count=point.neuron_count,
                max_edges=point.max_edges,
                survivor_fraction=self.config.survivor_fraction,
                ltw_noise_std=group.ltw_noise_std,
                sprout_probability=self._sprout_probability_for(group, point),
                prune_probability=group.prune_probability,
                low_ltw_prune_threshold=group.low_ltw_prune_threshold,
                low_ltw_prune_probability=group.low_ltw_prune_probability,
                protected_edge_count=protected_count,
                protect_core_topology=group.protect_core_topology,
                protect_core_weights=group.protect_core_weights,
            ),
            device=device,
        )
        evolver.seed_from_edges(self.config.seed_edges)
        loop = EvolvingHeadlessAMMCLoop(
            environment,
            evolver,
            VectorizedTransducer(TransducerConfig(neuron_count=point.neuron_count)),
            EvolvingLoopConfig(
                epoch_steps=self.config.epoch_steps,
                active_edge_fitness_penalty=group.active_edge_fitness_penalty,
            ),
            logger=EvolutionTelemetryLogger(),
        ).to(device)
        return loop, generator

    def _sprout_probability_for(self, group: SparseEfficiencyGroupConfig, point: NeuronScalePoint) -> float:
        if not group.sprout_scale_by_capacity:
            return group.sprout_probability
        scale = self.config.reference_max_edges / point.max_edges
        return max(0.0, group.sprout_probability * scale)


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


def summarize_neuron_scaling_records(
    records: list[NeuronScalingGenerationRecord],
    *,
    threshold: float,
) -> list[NeuronScalingSummaryRecord]:
    summaries: list[NeuronScalingSummaryRecord] = []
    scale_keys = sorted({(row.neuron_count, row.max_edges) for row in records})
    for neuron_count, max_edges in scale_keys:
        scale_rows = [row for row in records if row.neuron_count == neuron_count and row.max_edges == max_edges]
        seeds = sorted({row.seed for row in scale_rows})
        final_rows = [
            max((row for row in scale_rows if row.seed == seed), key=lambda row: row.generation)
            for seed in seeds
        ]
        threshold_generations: list[int] = []
        for seed in seeds:
            seed_rows = sorted((row for row in scale_rows if row.seed == seed), key=lambda row: row.generation)
            reached = next((row.generation for row in seed_rows if row.all_time_best_fitness >= threshold), None)
            if reached is not None:
                threshold_generations.append(reached)

        final_best = [row.all_time_best_fitness for row in final_rows]
        final_active = [row.mean_active_synapses for row in final_rows]
        final_active_mean = _mean(final_active)
        summaries.append(
            NeuronScalingSummaryRecord(
                neuron_count=neuron_count,
                hidden_neurons=hidden_neuron_count(neuron_count),
                max_edges=max_edges,
                seeds=len(seeds),
                final_mean_best_fitness=_mean(final_best) if final_best else 0.0,
                final_std_best_fitness=_std(final_best) if final_best else 0.0,
                final_mean_active_synapses=final_active_mean,
                final_active_edge_utilization=(final_active_mean / max_edges) if max_edges else 0.0,
                final_fitness_per_active_synapse=(
                    (_mean(final_best) / final_active_mean) if final_best and final_active_mean else 0.0
                ),
                threshold_success_rate=(len(threshold_generations) / len(seeds)) if seeds else 0.0,
                mean_generation_to_threshold=_mean(threshold_generations) if threshold_generations else None,
            )
        )
    return summaries


def summarize_sparse_efficiency_records(
    records: list[SparseEfficiencyGenerationRecord],
    *,
    threshold: float,
) -> list[SparseEfficiencySummaryRecord]:
    summaries: list[SparseEfficiencySummaryRecord] = []
    keys = sorted({(row.group, row.neuron_count, row.max_edges) for row in records})
    for group, neuron_count, max_edges in keys:
        group_rows = [
            row
            for row in records
            if row.group == group and row.neuron_count == neuron_count and row.max_edges == max_edges
        ]
        seeds = sorted({row.seed for row in group_rows})
        final_rows = [
            max((row for row in group_rows if row.seed == seed), key=lambda row: row.generation)
            for seed in seeds
        ]
        threshold_generations: list[int] = []
        for seed in seeds:
            seed_rows = sorted((row for row in group_rows if row.seed == seed), key=lambda row: row.generation)
            reached = next((row.generation for row in seed_rows if row.all_time_best_fitness >= threshold), None)
            if reached is not None:
                threshold_generations.append(reached)

        final_best = [row.all_time_best_fitness for row in final_rows]
        final_selection = [row.selection_best_fitness for row in final_rows]
        final_active = [row.mean_active_synapses for row in final_rows]
        final_utilization = [row.active_edge_utilization for row in final_rows]
        final_fitness_per_edge = [row.fitness_per_active_synapse for row in final_rows]
        final_hidden_fraction = [row.mean_hidden_edge_fraction for row in final_rows]
        final_direct_fraction = [row.mean_direct_sensor_motor_fraction for row in final_rows]
        summaries.append(
            SparseEfficiencySummaryRecord(
                group=group,
                neuron_count=neuron_count,
                hidden_neurons=hidden_neuron_count(neuron_count),
                max_edges=max_edges,
                seeds=len(seeds),
                final_mean_best_fitness=_mean(final_best) if final_best else 0.0,
                final_std_best_fitness=_std(final_best) if final_best else 0.0,
                final_mean_selection_best_fitness=_mean(final_selection) if final_selection else 0.0,
                final_mean_active_synapses=_mean(final_active) if final_active else 0.0,
                final_active_edge_utilization=_mean(final_utilization) if final_utilization else 0.0,
                final_fitness_per_active_synapse=_mean(final_fitness_per_edge) if final_fitness_per_edge else 0.0,
                final_mean_hidden_edge_fraction=_mean(final_hidden_fraction) if final_hidden_fraction else 0.0,
                final_mean_direct_sensor_motor_fraction=_mean(final_direct_fraction) if final_direct_fraction else 0.0,
                threshold_success_rate=(len(threshold_generations) / len(seeds)) if seeds else 0.0,
                mean_generation_to_threshold=_mean(threshold_generations) if threshold_generations else None,
            )
        )
    return summaries


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


def plot_neuron_scaling_result(result: NeuronScalingResult, path: str | Path | None = None, *, show: bool = False):
    if not result.summary:
        raise ValueError("no neuron scaling summary records to plot")
    import matplotlib.pyplot as plt

    rows = sorted(result.summary, key=lambda row: row.neuron_count)
    x = [row.neuron_count for row in rows]
    fitness = [row.final_mean_best_fitness for row in rows]
    fitness_std = [row.final_std_best_fitness for row in rows]
    active = [row.final_mean_active_synapses for row in rows]
    utilization = [row.final_active_edge_utilization * 100.0 for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].errorbar(
        x,
        fitness,
        yerr=fitness_std,
        marker="o",
        color="#38bdf8",
        linewidth=2,
        capsize=4,
        label="Final mean best fitness",
    )
    axes[0].set_title("AMMC Gen-5 Neuron Scaling")
    axes[0].set_ylabel("Best fitness")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(x, active, marker="o", color="#f59e0b", linewidth=2, label="Mean active synapses")
    axes[1].set_xlabel("Neuron count")
    axes[1].set_ylabel("Active synapses")
    axes[1].grid(True, alpha=0.25)
    twin = axes[1].twinx()
    twin.plot(x, utilization, marker="s", color="#22c55e", linewidth=2, label="Edge utilization")
    twin.set_ylabel("Edge utilization (%)")

    handles, labels = axes[1].get_legend_handles_labels()
    twin_handles, twin_labels = twin.get_legend_handles_labels()
    axes[1].legend(handles + twin_handles, labels + twin_labels, loc="best")

    fig.tight_layout()
    if path is not None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160)
    if show:
        plt.show()
    return fig


def plot_sparse_efficiency_result(result: SparseEfficiencyResult, path: str | Path | None = None, *, show: bool = False):
    if not result.summary:
        raise ValueError("no sparse-efficiency summary records to plot")
    import matplotlib.pyplot as plt

    groups = sorted({row.group for row in result.summary})
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    for group in groups:
        rows = sorted((row for row in result.summary if row.group == group), key=lambda row: row.neuron_count)
        x = [row.neuron_count for row in rows]
        axes[0].plot(
            x,
            [row.final_mean_best_fitness for row in rows],
            marker="o",
            linewidth=2,
            label=group,
        )
        axes[1].plot(
            x,
            [row.final_mean_active_synapses for row in rows],
            marker="o",
            linewidth=2,
            label=group,
        )
        axes[2].plot(
            x,
            [row.final_fitness_per_active_synapse for row in rows],
            marker="o",
            linewidth=2,
            label=group,
        )

    axes[0].set_title("AMMC Gen-5 Sparse-Efficiency Ablation")
    axes[0].set_ylabel("Best fitness")
    axes[1].set_ylabel("Active synapses")
    axes[2].set_ylabel("Fitness / active synapse")
    axes[2].set_xlabel("Neuron count")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize="small")

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
