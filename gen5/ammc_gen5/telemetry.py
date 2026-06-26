"""Telemetry and plotting utilities for headless AMMC evolution."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class EvolutionTelemetryRecord:
    """One epoch-level telemetry point."""

    generation: int
    epoch: int
    max_fitness: float
    mean_population_fitness: float
    mean_active_synapses: float
    sprout_count: int = 0
    prune_count: int = 0
    ltw_mutation_count: int = 0


class EvolutionTelemetryLogger:
    """Record and plot evolution curves for a headless Gen-5 run.

    The logger is intentionally dependency-light. JSON and CSV work with the
    Python standard library. Plotting imports matplotlib lazily, so Colab can
    render figures while local syntax checks do not need matplotlib installed.
    """

    def __init__(self) -> None:
        self.records: list[EvolutionTelemetryRecord] = []

    def log_epoch(self, report: dict[str, Any], evolver=None) -> EvolutionTelemetryRecord:
        """Append one epoch record from an `EvolvingHeadlessAMMCLoop` report."""

        mean_active = self._mean_active_synapses(evolver)
        record = EvolutionTelemetryRecord(
            generation=int(report.get("completed_generation", report.get("epoch", len(self.records) + 1))),
            epoch=int(report.get("epoch", len(self.records) + 1)),
            max_fitness=self._float(report.get("best_fitness", 0.0)),
            mean_population_fitness=self._float(report.get("mean_fitness", 0.0)),
            mean_active_synapses=mean_active,
            sprout_count=int(report.get("sprout_count", 0)),
            prune_count=int(report.get("prune_count", 0)),
            ltw_mutation_count=int(report.get("ltw_mutation_count", 0)),
        )
        self.records.append(record)
        return record

    def latest(self) -> EvolutionTelemetryRecord | None:
        return self.records[-1] if self.records else None

    def to_rows(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.records]

    def save_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_rows(), indent=2) + "\n", encoding="utf-8")
        return output

    def save_csv(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        rows = self.to_rows()
        fieldnames = list(asdict(EvolutionTelemetryRecord(0, 0, 0.0, 0.0, 0.0)).keys())
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return output

    def plot(self, path: str | Path | None = None, *, show: bool = False):
        """Plot max fitness, mean fitness, and mean active synapses.

        Returns the matplotlib figure. If `path` is provided, the figure is
        written to disk.
        """

        if not self.records:
            raise ValueError("no telemetry records are available to plot")

        import matplotlib.pyplot as plt  # lazy optional dependency

        generations = [record.generation for record in self.records]
        max_fitness = [record.max_fitness for record in self.records]
        mean_fitness = [record.mean_population_fitness for record in self.records]
        mean_synapses = [record.mean_active_synapses for record in self.records]

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(generations, max_fitness, color="#38bdf8", linewidth=2)
        axes[0].set_ylabel("Max fitness")
        axes[0].grid(True, alpha=0.25)

        axes[1].plot(generations, mean_fitness, color="#a7f3d0", linewidth=2)
        axes[1].set_ylabel("Mean fitness")
        axes[1].grid(True, alpha=0.25)

        axes[2].plot(generations, mean_synapses, color="#fbbf24", linewidth=2)
        axes[2].set_ylabel("Mean active synapses")
        axes[2].set_xlabel("Generation")
        axes[2].grid(True, alpha=0.25)

        fig.suptitle("AMMC Gen-5 Evolution Telemetry")
        fig.tight_layout()
        if path is not None:
            output = Path(path)
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=160)
        if show:
            plt.show()
        return fig

    @staticmethod
    def _float(value: Any) -> float:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "item"):
            value = value.item()
        return float(value)

    @classmethod
    def _mean_active_synapses(cls, evolver) -> float:
        if evolver is None:
            return 0.0
        counts = evolver.active_edge_counts()
        if hasattr(counts, "float"):
            counts = counts.float()
        if hasattr(counts, "mean"):
            counts = counts.mean()
        return cls._float(counts)

