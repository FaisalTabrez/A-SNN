"""Champion genome export for the Gen-5 -> Gen-4 visual bridge."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ChampionExportConfig:
    """Browser-compatible export settings."""

    coordinate_width: float = 1028.0
    coordinate_height: float = 743.0
    dendrites_per_neuron: int = 4
    organism_id: str = "Champion"
    include_inactive_edges: bool = False


@dataclass(frozen=True)
class ChampionExportResult:
    """Paths and summary for an exported champion bundle."""

    output_dir: Path
    connectome_path: Path
    weights_path: Path
    adjacency_path: Path
    active_edges: int
    skipped_duplicate_edges: int
    neuron_count: int
    fitness: float | None


class ChampionExporter:
    """Export a Gen-5 sparse genome into Gen-4 browser-compatible files.

    Files written:

    - `champion_connectome.json`: load this first in the browser sandbox.
    - `colab_weights.json`: import this second with "Import PyTorch Weights".
    - `champion_sparse_adjacency.json`: analysis-friendly sparse adjacency.
    """

    def __init__(self, config: ChampionExportConfig | None = None) -> None:
        self.config = config or ChampionExportConfig()

    def export_best_from_fitness(
        self,
        evolver,
        fitness,
        output_dir: str | Path,
        *,
        organism_id: str | None = None,
    ) -> ChampionExportResult:
        """Select `argmax(fitness)` from a live evolver and export it."""

        best_index = self._argmax_index(fitness)
        best_fitness = self._value_at(fitness, best_index)
        return self.export_from_evolver(
            evolver,
            best_index,
            output_dir,
            organism_id=organism_id,
            fitness=best_fitness,
        )

    def export_from_evolver(
        self,
        evolver,
        genome_index: int,
        output_dir: str | Path,
        *,
        organism_id: str | None = None,
        fitness: float | None = None,
    ) -> ChampionExportResult:
        snapshot = evolver.snapshot_genome(genome_index, to_cpu=True)
        neuron_count = int(getattr(evolver, "neuron_count", len(snapshot.get("sources", []))))
        return self.export_from_snapshot(
            snapshot,
            output_dir,
            neuron_count=neuron_count,
            organism_id=organism_id,
            fitness=fitness,
        )

    def export_from_snapshot(
        self,
        snapshot: dict[str, Any],
        output_dir: str | Path,
        *,
        neuron_count: int,
        organism_id: str | None = None,
        fitness: float | None = None,
    ) -> ChampionExportResult:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        organism = organism_id or self.config.organism_id
        edge_rows, skipped = self._edge_rows(snapshot)
        neurons = self._neurons(neuron_count)
        connectome = self._connectome(neurons, edge_rows, organism, fitness)
        weights = self._weights(edge_rows, organism, fitness)
        adjacency = self._adjacency(edge_rows, organism, fitness, neuron_count)

        connectome_path = output / "champion_connectome.json"
        weights_path = output / "colab_weights.json"
        adjacency_path = output / "champion_sparse_adjacency.json"

        self._write_json(connectome_path, connectome)
        self._write_json(weights_path, weights)
        self._write_json(adjacency_path, adjacency)

        return ChampionExportResult(
            output_dir=output,
            connectome_path=connectome_path,
            weights_path=weights_path,
            adjacency_path=adjacency_path,
            active_edges=len(edge_rows),
            skipped_duplicate_edges=skipped,
            neuron_count=neuron_count,
            fitness=fitness,
        )

    def _edge_rows(self, snapshot: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
        sources = self._as_list(snapshot["sources"])
        targets = self._as_list(snapshot["targets"])
        active = self._as_list(snapshot["active_mask"])
        signs = self._as_list(snapshot.get("signs", [1.0] * len(sources)))
        delays = self._as_list(snapshot.get("delay_steps", [0] * len(sources)))
        stw = self._as_list(snapshot.get("short_term_weight", [0.0] * len(sources)))
        ltw = self._as_list(snapshot.get("long_term_weight", [0.0] * len(sources)))

        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        skipped = 0
        for slot, (source, target) in enumerate(zip(sources, targets)):
            if not self.config.include_inactive_edges and not bool(active[slot]):
                continue
            source_index = int(source)
            target_index = int(target)
            if source_index == target_index:
                skipped += 1
                continue
            source_id = self._neuron_id(source_index)
            target_id = self._neuron_id(target_index)
            dendrite_id = f"{target_id}:D{slot % self.config.dendrites_per_neuron + 1}"
            key = (source_id, target_id, dendrite_id)
            if key in seen:
                skipped += 1
                continue
            seen.add(key)
            long_term = self._clamp(float(ltw[slot]), 0.0, 1.0)
            short_term = self._clamp(float(stw[slot]), 0.0, max(0.0, 1.2 - long_term))
            sign = -1.0 if float(signs[slot]) < 0 else 1.0
            rows.append(
                {
                    "edge_index": len(rows),
                    "slot": slot,
                    "source_index": source_index,
                    "target_index": target_index,
                    "source_id": source_id,
                    "target_id": target_id,
                    "dendrite_id": dendrite_id,
                    "dendrite_index": slot % self.config.dendrites_per_neuron,
                    "short_term_weight": short_term,
                    "long_term_weight": long_term,
                    "effective_weight": short_term + long_term,
                    "sign": sign,
                    "kind": "inhibitory" if sign < 0 else "excitatory",
                    "delay_steps": int(delays[slot]),
                }
            )
        return rows, skipped

    def _neurons(self, neuron_count: int) -> list[dict[str, Any]]:
        return [self._neuron(index, neuron_count) for index in range(neuron_count)]

    def _neuron(self, index: int, neuron_count: int) -> dict[str, Any]:
        direction_names = ["north", "east", "south", "west"]
        direction_labels = ["North", "East", "South", "West"]

        if index < 8:
            group_index = index % 4
            sensor_group = "Food" if index < 4 else "Toxin"
            role = "sensor"
            direction = direction_names[group_index]
            sensor_kind = sensor_group.lower()
            label = f"{sensor_group} {direction_labels[group_index]} Sensor"
            type_name = f"{sensor_group}_Sensor"
            x = 0.13 * self.config.coordinate_width
            y = (0.16 + group_index * 0.18 + (0.04 if index >= 4 else 0.0)) * self.config.coordinate_height
        elif index < 12:
            group_index = index - 8
            role = "motor"
            direction = direction_names[group_index]
            sensor_kind = None
            label = f"{direction_labels[group_index]} Motor"
            type_name = "motor"
            x = 0.86 * self.config.coordinate_width
            y = (0.2 + group_index * 0.16) * self.config.coordinate_height
        else:
            hidden_index = index - 12
            hidden_count = max(1, neuron_count - 12)
            role = None
            direction = None
            sensor_kind = None
            label = f"Hidden {hidden_index + 1}"
            type_name = "hidden"
            x = 0.5 * self.config.coordinate_width
            y = (0.12 + (hidden_index + 1) / (hidden_count + 1) * 0.76) * self.config.coordinate_height

        return {
            "id": self._neuron_id(index),
            "x": round(x, 4),
            "y": round(y, 4),
            "layer": "gen5",
            "type": type_name,
            "label": label,
            "kind": "excitatory",
            "dendritePhase": -0.62,
            "embodimentRole": role,
            "embodimentDirection": direction,
            "embodimentSensorKind": sensor_kind,
        }

    def _connectome(
        self,
        neurons: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        organism_id: str,
        fitness: float | None,
    ) -> dict[str, Any]:
        return {
            "schema": "AMMC-SNN/connectome",
            "version": 1,
            "exportedAt": self._now(),
            "coordinateSpace": {
                "width": self.config.coordinate_width,
                "height": self.config.coordinate_height,
            },
            "organism": {"id": organism_id, "parentId": "Gen5-Champion"},
            "gen5": {
                "fitness": fitness,
                "sensorChannels": [
                    "food_north",
                    "food_east",
                    "food_south",
                    "food_west",
                    "toxin_north",
                    "toxin_east",
                    "toxin_south",
                    "toxin_west",
                ],
                "motorChannels": ["north", "east", "south", "west"],
                "motorAssist": True,
                "note": "Gen-5 champion exported for the Gen-4 browser sandbox. Browser bridge mode preserves separate food/toxin directional sensor channels.",
            },
            "neurons": neurons,
            "synapses": [
                {
                    "sourceId": edge["source_id"],
                    "targetId": edge["target_id"],
                    "dendriteId": edge["dendrite_id"],
                    "weight": edge["effective_weight"],
                    "shortTermWeight": edge["short_term_weight"],
                    "longTermWeight": edge["long_term_weight"],
                    "kind": edge["kind"],
                    "strength": 1,
                    "minimumWeight": 0.001,
                    "minimumLongTermWeight": 0.001,
                    "decayRate": 0.000018,
                    "shortTermDecayRate": 0.000018,
                    "longTermDecayRate": 0.000000004,
                    "conductionSpeed": 0.34,
                }
                for edge in edges
            ],
        }

    def _weights(self, edges: list[dict[str, Any]], organism_id: str, fitness: float | None) -> dict[str, Any]:
        return {
            "schema": "AMMC-SNN/colab-weights",
            "version": 1,
            "organism_id": organism_id,
            "fitness": fitness,
            "sparse_adjacency": [
                [
                    edge["source_index"],
                    edge["target_index"],
                    edge["long_term_weight"],
                    edge["sign"],
                    edge["delay_steps"],
                ]
                for edge in edges
            ],
            "edges": [
                {
                    "edge_index": edge["edge_index"],
                    "source_id": edge["source_id"],
                    "target_id": edge["target_id"],
                    "dendrite_id": edge["dendrite_id"],
                    "weight": edge["long_term_weight"],
                }
                for edge in edges
            ],
        }

    def _adjacency(
        self,
        edges: list[dict[str, Any]],
        organism_id: str,
        fitness: float | None,
        neuron_count: int,
    ) -> dict[str, Any]:
        return {
            "schema": "AMMC-Gen5/sparse-adjacency",
            "version": 1,
            "organism_id": organism_id,
            "fitness": fitness,
            "neuron_count": neuron_count,
            "active_edges": len(edges),
            "columns": ["source_index", "target_index", "ltw", "stw", "sign", "delay_steps", "slot"],
            "rows": [
                [
                    edge["source_index"],
                    edge["target_index"],
                    edge["long_term_weight"],
                    edge["short_term_weight"],
                    edge["sign"],
                    edge["delay_steps"],
                    edge["slot"],
                ]
                for edge in edges
            ],
        }

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "tolist"):
            value = value.tolist()
        return list(value)

    @staticmethod
    def _argmax_index(value: Any) -> int:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "argmax"):
            result = value.argmax()
            if hasattr(result, "item"):
                return int(result.item())
            return int(result)
        values = list(value)
        return max(range(len(values)), key=lambda index: values[index])

    @staticmethod
    def _value_at(value: Any, index: int) -> float:
        if hasattr(value, "detach"):
            value = value.detach()
        selected = value[index] if hasattr(value, "__getitem__") else list(value)[index]
        if hasattr(selected, "cpu"):
            selected = selected.cpu()
        if hasattr(selected, "item"):
            selected = selected.item()
        return float(selected)

    @staticmethod
    def _neuron_id(index: int) -> str:
        return f"N{index + 1}"

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
