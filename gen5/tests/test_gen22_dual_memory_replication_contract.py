from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.gen22_dual_memory_replication import (
    GEN22_ARMS,
    Gen22Config,
    _load_progress,
    _save_progress,
    available_gen22_arms,
    decide_gen22,
    disjoint_sensor_damage_indices,
)


class Gen22ContractTest(unittest.TestCase):
    def test_frozen_protocol(self) -> None:
        config = Gen22Config()
        self.assertEqual(available_gen22_arms(), GEN22_ARMS)
        self.assertEqual(config.seeds, (421, 422, 423, 424, 425))
        self.assertEqual(config.minimum_a_retention_gain_vs_single, 0.01)
        self.assertEqual(config.maximum_b_accuracy_cost_vs_single, 0.005)
        self.assertEqual(config.minimum_qualifying_seed_count, 3)

    def test_damage_banks_are_deterministic_and_disjoint(self) -> None:
        first = disjoint_sensor_damage_indices(100, 0.35, seed=12)
        second = disjoint_sensor_damage_indices(100, 0.35, seed=12)
        self.assertEqual(first, second)
        self.assertEqual(len(first[0]), 35)
        self.assertEqual(len(first[1]), 35)
        self.assertFalse(set(first[0]) & set(first[1]))

    def test_decision_requires_aggregate_and_three_paired_seeds(self) -> None:
        rows = []
        for seed in Gen22Config().seeds:
            for arm in GEN22_ARMS:
                rows.append({
                    "seed": seed, "arm": arm,
                    "after_b_a_accuracy": 0.50 if arm == "dual_memory" else 0.48,
                    "after_b_b_accuracy": 0.51,
                    "a_forgetting_after_b": 0.01, "clean_retention_drop": 0.01,
                    "stability_plasticity_score": 0.505 if arm == "dual_memory" else 0.49,
                    "a_retention_gain_vs_single": 0.02 if arm == "dual_memory" else 0.0,
                    "b_accuracy_gain_vs_single": 0.0,
                    "stability_plasticity_gain_vs_single": 0.015 if arm == "dual_memory" else 0.0,
                    "ltw_causal_margin": 0.02 if arm == "dual_memory" else 0.0,
                    "consolidation_identity_margin": 0.015 if arm == "dual_memory" else 0.0,
                    "allocated_slots": 100, "active_slots": 35, "adapter_memory_bytes": 1200,
                })
        self.assertEqual(decide_gen22(rows, Gen22Config())["status"], "pass")
        dual_rows = [row for row in rows if row["arm"] == "dual_memory"]
        for row in dual_rows[:3]:
            row["a_retention_gain_vs_single"] = 0.0
        self.assertEqual(decide_gen22(rows, Gen22Config())["status"], "stop")

    def test_progress_round_trip(self) -> None:
        config = Gen22Config()
        rows = [{"seed": 421, "arm": "static_backbone"}]
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "progress.json"
            _save_progress(path, config, rows)
            self.assertEqual(_load_progress(path, config), rows)


if __name__ == "__main__":
    unittest.main()
