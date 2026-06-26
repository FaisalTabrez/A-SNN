from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5 import EvolutionTelemetryLogger


class FakeCounts:
    def float(self):
        return self

    def mean(self):
        return self

    def item(self):
        return 12.5


class FakeEvolver:
    def active_edge_counts(self):
        return FakeCounts()


class TelemetryContractTest(unittest.TestCase):
    def test_log_and_save_standard_outputs(self) -> None:
        logger = EvolutionTelemetryLogger()
        record = logger.log_epoch(
            {
                "completed_generation": 7,
                "epoch": 7,
                "best_fitness": 42.0,
                "mean_fitness": 3.5,
                "sprout_count": 9,
                "prune_count": 2,
                "ltw_mutation_count": 128,
            },
            FakeEvolver(),
        )

        self.assertEqual(record.generation, 7)
        self.assertEqual(record.max_fitness, 42.0)
        self.assertEqual(record.mean_population_fitness, 3.5)
        self.assertEqual(record.mean_active_synapses, 12.5)

        with tempfile.TemporaryDirectory() as tmp:
            json_path = logger.save_json(pathlib.Path(tmp) / "telemetry.json")
            csv_path = logger.save_csv(pathlib.Path(tmp) / "telemetry.csv")
            self.assertTrue(json_path.exists())
            self.assertTrue(csv_path.exists())
            self.assertIn("max_fitness", csv_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

