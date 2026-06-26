from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5 import ChampionExporter


class ChampionExportContractTest(unittest.TestCase):
    def test_exports_browser_connectome_and_weight_payload(self) -> None:
        snapshot = {
            "sources": [0, 1, 0, 2, 0],
            "targets": [8, 9, 0, 10, 8],
            "active_mask": [True, True, True, True, True],
            "signs": [1, -1, 1, 1, 1],
            "delay_steps": [2, 3, 4, 5, 6],
            "short_term_weight": [0.1, 0.0, 0.2, 0.0, 0.4],
            "long_term_weight": [0.7, 0.6, 0.5, 1.2, 0.3],
        }

        with tempfile.TemporaryDirectory() as tmp:
            result = ChampionExporter().export_from_snapshot(
                snapshot,
                tmp,
                neuron_count=16,
                organism_id="TestChampion",
                fitness=12.0,
            )

            self.assertEqual(result.neuron_count, 16)
            # slot 2 is self-edge; slot 4 duplicates slot 0's source/target/D1.
            self.assertEqual(result.active_edges, 3)
            self.assertEqual(result.skipped_duplicate_edges, 2)

            connectome = json.loads(result.connectome_path.read_text(encoding="utf-8"))
            weights = json.loads(result.weights_path.read_text(encoding="utf-8"))
            adjacency = json.loads(result.adjacency_path.read_text(encoding="utf-8"))

            self.assertEqual(connectome["schema"], "AMMC-SNN/connectome")
            self.assertEqual(weights["schema"], "AMMC-SNN/colab-weights")
            self.assertEqual(adjacency["schema"], "AMMC-Gen5/sparse-adjacency")
            self.assertTrue(connectome["gen5"]["motorAssist"])
            self.assertEqual(len(connectome["neurons"]), 16)
            self.assertEqual(len(connectome["synapses"]), 3)
            self.assertEqual(connectome["neurons"][0]["embodimentSensorKind"], "food")
            self.assertEqual(connectome["neurons"][4]["embodimentSensorKind"], "toxin")
            self.assertIsNone(connectome["neurons"][8]["embodimentSensorKind"])
            self.assertEqual(len(weights["edges"]), 3)
            self.assertEqual(weights["edges"][0]["source_id"], connectome["synapses"][0]["sourceId"])
            self.assertEqual(weights["edges"][0]["target_id"], connectome["synapses"][0]["targetId"])
            self.assertEqual(weights["edges"][0]["dendrite_id"], connectome["synapses"][0]["dendriteId"])
            self.assertEqual(weights["edges"][2]["weight"], 1.0)


if __name__ == "__main__":
    unittest.main()
