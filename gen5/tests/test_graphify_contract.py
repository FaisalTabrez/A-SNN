import json
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOOLS))

from verify_graphify_contract import verify_graph


class GraphifyContractTest(unittest.TestCase):
    def test_graphify_ignore_keeps_the_corpus_bounded(self):
        policy = (REPOSITORY_ROOT / ".graphifyignore").read_text(encoding="utf-8")
        self.assertIn("obsidian_vault/", policy)
        self.assertIn("*.zip", policy)
        self.assertIn("gen5/outputs/**", policy)
        self.assertIn("!gen5/outputs/**/*.md", policy)

    def test_verifier_accepts_implementation_and_research_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            graph = Path(directory) / "graph.json"
            graph.write_text(
                json.dumps(
                    {
                        "nodes": [
                            {"source_file": "gen5/ammc_gen5/runtime.py"},
                            {"source_file": "gen5/examples/gen30_example.py"},
                            {"source_file": "gen5/tests/test_runtime.py"},
                            {"source_file": "research.md"},
                            {"source_file": "gen5/docs/PRIMARY_EVIDENCE_TRACK_ROADMAP.md"},
                            {"source_file": "gen5/docs/LTH_INFORMED_ITINERARY_REVIEW.md"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            result = verify_graph(graph, require_research=True)
            self.assertTrue(result["passed"])
            self.assertEqual(result["missing_implementation"], [])
            self.assertEqual(result["missing_research"], [])

    def test_verifier_separates_code_only_from_research_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            graph = Path(directory) / "graph.json"
            graph.write_text(
                json.dumps(
                    {
                        "nodes": [
                            {"path": "gen5/ammc_gen5/runtime.py"},
                            {"path": "gen5/examples/example.py"},
                            {"path": "gen5/tests/test_example.py"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(verify_graph(graph)["passed"])
            research_result = verify_graph(graph, require_research=True)
            self.assertFalse(research_result["passed"])
            self.assertEqual(len(research_result["missing_research"]), 3)


if __name__ == "__main__":
    unittest.main()
