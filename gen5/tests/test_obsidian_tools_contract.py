import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from obsidian_retriever import parse_frontmatter, query_vault_context
from obsidian_sync import create_experiment_note
from build_obsidian_graph import ARTIFACTS, DECISIONS, build_graph


class ObsidianToolsContractTest(unittest.TestCase):
    def test_sync_note_is_retrievable_and_updates_deterministically(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            vault = Path(temporary_directory) / "vault"
            note = create_experiment_note(vault, "run-01", "sprint41", "example.py", "SHD", "Topology", "Hypothesis", {"accuracy": 0.9, "latency_ms": 4})
            metadata, _ = parse_frontmatter(note)
            self.assertEqual(metadata["run_id"], "run-01")
            self.assertEqual(metadata["metrics"]["accuracy"], 0.9)
            context = query_vault_context(vault, sprint_query="sprint41", tags=["experiment"])
            self.assertIn("run-01", context)
            self.assertIn("accuracy", context)
            create_experiment_note(vault, "run-01", "sprint41", "example.py", "SHD", "Topology", "Hypothesis", json.loads('{"accuracy": 1.0}'))
            self.assertEqual(len(list((vault / "Experiments").glob("*.md"))), 1)

    def test_graph_builder_creates_linked_artifact_notes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "outputs" / "shd_sparse_width_cuda_2026-08-10"
            source.mkdir(parents=True)
            (source / "shd_sparse_width.json").write_text(
                json.dumps({"summary": {"mean_test_accuracy": 0.8}}), encoding="utf-8"
            )
            self.assertEqual(build_graph(root / "vault", root / "outputs"), 1)
            experiment = next((root / "vault" / "Experiments").glob("*.md"))
            metadata, body = parse_frontmatter(experiment)
            self.assertEqual(metadata["sprint_id"], "sprint41")
            self.assertEqual(metadata["metrics"]["mean_test_accuracy"], 0.8)
            self.assertIn("[[Sparse-Width-Scaling]]", body)
            note_names = {note.stem for note in (root / "vault").rglob("*.md")}
            links = {
                target
                for note in (root / "vault").rglob("*.md")
                for target in re.findall(r"\[\[([^]|#]+)", note.read_text(encoding="utf-8"))
            }
            self.assertFalse(links - note_names, f"Unresolved graph links: {sorted(links - note_names)}")

    def test_graph_builder_marks_gen30_as_active_results_pending(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            build_graph(root / "vault", root / "outputs")
            state = (root / "vault" / "Current State.md").read_text(encoding="utf-8")
            gen30 = (root / "vault" / "Sprints" / "Gen-30.md").read_text(encoding="utf-8")
            self.assertIn("[[Gen-30]] is the current program position", state)
            self.assertIn("no result or mechanism claim exists", state)
            self.assertIn("active-implementation-complete-results-pending", gen30)
            self.assertIn("Do not enable structural plasticity", state)

    def test_graph_builder_covers_every_phase_after_sprint41(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            build_graph(root / "vault", root / "outputs")
            sprint_names = {note.stem for note in (root / "vault" / "Sprints").glob("*.md")}
            expected = {
                *(f"Sprint-{number}" for number in range(42, 50)),
                *(f"Gen-{number}" for number in range(20, 31)),
            }
            self.assertFalse(expected - sprint_names)
            decision_names = {note.stem for note in (root / "vault" / "Decisions").glob("*.md")}
            self.assertEqual(
                {f"{name} Decision" for name in expected},
                decision_names,
            )
            self.assertEqual(len(DECISIONS), 19)

    def test_graph_builder_imports_completed_gen9_through_gen18_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for generation in range(9, 19):
                name = f"gen{generation}_sample_cuda_2026-08-10"
                source = root / "outputs" / name
                source.mkdir(parents=True)
                # Only Gen-9's real artifact filename differs from its generation prefix.
                artifact = "gen9_continual_adaptation.json" if generation == 9 else f"gen{generation}_" + {
                    10: "robust_representation", 11: "plastic_adapter", 12: "associative_memory",
                    13: "local_plasticity", 14: "reward_eligibility", 15: "reward_baseline",
                    16: "local_score_credit", 17: "sparse_spiking_credit", 18: "local_credit_replication",
                }[generation] + ".json"
                (source / artifact).write_text(json.dumps({"decision": {"status": "stop"}}), encoding="utf-8")
            # Point the test at the actual expected directory names by copying the generated inputs.
            outputs = root / "outputs"
            for entry in ARTIFACTS:
                if entry[0] not in {f"gen{number}" for number in range(9, 19)}:
                    continue
                expected = outputs / Path(entry[1]).parent
                expected.mkdir(parents=True, exist_ok=True)
                (expected / Path(entry[1]).name).write_text(json.dumps({"decision": {"status": "stop"}}), encoding="utf-8")
            build_graph(root / "vault", outputs)
            imported = {note.name for note in (root / "vault" / "Experiments").glob("EXP-GEN*.md")}
            self.assertTrue(all(any(name.startswith(f"EXP-GEN{number}-") for name in imported) for number in range(9, 19)))


if __name__ == "__main__":
    unittest.main()
