from __future__ import annotations

import hashlib
import json
import pathlib
import tempfile
import unittest
import sys

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / "ammc_gen5"
sys.path.insert(0, str(PACKAGE))

from gen29_program_closure import (  # noqa: E402
    CLAIMS,
    EVIDENCE_DOCUMENTS,
    bundle_gen29_artifacts,
    run_gen29,
)


class Gen29ProgramClosureContractTest(unittest.TestCase):
    def test_repository_evidence_chain_closes_program(self):
        root = pathlib.Path(__file__).resolve().parents[2]
        result = run_gen29(root)
        self.assertEqual(len(result.sources), 12)
        self.assertEqual(len(result.claims), 12)
        self.assertEqual(result.decision["status"], "complete")
        self.assertEqual(result.decision["supported_adaptive_mechanisms"], [])
        self.assertFalse(result.decision["hardware_energy_claim_authorized"])
        self.assertEqual(
            result.decision["next_milestone"],
            "new_mechanism_theory_and_preregistered_causal_microtask",
        )

    def test_claims_keep_supported_rejected_and_untested_separate(self):
        statuses = {row[2] for row in CLAIMS}
        self.assertTrue({"supported", "rejected", "untested"} <= statuses)
        self.assertEqual(len(EVIDENCE_DOCUMENTS), len(set(EVIDENCE_DOCUMENTS)))

    def test_bundle_manifest_matches_saved_files(self):
        root = pathlib.Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = run_gen29(root)
            paths = result.save(temporary_directory)
            bundled = bundle_gen29_artifacts(paths, temporary_directory)
            manifest = json.loads(pathlib.Path(bundled["manifest"]).read_text(encoding="utf-8"))
            for row in manifest["files"]:
                path = pathlib.Path(temporary_directory) / row["name"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["sha256"])


if __name__ == "__main__":
    unittest.main()
