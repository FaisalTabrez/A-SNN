from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5.evidence_synthesis import EVIDENCE_FILENAMES


class EvidenceSynthesisContractTest(unittest.TestCase):
    def test_required_evidence_chain_is_complete(self) -> None:
        self.assertEqual(tuple(EVIDENCE_FILENAMES), (
            "phase44", "phase45", "phase46", "phase47", "phase48", "phase49"
        ))
        self.assertEqual(len(set(EVIDENCE_FILENAMES.values())), 6)


if __name__ == "__main__":
    unittest.main()
