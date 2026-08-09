"""Sprint 50 CLI: synthesize the final Gen-5 SHD/SSC evidence ledger."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5 import synthesize_gen5_evidence  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the Phase 50 Gen-5 evidence report")
    parser.add_argument("--evidence-root", default="gen5/outputs")
    parser.add_argument("--output-dir", default="gen5/outputs/gen5_evidence_synthesis")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = synthesize_gen5_evidence(args.evidence_root)
    paths = result.save(args.output_dir, plot=not args.no_plot)
    print(json.dumps({"paths": paths, "claims": result.claims}, indent=2))


if __name__ == "__main__":
    main()
