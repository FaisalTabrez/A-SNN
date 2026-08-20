"""Generate the deterministic Gen-29 causal evidence closure."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ammc_gen5"))

from gen29_program_closure import bundle_gen29_artifacts, run_gen29  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(ROOT.parent))
    parser.add_argument("--output-dir", default="gen5_outputs/gen29_program_closure")
    args = parser.parse_args()
    result = run_gen29(args.repo_root)
    paths = result.save(args.output_dir)
    paths.update(bundle_gen29_artifacts(paths, args.output_dir))
    print(json.dumps({"paths": paths, "decision": result.decision}, indent=2))


if __name__ == "__main__":
    main()
