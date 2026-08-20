"""Run the frozen Gen-30 dendritic predictive-credit causal microtask."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5 import (  # noqa: E402
    Gen30Config,
    available_gen30_arms,
    bundle_gen30_artifacts,
    run_gen30,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="gen5_outputs/gen30_dendritic_predictive_credit_cuda")
    parser.add_argument("--progress-path", default=None)
    parser.add_argument("--list-arms", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if args.list_arms:
        print("\n".join(available_gen30_arms()))
        return
    progress = args.progress_path or str(
        pathlib.Path(args.output_dir) / "gen30_dendritic_predictive_credit_progress.json"
    )
    result = run_gen30(Gen30Config(), device=args.device, progress_path=progress)
    paths = result.save(args.output_dir, plot=not args.no_plot)
    paths["progress"] = progress
    paths.update(bundle_gen30_artifacts(paths, args.output_dir))
    print(json.dumps({
        "paths": paths,
        "device": result.device,
        "task": result.task,
        "decision": result.decision,
        "summary": result.summary,
    }, indent=2))


if __name__ == "__main__":
    main()
