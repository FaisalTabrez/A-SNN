"""Run the preregistered Gen-21 matched causal mechanism benchmark."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5 import (  # noqa: E402
    Gen21Config,
    available_gen21_arms,
    bundle_gen21_artifacts,
    run_gen21,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-arms", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", default="gen5_data/ssc")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--source-train-samples", type=int, default=20_000)
    parser.add_argument("--validation-samples", type=int, default=6_000)
    parser.add_argument("--test-samples", type=int, default=8_000)
    parser.add_argument("--source-epochs", type=int, default=12)
    parser.add_argument("--adaptation-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", default="gen5_outputs/gen21_matched_causal_mechanisms_cuda")
    parser.add_argument("--progress-path", default=None)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_arms:
        print("\n".join(available_gen21_arms()))
        return
    config = Gen21Config(
        data_root=args.data_root,
        download=not args.no_download,
        source_train_samples=args.source_train_samples,
        validation_samples=args.validation_samples,
        test_samples=args.test_samples,
        source_epochs=args.source_epochs,
        adaptation_epochs=args.adaptation_epochs,
        batch_size=args.batch_size,
    )
    progress = args.progress_path or str(
        pathlib.Path(args.output_dir) / "gen21_matched_causal_mechanisms_progress.json"
    )
    result = run_gen21(config, device=args.device, progress_path=progress)
    paths = result.save(args.output_dir, plot=not args.no_plot)
    paths["progress"] = progress
    paths.update(bundle_gen21_artifacts(paths, args.output_dir))
    print(json.dumps({
        "paths": paths,
        "device": result.device,
        "dataset": result.dataset,
        "promoted_arms": result.promoted_arms,
        "decision": result.decision,
        "summary": result.summary,
    }, indent=2))


if __name__ == "__main__":
    main()
