"""Run the Gen-24 compiled residual-state systems benchmark on SSC."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ammc_gen5 import (  # noqa: E402
    Gen24Config,
    available_gen24_models,
    bundle_gen24_artifacts,
    run_gen24,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default="gen5_data/ssc")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--test-samples", type=int, default=2048)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 32, 256])
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--measurement-iterations", type=int, default=40)
    parser.add_argument("--measurement-repeats", type=int, default=3)
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--output-dir", default="gen5_outputs/gen24_compiled_residual_state_cuda")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if args.list_models:
        print("\n".join(available_gen24_models()))
        return
    config = Gen24Config(
        data_root=args.data_root,
        download=not args.no_download,
        test_samples=args.test_samples,
        batch_sizes=tuple(args.batch_sizes),
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
        measurement_repeats=args.measurement_repeats,
        compile_mode=args.compile_mode,
    )
    result = run_gen24(config, device=args.device)
    paths = result.save(args.output_dir, plot=not args.no_plot)
    paths.update(bundle_gen24_artifacts(paths, args.output_dir))
    print(json.dumps({
        "paths": paths,
        "device": result.device,
        "architecture": result.architecture,
        "decision": result.decision,
        "summary": result.summary,
    }, indent=2))


if __name__ == "__main__":
    main()
