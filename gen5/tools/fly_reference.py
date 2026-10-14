"""Run a pinned whole-fly reference diagnostic, not an A-SNN mechanism claim."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ammc_gen5.artifact_integrity import bind_protocol, identity, runtime_identity, seal_files, verify_files

REVISION = "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"
SUGAR = [720575940624963786, 720575940630233916, 720575940637568838,
         720575940638202345, 720575940617000768, 720575940630797113,
         720575940632889389, 720575940621754367, 720575940621502051,
         720575940640649691, 720575940639332736, 720575940616885538,
         720575940639198653, 720575940620900446, 720575940617937543,
         720575940632425919, 720575940633143833, 720575940612670570,
         720575940628853239, 720575940629176663, 720575940611875570]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "reference"), default="smoke")
    args = parser.parse_args()
    root, output = args.upstream.resolve(), args.output_dir.resolve()
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if head != REVISION or dirty:
        raise ValueError("Fly source/data must be clean at the pinned revision")
    import brian2 as b2
    import pandas as pd
    sys.path.insert(0, str(root))
    import model
    import utils
    files = [root / "2023_03_23_completeness_630_final.csv", root / "2023_03_23_connectivity_630_final.parquet"]
    output.mkdir(parents=True, exist_ok=True)
    # Python codegen is portable and avoids silently installing a C++ toolchain.
    # It is slower than the author's compiled environment; no throughput claim.
    b2.prefs.codegen.target = "numpy"
    n, duration = (1, 0.1) if args.profile == "smoke" else (30, 1.0)
    lock = {"schema": 2, "runtime": runtime_identity(), "kind": "reference_diagnostic_not_ASNN_mechanism", "upstream": head,
            "profile": args.profile, "seed": 310, "n_run": n, "duration_seconds": duration,
            "brian2": b2.__version__, "python": sys.version, "codegen": "numpy",
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "inputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    protocol = output / "fly_protocol.json"
    bind_protocol(protocol, lock, 'Fly protocol')
    for name, hz in [("sugarR", 150), ("sugarR_100Hz", 100)]:
        binding = {'protocol': identity(lock), 'experiment': name, 'rate_hz': hz}
        if (output / (name + '.parquet')).exists():
            verify_files(output, [name + '.parquet'], name + '_integrity.json', binding)
        b2.seed(310)
        params = dict(model.default_params, n_run=n, t_run=duration * b2.second, r_poi=hz * b2.Hz)
        model.run_exp(name, SUGAR, path_res=str(output), path_comp=str(files[0]), path_con=str(files[1]),
                      params=params, n_proc=1, force_overwrite=False)
        seal_files(output, [name + '.parquet'], name + '_integrity.json', binding)
    combined = pd.concat([pd.read_parquet(output / (name + ".parquet")) for name in ("sugarR", "sugarR_100Hz")])
    rates, deviations = utils.get_rate(combined, duration, n)
    rates.to_csv(output / "firing_rates_hz.csv")
    deviations.to_csv(output / "firing_rates_std_hz.csv")
    # Compare author examples descriptively only: different stochastic trials.
    author = pd.concat([pd.read_parquet(root / "results" / "example" / (name + ".parquet"))
                        for name in ("sugarR", "sugarR_100Hz")])
    author_rates, _ = utils.get_rate(author, 1.0, 30)
    author_rates.to_csv(output / "author_example_rates_hz.csv")
    target = 720575940660219265
    print("MN9 reference readout (Hz; absent spiking rows are zero):")
    print(rates.reindex([target]).fillna(0))
    print("Author example (different stochastic runs):")
    print(author_rates.reindex([target]).fillna(0))
    print("No paper-reproduction or A-SNN causal gate is claimed by this diagnostic.")


if __name__ == "__main__":
    main()
