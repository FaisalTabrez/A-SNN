"""Pinned FiTS native-recipe reference, selecting on validation and testing once.

This is not an Evidence-1 matched control. Run in a subprocess; the adapter
replaces only the upstream fit function and leaves its data/model unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ammc_gen5.artifact_integrity import bind_protocol, identity, runtime_identity, seal_files, verify_files

FITS_REVISION = "8fe7e376bae13fe75f357dcfbff6adf9c45f1077"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def validation_fit(model, cfg, out_dir, device, train_loader, test_loader,
                   valid_loader=None, prep=None, extra_config=None):
    if valid_loader is None:
        raise ValueError("Final-test selection is forbidden; use SHD valid_split or SSC official validation")
    from common.training import create_optimizer, run_epoch
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    configuration = {**vars(cfg), **(extra_config or {})}
    config_path = out_dir / "validation_reference_config.json"
    managed = ['validation_reference_summary.json', 'validation_reference_checkpoint.pt', 'validation_reference_history.json']
    if not config_path.exists() and any((out_dir / name).exists() for name in managed):
        raise ValueError('Unbound FiTS artifacts; choose a fresh output directory')
    if config_path.exists() and identity(json.loads(config_path.read_text())) != identity(configuration):
        raise ValueError("FiTS configuration changed; use a new results directory")
    write_json(config_path, configuration)
    binding = {'schema': 2, 'configuration': identity(configuration),
               'protocol': sha256(out_dir / 'validation_reference_protocol.json') if (out_dir / 'validation_reference_protocol.json').exists() else None}
    summary_path = out_dir / "validation_reference_summary.json"
    if summary_path.exists():
        verify_files(out_dir, managed, 'validation_reference_complete_integrity.json', binding)
        summary = json.loads(summary_path.read_text())
        if summary.get('selection') != 'validation_only' or summary.get('test_evaluations') != 1:
            raise ValueError('Invalid FiTS summary contract')
        return summary
    optimizer, scheduler, guard = create_optimizer(model, cfg)
    criterion = torch.nn.CrossEntropyLoss()
    checkpoint = out_dir / "validation_reference_checkpoint.pt"
    first, best, best_epoch, best_state, history = 1, -1.0, 0, None, []
    if checkpoint.exists():
        verify_files(out_dir, [checkpoint.name], 'validation_reference_checkpoint_integrity.json', binding)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved.get('binding') != binding:
            raise ValueError('FiTS checkpoint binding changed')
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        first = saved["epoch"] + 1
        best, best_epoch, best_state, history = saved["best"], saved["best_epoch"], saved["best_state"], saved["history"]
        random.setstate(saved["random"])
        np.random.set_state(saved["numpy"])
        torch.set_rng_state(saved["torch"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved["cuda"])
    for epoch in range(first, cfg.epochs + 1):
        loss, accuracy = run_epoch(model, train_loader, criterion, device, optimizer, prep)
        _, validation = run_epoch(model, valid_loader, criterion, device, prep=prep)
        if not np.isfinite(loss) or not np.isfinite(validation):
            raise RuntimeError("Nonfinite FiTS training/validation metric")
        history.append({"epoch": epoch, "train_loss": loss, "train_acc": accuracy, "valid_acc": validation})
        if validation > best:
            best, best_epoch = validation, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        scheduler.step()
        if guard is not None:
            guard.reset_counter()
        temporary = checkpoint.with_suffix(".tmp")
        torch.save({"binding": binding, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "epoch": epoch, "best": best, "best_epoch": best_epoch, "best_state": best_state, "history": history,
                    "random": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, temporary)
        temporary.replace(checkpoint)
        seal_files(out_dir, [checkpoint.name], 'validation_reference_checkpoint_integrity.json', binding)
        print(f"FiTS epoch {epoch}/{cfg.epochs} validation={validation:.4f}", flush=True)
    model.load_state_dict(best_state)
    _, test = run_epoch(model, test_loader, criterion, device, prep=prep)
    if not np.isfinite(test):
        raise FloatingPointError('Nonfinite FiTS final evaluation')
    summary = {"selection": "validation_only", "best_epoch": best_epoch, "best_validation_accuracy": best,
               "test_accuracy": test, "test_evaluations": 1, "kind": "native_recipe_not_matched_Evidence1_control"}
    write_json(out_dir / "validation_reference_history.json", history)
    write_json(summary_path, summary)
    seal_files(out_dir, managed, 'validation_reference_complete_integrity.json', binding)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--dataset", choices=("shd", "ssc"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=310)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    upstream = args.upstream.resolve()
    actual = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if actual != FITS_REVISION or dirty:
        raise ValueError("FiTS checkout must be clean at the pinned revision")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = args.dataset.upper()
    recipe = upstream / "config" / dataset / ("fits_o2_w256.yaml" if dataset == "SHD" else "fits_o1_w512.yaml")
    data = args.data_root.resolve() / args.dataset
    files = {name: sha256(data / name) for name in
             (["shd_train.h5", "shd_test.h5"] if dataset == "SHD" else ["ssc_train.h5", "ssc_valid.h5", "ssc_test.h5"])}
    signature = {"schema": 2, "runtime": runtime_identity(torch), "upstream": actual, "adapter": sha256(__file__), "recipe": sha256(recipe), "raw_data_sha256": files,
                 "torch": str(torch.__version__), "python": sys.version, "seed": args.seed, "epochs": args.epochs,
                 "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    lock = args.output_dir / "validation_reference_protocol.json"
    bind_protocol(lock, signature, 'FiTS protocol/environment/data')
    sys.path.insert(0, str(upstream))
    module = importlib.import_module(f"{dataset}.scripts.train.train_chunked")
    module.fit = validation_fit
    sys.argv = ["fits_reference", "--config", str(recipe), "--data-dir", str(data),
                "--results-dir", str(args.output_dir.resolve()), "--seed", str(args.seed),
                "--epochs", str(args.epochs), "--workers", "0", "--backend", "torch"]
    if dataset == "SHD":
        sys.argv.extend(["--eval-mode", "valid_split"])
    module.main()


if __name__ == "__main__":
    main()
