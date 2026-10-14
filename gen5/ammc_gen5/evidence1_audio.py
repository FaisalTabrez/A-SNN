"""Canonical paired audio evidence, with resumable validation-selected training.

The smoke profile uses synthetic data and cannot return a research decision.
Official confirmation compares capacity-matched residual LIF, Conv1D, TCN and
GRU. Causal interventions reuse the full residual checkpoint without training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import statistics
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from scipy.stats import t as student_t
from .artifact_integrity import bind_protocol, identity, runtime_identity, seal_files, verify_files

from .runtime import seed_everything
from .shd_benchmark import SHDConfig, load_shd_tensors, ensure_shd_files
from .shd_calibrated_baselines import TemporalConvClassifier, matched_temporal_conv_channels
from .shd_matched_baselines import GRUTemporalClassifier, matched_gru_hidden_units
from .shd_state_placement_diagnostic import (
    ResidualTemporalConvStateClassifier, matched_temporal_conv_residual_channels,
    _multiscale_features,
)
from .shd_validation_checkpoint import _stratified_split
from .ssc_benchmark import load_ssc_tensors, ensure_ssc_files
from .ssc_efficiency_baselines import TemporalDilatedTCNClassifier, matched_temporal_tcn_channels

ARMS = ("residual_lif", "conv1d", "tcn", "gru")
CONFIRMATION_SEEDS = tuple(range(310, 320))
IMPLEMENTATION_SCHEMA = 2


@dataclass(frozen=True)
class EvidenceConfig:
    profile: str = "smoke"
    datasets: tuple[str, ...] = ("shd", "ssc")
    seeds: tuple[int, ...] = (310,)
    epochs: int = 2
    batch_size: int = 16
    target_parameters: int = 2000
    timesteps: int = 8
    learning_rate: float = 0.003
    weight_decay: float = 0.0001
    validation_fraction: float = 0.2
    split_seed: int = 2026
    temporal_levels: tuple[int, ...] = (1, 2, 4)
    shuffle_repeats: int = 3
    accuracy_margin: float = 0.01
    compile_model: bool = False
    benchmark_iterations: int = 20
    benchmark_repeats: int = 3

    @classmethod
    def confirmation(cls):
        return cls(profile="confirmation", seeds=CONFIRMATION_SEEDS, epochs=15,
                   batch_size=256, target_parameters=133631, timesteps=64,
                   temporal_levels=(1, 2, 4, 8), compile_model=True)


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_fingerprint():
    package = Path(__file__).resolve().parent
    # Include transitive helpers too; a loader/activation change must not resume
    # an old run merely because the directly imported module is unchanged.
    return {p.relative_to(package).as_posix(): digest(p) for p in sorted(package.rglob("*.py"))}


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class CausalResidual(ResidualTemporalConvStateClassifier):
    """Use global sample partners rather than within-batch state rotations."""
    def forward(self, events, *, intervention="full", partner_events=None, return_event_rate=False):
        direct, state, final, activity = self.encode_trace(events)
        if intervention == "zero_state":
            state, final = torch.zeros_like(state), torch.zeros_like(final)
        elif intervention == "shuffled_state":
            if partner_events is None:
                raise ValueError("Shuffled state requires globally deranged partner events")
            _, state, final, _ = self.encode_trace(partner_events)
        elif intervention != "full":
            raise ValueError(f"Unknown intervention: {intervention}")
        features = _multiscale_features(direct, self.temporal_levels)
        features.extend(_multiscale_features(state, self.temporal_levels))
        logits = self.classifier(torch.cat([*features, final], dim=1))
        return (logits, activity) if return_event_rate else logits


def make_model(arm, config, inputs, classes, duration):
    base = SHDConfig(input_neurons=inputs, classes=classes, timesteps=config.timesteps,
                     duration_seconds=duration)
    common = dict(kernel_size=5, temporal_levels=config.temporal_levels)
    if arm == "residual_lif":
        width, _ = matched_temporal_conv_residual_channels(inputs, classes, config.target_parameters, **common)
        model = CausalResidual(base, channels=width, dynamics="lif", surrogate_slope=10.0, **common)
    elif arm == "conv1d":
        width, _ = matched_temporal_conv_channels(inputs, classes, config.target_parameters, **common)
        model = TemporalConvClassifier(base, channels=width, **common)
    elif arm == "tcn":
        options = dict(input_kernel_size=5, hidden_kernel_size=3, temporal_levels=config.temporal_levels)
        width, _ = matched_temporal_tcn_channels(inputs, classes, config.target_parameters, **options)
        model = TemporalDilatedTCNClassifier(base, channels=width, dilation=2, **options)
    elif arm == "gru":
        width, _ = matched_gru_hidden_units(inputs, classes, config.target_parameters)
        model = GRUTemporalClassifier(base, hidden_units=width)
    else:
        raise ValueError(arm)
    count = sum(parameter.numel() for parameter in model.parameters())
    if count > config.target_parameters:
        raise ValueError(f"{arm} exceeds parameter budget: {count}")
    if config.profile == "confirmation" and count < 0.95 * config.target_parameters:
        raise ValueError(f"{arm} has less than 95% of the matched budget")
    return model


def synthetic_data(config, dataset):
    generator = torch.Generator().manual_seed(config.split_seed + (dataset == "ssc"))
    def split(samples):
        events = (torch.rand(samples, config.timesteps, 12, generator=generator) < 0.12).to(torch.uint8)
        labels = torch.arange(samples) % 3
        events[torch.arange(samples), -1, labels] = 1
        return events, labels
    return (*split(48), *split(18), *split(19)), {"kind": "synthetic", "inputs": 12, "classes": 3, "duration": 1.0}


def official_data(config, dataset, data_root):
    root = Path(data_root) / dataset
    root.mkdir(parents=True, exist_ok=True)
    ensure = ensure_shd_files if dataset == "shd" else ensure_ssc_files
    ensure(root, download=True)
    files = {p.name: digest(p) for p in sorted(root.glob("*.h5"))}
    binding = {"schema": IMPLEMENTATION_SCHEMA, "raw_sha256": files, "timesteps": config.timesteps,
               "duration": 1.4 if dataset == "shd" else 1.0,
               "source_sha256": source_fingerprint()}
    binding_path = root / "evidence1_cache_sources.json"
    existing_caches = list(root.glob(f"*t{config.timesteps}_*.pt"))
    receipt = 'evidence1_encoded_integrity.json'
    if binding_path.exists():
        if json.loads(binding_path.read_text()) != binding:
            raise ValueError("Raw data/encoder changed; use a fresh data directory (no cache is deleted)")
    elif existing_caches:
        raise ValueError("Unbound historical encoded caches; use a fresh Evidence-1 data directory")
    else:
        atomic_json(binding_path, binding)
    if existing_caches or (root / receipt).exists():
        verify_files(root, sorted(p.name for p in existing_caches), receipt, binding)
    base = SHDConfig(data_root=str(root), input_neurons=700, classes=20 if dataset == "shd" else 35,
                     timesteps=config.timesteps, duration_seconds=1.4 if dataset == "shd" else 1.0,
                     data_seed=config.split_seed, train_samples=0, test_samples=0)
    if dataset == "shd":
        x, y, test_x, test_y = load_shd_tensors(base)
        parts = (*_stratified_split(x, y, fraction=config.validation_fraction, seed=config.split_seed), test_x, test_y)
    else:
        parts = load_ssc_tensors(base, validation_samples=0)
    caches = {p.name: digest(p) for p in sorted(root.glob(f"*t{config.timesteps}_*.pt"))}
    if not existing_caches:
        seal_files(root, sorted(caches), receipt, binding)
    return parts, {"kind": "official", "inputs": 700, "classes": base.classes,
                   "duration": base.duration_seconds, "raw_sha256": files, "cache_sha256": caches,
                   "split_seed": config.split_seed, "validation_fraction": config.validation_fraction if dataset == "shd" else "official"}


def derangement(samples, seed):
    if samples < 2:
        raise ValueError("State specificity needs at least two samples")
    order = torch.randperm(samples, generator=torch.Generator().manual_seed(seed))
    partners = torch.empty_like(order)
    partners[order] = order.roll(1)
    return partners


def evaluate(model, events, labels, config, device, *, intervention="full", seed=0):
    if len(labels) == 0 or len(events) != len(labels):
        raise ValueError('Evaluation needs nonempty, aligned events and labels')
    model.eval()
    partners = derangement(len(labels), seed) if intervention == "shuffled_state" else None
    correct, activity_sum = 0, 0.0
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(labels), config.batch_size):
            stop = start + config.batch_size
            batch = events[start:stop].to(device)
            kwargs = dict(return_event_rate=True)
            if isinstance(model, CausalResidual):
                kwargs.update(intervention=intervention)
                if partners is not None:
                    kwargs["partner_events"] = events[partners[start:stop]].to(device)
            logits, activity = model(batch, **kwargs)
            require_finite(logits, 'evaluation logits')
            require_finite(torch.as_tensor(activity), 'evaluation activity')
            if logits.ndim != 2 or logits.shape[0] != len(batch):
                raise ValueError('Invalid evaluation logit shape')
            prediction = logits.argmax(1).cpu()
            correct += int((prediction == labels[start:stop]).sum())
            predictions.extend(prediction.tolist())
            activity_sum += float(activity) * len(prediction)
    return {"accuracy": correct / len(labels), "activity": activity_sum / len(labels),
            "prediction_sha256": hashlib.sha256(bytes(predictions)).hexdigest()}


def require_finite(value, label):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f'Nonfinite {label}; terminal numerical failure')


def train(model, parts, config, seed, device, path, *, trial_context=None):
    path = Path(path)
    x, y, valid_x, valid_y = parts[:4]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    start_epoch, best_accuracy, best_state, best_epoch, elapsed, peak_train = 0, -1.0, None, 0, 0.0, 0.0
    initial_hash = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    # Direct helper calls may extend the epoch budget, but cannot change the trial.
    training_config = {k: v for k, v in asdict(config).items() if k != 'epochs'}
    binding = {'schema': IMPLEMENTATION_SCHEMA, 'seed': seed, 'model': type(model).__name__,
               'configuration': identity(training_config), 'trial_context': trial_context}
    receipt = path.name + '.integrity.json'
    history = []
    if path.exists():
        verify_files(path.parent, [path.name], receipt, binding)
        saved = torch.load(path, map_location=device, weights_only=True)
        if saved.get('binding') != binding or saved.get('initial_parameter_sha256') != initial_hash:
            raise ValueError('Checkpoint trial/initialization identity changed')
        if not 0 < saved['epoch'] <= config.epochs or saved['best_epoch'] > saved['epoch']:
            raise ValueError('Invalid checkpoint epoch history')
        initial_hash = saved['initial_parameter_sha256']
        model.load_state_dict(saved["last_state"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch, best_accuracy, best_epoch = saved["epoch"], saved["best_accuracy"], saved["best_epoch"]
        best_state, elapsed, history = saved["best_state"], saved["train_seconds"], saved["history"]
        peak_train = saved.get("cuda_peak_train_mb", 0.0)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, config.epochs):
        seed_everything(seed + epoch * 10007, device)
        order = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed + epoch * 10007 + 50000))
        model.train()
        _sync(device)
        started = time.perf_counter()
        for offset in range(0, len(y), config.batch_size):
            index = order[offset:offset + config.batch_size]
            batch = x[index].to(device)
            target = y[index].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(batch), target)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite loss; terminal numerical failure")
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None:
                    require_finite(parameter.grad, 'training gradient')
            optimizer.step()
            for parameter in model.parameters():
                require_finite(parameter, 'updated parameter')
        valid = evaluate(model, valid_x, valid_y, config, device)["accuracy"]
        _sync(device)
        elapsed += time.perf_counter() - started
        history.append({"epoch": epoch + 1, "validation_accuracy": valid})
        if valid > best_accuracy:
            best_accuracy, best_epoch = valid, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        temporary = path.with_suffix(".tmp")
        if device.type == "cuda":
            peak_train = max(peak_train, torch.cuda.max_memory_allocated(device) / 2**20)
        torch.save({"binding": binding, "initial_parameter_sha256": initial_hash,
                    "last_state": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1, "best_accuracy": best_accuracy, "best_epoch": best_epoch,
                    "best_state": best_state, "train_seconds": elapsed, "history": history,
                    "cuda_peak_train_mb": peak_train}, temporary)
        temporary.replace(path)
        seal_files(path.parent, [path.name], receipt, binding)
        print(f"  epoch {epoch+1}/{config.epochs} validation={valid:.4f}", flush=True)
    model.load_state_dict(best_state)
    return {"best_validation_accuracy": best_accuracy, "best_epoch": best_epoch,
            "train_seconds": elapsed, "initial_parameter_sha256": initial_hash,
            "cuda_peak_train_mb": peak_train if device.type == "cuda" else None}


def benchmark(model, validation, config, device):
    if not config.compile_model or device.type != "cuda":
        return {"status": "not_measured", "reason": "CUDA compilation disabled or unavailable"}
    batch = validation[:min(config.batch_size, len(validation))].to(device)
    model.eval()
    def measure(function):
        with torch.inference_mode():
            for _ in range(5):
                function(batch)
            _sync(device)
            timings = []
            for _ in range(config.benchmark_repeats):
                started = time.perf_counter()
                for _ in range(config.benchmark_iterations):
                    function(batch)
                _sync(device)
                timings.append(time.perf_counter() - started)
        return len(batch) * config.benchmark_iterations / statistics.median(timings)
    eager = measure(model)
    try:
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        compiled = torch.compile(model, fullgraph=True, mode="reduce-overhead")
        with torch.inference_mode():
            compiled(batch)
            _sync(device)
        first_shape_compilation = time.perf_counter() - started
        agreement, total, maximum_difference = 0, 0, 0.0
        # Use validation for compiler fidelity; no checkpoint or model selection occurs here.
        with torch.inference_mode():
            for offset in range(0, len(validation), config.batch_size):
                probe = validation[offset:offset + config.batch_size].to(device)
                a, b = model(probe), compiled(probe)
                require_finite(a, 'eager fidelity logits')
                require_finite(b, 'compiled fidelity logits')
                require_finite(a - b, 'fidelity difference')
                maximum_difference = max(maximum_difference, float((a - b).abs().max()))
                agreement += int((a.argmax(1) == b.argmax(1)).sum())
                total += len(probe)
        rate = measure(compiled)
        return {"status": "measured", "first_shape_compile_seconds": first_shape_compilation,
                "cuda_peak_inference_mb": torch.cuda.max_memory_allocated(device) / 2**20,
                "eager_examples_per_second": eager, "compiled_examples_per_second": rate,
                "speedup": rate / eager, "prediction_agreement": agreement / total,
                "maximum_logit_difference": maximum_difference,
                "fidelity_samples": total, "timing_batch_size": len(batch)}
    except Exception as error:
        return {"status": "numerical_failure" if isinstance(error, FloatingPointError) else "compile_failed",
                "error": f"{type(error).__name__}: {error}",
                "eager_examples_per_second": eager}


def confidence(values):
    values = list(values)
    if not values or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError('Confidence estimates require finite numeric observations')
    mean = statistics.fmean(values)
    if len(values) < 2:
        return {"mean": mean, "standard_deviation": None, "ci95_lower": None, "ci95_upper": None, "n": len(values)}
    half = float(student_t.ppf(0.975, len(values) - 1)) * statistics.stdev(values) / math.sqrt(len(values))
    return {"mean": mean, "standard_deviation": statistics.stdev(values),
            "ci95_lower": mean - half, "ci95_upper": mean + half, "n": len(values)}


def validate_record(row, config, *, complete_record=False):
    def number(key, low=0., high=None):
        value = row[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < low or (high is not None and value > high):
            raise ValueError(f'Invalid trial metric: {key}')
    if (row.get('dataset') not in config.datasets or row.get('arm') not in ARMS or
            isinstance(row.get('seed'), bool) or not isinstance(row.get('seed'), int) or row.get('seed') not in config.seeds):
        raise ValueError('Unexpected trial identity')
    number('accuracy', high=1.)
    systems = row.get('systems', {})
    if systems.get('status') not in {'measured', 'not_measured', 'compile_failed', 'numerical_failure'}:
        raise ValueError('Invalid systems status')
    if systems.get('status') == 'measured':
        for key in ['speedup', 'prediction_agreement', 'maximum_logit_difference']:
            value = systems.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'Invalid systems metric: {key}')
        if systems['prediction_agreement'] > 1:
            raise ValueError('Invalid prediction agreement')
    if row['arm'] == 'residual_lif':
        for key in ['zero_state_accuracy', 'shuffled_state_accuracy']:
            number(key, high=1.)
    if complete_record:
        for key in ['best_validation_accuracy', 'activity']:
            number(key, high=1. if key == 'best_validation_accuracy' else None)
        number('train_seconds')
        if isinstance(row.get('best_epoch'), bool) or not isinstance(row.get('best_epoch'), int) or not 1 <= row['best_epoch'] <= config.epochs:
            raise ValueError('Invalid selected epoch')
        count = row.get('parameters')
        if isinstance(count, bool) or not isinstance(count, int) or not 0 < count <= config.target_parameters or (config.profile == 'confirmation' and count < .95 * config.target_parameters):
            raise ValueError('Invalid parameter accounting')
        for key in ['initial_parameter_sha256', 'prediction_sha256']:
            if not isinstance(row.get(key), str) or not re.fullmatch('[0-9a-f]{64}', row[key]):
                raise ValueError(f'Invalid hash: {key}')
        if row['arm'] == 'residual_lif':
            shuffles = row.get('shuffle_repeat_accuracies', [])
            if len(shuffles) != config.shuffle_repeats or any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in shuffles):
                raise ValueError('Invalid shuffle repetitions')
            if abs(statistics.fmean(shuffles) - row['shuffled_state_accuracy']) > 1e-12:
                raise ValueError('Shuffle mean mismatch')


def summarize(records, config):
    if not config.datasets or not config.seeds or len(set(config.datasets)) != len(config.datasets):
        raise ValueError('Invalid registered dataset/seed sets')
    expected = {(d, s, a) for d in config.datasets for s in config.seeds for a in ARMS}
    seen = set()
    for row in records:
        validate_record(row, config)
        key = (row['dataset'], row['seed'], row['arm'])
        if key in seen or key not in expected:
            raise ValueError('Duplicate or unexpected trial key')
        seen.add(key)
    summaries = {}
    complete = seen == expected
    for dataset in config.datasets:
        groups = {arm: {r["seed"]: r for r in records if r["dataset"] == dataset and r["arm"] == arm} for arm in ARMS}
        if any(set(group) != set(config.seeds) for group in groups.values()):
            continue
        residual = groups["residual_lif"]
        causal = {mode: confidence(residual[s]["accuracy"] - residual[s][mode + "_accuracy"] for s in config.seeds)
                  for mode in ("zero_state", "shuffled_state")}
        parity = {arm: confidence(residual[s]["accuracy"] - groups[arm][s]["accuracy"] for s in config.seeds)
                  for arm in ("conv1d", "tcn", "gru")}
        positive = lambda estimate, margin=0: estimate["ci95_lower"] is not None and estimate["ci95_lower"] > margin
        systems = all(r["systems"].get("status") == "measured" and r["systems"].get("speedup", 0) > 1
                      and r["systems"].get("prediction_agreement", 0) == 1.0
                      and r["systems"].get("maximum_logit_difference", 1) <= 1e-4 for r in residual.values())
        summaries[dataset] = {"accuracy": {arm: confidence(r["accuracy"] for r in groups[arm].values()) for arm in ARMS},
                              "causal_deltas": causal, "paired_baseline_deltas": parity,
                              "causal_gate": all(positive(c) for c in causal.values()),
                              "accuracy_gate": all(positive(c, -config.accuracy_margin) for c in parity.values()),
                              "systems_gate": systems}
    eligible = complete and set(summaries) == set(config.datasets) and config == EvidenceConfig.confirmation()
    if not eligible:
        decision = "smoke_only" if config.profile == "smoke" else "incomplete_or_modified_protocol"
    elif not all(s["causal_gate"] for s in summaries.values()):
        decision = "stop_causal_claim"
    elif not all(s["accuracy_gate"] for s in summaries.values()):
        decision = "causal_audit_only"
    elif not all(s["systems_gate"] for s in summaries.values()):
        decision = "model_gates_pass_systems_unconfirmed"
    else:
        decision = "model_and_systems_gates_pass_replication_pending"
    return {"decision": decision, "replication_gate": "pending_independent_clean_clone",
            "complete": complete, "datasets": summaries}


def bundle(output):
    paths = sorted(p for p in output.rglob("*") if p.is_file() and p.suffix not in {".tmp", ".zip"} and p.name != "manifest.json")
    manifest = {p.relative_to(output).as_posix(): {"sha256": digest(p), "bytes": p.stat().st_size} for p in paths}
    atomic_json(output / "manifest.json", {"schema": 1, "files": manifest})
    archive = output / "evidence1_bundle.zip"
    temporary = output / "evidence1_bundle.zip.tmp"
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as zipped:
        for p in [*paths, output / "manifest.json"]:
            zipped.write(p, p.relative_to(output).as_posix())
    temporary.replace(archive)
    return archive


def run(config, output, data_root, device="auto"):
    output, data_root = Path(output), Path(data_root)
    output.mkdir(parents=True, exist_ok=True)
    if config.profile not in {"smoke", "confirmation"} or not config.datasets or any(d not in {"shd", "ssc"} for d in config.datasets):
        raise ValueError("Invalid profile/dataset configuration")
    if len(set(config.datasets)) != len(config.datasets) or len(set(config.seeds)) != len(config.seeds) or not config.seeds or config.epochs < 1 or config.batch_size < 2:
        raise ValueError("Seeds must be unique; epochs positive; batch size at least two")
    resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    if resolved.type not in {"cuda", "cpu"}:
        raise ValueError("This confirmation runner supports CPU/CUDA only")
    if resolved.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    lock = {"schema": IMPLEMENTATION_SCHEMA, "runtime": runtime_identity(torch),
            "config": asdict(config), "source_sha256": source_fingerprint(), "torch_version": str(torch.__version__),
            "device": str(resolved), "python": platform.python_version(), "platform": platform.platform(),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(resolved) if resolved.type == "cuda" else None}
    signature = hashlib.sha256(json.dumps(lock, sort_keys=True).encode()).hexdigest()
    lock["sha256"] = signature
    if (output / "protocol.json").exists():
        if json.loads((output / "protocol.json").read_text())["sha256"] != signature:
            raise ValueError("Protocol/source/environment changed: choose a new output directory")
        bind_protocol(output / 'protocol.json', lock, 'Protocol/source/environment')
    else:
        bind_protocol(output / 'protocol.json', lock, 'Protocol/source/environment')
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    atomic_json(output / "environment.json", {"git_commit": commit, "runtime": lock['runtime'], "python": sys.version, "torch": str(torch.__version__),
                "cuda": torch.version.cuda, "device": str(resolved), "gpu": torch.cuda.get_device_name(resolved) if resolved.type == "cuda" else None})
    if resolved.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    records = []
    for dataset in config.datasets:
        parts, metadata = synthetic_data(config, dataset) if config.profile == "smoke" else official_data(config, dataset, data_root)
        provenance_path = output / f"{dataset}_data.json"
        metadata["split_shapes"] = [list(t.shape) for t in parts]
        if provenance_path.exists() and json.loads(provenance_path.read_text()) != metadata:
            raise ValueError("Dataset provenance changed: choose a new output directory")
        atomic_json(provenance_path, metadata)
        for seed in config.seeds:
            for arm in ARMS:
                trial = output / f"{dataset}_{seed}_{arm}"
                trial.mkdir(exist_ok=True)
                record_path = trial / "record.json"
                context = {'protocol_sha256': signature, 'dataset_sha256': identity(metadata),
                           'dataset': dataset, 'seed': seed, 'arm': arm}
                if record_path.exists():
                    verify_files(trial, ['record.json', 'checkpoint.pt'], 'trial_integrity.json', context)
                    existing = json.loads(record_path.read_text())
                    validate_record(existing, config, complete_record=True)
                    if (existing['dataset'], existing['seed'], existing['arm']) != (dataset, seed, arm):
                        raise ValueError('Completed record belongs to a different trial')
                    records.append(existing)
                    continue
                print(f"[{dataset}] seed={seed} arm={arm}", flush=True)
                seed_everything(seed, resolved)
                model = make_model(arm, config, metadata["inputs"], metadata["classes"], metadata["duration"]).to(resolved)
                training = train(model, parts, config, seed, resolved, trial / "checkpoint.pt", trial_context=context)
                metrics = evaluate(model, parts[4], parts[5], config, resolved)
                record = {"dataset": dataset, "seed": seed, "arm": arm, **training, **metrics,
                          "parameters": sum(p.numel() for p in model.parameters()),
                          "activity_kind": "spike_density" if arm == "residual_lif" else "analog_activation",
                          "systems": benchmark(model, parts[2], config, resolved)}
                if arm == "residual_lif":
                    record["zero_state_accuracy"] = evaluate(model, parts[4], parts[5], config, resolved, intervention="zero_state")["accuracy"]
                    shuffled = [evaluate(model, parts[4], parts[5], config, resolved, intervention="shuffled_state", seed=seed + 80000 + i)["accuracy"] for i in range(config.shuffle_repeats)]
                    record["shuffled_state_accuracy"] = statistics.fmean(shuffled)
                    record["shuffle_repeat_accuracies"] = shuffled
                validate_record(record, config, complete_record=True)
                atomic_json(record_path, record)
                seal_files(trial, ['record.json', 'checkpoint.pt'], 'trial_integrity.json', context)
                records.append(record)
                atomic_json(output / "results.json", {"protocol_sha256": signature, "records": records, "summary": summarize(records, config)})
                del model
                if resolved.type == "cuda":
                    torch.cuda.empty_cache()
        del parts
    payload = {"protocol_sha256": signature, "records": records, "summary": summarize(records, config)}
    atomic_json(output / "results.json", payload)
    columns = ("dataset", "seed", "arm", "accuracy", "best_validation_accuracy", "best_epoch", "parameters", "activity", "activity_kind", "train_seconds")
    with (output / "records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(payload["summary"], indent=2), flush=True)
    print(f"Checksummed bundle: {bundle(output)}", flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "confirmation"), default="smoke")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("gen5_data"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    return run(EvidenceConfig.confirmation() if args.profile == "confirmation" else EvidenceConfig(), args.output_dir, args.data_root, args.device)


if __name__ == "__main__":
    main()
