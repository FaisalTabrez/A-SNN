"""Phase 31 transfer benchmark on Spiking Heidelberg Digits (SHD).

This phase deliberately stops tuning the row-sequential MNIST proxy. It tests
whether the fixed heterogeneous delays retained from Phase 29 transfer to a
dataset whose timing is part of the observation itself. The official SHD HDF5
files are downloaded directly from Zenke Lab and deterministically binned into
event tensors. Event-count controls discard order; paired sparse arms differ
only in whether recurrent edges execute fixed 0/1/2 distance delays.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import pathlib
import shutil
import statistics
import time
from typing import Iterable
import urllib.request

from .delayed_sequential_mnist import assign_fixed_delays, delayed_sparse_current
from .dynamic_sparse import DynamicSparseLinear
from .event_mnist import build_event_reservoir_edges, nn, torch
from .runtime import device_kind, mark_step, resolve_device, seed_everything, sync
from .trainable_temporal_mnist import SurrogateSpike


SHD_BASE_URL = "https://zenkelab.org/datasets"
SHD_FILES = ("shd_train.h5.gz", "shd_test.h5.gz")


@dataclass(frozen=True)
class SHDConfig:
    """Registered Phase 31 dataset, graph, and optimizer configuration."""

    seeds: tuple[int, ...] = (42, 43, 44)
    train_samples: int = 0
    test_samples: int = 0
    input_neurons: int = 700
    classes: int = 20
    timesteps: int = 64
    duration_seconds: float = 1.4
    hidden_neurons: int = 128
    sensor_fanout: int = 1
    recurrent_fanout: int = 4
    max_edges: int = 2048
    reservoir_leak: float = 0.90
    reservoir_threshold: float = 1.0
    input_gain: float = 1.0
    count_hidden_units: int = 128
    epochs: int = 15
    warmup_epochs: int = 5
    learning_rate: float = 0.003
    reservoir_learning_rate: float = 0.0003
    weight_decay: float = 0.0001
    batch_size: int = 256
    data_seed: int = 2026
    data_root: str = "gen5_data/shd"
    download: bool = True

    @property
    def neuron_count(self) -> int:
        return self.input_neurons + self.hidden_neurons


@dataclass(frozen=True)
class SHDArm:
    name: str
    model_kind: str
    delay_pattern: str
    max_delay_steps: int


SHD_ARMS = (
    SHDArm("event_count_linear", "count_linear", "none", 0),
    SHDArm("event_count_mlp", "count_mlp", "none", 0),
    SHDArm("sparse_no_delay_warm_all", "sparse", "none", 0),
    SHDArm(
        "sparse_distance012_warm_all",
        "sparse",
        "distance_0_2",
        2,
    ),
)


def available_shd_arms() -> tuple[str, ...]:
    return tuple(arm.name for arm in SHD_ARMS)


class SHDEventCountClassifier(nn.Module):
    """Timing-ablated classifier over per-channel event counts."""

    def __init__(self, config: SHDConfig, *, kind: str) -> None:
        super().__init__()
        if kind == "linear":
            self.network = nn.Linear(config.input_neurons, config.classes)
        elif kind == "mlp":
            self.network = nn.Sequential(
                nn.Linear(config.input_neurons, config.count_hidden_units),
                nn.ReLU(),
                nn.Linear(config.count_hidden_units, config.classes),
            )
        else:
            raise ValueError(f"unsupported count classifier: {kind}")

    def forward(self, events, *, return_event_rate: bool = False):  # type: ignore[override]
        if events.ndim != 3:
            raise ValueError("events must have shape [batch, time, channels]")
        counts = events.to(torch.float32).sum(dim=1)
        counts = counts / counts.amax(dim=1, keepdim=True).clamp_min(1.0)
        logits = self.network(counts)
        if return_event_rate:
            return logits, events.to(torch.float32).mean()
        return logits


class SHDSparseClassifier(nn.Module):
    """Sparse recurrent LIF classifier with optional fixed delay buckets."""

    def __init__(
        self,
        config: SHDConfig,
        *,
        seed: int,
        delay_pattern: str,
        max_delay_steps: int,
        surrogate_slope: float,
        device,
    ) -> None:
        if torch is None:
            raise ImportError("Phase 31 SHD benchmark requires PyTorch")
        super().__init__()
        self.config = config
        self.input_neurons = config.input_neurons
        self.hidden_neurons = config.hidden_neurons
        self.neuron_count = config.neuron_count
        self.max_delay_steps = int(max_delay_steps)
        self.surrogate_slope = float(surrogate_slope)
        edges = build_event_reservoir_edges(
            config.input_neurons,
            config.hidden_neurons,
            sensor_fanout=config.sensor_fanout,
            recurrent_fanout=config.recurrent_fanout,
            seed=seed,
        )
        if len(edges) > config.max_edges:
            raise ValueError(
                f"SHD reservoir requires {len(edges)} edges; "
                f"capacity is {config.max_edges}"
            )
        self.graph = DynamicSparseLinear(
            config.neuron_count,
            config.neuron_count,
            config.max_edges,
            device=device,
        )
        self.graph.load_edges(edges)
        self.graph.short_term_weight.requires_grad_(False)
        self.graph.long_term_weight.requires_grad_(True)
        assign_fixed_delays(
            self,
            pattern=delay_pattern,
            max_delay_steps=max_delay_steps,
            seed=seed,
        )
        self.readout = nn.Linear(config.hidden_neurons * 2, config.classes)

    @property
    def active_edge_count(self) -> int:
        return self.graph.active_edge_count

    def forward(self, events, *, return_event_rate: bool = False):  # type: ignore[override]
        if events.ndim != 3:
            raise ValueError("events must have shape [batch, time, channels]")
        if events.shape[2] != self.input_neurons:
            raise ValueError(f"events must have {self.input_neurons} channels")
        membrane = events.new_zeros(
            (events.shape[0], self.hidden_neurons), dtype=torch.float32
        )
        hidden_spikes = torch.zeros_like(membrane)
        accumulated_spikes = torch.zeros_like(membrane)
        zero_state = events.new_zeros(
            (events.shape[0], self.neuron_count), dtype=torch.float32
        )
        history: list = []
        for step in range(events.shape[1]):
            sensor_events = events[:, step, :].to(torch.float32) * self.config.input_gain
            network_state = torch.cat((sensor_events, hidden_spikes), dim=1)
            history.insert(0, network_state)
            if len(history) > self.max_delay_steps + 1:
                history.pop()
            current = delayed_sparse_current(
                self.graph,
                history,
                zero_state=zero_state,
                max_delay_steps=self.max_delay_steps,
            )[:, self.input_neurons :]
            pre_reset = membrane * self.config.reservoir_leak + current
            hidden_spikes = SurrogateSpike.apply(
                pre_reset - self.config.reservoir_threshold,
                self.surrogate_slope,
            )
            membrane = pre_reset - hidden_spikes * self.config.reservoir_threshold
            accumulated_spikes = accumulated_spikes + hidden_spikes
        mean_spikes = accumulated_spikes / events.shape[1]
        logits = self.readout(torch.cat((mean_spikes, membrane), dim=1))
        if return_event_rate:
            return logits, mean_spikes.mean()
        return logits

    def clamp_ltw(self, minimum: float, maximum: float) -> None:
        with torch.no_grad():
            self.graph.long_term_weight.clamp_(minimum, maximum)
            self.graph.long_term_weight.mul_(
                self.graph.active_mask.to(self.graph.long_term_weight.dtype)
            )
            self.graph.short_term_weight.zero_()


@dataclass
class SHDBenchmarkResult:
    config: SHDConfig
    device: str
    surrogate_slope: float
    arms: list[dict]
    records: list[dict]
    summary: list[dict]

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "shd_benchmark.json"
        records_path = output / "shd_benchmark_records.csv"
        summary_path = output / "shd_benchmark_summary.csv"
        payload = {
            "config": asdict(self.config),
            "device": self.device,
            "surrogate_slope": self.surrogate_slope,
            "arms": self.arms,
            "records": self.records,
            "summary": self.summary,
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        _write_csv(records_path, self.records)
        _write_csv(summary_path, self.summary)
        paths = {
            "json": str(json_path),
            "records_csv": str(records_path),
            "summary_csv": str(summary_path),
        }
        if plot:
            plot_path = output / "shd_benchmark_summary.png"
            plot_shd_benchmark(self.summary, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def bin_shd_events(
    times,
    units,
    *,
    timesteps: int,
    input_neurons: int,
    duration_seconds: float,
):
    """Bin one variable-length SHD event stream into a dense uint8 tensor."""

    if torch is None:
        raise ImportError("Phase 31 SHD benchmark requires PyTorch")
    if timesteps <= 1 or input_neurons <= 0 or duration_seconds <= 0:
        raise ValueError("invalid SHD binning dimensions")
    time_tensor = torch.as_tensor(times, dtype=torch.float64)
    unit_tensor = torch.as_tensor(units, dtype=torch.long)
    if time_tensor.numel() != unit_tensor.numel():
        raise ValueError("times and units must have equal length")
    output = torch.zeros((timesteps, input_neurons), dtype=torch.uint8)
    if time_tensor.numel() == 0:
        return output
    bins = torch.floor(time_tensor * timesteps / duration_seconds).to(torch.long)
    valid = (
        (bins >= 0)
        & (bins < timesteps)
        & (unit_tensor >= 0)
        & (unit_tensor < input_neurons)
    )
    output[bins[valid], unit_tensor[valid]] = 1
    return output


def load_shd_tensors(config: SHDConfig):
    """Download, verify, bin, and cache the official SHD train/test splits."""

    if torch is None:
        raise ImportError("Phase 31 SHD benchmark requires PyTorch")
    root = pathlib.Path(config.data_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    paths = ensure_shd_files(root, download=config.download)
    train_events, train_labels = _load_or_build_split(
        paths["train"], root, "train", config
    )
    test_events, test_labels = _load_or_build_split(
        paths["test"], root, "test", config
    )
    generator = torch.Generator(device="cpu").manual_seed(config.data_seed)
    if config.train_samples > 0 and config.train_samples < train_events.shape[0]:
        indices = torch.randperm(train_events.shape[0], generator=generator)[
            : config.train_samples
        ]
        train_events = train_events.index_select(0, indices)
        train_labels = train_labels.index_select(0, indices)
    if config.test_samples > 0 and config.test_samples < test_events.shape[0]:
        indices = torch.randperm(test_events.shape[0], generator=generator)[
            : config.test_samples
        ]
        test_events = test_events.index_select(0, indices)
        test_labels = test_labels.index_select(0, indices)
    return train_events, train_labels, test_events, test_labels


def ensure_shd_files(root: pathlib.Path, *, download: bool) -> dict[str, pathlib.Path]:
    """Return decompressed official files, downloading with MD5 verification."""

    resolved = {
        "train": root / "shd_train.h5",
        "test": root / "shd_test.h5",
    }
    if all(path.exists() for path in resolved.values()):
        return resolved
    if not download:
        missing = [str(path) for path in resolved.values() if not path.exists()]
        raise FileNotFoundError("missing SHD files: " + ", ".join(missing))
    md5_text = _download_bytes(f"{SHD_BASE_URL}/md5sums.txt").decode("utf-8")
    hashes = {
        fields[1]: fields[0]
        for line in md5_text.splitlines()
        if len(fields := line.split()) == 2
    }
    for filename in SHD_FILES:
        gz_path = root / filename
        h5_path = root / filename.removesuffix(".gz")
        if h5_path.exists():
            continue
        if not gz_path.exists():
            print(f"Downloading {filename} from Zenke Lab...")
            _download_to(f"{SHD_BASE_URL}/{filename}", gz_path)
        expected = hashes.get(filename)
        if expected and _md5(gz_path) != expected:
            raise RuntimeError(f"MD5 verification failed for {gz_path}")
        temporary = h5_path.with_suffix(h5_path.suffix + ".part")
        with gzip.open(gz_path, "rb") as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
        temporary.replace(h5_path)
    return resolved


def run_shd_benchmark(
    config: SHDConfig,
    *,
    device="auto",
    surrogate_slope: float = 10.0,
    arm_names: Iterable[str] | None = None,
    ltw_minimum: float = 0.0,
    ltw_maximum: float = 1.0,
) -> SHDBenchmarkResult:
    if torch is None:
        raise ImportError("Phase 31 SHD benchmark requires PyTorch")
    arms = _select_arms(arm_names)
    _validate_config(config, surrogate_slope, ltw_minimum, ltw_maximum)
    resolved = resolve_device(device)
    train_events, train_labels, test_events, test_labels = load_shd_tensors(config)
    records: list[dict] = []
    for seed in config.seeds:
        for arm in arms:
            seed_everything(seed, device=resolved)
            if arm.model_kind.startswith("count_"):
                model = SHDEventCountClassifier(
                    config, kind=arm.model_kind.removeprefix("count_")
                ).to(resolved)
                initial_ltw = None
                initial_event_rate = 0.0
            else:
                model = SHDSparseClassifier(
                    config,
                    seed=seed,
                    delay_pattern=arm.delay_pattern,
                    max_delay_steps=arm.max_delay_steps,
                    surrogate_slope=surrogate_slope,
                    device=resolved,
                ).to(resolved)
                initial_ltw = model.graph.long_term_weight.detach().clone()
                _, _, initial_event_rate = _measure(
                    model,
                    test_events,
                    test_labels,
                    config.batch_size,
                    resolved,
                )
            train_seconds = _train_model(
                model,
                train_events,
                train_labels,
                config,
                seed=seed,
                device=resolved,
                ltw_minimum=ltw_minimum,
                ltw_maximum=ltw_maximum,
            )
            train_accuracy, _, _ = _measure(
                model, train_events, train_labels, config.batch_size, resolved
            )
            test_accuracy, inference_seconds, final_event_rate = _measure(
                model, test_events, test_labels, config.batch_size, resolved
            )
            active_edges = recurrent_edges = delayed_edges = 0
            mean_recurrent_delay = mean_ltw_change = 0.0
            lower_saturation = upper_saturation = 0.0
            if isinstance(model, SHDSparseClassifier):
                active = model.graph.active_mask
                recurrent = active & (model.graph.sources >= config.input_neurons)
                delays = model.graph.delay_steps[recurrent]
                active_edges = int(active.sum().item())
                recurrent_edges = int(recurrent.sum().item())
                delayed_edges = int((delays > 0).sum().item())
                mean_recurrent_delay = float(delays.to(torch.float32).mean().item())
                current_ltw = model.graph.long_term_weight.detach()
                mean_ltw_change = float(
                    (current_ltw[active] - initial_ltw[active]).abs().mean().item()
                )
                lower_saturation = float(
                    (current_ltw[active] <= ltw_minimum + 1e-6)
                    .to(torch.float32)
                    .mean()
                    .item()
                )
                upper_saturation = float(
                    (current_ltw[active] >= ltw_maximum - 1e-6)
                    .to(torch.float32)
                    .mean()
                    .item()
                )
            allocated_trainable_parameters = sum(
                parameter.numel() for parameter in model.parameters()
                if parameter.requires_grad
            )
            effective_trainable_parameters = allocated_trainable_parameters
            if isinstance(model, SHDSparseClassifier):
                effective_trainable_parameters = sum(
                    parameter.numel() for parameter in model.readout.parameters()
                ) + active_edges
            records.append(
                {
                    "seed": int(seed),
                    "arm": arm.name,
                    "model_kind": arm.model_kind,
                    "delay_pattern": arm.delay_pattern,
                    "train_samples": int(train_events.shape[0]),
                    "test_samples": int(test_events.shape[0]),
                    "train_accuracy": float(train_accuracy),
                    "test_accuracy": float(test_accuracy),
                    "active_edges": int(active_edges),
                    "recurrent_edges": int(recurrent_edges),
                    "delayed_edges": int(delayed_edges),
                    "mean_recurrent_delay": float(mean_recurrent_delay),
                    "effective_trainable_parameters": int(
                        effective_trainable_parameters
                    ),
                    "allocated_trainable_parameters": int(
                        allocated_trainable_parameters
                    ),
                    "initial_hidden_event_rate": float(initial_event_rate),
                    "final_hidden_event_rate": float(final_event_rate),
                    "event_rate_ratio": float(
                        final_event_rate / max(initial_event_rate, 1e-12)
                        if initial_event_rate > 0.0 else 0.0
                    ),
                    "mean_absolute_ltw_change": float(mean_ltw_change),
                    "lower_ltw_saturation_rate": float(lower_saturation),
                    "upper_ltw_saturation_rate": float(upper_saturation),
                    "train_seconds": float(train_seconds),
                    "inference_seconds": float(inference_seconds),
                    "test_examples_per_second": float(
                        test_events.shape[0] / max(inference_seconds, 1e-12)
                    ),
                }
            )
    _attach_delay_comparisons(records)
    return SHDBenchmarkResult(
        config=config,
        device=device_kind(resolved),
        surrogate_slope=float(surrogate_slope),
        arms=[asdict(arm) for arm in arms],
        records=records,
        summary=summarize_shd_benchmark(records, arms=arms),
    )


def summarize_shd_benchmark(
    records: Iterable[dict], *, arms: Iterable[SHDArm] = SHD_ARMS
) -> list[dict]:
    rows = list(records)
    summary: list[dict] = []
    for arm in arms:
        group = [row for row in rows if row["arm"] == arm.name]
        if not group:
            continue
        gains = [float(row["accuracy_gain_vs_no_delay"]) for row in group]
        paired_rates = [
            float(row["event_rate_vs_no_delay"])
            for row in group
            if float(row["event_rate_vs_no_delay"]) > 0.0
        ]
        summary.append(
            {
                "arm": arm.name,
                "model_kind": arm.model_kind,
                "delay_pattern": arm.delay_pattern,
                "seeds": len(group),
                "mean_test_accuracy": statistics.fmean(
                    float(row["test_accuracy"]) for row in group
                ),
                "std_test_accuracy": statistics.pstdev(
                    float(row["test_accuracy"]) for row in group
                ),
                "mean_accuracy_gain_vs_no_delay": statistics.fmean(gains),
                "improved_seed_count": sum(gain > 0.0 for gain in gains),
                "practical_gain_seed_count": sum(gain >= 0.01 for gain in gains),
                "active_edges": int(group[0]["active_edges"]),
                "mean_delayed_edges": statistics.fmean(
                    int(row["delayed_edges"]) for row in group
                ),
                "mean_recurrent_delay": statistics.fmean(
                    float(row["mean_recurrent_delay"]) for row in group
                ),
                "effective_trainable_parameters": int(
                    group[0]["effective_trainable_parameters"]
                ),
                "allocated_trainable_parameters": int(
                    group[0]["allocated_trainable_parameters"]
                ),
                "mean_event_rate_ratio": statistics.fmean(
                    float(row["event_rate_ratio"]) for row in group
                ),
                "mean_event_rate_vs_no_delay": (
                    statistics.fmean(paired_rates) if paired_rates else 0.0
                ),
                "mean_absolute_ltw_change": statistics.fmean(
                    float(row["mean_absolute_ltw_change"]) for row in group
                ),
                "mean_lower_ltw_saturation_rate": statistics.fmean(
                    float(row["lower_ltw_saturation_rate"]) for row in group
                ),
                "mean_upper_ltw_saturation_rate": statistics.fmean(
                    float(row["upper_ltw_saturation_rate"]) for row in group
                ),
                "mean_train_seconds": statistics.fmean(
                    float(row["train_seconds"]) for row in group
                ),
            }
        )
    return summary


def plot_shd_benchmark(summary: list[dict], path: str | pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    names = [row["arm"] for row in summary]
    accuracy = [100.0 * float(row["mean_test_accuracy"]) for row in summary]
    error = [100.0 * float(row["std_test_accuracy"]) for row in summary]
    gains = [100.0 * float(row["mean_accuracy_gain_vs_no_delay"]) for row in summary]
    parameters = [int(row["effective_trainable_parameters"]) for row in summary]
    figure, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)
    axes[0].bar(names, accuracy, yerr=error, color="#35b4f2", capsize=5)
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_title("AMMC Gen-5 Phase 31: SHD temporal transfer")
    axes[1].bar(names, gains, color="#ffb31a")
    axes[1].axhline(1.0, color="#bd3d3a", linestyle="--", label="+1 point gate")
    axes[1].set_ylabel("Gain vs paired no-delay (points)")
    axes[1].legend()
    axes[2].bar(names, parameters, color="#48c78e")
    axes[2].set_ylabel("Trainable parameters")
    axes[2].set_yscale("log")
    for axis in axes:
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.25)
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _load_or_build_split(
    h5_path: pathlib.Path,
    root: pathlib.Path,
    split: str,
    config: SHDConfig,
):
    duration_ms = round(config.duration_seconds * 1000)
    cache = root / (
        f"shd_{split}_t{config.timesteps}_c{config.input_neurons}_"
        f"d{duration_ms}ms.pt"
    )
    if cache.exists():
        payload = torch.load(cache, map_location="cpu", weights_only=True)
        return payload["events"], payload["labels"]
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Phase 31 preprocessing requires h5py") from exc
    print(f"Binning {h5_path.name} into {config.timesteps} temporal steps...")
    with h5py.File(h5_path, "r") as handle:
        labels = torch.as_tensor(handle["labels"][:], dtype=torch.long)
        events = torch.zeros(
            (labels.shape[0], config.timesteps, config.input_neurons),
            dtype=torch.uint8,
        )
        times = handle["spikes"]["times"]
        units = handle["spikes"]["units"]
        for index in range(labels.shape[0]):
            events[index] = bin_shd_events(
                times[index],
                units[index],
                timesteps=config.timesteps,
                input_neurons=config.input_neurons,
                duration_seconds=config.duration_seconds,
            )
            if (index + 1) % 1000 == 0:
                print(f"  {split}: {index + 1}/{labels.shape[0]}")
    temporary = cache.with_suffix(cache.suffix + ".part")
    torch.save({"events": events, "labels": labels}, temporary)
    temporary.replace(cache)
    return events, labels


def _train_model(
    model,
    events,
    labels,
    config: SHDConfig,
    *,
    seed: int,
    device,
    ltw_minimum: float,
    ltw_maximum: float,
) -> float:
    readout_parameters = list(model.parameters())
    parameter_groups = [{"params": readout_parameters, "lr": config.learning_rate}]
    if isinstance(model, SHDSparseClassifier):
        readout_parameters = list(model.readout.parameters())
        parameter_groups = [
            {"params": readout_parameters, "lr": config.learning_rate},
            {
                "params": [model.graph.long_term_weight],
                "lr": 0.0 if config.warmup_epochs > 0 else config.reservoir_learning_rate,
            },
        ]
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 50_000)
    model.train()
    sync(device)
    start = time.perf_counter()
    for epoch in range(config.epochs):
        if isinstance(model, SHDSparseClassifier) and epoch == config.warmup_epochs:
            optimizer.param_groups[1]["lr"] = config.reservoir_learning_rate
        order = torch.randperm(events.shape[0], generator=generator)
        for offset in range(0, events.shape[0], config.batch_size):
            index = order[offset : offset + config.batch_size]
            batch_events = events.index_select(0, index).to(device)
            batch_labels = labels.index_select(0, index).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_events)
            loss = torch.nn.functional.cross_entropy(logits, batch_labels)
            loss.backward()
            optimizer.step()
            if isinstance(model, SHDSparseClassifier):
                model.clamp_ltw(ltw_minimum, ltw_maximum)
            mark_step(device)
    sync(device)
    return time.perf_counter() - start


def _measure(model, events, labels, batch_size: int, device):
    model.eval()
    correct = 0
    total = 0
    weighted_event_rate = 0.0
    sync(device)
    start = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, events.shape[0], batch_size):
            batch_events = events[offset : offset + batch_size].to(device)
            batch_labels = labels[offset : offset + batch_size].to(device)
            logits, event_rate = model(batch_events, return_event_rate=True)
            correct += int((logits.argmax(dim=1) == batch_labels).sum().item())
            total += int(batch_labels.shape[0])
            weighted_event_rate += float(event_rate.item()) * batch_labels.shape[0]
            mark_step(device)
    sync(device)
    seconds = time.perf_counter() - start
    return correct / max(total, 1), seconds, weighted_event_rate / max(total, 1)


def _attach_delay_comparisons(records: list[dict]) -> None:
    controls = {
        int(row["seed"]): row
        for row in records
        if row["arm"] == "sparse_no_delay_warm_all"
    }
    for row in records:
        control = controls.get(int(row["seed"]))
        if control is None or row["model_kind"] != "sparse":
            row["accuracy_gain_vs_no_delay"] = 0.0
            row["event_rate_vs_no_delay"] = 0.0
            continue
        row["accuracy_gain_vs_no_delay"] = float(row["test_accuracy"]) - float(
            control["test_accuracy"]
        )
        denominator = max(float(control["final_hidden_event_rate"]), 1e-12)
        row["event_rate_vs_no_delay"] = float(row["final_hidden_event_rate"]) / denominator


def _select_arms(names: Iterable[str] | None) -> tuple[SHDArm, ...]:
    if names is None:
        return SHD_ARMS
    lookup = {arm.name: arm for arm in SHD_ARMS}
    selected = []
    for name in names:
        if name not in lookup:
            raise ValueError(f"unknown SHD arm: {name}")
        selected.append(lookup[name])
    if not selected:
        raise ValueError("at least one SHD arm is required")
    return tuple(selected)


def _validate_config(
    config: SHDConfig,
    surrogate_slope: float,
    ltw_minimum: float,
    ltw_maximum: float,
) -> None:
    if not config.seeds:
        raise ValueError("at least one seed is required")
    if config.input_neurons != 700 or config.classes != 20:
        raise ValueError("official SHD requires 700 inputs and 20 classes")
    if config.timesteps < 3 or config.duration_seconds <= 0:
        raise ValueError("SHD timing configuration is invalid")
    if config.epochs <= 0 or not 0 <= config.warmup_epochs <= config.epochs:
        raise ValueError("invalid epoch schedule")
    if config.batch_size <= 0 or config.train_samples < 0 or config.test_samples < 0:
        raise ValueError("invalid SHD sample or batch configuration")
    required_edges = (
        config.input_neurons * config.sensor_fanout
        + config.hidden_neurons * config.recurrent_fanout
    )
    if required_edges > config.max_edges:
        raise ValueError(f"SHD graph requires {required_edges} edge slots")
    if surrogate_slope <= 0 or ltw_minimum >= ltw_maximum:
        raise ValueError("invalid surrogate or LTW range")


def _download_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "AMMC-Gen5/1.0"})
    with urllib.request.urlopen(request) as response:
        return response.read()


def _download_to(url: str, destination: pathlib.Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "AMMC-Gen5/1.0"})
    with urllib.request.urlopen(request) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    temporary.replace(destination)


def _md5(path: pathlib.Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
