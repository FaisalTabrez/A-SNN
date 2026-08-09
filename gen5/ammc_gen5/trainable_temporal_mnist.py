"""Phase 21 fixed-topology LTW training for temporal event-coded MNIST."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import pathlib
import statistics
import time
from typing import Iterable

from .dynamic_sparse import DynamicSparseLinear
from .event_mnist import (
    EventMNISTConfig,
    _Classifier,
    _matched_raw_hidden_units,
    build_event_reservoir_edges,
    latency_encode,
    load_mnist_tensors,
    nn,
    torch,
)
from .runtime import device_kind, mark_step, resolve_device, seed_everything, sync

try:  # pragma: no cover - accelerator dependent
    from torch.autograd import Function
except Exception:  # pragma: no cover
    Function = object


TRAINABLE_TEMPORAL_GROUPS = (
    "raw_linear",
    "raw_mlp",
    "frozen_temporal_linear",
    "frozen_temporal_mlp",
    "trained_ltw_temporal_linear",
    "trained_ltw_temporal_mlp",
)


def _require_torch() -> None:
    if torch is None:
        raise ImportError("Phase 21 fixed-topology LTW training requires PyTorch")


class SurrogateSpike(Function):
    """Hard threshold forward pass with a fast-sigmoid surrogate gradient."""

    @staticmethod
    def forward(ctx, voltage, slope):  # type: ignore[override]
        ctx.save_for_backward(voltage)
        ctx.slope = float(slope)
        return (voltage >= 0).to(voltage.dtype)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        (voltage,) = ctx.saved_tensors
        denominator = (1.0 + ctx.slope * voltage.abs()).pow(2)
        return grad_output / denominator, None


class SparseTemporalClassifier(nn.Module):
    """Fixed sparse topology with optionally trainable LTWs."""

    def __init__(
        self,
        config: EventMNISTConfig,
        *,
        seed: int,
        classifier: str,
        hidden_units: int,
        train_ltw: bool,
        surrogate_slope: float,
        device,
    ) -> None:
        _require_torch()
        super().__init__()
        self.config = config
        self.train_ltw = bool(train_ltw)
        self.surrogate_slope = float(surrogate_slope)
        edges = build_event_reservoir_edges(
            config.sensor_neurons,
            config.hidden_neurons,
            sensor_fanout=config.sensor_fanout,
            recurrent_fanout=config.recurrent_fanout,
            seed=seed,
        )
        if len(edges) > config.max_edges:
            raise ValueError(f"reservoir requires {len(edges)} edges, capacity is {config.max_edges}")
        self.graph = DynamicSparseLinear(
            config.neuron_count,
            config.neuron_count,
            config.max_edges,
            device=device,
        )
        self.graph.load_edges(edges)
        self.graph.short_term_weight.requires_grad_(False)
        self.graph.long_term_weight.requires_grad_(self.train_ltw)
        feature_dim = config.timesteps * config.neuron_count
        self.readout = _Classifier(feature_dim, kind=classifier, hidden_units=hidden_units)

    @property
    def active_edge_count(self) -> int:
        return self.graph.active_edge_count

    def forward(self, pixels, *, return_event_rate: bool = False):  # type: ignore[override]
        events = latency_encode(pixels, self.config.timesteps, self.config.event_threshold)
        membrane = pixels.new_zeros((pixels.shape[0], self.config.neuron_count))
        spikes = torch.zeros_like(membrane)
        states = []
        hidden_events = pixels.new_zeros(())
        for step in range(self.config.timesteps):
            injected = torch.zeros_like(membrane)
            injected[:, : self.config.sensor_neurons] = events[step] * self.config.input_gain
            pre_reset = membrane * self.config.reservoir_leak + injected + self.graph(spikes)
            spikes = SurrogateSpike.apply(
                pre_reset - self.config.reservoir_threshold,
                self.surrogate_slope,
            )
            membrane = pre_reset - spikes * self.config.reservoir_threshold
            states.append(pre_reset)
            hidden_events = hidden_events + spikes[:, self.config.sensor_neurons :].mean()
        features = torch.stack(states, dim=1).reshape(pixels.shape[0], -1)
        logits = self.readout(features)
        if return_event_rate:
            return logits, hidden_events / self.config.timesteps
        return logits

    def clamp_ltw(self, minimum: float, maximum: float) -> None:
        with torch.no_grad():
            self.graph.long_term_weight.clamp_(minimum, maximum)
            self.graph.long_term_weight.mul_(self.graph.active_mask.to(self.graph.long_term_weight.dtype))
            self.graph.short_term_weight.zero_()


@dataclass
class TrainableTemporalMNISTResult:
    config: EventMNISTConfig
    device: str
    reservoir_learning_rate: float
    surrogate_slope: float
    records: list[dict]
    summary: list[dict]

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "trainable_temporal_mnist.json"
        records_path = output / "trainable_temporal_mnist_records.csv"
        summary_path = output / "trainable_temporal_mnist_summary.csv"
        payload = {
            "config": asdict(self.config),
            "device": self.device,
            "reservoir_learning_rate": self.reservoir_learning_rate,
            "surrogate_slope": self.surrogate_slope,
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
            plot_path = output / "trainable_temporal_mnist_summary.png"
            plot_trainable_temporal_mnist(self.summary, plot_path)
            paths["plot"] = str(plot_path)
        return paths


def run_trainable_temporal_mnist(
    config: EventMNISTConfig,
    *,
    device="auto",
    reservoir_learning_rate: float = 0.001,
    surrogate_slope: float = 10.0,
    ltw_minimum: float = 0.0,
    ltw_maximum: float = 1.0,
) -> TrainableTemporalMNISTResult:
    """Run raw, frozen, and LTW-trained fixed-topology comparisons."""

    _require_torch()
    _validate(config, reservoir_learning_rate, surrogate_slope, ltw_minimum, ltw_maximum)
    resolved = resolve_device(device)
    train_pixels, train_labels, test_pixels, test_labels = load_mnist_tensors(config)
    records: list[dict] = []

    for seed in config.seeds:
        for group_index, group in enumerate(TRAINABLE_TEMPORAL_GROUPS):
            seed_everything(seed + group_index * 10_000, device=resolved)
            model, train_ltw = _build_model(
                group,
                config,
                seed=seed,
                surrogate_slope=surrogate_slope,
                device=resolved,
            )
            model = model.to(resolved)
            initial_ltw = None
            if isinstance(model, SparseTemporalClassifier):
                initial_ltw = model.graph.long_term_weight.detach().clone()
            train_seconds = _train_model(
                model,
                train_pixels,
                train_labels,
                config,
                seed=seed,
                device=resolved,
                train_ltw=train_ltw,
                reservoir_learning_rate=reservoir_learning_rate,
                ltw_minimum=ltw_minimum,
                ltw_maximum=ltw_maximum,
            )
            train_accuracy, _, _ = _measure(model, train_pixels, train_labels, config.batch_size, resolved)
            test_accuracy, inference_seconds, hidden_event_rate = _measure(
                model, test_pixels, test_labels, config.batch_size, resolved
            )
            active_edges = model.active_edge_count if isinstance(model, SparseTemporalClassifier) else 0
            ltw_change = 0.0
            mean_ltw = 0.0
            if isinstance(model, SparseTemporalClassifier):
                active = model.graph.active_mask
                current = model.graph.long_term_weight.detach()
                mean_ltw = float(current[active].mean().item())
                if initial_ltw is not None:
                    ltw_change = float((current[active] - initial_ltw[active]).abs().mean().item())
            trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            readout_parameters = _readout_parameter_count(model)
            effective_trainable = readout_parameters + (active_edges if train_ltw else 0)
            records.append(
                {
                    "seed": int(seed),
                    "group": group,
                    "train_samples": int(train_pixels.shape[0]),
                    "test_samples": int(test_pixels.shape[0]),
                    "train_accuracy": float(train_accuracy),
                    "test_accuracy": float(test_accuracy),
                    "active_edges": int(active_edges),
                    "readout_parameters": int(readout_parameters),
                    "optimizer_parameters": int(trainable_parameters),
                    "effective_trainable_parameters": int(effective_trainable),
                    "mean_hidden_event_rate": float(hidden_event_rate),
                    "mean_ltw": mean_ltw,
                    "mean_absolute_ltw_change": ltw_change,
                    "train_seconds": float(train_seconds),
                    "inference_seconds": float(inference_seconds),
                    "end_to_end_examples_per_second": float(test_pixels.shape[0] / max(inference_seconds, 1e-12)),
                }
            )

    return TrainableTemporalMNISTResult(
        config=config,
        device=device_kind(resolved),
        reservoir_learning_rate=float(reservoir_learning_rate),
        surrogate_slope=float(surrogate_slope),
        records=records,
        summary=summarize_trainable_temporal_mnist(records),
    )


def summarize_trainable_temporal_mnist(records: Iterable[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in records:
        grouped.setdefault(str(row["group"]), []).append(row)
    summary: list[dict] = []
    for group in TRAINABLE_TEMPORAL_GROUPS:
        rows = grouped.get(group, [])
        if not rows:
            continue
        accuracy = [float(row["test_accuracy"]) for row in rows]
        summary.append(
            {
                "group": group,
                "seeds": len(rows),
                "mean_test_accuracy": statistics.fmean(accuracy),
                "std_test_accuracy": statistics.pstdev(accuracy),
                "mean_train_accuracy": statistics.fmean(float(row["train_accuracy"]) for row in rows),
                "active_edges": int(rows[0]["active_edges"]),
                "readout_parameters": int(rows[0]["readout_parameters"]),
                "optimizer_parameters": int(rows[0]["optimizer_parameters"]),
                "effective_trainable_parameters": int(rows[0]["effective_trainable_parameters"]),
                "mean_hidden_event_rate": statistics.fmean(float(row["mean_hidden_event_rate"]) for row in rows),
                "mean_ltw": statistics.fmean(float(row["mean_ltw"]) for row in rows),
                "mean_absolute_ltw_change": statistics.fmean(
                    float(row["mean_absolute_ltw_change"]) for row in rows
                ),
                "mean_train_seconds": statistics.fmean(float(row["train_seconds"]) for row in rows),
                "mean_end_to_end_examples_per_second": statistics.fmean(
                    float(row["end_to_end_examples_per_second"]) for row in rows
                ),
            }
        )
    return summary


def plot_trainable_temporal_mnist(summary: list[dict], path: str | pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    labels = [row["group"].replace("_", "\n") for row in summary]
    accuracy = [100.0 * float(row["mean_test_accuracy"]) for row in summary]
    errors = [100.0 * float(row["std_test_accuracy"]) for row in summary]
    figure, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
    axes[0].bar(labels, accuracy, yerr=errors, capsize=4, color="#35b4f2")
    axes[0].set_ylabel("Engineering-validation accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_title("AMMC Gen-5 Phase 21: Fixed-Topology LTW Training")
    axes[1].bar(labels, [float(row["mean_absolute_ltw_change"]) for row in summary], color="#ffb31a")
    axes[1].set_ylabel("Mean absolute LTW change")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _build_model(group, config, *, seed, surrogate_slope, device):
    if group == "raw_linear":
        return _Classifier(config.sensor_neurons, kind="linear", hidden_units=1), False
    if group == "raw_mlp":
        width = _matched_raw_hidden_units(
            config.sensor_neurons, config.neuron_count * 2, config.readout_hidden_units
        )
        return _Classifier(config.sensor_neurons, kind="mlp", hidden_units=width), False
    classifier = "mlp" if group.endswith("mlp") else "linear"
    feature_dim = config.timesteps * config.neuron_count
    width = _matched_raw_hidden_units(feature_dim, config.neuron_count * 2, config.readout_hidden_units)
    train_ltw = group.startswith("trained_ltw_")
    return (
        SparseTemporalClassifier(
            config,
            seed=seed,
            classifier=classifier,
            hidden_units=width,
            train_ltw=train_ltw,
            surrogate_slope=surrogate_slope,
            device=device,
        ),
        train_ltw,
    )


def _train_model(
    model,
    pixels,
    labels,
    config,
    *,
    seed,
    device,
    train_ltw,
    reservoir_learning_rate,
    ltw_minimum,
    ltw_maximum,
):
    if train_ltw:
        optimizer = torch.optim.AdamW(
            [
                {"params": list(model.readout.parameters()), "lr": config.learning_rate, "weight_decay": config.weight_decay},
                {"params": [model.graph.long_term_weight], "lr": reservoir_learning_rate, "weight_decay": 0.0},
            ]
        )
    else:
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    criterion = nn.CrossEntropyLoss()
    model.train()
    start_time = time.perf_counter()
    for epoch in range(config.epochs):
        generator = torch.Generator().manual_seed(seed * 1000 + epoch)
        order = torch.randperm(pixels.shape[0], generator=generator)
        for start in range(0, order.numel(), config.batch_size):
            index = order[start : start + config.batch_size]
            batch = pixels.index_select(0, index).to(device)
            target = labels.index_select(0, index).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), target)
            loss.backward()
            optimizer.step()
            if train_ltw:
                model.clamp_ltw(ltw_minimum, ltw_maximum)
            mark_step(device)
    sync(device)
    return time.perf_counter() - start_time


def _measure(model, pixels, labels, batch_size, device):
    model.eval()
    correct = 0
    event_rate_sum = 0.0
    batches = 0
    start_time = time.perf_counter()
    with torch.no_grad():
        for start in range(0, pixels.shape[0], batch_size):
            batch = pixels[start : start + batch_size].to(device)
            if isinstance(model, SparseTemporalClassifier):
                logits, event_rate = model(batch, return_event_rate=True)
                event_rate_sum += float(event_rate.item())
            else:
                logits = model(batch)
            prediction = logits.argmax(dim=1).cpu()
            correct += int((prediction == labels[start : start + batch_size]).sum().item())
            batches += 1
            mark_step(device)
    sync(device)
    seconds = time.perf_counter() - start_time
    return correct / pixels.shape[0], seconds, event_rate_sum / max(batches, 1)


def _readout_parameter_count(model) -> int:
    parameters = model.readout.parameters() if isinstance(model, SparseTemporalClassifier) else model.parameters()
    return sum(parameter.numel() for parameter in parameters)


def _validate(config, reservoir_learning_rate, surrogate_slope, ltw_minimum, ltw_maximum):
    if not config.seeds or config.train_samples <= 0 or config.test_samples <= 0:
        raise ValueError("seeds and positive sample counts are required")
    if reservoir_learning_rate <= 0 or surrogate_slope <= 0:
        raise ValueError("reservoir learning rate and surrogate slope must be positive")
    if ltw_minimum < 0 or ltw_maximum <= ltw_minimum:
        raise ValueError("LTW bounds must satisfy 0 <= minimum < maximum")
    required = config.sensor_neurons * config.sensor_fanout + config.hidden_neurons * config.recurrent_fanout
    if required > config.max_edges:
        raise ValueError(f"topology requires {required} edges but max_edges is {config.max_edges}")


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
