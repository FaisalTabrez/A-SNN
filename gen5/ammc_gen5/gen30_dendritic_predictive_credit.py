"""Gen-30 fixed-topology dendritic predictive-credit causal microtask."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import pathlib
import random
import statistics
import time
import zipfile

try:  # pragma: no cover - exercised in PyTorch runtimes
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


GEN30_ARMS = (
    "static",
    "bptt",
    "eprop_broadcast",
    "dendritic_predictive_credit",
    "dpc_shuffled_apical",
    "dpc_no_eligibility",
    "dpc_shuffled_modulator",
)


@dataclass(frozen=True)
class Gen30Config:
    seeds: tuple[int, ...] = tuple(range(42, 52))
    cue_classes: int = 4
    input_channels: int = 11
    hidden_neurons: int = 64
    timesteps: int = 24
    context_time: int = 4
    query_time: int = 23
    distractor_start: int = 7
    distractor_end: int = 17
    distractor_probability: float = 0.08
    train_samples_per_context: int = 2048
    test_samples_per_context: int = 1024
    batch_size: int = 256
    phase_a_epochs: int = 10
    phase_b_epochs: int = 10
    membrane_decay: float = 0.90
    trace_decay: float = 0.95
    threshold: float = 0.75
    surrogate_slope: float = 8.0
    bptt_learning_rate: float = 0.003
    local_learning_rate: float = 0.03
    predictor_learning_rate: float = 0.01
    update_clip: float = 0.05
    weight_clip: float = 2.0
    minimum_b_accuracy: float = 0.80
    minimum_a_retention: float = 0.75
    maximum_a_retention_drop: float = 0.05
    maximum_gap_vs_eprop: float = 0.05
    minimum_causal_margin: float = 0.10
    minimum_spike_activity: float = 0.01
    maximum_spike_activity: float = 0.30
    minimum_qualified_seeds: int = 8


@dataclass
class Gen30Result:
    config: dict
    device: str
    task: dict
    records: list[dict]
    summary: list[dict]
    decision: dict

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / "gen30_dendritic_predictive_credit.json"
        records_path = output / "gen30_dendritic_predictive_credit_records.csv"
        summary_path = output / "gen30_dendritic_predictive_credit_summary.csv"
        json_path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        _write_csv(records_path, self.records)
        _write_csv(summary_path, self.summary)
        paths = {"json": str(json_path), "records_csv": str(records_path), "summary_csv": str(summary_path)}
        if plot:
            plot_path = output / "gen30_dendritic_predictive_credit.png"
            plot_gen30(self, plot_path)
            paths["plot"] = str(plot_path)
        return paths


if torch is not None:  # pragma: no branch
    class _SurrogateSpike(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, slope):
            ctx.save_for_backward(value)
            ctx.slope = slope
            return (value >= 0).to(value.dtype)

        @staticmethod
        def backward(ctx, gradient):
            (value,) = ctx.saved_tensors
            scale = torch.sigmoid(ctx.slope * value)
            return gradient * ctx.slope * scale * (1.0 - scale), None


_ModuleBase = nn.Module if nn is not None else object


class DendriticCreditNetwork(_ModuleBase):
    """Fixed decoder with trainable basal input and recurrent synapses."""

    def __init__(self, config: Gen30Config):
        if torch is None:
            raise ImportError("Gen-30 requires PyTorch")
        super().__init__()
        hidden, inputs, classes = config.hidden_neurons, config.input_channels, config.cue_classes
        self.config = config
        self.w_in = nn.Parameter(torch.randn(hidden, inputs) / math.sqrt(inputs))
        self.w_rec = nn.Parameter(torch.randn(hidden, hidden) * (0.35 / math.sqrt(hidden)))
        self.predictor_gain = nn.Parameter(torch.zeros(hidden), requires_grad=False)
        decoder = torch.randn(classes, hidden) / math.sqrt(hidden)
        decoder = torch.nn.functional.normalize(decoder, dim=1)
        # Fixed symmetric feedback isolates the local update rule. An unrelated
        # random feedback matrix would turn this experiment into a feedback-
        # alignment test and confound a negative credit-assignment result.
        feedback = decoder.clone()
        self.register_buffer("decoder", decoder)
        self.register_buffer("feedback", feedback)

    def forward(self, events, *, return_activity: bool = False):
        batch = int(events.shape[0])
        membrane = events.new_zeros((batch, self.config.hidden_neurons))
        spikes = torch.zeros_like(membrane)
        accumulated = torch.zeros_like(membrane)
        activity = events.new_zeros(())
        for step in range(self.config.timesteps):
            current = torch.nn.functional.linear(events[:, step], self.w_in)
            current = current + torch.nn.functional.linear(spikes, self.w_rec)
            pre_reset = self.config.membrane_decay * membrane + current
            spikes = _SurrogateSpike.apply(pre_reset - self.config.threshold, self.config.surrogate_slope)
            membrane = pre_reset - spikes * self.config.threshold
            accumulated = accumulated + spikes
            activity = activity + spikes.mean()
        logits = torch.nn.functional.linear(accumulated / self.config.timesteps, self.decoder)
        if return_activity:
            return logits, activity / self.config.timesteps
        return logits


def available_gen30_arms() -> tuple[str, ...]:
    return GEN30_ARMS


def generate_contextual_binding(
    config: Gen30Config, *, samples: int, context: int, seed: int
):
    """Generate paired delayed contextual-binding examples on CPU."""
    if torch is None:
        raise ImportError("Gen-30 task generation requires PyTorch")
    if context not in (0, 1):
        raise ValueError("context must be 0 or 1")
    generator = torch.Generator().manual_seed(int(seed))
    cues = torch.randint(config.cue_classes, (samples,), generator=generator)
    events = torch.zeros(samples, config.timesteps, config.input_channels)
    rows = torch.arange(samples)
    events[rows, 0, cues] = 1.0
    events[:, config.context_time, config.cue_classes + context] = 1.0
    distractor_channels = config.input_channels - config.cue_classes - 3
    distractor_shape = (samples, config.distractor_end - config.distractor_start, distractor_channels)
    distractors = (
        torch.rand(distractor_shape, generator=generator) < config.distractor_probability
    ).to(events.dtype)
    events[
        :, config.distractor_start:config.distractor_end, config.cue_classes + 2:-1
    ] = distractors
    events[:, config.query_time, -1] = 1.0
    permutation = torch.tensor((1, 0, 3, 2), dtype=torch.long)
    labels = cues.clone() if context == 0 else permutation[cues]
    return events, labels


def run_gen30(
    config: Gen30Config = Gen30Config(), *, device: str = "auto",
    progress_path: str | pathlib.Path | None = None,
) -> Gen30Result:
    _validate_config(config)
    if torch is None:
        raise ImportError("Gen-30 requires PyTorch")
    resolved = _resolve_device(device)
    signature = hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()
    progress = _load_progress(progress_path, signature)
    records = list(progress.get("records", []))
    completed = {(int(row["seed"]), row["arm"]) for row in records}
    datasets = _build_datasets(config)
    for seed in config.seeds:
        _seed_everything(seed)
        initial = DendriticCreditNetwork(config).state_dict()
        for arm_index, arm in enumerate(GEN30_ARMS):
            if (seed, arm) in completed:
                continue
            _seed_everything(seed * 1000 + arm_index)
            model = DendriticCreditNetwork(config).to(resolved)
            model.load_state_dict(initial)
            started = time.perf_counter()
            if arm != "static":
                _train_stage(
                    model, datasets["train_a"], arm, config, resolved,
                    seed * 101 + 1, config.phase_a_epochs,
                )
            after_a_a, activity_a = _evaluate(model, datasets["test_a"], config, resolved)
            after_a_b, _ = _evaluate(model, datasets["test_b"], config, resolved)
            if arm != "static":
                _train_stage(
                    model, datasets["train_b"], arm, config, resolved,
                    seed * 101 + 2, config.phase_b_epochs,
                )
            after_b_a, activity_b = _evaluate(model, datasets["test_a"], config, resolved)
            after_b_b, _ = _evaluate(model, datasets["test_b"], config, resolved)
            records.append({
                "seed": int(seed),
                "arm": arm,
                "after_a_a_accuracy": after_a_a,
                "after_a_b_accuracy": after_a_b,
                "after_b_a_accuracy": after_b_a,
                "after_b_b_accuracy": after_b_b,
                "a_retention_drop": after_a_a - after_b_a,
                "joint_after_b_accuracy": 0.5 * (after_b_a + after_b_b),
                "mean_spike_activity": 0.5 * (activity_a + activity_b),
                "trainable_synapses": int(model.w_in.numel() + model.w_rec.numel()),
                "seconds": time.perf_counter() - started,
            })
            _save_progress(progress_path, signature, records)
    summary = summarize_gen30(records)
    decision = decide_gen30(records, summary, config)
    _save_progress(progress_path, signature, records, decision=decision, complete=True)
    return Gen30Result(
        config=asdict(config),
        device=str(resolved),
        task={
            "name": "Delayed Contextual Binding",
            "context_a_mapping": [0, 1, 2, 3],
            "context_b_mapping": [1, 0, 3, 2],
            "structural_plasticity_enabled": False,
            "global_bptt_used_by_dpc": False,
        },
        records=records,
        summary=summary,
        decision=decision,
    )


def _build_datasets(config):
    return {
        "train_a": generate_contextual_binding(config, samples=config.train_samples_per_context, context=0, seed=3001),
        "train_b": generate_contextual_binding(config, samples=config.train_samples_per_context, context=1, seed=3002),
        "test_a": generate_contextual_binding(config, samples=config.test_samples_per_context, context=0, seed=4001),
        "test_b": generate_contextual_binding(config, samples=config.test_samples_per_context, context=1, seed=4002),
    }


def _train_stage(model, dataset, arm, config, device, order_seed, epochs):
    events, labels = dataset
    model.train()
    if arm == "bptt":
        optimizer = torch.optim.Adam((model.w_in, model.w_rec), lr=config.bptt_learning_rate)
    else:
        optimizer = None
    for epoch in range(epochs):
        generator = torch.Generator().manual_seed(order_seed + epoch)
        order = torch.randperm(len(events), generator=generator)
        for start in range(0, len(events), config.batch_size):
            index = order[start:start + config.batch_size]
            batch = events[index].to(device)
            target = labels[index].to(device)
            if arm == "bptt":
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(batch), target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_((model.w_in, model.w_rec), 1.0)
                optimizer.step()
            else:
                _manual_local_step(model, batch, target, arm, config)


def _manual_local_step(model, events, labels, arm, config):
    batch = int(events.shape[0])
    hidden = config.hidden_neurons
    with torch.no_grad():
        membrane = events.new_zeros((batch, hidden))
        spikes = torch.zeros_like(membrane)
        accumulated = torch.zeros_like(membrane)
        membrane_accumulated = torch.zeros_like(membrane)
        input_trace = events.new_zeros((batch, config.input_channels))
        recurrent_trace = events.new_zeros((batch, hidden))
        eligibility_in = events.new_zeros((batch, hidden, config.input_channels))
        eligibility_rec = events.new_zeros((batch, hidden, hidden))
        for step in range(config.timesteps):
            previous_spikes = spikes
            input_trace = config.trace_decay * input_trace + events[:, step]
            recurrent_trace = config.trace_decay * recurrent_trace + previous_spikes
            current = torch.nn.functional.linear(events[:, step], model.w_in)
            current = current + torch.nn.functional.linear(previous_spikes, model.w_rec)
            pre_reset = config.membrane_decay * membrane + current
            spikes = (pre_reset >= config.threshold).to(events.dtype)
            sensitivity_probability = torch.sigmoid(
                config.surrogate_slope * (pre_reset - config.threshold)
            )
            sensitivity = (
                config.surrogate_slope
                * sensitivity_probability
                * (1.0 - sensitivity_probability)
            )
            coincidence_in = sensitivity.unsqueeze(2) * input_trace.unsqueeze(1)
            coincidence_rec = sensitivity.unsqueeze(2) * recurrent_trace.unsqueeze(1)
            if arm == "dpc_no_eligibility":
                eligibility_in = coincidence_in
                eligibility_rec = coincidence_rec
            else:
                eligibility_in = config.trace_decay * eligibility_in + coincidence_in
                eligibility_rec = config.trace_decay * eligibility_rec + coincidence_rec
            membrane = pre_reset - spikes * config.threshold
            accumulated = accumulated + spikes
            membrane_accumulated = membrane_accumulated + pre_reset
        hidden_mean = accumulated / config.timesteps
        logits = torch.nn.functional.linear(hidden_mean, model.decoder)
        probabilities = torch.softmax(logits, dim=1)
        targets = torch.nn.functional.one_hot(labels, config.cue_classes).to(probabilities.dtype)
        output_error = targets - probabilities
        if arm == "dpc_shuffled_modulator":
            permutation = torch.randperm(batch, device=events.device)
            output_error = output_error[permutation]
        apical_target = output_error @ model.feedback
        if arm == "dpc_shuffled_apical":
            neuron_permutation = torch.randperm(hidden, device=events.device)
            apical_target = apical_target[:, neuron_permutation]
        if arm == "eprop_broadcast":
            local_error = apical_target
        else:
            local_state = torch.tanh(membrane_accumulated / config.timesteps)
            local_prediction = local_state * model.predictor_gain
            local_error = apical_target - local_prediction
            predictor_update = (local_error * local_state).mean(dim=0)
            model.predictor_gain.add_(config.predictor_learning_rate * predictor_update)
            model.predictor_gain.clamp_(-2.0, 2.0)
        update_in = (local_error.unsqueeze(2) * eligibility_in).mean(dim=0)
        update_rec = (local_error.unsqueeze(2) * eligibility_rec).mean(dim=0)
        update_in = update_in.clamp(-config.update_clip, config.update_clip)
        update_rec = update_rec.clamp(-config.update_clip, config.update_clip)
        model.w_in.add_(config.local_learning_rate * update_in)
        model.w_rec.add_(config.local_learning_rate * update_rec)
        model.w_in.clamp_(-config.weight_clip, config.weight_clip)
        model.w_rec.clamp_(-config.weight_clip, config.weight_clip)


def _evaluate(model, dataset, config, device):
    events, labels = dataset
    correct = 0
    activity_total = 0.0
    batches = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(events), config.batch_size):
            batch = events[start:start + config.batch_size].to(device)
            target = labels[start:start + config.batch_size].to(device)
            logits, activity = model(batch, return_activity=True)
            correct += int((logits.argmax(1) == target).sum().item())
            activity_total += float(activity.item())
            batches += 1
    return correct / len(events), activity_total / max(batches, 1)


def summarize_gen30(records):
    summary = []
    for arm in GEN30_ARMS:
        group = [row for row in records if row["arm"] == arm]
        if not group:
            continue
        summary.append({
            "arm": arm,
            "seeds": len(group),
            "mean_after_a_a_accuracy": statistics.fmean(row["after_a_a_accuracy"] for row in group),
            "mean_after_b_a_accuracy": statistics.fmean(row["after_b_a_accuracy"] for row in group),
            "mean_after_b_b_accuracy": statistics.fmean(row["after_b_b_accuracy"] for row in group),
            "mean_a_retention_drop": statistics.fmean(row["a_retention_drop"] for row in group),
            "mean_joint_after_b_accuracy": statistics.fmean(row["joint_after_b_accuracy"] for row in group),
            "std_joint_after_b_accuracy": statistics.pstdev(row["joint_after_b_accuracy"] for row in group),
            "mean_spike_activity": statistics.fmean(row["mean_spike_activity"] for row in group),
            "mean_seconds": statistics.fmean(row["seconds"] for row in group),
            "trainable_synapses": int(group[0]["trainable_synapses"]),
        })
    return summary


def decide_gen30(records, summary, config):
    by_arm = {row["arm"]: row for row in summary}
    required = set(GEN30_ARMS)
    if not required <= set(by_arm):
        return {"status": "stop", "reason": "incomplete arms", "next_milestone": "complete_gen30"}
    dpc = by_arm["dendritic_predictive_credit"]
    eprop = by_arm["eprop_broadcast"]
    controls = (
        by_arm["dpc_shuffled_apical"],
        by_arm["dpc_no_eligibility"],
        by_arm["dpc_shuffled_modulator"],
    )
    margins = [dpc["mean_joint_after_b_accuracy"] - row["mean_joint_after_b_accuracy"] for row in controls]
    dpc_seed_rows = [row for row in records if row["arm"] == "dendritic_predictive_credit"]
    qualified_seeds = sum(
        row["after_b_b_accuracy"] >= config.minimum_b_accuracy
        and row["after_b_a_accuracy"] >= config.minimum_a_retention
        and row["a_retention_drop"] <= config.maximum_a_retention_drop
        for row in dpc_seed_rows
    )
    gates = {
        "b_accuracy": dpc["mean_after_b_b_accuracy"] >= config.minimum_b_accuracy,
        "a_retention": dpc["mean_after_b_a_accuracy"] >= config.minimum_a_retention,
        "retention_drop": dpc["mean_a_retention_drop"] <= config.maximum_a_retention_drop,
        "eprop_parity": dpc["mean_joint_after_b_accuracy"] >= eprop["mean_joint_after_b_accuracy"] - config.maximum_gap_vs_eprop,
        "causal_controls": min(margins) >= config.minimum_causal_margin,
        "activity": config.minimum_spike_activity <= dpc["mean_spike_activity"] <= config.maximum_spike_activity,
        "seed_replication": qualified_seeds >= config.minimum_qualified_seeds,
    }
    passed = all(gates.values())
    return {
        "status": "pass" if passed else "stop",
        "gates": gates,
        "qualified_seed_count": int(qualified_seeds),
        "causal_margins": {
            "vs_shuffled_apical": margins[0],
            "vs_no_eligibility": margins[1],
            "vs_shuffled_modulator": margins[2],
        },
        "structural_plasticity_claim_authorized": False,
        "hardware_energy_claim_authorized": False,
        "next_milestone": "fixed_topology_ssc_transfer" if passed else "redesign_local_credit_before_real_data",
    }


def plot_gen30(result: Gen30Result, path: str | pathlib.Path):
    import matplotlib.pyplot as plt
    labels = [row["arm"] for row in result.summary]
    joint = [row["mean_joint_after_b_accuracy"] for row in result.summary]
    retained = [row["mean_after_b_a_accuracy"] for row in result.summary]
    learned = [row["mean_after_b_b_accuracy"] for row in result.summary]
    x = range(len(labels))
    figure, axis = plt.subplots(figsize=(13, 6), constrained_layout=True)
    width = 0.26
    axis.bar([value - width for value in x], retained, width, label="Context A retained")
    axis.bar(list(x), learned, width, label="Context B learned")
    axis.bar([value + width for value in x], joint, width, label="Joint")
    axis.axhline(0.80, color="black", linestyle="--", linewidth=1)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Accuracy")
    axis.set_title("Gen-30 Delayed Contextual Binding")
    axis.set_xticks(list(x), labels, rotation=25, ha="right")
    axis.legend()
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def bundle_gen30_artifacts(paths, output_dir):
    output = pathlib.Path(output_dir)
    files = [pathlib.Path(value) for value in paths.values() if pathlib.Path(value).is_file()]
    manifest = output / "gen30_dendritic_predictive_credit_manifest.json"
    manifest.write_text(json.dumps({
        "files": [{"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in files]
    }, indent=2) + "\n", encoding="utf-8")
    archive = output / "gen30_dendritic_predictive_credit_bundle.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in files + [manifest]:
            bundle.write(path, arcname=path.name)
    return {"manifest": str(manifest), "bundle": str(archive)}


def _resolve_device(device):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return resolved


def _seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_progress(path, signature):
    if path is None or not pathlib.Path(path).is_file():
        return {}
    payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if payload.get("signature") != signature:
        raise ValueError("Gen-30 progress signature does not match the frozen configuration")
    return payload


def _save_progress(path, signature, records, **extra):
    if path is None:
        return
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"signature": signature, "records": records, **extra}
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def _validate_config(config):
    if config.cue_classes != 4 or config.input_channels != 11:
        raise ValueError("Gen-30 freezes four cues and eleven input channels")
    if config.context_time <= 0 or config.query_time >= config.timesteps:
        raise ValueError("invalid task timing")
    if config.distractor_end > config.query_time:
        raise ValueError("distractors must end before the query")
    if len(config.seeds) != 10:
        raise ValueError("Gen-30 requires ten confirmation seeds")
    if config.minimum_qualified_seeds != 8:
        raise ValueError("Gen-30 freezes an 8/10 seed gate")


def _write_csv(path, rows):
    if not rows:
        pathlib.Path(path).write_text("", encoding="utf-8")
        return
    with pathlib.Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
