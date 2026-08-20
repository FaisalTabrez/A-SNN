"""Gen-27 trained residual-LIF robustness to sparse accumulation differences."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import pathlib
import statistics
import time
import zipfile

from .event_mnist import torch
from .gen25_event_driven_sparse_audit import ResidualLIFStateHead, sparse_temporal_currents
from .runtime import device_kind, mark_step, resolve_device, seed_everything
from .shd_benchmark import SHDConfig
from .shd_state_placement_diagnostic import (
    ResidualTemporalConvStateClassifier,
    _multiscale_features,
    matched_temporal_conv_residual_channels,
)
from .shd_validation_checkpoint import _train_validation_selected
from .ssc_benchmark import load_ssc_tensors


GEN27_ARMS = ("dense_reference", "sparse_operator", "shuffled_error_control")


@dataclass(frozen=True)
class Gen27Config:
    seeds: tuple[int, ...] = (651, 652, 653)
    input_neurons: int = 700
    classes: int = 35
    timesteps: int = 64
    duration_seconds: float = 1.0
    data_root: str = "gen5_data/ssc"
    download: bool = True
    source_train_samples: int = 20_000
    validation_samples: int = 2_999
    test_samples: int = 8_000
    epochs: int = 12
    learning_rate: float = 0.003
    weight_decay: float = 0.0001
    batch_size: int = 256
    target_parameters: int = 133_631
    temporal_levels: tuple[int, ...] = (1, 2, 4, 8)
    kernel_size: int = 5
    surrogate_slope: float = 10.0
    near_threshold_band: float = 1e-3
    maximum_accuracy_change: float = 0.001
    minimum_prediction_agreement: float = 0.999
    maximum_spike_disagreement: float = 1e-4


@dataclass
class Gen27Result:
    config: dict
    device: str
    dataset: dict
    architecture: dict
    records: list[dict]
    summary: list[dict]
    decision: dict

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output = pathlib.Path(output_dir); output.mkdir(parents=True, exist_ok=True)
        json_path = output / "gen27_trained_threshold_robustness.json"
        records_path = output / "gen27_trained_threshold_robustness_records.csv"
        summary_path = output / "gen27_trained_threshold_robustness_summary.csv"
        json_path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        _write_csv(records_path, self.records); _write_csv(summary_path, self.summary)
        paths = {"json": str(json_path), "records_csv": str(records_path), "summary_csv": str(summary_path)}
        if plot:
            plot_path = output / "gen27_trained_threshold_robustness.png"
            plot_gen27(self, plot_path); paths["plot"] = str(plot_path)
        return paths


def run_gen27(config: Gen27Config = Gen27Config(), *, device="auto", progress_path=None, dataset=None) -> Gen27Result:
    _validate_config(config)
    resolved = resolve_device(device)
    data = dataset if dataset is not None else _load_data(config)
    channels, parameters = matched_temporal_conv_residual_channels(
        config.input_neurons, config.classes, config.target_parameters,
        kernel_size=config.kernel_size, temporal_levels=config.temporal_levels,
    )
    records = _load_progress(progress_path, config)
    completed = {int(row["seed"]) for row in records if row["arm"] == "sparse_operator"}
    for seed in config.seeds:
        if seed in completed:
            continue
        seed_everything(seed, device=resolved)
        model_config = _model_config(config, seed)
        model = ResidualTemporalConvStateClassifier(
            model_config, channels=channels, kernel_size=config.kernel_size,
            temporal_levels=config.temporal_levels, dynamics="lif",
            surrogate_slope=config.surrogate_slope,
        ).to(resolved)
        training = _train_validation_selected(
            model, data[0], data[1], data[2], data[3], model_config,
            seed=seed, device=resolved,
        )
        model.load_state_dict(training["best_state"]); model.eval()
        seed_records = _evaluate_trained_model(
            model, data[4], data[5], config, seed, training, resolved
        )
        records.extend(seed_records); _save_progress(progress_path, config, records)
    summary = summarize_gen27(records)
    return Gen27Result(
        config=asdict(config), device=device_kind(resolved),
        dataset={
            "name": "Spiking Speech Commands", "train_samples": int(data[0].shape[0]),
            "validation_samples": int(data[2].shape[0]), "test_samples": int(data[4].shape[0]),
        },
        architecture={
            "model": "validation-selected Phase-49 residual LIF", "channels": channels,
            "trainable_parameters": parameters,
            "operator_change_only": True,
        },
        records=records, summary=summary, decision=decide_gen27(records, config),
    )


@torch.inference_mode()
def _evaluate_trained_model(model, events, labels, config, seed, training, device):
    head = ResidualLIFStateHead(model).to(device).eval()
    totals = {
        arm: {"correct": 0, "agreement": 0, "spike_diff": 0.0, "spikes": 0,
              "logit_max": 0.0, "current_max": 0.0, "near": 0.0, "units": 0}
        for arm in GEN27_ARMS
    }
    started = time.perf_counter()
    for offset in range(0, events.shape[0], config.batch_size):
        x = events[offset : offset + config.batch_size].to(device).to(torch.float32)
        y = labels[offset : offset + config.batch_size].to(device)
        dense_currents = model.temporal(x.transpose(1, 2)).transpose(1, 2)
        sparse_currents = sparse_temporal_currents(x, model.temporal)
        error = sparse_currents - dense_currents
        shuffled_currents = dense_currents + torch.roll(error, shifts=1, dims=0)
        dense_logits, dense_spikes, near = head_with_trace(head, dense_currents, config.near_threshold_band)
        dense_prediction = dense_logits.argmax(dim=1)
        candidates = {
            "dense_reference": (dense_currents, dense_logits, dense_spikes),
            "sparse_operator": (*head_with_trace(head, sparse_currents, config.near_threshold_band)[:2],),
            "shuffled_error_control": (*head_with_trace(head, shuffled_currents, config.near_threshold_band)[:2],),
        }
        # Reorder tuples from helper to (currents, logits, spikes).
        candidates["sparse_operator"] = (sparse_currents, candidates["sparse_operator"][0], candidates["sparse_operator"][1])
        candidates["shuffled_error_control"] = (shuffled_currents, candidates["shuffled_error_control"][0], candidates["shuffled_error_control"][1])
        for arm, (currents, logits, spikes) in candidates.items():
            state = totals[arm]
            prediction = logits.argmax(dim=1)
            state["correct"] += int((prediction == y).sum().item())
            state["agreement"] += int((prediction == dense_prediction).sum().item())
            state["spike_diff"] += float((spikes != dense_spikes).sum().item())
            state["spikes"] += int(spikes.numel())
            state["logit_max"] = max(state["logit_max"], float((logits - dense_logits).abs().max().item()))
            state["current_max"] = max(state["current_max"], float((currents - dense_currents).abs().max().item()))
            state["near"] += float(near)
            state["units"] += 1
        mark_step(device)
    examples = int(labels.shape[0]); wall = time.perf_counter() - started
    dense_accuracy = totals["dense_reference"]["correct"] / examples
    rows = []
    for arm in GEN27_ARMS:
        state = totals[arm]; accuracy = state["correct"] / examples
        rows.append({
            "seed": int(seed), "arm": arm,
            "accuracy": float(accuracy), "accuracy_change_vs_dense": float(accuracy - dense_accuracy),
            "prediction_agreement_vs_dense": float(state["agreement"] / examples),
            "spike_disagreement_rate": float(state["spike_diff"] / max(state["spikes"], 1)),
            "maximum_logit_difference": float(state["logit_max"]),
            "maximum_current_difference": float(state["current_max"]),
            "mean_near_threshold_fraction": float(state["near"] / max(state["units"], 1)),
            "best_epoch": int(training["best_epoch"]),
            "best_validation_accuracy": float(training["best_validation_accuracy"]),
            "train_seconds": float(training["train_seconds"]), "evaluation_seconds": float(wall),
        })
    return rows


def head_with_trace(head, currents, near_threshold_band):
    direct = torch.relu(currents); leak = torch.sigmoid(head.leak_logit)
    threshold = torch.nn.functional.softplus(head.threshold_raw).clamp_min(1e-3)
    membrane = currents.new_zeros((currents.shape[0], head.channels)); states=[]; near=0.0
    for step in range(currents.shape[1]):
        pre_reset = leak * membrane + currents[:, step]
        margin = pre_reset - threshold
        near += float((margin.abs() <= near_threshold_band).to(torch.float32).mean().item())
        state = (margin >= 0).to(currents.dtype)
        membrane = pre_reset - state * threshold; states.append(state)
    spikes = torch.stack(states, dim=1)
    features = _multiscale_features(direct, head.temporal_levels)
    features.extend(_multiscale_features(spikes, head.temporal_levels)); features.append(membrane / threshold)
    return head.classifier(torch.cat(features, dim=1)), spikes, near / int(currents.shape[1])


def summarize_gen27(records):
    summary=[]
    for arm in GEN27_ARMS:
        group=[row for row in records if row["arm"]==arm]
        if not group: continue
        summary.append({
            "arm":arm,"seeds":len(group),
            "mean_accuracy":statistics.fmean(float(row["accuracy"]) for row in group),
            "std_accuracy":statistics.pstdev(float(row["accuracy"]) for row in group) if len(group)>1 else 0.0,
            "mean_accuracy_change_vs_dense":statistics.fmean(float(row["accuracy_change_vs_dense"]) for row in group),
            "minimum_prediction_agreement_vs_dense":min(float(row["prediction_agreement_vs_dense"]) for row in group),
            "mean_spike_disagreement_rate":statistics.fmean(float(row["spike_disagreement_rate"]) for row in group),
            "maximum_logit_difference":max(float(row["maximum_logit_difference"]) for row in group),
            "maximum_current_difference":max(float(row["maximum_current_difference"]) for row in group),
            "mean_near_threshold_fraction":statistics.fmean(float(row["mean_near_threshold_fraction"]) for row in group),
        })
    return summary


def decide_gen27(records, config):
    sparse=[row for row in records if row["arm"]=="sparse_operator"]
    accuracy_stable=bool(sparse) and all(abs(float(row["accuracy_change_vs_dense"]))<=config.maximum_accuracy_change for row in sparse)
    predictions_stable=bool(sparse) and all(float(row["prediction_agreement_vs_dense"])>=config.minimum_prediction_agreement for row in sparse)
    spikes_stable=bool(sparse) and all(float(row["spike_disagreement_rate"])<=config.maximum_spike_disagreement for row in sparse)
    passed=bool(accuracy_stable and predictions_stable and spikes_stable)
    return {
        "status":"pass" if passed else "stop",
        "trained_accuracy_stable":bool(accuracy_stable),
        "trained_predictions_stable":bool(predictions_stable),
        "trained_spikes_stable":bool(spikes_stable),
        "behavioral_sparse_semantics_supported":passed,
        "hardware_energy_claim_authorized":False,
        "next_milestone":"custom_triton_event_kernel" if passed else "threshold_margin_training",
    }


def _load_data(config):
    return load_ssc_tensors(_model_config(config, config.seeds[0]), validation_samples=config.validation_samples)


def _model_config(config, seed):
    return SHDConfig(
        seeds=(seed,),train_samples=config.source_train_samples,test_samples=config.test_samples,
        input_neurons=config.input_neurons,classes=config.classes,timesteps=config.timesteps,
        duration_seconds=config.duration_seconds,hidden_neurons=128,max_edges=4096,
        epochs=config.epochs,warmup_epochs=0,learning_rate=config.learning_rate,
        reservoir_learning_rate=0.0,weight_decay=config.weight_decay,batch_size=config.batch_size,
        data_root=config.data_root,download=config.download,
    )


def plot_gen27(result,path):
    import matplotlib.pyplot as plt
    labels=[row["arm"].replace("_","\n") for row in result.summary];x=range(len(labels))
    fig,axes=plt.subplots(2,1,figsize=(10,8),constrained_layout=True)
    axes[0].bar(x,[100*row["mean_accuracy"] for row in result.summary],color="#35b4f2");axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_title("AMMC Gen-27 trained threshold robustness")
    axes[1].bar(x,[row["mean_spike_disagreement_rate"] for row in result.summary],color="#ffb31a");axes[1].set_ylabel("Spike disagreement rate")
    for axis in axes:axis.set_xticks(list(x),labels);axis.grid(axis="y",alpha=.25)
    destination=pathlib.Path(path);destination.parent.mkdir(parents=True,exist_ok=True);fig.savefig(destination,dpi=180);plt.close(fig)


def bundle_gen27_artifacts(paths,output_dir):
    output=pathlib.Path(output_dir);files=[pathlib.Path(value) for value in paths.values() if pathlib.Path(value).is_file()]
    manifest=output/"gen27_trained_threshold_robustness_manifest.json"
    manifest.write_text(json.dumps({"files":[{"name":file.name,"sha256":hashlib.sha256(file.read_bytes()).hexdigest()} for file in files]},indent=2)+"\n",encoding="utf-8")
    archive=output/"gen27_trained_threshold_robustness_bundle.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as bundle:
        for file in files+[manifest]:bundle.write(file,arcname=file.name)
    return {"manifest":str(manifest),"bundle":str(archive)}


def _save_progress(path,config,records):
    if path is None:return
    destination=pathlib.Path(path);destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps({"config":asdict(config),"records":records},indent=2)+"\n",encoding="utf-8")


def _load_progress(path,config):
    if path is None or not pathlib.Path(path).exists():return []
    try:payload=json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError):return []
    return list(payload.get("records",[])) if payload.get("config")==json.loads(json.dumps(asdict(config))) else []


def _validate_config(config):
    if config.input_neurons!=700 or config.classes!=35:raise ValueError("Gen-27 is frozen for SSC")
    if len(config.seeds)<3:raise ValueError("Gen-27 requires at least three trained seeds")


def _write_csv(path,rows):
    if not rows:pathlib.Path(path).write_text("",encoding="utf-8");return
    with pathlib.Path(path).open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
