"""Gen-28 Triton event-native temporal input-kernel audit."""

import csv
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import pathlib
import statistics
import time
import zipfile

from .event_mnist import torch
from .gen24_compiled_residual_state import _benchmark_callable
from .gen25_event_driven_sparse_audit import (
    DenseResidualPipeline,
    ResidualLIFStateHead,
    sparse_temporal_currents,
)
from .runtime import device_kind, resolve_device, seed_everything, sync
from .shd_benchmark import SHDConfig
from .shd_state_placement_diagnostic import (
    ResidualTemporalConvStateClassifier,
    matched_temporal_conv_residual_channels,
)
from .ssc_benchmark import load_ssc_tensors


GEN28_RUNTIMES = ("compiled_dense", "triton_event_native", "triton_end_to_end")


@dataclass(frozen=True)
class Gen28Config:
    seeds: tuple[int, ...] = (661, 662, 663)
    input_neurons: int = 700
    classes: int = 35
    timesteps: int = 64
    data_root: str = "gen5_data/ssc"
    download: bool = True
    test_samples: int = 256
    batch_sizes: tuple[int, ...] = (1, 32, 256)
    density_batch_size: int = 32
    synthetic_densities: tuple[float, ...] = (0.001, 0.005, 0.01)
    target_parameters: int = 133_631
    temporal_levels: tuple[int, ...] = (1, 2, 4, 8)
    kernel_size: int = 5
    surrogate_slope: float = 10.0
    compile_mode: str = "reduce-overhead"
    warmup_iterations: int = 3
    measurement_iterations: int = 10
    measurement_repeats: int = 3
    event_block_size: int = 32
    channel_block_size: int = 32
    maximum_current_difference_vs_coo: float = 1e-3
    minimum_prediction_agreement_vs_dense: float = 0.999
    minimum_real_event_native_speed_ratio: float = 1.0


@dataclass
class Gen28Result:
    config: dict
    device: str
    dataset: dict
    architecture: dict
    records: list[dict]
    summary: list[dict]
    decision: dict

    def save(self, output_dir: str | pathlib.Path, *, plot: bool = True) -> dict[str, str]:
        output=pathlib.Path(output_dir);output.mkdir(parents=True,exist_ok=True)
        json_path=output/"gen28_triton_event_kernel.json";records_path=output/"gen28_triton_event_kernel_records.csv";summary_path=output/"gen28_triton_event_kernel_summary.csv"
        json_path.write_text(json.dumps(asdict(self),indent=2)+"\n",encoding="utf-8");_write_csv(records_path,self.records);_write_csv(summary_path,self.summary)
        paths={"json":str(json_path),"records_csv":str(records_path),"summary_csv":str(summary_path)}
        if plot:
            plot_path=output/"gen28_triton_event_kernel.png";plot_gen28(self,plot_path);paths["plot"]=str(plot_path)
        return paths


def build_event_list(events):
    nonzero=torch.nonzero(events,as_tuple=False)
    if nonzero.numel()==0:
        empty=torch.empty(0,dtype=torch.int32,device=events.device)
        return empty,empty.clone(),empty.clone(),torch.empty(0,dtype=torch.float32,device=events.device)
    return (
        nonzero[:,0].to(torch.int32),nonzero[:,1].to(torch.int32),nonzero[:,2].to(torch.int32),
        events[nonzero[:,0],nonzero[:,1],nonzero[:,2]].to(torch.float32),
    )


@lru_cache(maxsize=1)
def _load_triton_kernel():
    try:
        import triton
        import triton.language as tl
    except ImportError as error:
        raise ImportError("Gen-28 requires Triton from a CUDA PyTorch runtime") from error

    @triton.jit
    def event_conv_kernel(
        event_b,event_t,event_i,event_v,weight,output,n_events,
        timesteps:tl.constexpr,input_neurons:tl.constexpr,channels:tl.constexpr,
        kernel_size:tl.constexpr,block_events:tl.constexpr,block_channels:tl.constexpr,
    ):
        event_offsets=tl.program_id(0)*block_events+tl.arange(0,block_events)
        channel_offsets=tl.program_id(1)*block_channels+tl.arange(0,block_channels)
        event_mask=event_offsets<n_events;channel_mask=channel_offsets<channels
        batch_index=tl.load(event_b+event_offsets,mask=event_mask,other=0)
        event_time=tl.load(event_t+event_offsets,mask=event_mask,other=0)
        input_index=tl.load(event_i+event_offsets,mask=event_mask,other=0)
        value=tl.load(event_v+event_offsets,mask=event_mask,other=0.0)
        for kernel_index in tl.static_range(0,kernel_size):
            output_time=event_time-kernel_index+kernel_size//2
            valid_time=(output_time>=0)&(output_time<timesteps)
            weight_offset=(channel_offsets[None,:]*input_neurons+input_index[:,None])*kernel_size+kernel_index
            weights=tl.load(weight+weight_offset,mask=event_mask[:,None]&channel_mask[None,:],other=0.0)
            output_offset=(batch_index[:,None]*timesteps+output_time[:,None])*channels+channel_offsets[None,:]
            tl.atomic_add(output+output_offset,value[:,None]*weights,
                mask=event_mask[:,None]&channel_mask[None,:]&valid_time[:,None])
    return triton,event_conv_kernel


def triton_event_currents(event_list,temporal,*,batch_size,timesteps,block_events=32,block_channels=32):
    triton,kernel=_load_triton_kernel();event_b,event_t,event_i,event_v=event_list
    channels=int(temporal.weight.shape[0]);input_neurons=int(temporal.weight.shape[1]);kernel_size=int(temporal.weight.shape[2])
    output=temporal.bias.to(torch.float32).view(1,1,channels).expand(batch_size,timesteps,channels).clone()
    n_events=int(event_v.numel())
    if n_events:
        grid=(triton.cdiv(n_events,block_events),triton.cdiv(channels,block_channels))
        kernel[grid](event_b,event_t,event_i,event_v,temporal.weight,output,n_events=n_events,
            timesteps=timesteps,input_neurons=input_neurons,channels=channels,kernel_size=kernel_size,
            block_events=block_events,block_channels=block_channels)
    return output


class EventNativePipeline:
    def __init__(self,event_list,temporal,head,batch_size,timesteps,config):
        self.event_list=event_list;self.temporal=temporal;self.head=head;self.batch_size=batch_size;self.timesteps=timesteps;self.config=config
    def __call__(self,_events):
        currents=triton_event_currents(self.event_list,self.temporal,batch_size=self.batch_size,timesteps=self.timesteps,
            block_events=self.config.event_block_size,block_channels=self.config.channel_block_size)
        return self.head(currents)


class EndToEndTritonPipeline:
    def __init__(self,temporal,head,config):self.temporal=temporal;self.head=head;self.config=config
    def __call__(self,events):
        currents=triton_event_currents(build_event_list(events),self.temporal,batch_size=int(events.shape[0]),timesteps=int(events.shape[1]),
            block_events=self.config.event_block_size,block_channels=self.config.channel_block_size)
        return self.head(currents)


def run_gen28(config:Gen28Config=Gen28Config(),*,device="auto",dataset=None)->Gen28Result:
    _validate_config(config);resolved=resolve_device(device)
    if device_kind(resolved)!="cuda":raise ValueError("Gen-28 requires a CUDA runtime")
    _load_triton_kernel()
    test_events,test_labels=dataset if dataset is not None else _load_test_data(config)
    channels,parameters=matched_temporal_conv_residual_channels(config.input_neurons,config.classes,config.target_parameters,
        kernel_size=config.kernel_size,temporal_levels=config.temporal_levels)
    records=[]
    for seed in config.seeds:
        seed_everything(seed,device=resolved)
        source=ResidualTemporalConvStateClassifier(_model_config(config),channels=channels,kernel_size=config.kernel_size,
            temporal_levels=config.temporal_levels,dynamics="lif",surrogate_slope=config.surrogate_slope).to(resolved).eval()
        for parameter in source.parameters():parameter.requires_grad_(False)
        dense=torch.compile(DenseResidualPipeline(source).to(resolved).eval(),mode=config.compile_mode)
        head=torch.compile(ResidualLIFStateHead(source).to(resolved).eval(),mode=config.compile_mode)
        for batch_size in config.batch_sizes:
            real=test_events[:batch_size].to(resolved).to(torch.float32)
            workloads=[("real_ssc",real,float(real.mean().item()))]
            if batch_size==config.density_batch_size:
                for density in config.synthetic_densities:
                    workloads.append((f"synthetic_{density:.4f}",_fixed_density_events(batch_size,config,density,seed,resolved),density))
            for workload,batch,density in workloads:
                records.extend(_measure_workload(seed,workload,batch,density,source,dense,head,config,resolved))
    summary=summarize_gen28(records)
    return Gen28Result(config=asdict(config),device=device_kind(resolved),
        dataset={"name":"Spiking Speech Commands","test_samples":int(test_events.shape[0]),"labels_loaded_for_identity_only":int(test_labels.shape[0])},
        architecture={"model":"Phase-49 residual LIF","channels":channels,"trainable_parameters":parameters,
            "kernel":"Triton event scatter with atomic accumulation","event_native_conversion_excluded":True},
        records=records,summary=summary,decision=decide_gen28(records,config))


def _measure_workload(seed,workload,batch,density,source,dense,head,config,device):
    event_list=build_event_list(batch);native=EventNativePipeline(event_list,source.temporal,head,int(batch.shape[0]),int(batch.shape[1]),config)
    end_to_end=EndToEndTritonPipeline(source.temporal,head,config)
    with torch.inference_mode():
        dense_logits=dense(batch);coo_currents=sparse_temporal_currents(batch,source.temporal)
        triton_currents=triton_event_currents(event_list,source.temporal,batch_size=int(batch.shape[0]),timesteps=int(batch.shape[1]),
            block_events=config.event_block_size,block_channels=config.channel_block_size)
        triton_logits=head(triton_currents)
    sync(device)
    current_difference=float((triton_currents-coo_currents).abs().max().item())
    agreement=float((triton_logits.argmax(1)==dense_logits.argmax(1)).to(torch.float32).mean().item())
    metrics={}
    for runtime,callable_model in (("compiled_dense",dense),("triton_event_native",native),("triton_end_to_end",end_to_end)):
        metrics[runtime]=_benchmark_callable(callable_model,batch,device,warmup_iterations=config.warmup_iterations,
            measurement_iterations=config.measurement_iterations,measurement_repeats=config.measurement_repeats)
    dense_rate=metrics["compiled_dense"]["examples_per_second"];rows=[]
    for runtime in GEN28_RUNTIMES:
        row={"seed":int(seed),"workload":workload,"batch_size":int(batch.shape[0]),"event_density":float(density),
            "active_events":int(event_list[3].numel()),"runtime":runtime,**metrics[runtime],
            "speed_ratio_vs_dense":float(metrics[runtime]["examples_per_second"]/max(dense_rate,1e-12)),
            "maximum_current_difference_vs_coo":current_difference if runtime!="compiled_dense" else 0.0,
            "prediction_agreement_vs_dense":agreement if runtime!="compiled_dense" else 1.0}
        rows.append(row)
    return rows


def summarize_gen28(records):
    summary=[];keys=sorted({(row["workload"],int(row["batch_size"]),row["runtime"]) for row in records})
    for workload,batch_size,runtime in keys:
        group=[row for row in records if row["workload"]==workload and int(row["batch_size"])==batch_size and row["runtime"]==runtime]
        summary.append({"workload":workload,"batch_size":batch_size,"runtime":runtime,"seeds":len(group),
            "mean_event_density":statistics.fmean(float(row["event_density"]) for row in group),
            "mean_examples_per_second":statistics.fmean(float(row["examples_per_second"]) for row in group),
            "mean_speed_ratio_vs_dense":statistics.fmean(float(row["speed_ratio_vs_dense"]) for row in group),
            "maximum_current_difference_vs_coo":max(float(row["maximum_current_difference_vs_coo"]) for row in group),
            "minimum_prediction_agreement_vs_dense":min(float(row["prediction_agreement_vs_dense"]) for row in group),
            "maximum_cuda_peak_memory_mb":max(float(row["cuda_peak_memory_mb"]) for row in group)})
    return summary


def decide_gen28(records,config):
    native=[row for row in records if row["runtime"]=="triton_event_native"]
    numerical=bool(native) and all(float(row["maximum_current_difference_vs_coo"])<=config.maximum_current_difference_vs_coo and
        float(row["prediction_agreement_vs_dense"])>=config.minimum_prediction_agreement_vs_dense for row in native)
    primary=[row for row in native if row["workload"]=="real_ssc" and int(row["batch_size"])==max(config.batch_sizes)]
    real_ratio=statistics.fmean(float(row["speed_ratio_vs_dense"]) for row in primary) if primary else 0.0
    low=[row for row in native if row["workload"].startswith("synthetic_")]
    best_low=max((float(row["speed_ratio_vs_dense"]) for row in low),default=0.0)
    end=[row for row in records if row["runtime"]=="triton_end_to_end" and row["workload"]=="real_ssc" and int(row["batch_size"])==max(config.batch_sizes)]
    end_ratio=statistics.fmean(float(row["speed_ratio_vs_dense"]) for row in end) if end else 0.0
    native_pass=bool(numerical and real_ratio>=config.minimum_real_event_native_speed_ratio)
    return {"status":"pass" if native_pass else "stop","numerical_contract_passed":bool(numerical),
        "mean_real_event_native_speed_ratio":float(real_ratio),"mean_real_end_to_end_speed_ratio":float(end_ratio),
        "best_low_density_event_native_speed_ratio":float(best_low),"sensor_native_kernel_supported":native_pass,
        "dense_cache_kernel_supported":bool(native_pass and end_ratio>=1.0),"hardware_energy_claim_authorized":False,
        "next_milestone":"event_stream_integration" if native_pass else "density_gated_hybrid" if numerical and best_low>=1.0 else "close_event_sparse_software_path"}


def _load_test_data(config):
    data=load_ssc_tensors(_model_config(config),validation_samples=1);return data[4],data[5]
def _model_config(config):
    return SHDConfig(seeds=config.seeds,train_samples=1,test_samples=config.test_samples,input_neurons=config.input_neurons,
        classes=config.classes,timesteps=config.timesteps,hidden_neurons=128,max_edges=4096,epochs=1,warmup_epochs=0,
        batch_size=max(config.batch_sizes),data_root=config.data_root,download=config.download)
def _fixed_density_events(batch_size,config,density,seed,device):
    total=batch_size*config.timesteps*config.input_neurons;active=max(1,round(total*density));generator=torch.Generator().manual_seed(seed+round(density*1_000_000))
    indices=torch.randperm(total,generator=generator)[:active];flat=torch.zeros(total);flat[indices]=1.0;return flat.reshape(batch_size,config.timesteps,config.input_neurons).to(device)


def plot_gen28(result,path):
    import matplotlib.pyplot as plt
    native=[row for row in result.summary if row["runtime"]=="triton_event_native"]
    labels=[f"{row['workload']}\nB={row['batch_size']}" for row in native];x=range(len(labels))
    fig,axis=plt.subplots(figsize=(13,6),constrained_layout=True);axis.bar(x,[row["mean_speed_ratio_vs_dense"] for row in native],color="#35b4f2")
    axis.axhline(1.0,color="black",linestyle="--");axis.set_title("AMMC Gen-28 Triton event-native kernel");axis.set_ylabel("Triton / compiled-dense throughput")
    axis.set_xticks(list(x),labels,rotation=25,ha="right");axis.grid(axis="y",alpha=.25);destination=pathlib.Path(path);destination.parent.mkdir(parents=True,exist_ok=True);fig.savefig(destination,dpi=180);plt.close(fig)
def bundle_gen28_artifacts(paths,output_dir):
    output=pathlib.Path(output_dir);files=[pathlib.Path(value) for value in paths.values() if pathlib.Path(value).is_file()];manifest=output/"gen28_triton_event_kernel_manifest.json"
    manifest.write_text(json.dumps({"files":[{"name":file.name,"sha256":hashlib.sha256(file.read_bytes()).hexdigest()} for file in files]},indent=2)+"\n",encoding="utf-8")
    archive=output/"gen28_triton_event_kernel_bundle.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as bundle:
        for file in files+[manifest]:bundle.write(file,arcname=file.name)
    return {"manifest":str(manifest),"bundle":str(archive)}
def _validate_config(config):
    if config.input_neurons!=700 or config.classes!=35:raise ValueError("Gen-28 is frozen for SSC")
    if len(config.seeds)<3:raise ValueError("Gen-28 requires three timing seeds")
    if config.density_batch_size not in config.batch_sizes:raise ValueError("density batch must be registered")
def _write_csv(path,rows):
    if not rows:pathlib.Path(path).write_text("",encoding="utf-8");return
    with pathlib.Path(path).open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
