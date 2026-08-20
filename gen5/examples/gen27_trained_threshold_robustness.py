"""Run Gen-27 trained SSC sparse-threshold robustness."""
from __future__ import annotations
import argparse,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ammc_gen5 import Gen27Config,bundle_gen27_artifacts,run_gen27  # noqa: E402

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",default="cuda");parser.add_argument("--data-root",default="gen5_data/ssc")
    parser.add_argument("--no-download",action="store_true");parser.add_argument("--source-train-samples",type=int,default=20_000)
    parser.add_argument("--validation-samples",type=int,default=2_999);parser.add_argument("--test-samples",type=int,default=8_000)
    parser.add_argument("--epochs",type=int,default=12);parser.add_argument("--batch-size",type=int,default=256)
    parser.add_argument("--output-dir",default="gen5_outputs/gen27_trained_threshold_robustness_cuda")
    parser.add_argument("--progress-path",default=None);parser.add_argument("--no-plot",action="store_true");args=parser.parse_args()
    config=Gen27Config(data_root=args.data_root,download=not args.no_download,source_train_samples=args.source_train_samples,
        validation_samples=args.validation_samples,test_samples=args.test_samples,epochs=args.epochs,batch_size=args.batch_size)
    progress=args.progress_path or str(pathlib.Path(args.output_dir)/"gen27_trained_threshold_robustness_progress.json")
    result=run_gen27(config,device=args.device,progress_path=progress);paths=result.save(args.output_dir,plot=not args.no_plot);paths["progress"]=progress
    paths.update(bundle_gen27_artifacts(paths,args.output_dir));print(json.dumps({"paths":paths,"device":result.device,"decision":result.decision,"summary":result.summary},indent=2))
if __name__=="__main__":main()
