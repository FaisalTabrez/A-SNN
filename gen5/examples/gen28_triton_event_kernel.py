"""Run Gen-28 Triton event-native kernel audit."""
from __future__ import annotations
import argparse,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ammc_gen5 import Gen28Config,bundle_gen28_artifacts,run_gen28  # noqa: E402
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--device",default="cuda");parser.add_argument("--data-root",default="gen5_data/ssc")
    parser.add_argument("--no-download",action="store_true");parser.add_argument("--test-samples",type=int,default=256);parser.add_argument("--batch-sizes",nargs="+",type=int,default=[1,32,256])
    parser.add_argument("--output-dir",default="gen5_outputs/gen28_triton_event_kernel_cuda");parser.add_argument("--no-plot",action="store_true");args=parser.parse_args()
    config=Gen28Config(data_root=args.data_root,download=not args.no_download,test_samples=args.test_samples,batch_sizes=tuple(args.batch_sizes))
    result=run_gen28(config,device=args.device);paths=result.save(args.output_dir,plot=not args.no_plot);paths.update(bundle_gen28_artifacts(paths,args.output_dir))
    print(json.dumps({"paths":paths,"device":result.device,"decision":result.decision,"summary":result.summary},indent=2))
if __name__=="__main__":main()
