#!/usr/bin/env python3
"""Create new synthetic reference data without changing any NSE observations."""
import argparse,json,sys,time,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.mentor_dh_pinn.regular_pinn_data import DOMAIN,draw_points,teacher_labels

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--factors",type=int,choices=(1,2),required=True)
    ap.add_argument("--train",type=int,default=131072);ap.add_argument("--validation",type=int,default=16384)
    ap.add_argument("--seed",type=int,default=906100)
    ap.add_argument('--full-gradients',action='store_true');args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    report={"domain":DOMAIN,"factors":args.factors,"seed":args.seed,"source":"synthetic only",
            "teacher":"128-node Fourier; every price checked at 96 nodes, tolerance 1e-9",
            "training_labels":"g=log(IV/sqrt(expected_variance)), exact implicit parameter sensitivities",
            "rejections":"All original candidates retained; usability mask explicit","splits":{}}
    start=time.perf_counter()
    for split,n,seed in (("train",args.train,args.seed),("validation",args.validation,args.seed+1)):
        q=draw_points(n,args.factors,seed)
        d=teacher_labels(q,args.factors,full_gradients=args.full_gradients)
        path=args.out/f"{split}.npz";np.savez_compressed(path,**d)
        report["splits"][split]={"seed":seed,"candidates":n,"usable":int(d["usable"].sum()),
                "sha256":hashlib.file_digest(path.open("rb"),"sha256").hexdigest()}
        print(split,report["splits"][split],flush=True)
    col=draw_points(18000,args.factors,args.seed+2,collocation=True)
    np.savez_compressed(args.out/"collocation.npz",q=col)
    report["collocation_points"]=len(col);report["seconds"]=time.perf_counter()-start
    report['full_geometry_gradients']=args.full_gradients
    report["source_sha256"]={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (Path(__file__),ROOT/"src/mentor_dh_pinn/regular_pinn_data.py",ROOT/"src/mentor_dh_pinn/torch_pricer.py")}
    (args.out/"manifest.json").write_text(json.dumps(report,indent=2))
    print("complete",report["seconds"],flush=True)

if __name__=="__main__":main()
