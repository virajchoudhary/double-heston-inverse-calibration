#!/usr/bin/env python3
"""Rebuild the fresh synthetic training/validation corpus used by the repair protocol."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.mentor_dh_pinn.dataset_v6 import build


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    out=ROOT/"outputs/calibration_repair/synthetic"
    out.mkdir(parents=True,exist_ok=True)
    manifest={"source":"synthetic Double Heston characteristic-function prices; not NSE quotes",
              "nodes":64,"storage_dtype":"float64","datasets":{}}
    parameter_sets=[]
    for name,n,seed in (("train",20000,905001),("validation",2000,905002)):
        path=out/f"v6_{name}.npz"
        if not path.exists():
            d=build(n,seed,storage_dtype=np.float64)
            np.savez_compressed(path,**d)
        else:
            d=dict(np.load(path,allow_pickle=False))
        if len(d["params"])!=n or d["clean"].dtype!=np.float64:
            raise ValueError(f"Unexpected existing corpus: {path}")
        keys={row.tobytes() for row in d["params"]}
        parameter_sets.append(keys)
        manifest["datasets"][name]={"seed":seed,"candidates":n,"valid_surfaces":int(d["ok"].sum()),
            "valid_quotes":int(d["n_quotes"][d["ok"]].sum()),
            "rejected_surfaces":int((~d["ok"]).sum()),"duplicate_parameter_rows":n-len(keys),
            "sha256":hashlib.file_digest(path.open("rb"),"sha256").hexdigest()}
    manifest["train_validation_parameter_overlap"]=len(parameter_sets[0]&parameter_sets[1])
    if manifest["train_validation_parameter_overlap"]:
        raise ValueError("Synthetic split overlap")
    stats=ROOT/"outputs/unified_v6/latent_standardisation.json"
    destination=out/stats.name
    if not destination.exists():
        destination.write_bytes(stats.read_bytes())
    if destination.read_bytes()!=stats.read_bytes():
        raise ValueError("Latent standardisation changed from the base training-only statistics")
    manifest["latent_standardisation"]="Inherited from the base synthetic training set only"
    manifest["generator_sha256"]=hashlib.sha256((ROOT/"src/mentor_dh_pinn/dataset_v6.py").read_bytes()).hexdigest()
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))


if __name__=="__main__":
    main()
