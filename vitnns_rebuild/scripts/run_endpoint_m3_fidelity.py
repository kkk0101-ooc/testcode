#!/usr/bin/env python3
"""Execute M3-A independent NNS equivalence on five graph families."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["OPENBLAS_NUM_THREADS"]="1"
import argparse, json, time
from pathlib import Path
import sys
import torch
REPO=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(REPO/"src"))
from vitnns.endpoint_m3_fidelity import FidelityNetwork,compare_learning_images

KINDS=("fcn","cnn","cnn_maxpool","residual_mlp","residual_cnn")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--spec",type=Path,default=REPO/"config/nns_vit_endpoint/endpoint_m3.json")
    p.add_argument("--output",type=Path,required=True); a=p.parse_args(); spec=json.loads(a.spec.read_text())["equivalence"]
    started=time.monotonic(); records=[]
    for dtype_name,dtype in (("float64",torch.float64),("float32",torch.float32)):
        tolerance=spec[dtype_name]
        for kind in KINDS:
            model=FidelityNetwork(kind,dtype).eval(); generator=torch.Generator().manual_seed(88000+KINDS.index(kind))
            for sample in range(spec["random_inputs_per_model_dtype"]):
                x=torch.randn(model.input_shape,generator=generator,dtype=dtype)
                rows=compare_learning_images(model,x,tolerance["atol"],tolerance["rtol"])
                for row in rows: row.update(model=kind,dtype=dtype_name,sample=sample)
                records.extend(rows)
    failures=[r for r in records if r["max_tolerance_ratio"]>1]
    summary={
        "status":"PASS" if not failures else "FAIL", "independent_methods":["explicit_layerwise_learning_image","torch_autograd_x_times_gradient"],
        "models":list(KINDS), "random_inputs_per_model_dtype":spec["random_inputs_per_model_dtype"],
        "comparison_count":len(records), "failure_count":len(failures),
        "max_relative_l2":max(r["relative_l2"] for r in records), "max_tolerance_ratio":max(r["max_tolerance_ratio"] for r in records),
        "by_dtype":{d:{"comparisons":sum(r["dtype"]==d for r in records),"max_relative_l2":max(r["relative_l2"] for r in records if r["dtype"]==d),"max_tolerance_ratio":max(r["max_tolerance_ratio"] for r in records if r["dtype"]==d)} for d in ("float64","float32")},
        "records":records, "wall_seconds":time.monotonic()-started,
    }
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({k:summary[k] for k in ("status","comparison_count","max_relative_l2","max_tolerance_ratio","wall_seconds")}))
    if failures: raise SystemExit(1)
if __name__=="__main__": main()
