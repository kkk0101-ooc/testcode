#!/usr/bin/env python3
"""Compare analytic ViT-operator JVPs to autograd JVP/VJP and FD curves."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""; os.environ["OPENBLAS_NUM_THREADS"]="1"
import argparse,json,time
from pathlib import Path
import sys,torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/"src"))
from vitnns.endpoint_m3_operators import cases

def norm(x):return float(x.norm())
def main():
    p=argparse.ArgumentParser();p.add_argument("--spec",type=Path,default=REPO/"config/nns_vit_endpoint/endpoint_m3.json");p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    spec=json.loads(a.spec.read_text())["operator_validation"];started=time.monotonic();records=[]
    for operator,primals,function,analytic in cases():
        for direction_id in range(spec["directions_per_operator"]):
            g=torch.Generator().manual_seed(91900+100*list(x[0] for x in cases()).index(operator)+direction_id)
            directions=tuple(torch.randn(x.shape,generator=g,dtype=x.dtype) for x in primals)
            explicit=analytic(primals,directions)
            automatic=torch.autograd.functional.jvp(function,primals,directions,create_graph=False,strict=True)[1]
            jvp_error=norm(explicit-automatic)/(norm(automatic)+1e-30)
            cotangent=torch.randn(explicit.shape,generator=g,dtype=explicit.dtype)
            differentiable=tuple(x.detach().clone().requires_grad_(True) for x in primals);output=function(*differentiable)
            vjp=torch.autograd.grad((output*cotangent).sum(),differentiable)
            lhs=float((explicit*cotangent).sum());rhs=sum(float((d*v).sum()) for d,v in zip(directions,vjp))
            identity_error=abs(lhs-rhs)/(max(abs(lhs),abs(rhs),1e-30))
            curve=[]
            for epsilon in spec["epsilons"]:
                plus=function(*(x+epsilon*d for x,d in zip(primals,directions)));minus=function(*(x-epsilon*d for x,d in zip(primals,directions)))
                fd=(plus-minus)/(2*epsilon);curve.append({"epsilon":epsilon,"relative_error":norm(fd-explicit)/(norm(explicit)+1e-30),"absolute_error":norm(fd-explicit)})
            records.append({"operator":operator,"direction":direction_id,"analytic_autograd_jvp_relative_error":jvp_error,"jvp_vjp_identity_relative_error":identity_error,"finite_difference_curve":curve,"finite_difference_best_relative_error":min(x["relative_error"] for x in curve)})
    failures=[r for r in records if r["analytic_autograd_jvp_relative_error"]>spec["jvp_rtol"] or r["jvp_vjp_identity_relative_error"]>spec["vjp_identity_atol"] or r["finite_difference_best_relative_error"]>spec["finite_difference_best_relative_error_max"]]
    result={"status":"PASS" if not failures else "FAIL","operators":sorted({r["operator"] for r in records}),"directions_per_operator":spec["directions_per_operator"],"record_count":len(records),"failure_count":len(failures),"max_analytic_autograd_jvp_relative_error":max(r["analytic_autograd_jvp_relative_error"] for r in records),"max_jvp_vjp_identity_relative_error":max(r["jvp_vjp_identity_relative_error"] for r in records),"max_best_finite_difference_relative_error":max(r["finite_difference_best_relative_error"] for r in records),"records":records,"wall_seconds":time.monotonic()-started}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps({k:result[k] for k in result if k not in {"records","operators"}}))
    if failures:raise SystemExit(1)
if __name__=="__main__":main()
