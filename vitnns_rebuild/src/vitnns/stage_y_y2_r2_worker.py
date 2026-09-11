"""One-visible-device, single-attempt missing-unit worker."""
from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

from . import stage_y_y2 as y2
from . import stage_y_y2_r2 as r2


def check_environment() -> None:
    r2.require(os.environ.get("CUBLAS_WORKSPACE_CONFIG")==":4096:8","CUBLAS_WORKSPACE_CONFIG must equal :4096:8 before Torch initialization",r2.StartupError)
    r2.require(os.path.realpath(sys.executable)==os.path.realpath(r2.PYTHON),"worker Python executable drift",r2.StartupError)


def publish_success(repo: Path, unit: Path, order: int, shard: int, env_digest: str,
                    result: dict[str,Any], quadrature: list[dict[str,Any]], arrays: dict[str,Any],
                    resource: dict[str,Any]) -> dict[str,Any]:
    from . import stage_y_y2_worker as engine
    publish_started=time.monotonic();publish_cpu=time.process_time()
    rows=[]
    r2.require(tuple(sorted(arrays))==r2.EXPECTED_ARRAY_NAMES,"computed array roster drift")
    for name,value in sorted(arrays.items()):
        row=engine._write_array_new(unit/"arrays"/(name+".npy"),value)
        row["path"]=(unit/"arrays"/(name+".npy")).relative_to(repo).as_posix()
        r2.validate_array_record(repo,row);rows.append(row)
    science=r2.signed("scientific_result","VALID",{key:result[key] for key in ("target_identity","clean_logit","baseline_logit","delta","selected_steps","closure","ablations","deletion_operator","calls","phase_telemetry")})
    r2.write_json_new(unit/"scientific_result.json",science)
    attempt_id=f"unit-{order:03d}-scientific-001"
    r2.write_json_new(unit/"attempts/attempt-001.json",r2.signed("attempt","PASS",{"attempt_id":attempt_id,"selection_order":order,"retry_index":1,"quadrature_attempts":quadrature}))
    r2.write_json_new(unit/"array_inventory.json",r2.signed("array_inventory","PASS",{"selection_order":order,"array_records":rows,"records_digest":r2.canonical_digest(rows)}))
    resource={**resource,"cpu_seconds":resource["cpu_seconds"]+time.process_time()-publish_cpu,"wall_seconds":resource["wall_seconds"]+time.monotonic()-publish_started,"retained_bytes":sum(p.stat().st_size for p in unit.rglob("*") if p.is_file()),"calls":result["calls"],"download_bytes":0}
    r2.write_json_new(unit/"resource_record.json",r2.signed("resource_record","PASS",resource))
    cohort=r2.cohort_by_order(repo)[order];old_path,_=r2.terminal_by_order(repo)[order]
    payload={"selection_order":order,"source_object_id":cohort["source_object_id"],"source_type":"Y2_R2_COMPUTED","shard_id":shard,
             "attempt_id":attempt_id,"condition_id":"clean","failure_code":None,"target_identity":result["target_identity"],
             "source_binding":{"cohort_digest":r2.COHORT_DIGEST,"file_sha256":cohort["file_sha256"]},
             "checkpoint_sha256":engine.MODEL_SHA256,"environment_digest":env_digest,"numeric_contract_digest":engine.NUMERIC_CONTRACT_DIGEST,
             "quadrature_attempts":quadrature,"resource_telemetry":resource,"scientific_result":r2.file_record(unit/"scientific_result.json",repo),
             "array_inventory":r2.file_record(unit/"array_inventory.json",repo),"resource_record":r2.file_record(unit/"resource_record.json",repo),
             "array_records":rows,"attempt_records":[r2.file_record(unit/"attempts/attempt-001.json",repo)],
             "predecessor_terminal":r2.file_record(old_path,repo),"retry_index":1,"scientific_scope_changed":False,"target_metric_tolerance_changed":False}
    terminal=r2.signed("unit_terminal","VALID",payload)
    r2.validate_attempts(repo,terminal,unit)
    closure=r2.validate_quadrature_arrays(terminal,arrays)
    r2.validate_scientific_result(repo,terminal,unit,closure)
    r2.publish_terminal(unit/"completion/unit.pending.json",unit/"completion/unit.json",terminal)
    return terminal


def execute(repo: Path, shard_id: int) -> dict[str,Any]:
    check_environment();r2.validate_implementation(repo)
    root=repo/r2.ROOT;sr=root/f"shards/shard-{shard_id:02d}";started=time.monotonic()
    gate=r2.read_json(root/"startup_gate.json");r2.validate_signed(gate)
    manifest=r2.read_json(sr/"shard_manifest.json");r2.validate_signed(manifest)
    k=gate["payload"]["locked_worker_count"];orders=r2.shard_orders(shard_id,k)
    r2.require(manifest["payload"]["selection_orders"]==orders and manifest["payload"]["shard_id"]==shard_id,"worker manifest ownership drift")
    r2.require(os.environ.get("STAGE_Y2_R2_LAUNCH_AUTHORIZATION")==gate["canonical_digest"],"worker authorization drift")
    expected=gate["payload"]["devices"][shard_id];visible=os.environ.get("CUDA_VISIBLE_DEVICES")
    r2.require(visible==str(expected["physical_index"]),"worker GPU assignment drift")
    import torch
    import torchvision
    import timm
    from . import stage_y_y2_worker as engine
    from .modeling import load_canonical_model
    r2.require(torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=="NVIDIA H20","worker must see one H20")
    torch.cuda.set_device(0);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    y2.activate_recovery_contract()
    identity=y2.build_environment_identity(torch,torchvision,timm,driver=expected["driver"],checkpoint_sha256=engine.MODEL_SHA256,numeric_contract_digest=engine.NUMERIC_CONTRACT_DIGEST)
    env_digest=y2.environment_digest(identity)
    r2.require(identity==expected["environment_identity"] and env_digest==expected["environment_digest"],"worker environment drift")
    worker={"physical_index":expected["physical_index"],"cuda_visible_devices":visible,"gpu_uuid":expected["gpu_uuid"],"pci_bus_id":expected["pci_bus_id"],"device_name":torch.cuda.get_device_name(0),"hostname":socket.gethostname(),"pid":os.getpid(),"environment_identity":identity,"environment_digest":env_digest}
    r2.write_json_new(sr/"worker_identity.json",r2.signed("worker_identity","PASS",worker))
    r2.require(r2.sha_file(repo/engine.MODEL_PATH)==engine.MODEL_SHA256,"checkpoint digest drift")
    fixed_start=time.monotonic();fixed_cpu=time.process_time();torch.cuda.reset_peak_memory_stats(0)
    g0=torch.cuda.Event(enable_timing=True);g1=torch.cuda.Event(enable_timing=True);g0.record()
    model=load_canonical_model(repo/engine.MODEL_PATH,device="cuda:0",unfused_attention=True,expected_state_dict_sha256=engine.MODEL_SHA256,expected_model_name=engine.MODEL_NAME,expected_model_identity=engine.MODEL_LOADER_IDENTITY,expected_canonical_tensor_digest=engine.MODEL_TENSOR_DIGEST).eval()
    g1.record();torch.cuda.synchronize()
    fixed={"gpu_active_seconds":g0.elapsed_time(g1)/1000,"cpu_seconds":time.process_time()-fixed_cpu,"wall_seconds":time.monotonic()-fixed_start,"peak_reserved_gpu_bytes":int(torch.cuda.max_memory_reserved(0)),"retained_bytes":(sr/"worker_identity.json").stat().st_size,"download_bytes":0}
    spec={"execution":r2.read_json(repo/r2.RUN_SPEC)["scientific_contract"]}
    counts=dict.fromkeys(("VALID","FAILED_NUMERICAL","BLOCKED_PROVENANCE","BLOCKED_SOURCE_DRIFT","BLOCKED_ACCESS","BLOCKED_INFRASTRUCTURE"),0)
    cohort=r2.cohort_by_order(repo)
    for order in orders:
        r2.require(not (root/"failure.json").exists(),"global failure stops new work")
        r2.validate_implementation(repo)
        unit=sr/f"units/unit-{order:03d}"
        r2.require(not unit.exists(),"unit output already exists")
        r2.write_json_new(unit/"completion/unit.pending.json",r2.signed("unit_pending","PENDING",{"selection_order":order,"source_object_id":cohort[order]["source_object_id"],"attempt_id":f"unit-{order:03d}-scientific-001","retry_index":1}))
        print(json.dumps({"event":"unit_started","shard_id":shard_id,"selection_order":order}),flush=True)
        ustart=time.monotonic();ucpu=time.process_time();torch.cuda.reset_peak_memory_stats(0)
        g0=torch.cuda.Event(enable_timing=True);g1=torch.cuda.Event(enable_timing=True);g0.record()
        try:
            image=engine._load_selected_image(repo,cohort[order],"cuda:0")
            result,quadrature,arrays=engine._compute_unit(model,image,cohort[order],spec)
            g1.record();torch.cuda.synchronize()
            resource={"gpu_active_seconds":g0.elapsed_time(g1)/1000,"cpu_seconds":time.process_time()-ucpu,"wall_seconds":time.monotonic()-ustart,"peak_allocated_gpu_bytes":int(torch.cuda.max_memory_allocated(0)),"peak_reserved_gpu_bytes":int(torch.cuda.max_memory_reserved(0)),"retained_bytes":0,"download_bytes":0}
            publish_success(repo,unit,order,shard_id,env_digest,result,quadrature,arrays,resource)
            counts["VALID"]+=1
            print(json.dumps({"event":"unit_terminal","selection_order":order,"status":"VALID","shard_id":shard_id}),flush=True)
        except Exception as error:
            code=getattr(error,"code","BLOCKED_INFRASTRUCTURE")
            if code not in counts: code="BLOCKED_PROVENANCE"
            r2.publish_terminal(unit/"completion/unit.pending.json",unit/"completion/unit.json",r2.signed("failure",code,{"phase":f"UNIT_{order}","failure_type":type(error).__name__,"message":str(error),"next_authorization":"NONE"}))
            raise
    terminal=r2.signed("shard_terminal","PASS",{"shard_id":shard_id,"unit_count":len(orders),"terminal_counts":counts,"fixed_initialization":fixed,"worker_wall_seconds":time.monotonic()-started,"completed_unix_ns":time.time_ns()})
    r2.publish_terminal(sr/"completion/shard.pending.json",sr/"completion/shard.json",terminal)
    return {"status":"PASS","shard_id":shard_id,"terminal_counts":counts}


def main() -> int:
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--repo-root",type=Path,required=True);p.add_argument("--shard-id",type=int,required=True);a=p.parse_args()
    try: result=execute(a.repo_root.resolve(),a.shard_id)
    except Exception as error:
        print(json.dumps({"status":getattr(error,"code","BLOCKED_INFRASTRUCTURE"),"error":str(error),"next_authorization":"NONE"}),flush=True)
        return 2
    print(json.dumps(result),flush=True)
    return 0

if __name__=="__main__": raise SystemExit(main())
