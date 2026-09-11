"""One-visible-H20 worker for the 13-unit Stage Y2-R2 missing roster."""
from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import Any

from . import stage_y_y2 as y2
from . import stage_y_y2_r2 as r2
from . import stage_y_y2_worker as engine

if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != y2.REQUIRED_CUBLAS_WORKSPACE_CONFIG:
    raise y2.StageYY2ScheduleError("CUBLAS_WORKSPACE_CONFIG must equal :4096:8 before Torch initialization")


def execute(repo: Path, shard_id: int) -> dict[str,Any]:
    root=repo/r2.ROOT; shard_root=root/"shards"/f"shard-{shard_id:02d}"; started=time.monotonic()
    manifest=r2.read_json(shard_root/"shard_manifest.json"); r2.validate_signed(manifest)
    gate=r2.read_json(root/"startup_gate.json"); r2.validate_signed(gate)
    if os.environ.get("STAGE_Y2_R2_LAUNCH_AUTHORIZATION")!=gate["canonical_digest"]: raise y2.StageYY2ScheduleError("worker launch authorization drift")
    expected=gate["payload"]["devices"][shard_id]; visible=os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible!=str(expected["physical_index"]): raise y2.StageYY2ScheduleError("worker GPU assignment drift")
    import torch
    import torchvision
    import timm
    from .modeling import load_canonical_model
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_name(0)!="NVIDIA H20": raise y2.StageYY2ScheduleError("worker must see exactly one H20")
    torch.cuda.set_device(0); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    identity={
        "python_executable_realpath":os.path.realpath(__import__("sys").executable),"python_version":__import__("platform").python_version(),
        "torch_version":torch.__version__,"torchvision_version":torchvision.__version__,"timm_version":timm.__version__,"cuda_version":torch.version.cuda,"driver":expected["driver"],"device_name":torch.cuda.get_device_name(0),
        "cublas_workspace_config":os.environ.get("CUBLAS_WORKSPACE_CONFIG"),"deterministic_algorithms_enabled":torch.are_deterministic_algorithms_enabled(),"deterministic_warn_only_enabled":torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark":torch.backends.cudnn.benchmark,"cudnn_deterministic":torch.backends.cudnn.deterministic,"cuda_matmul_allow_tf32":torch.backends.cuda.matmul.allow_tf32,"cudnn_allow_tf32":torch.backends.cudnn.allow_tf32,
        "checkpoint_sha256":engine.MODEL_SHA256,"numeric_contract_digest":engine.NUMERIC_CONTRACT_DIGEST,
    }
    env_digest=r2.canonical_digest(identity)
    if env_digest!=expected["environment_digest"]: raise y2.StageYY2ScheduleError("worker environment digest drift")
    worker={"physical_index":expected["physical_index"],"cuda_visible_devices":visible,"gpu_uuid":expected["gpu_uuid"],"pci_bus_id":expected["pci_bus_id"],"device_name":torch.cuda.get_device_name(0),"hostname":socket.gethostname(),"pid":os.getpid(),"environment_identity":identity,"environment_digest":env_digest}
    r2.write_json_new(shard_root/"worker_identity.json",r2.signed("stage_y_y2_r2_worker_identity","PASS",{"artifact_id":"worker_identity",**worker}))
    if r2.sha_file(repo/engine.MODEL_PATH)!=engine.MODEL_SHA256: raise r2.SourceDriftError("checkpoint SHA-256 drift")
    fixed_start=time.monotonic(); fixed_cpu=time.process_time(); torch.cuda.reset_peak_memory_stats(0); ge0=torch.cuda.Event(enable_timing=True);ge1=torch.cuda.Event(enable_timing=True);ge0.record()
    model=load_canonical_model(repo/engine.MODEL_PATH,device="cuda:0",unfused_attention=True,expected_state_dict_sha256=engine.MODEL_SHA256,expected_model_name=engine.MODEL_NAME,expected_model_identity=engine.MODEL_LOADER_IDENTITY,expected_canonical_tensor_digest=engine.MODEL_TENSOR_DIGEST).eval()
    ge1.record();torch.cuda.synchronize(); fixed={"gpu_active_seconds":ge0.elapsed_time(ge1)/1000.0,"cpu_seconds":time.process_time()-fixed_cpu,"wall_seconds":time.monotonic()-fixed_start,"peak_reserved_gpu_bytes":int(torch.cuda.max_memory_reserved(0)),"retained_bytes":(shard_root/"worker_identity.json").stat().st_size,"download_bytes":0}
    spec={"execution":r2.read_json(repo/r2.RUN_SPEC)["scientific_contract"]}; counts={"VALID":0,"FAILED_NUMERICAL":0,"BLOCKED_PROVENANCE":0,"BLOCKED_INFRASTRUCTURE":0}
    cohort=r2.cohort_by_order(repo); old=r2.terminal_by_order(repo)
    for order in manifest["payload"]["selection_orders"]:
        unit_root=shard_root/"units"/f"unit-{order:03d}"; pending=unit_root/"completion/unit.pending.json"
        r2.write_json_new(pending,r2.signed("stage_y_y2_r2_unit_pending","PENDING",{"artifact_id":f"unit-{order:03d}","selection_order":order,"source_object_id":cohort[order]["source_object_id"],"terminal_published":False,"next_authorization":"NONE_WHILE_PENDING"}))
        ustart=time.monotonic();ucpu=time.process_time();torch.cuda.reset_peak_memory_stats(0);g0=torch.cuda.Event(enable_timing=True);g1=torch.cuda.Event(enable_timing=True);g0.record();result={}
        try:
            image=engine._load_selected_image(repo,cohort[order],"cuda:0")
            result,quadrature,arrays=engine._compute_unit(model,image,cohort[order],spec)
            records=[engine._write_array_new(unit_root/"arrays"/f"{name}.npy",value) for name,value in sorted(arrays.items())]
            science=r2.signed("stage_y_y2_r2_scientific_result","VALID",{"artifact_id":f"scientific-{order:03d}","selection_order":order,"source_object_id":cohort[order]["source_object_id"],"target_identity":result["target_identity"],"clean_logit":result["clean_logit"],"baseline_logit":result["baseline_logit"],"delta":result["delta"],"selected_steps":result["selected_steps"],"closure":result["closure"],"ablations":result["ablations"],"deletion_operator":result["deletion_operator"],"calls":result["calls"],"phase_telemetry":result["phase_telemetry"]})
            r2.write_json_new(unit_root/"scientific_result.json",science)
            attempt=r2.signed("stage_y_y2_r2_scientific_attempt","PASS",{"artifact_id":f"attempt-{order:03d}-1","selection_order":order,"retry_index":1,"quadrature_attempts":quadrature})
            r2.write_json_new(unit_root/"attempts/attempt-001.json",attempt)
            status="VALID"; failure=None
        except engine.UnitNumericalError as error:
            status="FAILED_NUMERICAL";failure=str(error);quadrature=error.attempts;records=[];science=None
        except Exception as error:
            status="BLOCKED_INFRASTRUCTURE";failure=f"{type(error).__name__}:{error}";quadrature=[];records=[];science=None
        g1.record();torch.cuda.synchronize();resource={"gpu_active_seconds":g0.elapsed_time(g1)/1000.0,"cpu_seconds":time.process_time()-ucpu,"wall_seconds":time.monotonic()-ustart,"peak_allocated_gpu_bytes":int(torch.cuda.max_memory_allocated(0)),"peak_reserved_gpu_bytes":int(torch.cuda.max_memory_reserved(0)),"retained_bytes":sum(p.stat().st_size for p in unit_root.rglob("*") if p.is_file()),"download_bytes":0,"calls":result.get("calls",{}) if status=="VALID" else {}}
        old_path,old_terminal=old[order]
        payload={"artifact_id":f"unit-{order:03d}","selection_order":order,"source_object_id":cohort[order]["source_object_id"],"source_type":"Y2_R2_COMPUTED","shard_id":shard_id,"condition_id":"clean","failure_code":failure,"target_identity":result["target_identity"] if status=="VALID" else {},"source_binding":{"cohort_digest":"156eab4bd86ff9979abc6acf8b06af3aa5a90e90cd02edef7713539b147ce277","file_sha256":cohort[order]["file_sha256"]},"checkpoint_sha256":engine.MODEL_SHA256,"environment_digest":env_digest,"numeric_contract_digest":engine.NUMERIC_CONTRACT_DIGEST,"quadrature_attempts":quadrature,"resource_telemetry":resource,"scientific_result":r2.file_record(unit_root/"scientific_result.json",repo) if status=="VALID" else None,"array_records":records,"attempt_records":[r2.file_record(unit_root/"attempts/attempt-001.json",repo)] if status=="VALID" else [],"predecessor_terminal":r2.file_record(old_path,repo),"retry_index":1,"scientific_scope_changed":False}
        r2.publish_terminal(pending,unit_root/"completion/unit.json",r2.signed("stage_y_y2_r2_unit_terminal",status,payload));counts[status]+=1
        if status!="VALID": raise r2.NumericalError(f"unit {order} failed: {failure}")
    shard_terminal=r2.signed("stage_y_y2_r2_shard_terminal","PASS",{"artifact_id":f"shard-{shard_id:02d}","shard_id":shard_id,"unit_count":len(manifest["payload"]["selection_orders"]),"terminal_counts":counts,"resource_telemetry":{"fixed_initialization":fixed,"worker_wall_seconds":time.monotonic()-started,"download_bytes":0}})
    r2.publish_terminal(shard_root/"completion/shard.pending.json",shard_root/"completion/shard.json",shard_terminal)
    return {"status":"PASS","shard_id":shard_id,"terminal_counts":counts}


def main() -> int:
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--repo-root",type=Path,required=True);parser.add_argument("--shard-id",type=int,required=True);args=parser.parse_args()
    try: result=execute(args.repo_root.resolve(),args.shard_id)
    except Exception as error: print(json.dumps({"status":getattr(error,"code","BLOCKED_INFRASTRUCTURE"),"error":str(error),"next_authorization":"NONE"},sort_keys=True));return 2
    print(json.dumps(result,sort_keys=True));return 0

if __name__=="__main__": raise SystemExit(main())
