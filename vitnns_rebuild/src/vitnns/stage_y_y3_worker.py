"""One-visible-H20 worker for frozen Y3 non-clean units."""
from __future__ import annotations
import os
if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
    raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must equal :4096:8 before Torch import")

import argparse
import json
import math
import socket
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from . import stage_y_y3 as y3
from . import stage_y_y2_r2 as r2

class UnitNumericalError(y3.NumericalError):
    def __init__(self,message,attempts,calls):super().__init__(message);self.attempts=attempts;self.calls=calls
class UnitInfrastructureError(y3.InfrastructureError):
    def __init__(self,message,attempts,calls):super().__init__(message);self.attempts=attempts;self.calls=calls

def normalize_attempt(value: Mapping[str,Any]) -> dict[str,Any]:
    return {"steps":value.get("steps"),"all_contracts_pass":bool(value.get("all_contracts_pass",False)),"input_sum":value.get("input_sum"),"input_delta":value.get("input_delta"),"input_abs_error":value.get("input_abs_error"),"threshold":value.get("threshold"),"closure_status":value.get("closure_status"),"identity_max_abs_error":value.get("identity_max_abs_error"),"wall_seconds":float(value.get("wall_seconds",0)),"array_digest":value.get("array_digest"),"failure_code":value.get("failure_code"),"failure_phase":value.get("failure_phase")}

def capture_native(model: Any, image: Any, target: int, audit: bool):
    import torch
    from . import stage_y_y1 as y1
    captured=[];handles=[]
    if audit:
        for block in model.blocks:
            handles.append(block.attn.attn_drop.register_forward_hook(lambda _m,_a,out: captured.append(out.detach().cpu().contiguous())))
    try:
        with torch.no_grad():native=model(image)
    finally:
        for handle in reversed(handles):handle.remove()
    routed=y1.native_forward_capture(model,image)
    error=float((native-routed.logits).abs().max())
    if error>=1e-6:raise y3.NumericalError("endpoint parity failure")
    if audit and (len(captured)!=12 or any(x.shape!=(1,12,197,197) for x in captured)):raise y3.ScientificValidityError("raw attention capture roster drift")
    return native,int(native.argmax(1)),float(native[0,target]),error,[x[0].numpy().astype("<f4",copy=False) for x in captured]

def compute_unit(model: Any, image: Any, row: Mapping[str,Any], condition: str, fixed_target: int, clean_logit: float, clean_baseline: float, spec: Mapping[str,Any]):
    import torch
    from . import stage_y_y1 as y1
    from .stage_x_nns_v2 import input_ig
    transformed,metadata=y1.transform_image(image,condition,row["source_object_id"]);audit=row["selection_order"] in y3.AUDIT_OVERLAP
    started=time.monotonic();cpu_started=time.process_time();torch.cuda.reset_peak_memory_stats(transformed.device)
    g0=torch.cuda.Event(enable_timing=True);g1=torch.cuda.Event(enable_timing=True);g0.record()
    native,predicted,target_logit,endpoint_error,raw_attention=capture_native(model,transformed,fixed_target,audit)
    baseline=torch.zeros_like(transformed)
    with torch.no_grad():baseline_logit=float(model(baseline)[0,fixed_target])
    calls={"forward_calls":3,"backward_calls":0,"jvp_calls":0,"jvp_direction_lanes":0,"vjp_calls":0,"quadrature_alpha_batches":0}
    attempts=[];selected_arrays=None;selected_closure=None;selected_integration=None;selected=None
    for steps in spec["scientific_contract"]["quadrature_schedule"]:
        ast=time.monotonic()
        try:candidate=input_ig(model,transformed,baseline,fixed_target,int(steps),int(spec["scientific_contract"]["alpha_chunk"]))
        except Exception as error:
            attempts.append(normalize_attempt({"steps":steps,"all_contracts_pass":False,"wall_seconds":time.monotonic()-ast,"failure_code":f"{type(error).__name__}:{error}"}));raise UnitInfrastructureError(str(error),attempts,calls) from error
        batches=math.ceil(int(steps)/int(spec["scientific_contract"]["alpha_chunk"]));calls["forward_calls"]+=batches;calls["backward_calls"]+=batches
        delta=target_logit-baseline_logit;error=abs(float(candidate.double().sum())-delta);threshold=1e-4+0.005*abs(delta)
        closure=None;identity_error=None;integration=None;arrays=None
        if error<=threshold:
            integration=y1.integrate_routes(model,transformed,baseline,fixed_target,int(steps),alpha_chunk=int(spec["scientific_contract"]["alpha_chunk"]),source_patch_chunk=int(spec["scientific_contract"]["source_patch_jvp_chunk"]),component_vjp_chunk=int(spec["scientific_contract"]["component_vjp_chunk"]))
            calls={k:calls.get(k,0)+integration.calls.get(k,0) for k in set(calls)|set(integration.calls)}
            identity_error=float((integration.attribution-candidate).abs().max());arrays=y1.materialize_arrays(integration);closure=y1.validate_closure(arrays,delta)
        passed=error<=threshold and closure is not None and closure["status"]=="PASS" and identity_error is not None and identity_error<=1e-5
        attempts.append(normalize_attempt({"steps":steps,"all_contracts_pass":passed,"input_sum":float(candidate.double().sum()),"input_delta":delta,"input_abs_error":error,"threshold":threshold,"closure_status":None if closure is None else closure["status"],"identity_max_abs_error":identity_error,"wall_seconds":time.monotonic()-ast,"array_digest":y1.array_digest(candidate)}))
        if passed:selected=int(steps);selected_arrays=arrays;selected_closure=closure;selected_integration=integration;break
    if selected is None or selected_arrays is None or selected_integration is None:raise UnitNumericalError("Y3 unit nonconverged at 8192",attempts,calls)
    head_maps={b:selected_arrays[f"b{b:02d}_first_heads_cell"] for b in range(12)};attention_maps={b:selected_arrays[f"b{b:02d}_first_attention_cell"] for b in range(12)}
    ablations,acalls=y1.run_head_ablations(model,transformed,fixed_target,clean_logit,clean_baseline,head_maps,row["source_object_id"]);deletion,dcalls=y1.run_deletion_operator(model,transformed,fixed_target,clean_logit,clean_baseline,attention_maps,row["source_object_id"],int(spec["scientific_contract"]["deletion_forward_batch"]))
    for extra in (acalls,dcalls):calls={k:calls.get(k,0)+extra.get(k,0) for k in set(calls)|set(extra)}
    mask,w=build_inverse_arrays(transformed,metadata)
    selected_arrays["valid_mask"]=mask;selected_arrays["fully_valid_patch_mask"]=mask.reshape(14,16,14,16).all((1,3)).reshape(196);selected_arrays["patch_inverse_operator"]=w
    if audit:
        pixel=selected_arrays["input_attribution"].astype(np.float64).sum(0).astype("<f4");selected_arrays["positive_input_map"]=np.maximum(pixel,0).astype("<f4");selected_arrays["negative_input_map"]=np.minimum(pixel,0).astype("<f4")
        for name in y3.AUDIT_SIGNED_NAMES:
            value=selected_arrays[name];selected_arrays["positive_"+name]=np.maximum(value,0).astype("<f4");selected_arrays["negative_"+name]=np.minimum(value,0).astype("<f4")
        for b,value in enumerate(raw_attention):selected_arrays[f"raw_attention_b{b:02d}"]=value
    g1.record();torch.cuda.synchronize()
    resource={"gpu_active_seconds":g0.elapsed_time(g1)/1000,"cpu_seconds":time.process_time()-cpu_started,"wall_seconds":time.monotonic()-started,"peak_allocated_gpu_bytes":int(torch.cuda.max_memory_allocated(0)),"peak_reserved_gpu_bytes":int(torch.cuda.max_memory_reserved(0)),"retained_bytes":0,"download_bytes":0,"calls":calls}
    legacy={"target_identity":{"class_id":fixed_target,"rule":"clean predicted-class raw logit"},"clean_logit":target_logit,"baseline_logit":baseline_logit,"delta":target_logit-baseline_logit,"selected_steps":selected,"closure":selected_closure,"ablations":ablations,"deletion_operator":deletion,"calls":calls,"phase_telemetry":selected_integration.phase_telemetry}
    prediction={"fixed_target":legacy["target_identity"],"clean_target_logit":clean_logit,"clean_baseline_logit":clean_baseline,"transformed_predicted_class":predicted,"transformed_fixed_target_logit":target_logit,"baseline_target_logit":baseline_logit,"transformed_target_delta":target_logit-baseline_logit,"prediction_changed":predicted!=fixed_target,"target_reselected":False,"clean_forward_executed":False}
    return legacy,prediction,metadata,endpoint_error,attempts,selected_arrays,resource

def build_inverse_arrays(transformed: Any, metadata: Mapping[str,Any]):
    import torch
    from . import stage_y_y1 as y1
    seed=torch.ones((224,224),device=transformed.device,dtype=torch.float32);_,mask=y1.inverse_map(seed,metadata)
    w=np.zeros((196,196),dtype=np.float32)
    if metadata["inverse"]=="identity":w[:]=np.eye(196,dtype=np.float32)
    else:
        for patch in range(196):
            py,px=divmod(patch,14);basis=torch.zeros((224,224),device=transformed.device,dtype=torch.float32);basis[py*16:(py+1)*16,px*16:(px+1)*16]=1/256
            restored,_=y1.inverse_map(basis,metadata);w[:,patch]=restored.astype(np.float64).reshape(14,16,14,16).sum((1,3)).reshape(196).astype(np.float32)
    if not np.isfinite(w).all():raise y3.ScientificValidityError("nonfinite patch inverse operator")
    return np.ascontiguousarray(mask,dtype=np.bool_),np.ascontiguousarray(w,dtype="<f4")

def execute(repo: Path, shard_id: int) -> dict[str,Any]:
    import torch,torchvision,timm
    from . import stage_y_y2_worker as engine
    from .modeling import load_canonical_model
    root=repo/y3.ROOT;y3.verify_implementation(repo);gate=y3.read(root/"startup_gate.json");y3.validate_document(gate);k=gate["payload"]["locked_worker_count"]
    y3.require(0<=shard_id<k,"invalid shard",y3.ScheduleError);manifest=y3.read(root/f"shards/shard-{shard_id:02d}/shard_manifest.json");y3.validate_document(manifest)
    expected=gate["payload"]["devices"][shard_id];y3.require(os.environ.get("STAGE_Y3_LAUNCH_AUTHORIZATION")==gate["canonical_digest"] and os.environ.get("CUDA_VISIBLE_DEVICES")==str(expected["physical_index"]),"worker launch authorization drift",y3.ScheduleError)
    y3.require(torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=="NVIDIA H20","worker must see one H20",y3.ScheduleError);torch.cuda.set_device(0);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    identity=r2.y2.build_environment_identity(torch,torchvision,timm,driver=expected["driver"],checkpoint_sha256=r2.CHECKPOINT_SHA,numeric_contract_digest=r2.NUMERIC_CONTRACT);digest=r2.y2.environment_digest(identity);y3.require(identity==expected["environment_identity"] and digest==expected["environment_digest"],"worker environment identity drift",y3.ScheduleError)
    sr=root/f"shards/shard-{shard_id:02d}";worker={"physical_index":expected["physical_index"],"cuda_visible_devices":os.environ["CUDA_VISIBLE_DEVICES"],"gpu_uuid":expected["gpu_uuid"],"pci_bus_id":expected["pci_bus_id"],"device_name":"NVIDIA H20","hostname":socket.gethostname(),"pid":os.getpid(),"environment_identity":identity,"environment_digest":digest};y3.write(sr/"worker_identity.json","worker_identity",worker)
    fixed_start=time.monotonic();fixed_cpu=time.process_time();torch.cuda.reset_peak_memory_stats(0);g0=torch.cuda.Event(enable_timing=True);g1=torch.cuda.Event(enable_timing=True);g0.record();model=load_canonical_model(repo/engine.MODEL_PATH,device="cuda:0",unfused_attention=True,expected_state_dict_sha256=engine.MODEL_SHA256,expected_model_name=engine.MODEL_NAME,expected_model_identity=engine.MODEL_LOADER_IDENTITY,expected_canonical_tensor_digest=engine.MODEL_TENSOR_DIGEST).eval();g1.record();torch.cuda.synchronize()
    zero_calls=dict.fromkeys(r2.CALL_KEYS,0);fixed={"gpu_active_seconds":g0.elapsed_time(g1)/1000,"cpu_seconds":time.process_time()-fixed_cpu,"wall_seconds":time.monotonic()-fixed_start,"peak_allocated_gpu_bytes":int(torch.cuda.max_memory_allocated(0)),"peak_reserved_gpu_bytes":int(torch.cuda.max_memory_reserved(0)),"retained_bytes":0,"download_bytes":0,"calls":zero_calls}
    spec=y3.read(repo/y3.SPEC);clean={x["subset_rank"]:x for x in y3.read_clean_manifest(repo)};counts=dict.fromkeys(y3.TERMINAL_STATES,0);started=time.monotonic();cpu_total=fixed["cpu_seconds"];gpu_total=fixed["gpu_active_seconds"];stage_block=None
    r3_resources=[]
    for inherited_index in y3.INHERITED_INDICES:
        predecessor=next((repo/y3.R3_ROOT).glob(f"shards/shard-*/units/unit-{inherited_index:03d}/resource.json"));r3_resources.append(y3.read(predecessor)["payload"])
    prior_gpu=sum(x["gpu_active_seconds"] for x in r3_resources)+y3.R3_INTERRUPTED_GPU_RESERVE_SECONDS;prior_cpu=sum(x["cpu_seconds"] for x in r3_resources)+y3.R3_INTERRUPTED_CPU_RESERVE_SECONDS
    prior_retained=sum(p.stat().st_size for historical in (repo/"evidence/stage_y/y3_perturbation_formal_20260910",repo/y3.R3_ROOT) for p in historical.rglob("*") if p.is_file())
    for coord in manifest["payload"]["records"]:
        unit=sr/f"units/unit-{coord['unit_index']:03d}";attempt_id=f"unit-{coord['unit_index']:03d}-scientific-001";y3.write(unit/"completion/unit.pending.json","unit_pending",{**{x:coord[x] for x in ("unit_index","subset_rank","selection_order","condition_index","condition_id","source_object_id")},"attempt_id":attempt_id},"PENDING")
        status="VALID";failure=None;records=[];attempts=[];science_record=None;transform_record=None;prediction_record=None;inventory_record=None;resource=None
        try:
            if stage_block is not None:raise stage_block("prior blocking terminal stops scientific starts on this locked shard")
            retained=prior_retained+sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
            available_gpu=y3.CAPS["h20_gpu_hours"]*3600-prior_gpu;available_cpu=y3.CAPS["cpu_hours"]*3600-prior_cpu
            if gpu_total>available_gpu/k or cpu_total>available_cpu/k or (y3.R3_INTERRUPTED_ELAPSED_UPPER_BOUND_SECONDS+time.monotonic()-started)/3600>y3.CAPS["wall_hours"] or retained>y3.CAPS["retained_storage_gib"]*2**30 or int(torch.cuda.max_memory_reserved(0))>y3.CAPS["peak_reserved_gpu_gib_per_process"]*2**30:raise y3.BudgetError("Y3 hard cap exhausted before unit start")
            c=clean[coord["subset_rank"]];image=engine._load_selected_image(repo,{"tar_member_name":coord["source_object_id"].split(":",1)[1].split("/")[-1],"file_sha256":coord["source_jpeg_sha256"] if "source_jpeg_sha256" in coord else y3.read(repo/r2.Y2_ROOT/"cohort_registry.json")["payload"]["records"][coord["selection_order"]]["file_sha256"]},"cuda:0")
            legacy,prediction,metadata,endpoint,attempts,arrays,resource=compute_unit(model,image,coord,coord["condition_id"],c["target_identity"]["class_id"],c["clean_target_logit"],c["clean_baseline_logit"],spec)
            for name,value in sorted(arrays.items()):records.append(y3.write_array(unit/"arrays"/(name+".npy"),value,repo))
            resource["retained_bytes"]=sum(x["bytes"] for x in records);gpu_total+=resource["gpu_active_seconds"];cpu_total+=resource["cpu_seconds"]
            registry=next(x for x in y3.transform_records(repo) if x["id"]==coord["condition_id"]);mask=arrays["valid_mask"]
            transform={"condition_id":coord["condition_id"],"registry_record":registry,"adapter_metadata":metadata,"source_coordinate_convention":"zero-based pixel-center y,x on clean 224x224","transformed_coordinate_convention":"zero-based pixel-center y,x on transformed 224x224","valid_threshold":0.999999,"noise_seed":metadata.get("seed"),"valid_domain_coverage":float(mask.mean()),"inverse_operator_contract":"W[clean_patch,transformed_patch]; W @ T_transformed @ W.T; W @ C_transformed; fully valid pairs only"}
            y3.write(unit/"transform.json","transform",transform);transform_record=y3.file_record(unit/"transform.json",repo);y3.write(unit/"prediction.json","prediction",prediction);prediction_record=y3.file_record(unit/"prediction.json",repo);y3.write(unit/"resource.json","resource",resource);y3.write(unit/"array_inventory.json","array_inventory",{"unit_index":coord["unit_index"],"records":records,"record_count":len(records),"records_digest":y3.canonical(records)});inventory_record=y3.file_record(unit/"array_inventory.json",repo)
            science={"y2_compatible":legacy,"transformed_prediction":prediction,"endpoint_parity_max_abs_error":endpoint,"endpoint_parity_pass":endpoint<1e-6,"transform":transform,"inverse_materialization":{"stored_duplicate_inverse_arrays":False,"operator_name":"patch_inverse_operator","matrix_formula":"W @ T_transformed @ W.T","cls_formula":"W @ C_transformed","recompute_dtype":"float64_then_float32_little_endian_c_contiguous","fully_valid_pairs_only":True},"raw_attention_role":"descriptive structural reference only","audit_overlap":coord["selection_order"] in y3.AUDIT_OVERLAP};y3.write(unit/"scientific_result.json","scientific_result",science,"VALID");science_record=y3.file_record(unit/"scientific_result.json",repo)
        except y3.Y3Error as error:status=error.code;failure=str(error);attempts=getattr(error,"attempts",attempts)
        except Exception as error:status="BLOCKED_INFRASTRUCTURE";failure=f"{type(error).__name__}:{error}"
        if resource is None:resource={"gpu_active_seconds":0,"cpu_seconds":0,"wall_seconds":0,"peak_allocated_gpu_bytes":int(torch.cuda.max_memory_allocated(0)),"peak_reserved_gpu_bytes":int(torch.cuda.max_memory_reserved(0)),"retained_bytes":sum(p.stat().st_size for p in unit.rglob("*") if p.is_file()),"download_bytes":0,"calls":zero_calls};y3.write(unit/"resource.json","resource",resource)
        if not attempts:attempts=[normalize_attempt({"all_contracts_pass":False,"wall_seconds":0,"failure_code":failure,"failure_phase":"unit"})]
        for i,attempt in enumerate(attempts):y3.write(unit/f"attempts/attempt-{i:03d}.json","attempt",{"unit_index":coord["unit_index"],"attempt_id":attempt_id,"attempt_index":i,"record":attempt},status if status!="VALID" else "PASS")
        attempt_records=[y3.file_record(p,repo) for p in sorted((unit/"attempts").glob("*.json"))];cohort=y3.read(repo/r2.Y2_ROOT/"cohort_registry.json")["payload"]["records"][coord["selection_order"]]
        payload={**{x:coord[x] for x in ("unit_index","subset_rank","selection_order","condition_index","condition_id","source_object_id")},"shard_id":shard_id,"attempt_id":attempt_id,"failure_code":failure,"fixed_target":coord["fixed_target"],"source_terminal":coord["clean_source_terminal"],"source_jpeg_sha256":cohort["file_sha256"],"checkpoint_sha256":r2.CHECKPOINT_SHA,"environment_digest":digest,"transform_record":transform_record,"prediction_record":prediction_record,"scientific_result":science_record,"resource_record":y3.file_record(unit/"resource.json",repo),"array_inventory":inventory_record,"array_records":records,"attempt_records":attempt_records,"quadrature_attempts":attempts,"clean_rerun":False,"retry_count":1 if coord["unit_index"] in y3.INTERRUPTED_INDICES else 0,"evidence_mode":"NEW_COMPUTE","predecessor_terminal":None}
        y3.publish_terminal(unit/"completion/unit.pending.json",unit/"completion/unit.json",y3.signed("unit_terminal",payload,status));counts[status]+=1;print(json.dumps({"event":"unit_terminal","unit_index":coord["unit_index"],"condition":coord["condition_id"],"status":status,"shard_id":shard_id}),flush=True)
        if status in y3.BLOCKING_STATES and stage_block is None:stage_block={"BLOCKED_BUDGET":y3.BudgetError,"BLOCKED_RESOURCE_SCHEDULE":y3.ScheduleError,"BLOCKED_PROVENANCE":y3.ProvenanceError,"BLOCKED_ACCESS":y3.AccessError,"BLOCKED_SCIENTIFIC_VALIDITY":y3.ScientificValidityError}.get(status,y3.InfrastructureError)
    terminal=y3.signed("shard_terminal",{"shard_id":shard_id,"shard_count":k,"expected_unit_count":manifest["payload"]["unit_count"],"terminal_count":sum(counts.values()),"terminal_counts":counts,"fixed_initialization_resource":fixed,"worker_wall_seconds":time.monotonic()-started,"completed_unix_ns":time.time_ns()},"PASS" if sum(counts.values())==manifest["payload"]["unit_count"] and not any(counts[x] for x in y3.BLOCKING_STATES) else next((x for x in y3.BLOCKING_STATES if counts[x]),"BLOCKED_INFRASTRUCTURE"))
    y3.publish_terminal(sr/"completion/shard.pending.json",sr/"completion/shard.json",terminal);return {"status":terminal["status"],"shard_id":shard_id,"terminal_counts":counts}

def main() -> int:
    p=argparse.ArgumentParser();p.add_argument("--repo-root",type=Path,required=True);p.add_argument("--shard-id",type=int,required=True);a=p.parse_args()
    try:result=execute(a.repo_root.resolve(),a.shard_id)
    except Exception as error:print(json.dumps({"status":getattr(error,"code","BLOCKED_INFRASTRUCTURE"),"error":str(error),"next_authorization":"NONE"}),flush=True);return 2
    print(json.dumps(result,sort_keys=True),flush=True);return 0
if __name__=="__main__":raise SystemExit(main())
