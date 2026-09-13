"""Single-GPU Y3-R5 worker; CUBLAS guard precedes Torch import."""
from __future__ import annotations
import os
if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must equal :4096:8 before Torch import")
import argparse,json,socket,time
from pathlib import Path
from . import stage_y_y3 as y3
from . import stage_y_y3_r5 as r5

def execute(repo:Path)->dict:
    import torch,torchvision,timm
    from . import stage_y_y2_r2 as r2
    from . import stage_y_y2_worker as engine
    from .modeling import load_canonical_model
    from .stage_y_y3_worker import compute_unit,normalize_attempt
    root=repo/r5.ROOT;r5.verify_implementation(repo);gate=r5.read(root/"startup_gate.json");r5.validate_document(gate)
    r5.require(os.environ.get("CUDA_VISIBLE_DEVICES")=="1" and os.environ.get("STAGE_Y3_R5_AUTHORIZATION")==gate["canonical_digest"],"worker authorization drift",r5.ScheduleError)
    expected=gate["payload"]["devices"][0];r5.require(torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=="NVIDIA H20","worker must see one H20",r5.ScheduleError)
    torch.cuda.set_device(0);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    identity=r2.y2.build_environment_identity(torch,torchvision,timm,driver=expected["driver"],checkpoint_sha256=r2.CHECKPOINT_SHA,numeric_contract_digest=r2.NUMERIC_CONTRACT);digest=r2.y2.environment_digest(identity)
    r5.require(identity==expected["environment_identity"] and digest==expected["environment_digest"],"worker environment drift",r5.ScheduleError)
    sr=root/"shards/shard-01";r5.write(sr/"worker_identity.json","worker_identity",{"physical_index":1,"cuda_visible_devices":"1","gpu_uuid":expected["gpu_uuid"],"pci_bus_id":expected["pci_bus_id"],"device_name":"NVIDIA H20","hostname":socket.gethostname(),"pid":os.getpid(),"environment_identity":identity,"environment_digest":digest})
    fixed_start=time.monotonic();fixed_cpu=time.process_time();torch.cuda.reset_peak_memory_stats(0);g0=torch.cuda.Event(enable_timing=True);g1=torch.cuda.Event(enable_timing=True);g0.record();model=load_canonical_model(repo/engine.MODEL_PATH,device="cuda:0",unfused_attention=True,expected_state_dict_sha256=engine.MODEL_SHA256,expected_model_name=engine.MODEL_NAME,expected_model_identity=engine.MODEL_LOADER_IDENTITY,expected_canonical_tensor_digest=engine.MODEL_TENSOR_DIGEST).eval();g1.record();torch.cuda.synchronize()
    zero=dict.fromkeys(r2.CALL_KEYS,0);fixed={"gpu_active_seconds":g0.elapsed_time(g1)/1000,"cpu_seconds":time.process_time()-fixed_cpu,"wall_seconds":time.monotonic()-fixed_start,"peak_allocated_gpu_bytes":int(torch.cuda.max_memory_allocated(0)),"peak_reserved_gpu_bytes":int(torch.cuda.max_memory_reserved(0)),"retained_bytes":0,"download_bytes":0,"calls":zero}
    source_ledger=y3.read(repo/r5.SOURCE_ROOT/"resource_ledger.json")["payload"];admission=r5.read(root/"budget_admission.json")["payload"];maximum=admission["maximum_unit"];coords=r5.read(root/"coordinate_manifest.json")["payload"]["records"];clean={x["subset_rank"]:x for x in y3.read(repo/r5.SOURCE_ROOT/"clean_reference_manifest.json")["payload"]["records"]};spec=y3.read(repo/y3.SPEC)
    cohort=y3.read(repo/r2.Y2_ROOT/"cohort_registry.json")["payload"]["records"];gpu=fixed["gpu_active_seconds"];cpu=fixed["cpu_seconds"];started=time.monotonic();counts=dict.fromkeys(y3.TERMINAL_STATES,0);stage_block=None
    for coord in coords:
        unit=sr/f"units/unit-{coord['unit_index']:03d}";attempt_id=f"unit-{coord['unit_index']:03d}-r5-scientific-001";r5.write(unit/"completion/unit.pending.json","unit_pending",{**{k:coord[k] for k in ("unit_index","subset_rank","selection_order","condition_index","condition_id","source_object_id")},"attempt_id":attempt_id},"PENDING")
        status="VALID";failure=None;records=[];attempts=[];science_record=transform_record=prediction_record=inventory_record=None;resource=None
        try:
            if stage_block is not None:raise stage_block("prior blocking terminal stops new scientific starts")
            retained=source_ledger["retained_bytes"]+sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
            combined_gpu=source_ledger["gpu_active_seconds"]+gpu;combined_cpu=source_ledger["cpu_seconds"]+cpu;combined_wall=source_ledger["stage_wall_seconds"]+(time.monotonic()-started)
            if combined_gpu+maximum["gpu_active_seconds"]>=y3.CAPS["h20_gpu_hours"]*3600 or combined_cpu+maximum["cpu_seconds"]>=y3.CAPS["cpu_hours"]*3600 or combined_wall+maximum["wall_seconds"]>=y3.CAPS["wall_hours"]*3600 or retained+maximum["retained_bytes"]>=y3.CAPS["retained_storage_gib"]*2**30 or max(int(torch.cuda.max_memory_reserved(0)),maximum["peak_reserved_gpu_bytes"])>=y3.CAPS["peak_reserved_gpu_gib_per_process"]*2**30:raise r5.BudgetError("global cumulative Y3 cap cannot admit one conservative R5 unit")
            row=cohort[coord["selection_order"]];image=engine._load_selected_image(repo,{"tar_member_name":coord["source_object_id"].split(":",1)[1].split("/")[-1],"file_sha256":row["file_sha256"]},"cuda:0");c=clean[coord["subset_rank"]]
            legacy,prediction,metadata,endpoint,attempts,arrays,resource=compute_unit(model,image,coord,coord["condition_id"],c["target_identity"]["class_id"],c["clean_target_logit"],c["clean_baseline_logit"],spec)
            for name,value in sorted(arrays.items()):records.append(y3.write_array(unit/"arrays"/(name+".npy"),value,repo))
            resource["retained_bytes"]=sum(x["bytes"] for x in records);gpu+=resource["gpu_active_seconds"];cpu+=resource["cpu_seconds"]
            registry=next(x for x in y3.transform_records(repo) if x["id"]==coord["condition_id"]);mask=arrays["valid_mask"];transform={"condition_id":coord["condition_id"],"registry_record":registry,"adapter_metadata":metadata,"source_coordinate_convention":"zero-based pixel-center y,x on clean 224x224","transformed_coordinate_convention":"zero-based pixel-center y,x on transformed 224x224","valid_threshold":0.999999,"noise_seed":metadata.get("seed"),"valid_domain_coverage":float(mask.mean()),"inverse_operator_contract":"W[clean_patch,transformed_patch]; W @ T_transformed @ W.T; W @ C_transformed; fully valid pairs only"}
            r5.write(unit/"transform.json","transform",transform);transform_record=r5.file_record(unit/"transform.json",repo);r5.write(unit/"prediction.json","prediction",prediction);prediction_record=r5.file_record(unit/"prediction.json",repo);r5.write(unit/"resource.json","resource",resource);r5.write(unit/"array_inventory.json","array_inventory",{"unit_index":coord["unit_index"],"records":records,"record_count":len(records),"records_digest":r5.canonical(records)});inventory_record=r5.file_record(unit/"array_inventory.json",repo)
            science={"y2_compatible":legacy,"transformed_prediction":prediction,"endpoint_parity_max_abs_error":endpoint,"endpoint_parity_pass":endpoint<1e-6,"transform":transform,"inverse_materialization":{"stored_duplicate_inverse_arrays":False,"operator_name":"patch_inverse_operator","matrix_formula":"W @ T_transformed @ W.T","cls_formula":"W @ C_transformed","recompute_dtype":"float64_then_float32_little_endian_c_contiguous","fully_valid_pairs_only":True},"raw_attention_role":"descriptive structural reference only","audit_overlap":coord["selection_order"] in y3.AUDIT_OVERLAP};r5.write(unit/"scientific_result.json","scientific_result",science);science_record=r5.file_record(unit/"scientific_result.json",repo)
        except (r5.R5Error,y3.Y3Error) as error:status=getattr(error,"code","BLOCKED_INFRASTRUCTURE");failure=str(error);attempts=getattr(error,"attempts",attempts)
        except Exception as error:status="BLOCKED_INFRASTRUCTURE";failure=f"{type(error).__name__}:{error}"
        if resource is None:resource={"gpu_active_seconds":0,"cpu_seconds":0,"wall_seconds":0,"peak_allocated_gpu_bytes":int(torch.cuda.max_memory_allocated(0)),"peak_reserved_gpu_bytes":int(torch.cuda.max_memory_reserved(0)),"retained_bytes":sum(p.stat().st_size for p in unit.rglob("*") if p.is_file()),"download_bytes":0,"calls":zero};r5.write(unit/"resource.json","resource",resource)
        if not attempts:attempts=[normalize_attempt({"all_contracts_pass":False,"wall_seconds":0,"failure_code":failure,"failure_phase":"unit"})]
        for i,attempt in enumerate(attempts):r5.write(unit/f"attempts/attempt-{i:03d}.json","attempt",{"unit_index":coord["unit_index"],"attempt_id":attempt_id,"attempt_index":i,"record":attempt},status if status!="VALID" else "PASS")
        attempt_records=[r5.file_record(p,repo) for p in sorted((unit/"attempts").glob("*.json"))];row=cohort[coord["selection_order"]];payload={**{k:coord[k] for k in ("unit_index","subset_rank","selection_order","condition_index","condition_id","source_object_id")},"shard_id":1,"attempt_id":attempt_id,"failure_code":failure,"fixed_target":coord["fixed_target"],"source_terminal":coord["source_terminal"],"source_jpeg_sha256":row["file_sha256"],"checkpoint_sha256":r2.CHECKPOINT_SHA,"environment_digest":digest,"transform_record":transform_record,"prediction_record":prediction_record,"scientific_result":science_record,"resource_record":r5.file_record(unit/"resource.json",repo),"array_inventory":inventory_record,"array_records":records,"attempt_records":attempt_records,"quadrature_attempts":attempts,"clean_rerun":False,"retry_count":0,"evidence_mode":"NEW_COMPUTE","predecessor_terminal":None}
        r5.publish_terminal(unit/"completion/unit.pending.json",unit/"completion/unit.json",r5.signed("unit_terminal",payload,status));counts[status]+=1;print(json.dumps({"event":"unit_terminal","unit_index":coord["unit_index"],"status":status}),flush=True)
        if status in y3.BLOCKING_STATES and stage_block is None:stage_block={"BLOCKED_BUDGET":r5.BudgetError,"BLOCKED_PROVENANCE":r5.ProvenanceError,"BLOCKED_RESOURCE_SCHEDULE":r5.ScheduleError}.get(status,r5.InfrastructureError)
    terminal=r5.signed("shard_terminal",{"logical_shard_id":1,"expected_unit_count":5,"terminal_count":5,"terminal_counts":counts,"fixed_initialization_resource":fixed,"worker_wall_seconds":time.monotonic()-started,"completed_unix_ns":time.time_ns()},"PASS" if counts["VALID"]==5 else next((x for x in y3.BLOCKING_STATES if counts[x]),"BLOCKED_INFRASTRUCTURE"));r5.publish_terminal(sr/"completion/shard.pending.json",sr/"completion/shard.json",terminal);return {"status":terminal["status"],"terminal_counts":counts}
def main()->int:
    p=argparse.ArgumentParser();p.add_argument("--repo-root",type=Path,required=True);a=p.parse_args()
    try:result=execute(a.repo_root.resolve())
    except Exception as error:print(json.dumps({"status":getattr(error,"code","BLOCKED_INFRASTRUCTURE"),"error":str(error),"next_authorization":"NONE"}));return 2
    print(json.dumps(result,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
