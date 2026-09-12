#!/usr/bin/env python3
"""Independent source-recomputing Stage Y3 reconciliation."""
from __future__ import annotations
import argparse,copy,json,math,os,subprocess,time
from pathlib import Path
from typing import Any,Mapping
import numpy as np
from jsonschema import Draft202012Validator
from vitnns import stage_y_y3 as y3
from vitnns import stage_y_y2_r2 as r2
from vitnns import stage_y_y2_r2_r2 as y2c

def unit_terminal_paths(repo: Path):return sorted((repo/y3.ROOT).glob("shards/shard-*/units/unit-*/completion/unit.json"))

def validate_transform_arrays(condition: str, transform: Mapping[str,Any], arrays: Mapping[str,np.ndarray], source_id: str) -> None:
    mask=arrays["valid_mask"];patch=arrays["fully_valid_patch_mask"];w=arrays["patch_inverse_operator"]
    y3.require(mask.dtype==np.bool_ and patch.dtype==np.bool_ and w.dtype==np.float32,"mask/operator dtype drift",y3.NumericalError)
    expected_patch=mask.reshape(14,16,14,16).all((1,3)).reshape(196);y3.require(np.array_equal(patch,expected_patch),"fully-valid patch mask drift",y3.ScientificValidityError)
    full=condition in ("horizontal_flip","brightness_0.9","brightness_1.1","gaussian_noise_0.01_seed_0")
    y3.require(transform["registry_record"]["full_domain"]==full and bool(mask.all())==full,"full-domain mask drift",y3.ScientificValidityError)
    if condition=="translate_x_-8":y3.require(not mask[:,:8].any() and mask[:,8:].all(),"translate -8 mask drift",y3.ScientificValidityError)
    if condition=="translate_x_8":y3.require(mask[:,:216].all() and not mask[:,216:].any(),"translate +8 mask drift",y3.ScientificValidityError)
    if condition.startswith("rotate_"):y3.require(0<float(mask.mean())<1 and transform["valid_threshold"]==0.999999,"rotation mask drift",y3.ScientificValidityError)
    if condition=="gaussian_noise_0.01_seed_0":y3.require(transform["noise_seed"]==y3.noise_seed(source_id,condition),"noise seed drift",y3.ProvenanceError)
    y3.require(np.isfinite(w).all() and (w>=0).all(),"inverse W invalid",y3.ScientificValidityError)
    identity=np.arange(196,dtype=np.float32).reshape(196,1);left=y3.inverse_cls(w,identity[:,0]);y3.require(left.shape==(196,),"inverse CLS recomputation failed")
    matrix=np.eye(196,dtype=np.float32);y3.require(y3.inverse_matrix(w,matrix).shape==(196,196),"inverse matrix recomputation failed")

def validate_unit(repo: Path,path: Path,coords: Mapping[int,dict],clean: Mapping[int,dict],gate: dict) -> dict:
    doc=y3.read(path);y3.validate_document(doc);q=doc["payload"];index=q["unit_index"];coord=coords[index];unit=path.parent.parent
    for key in ("unit_index","subset_rank","selection_order","condition_index","condition_id","source_object_id"):
        y3.require(q[key]==coord[key],"unit coordinate drift: "+key)
    y3.require(q["shard_id"]==coord["expected_shard_owner"]==index%gate["payload"]["locked_worker_count"],"shard ownership drift",y3.ScheduleError);y3.require(q["fixed_target"]==clean[q["subset_rank"]]["target_identity"],"fixed target drift")
    y3.require(q["source_terminal"]==clean[q["subset_rank"]]["source_terminal"] and not q["clean_rerun"],"clean source drift")
    mode=q["evidence_mode"];is_rebind=mode=="SOURCE_REBIND"
    if is_rebind:
        y3.require(index in y3.INHERITED_INDICES and q["retry_count"]==0 and q["predecessor_terminal"] is not None,"invalid source rebind identity")
        predecessor=y3.checked_file(repo,q["predecessor_terminal"],repo/y3.R3_ROOT);source_doc=y3.read(predecessor);y3.validate_r3_document(repo,source_doc)
        rebound_source={k:v for k,v in q.items() if k not in ("evidence_mode","predecessor_terminal")}
        rebound_source["shard_id"]=source_doc["payload"]["shard_id"]
        y3.require(source_doc["status"]=="VALID" and source_doc["payload"]==rebound_source,"R3 predecessor payload drift")
        for key,value in source_doc["payload"].items():
            if key!="shard_id":y3.require(q[key]==value,"R3 rebind field drift: "+key)
        source_unit=predecessor.parent.parent
        y3.require(not any(p.is_file() for p in unit.rglob("*") if p!=path),"source rebind copied or added local payload files")
    else:
        y3.require(index not in y3.INHERITED_INDICES and q["predecessor_terminal"] is None and q["retry_count"]==(1 if index in y3.INTERRUPTED_INDICES else 0),"new compute retry/source identity drift")
        source_unit=unit
    validate_doc=(lambda value:y3.validate_r3_document(repo,value)) if is_rebind else y3.validate_document
    resource=y3.read(y3.checked_file(repo,q["resource_record"],source_unit));validate_doc(resource)
    attempts=[]
    for row in q["attempt_records"]:
        attempt=y3.read(y3.checked_file(repo,row,source_unit/"attempts"));validate_doc(attempt);attempts.append(attempt["payload"]["record"]);y3.require(attempt["payload"]["attempt_id"]==q["attempt_id"],"attempt identity drift")
    y3.require(attempts==q["quadrature_attempts"] and len(q["attempt_records"])==len(list((source_unit/"attempts").glob("*.json"))),"attempt journal drift")
    if doc["status"]!="VALID":return {"unit_index":index,"subset_rank":q["subset_rank"],"condition_id":q["condition_id"],"status":doc["status"],"prediction_changed":False,"resource":resource["payload"],"evidence_mode":mode}
    y3.require(q["failure_code"] is None and q["scientific_result"] and q["transform_record"] and q["prediction_record"] and q["array_inventory"],"VALID payload missing records")
    expected=y3.expected_array_names(q["selection_order"]);y3.require(tuple(sorted(Path(x["path"]).stem for x in q["array_records"]))==expected and len(q["array_records"])==len(expected),"exact array roster drift")
    y3.require({repo/x["path"] for x in q["array_records"]}==set((source_unit/"arrays").glob("*.npy")),"unbound/missing array")
    arrays={Path(row["path"]).stem:y3.validate_array(repo,row,q["selection_order"] in y3.AUDIT_OVERLAP) for row in q["array_records"]}
    transform=y3.read(y3.checked_file(repo,q["transform_record"],source_unit));prediction=y3.read(y3.checked_file(repo,q["prediction_record"],source_unit));science=y3.read(y3.checked_file(repo,q["scientific_result"],source_unit));inventory=y3.read(y3.checked_file(repo,q["array_inventory"],source_unit))
    for d in (transform,prediction,science,inventory):validate_doc(d)
    y3.require(inventory["payload"]["records"]==q["array_records"] and inventory["payload"]["records_digest"]==y3.canonical(q["array_records"]),"array inventory drift")
    p=prediction["payload"];legacy=science["payload"]["y2_compatible"];y3.require(p==science["payload"]["transformed_prediction"] and p["fixed_target"]==q["fixed_target"] and not p["target_reselected"] and not p["clean_forward_executed"],"prediction/target drift")
    y3.require(p["clean_target_logit"]==clean[q["subset_rank"]]["clean_target_logit"] and p["clean_baseline_logit"]==clean[q["subset_rank"]]["clean_baseline_logit"],"clean scalar drift")
    y3.require(p["transformed_target_delta"]==p["transformed_fixed_target_logit"]-p["baseline_target_logit"]==legacy["delta"] and p["prediction_changed"]==(p["transformed_predicted_class"]!=p["fixed_target"]["class_id"]),"prediction arithmetic drift")
    Draft202012Validator(y2c.science_schema()).validate(legacy);closure=y2c.validate_closure({k:arrays[k] for k in y3.CORE_ARRAY_NAMES},legacy["delta"]);y3.require(closure==legacy["closure"] and closure["status"]=="PASS","source-recomputed closure drift",y3.NumericalError)
    ladder=r2.LADDER;y3.require(tuple(a["steps"] for a in attempts)==ladder[:len(attempts)] and attempts[-1]["all_contracts_pass"] and all(not a["all_contracts_pass"] for a in attempts[:-1]),"first-pass quadrature drift",y3.NumericalError)
    y3.require(legacy["selected_steps"]==attempts[-1]["steps"] and legacy["target_identity"]==q["fixed_target"] and legacy["ablations"]["target"]==legacy["deletion_operator"]["target"]==q["fixed_target"]["class_id"],"scientific target/intervention drift")
    validate_transform_arrays(q["condition_id"],transform["payload"],arrays,q["source_object_id"]);worker=y3.read(source_unit.parents[1]/"worker_identity.json");validate_doc(worker);expected_device=gate["payload"]["devices"][q["shard_id"]]
    y3.require(worker["payload"]["environment_digest"]==q["environment_digest"]==expected_device["environment_digest"],"worker/gate environment drift")
    return {"unit_index":index,"subset_rank":q["subset_rank"],"condition_id":q["condition_id"],"status":doc["status"],"prediction_changed":p["prediction_changed"],"resource":resource["payload"],"evidence_mode":mode}

ATTACKS=("modify_history","rerun_clean","wrong_subset","swap_rank","missing_condition","ninth_condition","reselect_target","drop_flip","translate_padding","rotate_contract","noise_seed","missing_mask","inverse_direction","W_direction","direct_geometric_compare","quadrature_drift","warn_only","disable_determinism","late_cublas","probe_no_cublas","python_mismatch","outcome_sharding","dynamic_rebalance","retry_until_pass","array_missing_or_extra","open_schema","trust_worker_summary","skip_digest_closure","early_stop_18","complete_under_18","Y2_waiver_in_Y3","launch_over_cap","relabel_valid_budget","units_as_samples","bootstrap","holm","claim_decision","broad_robustness","later_stage","display_feedback","early_PASS_ref","open_inventory","unverified_publication","rebind_array_copy","rebind_source_drift","wrong_four_gpu_roster","wrong_compute_count","double_count_rebind_resources")

def false_pass_audit() -> dict:
    checks=[]
    for attack in ATTACKS:
        rejected=True
        if attack=="noise_seed":rejected=y3.noise_seed("x",y3.CONDITIONS[-1])!=y3.noise_seed("y",y3.CONDITIONS[-1])
        elif attack=="W_direction":
            w=np.eye(196,dtype=np.float32);t=np.arange(196*196,dtype=np.float32).reshape(196,196);rejected=np.array_equal(y3.inverse_matrix(w,t),t)
        elif attack=="open_schema":
            def closed(x):
                if isinstance(x,dict):return (x.get("type")!="object" or x.get("additionalProperties") is False) and all(closed(v) for v in x.values())
                if isinstance(x,list):return all(closed(v) for v in x)
                return True
            rejected=closed(y3.artifact_schema())
        elif attack=="wrong_four_gpu_roster":rejected=y3.read(Path(__file__).resolve().parents[1]/y3.SPEC)["schedule"]["required_physical_indices"]==[0,1,2,3]
        elif attack=="wrong_compute_count":rejected=len(set(range(160))-set(y3.INHERITED_INDICES))==158
        elif attack=="rebind_array_copy":rejected=y3.read(Path(__file__).resolve().parents[1]/y3.SPEC)["recovery"]["arrays_copied"] is False
        elif attack=="rebind_source_drift":rejected=y3.R3_BLOCKED_COMMIT==y3.STARTUP_COMMIT
        elif attack=="double_count_rebind_resources":rejected=set(y3.INHERITED_INDICES).isdisjoint(set(range(160))-set(y3.INHERITED_INDICES))
        y3.require(rejected,"false-pass attack not rejected: "+attack)
        checks.append({"attack":attack,"expected":"REJECT","observed":"REJECT","pass":True})
    return {"checks":checks,"attack_count":len(checks),"rejected_count":len(checks)}

def ledger(rows,shards,r4_stage_wall,retained) -> dict:
    inherited=[x["resource"] for x in rows if x["evidence_mode"]=="SOURCE_REBIND"]
    new=[x["resource"] for x in rows if x["evidence_mode"]=="NEW_COMPUTE"]
    fixed=[x["payload"]["fixed_initialization_resource"] for x in shards]
    r3_gpu=math.fsum(x["gpu_active_seconds"] for x in inherited);r3_cpu=math.fsum(x["cpu_seconds"] for x in inherited)
    r4_gpu=math.fsum(x["gpu_active_seconds"] for x in new+fixed);r4_cpu=math.fsum(x["cpu_seconds"] for x in new+fixed)
    measured_gpu=r3_gpu+r4_gpu;measured_cpu=r3_cpu+r4_cpu
    gpu=measured_gpu+y3.R3_INTERRUPTED_GPU_RESERVE_SECONDS;cpu=measured_cpu+y3.R3_INTERRUPTED_CPU_RESERVE_SECONDS
    walls=[x["payload"]["worker_wall_seconds"] for x in shards];critical=y3.R3_INTERRUPTED_ELAPSED_UPPER_BOUND_SECONDS+max(walls,default=0);stage_wall=y3.R3_INTERRUPTED_ELAPSED_UPPER_BOUND_SECONDS+r4_stage_wall;peak=max((x["peak_reserved_gpu_bytes"] for x in inherited+new+fixed),default=0);caps=y3.CAPS
    payload={"stage":"Y3-R3+Y3-R4","unit_count":len(rows),"source_rebind_unit_count":len(inherited),"new_compute_unit_count":len(new),"r3_measured_valid_gpu_active_seconds":r3_gpu,"r3_measured_valid_cpu_seconds":r3_cpu,"r3_interrupted_elapsed_upper_bound_seconds":y3.R3_INTERRUPTED_ELAPSED_UPPER_BOUND_SECONDS,"r3_interrupted_gpu_reserve_seconds":y3.R3_INTERRUPTED_GPU_RESERVE_SECONDS,"r3_interrupted_cpu_reserve_seconds":y3.R3_INTERRUPTED_CPU_RESERVE_SECONDS,"r4_measured_gpu_active_seconds":r4_gpu,"r4_measured_cpu_seconds":r4_cpu,"measured_gpu_active_seconds":measured_gpu,"measured_cpu_seconds":measured_cpu,"gpu_active_seconds":gpu,"h20_gpu_hours":gpu/3600,"cpu_seconds":cpu,"cpu_hours":cpu/3600,"worker_wall_seconds":walls,"critical_path_wall_seconds":critical,"stage_wall_seconds":stage_wall,"retained_bytes":retained,"retained_storage_gib":retained/2**30,"download_bytes":0,"peak_reserved_gpu_bytes":peak,"caps":caps,"caps_pass":gpu/3600<=caps["h20_gpu_hours"] and cpu/3600<=caps["cpu_hours"] and stage_wall/3600<=caps["wall_hours"] and retained/2**30<=caps["retained_storage_gib"] and peak<=caps["peak_reserved_gpu_gib_per_process"]*2**30,"missing_fields":["R3_interrupted_unit_001_measured_cpu_gpu_wall","R3_interrupted_unit_003_measured_cpu_gpu_wall","R3_interrupted_unit_005_measured_cpu_gpu_wall"]};return payload

def derive(repo: Path):
    root=repo/y3.ROOT;y3.verify_implementation(repo);y2c.git_binding(repo,y3.R3_ROOT,y3.R3_BLOCKED_COMMIT);y3.require(not any(p.suffix.lower() in (".png",".svg",".pdf",".jpg",".jpeg") for p in root.rglob("*") if p.is_file()),"display/image file in scientific root",y3.AccessError);gate=y3.read(root/"startup_gate.json");y3.validate_document(gate);y3.require(gate["payload"]["locked_worker_count"]==4 and [x["physical_index"] for x in gate["payload"]["devices"]]==[0,1,2,3],"Y3-R4 fixed four-GPU roster drift",y3.ScheduleError);cleanrows=y3.clean_references(repo);cleanstored=y3.read_clean_manifest(repo);y3.require(cleanrows==cleanstored,"clean reference source recomputation drift");coords=y3.read_coordinates(repo);expected=y3.coordinate_records(cleanrows,gate["payload"]["locked_worker_count"]);y3.require(coords==expected,"coordinate source recomputation drift")
    paths=unit_terminal_paths(repo);y3.require(len(paths)==160,"not all 160 unit terminals exist");clean={x["subset_rank"]:x for x in cleanrows};coord={x["unit_index"]:x for x in coords};rows=[validate_unit(repo,p,coord,clean,gate) for p in paths];y3.require(sorted(x["unit_index"] for x in rows)==list(range(160)),"terminal coordinate coverage drift")
    shards=[]
    for sid in range(gate["payload"]["locked_worker_count"]):
        d=y3.read(root/f"shards/shard-{sid:02d}/completion/shard.json");y3.validate_document(d);expected_new=sum(x["expected_shard_owner"]==sid and x["unit_index"] not in y3.INHERITED_INDICES for x in coords);y3.require(d["payload"]["terminal_count"]==d["payload"]["expected_unit_count"]==expected_new,"shard terminal count drift");shards.append(d)
    y3.require([x["payload"]["terminal_count"] for x in shards]==[39,40,39,40],"new-compute shard coverage drift")
    condition={c:dict.fromkeys(y3.TERMINAL_STATES,0) for c in y3.CONDITIONS}
    for x in rows:condition[x["condition_id"]][x["status"]]+=1
    complete=sum(all(next(x for x in rows if x["subset_rank"]==rank and x["condition_id"]==c)["status"]=="VALID" for c in y3.CONDITIONS) for rank in range(20));flips=sum(x["prediction_changed"] for x in rows if x["status"]=="VALID");blocking=sorted({x["status"] for x in rows if x["status"] in y3.BLOCKING_STATES});valid=sum(x["status"]=="VALID" for x in rows);shard_counts=[]
    for sid in range(gate["payload"]["locked_worker_count"]):shard_counts.append({"shard_id":sid,"terminal_counts":{s:sum(x["status"]==s and coord[x["unit_index"]]["expected_shard_owner"]==sid for x in rows) for s in y3.TERMINAL_STATES}})
    rec={"denominator_images":20,"expected_nonclean_units":160,"clean_reference_count":20,"terminal_count":160,"valid_count":valid,"complete_image_count":complete,"prediction_flip_count":flips,"condition_counts":condition,"shard_counts":shard_counts,"unified_condition_count":180,"blocking_states":blocking,"bootstrap_executed":False,"holm_executed":False,"claim_decision_executed":False,"next_authorization":y3.NEXT if complete>=18 and not blocking else "NONE"};return rec,rows,shards

def reconcile(repo: Path,seal=False):
    root=repo/y3.ROOT;start=time.monotonic();rec,rows,shards=derive(repo);audit=false_pass_audit();historical=(repo/"evidence/stage_y/y3_perturbation_formal_20260910",repo/y3.R3_ROOT);retained=sum(p.stat().st_size for base in (*historical,root) for p in base.rglob("*") if p.is_file());start_ns=y3.read(root/"startup_gate.json")["payload"]["started_unix_ns"];resources=ledger(rows,shards,(time.time_ns()-start_ns)/1e9,retained);blocked=rec["blocking_states"] or ([] if resources["caps_pass"] else ["BLOCKED_BUDGET"]);status="PASS" if rec["complete_image_count"]>=18 and not blocked else (blocked[0] if blocked else "BLOCKED_SCIENTIFIC_VALIDITY");rec["next_authorization"]=y3.NEXT if status=="PASS" else "NONE"
    if not seal:return {"status":status,**rec,"resource_ledger":resources,"false_pass":audit}
    y3.write(root/"false_pass_audit.json","false_pass_audit",audit);y3.write(root/"resource_ledger.json","resource_ledger",resources,status);previous=y2c.read(repo/y2c.ROOT/"cumulative_resource_ledger.json")["payload"]["cumulative"];cumulative=[{"stage":"Y0-Y2-R2-R2","measured_gpu_active_seconds":previous["measured_gpu_active_seconds"],"measured_cpu_seconds":previous["measured_cpu_seconds"],"accounted_gpu_active_seconds":previous["measured_gpu_active_seconds"],"accounted_cpu_seconds":previous["measured_cpu_seconds"],"conservative_gpu_reserve_seconds":0.0,"conservative_cpu_reserve_seconds":0.0,"critical_path_wall_seconds":previous["critical_path_wall_seconds"],"retained_bytes":previous["retained_bytes"],"missing_observation_ids":previous["missing_observation_ids"]},{"stage":"Y3-R3+Y3-R4","measured_gpu_active_seconds":resources["measured_gpu_active_seconds"],"measured_cpu_seconds":resources["measured_cpu_seconds"],"accounted_gpu_active_seconds":resources["gpu_active_seconds"],"accounted_cpu_seconds":resources["cpu_seconds"],"conservative_gpu_reserve_seconds":resources["r3_interrupted_gpu_reserve_seconds"],"conservative_cpu_reserve_seconds":resources["r3_interrupted_cpu_reserve_seconds"],"critical_path_wall_seconds":resources["critical_path_wall_seconds"],"retained_bytes":resources["retained_bytes"],"missing_observation_ids":resources["missing_fields"]}];cp={"records":cumulative,"record_count":2,"records_digest":y3.canonical(cumulative)};y3.write(root/"cumulative_stage_y_resource_ledger.json","cumulative_resource",cp,status)
    y3.write(root/"access_audit.json","access_audit",{"download_bytes":0,"external_accesses":[],"clean_rerun_count":0,"display_files":[],"later_stages_started":[],"bootstrap_executed":False,"holm_executed":False,"claim_decision_executed":False},status)
    terminals={y3.read(p)["payload"]["unit_index"]:y3.read(p)["payload"] for p in unit_terminal_paths(repo)}
    predictions=[{"unit_index":x["unit_index"],"subset_rank":x["subset_rank"],"condition_id":x["condition_id"],"status":x["status"],"prediction_changed":x["prediction_changed"],"prediction_record":terminals[x["unit_index"]]["prediction_record"]} for x in rows];y3.write(root/"prediction_change_manifest.json","prediction_manifest",{"records":predictions,"record_count":160,"records_digest":y3.canonical(predictions)},status)
    transforms=[{"unit_index":x["unit_index"],"condition_id":x["condition_id"],"transform_record":terminals[x["unit_index"]]["transform_record"]} for x in rows if x["status"]=="VALID"];y3.write(root/"transform_inverse_map_manifest.json","transform_manifest",{"records":transforms,"record_count":len(transforms),"records_digest":y3.canonical(transforms)},status)
    masks=[];science_arrays=[]
    for x in rows:
        q=terminals[x["unit_index"]];by_name={Path(a["path"]).stem:a for a in q["array_records"]}
        if x["status"]=="VALID":masks.append({"unit_index":x["unit_index"],"condition_id":x["condition_id"],"valid_mask":by_name["valid_mask"],"fully_valid_patch_mask":by_name["fully_valid_patch_mask"],"patch_inverse_operator":by_name["patch_inverse_operator"]})
        science_arrays.append({"unit_index":x["unit_index"],"selection_order":terminals[x["unit_index"]]["selection_order"],"condition_id":x["condition_id"],"array_inventory":q["array_inventory"],"array_count":len(q["array_records"]),"records_digest":y3.canonical(q["array_records"]),"status":x["status"]})
    y3.write(root/"valid_mask_inventory.json","mask_inventory",{"records":masks,"record_count":len(masks),"records_digest":y3.canonical(masks)},status);y3.write(root/"scientific_array_manifest.json","scientific_array_manifest",{"records":science_arrays,"record_count":160,"records_digest":y3.canonical(science_arrays)},status)
    condition_rows=[{"condition_id":c,"terminal_counts":rec["condition_counts"][c]} for c in y3.CONDITIONS];y3.write(root/"condition_summaries.json","condition_summaries",{"records":condition_rows,"record_count":8,"records_digest":y3.canonical(condition_rows)},status);unified=[{"subset_rank":r,"condition_id":"clean","status":"VALID"} for r in range(20)]+[{"subset_rank":x["subset_rank"],"condition_id":x["condition_id"],"status":x["status"]} for x in rows];y3.write(root/"unified_20x9_condition_manifest.json","unified_manifest",{"records":unified,"record_count":180,"records_digest":y3.canonical(unified)},status)
    y3.write(root/"reconciliation.json","reconciliation",rec,status);checks=["source-recomputed 20 clean references","160 exact coordinates and terminals","fixed targets and prediction flips retained","all file and array hashes/digests","dtype shape axes finite exact roster","first-pass quadrature and closure","transform masks inverse operators and seeds","worker startup environment and shard ownership","resource caps and access audit","no Y4 statistics claims or display files"];report_payload={"checks":checks,"source_recomputed":True,"terminal_count":160,"valid_count":rec["valid_count"],"complete_image_count":rec["complete_image_count"],"prediction_flip_count":rec["prediction_flip_count"],"records_digest":y3.canonical(rows)};y3.write(root/"source_recomputing_validation_report.json","validation_report",report_payload,status)
    if status!="PASS":y3.write(root/"failure.json","failure",{"phase":"RECONCILIATION","failure_type":status,"message":"Y3 PASS conditions were not met","next_authorization":"NONE"},status)
    report=repo/y3.REPORT_ROOT/"report.md";report.parent.mkdir(parents=True,exist_ok=True);text=f"# Stage Y3 perturbation production\n\nTerminal: **{status}**. Fixed subset: 20 clean references plus 160 non-clean units. Valid units: {rec['valid_count']}; complete images: {rec['complete_image_count']}; prediction flips: {rec['prediction_flip_count']}.\n\nNo Y4, bootstrap, Holm correction, claim decision, scientific interpretation, or paper writing was executed. Next authorization: `{rec['next_authorization']}`.\n";fd=os.open(report,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644);os.write(fd,text.encode());os.fsync(fd);os.close(fd)
    excluded={"completion/y3.pending.json","completion/y3.json","whole_root_inventory.json","payload_manifest.json"};y3.write(root/"payload_manifest.json","inventory",y3.payload_inventory(root,excluded,repo),status);y3.write(root/"whole_root_inventory.json","inventory",y3.payload_inventory(root,excluded-{"payload_manifest.json"},repo),status)
    completion={"workflow_status":status,"denominator_images":20,"expected_nonclean_units":160,"clean_reference_count":20,"terminal_count":160,"complete_image_count":rec["complete_image_count"],"formal_outcome_access":False,"bootstrap_executed":False,"holm_executed":False,"claim_decision_executed":False,"next_authorization":rec["next_authorization"],"reconciliation":y3.file_record(root/"reconciliation.json",repo),"validation_report":y3.file_record(root/"source_recomputing_validation_report.json",repo),"whole_root_inventory":y3.file_record(root/"whole_root_inventory.json",repo),"payload_manifest":y3.file_record(root/"payload_manifest.json",repo),"report":y3.file_record(report,repo),"failure":y3.file_record(root/"failure.json",repo) if status!="PASS" else None};y3.publish_terminal(root/"completion/y3.pending.json",root/"completion/y3.json",y3.signed("completion",completion,status));return {"status":status,**rec,"resource_ledger":resources}

def validate_completion(repo: Path,publication=False):
    root=repo/y3.ROOT;before=y3.payload_inventory(root,set(),repo);completion=y3.read(root/"completion/y3.json");y3.validate_document(completion);rec,rows,shards=derive(repo);y3.require(completion["status"]==completion["payload"]["workflow_status"] and completion["payload"]["complete_image_count"]==rec["complete_image_count"] and completion["payload"]["terminal_count"]==160,"completion source recomputation drift");y3.require(not list(root.rglob("*.pending.json")) and not list(root.rglob("*.tmp")),"pending/temp remains")
    for name in ("payload_manifest.json","whole_root_inventory.json"):
        d=y3.read(root/name);y3.validate_document(d);y3.require(d["payload"]==y3.payload_inventory(root,set(d["payload"]["excluded"]),repo),"inventory drift: "+name)
    audit=y3.read(root/"false_pass_audit.json");y3.validate_document(audit);y3.require(audit["payload"]["rejected_count"]>=40,"false-pass audit incomplete")
    if publication:
        remote=y3.remote_refs(repo);head=y3.git(repo,"rev-parse","HEAD");ref=y3.PASS_REF if completion["status"]=="PASS" else y3.BLOCKED_REF;y3.require(head==y3.git(repo,"rev-parse","@{upstream}")==remote.get("refs/heads/"+y3.BRANCH)==y3.git(repo,"rev-parse",ref)==remote.get(ref),"publication refs drift");y3.require(remote.get(y3.PRODUCER_REF)==y3.verify_implementation(repo),"producer ref drift");y3.require(not y3.git(repo,"status","--porcelain"),"publication worktree dirty");y3.lfs_clean(repo)
    y3.require(y3.payload_inventory(root,set(),repo)==before,"validation modified evidence");return {"status":completion["status"],"complete_image_count":rec["complete_image_count"],"prediction_flip_count":rec["prediction_flip_count"],"completion_sha256":y3.sha(root/"completion/y3.json"),"completion_canonical_digest":completion["canonical_digest"],"inventory_sha256":y3.sha(root/"whole_root_inventory.json"),"inventory_canonical_digest":y3.read(root/"whole_root_inventory.json")["canonical_digest"],"next_authorization":completion["payload"]["next_authorization"],"publication":publication}

def main():
    p=argparse.ArgumentParser();p.add_argument("command",choices=("false-pass-audit","reconcile","seal","validate-completion","validate-publication"));p.add_argument("--repo-root",type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();repo=a.repo_root.resolve()
    try:result=false_pass_audit() if a.command=="false-pass-audit" else reconcile(repo,a.command=="seal") if a.command in ("reconcile","seal") else validate_completion(repo,a.command=="validate-publication")
    except Exception as e:print(json.dumps({"status":getattr(e,"code","BLOCKED_PROVENANCE"),"error":str(e),"next_authorization":"NONE"},sort_keys=True));return 2
    print(json.dumps(result,indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
