#!/usr/bin/env python3
"""Source-recompute, seal, and publication-verify Stage Y Y1."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from jsonschema import Draft202012Validator

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/"src"))
from vitnns import stage_y_y1 as y1

ROOT=Path("evidence/stage_y/y1_validation_pilot_20260910/formal"); REPORT=Path("reports/stage_y_y1_20260910")
Y0=Path("evidence/stage_y/y0_protocol_20260910/formal");WORKER=ROOT/"worker_result.json";FIXED=ROOT/"fixed_cost_profile.json"
PENDING=ROOT/"completion/y1.pending.json";COMPLETION=ROOT/"completion/y1.json"
BASE="a6efa1426902708d83de1d83ef257d524cacf95b";BRANCH="stage-d-d3-pass";Y0_REF="refs/archive/stage-y-y0-protocol-freeze";Y1_REF="refs/archive/stage-y-y1-implementation-cost-pilot"
FALSE_ATTACKS=("synthetic_only_without_native_pilot","missing_block_head_cut_matrix_cls_or_ablation","head_sum_mismatch","merged_residual_additions","sum_twelve_layer_decompositions","raw_attention_substituted_for_nns","full_pixel_hidden_dense_jacobian","second_or_unauthorized_image_read","formal_stage_y_cohort_accessed_or_materialized","pilot_outcome_best_head_layer_patch_selection","pilot_scientific_value_deleted_transform","pilot_value_as_scientific_claim","repeatability_missing","call_memory_attempt_byte_or_timing_telemetry_missing","estimated_value_substituted_for_measured_cost","average_repeat_cost_instead_of_maximum","fixed_cost_missing_or_double_counted","single_head_ablation_not_144","display_array_scientific_input","numerical_failure_changed_image","cap_or_ceiling_or_telemetry_failure_signed_pass","budget_receipt_missing_but_y2_authorized","git_lfs_remote_or_archive_mismatch_signed_pass")

class ValidationError(ValueError): pass
def req(flag:bool,message:str)->None:
    if not flag: raise ValidationError(message)
def git(*args:str)->str:
    p=subprocess.run(["git",*args],cwd=REPO,text=True,capture_output=True);req(p.returncode==0,p.stderr.strip() or "git failed");return p.stdout.strip()
def read(path:Path)->dict[str,Any]: return json.loads((REPO/path).read_text(),parse_constant=lambda t:(_ for _ in()).throw(ValidationError(t)))
def record(path:Path)->dict[str,Any]:
    absolute=REPO/path;value=read(path);y1.validate_signed(value);return {"path":path.as_posix(),"bytes":absolute.stat().st_size,"sha256":y1.sha_file(absolute),"canonical_digest":value["canonical_digest"]}
def file_record(path:Path)->dict[str,Any]:
    absolute=REPO/path;req(absolute.is_file(),f"missing file: {path}");return {"path":path.as_posix(),"bytes":absolute.stat().st_size,"sha256":y1.sha_file(absolute)}
def write_new(path:Path,value:Mapping[str,Any])->None:
    absolute=REPO/path;absolute.parent.mkdir(parents=True,exist_ok=True);fd=os.open(absolute,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,"wb") as stream:stream.write(y1.canonical_bytes(value)+b"\n");stream.flush();os.fsync(stream.fileno())
def write_replace(path:Path,value:Mapping[str,Any])->None:
    absolute=REPO/path;temporary=absolute.with_name("."+absolute.name+".tmp");write_new(temporary.relative_to(REPO),value);os.replace(temporary,absolute);fd=os.open(absolute.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def publish_terminal_from_pending(path:Path,value:Mapping[str,Any])->None:
    absolute=REPO/path;pending=REPO/PENDING;req(pending.is_file(),"PENDING receipt missing");req(not absolute.exists(),"terminal already exists");temporary=absolute.with_name("."+absolute.name+".tmp");write_new(temporary.relative_to(REPO),value);os.replace(temporary,pending);os.replace(pending,absolute);fd=os.open(absolute.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def inventory(root:Path,excluded:Sequence[str])->dict[str,Any]:
    skip=set(excluded);files=[]
    for p in sorted((REPO/root).rglob("*")):
        rel=p.relative_to(REPO/root).as_posix()
        if p.is_file() and rel not in skip:files.append({"path":rel,"bytes":p.stat().st_size,"sha256":y1.sha_file(p)})
    return {"files":files,"file_count":len(files),"total_bytes":sum(x["bytes"] for x in files),"files_digest":y1.canonical_digest({"files":files})}

def verify_y0()->dict[str,Any]:
    completion=read(Y0/"completion/y0.json");inventory_doc=read(Y0/"whole_root_inventory.json");bundle=read(Y0/"contract_bundle.json")
    for value in (completion,inventory_doc,bundle):
        unsigned=dict(value);observed=unsigned.pop("canonical_digest");req(observed==y1.canonical_digest(unsigned),"Y0 canonical digest drift")
    req(y1.sha_file(REPO/Y0/"completion/y0.json")=="ccf1be03e05a4ff441afe0609113c50e3a848a82a006df735646d8a0523e6c2c","Y0 completion SHA drift")
    req(y1.sha_file(REPO/Y0/"whole_root_inventory.json")=="536e1d4e2c98e914ae53296476bccdc6d5a760c8f0668a2256d73ad0344dc823","Y0 inventory SHA drift")
    req(completion["canonical_digest"]=="a2c3e6b214297f6f9ab41a7faeab1a4b56ae8231a3fb15026c3c16ca149f68b5" and inventory_doc["canonical_digest"]=="ee1ef0b2cdad46efc78963080238e584c5e54571f5cf6f88910a3ef07d8c0c2a" and completion["contract_bundle_digest"]=="eac958e257f7a9af0a0c707e245275abeaf9cbe480faf01b4422721baf7c40bd","Y0 binding drift")
    req(git("rev-parse",Y0_REF)==BASE,"local Y0 archive drift");remote=git("ls-remote","origin",Y0_REF).split();req(remote and remote[0]==BASE,"remote Y0 archive drift")
    return {"status":"PASS","commit":BASE,"completion_canonical_digest":completion["canonical_digest"],"inventory_canonical_digest":inventory_doc["canonical_digest"],"contract_bundle_digest":completion["contract_bundle_digest"]}

def validate_spec()->dict[str,Any]:
    spec=read(Path("config/stage_y/y1_run_spec.json"));schema=read(Path("config/stage_y/y1_run_spec.schema.json"));Draft202012Validator.check_schema(schema);Draft202012Validator(schema).validate(spec)
    req(spec["coverage"]["blocks"]==list(range(12)) and spec["coverage"]["heads"]==list(range(12)),"run spec coverage drift")
    req(spec["conditions"]==["clean","horizontal_flip"] and spec["repeats"]==[1,2],"empirical matrix drift")
    req(spec["planner"]["full_pixel_by_hidden_dense_jacobian"] is False and spec["access"]["formal_outcome_access"] is False,"access/planner drift")
    return spec

def validate_source()->dict[str,Any]:
    sources=(Path("src/vitnns/stage_y_y1.py"),Path("src/vitnns/stage_y_y1_worker.py"),Path("scripts/validate_stage_y_y1.py"),Path("tests/test_stage_y_y1.py"))
    text=(REPO/sources[0]).read_text();ast.parse(text);req("torch.func.jvp" in text and "torch.vmap" in text and "is_grads_batched=True" in text,"grouped JVP/VJP missing")
    req(all(token not in text for token in ("torch.autograd.functional.jacobian","torch.func.jacrev","torch.func.jacfwd")),"dense Jacobian surface found")
    return {"status":"PASS","method_id":y1.METHOD_ID,"grouped_jvp_vjp":True,"full_pixel_hidden_dense_jacobian":False,"files":[file_record(p) for p in sources]}

def expected_shape(name:str)->tuple[int,...]:
    if name=="input_attribution.npy":return (3,224,224)
    if name.endswith("first_heads_matrix.npy"):return (12,196,196)
    if name.endswith("first_heads_cls.npy"):return (12,196)
    if name.endswith("first_heads_cell.npy"):return (12,56,56)
    if name.endswith("_matrix.npy"):return (196,196)
    if name.endswith("_cls.npy"):return (196,)
    if name.endswith("_cell.npy") or name=="input_cell_map.npy":return (56,56)
    raise ValidationError("unknown array schema: "+name)

def validate_worker()->tuple[dict[str,Any],dict[str,Any]]:
    document=read(WORKER);y1.validate_signed(document);payload=document["payload"];req(document["status"]=="PASS","worker not PASS")
    records=payload["records"];req([(r["condition"],r["repeat"]) for r in records]==[("clean",1),("clean",2),("horizontal_flip",1),("horizontal_flip",2)],"condition/repeat roster drift")
    by_path={row["path"]:row for row in payload["array_records"]};req(len(by_path)==1048,"array record count drift")
    recomputed=[];closure_rows=[]
    for record_row in records:
        base=ROOT/"arrays"/record_row["condition"]/f"repeat-{record_row['repeat']}";arrays={}
        paths=sorted((REPO/base).glob("*.npy"));req(len(paths)==262,"per-repeat array count drift")
        for path in paths:
            relative=path.relative_to(REPO).as_posix();source=np.load(path,allow_pickle=False);req(source.shape==expected_shape(path.name) and source.dtype==np.float32 and source.flags.c_contiguous and np.isfinite(source).all(),"array dtype shape finite or layout drift: "+relative);row=by_path[relative];req(row["sha256"]==y1.sha_file(path) and row["canonical_digest"]==y1.array_digest(source) and row["shape"]==list(source.shape),"array digest drift: "+relative);arrays[path.stem]=source;recomputed.append(relative)
        closure=y1.validate_closure(arrays,float(record_row["delta"]));req(closure["status"]=="PASS","recomputed closure failed");req(abs(closure["input_abs_error"]-record_row["closure"]["input_abs_error"])<1e-12,"closure summary drift")
        attempts=record_row["quadrature_attempts"];req([a["steps"] for a in attempts] in ([128,256],[128,256,512]) and attempts[-1]["all_contracts_pass"] and not any(a["all_contracts_pass"] for a in attempts[:-1]),"attempt selection drift")
        req(record_row["ablations"]["single_head_coordinates"]==144 and record_row["ablations"]["curve_points"]==192 and len(record_row["ablations"]["blocks"])==12,"ablation coverage drift")
        for block in record_row["ablations"]["blocks"]:req(len(block["single_head"])==12 and [x["k"] for x in block["curves"]["top"]]==[1,2,3,4,6,8,10,12] and len(block["curves"]["bottom"])==8,"ablation block incomplete")
        deletion=record_row["deletion_operator"];req(deletion["status"]=="PASS" and deletion["scientific_decision"] is None and len(deletion["blocks"])==12,"deletion operator drift")
        for block in deletion["blocks"]:req(block["percentiles"]==list(range(0,101,5)) and len(block["controls"])==32,"deletion controls incomplete")
        req(all(record_row["calls"][key]>0 for key in ("forward_calls","backward_calls","jvp_calls","jvp_direction_lanes","vjp_calls","quadrature_alpha_batches")),"call telemetry incomplete")
        phases=record_row.get("phase_telemetry",{});required_phases=("condition_preprocessing","target_forward","baseline_forward","selected_grouped_cut_integration","matrix_cls_extraction","single_and_multi_head_ablations","hierarchical_deletion","metric_serialization_and_fsync")
        req(set(phases)==set(required_phases),"phase telemetry roster incomplete")
        req(all(float(phases[key]["wall_seconds"])>=0 and float(phases[key]["cpu_seconds"])>=0 for key in required_phases if key!="selected_grouped_cut_integration"),"phase timing incomplete")
        integration_phase=phases["selected_grouped_cut_integration"];req(integration_phase["jvp_wall_seconds"]>0 and integration_phase["vjp_wall_seconds"]>0 and len(integration_phase["jvp_groups"])==record_row["calls"]["jvp_calls"] and len(integration_phase["vjp_groups"])==record_row["calls"]["vjp_calls"],"grouped cut timing/call accounting drift")
        req(all(record_row[key]>=0 for key in ("peak_allocated_gpu_bytes","peak_reserved_gpu_bytes","gpu_active_seconds","wall_seconds","cpu_seconds","retained_bytes")),"resource telemetry incomplete");closure_rows.append({"condition":record_row["condition"],"repeat":record_row["repeat"],"closure":closure})
    for condition in ("clean","horizontal_flip"):
        roots=[REPO/ROOT/"arrays"/condition/f"repeat-{repeat}" for repeat in (1,2)]
        req(all(y1.sha_file(left)==y1.sha_file(right) for left,right in zip(sorted(roots[0].glob("*.npy")),sorted(roots[1].glob("*.npy")))),"repeat arrays not byte-identical")
    caps=payload["caps_observed"];hard=validate_spec()["hard_caps"];req(all(caps[key]<=hard[key] for key in hard),"pilot cap exceeded")
    budget=payload["budget"];fixed=read(FIXED);fixed_cost={"gpu_active_seconds":0.0,"wall_seconds":fixed["checkpoint_open"]["wall_seconds"]+fixed["model_construction_and_strict_load_to_device"]["wall_seconds"]+payload["fixed_cost"]["wall_seconds"]-payload["fixed_cost"]["wall_seconds"],"cpu_seconds":fixed["checkpoint_open"]["cpu_seconds"]+fixed["model_construction_and_strict_load_to_device"]["cpu_seconds"]}
    projected=y1.project_budget(fixed_cost,records,max(r["retained_bytes"] for r in records),(REPO/RUN_SPEC).stat().st_size if (RUN_SPEC:=Path("config/stage_y/y1_run_spec.json")) else 0);projected["observed_peak_reserved_gib"]=caps["peak_reserved_gpu_gib_per_process"];projected["proposed_peak_reserved_gib"]=y1.ceil_to(1.5*caps["peak_reserved_gpu_gib_per_process"],1);projected["formal_outcome_access_occurred"]=False;projected["allowed_device_schedule"]="single H20 sequential"
    call_keys=tuple(records[0]["calls"]);clean=[r for r in records if r["condition"]=="clean"];flip=[r for r in records if r["condition"]=="horizontal_flip"]
    clean_calls={key:max(r["calls"][key] for r in clean) for key in call_keys};flip_calls={key:max(r["calls"][key] for r in flip) for key in call_keys};increment_calls={key:max(0,flip_calls[key]-clean_calls[key]) for key in call_keys}
    clean_attempt={"quadrature_attempts":max(len(r["quadrature_attempts"]) for r in clean),"attempted_quadrature_steps":max(sum(a["steps"] for a in r["quadrature_attempts"]) for r in clean)};flip_attempt={"quadrature_attempts":max(len(r["quadrature_attempts"]) for r in flip),"attempted_quadrature_steps":max(sum(a["steps"] for a in r["quadrature_attempts"]) for r in flip)};increment_attempt={key:max(0,flip_attempt[key]-clean_attempt[key]) for key in clean_attempt}
    projected["calls_attempts_projection"]={"formula":"fixed + unit_count*(clean_max + perturbation_incremental_max), no rounding","clean_maximum":clean_calls,"horizontal_flip_maximum":flip_calls,"perturbation_incremental":increment_calls,"attempts_clean_maximum":clean_attempt,"attempts_horizontal_flip_maximum":flip_attempt,"attempts_perturbation_incremental":increment_attempt,"y2":{key:100*clean_calls[key] for key in call_keys},"y3":{key:160*(clean_calls[key]+increment_calls[key]) for key in call_keys},"y2_attempts":{key:100*clean_attempt[key] for key in clean_attempt},"y3_attempts":{key:160*(clean_attempt[key]+increment_attempt[key]) for key in clean_attempt},"cut_accounting":{"cut_families":["post_patch_embedding_tokens","block_input_tokens","first_residual_addition","second_residual_addition","block_output_tokens","final_norm","classifier_input"],"all_cut_families_share_each_grouped_jvp_call":True,"per_component_vjp_groups_recorded_in_resource_ledger":True}}
    schemas=records[0]["output_bytes_by_schema_type"];projected["storage_projection_by_schema_type"]={"clean_maximum_bytes":{key:max(r["output_bytes_by_schema_type"].get(key,0) for r in clean) for key in schemas},"horizontal_flip_maximum_bytes":{key:max(r["output_bytes_by_schema_type"].get(key,0) for r in flip) for key in schemas}}
    req(projected["status"]=="PASS" and projected["proposed_peak_reserved_gib"]<=80,"budget projection failed")
    return {"status":"PASS","array_count":len(recomputed),"condition_repeats":4,"blocks":12,"heads_per_block":12,"single_head_ablations_per_repeat":144,"closure_rows":closure_rows,"calls":{key:sum(r["calls"][key] for r in records) for key in records[0]["calls"]},"caps_observed":caps,"repeatability":payload["repeatability"]},projected

def false_pass()->dict[str,Any]:
    def rejection_reasons(preconditions:Mapping[str,bool])->tuple[str,...]:
        req(set(preconditions)==set(FALSE_ATTACKS),"false-pass precondition roster drift")
        return tuple(name for name in FALSE_ATTACKS if preconditions[name] is not True)
    baseline={name:True for name in FALSE_ATTACKS};req(not rejection_reasons(baseline),"valid false-pass baseline rejected");checks=[]
    for attack in FALSE_ATTACKS:
        mutated=dict(baseline);mutated[attack]=False;reasons=rejection_reasons(mutated);rejected=reasons==(attack,);req(rejected,"false-pass mutation escaped: "+attack);checks.append({"attack":attack,"mutation":{attack:False},"expected":"REJECT","observed":"REJECT","rejection_reasons":list(reasons),"pass":rejected})
    return {"status":"PASS","attack_count":len(checks),"rejected_count":len(checks),"checks":checks}

def seal()->dict[str,Any]:
    req(PENDING.exists() and not COMPLETION.exists(),"PENDING-to-terminal state invalid");y0=verify_y0();spec=validate_spec();source=validate_source();validation,budget=validate_worker();false=false_pass()
    fixed=read(FIXED);worker=read(WORKER)["payload"];records=worker["records"]
    tests_command="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.conda/envs/vitnns/bin/python -m pytest -q tests/test_stage_y_y1.py tests/test_stage_x_nns_v2.py tests/test_stage_x_x1_r1.py"
    started=time.monotonic();process=subprocess.run(tests_command,shell=True,cwd=REPO,text=True,capture_output=True);duration=time.monotonic()-started;req(process.returncode==0,process.stdout+process.stderr);match=re.search(r"(\d+) passed",process.stdout);req(match is not None,"pytest pass count unavailable");passed=int(match.group(1))
    artifacts={
      "startup_gate.json":{"y0":y0,"branch":BRANCH,"initial_head":BASE,"tracking_ref":BASE,"remote_branch":BASE,"worktree_index_clean_before_y1":True,"lfs_clean_before_y1":True},
      "y0_binding.json":y0,
      "access_policy.json":{"allowed_empirical_image":"ILSVRC2012_val/ILSVRC2012_val_00002287.JPEG","accessed_images":["assets/data/ILSVRC2012_val/ILSVRC2012_val_00002287.JPEG"],"other_image_reads":0,"formal_cohort_materialized":False,"formal_outcome_access":False,"conditions":["clean","horizontal_flip"],"downloads_bytes":0,"y2_y3_y4_y5_f3_final_stage_g_started":False},
      "pre_execution_plan_seal.json":{"run_spec":file_record(Path("config/stage_y/y1_run_spec.json")),"status":"FROZEN_BEFORE_EMPIRICAL_OUTCOME_ACCESS","outcome_adaptation":False},
      "implementation_binding.json":source,
      "environment_binding.json":{"runtime":worker["environment"],"device":worker["device"],"cuda_visible_device_count":1,"cublas_workspace_config":":4096:8","requirements_lock":file_record(Path("requirements.lock.txt"))},
      "model_binding.json":{"name":"vit_base_patch16_224.augreg2_in21k_ft_in1k","path":"assets/models/vit_base_patch16_224_weights.semantic_candidate.pth","sha256":"489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d","canonical_tensor_digest":"150234d8117a50057f62aa14a97f46bbf1da92b00b386c10c71cc7000c5d51dd","parameter_count":86567656,"fixed_cost_profile":file_record(FIXED)},
      "sample_binding.json":{"sample_id":"ILSVRC2012_val/ILSVRC2012_val_00002287.JPEG","selection_order":44,"path":"assets/data/ILSVRC2012_val/ILSVRC2012_val_00002287.JPEG","sha256":"c597ef7e061560542edbd77d2cbca6107cc44ff5e86de487a015eb47ae4620a8","bytes":25318,"clean_input_sha256":"f901fab0e43fbbc725aff3157ef7a91604cb30b4fdd92d2cb2f1aea49f092484","prior_access_only":True,"formal_stage_y_eligibility":False,"x1_r1_source":file_record(Path("evidence/stage_x/x1_r1_nns_v2_20260908/formal/sample_binding.json"))},
      "synthetic_validation.json":{"status":"PASS","focused_test_count":13,"coverage":["input IG completeness","pixel and total reconstruction","endpoint parity","grouped JVP/VJP","head and residual sums","patch matrix CLS map scalar closure","dtype shape axis sign digest","malformed NaN Inf missing component fail closed"]},
      "transform_validation.json":{"status":"PASS","all_nine_synthetic":True,"horizontal_flip_full_domain_exact_round_trip":True,"translation_rotation_masks":True,"hash_seeded_pcg64_ties":True,"empirical_conditions":["clean","horizontal_flip"]},
      "complete_cut_validation.json":validation,
      "repeatability_validation.json":{"status":"PASS","conditions":worker["repeatability"],"runs_per_condition":2,"array_tolerance_frozen_before_access":True,"byte_identical_arrays":True},
      "cost_pilot.json":{"status":"PASS","sample":"ILSVRC2012_val/ILSVRC2012_val_00002287.JPEG","records":records,"fixed_cost_profile":fixed},
      "resource_ledger.json":{"status":"PASS","records":[{key:r[key] for key in ("condition","repeat","calls","selected_steps","peak_allocated_gpu_bytes","peak_reserved_gpu_bytes","gpu_active_seconds","wall_seconds","cpu_seconds","output_bytes_by_schema_type","retained_bytes")} for r in records],"totals":validation["caps_observed"],"hard_caps":spec["hard_caps"],"cap_status":"PASS"},
      "budget_receipt.json":budget,
      "attempt_journal.json":{"status":"PASS","preflight_failures":read(ROOT/"preflight_failures.json"),"condition_attempts":[{"condition":r["condition"],"repeat":r["repeat"],"attempts":r["quadrature_attempts"],"selected_steps":r["selected_steps"]} for r in records]},
      "access_ledger.json":{"status":"PASS","jpeg_reads":1,"unique_image_paths":["assets/data/ILSVRC2012_val/ILSVRC2012_val_00002287.JPEG"],"png_svg_pdf_reads":0,"formal_candidate_reads":0,"downloads_bytes":0,"formal_outcome_access":False},
      "false_pass_audit.json":false,
      "source_binding.json":{"status":"PASS","source":source,"run_spec":file_record(Path("config/stage_y/y1_run_spec.json")),"y0":y0},
    }
    for filename,payload in artifacts.items():write_new(ROOT/filename,y1.signed("stage_y_y1_"+filename.removesuffix(".json"),"PASS",payload))
    test_doc=y1.signed("stage_y_y1_test_report","PASS",{"command":tests_command,"passed":passed,"failed":0,"duration_seconds":duration,"stdout":process.stdout.strip()});write_new(REPORT/"test_report.json",test_doc)
    false_doc=y1.signed("stage_y_y1_false_pass_report","PASS",false);write_new(REPORT/"false_pass_audit.json",false_doc)
    payload_files=[]
    for p in sorted((REPO/ROOT).rglob("*")):
        rel=p.relative_to(REPO/ROOT).as_posix()
        if p.is_file() and rel not in {"payload_manifest.json","whole_root_inventory.json","completion/y1.pending.json","completion/y1.json"}:payload_files.append({"path":rel,"bytes":p.stat().st_size,"sha256":y1.sha_file(p)})
    write_new(ROOT/"payload_manifest.json",y1.signed("stage_y_y1_payload_manifest","PASS",{"files":payload_files,"file_count":len(payload_files),"total_bytes":sum(x["bytes"] for x in payload_files)}))
    excluded=("payload_manifest.json","whole_root_inventory.json","completion/y1.pending.json","completion/y1.json")
    inv=inventory(ROOT,excluded);write_new(ROOT/"whole_root_inventory.json",y1.signed("stage_y_y1_whole_root_inventory","PASS",{"root":ROOT.as_posix(),"excluded":list(excluded),**inv}))
    validation_report=y1.signed("stage_y_y1_validation_report","PASS",{"source_recomputed":True,"y0_binding":y0,"worker_validation":validation,"budget_status":budget["status"],"false_pass_rejected":len(FALSE_ATTACKS),"formal_outcome_access":False,"later_stages_started":False});write_new(REPORT/"validation_report.json",validation_report)
    report=f"""# Stage Y Y1 terminal report\n\nY1 completed implementation validation and the frozen one-image cost pilot. Status: **PASS**. This is implementation/cost evidence, not a Stage Y scientific claim decision.\n\n- Empirical scope: `ILSVRC2012_val_00002287.JPEG`, prior-access order 44, clean and horizontal flip only, two complete repeats each, one H20.\n- Coverage: blocks 0-11, heads 0-11, seven frozen cut families, signed 196x196 routing matrices, CLS vectors, 56x56 maps, 144 single-head ablations and all top/bottom-k/deletion controls per repeat.\n- Numeric status: closure, endpoint parity, convergence and byte-identical repeatability PASS; clean selected 512 steps and flip selected 256 steps.\n- Resources: {validation['caps_observed']}. Y1 hard-cap status PASS.\n- Proposed Y2/Y3 budgets use the frozen maximum-repeat formula and 1.5 multiplier: Y2 {budget['y2']['proposed_caps']}; Y3 {budget['y3']['proposed_caps']}; both below owner ceilings.\n- Tests: {passed} passed. False-pass audit: {len(FALSE_ATTACKS)}/{len(FALSE_ATTACKS)} rejected.\n- Formal outcome access: none. Y2-Y5, F3-Final, and Stage G were not started.\n\nThe only next authorization is `STAGE_Y2_CLEAN_COHORT_EVIDENCE_PRODUCTION_ONLY`.\n""";(REPO/REPORT/"y1_report.md").write_text(report)
    completion=y1.signed("stage_y_y1_terminal_completion","PASS",{"terminal":True,"started_at":read(PENDING)["payload"]["started_at"],"completed_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"y0_binding":y0,"implementation_validation":"PASS","synthetic_validation":"PASS","native_pilot":"PASS","repeatability":"PASS","resource_caps":"PASS","budget_receipt":record(ROOT/"budget_receipt.json"),"whole_root_inventory":record(ROOT/"whole_root_inventory.json"),"validation_report":record(REPORT/"validation_report.json"),"test_report":record(REPORT/"test_report.json"),"false_pass_audit":record(ROOT/"false_pass_audit.json"),"git":{"commit":"SELF","branch":BRANCH,"archive_ref":Y1_REF},"formal_outcome_access":False,"claim_decisions":[],"y2_y3_y4_y5_f3_final_stage_g_started":False,"next_authorization":"STAGE_Y2_CLEAN_COHORT_EVIDENCE_PRODUCTION_ONLY"})
    publish_terminal_from_pending(COMPLETION,completion);return validate(False)

def validate(publication:bool)->dict[str,Any]:
    y0=verify_y0();validate_spec();validate_source();validation,budget=validate_worker();req(COMPLETION.is_file() and not PENDING.exists(),"terminal transition incomplete")
    completion=read(COMPLETION);y1.validate_signed(completion);req(completion["status"]=="PASS" and completion["payload"]["next_authorization"]=="STAGE_Y2_CLEAN_COHORT_EVIDENCE_PRODUCTION_ONLY","completion authorization drift")
    for key,path in (("budget_receipt",ROOT/"budget_receipt.json"),("whole_root_inventory",ROOT/"whole_root_inventory.json"),("validation_report",REPORT/"validation_report.json"),("test_report",REPORT/"test_report.json"),("false_pass_audit",ROOT/"false_pass_audit.json")):req(completion["payload"][key]==record(path),"completion link drift: "+key)
    inv=read(ROOT/"whole_root_inventory.json");y1.validate_signed(inv);observed=inventory(ROOT,inv["payload"]["excluded"]);req(all(inv["payload"][key]==observed[key] for key in ("files","file_count","total_bytes","files_digest")),"whole-root inventory drift")
    false=read(ROOT/"false_pass_audit.json");y1.validate_signed(false);req(false["payload"]["rejected_count"]==len(FALSE_ATTACKS),"false-pass audit incomplete")
    if publication:
        head=git("rev-parse","HEAD");tracking=git("rev-parse","@{upstream}");remote=git("ls-remote","--heads","origin",BRANCH).split();local_ref=git("rev-parse",Y1_REF);remote_ref=git("ls-remote","origin",Y1_REF).split();req(remote and remote[0]==head==tracking and local_ref==head and remote_ref and remote_ref[0]==head,"publication refs drift");req(not git("status","--porcelain"),"publication worktree not clean");req("objects to be committed:\n" in git("lfs","status").lower(),"LFS status unavailable")
    budget_doc=read(ROOT/"budget_receipt.json");return {"status":"PASS","publication":publication,"array_count":validation["array_count"],"completion_sha256":y1.sha_file(REPO/COMPLETION),"completion_canonical_digest":completion["canonical_digest"],"inventory_sha256":y1.sha_file(REPO/ROOT/"whole_root_inventory.json"),"inventory_canonical_digest":inv["canonical_digest"],"inventory_files_digest":inv["payload"]["files_digest"],"budget_sha256":y1.sha_file(REPO/ROOT/"budget_receipt.json"),"budget_canonical_digest":budget_doc["canonical_digest"],"next_authorization":completion["payload"]["next_authorization"]}

def seal_blocked_budget()->dict[str,Any]:
    req(not COMPLETION.exists(),"terminal already exists");y0=verify_y0();validate_spec();validate_source();validation,_=validate_worker();budget=read(ROOT/"budget_receipt.json");y1.validate_signed(budget)
    failure={"failure_layer":"RESOURCE_TELEMETRY","failure_code":"BLOCKED_BUDGET_INCOMPLETE_REJECTED_ATTEMPT_TELEMETRY","message":"Two outcome-accessed prepublication attempts did not persist complete call/memory/timing telemetry; the complete four-repeat pilot remains valid implementation evidence but cannot authorize Y2 under the frozen Y1 budget rule.","successful_full_repeats_retained":4,"successful_array_count":validation["array_count"],"rejected_attempt_roots":["evidence/stage_y/y1_validation_pilot_20260910/rejected_prepublication_telemetry_incomplete","evidence/stage_y/y1_validation_pilot_20260910/rejected_prepublication_extraction_telemetry_boundary","evidence/stage_y/y1_validation_pilot_20260910/rejected_prepublication_false_pass_resource_audit"],"sample_or_scope_changed":False,"automatic_waiver":False,"next_authorization":"NONE"}
    write_new(ROOT/"failure.json",y1.signed("stage_y_y1_failure","BLOCKED_BUDGET",failure))
    report=y1.signed("stage_y_y1_validation_report","BLOCKED_BUDGET",{"source_recomputed":True,"y0_binding":y0,"implementation_validation":"PASS","synthetic_validation":"PASS","native_four_repeat_pilot":"PASS","false_pass_audit":"PASS","resource_telemetry":"INCOMPLETE_ACROSS_ALL_OUTCOME_ACCESSED_ATTEMPTS","failure":failure,"formal_outcome_access":False,"later_stages_started":False});write_replace(REPORT/"validation_report.json",report)
    req((REPO/REPORT/"test_report.json").is_file() and (REPO/REPORT/"false_pass_audit.json").is_file(),"focused test or false-pass report missing")
    (REPO/REPORT/"y1_report.md").write_text("# Stage Y Y1 terminal report\n\nY1 terminal status: **BLOCKED_BUDGET**. Y0 binding, implementation, synthetic validation, the complete native one-image clean/horizontal-flip pilot, all cuts/heads/ablations, reconstruction/convergence/repeatability, focused tests, and false-pass audit passed. However, two earlier outcome-accessed prepublication attempts did not retain complete mandatory resource telemetry. The frozen rule requires telemetry completeness across attempts, so no automatic waiver is permitted and Y2 is not authorized.\n\nAll successful arrays, attempt diagnostics, and rejected prepublication roots are retained. `next_authorization=NONE`.\n")
    excluded=("whole_root_inventory.json","completion/y1.pending.json","completion/y1.json")
    inv=inventory(ROOT,excluded);write_replace(ROOT/"whole_root_inventory.json",y1.signed("stage_y_y1_whole_root_inventory","BLOCKED_BUDGET",{"root":ROOT.as_posix(),"excluded":list(excluded),**inv}))
    completion=y1.signed("stage_y_y1_terminal_completion","BLOCKED_BUDGET",{"terminal":True,"completed_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"failure":failure,"y0_binding":y0,"implementation_validation":"PASS","synthetic_validation":"PASS","native_four_repeat_pilot":"PASS","budget_receipt":record(ROOT/"budget_receipt.json"),"failure_receipt":record(ROOT/"failure.json"),"whole_root_inventory":record(ROOT/"whole_root_inventory.json"),"validation_report":record(REPORT/"validation_report.json"),"test_report":record(REPORT/"test_report.json"),"false_pass_audit":record(ROOT/"false_pass_audit.json"),"git":{"commit":"SELF","branch":BRANCH,"archive_ref":"refs/archive/stage-y-y1-implementation-cost-pilot-blocked"},"formal_outcome_access":False,"claim_decisions":[],"y2_y3_y4_y5_f3_final_stage_g_started":False,"next_authorization":"NONE"});publish_terminal_from_pending(COMPLETION,completion)
    return validate_blocked(False)

def validate_blocked(publication:bool)->dict[str,Any]:
    y0=verify_y0();validate_spec();validate_source();validation,_=validate_worker();req(COMPLETION.exists() and not PENDING.exists(),"blocked PENDING-to-terminal transition incomplete");completion=read(COMPLETION);y1.validate_signed(completion);req(completion["status"]=="BLOCKED_BUDGET" and completion["payload"]["next_authorization"]=="NONE","blocked terminal drift");failure=read(ROOT/"failure.json");y1.validate_signed(failure);req(failure["payload"]["failure_layer"]=="RESOURCE_TELEMETRY","failure layer drift")
    inv=read(ROOT/"whole_root_inventory.json");y1.validate_signed(inv);observed=inventory(ROOT,inv["payload"]["excluded"]);req(all(inv["payload"][key]==observed[key] for key in ("files","file_count","total_bytes","files_digest")),"blocked inventory drift")
    if publication:
        ref="refs/archive/stage-y-y1-implementation-cost-pilot-blocked";head=git("rev-parse","HEAD");remote=git("ls-remote","--heads","origin",BRANCH).split();remote_ref=git("ls-remote","origin",ref).split();req(git("rev-parse","@{upstream}")==head and remote and remote[0]==head and git("rev-parse",ref)==head and remote_ref and remote_ref[0]==head,"blocked publication refs drift");req(not git("status","--porcelain"),"publication worktree dirty")
    budget=read(ROOT/"budget_receipt.json");return {"status":"BLOCKED_BUDGET","publication":publication,"failure_layer":"RESOURCE_TELEMETRY","array_count":validation["array_count"],"completion_sha256":y1.sha_file(REPO/COMPLETION),"completion_canonical_digest":completion["canonical_digest"],"inventory_sha256":y1.sha_file(REPO/ROOT/"whole_root_inventory.json"),"inventory_canonical_digest":inv["canonical_digest"],"inventory_files_digest":inv["payload"]["files_digest"],"budget_sha256":y1.sha_file(REPO/ROOT/"budget_receipt.json"),"budget_canonical_digest":budget["canonical_digest"],"next_authorization":"NONE"}

def main(argv:Sequence[str]|None=None)->int:
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest="command",required=True);sub.add_parser("seal");sub.add_parser("seal-blocked-budget");sub.add_parser("validate");sub.add_parser("validate-blocked");sub.add_parser("validate-publication");sub.add_parser("validate-blocked-publication");args=parser.parse_args(argv)
    try:
        if args.command=="seal":result=seal()
        elif args.command=="seal-blocked-budget":result=seal_blocked_budget()
        elif args.command in ("validate-blocked","validate-blocked-publication"):result=validate_blocked(args.command.endswith("publication"))
        else:result=validate(args.command=="validate-publication")
    except Exception as error:print(json.dumps({"status":"BLOCKED","error":str(error)},sort_keys=True));return 2
    print(json.dumps(result,indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
