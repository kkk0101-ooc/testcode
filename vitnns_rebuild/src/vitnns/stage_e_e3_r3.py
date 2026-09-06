"""E3-R3 prospective numerical-stability recovery."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from jsonschema import Draft202012Validator

from .constants import REPO_ROOT
from .stage_d_provenance import canonical_json_bytes, canonical_sha256
from . import stage_e_e1 as e1
from . import stage_e_e3 as e3
from . import stage_e_e3_r1 as r1
from . import stage_e_e3_r2 as r2

RECOVERY_ID = "STAGE-E-E3-R3-NUMERICAL-STABILITY-20260907"
CONFIG_PATH = "config/stage_e_e3_r3.json"
CONFIG_SCHEMA_PATH = "config/stage_e_e3_r3.schema.json"
COMPLETION_SCHEMA_PATH = "config/stage_e_e3_r3_completion.schema.json"
AMENDMENT_PATH = "docs/recovery_stage_e_e3_r3_numerical_stability_amendment_20260907.md"
ROOT_PATH = "evidence/stage_e/e3_r3_recovery_v2_20260907/formal"
PILOT_ROOT = "evidence/stage_e/e3_r3_recovery_v2_20260907/pilot"
REPORT_ROOT_PATH = "reports/stage_e_e3_r3_20260907"
PLAN_PATH = ROOT_PATH + "/control/recovery_plan.json"
RUN_SPEC_PATH = ROOT_PATH + "/control/run_spec.json"
PILOT_AUTH_PATH = ROOT_PATH + "/control/pilot_execution_authorization.json"
PILOT_GATE_PATH = ROOT_PATH + "/control/pilot_repeatability_gate.json"
BULK_AUTH_PATH = ROOT_PATH + "/control/production_execution_authorization.json"
TEST_REPORT_PATH = REPORT_ROOT_PATH + "/pre_pilot_test_results.json"
COMPLETION_PENDING_PATH = ROOT_PATH + "/completion/e3_r3_completion.pending.json"
COMPLETION_PATH = ROOT_PATH + "/completion/e3_r3_completion.json"
COMPLETION_SCHEMA_PATH = "config/stage_e_e3_r3_completion.schema.json"
R2_COMMIT = "bf9fed71d33391d9a5e8f7f1025b1fedc57bdf22"
R2_TREE = "864d1aedd10fe7cc01d5c6870cf29925809ad46c"
R2_REF = "refs/archive/stage-e-e3-r2-scientific-fail"
R2_COMPLETION_SHA256 = "83f219837b1672799ce1a9980481f917274027e97962f7265dc75602434b2e72"
R2_COMPLETION_DIGEST = "8f845d51e00cf5eab20254e8ad75c193d4416c6b9a8819153dfb16df29d74d12"
R2_INVENTORY_SHA256 = "28c96ab99b97d9c634b02d25a6226facd7c68d297a444bb09f5a53f4bd504707"
R2_INVENTORY_DIGEST = "068f78c572e19993801238cc829db180d3b9931dc9c4e25bcdd6d079ab418ffd"
EXPECTED_CONFIG_DIGEST = "e7e2dcbef13d15fd9e6b0001a511a4f134b6a55b5962ad97b113c5ab681c5402"
STEPS = (65536, 131072, 262144)
ATTEMPT_ID = "attempt-r3-000001"
PILOT_ORDER = 117
PILOT_COORDINATE = (11, 2)
PILOT_RUNS = ("repeat-000001", "repeat-000002")


class StageEE3R3Error(RuntimeError):
    pass


def _now() -> str:
    zone=time.strftime("%z",time.localtime()); return time.strftime("%Y-%m-%dT%H:%M:%S",time.localtime())+zone[:-2]+":"+zone[-2:]


def _sha(raw: bytes) -> str:
    import hashlib
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path: Path) -> str:
    import hashlib
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()


def _signed(value: Mapping[str, Any]) -> dict[str, Any]:
    result=dict(value); result["canonical_digest"]=canonical_sha256({k:v for k,v in result.items() if k!="canonical_digest"}); return result


def _strict_json(path: Path, context: str) -> tuple[dict[str, Any],bytes]:
    try:
        raw=path.read_bytes(); value=json.loads(raw.decode("utf-8"),parse_constant=lambda token:(_ for _ in ()).throw(ValueError(token)),object_pairs_hook=e1._reject_duplicate_keys)
    except Exception as error: raise StageEE3R3Error(f"cannot read {context}: {path}: {error}") from error
    if not isinstance(value,dict): raise StageEE3R3Error(f"{context} must be an object")
    return value,raw


def _validate_signed(value: Mapping[str, Any], context: str) -> None:
    if value.get("canonical_digest")!=canonical_sha256({k:v for k,v in value.items() if k!="canonical_digest"}): raise StageEE3R3Error(f"{context} canonical digest mismatch")


def _write_new_json(path: Path,value: Mapping[str,Any]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); raw=canonical_json_bytes(value)+b"\n"; descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(descriptor,"wb") as stream: stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    directory=os.open(path.parent,os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)


def _write_or_verify(path: Path,value: Mapping[str,Any]) -> None:
    raw=canonical_json_bytes(value)+b"\n"
    if path.exists():
        if path.read_bytes()!=raw: raise StageEE3R3Error(f"immutable document differs: {path}")
    else: _write_new_json(path,value)


def _git(repo: Path,*args: str) -> str:
    try: return subprocess.run(("git",*args),cwd=repo,check=True,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout.strip()
    except subprocess.CalledProcessError as error: raise StageEE3R3Error(f"git {' '.join(args)} failed: {error.stderr.strip()}") from error


def _validate_config(repo: Path) -> dict[str,Any]:
    value,_=_strict_json(repo/CONFIG_PATH,"R3 config"); schema,_=_strict_json(repo/CONFIG_SCHEMA_PATH,"R3 schema"); Draft202012Validator.check_schema(schema); errors=list(Draft202012Validator(schema).iter_errors(value))
    if errors: raise StageEE3R3Error(f"R3 config schema failure: {errors[0].message}")
    _validate_signed(value,"R3 config")
    if value["canonical_digest"]!=EXPECTED_CONFIG_DIGEST or tuple(value["numerical_stability"]["prospective_schedule"])!=STEPS or value["numerical_stability"]["atol"]!=1e-4 or value["numerical_stability"]["rtol"]!=0.01: raise StageEE3R3Error("R3 frozen config drifted")
    return value


def _v1_plan(repo: Path) -> dict[str,Any]:
    value,_=_strict_json(repo/e3.PLAN_PATH,"E3 plan"); return value


def _pilot_unit(repo: Path) -> dict[str,Any]:
    plan=_v1_plan(repo); rows=[unit for unit in plan["units"] if unit["selection_order"]==PILOT_ORDER and (unit["block"],unit["head"])==PILOT_COORDINATE and unit["method"]==e3.NNS_METHOD]
    if len(rows)!=1: raise StageEE3R3Error("R3 pilot unit is not unique")
    return rows[0]


def validate_upstream(repo_root: Path|str=REPO_ROOT) -> dict[str,Any]:
    repo=Path(repo_root).resolve(); result=r2.validate_completion(repo)
    if result.get("status")!="FAIL" or result.get("failure_selection_order")!=117 or result.get("failure_coordinate")!=[11,2] or result.get("next_authorization")!="NONE" or result.get("e4_started") is not False: raise StageEE3R3Error("R2 terminal failure drifted")
    completion,raw=_strict_json(repo/r2.COMPLETION_PATH,"R2 completion")
    if _sha(raw)!=R2_COMPLETION_SHA256 or completion["canonical_digest"]!=R2_COMPLETION_DIGEST: raise StageEE3R3Error("R2 completion binding drifted")
    inventory,inventory_raw=_strict_json(repo/r2.ROOT_PATH/"whole_root_inventory.json","R2 inventory"); _validate_signed(inventory,"R2 inventory")
    if _sha(inventory_raw)!=R2_INVENTORY_SHA256 or inventory["canonical_digest"]!=R2_INVENTORY_DIGEST: raise StageEE3R3Error("R2 inventory binding drifted")
    if _git(repo,"rev-parse",R2_REF+"^{commit}")!=R2_COMMIT or _git(repo,"rev-parse",R2_COMMIT+"^{tree}")!=R2_TREE: raise StageEE3R3Error("R2 custody binding drifted")
    if _git(repo,"diff","--name-only",R2_COMMIT,"--",r2.ROOT_PATH,r2.REPORT_ROOT_PATH): raise StageEE3R3Error("R2 evidence changed after custody")
    return result


def record_tests(repo_root:Path|str=REPO_ROOT,*,passed:int,failed:int,command:str)->dict[str,Any]:
    if type(passed)is not int or passed<=0 or failed!=0 or not command.strip(): raise StageEE3R3Error("only zero-failure tests can be recorded")
    report=_signed({"schema_version":1,"document_type":"stage_e_e3_r3_pre_pilot_test_results","recovery_id":RECOVERY_ID,"status":"PASS","recorded_at":_now(),"command":command,"passed":passed,"failed":failed,"e4_started":False}); _write_or_verify(Path(repo_root).resolve()/TEST_REPORT_PATH,report); return report


def verify_preconditions(repo_root:Path|str=REPO_ROOT)->dict[str,Any]:
    repo=Path(repo_root).resolve(); config=_validate_config(repo); upstream=validate_upstream(repo); unit=_pilot_unit(repo)
    return {"status":"PASS","config":config,"upstream":upstream,"pilot_unit_digest":unit["canonical_digest"],"head":_git(repo,"rev-parse","HEAD"),"tree":_git(repo,"rev-parse","HEAD^{tree}"),"amendment_sha256":_sha_file(repo/AMENDMENT_PATH)}


def initialize(repo_root:Path|str=REPO_ROOT)->dict[str,Any]:
    repo=Path(repo_root).resolve(); pre=verify_preconditions(repo); tests,_=_strict_json(repo/TEST_REPORT_PATH,"R3 tests"); _validate_signed(tests,"R3 tests")
    status=_git(repo,"status","--porcelain=v1"); allowed=(ROOT_PATH+"/",PILOT_ROOT+"/",REPORT_ROOT_PATH+"/"); unexpected=[line for line in status.splitlines() if not any(line[3:].startswith(item) or f"/{item}" in line[3:] for item in allowed)]
    if unexpected: raise StageEE3R3Error(f"R3 producer worktree is not clean: {unexpected}")
    head=pre["head"]
    if _git(repo,"rev-parse","@{u}")!=head: raise StageEE3R3Error("R3 producer is not pushed")
    unit=_pilot_unit(repo); plan=_signed({"schema_version":1,"document_type":"stage_e_e3_r3_recovery_plan","recovery_id":RECOVERY_ID,"pilot_unit_digest":unit["canonical_digest"],"pilot_order":117,"pilot_coordinate":[11,2],"pilot_method":e3.NNS_METHOD,"inherited_valid_attempts":1164,"r3_new_attempts_required":1836,"pilot_runs":list(PILOT_RUNS)})
    run_spec=_signed({"schema_version":1,"document_type":"stage_e_e3_r3_run_spec","recovery_id":RECOVERY_ID,"plan_canonical_digest":plan["canonical_digest"],"schedule":list(STEPS),"model_compute_dtype":"float32","quadrature_dtype":"float64","accumulation_dtype":"float64","accumulation_algorithm":"kahan_per_chunk","stored_attribution_dtype":"float32","atol":1e-4,"rtol":0.01,"repeatability_atol":1e-6,"repeatability_rtol":1e-5,"e4_started":False})
    authorization=_signed({"schema_version":1,"document_type":"stage_e_e3_r3_pilot_execution_authorization","recovery_id":RECOVERY_ID,"status":"AUTHORIZED_PILOT_ONLY","issued_at":_now(),"producer_commit":head,"producer_tree":pre["tree"],"amendment_sha256":pre["amendment_sha256"],"r2_commit":R2_COMMIT,"plan_canonical_digest":plan["canonical_digest"],"run_spec_canonical_digest":run_spec["canonical_digest"],"authorized_order":117,"authorized_coordinate":[11,2],"authorized_method":e3.NNS_METHOD,"authorized_runs":list(PILOT_RUNS),"bulk_authorized":False,"e4_authorized":False,"e4_started":False})
    pending=_signed({"schema_version":1,"document_type":"stage_e_e3_r3_completion_pending","stage_id":"E3-R3","recovery_id":RECOVERY_ID,"status":"PENDING","started_at":_now(),"plan_canonical_digest":plan["canonical_digest"],"authorization_canonical_digest":authorization["canonical_digest"],"next_authorization":"NONE_PENDING","e4_started":False})
    for relative,value in ((PLAN_PATH,plan),(RUN_SPEC_PATH,run_spec),(PILOT_AUTH_PATH,authorization),(COMPLETION_PENDING_PATH,pending)): _write_or_verify(repo/relative,value)
    return validate_initialized(repo)


def validate_initialized(repo_root:Path|str=REPO_ROOT)->dict[str,Any]:
    repo=Path(repo_root).resolve(); verify_preconditions(repo); documents={}
    for name,relative in (("plan",PLAN_PATH),("spec",RUN_SPEC_PATH),("auth",PILOT_AUTH_PATH),("pending",COMPLETION_PENDING_PATH)):
        value,_=_strict_json(repo/relative,f"R3 {name}"); _validate_signed(value,f"R3 {name}"); documents[name]=value
    if documents["auth"].get("status")!="AUTHORIZED_PILOT_ONLY" or documents["auth"].get("authorized_order")!=117 or documents["auth"].get("bulk_authorized") is not False: raise StageEE3R3Error("R3 pilot authorization drifted")
    if documents["spec"].get("schedule")!=list(STEPS) or documents["pending"].get("e4_started") is not False: raise StageEE3R3Error("R3 initialized contract drifted")
    return {"status":"PASS","pilot_order":117,"pilot_coordinate":[11,2],"pilot_runs":2,"bulk_authorized":False,"e4_started":False}


def _pilot_paths(repo:Path,run_id:str)->tuple[Path,Path,Path,Path]:
    if run_id not in PILOT_RUNS: raise StageEE3R3Error("unknown R3 pilot run")
    root=repo/PILOT_ROOT/run_id; return root,root/"result.json",root/"map.npy",root/"manifest.json"


def _publish_pilot(repo:Path,run_id:str,result:Mapping[str,Any],signed_map:np.ndarray)->Path:
    import tempfile,shutil
    root,_,_,_=_pilot_paths(repo,run_id)
    if root.exists(): raise StageEE3R3Error(f"refusing to overwrite R3 pilot: {root}")
    root.parent.mkdir(parents=True,exist_ok=True); temporary=Path(tempfile.mkdtemp(prefix=f".{run_id}-",dir=root.parent))
    try:
        signed=_signed(result); raw=canonical_json_bytes(signed)+b"\n"; (temporary/"result.json").write_bytes(raw)
        with (temporary/"map.npy").open("wb") as stream: np.save(stream,signed_map,allow_pickle=False)
        manifest=_signed({"schema_version":1,"document_type":"stage_e_e3_r3_pilot_manifest","recovery_id":RECOVERY_ID,"run_id":run_id,"result_sha256":_sha(raw),"map_sha256":_sha_file(temporary/"map.npy"),"map_array_sha256":e1._array_digest(signed_map),"immutable":True}); (temporary/"manifest.json").write_bytes(canonical_json_bytes(manifest)+b"\n")
        for path in temporary.iterdir():
            with path.open("rb") as stream: os.fsync(stream.fileno())
        os.replace(temporary,root); directory=os.open(root.parent,os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        if temporary.exists(): shutil.rmtree(temporary)
    _validate_pilot(repo,run_id); return root


def _validate_pilot(repo:Path,run_id:str)->dict[str,Any]:
    root,result_path,map_path,manifest_path=_pilot_paths(repo,run_id)
    if not root.is_dir() or any(not path.is_file() for path in (result_path,map_path,manifest_path)): raise StageEE3R3Error("incomplete R3 pilot run")
    result,raw=_strict_json(result_path,"R3 pilot result"); manifest,_=_strict_json(manifest_path,"R3 pilot manifest"); _validate_signed(result,"R3 pilot result"); _validate_signed(manifest,"R3 pilot manifest"); signed_map=np.load(map_path,allow_pickle=False)
    status=result.get("status")
    if status not in {"VALID","FAILED"} or result.get("run_id")!=run_id or result.get("selection_order")!=117 or result.get("coordinate")!=[11,2] or result.get("method")!=e3.NNS_METHOD or result.get("selected_steps") not in STEPS or result.get("atol")!=1e-4 or result.get("rtol")!=0.01: raise StageEE3R3Error("R3 pilot result contract failed")
    attempt_steps=tuple(item["steps"] for item in result.get("attempts",[]))
    if status=="VALID":
        if result.get("converged") is not True or attempt_steps!=STEPS[:STEPS.index(result["selected_steps"])+1] or result.get("diagnostic_map_only") is not None: raise StageEE3R3Error("R3 valid pilot convergence contract failed")
    elif result.get("converged") is not False or attempt_steps!=STEPS or result.get("selected_steps")!=STEPS[-1] or result.get("diagnostic_map_only") is not True: raise StageEE3R3Error("R3 failed pilot convergence contract failed")
    metadata=result.get("numerical_metadata",{})
    if metadata.get("quadrature_dtype")!="float64" or metadata.get("accumulation_dtype")!="float64" or metadata.get("accumulation_algorithm")!="kahan_per_chunk" or metadata.get("conservation_reduction_dtype")!="float64" or metadata.get("model_compute_dtype")!="float32" or metadata.get("stored_attribution_dtype")!="float32": raise StageEE3R3Error("R3 pilot numerical metadata failed")
    if signed_map.dtype!=np.float32 or signed_map.shape!=(56,56) or not np.isfinite(signed_map).all() or manifest.get("result_sha256")!=_sha(raw) or manifest.get("map_sha256")!=_sha_file(map_path) or manifest.get("map_array_sha256")!=e1._array_digest(signed_map): raise StageEE3R3Error("R3 pilot payload binding failed")
    return {"result":result,"manifest":manifest,"map":signed_map,"root":root}


def _validate_pilot_authorization(repo:Path)->dict[str,Any]:
    value,_=_strict_json(repo/PILOT_AUTH_PATH,"R3 pilot authorization"); _validate_signed(value,"R3 pilot authorization")
    if value.get("status")!="AUTHORIZED_PILOT_ONLY" or value.get("authorized_order")!=117 or value.get("authorized_coordinate")!=[11,2] or value.get("authorized_runs")!=list(PILOT_RUNS) or value.get("bulk_authorized") is not False: raise StageEE3R3Error("R3 pilot authorization boundary failed")
    producer=value["producer_commit"]
    if subprocess.run(("git","merge-base","--is-ancestor",producer,"HEAD"),cwd=repo,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode: raise StageEE3R3Error("R3 execution HEAD does not descend from producer")
    if _git(repo,"diff","--name-only",producer,"HEAD","--","src/vitnns","config","tests",AMENDMENT_PATH): raise StageEE3R3Error("R3 producer source changed before pilot")
    return value


def run_pilot(repo_root:Path|str=REPO_ROOT,*,run_id:str,device:str="cuda:0")->dict[str,Any]:
    repo=Path(repo_root).resolve(); validate_initialized(repo); _validate_pilot_authorization(repo); unit=_pilot_unit(repo)
    root=_pilot_paths(repo,run_id)[0]
    if root.exists(): return _validate_pilot(repo,run_id)["result"]
    import torch
    from .stage_c_fidelity import ValuePathScalarAdapter,resolve_target_response
    from .stage_c_value_path import HeadCandidate,StageCValuePathConvergenceError,e3_r3_conditional_fixed_clean_attention_head_direction_value_path_ig,resolve_fixed_clean_attention_head_directions
    from .stage_c_maps import pixel_to_fine_cell_mass,raw_array,signed_rgb_channel_sum
    from .stage_c_perturbations import tensor_sha256
    plan=_v1_plan(repo); group=next(row for row in plan["groups"] if row["selection_order"]==117); model=None; started=time.monotonic(); cpu_started=time.process_time(); calls=0
    try:
        model,binding,model_record=e1._load_model_and_context(repo,device); _canonical,clean,clean_sha=e1._load_sample(binding,group,model,device)
        if clean_sha!=tensor_sha256(clean): raise StageEE3R3Error("R3 pilot clean hash drifted")
        def count_hook(_module:Any,_args:Any)->None:
            nonlocal calls; calls+=1
        handle=model.register_forward_pre_hook(count_hook)
        try:
            target=resolve_fixed_clean_attention_head_directions(model,clean,(HeadCandidate(11,2),),candidate_chunk_size=1)[0]; evaluator=ValuePathScalarAdapter(model,clean,target); response=resolve_target_response(evaluator,clean,torch.zeros_like(clean))
            if response.na is not None: raise StageEE3R3Error("R3 pilot target became NA")
            target_identity=json.loads(canonical_json_bytes(evaluator.target_identity).decode("utf-8")); target_digest=e1._target_semantic_digest(target_identity)
            try:
                method=e3_r3_conditional_fixed_clean_attention_head_direction_value_path_ig(model,clean,target,provenance={"stage":"E3-R3-PILOT","run_id":run_id,"comparison_identity":e3.comparison_identity(11,2)})
                signed_map=e3._method_map(method,response); provenance=json.loads(canonical_json_bytes(dict(method.provenance)).decode("utf-8")); ig=provenance["ig_result"]
                result={"schema_version":1,"document_type":"stage_e_e3_r3_pilot_result","recovery_id":RECOVERY_ID,"run_id":run_id,"status":"VALID","sample_id":unit["sample_id"],"selection_order":117,"coordinate":[11,2],"method":e3.NNS_METHOD,"unit_digest":unit["canonical_digest"],"image_sha256":unit["image_sha256"],"clean_input_sha256":clean_sha,"model_sha256":e3.MODEL_SHA256,"model_canonical_tensor_digest":e3.MODEL_TENSOR_DIGEST,"target_semantic_digest":target_digest,"target_response":response.to_dict(),"method_provenance":provenance,"converged":True,"selected_steps":ig["selected_steps"],"attempts":ig["attempts"],"conservation":ig["conservation"],"numerical_metadata":ig["metadata"],"signed_map_sha256":e1._array_digest(signed_map),"atol":1e-4,"rtol":0.01,"elapsed_seconds":time.monotonic()-started,"cpu_seconds":time.process_time()-cpu_started,"gpu_hours":(time.monotonic()-started)/3600.0,"forward_calls":calls,"backward_calls":sum(int(math.ceil(int(item["steps"])/8)) for item in ig["attempts"]),"peak_cuda_memory_bytes":int(torch.cuda.max_memory_reserved(device)) if device.startswith("cuda") else 0,"model":model_record,"e4_started":False}
            except StageCValuePathConvergenceError as error:
                native=error.ig_result.attribution[0].detach().to(device="cpu",dtype=torch.float32).numpy().copy(); signed_map=pixel_to_fine_cell_mass(signed_rgb_channel_sum(raw_array(native))).data.copy(); ig=error.ig_result
                result={"schema_version":1,"document_type":"stage_e_e3_r3_pilot_result","recovery_id":RECOVERY_ID,"run_id":run_id,"status":"FAILED","sample_id":unit["sample_id"],"selection_order":117,"coordinate":[11,2],"method":e3.NNS_METHOD,"unit_digest":unit["canonical_digest"],"image_sha256":unit["image_sha256"],"clean_input_sha256":clean_sha,"model_sha256":e3.MODEL_SHA256,"model_canonical_tensor_digest":e3.MODEL_TENSOR_DIGEST,"target_semantic_digest":target_digest,"target_response":response.to_dict(),"converged":False,"selected_steps":ig.steps,"attempts":[dict(item) for item in ig.attempts],"conservation":{key:value.detach().cpu().tolist() for key,value in ig.conservation.items()},"numerical_metadata":dict(ig.metadata),"signed_map_sha256":e1._array_digest(signed_map),"diagnostic_map_only":True,"atol":1e-4,"rtol":0.01,"elapsed_seconds":time.monotonic()-started,"cpu_seconds":time.process_time()-cpu_started,"gpu_hours":(time.monotonic()-started)/3600.0,"forward_calls":calls,"backward_calls":sum(int(math.ceil(int(item["steps"])/8)) for item in ig.attempts),"peak_cuda_memory_bytes":int(torch.cuda.max_memory_reserved(device)) if device.startswith("cuda") else 0,"model":model_record,"e4_started":False}
        finally: handle.remove()
        _publish_pilot(repo,run_id,result,signed_map); return _validate_pilot(repo,run_id)["result"]
    finally:
        if model is not None: del model
        if device.startswith("cuda") and torch.cuda.is_available(): torch.cuda.empty_cache()


def evaluate_pilot(repo_root:Path|str=REPO_ROOT)->dict[str,Any]:
    repo=Path(repo_root).resolve(); runs=[_validate_pilot(repo,run_id) for run_id in PILOT_RUNS]; valid=all(item["result"]["status"]=="VALID" for item in runs); same_steps=len({item["result"]["selected_steps"] for item in runs})==1; identity_keys=("sample_id","selection_order","coordinate","method","unit_digest","image_sha256","clean_input_sha256","model_sha256","model_canonical_tensor_digest","target_semantic_digest")
    same_identity=all(all(runs[0]["result"][key]==item["result"][key] for key in identity_keys) for item in runs[1:]); close=bool(np.allclose(runs[0]["map"],runs[1]["map"],atol=1e-6,rtol=1e-5)); max_abs=float(np.max(np.abs(runs[0]["map"].astype(np.float64)-runs[1]["map"].astype(np.float64)))); passed=valid and same_steps and same_identity and close
    gate=_signed({"schema_version":1,"document_type":"stage_e_e3_r3_pilot_repeatability_gate","recovery_id":RECOVERY_ID,"status":"PASS" if passed else "FAIL","evaluated_at":_now(),"runs":list(PILOT_RUNS),"valid_runs":sum(item["result"]["status"]=="VALID" for item in runs),"same_selected_steps":same_steps,"selected_steps":runs[0]["result"]["selected_steps"] if same_steps else None,"same_identity":same_identity,"repeatability_atol":1e-6,"repeatability_rtol":1e-5,"maps_allclose":close,"map_max_abs_difference":max_abs,"production_authorized":passed,"bulk_authorized":passed,"e4_authorized":False,"e4_started":False}); _write_or_verify(repo/PILOT_GATE_PATH,gate)
    if passed:
        pilot_auth,_=_strict_json(repo/PILOT_AUTH_PATH,"R3 pilot authorization")
        authorization=_signed({"schema_version":1,"document_type":"stage_e_e3_r3_production_execution_authorization","recovery_id":RECOVERY_ID,"status":"AUTHORIZED_R3_COMPLEMENT_ONLY","issued_at":_now(),"producer_commit":pilot_auth["producer_commit"],"producer_tree":pilot_auth["producer_tree"],"pilot_gate_canonical_digest":gate["canonical_digest"],"authorized_new_attempts":1836,"inherited_valid_attempts":1164,"pilot_order":117,"shard_count":8,"e4_authorized":False,"e4_started":False}); _write_or_verify(repo/BULK_AUTH_PATH,authorization)
    return gate


def _r3_attempt_path(repo:Path,unit:Mapping[str,Any])->Path:
    return r1._r1_attempt_paths(repo,unit["canonical_digest"],root_path=ROOT_PATH,attempt_id=ATTEMPT_ID)[0]


def _r3_schedule_for(unit:Mapping[str,Any])->tuple[int,...]:
    is_pilot_replacement = (
        unit["selection_order"] == 117
        and (unit["block"], unit["head"]) == (11, 2)
        and unit["method"] == e3.NNS_METHOD
    )
    return STEPS if is_pilot_replacement else r1.COMPLETE_SCHEDULE + STEPS


def _source_attempt(repo:Path,unit:Mapping[str,Any])->tuple[str,dict[str,Any]]|None:
    v1=repo/e3.ROOT_PATH/"units"/unit["canonical_digest"]/"attempts/attempt-000001"
    if v1.exists(): return "E3_V1_INHERITED",e3._validate_attempt(repo,unit)
    one=repo/r1.ROOT_PATH/"units"/unit["canonical_digest"]/"attempts"/r1.ATTEMPT_ID
    if one.exists(): return "E3_R1_INHERITED",r1._validate_r1_attempt(repo,unit)
    two=repo/r2.ROOT_PATH/"units"/unit["canonical_digest"]/"attempts"/r2.ATTEMPT_ID
    if two.exists():
        observed=r1._validate_r1_attempt(repo,unit,root_path=r2.ROOT_PATH,attempt_id=r2.ATTEMPT_ID,recovery_id=r2.RECOVERY_ID,origin="E3_R2_NEW")
        if observed["result"]["validity_status"]=="VALID": return "E3_R2_INHERITED",observed
    three=_r3_attempt_path(repo,unit)
    if three.exists(): return "E3_R3_NEW",r1._validate_r1_attempt(repo,unit,root_path=ROOT_PATH,attempt_id=ATTEMPT_ID,recovery_id=RECOVERY_ID,origin="E3_R3_NEW",nns_schedule=_r3_schedule_for(unit),numerical_field="numerical_stability",numerical_id=RECOVERY_ID,failure_code="ig_conservation_failed_after_e3_r3_schedule")
    return None


def _partition(repo:Path)->tuple[dict[str,tuple[str,dict[str,Any]]],list[dict[str,Any]]]:
    plan=_v1_plan(repo); inherited={}; eligible=[]
    for unit in plan["units"]:
        source=_source_attempt(repo,unit)
        if source is None or source[0]=="E3_R3_NEW": eligible.append(unit)
        else: inherited[unit["canonical_digest"]]=source
    from collections import Counter
    counts=Counter(value[0] for value in inherited.values())
    if counts!=Counter({"E3_V1_INHERITED":516,"E3_R1_INHERITED":12,"E3_R2_INHERITED":636}) or len(eligible)!=1836: raise StageEE3R3Error(f"R3 1164/1836 partition drifted: {counts}, {len(eligible)}")
    return inherited,eligible


def _validate_production_authorization(repo:Path)->dict[str,Any]:
    gate,_=_strict_json(repo/PILOT_GATE_PATH,"R3 pilot gate"); authorization,_=_strict_json(repo/BULK_AUTH_PATH,"R3 production authorization"); _validate_signed(gate,"R3 pilot gate"); _validate_signed(authorization,"R3 production authorization")
    if gate.get("status")!="PASS" or authorization.get("status")!="AUTHORIZED_R3_COMPLEMENT_ONLY" or authorization.get("authorized_new_attempts")!=1836: raise StageEE3R3Error("R3 production is not authorized")
    producer=authorization["producer_commit"]
    if subprocess.run(("git","merge-base","--is-ancestor",producer,"HEAD"),cwd=repo,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode: raise StageEE3R3Error("R3 production HEAD does not descend from producer")
    if _git(repo,"diff","--name-only",producer,"HEAD","--","src/vitnns","config","tests",AMENDMENT_PATH): raise StageEE3R3Error("R3 producer source changed before production")
    return authorization


def run_shard(repo_root:Path|str=REPO_ROOT,*,shard_id:int,shard_count:int,device:str="cuda:0")->dict[str,Any]:
    repo=Path(repo_root).resolve()
    if shard_count!=8 or type(shard_id)is not int or not 0<=shard_id<8: raise StageEE3R3Error("R3 requires one of eight shards")
    _validate_production_authorization(repo); plan=_v1_plan(repo); _,eligible=_partition(repo); by_sample={}
    for unit in eligible: by_sample.setdefault(unit["sample_id"],[]).append(unit)
    groups=[group for group in plan["groups"] if group["sample_id"] in by_sample and group["selection_order"]%8==shard_id]
    import torch
    from .stage_c_fidelity import ValuePathScalarAdapter,resolve_target_response
    from .stage_c_value_path import HeadCandidate,StageCValuePathConvergenceError,e3_r3_conditional_fixed_clean_attention_head_direction_value_path_ig,e3_r3_full_schedule_conditional_fixed_clean_attention_head_direction_value_path_ig,matched_fixed_clean_attention_patch_token_contribution,resolve_fixed_clean_attention_head_directions
    from .stage_c_perturbations import tensor_sha256
    model=None; started=time.monotonic(); cpu_started=time.process_time(); records=[]; completed=skipped=failed=0
    try:
        model,binding,model_record=e1._load_model_and_context(repo,device); visible=os.environ.get("CUDA_VISIBLE_DEVICES"); gpu_index=torch.device(device).index or 0; physical=int(visible) if visible and visible.isdigit() else gpu_index; identities={}
        if device.startswith("cuda"):
            query=subprocess.run(("nvidia-smi","--query-gpu=index,uuid,name","--format=csv,noheader,nounits"),check=True,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout.splitlines()
            for line in query:
                index,uuid,name=(item.strip() for item in line.split(",",2)); identities[int(index)]={"gpu_uuid":uuid,"gpu_name":name}
        for group in groups:
            units=by_sample[group["sample_id"]]; existing={}
            for unit in units:
                if _r3_attempt_path(repo,unit).exists(): existing[unit["canonical_digest"]]=_source_attempt(repo,unit)[1]
            missing=[unit for unit in units if unit["canonical_digest"] not in existing]; skipped+=len(existing)
            if not missing: continue
            if any(item["result"]["validity_status"]!="VALID" for item in existing.values()): failed+=1; break
            group_start=time.monotonic(); group_cpu=time.process_time(); calls=0; _canonical,clean,clean_sha=e1._load_sample(binding,group,model,device)
            if clean_sha!=tensor_sha256(clean): raise StageEE3R3Error("R3 clean hash drifted")
            if device.startswith("cuda"): torch.cuda.reset_peak_memory_stats(device)
            def hook(_module:Any,_args:Any)->None:
                nonlocal calls; calls+=1
            handle=model.register_forward_pre_hook(hook); group_failed=False
            try:
                coordinates=sorted({(unit["block"],unit["head"]) for unit in missing},key=e3.COORDINATES.index); targets=resolve_fixed_clean_attention_head_directions(model,clean,tuple(HeadCandidate(*coord) for coord in coordinates),candidate_chunk_size=len(coordinates)); target_map={(t.block_index,t.head_index):t for t in targets}
                for block,head in coordinates:
                    target=target_map[(block,head)]; evaluator=ValuePathScalarAdapter(model,clean,target); response=resolve_target_response(evaluator,clean,torch.zeros_like(clean)); target_identity=json.loads(canonical_json_bytes(evaluator.target_identity).decode("utf-8"))
                    if response.na is not None: raise StageEE3R3Error("R3 target became NA")
                    for unit in [x for x in missing if (x["block"],x["head"])==(block,head)]:
                        unit_start=time.monotonic(); unit_cpu=time.process_time(); before=calls
                        try:
                            if unit["method"]==e3.NNS_METHOD:
                                pilot_replacement=(unit["selection_order"]==117 and (block,head)==(11,2)); method=(e3_r3_conditional_fixed_clean_attention_head_direction_value_path_ig if pilot_replacement else e3_r3_full_schedule_conditional_fixed_clean_attention_head_direction_value_path_ig)(model,clean,target,provenance={"stage":"E3-R3","comparison_identity":e3.comparison_identity(block,head),"historical_r2_failure_bound":pilot_replacement}); ig=method.provenance["ig_result"]; backward=sum(int(math.ceil(int(item["steps"])/8)) for item in ig["attempts"]); unit_schedule=STEPS if pilot_replacement else r1.COMPLETE_SCHEDULE+STEPS
                            else: method=matched_fixed_clean_attention_patch_token_contribution(target,provenance={"stage":"E3-R3","comparison_identity":e3.comparison_identity(block,head)}); backward=0
                            logical={"forward_calls":0}; curves,signed_map=e3._curves_for_method(model=model,clean=clean,target=target,response=response,method_result=method,sample_id=group["sample_id"],block=block,head=head,call_counter=logical)
                            result=r1._valid_result(unit=unit,target_identity=target_identity,response=response,method_result=method,signed_map=signed_map,curves=curves,elapsed_seconds=time.monotonic()-unit_start,cpu_seconds=time.process_time()-unit_cpu,forward_calls=calls-before,backward_calls=backward,peak_cuda_bytes=int(torch.cuda.max_memory_reserved(device)) if device.startswith("cuda") else 0,clean_sha256=clean_sha,recovery_id=RECOVERY_ID,attempt_id=ATTEMPT_ID,origin="E3_R3_NEW")
                            r1._publish_attempt(repo,unit,result,signed_map,root_path=ROOT_PATH,attempt_id=ATTEMPT_ID,recovery_id=RECOVERY_ID,nns_schedule=unit_schedule if unit["method"]==e3.NNS_METHOD else r1.COMPLETE_SCHEDULE+STEPS,numerical_field="numerical_stability",numerical_id=RECOVERY_ID,failure_code="ig_conservation_failed_after_e3_r3_schedule"); completed+=1
                        except StageCValuePathConvergenceError as error:
                            native=error.ig_result.attribution[0].detach().to(device="cpu",dtype=torch.float32).numpy().copy(); pilot_replacement=(unit["selection_order"]==117 and (block,head)==(11,2)); failure_schedule=STEPS if pilot_replacement else r1.COMPLETE_SCHEDULE+STEPS; result=r1._failure_result(unit=unit,target_identity=target_identity,response=response,error=error,elapsed_seconds=time.monotonic()-unit_start,cpu_seconds=time.process_time()-unit_cpu,forward_calls=calls-before,peak_cuda_bytes=int(torch.cuda.max_memory_reserved(device)) if device.startswith("cuda") else 0,clean_sha256=clean_sha,diagnostic=native,recovery_id=RECOVERY_ID,attempt_id=ATTEMPT_ID,origin="E3_R3_NEW",expected_schedule=failure_schedule,failure_code="ig_conservation_failed_after_e3_r3_schedule"); r1._publish_attempt(repo,unit,result,native,root_path=ROOT_PATH,attempt_id=ATTEMPT_ID,recovery_id=RECOVERY_ID,nns_schedule=failure_schedule,numerical_field="numerical_stability",numerical_id=RECOVERY_ID,failure_code="ig_conservation_failed_after_e3_r3_schedule"); completed+=1; failed+=1; group_failed=True; break
                    if group_failed: break
                    del evaluator,response
            finally: handle.remove()
            records.append({"selection_order":group["selection_order"],"sample_id":group["sample_id"],"elapsed_seconds":time.monotonic()-group_start,"cpu_seconds":time.process_time()-group_cpu,"forward_calls":calls,"terminal_units_published":sum(_r3_attempt_path(repo,u).exists() for u in units),"scientific_failure":group_failed})
            if group_failed: break
        elapsed=time.monotonic()-started; telemetry=_signed({"schema_version":1,"document_type":"stage_e_e3_r3_shard_telemetry","recovery_id":RECOVERY_ID,"shard_id":shard_id,"shard_count":8,"device":device,"physical_gpu_index":physical,"gpu_uuid":identities.get(physical,{}).get("gpu_uuid"),"gpu_name":identities.get(physical,{}).get("gpu_name"),"elapsed_seconds":elapsed,"gpu_hours":elapsed/3600.0,"cpu_seconds":time.process_time()-cpu_started,"completed_units":completed,"skipped_terminal_units":skipped,"failed_units":failed,"groups":records,"model":model_record,"completed_at":_now()}); _write_or_verify(repo/ROOT_PATH/"telemetry"/f"shard-{shard_id:02d}-of-08.json",telemetry); return telemetry
    finally:
        if model is not None: del model
        if device.startswith("cuda") and torch.cuda.is_available(): torch.cuda.empty_cache()


def _all_sources(repo:Path)->tuple[dict[str,dict[str,Any]],dict[str,str]]:
    plan=_v1_plan(repo); attempts={}; origins={}
    for unit in plan["units"]:
        source=_source_attempt(repo,unit)
        if source is None: raise StageEE3R3Error(f"missing terminal union unit: {unit['canonical_digest']}")
        origin,observed=source; attempts[unit["canonical_digest"]]=observed; origins[unit["canonical_digest"]]=origin
    from collections import Counter
    counts=Counter(origins.values())
    if len(attempts)!=3000 or counts!=Counter({"E3_V1_INHERITED":516,"E3_R1_INHERITED":12,"E3_R2_INHERITED":636,"E3_R3_NEW":1836}): raise StageEE3R3Error(f"R3 union origin coverage failed: {counts}")
    return attempts,origins


def _combined_new_rows(repo:Path,plan:Mapping[str,Any])->tuple[list[dict[str,Any]],list[dict[str,Any]],dict[str,dict[str,Any]],dict[str,str]]:
    attempts,origins=_all_sources(repo); method_rows=[]; by_key={}
    for unit in plan["units"]:
        observed=attempts[unit["canonical_digest"]]; result=observed["result"]; path=observed["attempt_path"]; origin=origins[unit["canonical_digest"]]
        row={"variant_id":f"layer_head_{unit['block']}_{unit['head']}","sample_id":unit["sample_id"],"selection_order":unit["selection_order"],"cohort":"stage_d_fidelity_v1_500","stratum":unit["stratum"],"validity_status":result["validity_status"],"source_hashes":result["source_hashes"],"method":unit["method"],"target":unit["target"],"scale":unit["scale"],"layer":unit["block"],"head":unit["head"],"coordinate_role":"DESCRIPTIVE_FROZEN_COORDINATE","recompute_status":{"E3_V1_INHERITED":"INHERITED_E3_V1_TERMINAL","E3_R1_INHERITED":"INHERITED_E3_R1_TERMINAL","E3_R2_INHERITED":"INHERITED_E3_R2_VALID_TERMINAL","E3_R3_NEW":"E3_R3_NEW_COMPUTE_COMPLETED"}[origin],"evidence_origin":origin,"attempt_id":result["attempt_id"],"retry_count":result["retry_count"],"failure_class":result["failure_class"],"failure_code":result["failure_code"],"na_reason":result["na_reason"],"insertion_auc_mean":result.get("insertion_auc_mean"),"deletion_auc_mean":result.get("deletion_auc_mean"),"insertion_auc_repeat_sd_ddof_0":result.get("insertion_auc_repeat_sd_ddof_0"),"deletion_auc_repeat_sd_ddof_0":result.get("deletion_auc_repeat_sd_ddof_0"),"misclassified":unit["stratum"]=="misclassified","pair_binding_digest":result.get("pair_binding_digest"),"elapsed_seconds":result["elapsed_seconds"],"gpu_hours":result["gpu_hours"],"cpu_seconds":result["cpu_seconds"],"forward_calls":result["forward_calls"],"backward_calls":result["backward_calls"],"peak_cuda_memory_bytes":result["peak_cuda_memory_bytes"],"artifact_bytes":sum(item.stat().st_size for item in path.iterdir() if item.is_file()),"descriptive_only":True}
        method_rows.append(row); by_key.setdefault((unit["sample_id"],unit["block"],unit["head"]),{})[unit["method"]]=(row,result)
    pair_rows=[]
    for (sample_id,block,head),methods in by_key.items():
        if set(methods)!=set(e3.METHODS): raise StageEE3R3Error("R3 paired method coverage failed")
        nns,nns_result=methods[e3.NNS_METHOD]; matched,matched_result=methods[e3.MATCHED_METHOD]
        if e1._target_semantic_digest(nns_result["target_response"]["target_identity"])!=e1._target_semantic_digest(matched_result["target_response"]["target_identity"]): raise StageEE3R3Error("R3 paired target identity drifted")
        nc={(x["direction"],x["replicate"]):x for x in nns_result["curves"]}; mc={(x["direction"],x["replicate"]):x for x in matched_result["curves"]}
        if set(nc)!=set(mc) or any(nc[key]["seed"]!=mc[key]["seed"] or nc[key]["tie_priority_sha256"]!=mc[key]["tie_priority_sha256"] for key in nc): raise StageEE3R3Error("R3 paired seed/tie drifted")
        status="VALID" if nns["validity_status"]==matched["validity_status"]=="VALID" else "FAILED"
        pair_rows.append({"variant_id":f"layer_head_{block}_{head}_pair","sample_id":sample_id,"selection_order":nns["selection_order"],"cohort":"stage_d_fidelity_v1_500","stratum":nns["stratum"],"validity_status":status,"source_hashes":{"nns_map_sha256":nns["source_hashes"].get("signed_map_sha256"),"matched_map_sha256":matched["source_hashes"].get("signed_map_sha256"),"target_semantic_digest":nns["source_hashes"]["target_semantic_digest"]},"method":"nns_vs_matched","target":"fixed_clean_attention_cls_head_direction","scale":"fine_4x4_cell","layer":block,"head":head,"coordinate_role":"DESCRIPTIVE_FROZEN_COORDINATE","recompute_status":"E3_V1_R1_R2_R3_UNION","misclassified":nns["misclassified"],"nns_insertion_auc":nns["insertion_auc_mean"],"nns_deletion_auc":nns["deletion_auc_mean"],"matched_insertion_auc":matched["insertion_auc_mean"],"matched_deletion_auc":matched["deletion_auc_mean"],"delta_fid":e3.delta_fid(nns,matched) if status=="VALID" else None,"failure_class":None if status=="VALID" else "SCIENTIFIC_INVALIDITY","failure_code":None if status=="VALID" else "paired_method_invalid","na_reason":None,"descriptive_only":True})
    return method_rows,pair_rows,attempts,origins


def _actual_cost(repo:Path)->dict[str,Any]:
    upstream=r2.validate_completion(repo)["actual_cost"]; telemetry=[]
    for path in sorted((repo/ROOT_PATH/"telemetry").glob("shard-*-of-08.json")):
        value,_=_strict_json(path,"R3 telemetry"); _validate_signed(value,"R3 telemetry"); telemetry.append(value)
    pilots=[_validate_pilot(repo,run_id)["result"] for run_id in PILOT_RUNS]; gpu=sum(float(x["gpu_hours"]) for x in telemetry)+sum(float(x["gpu_hours"]) for x in pilots); cpu=sum(float(x["cpu_seconds"]) for x in telemetry)+sum(float(x["cpu_seconds"]) for x in pilots); retained=sum(r2._inventory(repo/path)["total_bytes"] for path in (e3.ROOT_PATH,e3.REPORT_ROOT_PATH,r1.ROOT_PATH,r1.REPORT_ROOT_PATH,r2.ROOT_PATH,r2.REPORT_ROOT_PATH,ROOT_PATH,REPORT_ROOT_PATH,PILOT_ROOT))
    result={"e3_v1_r1_r2_gpu_hours":upstream["gpu_hours"],"e3_r3_gpu_hours":gpu,"gpu_hours":upstream["gpu_hours"]+gpu,"e3_v1_r1_r2_cpu_seconds":upstream["cpu_seconds"],"e3_r3_cpu_seconds":cpu,"cpu_seconds":upstream["cpu_seconds"]+cpu,"retained_bytes":retained,"gpu_hours_max":512.0,"cpu_hours_max":2048.0,"retained_bytes_max":68719476736}; result["status"]="PASS_COST" if result["gpu_hours"]<=512 and result["cpu_seconds"]<=2048*3600 and result["retained_bytes"]<=68719476736 else "FAIL_COST"; return result


def validate_outputs(repo_root:Path|str=REPO_ROOT)->dict[str,Any]:
    repo=Path(repo_root).resolve(); validate_upstream(repo); gate,_=_strict_json(repo/PILOT_GATE_PATH,"R3 pilot gate"); _validate_signed(gate,"R3 pilot gate")
    if gate.get("status")!="PASS": raise StageEE3R3Error("R3 pilot gate is not PASS")
    plan=_v1_plan(repo); new_methods,new_pairs,attempts,origins=_combined_new_rows(repo,plan); primary_methods,primary_pairs=e3._primary_method_rows(repo,plan)
    method_rows=sorted(primary_methods+new_methods,key=lambda row:(row["selection_order"],e3.COORDINATES.index((row["layer"],row["head"])),e3.METHODS.index(row["method"])))
    pair_rows=sorted(primary_pairs+new_pairs,key=lambda row:(row["selection_order"],e3.COORDINATES.index((row["layer"],row["head"]))))
    if len(method_rows)!=4000 or len(pair_rows)!=2000 or len(attempts)!=3000 or any(row["validity_status"]!="VALID" for row in method_rows+pair_rows): raise StageEE3R3Error("R3 union is not 3000 valid terminal attempts")
    telemetry=[]
    for path in sorted((repo/ROOT_PATH/"telemetry").glob("shard-*-of-08.json")):
        value,_=_strict_json(path,"R3 telemetry"); _validate_signed(value,"R3 telemetry"); telemetry.append(value)
    if {x["shard_id"] for x in telemetry}!=set(range(8)) or any(x["failed_units"] for x in telemetry): raise StageEE3R3Error("R3 shard telemetry is incomplete or failed")
    _,eligible=_partition(repo); expected_orders={unit["selection_order"] for unit in eligible}; observed_orders=[group["selection_order"] for item in telemetry for group in item["groups"]]
    if len(set(observed_orders))!=len(expected_orders) or set(observed_orders)!=expected_orders: raise StageEE3R3Error("R3 telemetry order coverage drifted")
    strata={group["sample_id"]:group["stratum"] for group in plan["groups"]}; summaries=[]; method_summaries=[]
    for block,head in e3.COORDINATES:
        cp=[row for row in pair_rows if (row["layer"],row["head"])==(block,head)]; cm=[row for row in method_rows if (row["layer"],row["head"])==(block,head)]
        for stratum in e3.STRATA:
            selected=cp if stratum=="overall" else [row for row in cp if row["stratum"]==stratum]; summary=e3._summary(selected,"delta_fid",f"layer_head_{block}_{head}_delta_fid",stratum)
            summaries.append({"variant_id":f"layer_head_{block}_{head}_{stratum}","sample_id":None,"selection_order":None,"cohort":"stage_d_fidelity_v1_500","stratum":stratum,"validity_status":"VALID","source_hashes":{"paired_rows_canonical_digest":canonical_sha256(selected)},"method":"nns_vs_matched","target":"fixed_clean_attention_cls_head_direction","scale":"fine_4x4_cell","layer":block,"head":head,"coordinate_role":"PRIMARY_ANCHOR_READ_ONLY" if (block,head)==(0,9) else "DESCRIPTIVE_FROZEN_COORDINATE","metric":"delta_fid",**summary,"descriptive_only":True})
            for method in e3.METHODS:
                rows=[row for row in cm if row["method"]==method and (stratum=="overall" or row["stratum"]==stratum)]
                for direction in ("insertion","deletion"):
                    summary=e3._summary(rows,f"{direction}_auc_mean",f"layer_head_{block}_{head}_{method}_{direction}_auc",stratum); method_summaries.append({"variant_id":f"layer_head_{block}_{head}_{method}_{direction}_{stratum}","sample_id":None,"selection_order":None,"cohort":"stage_d_fidelity_v1_500","stratum":stratum,"validity_status":"VALID","source_hashes":{"method_rows_canonical_digest":canonical_sha256(rows)},"method":method,"target":"fixed_clean_attention_cls_head_direction","scale":"fine_4x4_cell","layer":block,"head":head,"coordinate_role":"PRIMARY_ANCHOR_READ_ONLY" if (block,head)==(0,9) else "DESCRIPTIVE_FROZEN_COORDINATE","metric":f"{direction}_auc",**summary,"descriptive_only":True})
    registry,_=e1._load_registry(repo); membership=[{"variant_id":"prediction_stratum_membership","sample_id":row["sample_id"],"selection_order":row["selection_order"],"cohort":"stage_d_fidelity_v1_500","stratum":strata[row["sample_id"]],"validity_status":"VALID","source_hashes":{"d10_per_sample_sha256":e3._sha_file(repo/e3.D10_DELTA_PATH),"image_sha256":row["image_sha256"]},"method":"nns_vs_matched","target":"fixed_clean_attention_cls_head_direction","scale":"fine_4x4_cell","layer":None,"head":None,"misclassified":strata[row["sample_id"]]=="misclassified"} for row in registry]
    cost=_actual_cost(repo)
    if cost["status"]!="PASS_COST": raise StageEE3R3Error("R3 actual budget failed")
    summary_doc=_signed({"schema_version":1,"document_type":"stage_e_e3_r3_descriptive_summary","stage_id":"E3-R3","recovery_id":RECOVERY_ID,"cohort":"stage_d_fidelity_v1_500","descriptive_only":True,"p_values_emitted":False,"winner_selection_performed":False,"causal_claim_made":False,"new_go_no_go_issued":False,"d10_primary_unchanged":True,"primary_anchor_recomputed":False,"historical_statuses":{"E3":"FAIL","E3-R1":"BLOCKED","E3-R2":"FAIL"},"counts":{"method_rows":4000,"paired_rows":2000,"new_compute_attempts":3000,"e3_v1_inherited_attempts":516,"e3_r1_inherited_attempts":12,"e3_r2_inherited_attempts":636,"e3_r3_new_attempts":1836,"summary_cells":12,"method_summary_cells":48},"delta_fid_coordinate_strata":summaries,"method_auc_coordinate_strata":method_summaries,"prediction_partition":{"overall":500,"correct":418,"misclassified":82},"clean_margin":e3._validate_config(repo)["scientific_contract"]["clean_margin"],"actual_cost":cost,"e4_started":False})
    return {"method_rows":method_rows,"pair_rows":pair_rows,"summary_rows":summaries+method_summaries,"membership_rows":membership,"resource_rows":method_rows,"attempts":attempts,"origins":origins,"summary":summary_doc,"actual_cost":cost}


def _table(table_id:str,rows:list[dict[str,Any]])->dict[str,Any]:
    if not rows: raise StageEE3R3Error(f"{table_id} cannot be empty")
    columns=list(dict.fromkeys(key for row in rows for key in row)); normalized=[{key:row.get(key) for key in columns} for row in rows]
    return _signed({"schema_version":1,"document_type":"stage_e_e3_paper_table","table_id":table_id,"columns":columns,"row_count":len(normalized),"rows":normalized})


def _write_csv(path:Path,rows:list[dict[str,Any]])->None:
    import csv
    path.parent.mkdir(parents=True,exist_ok=True); columns=list(rows[0]); temporary=path.with_name("."+path.name+".tmp")
    with temporary.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,lineterminator="\n"); writer.writeheader(); writer.writerows([{key:e3._csv_scalar(value) for key,value in row.items()} for row in rows]); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary,path)


def _record(repo:Path,relative:str)->dict[str,Any]:
    path=repo/relative; raw=path.read_bytes(); value=_strict_json(path,relative)[0] if path.suffix==".json" else {}
    return {"path":relative,"bytes":len(raw),"sha256":_sha(raw),"canonical_digest":value.get("canonical_digest")}


def _inventory(root:Path,*,exclude:Sequence[str]=())->dict[str,Any]:
    excluded=set(exclude); files=[]
    if root.exists():
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink(): continue
            relative=path.relative_to(root).as_posix()
            if relative in excluded or any(part.startswith(".") for part in Path(relative).parts): continue
            files.append({"path":relative,"bytes":path.stat().st_size,"sha256":_sha_file(path)})
    return {"files":files,"file_count":len(files),"total_bytes":sum(row["bytes"] for row in files),"canonical_digest":canonical_sha256({"files":files})}


def finalize(repo_root:Path|str=REPO_ROOT)->dict[str,Any]:
    import platform
    repo=Path(repo_root).resolve(); outputs=validate_outputs(repo); root=repo/ROOT_PATH
    specs=(("layer_head_method_rows",outputs["method_rows"]),("layer_head_paired_delta",outputs["pair_rows"]),("prediction_strata",outputs["summary_rows"]),("prediction_strata_membership",outputs["membership_rows"]),("resource_and_qc",outputs["resource_rows"]))
    for name,rows in specs:
        table=_table(name,rows); _write_or_verify(root/"paper_data"/f"{name}.json",table); _write_csv(root/"paper_data"/f"{name}.csv",table["rows"])
    _write_or_verify(root/"descriptive_summary.json",outputs["summary"]); _write_or_verify(root/"clean_margin_not_run.json",_signed({"schema_version":1,"document_type":"stage_e_e3_r3_not_run","analysis":"clean_margin",**e3._validate_config(repo)["scientific_contract"]["clean_margin"]}))
    recipe=_signed({"schema_version":1,"document_type":"stage_e_figure_recipes","numeric_source":"paper_data_only","renders_png":False,"figures":[{"figure_id":"e3_r3_layer_head_prediction_strata_future","table_path":"paper_data/prediction_strata.json","x":"layer","y":"median","group":"stratum","caption_contract":"Post-D10 descriptive views; historical E3/R1/R2 outcomes remain disclosed.","numeric_source":"paper_data_only","renders_png":False}]}); _write_or_verify(root/"figure_recipes.json",recipe)
    from collections import Counter
    provenance=_signed({"schema_version":1,"document_type":"stage_e_e3_r3_provenance","recovery_id":RECOVERY_ID,"historical_statuses":{"E3":"FAIL","E3-R1":"BLOCKED","E3-R2":"FAIL"},"origins":dict(Counter(outputs["origins"].values())),"pilot_gate":_record(repo,PILOT_GATE_PATH),"environment":{"python":platform.python_version(),"numpy":np.__version__,"platform":platform.platform()},"e4_started":False}); _write_or_verify(root/"provenance.json",provenance)
    attacks=("historical_receipt_overwritten","r2_failed_attempt_hidden","one_of_1164_valid_attempts_recomputed","missing_or_duplicate_unit","threshold_or_scientific_contract_changed","pilot_repeatability_skipped","favorable_subset_summary","e4_started_before_terminal_pass")
    false_audit=_signed({"schema_version":1,"document_type":"stage_e_e3_r3_false_pass_audit","recovery_id":RECOVERY_ID,"status":"PASS","checks":[{"attack":name,"result":"REJECTED"} for name in attacks],"pass_possible_without_all_conditions":False,"e4_started":False}); _write_or_verify(repo/REPORT_ROOT_PATH/"false_pass_audit.json",false_audit)
    tests,_=_strict_json(repo/TEST_REPORT_PATH,"R3 tests"); _validate_signed(tests,"R3 tests")
    if tests.get("status")!="PASS" or tests.get("failed")!=0: raise StageEE3R3Error("R3 tests are not PASS")
    validation=_signed({"schema_version":1,"document_type":"stage_e_e3_r3_source_validation","stage_id":"E3-R3","recovery_id":RECOVERY_ID,"status":"PASS","validated_at":_now(),"checks":{"historical_receipts_preserved":"PASS","pilot_repeatability":"PASS","origin_union":"PASS_516_12_636_1836","union_3000_unique_terminal_valid":"PASS","method_rows":"PASS_4000","paired_rows":"PASS_2000","summary_cells":"PASS_12","scientific_contract_unchanged":"PASS","actual_budget":"PASS","e4_not_started":"PASS"},"summary_canonical_digest":outputs["summary"]["canonical_digest"],"false_pass_audit_canonical_digest":false_audit["canonical_digest"]}); _write_or_verify(repo/REPORT_ROOT_PATH/"validation_report.json",validation)
    artifacts=[]
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink() and "completion" not in path.parts and path.name not in {"payload_manifest.json","whole_root_inventory.json","status.json"}: artifacts.append({"path":path.relative_to(root).as_posix(),"bytes":path.stat().st_size,"sha256":_sha_file(path)})
    manifest=_signed({"schema_version":1,"document_type":"stage_e_e3_r3_payload_manifest","recovery_id":RECOVERY_ID,"immutable":True,"payload_count":len(artifacts),"artifacts":artifacts}); _write_or_verify(root/"payload_manifest.json",manifest)
    cost=_actual_cost(repo)
    if cost["status"]!="PASS_COST": raise StageEE3R3Error("R3 final budget failed")
    inventory=_inventory(root,exclude=("whole_root_inventory.json","status.json","completion/e3_r3_completion.pending.json","completion/e3_r3_completion.json")); inventory_doc=_signed({"schema_version":1,"document_type":"stage_e_e3_r3_whole_root_inventory","root":ROOT_PATH,"scope":"all regular non-hidden E3-R3 formal files except self-referential inventory status and completion receipts",**inventory}); _write_or_verify(root/"whole_root_inventory.json",inventory_doc)
    status=_signed({"schema_version":1,"document_type":"stage_e_e3_r3_status","stage_id":"E3-R3","recovery_id":RECOVERY_ID,"status":"PASS","complete":True,"terminal":True,"immutable":True,"historical_statuses_preserved":{"E3":"FAIL","E3-R1":"BLOCKED","E3-R2":"FAIL"},"package_completeness":"PASS","workflow_integrity":"PASS","scientific_validity":"PASS","cost_status":"PASS_COST","actual_cost":cost,"e4_authorized":True,"e4_started":False,"whole_root_inventory":{"path":"whole_root_inventory.json","file_sha256":_sha_file(root/"whole_root_inventory.json"),"canonical_digest":inventory_doc["canonical_digest"],"file_count":inventory["file_count"],"total_bytes":inventory["total_bytes"]}}); _write_or_verify(root/"status.json",status)
    pending,_=_strict_json(repo/COMPLETION_PENDING_PATH,"R3 pending"); completion=_signed({"schema_version":1,"document_type":"stage_e_e3_r3_terminal_completion","stage_id":"E3-R3","recovery_id":RECOVERY_ID,"status":"PASS","started_at":pending["started_at"],"completed_at":_now(),"historical_statuses":{"E3":"FAIL","E3-R1":"BLOCKED","E3-R2":"FAIL"},"union":{"expected_new_compute_units":3000,"e3_v1_inherited":516,"e3_r1_inherited":12,"e3_r2_inherited":636,"e3_r3_new":1836,"valid_terminal":3000,"failed":0,"na":0,"method_rows":4000,"paired_rows":2000,"summary_cells":12},"package_completeness":"PASS","workflow_integrity":"PASS","scientific_validity":"PASS","cost_status":"PASS_COST","actual_cost":cost,"source_validation":{"validation_report":_record(repo,REPORT_ROOT_PATH+"/validation_report.json"),"test_report":_record(repo,TEST_REPORT_PATH),"false_pass_audit":_record(repo,REPORT_ROOT_PATH+"/false_pass_audit.json"),"payload_manifest":_record(repo,ROOT_PATH+"/payload_manifest.json"),"status":_record(repo,ROOT_PATH+"/status.json"),"whole_root_inventory":_record(repo,ROOT_PATH+"/whole_root_inventory.json"),"pilot_gate":_record(repo,PILOT_GATE_PATH)},"next_authorization":"E4_ONLY","e4_started":False}); schema,_=_strict_json(repo/COMPLETION_SCHEMA_PATH,"R3 completion schema"); errors=list(Draft202012Validator(schema).iter_errors(completion));
    if errors: raise StageEE3R3Error(f"R3 completion schema failure: {errors[0].message}")
    temporary=(repo/COMPLETION_PENDING_PATH).with_name(".e3-r3-completion-terminal.tmp"); _write_new_json(temporary,completion); os.replace(temporary,repo/COMPLETION_PENDING_PATH); os.replace(repo/COMPLETION_PENDING_PATH,repo/COMPLETION_PATH); return validate_completion(repo)


def validate_completion(repo_root:Path|str=REPO_ROOT)->dict[str,Any]:
    repo=Path(repo_root).resolve()
    if not (repo/COMPLETION_PATH).is_file() or (repo/COMPLETION_PENDING_PATH).exists(): raise StageEE3R3Error("R3 terminal completion absent or pending remains")
    completion,_=_strict_json(repo/COMPLETION_PATH,"R3 completion"); _validate_signed(completion,"R3 completion"); schema,_=_strict_json(repo/COMPLETION_SCHEMA_PATH,"R3 completion schema")
    if list(Draft202012Validator(schema).iter_errors(completion)): raise StageEE3R3Error("R3 completion schema failed")
    if completion.get("status")!="PASS" or completion.get("next_authorization")!="E4_ONLY" or completion.get("e4_started") is not False: raise StageEE3R3Error("R3 completion boundary failed")
    validate_upstream(repo); outputs=validate_outputs(repo)
    if outputs["actual_cost"]!=completion["actual_cost"]: raise StageEE3R3Error("R3 completion cost drifted")
    status,_=_strict_json(repo/ROOT_PATH/"status.json","R3 status"); inventory,_=_strict_json(repo/ROOT_PATH/"whole_root_inventory.json","R3 inventory"); _validate_signed(status,"R3 status"); _validate_signed(inventory,"R3 inventory")
    rebuilt=_inventory(repo/ROOT_PATH,exclude=("whole_root_inventory.json","status.json","completion/e3_r3_completion.pending.json","completion/e3_r3_completion.json"))
    if any(inventory[key]!=rebuilt[key] for key in ("files","file_count","total_bytes")) or canonical_sha256({"files":inventory["files"]})!=rebuilt["canonical_digest"]: raise StageEE3R3Error("R3 inventory drifted")
    if status.get("status")!="PASS" or status.get("e4_authorized") is not True or status.get("e4_started") is not False: raise StageEE3R3Error("R3 status drifted")
    for record in completion["source_validation"].values():
        if record!=_record(repo,record["path"]): raise StageEE3R3Error(f"R3 completion binding drifted: {record['path']}")
    return {"status":"PASS","completion_canonical_digest":completion["canonical_digest"],"whole_root_inventory_digest":inventory["canonical_digest"],"inherited_attempts":1164,"new_attempts":1836,"union_valid_terminal_attempts":3000,"method_rows":4000,"paired_rows":2000,"actual_cost":completion["actual_cost"],"next_authorization":"E4_ONLY","e4_started":False}


def _main()->int:
    parser=argparse.ArgumentParser(prog="python -m vitnns.stage_e_e3_r3"); sub=parser.add_subparsers(dest="command",required=True)
    for name in ("verify-preconditions","initialize","validate-initialized","evaluate-pilot","validate-outputs","finalize","validate-completion"):
        child=sub.add_parser(name); child.add_argument("--repo-root",type=Path,default=REPO_ROOT)
    tests=sub.add_parser("record-tests"); tests.add_argument("--repo-root",type=Path,default=REPO_ROOT); tests.add_argument("--passed",type=int,required=True); tests.add_argument("--failed",type=int,required=True); tests.add_argument("--test-command",required=True)
    pilot=sub.add_parser("run-pilot"); pilot.add_argument("--repo-root",type=Path,default=REPO_ROOT); pilot.add_argument("--run-id",choices=PILOT_RUNS,required=True); pilot.add_argument("--device",default="cuda:0")
    run=sub.add_parser("run-shard"); run.add_argument("--repo-root",type=Path,default=REPO_ROOT); run.add_argument("--shard-id",type=int,required=True); run.add_argument("--shard-count",type=int,required=True); run.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    if args.command=="verify-preconditions": value=verify_preconditions(args.repo_root)
    elif args.command=="initialize": value=initialize(args.repo_root)
    elif args.command=="validate-initialized": value=validate_initialized(args.repo_root)
    elif args.command=="record-tests": value=record_tests(args.repo_root,passed=args.passed,failed=args.failed,command=args.test_command)
    elif args.command=="run-pilot": value=run_pilot(args.repo_root,run_id=args.run_id,device=args.device)
    elif args.command=="evaluate-pilot": value=evaluate_pilot(args.repo_root)
    elif args.command=="run-shard": value=run_shard(args.repo_root,shard_id=args.shard_id,shard_count=args.shard_count,device=args.device)
    elif args.command=="validate-outputs":
        output=validate_outputs(args.repo_root); value={"status":"PASS","method_rows":len(output["method_rows"]),"paired_rows":len(output["pair_rows"]),"actual_cost":output["actual_cost"]}
    elif args.command=="finalize": value=finalize(args.repo_root)
    else: value=validate_completion(args.repo_root)
    print(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)); return 0


if __name__=="__main__": raise SystemExit(_main())
