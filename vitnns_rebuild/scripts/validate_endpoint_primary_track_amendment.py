#!/usr/bin/env python3
"""Validate the prospective Endpoint/Path track split without data/model access."""
from __future__ import annotations
import argparse,copy,hashlib,json,os,subprocess
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"]=""
REPO=Path(__file__).resolve().parents[1];CONFIG=REPO/"config/nns_vit_endpoint/endpoint_primary_track_amendment_r1.json";SCHEMA=REPO/"config/nns_vit_endpoint/endpoint_primary_track_amendment_r1.schema.json";REPORT=REPO/"reports/nns_vit_endpoint_scope_amendment_r1_20260918"
def require(c,m):
    if not c:raise ValueError(m)
def canonical(v):return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()
def load(path):return json.loads(path.read_text())
def closed(value,schema,path="root"):
    if "const" in schema:require(type(value)==type(schema["const"]) and value==schema["const"],"SCHEMA_CONST:"+path);return
    kind=schema.get("type")
    if kind=="object":
        require(isinstance(value,dict),"SCHEMA_OBJECT:"+path)
        if schema.get("additionalProperties") is False:require(set(value)==set(schema["required"])==set(schema["properties"]),"SCHEMA_KEYS:"+path)
        for key,item in value.items():
            if key in schema.get("properties",{}):closed(item,schema["properties"][key],path+"."+key)
    elif kind=="array":
        require(isinstance(value,list),"SCHEMA_ARRAY:"+path);require(len(value)>=schema.get("minItems",0) and len(value)<=schema.get("maxItems",len(value)),"SCHEMA_LENGTH:"+path)
        if schema.get("uniqueItems"):require(len({canonical(item) for item in value})==len(value),"SCHEMA_UNIQUE:"+path)
        for index,item in enumerate(value):closed(item,schema["items"],f"{path}[{index}]")
    elif kind=="string":require(isinstance(value,str) and len(value)>=schema.get("minLength",0),"SCHEMA_STRING:"+path)
    elif kind is None:require(isinstance(value,dict),"SCHEMA_OBJECT_OPEN:"+path)
    else:raise ValueError("UNSUPPORTED_SCHEMA:"+path)
def semantic(value):
    status=value["status"];require(status["endpoint_primary_track"]=="ENDPOINT_PRIMARY_TRACK_AUTHORIZED_BY_AMENDMENT" and status["path_dependent_track"]=="PATH_DEPENDENT_TRACK_BLOCKED" and status["ep2_r1"]=="EP2_R1_PATH_NUMERICAL_CALIBRATION_FAILED_REMAINS_IMMUTABLE" and status["original_ep3"]=="NOT_AUTHORIZED" and not status["new_scientific_outcome_access"],"STATUS")
    paths={"ep0_protocol_sha256":"manifests/nns_vit_endpoint/ep0_r1_20260918/protocol.json","ep0_target_registry_sha256":"manifests/nns_vit_endpoint/ep0_r1_20260918/target_registry.json","ep0_statistical_plan_sha256":"manifests/nns_vit_endpoint/ep0_r1_20260918/statistical_analysis_plan.json","ep0_intervention_registry_sha256":"manifests/nns_vit_endpoint/ep0_r1_20260918/intervention_registry.json","ep1_terminal_sha256":"reports/nns_vit_endpoint_ep1_r11_20260918/terminal_receipt.json","ep1_validation_sha256":"reports/nns_vit_endpoint_ep1_r11_20260918/validation_report.json","ep2_r1_terminal_sha256":"reports/nns_vit_endpoint_ep2_r1_20260918/terminal_receipt.json","ep2_r1_calibration_report_sha256":"reports/nns_vit_endpoint_ep2_r1_20260918/calibration_report.json"}
    for key,path in paths.items():require(sha(REPO/path)==value["immutable_bindings"][key],"BINDING:"+key)
    atlas=load(REPO/"manifests/nns_vit_endpoint/ep0_r1_20260918/evaluation_registry.json")["atlas"];require(digest(atlas)==value["immutable_bindings"]["ep0_atlas_contract_digest"],"ATLAS_BINDING")
    ep1=load(REPO/value["authority"]["ep1_terminal"]);ep2=load(REPO/value["authority"]["ep2_r1_terminal"]);require(ep1["status"]=="PASS_EP1_IMPLEMENTATION_AND_REGRESSION" and ep2["status"]=="EP2_NUMERICAL_CALIBRATION_FAILED" and not ep2["evaluation_access"] and not ep2["ep3_started"] and ep2["next_authorization"]=="NONE_EP2_RECOVERY_REQUIRED","PREDECESSORS")
    require(value["method_identity"]["numerical_identity"]=="Gradient x Input on the same registered internal scalar target" and not value["method_identity"]["ig_completeness_claimed"] and not value["method_identity"]["universal_path_ig_replacement_claimed"],"METHOD_BOUNDARY")
    stages={row["stage"]:row["status"] for row in value["endpoint_track"]};require(stages["ENDPOINT-E0"]=="THIS_AMENDMENT" and stages["ENDPOINT-E1"]=="SATISFIED_BY_EP1_REBIND" and stages["ENDPOINT-E2"]=="REQUIRES_EXPLICIT_USER_REQUEST","ENDPOINT_ROUTE");require(value["path_track"][1]["status"]=="FAILED_IMMUTABLE" and value["path_track"][2]["status"]=="BLOCKED_REQUIRES_NEW_PROSPECTIVE_AMENDMENT","PATH_ROUTE")
    scope=value["endpoint_e2_scope"];require(scope["models"]==["vit_b16","deit_s16"] and not scope["path_ig_allowed"] and not scope["input_fidelity_or_stability_allowed"] and not scope["finite_intervention_allowed"] and not scope["scientific_utility_claims_allowed"],"E2_SCOPE");require(value["statistical_binding"]["thresholds_preserved"] and value["statistical_binding"]["holm_scope_preserved"],"STATS_DRIFT");require(value["access"]=={"amendment_new_image_reads":0,"amendment_model_forwards":0,"amendment_gpu_jobs":0,"next_stage":"ENDPOINT-E2","next_authorization":"ENDPOINT_E2_ATLAS_REQUIRES_EXPLICIT_USER_REQUEST","path_dependent_authorization":"NONE"},"ACCESS")
def main():
    p=argparse.ArgumentParser();p.add_argument("--seal",action="store_true");a=p.parse_args();value=load(CONFIG);closed(value,load(SCHEMA));semantic(value);mutations=[("ep2_pass",lambda x:x["status"].__setitem__("ep2_r1","PASS")),("path_unblocked",lambda x:x["status"].__setitem__("path_dependent_track","PASS")),("original_ep3",lambda x:x["status"].__setitem__("original_ep3","AUTHORIZED")),("new_access",lambda x:x["status"].__setitem__("new_scientific_outcome_access",True)),("path_e2",lambda x:x["endpoint_e2_scope"].__setitem__("path_ig_allowed",True)),("drop_model",lambda x:x["endpoint_e2_scope"]["models"].pop()),("allow_utility",lambda x:x["endpoint_e2_scope"].__setitem__("scientific_utility_claims_allowed",True)),("authorize_without_request",lambda x:x["endpoint_track"][2].__setitem__("status","AUTHORIZED"))];attacks=[]
    for name,change in mutations:
        test=copy.deepcopy(value);change(test)
        try:closed(test,load(SCHEMA));semantic(test)
        except (ValueError,KeyError,IndexError):attacks.append({"case":name,"rejected":True})
        else:raise ValueError("FALSE_PASS:"+name)
    summary={"status":"PASS_ENDPOINT_PRIMARY_TRACK_SCOPE_AMENDMENT","endpoint_primary_track":"AUTHORIZED_BY_AMENDMENT","path_dependent_track":"BLOCKED","ep2_r1":"IMMUTABLE_FAILURE","original_ep3":"NOT_AUTHORIZED","endpoint_e1":"SATISFIED_BY_EP1_REBIND","next_authorization":value["access"]["next_authorization"],"new_image_reads":0,"model_forwards":0,"gpu_jobs":0,"mutation_tests":attacks,"amendment_digest":digest(value)}
    if a.seal:
        require(not REPORT.exists(),"REPORT_EXISTS");REPORT.mkdir(parents=True);files=[]
        for path in [CONFIG,SCHEMA,Path(__file__),REPO/"docs/NNS_ViT_Endpoint_Scanner主轨拆分修订_20260918.md"]:files.append({"path":path.relative_to(REPO).as_posix(),"bytes":path.stat().st_size,"sha256":sha(path)})
        (REPORT/"validation_report.json").write_text(json.dumps(summary,sort_keys=True,indent=2,ensure_ascii=False)+"\n");manifest={"files":files,"canonical_digest":digest(files)};(REPORT/"artifact_manifest.json").write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n");terminal={"status":summary["status"],"endpoint_primary_track":"ENDPOINT_PRIMARY_TRACK_AUTHORIZED_BY_AMENDMENT","path_dependent_track":"PATH_DEPENDENT_TRACK_BLOCKED","ep2_r1":"EP2_R1_REMAINS_IMMUTABLE","original_ep3":"NOT_AUTHORIZED","next_authorization":summary["next_authorization"],"validation_digest":digest(summary),"manifest_digest":manifest["canonical_digest"]};(REPORT/"terminal_receipt.json").write_text(json.dumps(terminal,sort_keys=True,indent=2)+"\n")
    elif REPORT.exists():
        manifest=load(REPORT/"artifact_manifest.json");require(manifest["canonical_digest"]==digest(manifest["files"]),"MANIFEST_DIGEST")
        for row in manifest["files"]:path=REPO/row["path"];require(path.is_file() and path.stat().st_size==row["bytes"] and sha(path)==row["sha256"],"ARTIFACT_HASH:"+row["path"])
        terminal=load(REPORT/"terminal_receipt.json");require(terminal["validation_digest"]==digest(load(REPORT/"validation_report.json")) and terminal["manifest_digest"]==manifest["canonical_digest"],"TERMINAL_BINDING");require(load(REPORT/"validation_report.json")==summary,"REPORT_RECOMPUTE")
    print(json.dumps({k:v for k,v in summary.items() if k!="mutation_tests"},sort_keys=True));print("negative mutations rejected:",len(attacks))
if __name__=="__main__":main()
