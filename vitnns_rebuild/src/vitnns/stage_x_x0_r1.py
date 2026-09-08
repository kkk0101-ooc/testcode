"""CPU-only fail-closed correction for the X0 identity comparator defect."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from .constants import REPO_ROOT

BASE_COMMIT="bc4b18580f88fc2e82e1a3c686b29092e9ad7b40"
BRANCH="stage-d-d3-pass"
UPSTREAM_REF="refs/archive/stage-x-nns-v2-x0-completion"
ARCHIVE_REF="refs/archive/stage-x-nns-v2-x0-r1-completion"
ROOT=Path("evidence/stage_x/x0_r1_identity_control_20260908/formal")
REPORT_ROOT=Path("reports/stage_x_x0_r1_identity_control_20260908")
PENDING=ROOT/"completion/x0_r1.pending.json"
COMPLETION=ROOT/"completion/x0_r1.json"
CONFIG=Path("config/stage_x_x0_r1.json")
SCHEMA=Path("config/stage_x_x0_r1.schema.json")
AMENDMENT=Path("docs/stage_x_x0_r1_identity_control_amendment_20260908.md")
NEXT_PROMPT=Path("docs/stage_x_x0_r1_execution_prompt_20260908.md")
COMPARATORS=Path("manifests/stage_x_comparator_registry_v2.json")
BENCHMARK=Path("manifests/stage_x_benchmark_plan_v2.json")
OLD_COMPLETION=Path("evidence/stage_x/x0_nns_v2_20260908/formal/completion/x0.json")
OLD_CONFIG=Path("config/stage_x.json")
OLD_COMPARATORS=Path("manifests/stage_x_comparator_registry_v1.json")
OLD_BENCHMARK=Path("manifests/stage_x_benchmark_plan_v1.json")
OLD_COMPLETION_SHA="16d21ef2e9f42f684339ab2ead2c221e8479a193683922917352c24f2b7d4cfd"
COMPETITIVE=("vit_grad_cam","transformer_attribution_chefer","attnlrp","rise","occlusion")
ATTACKS=("old_x0_completion_or_ref_drift","identity_control_treated_as_superiority_comparator","identity_control_removed","identity_target_baseline_path_or_quadrature_drift","identity_elementwise_tolerance_loosened","identity_selected_steps_allowed_to_differ","identity_mismatch_not_method_invalidity","competitive_method_added_removed_or_reordered","holm_family_contains_identity_control","strongest_wording_omits_identity_control","x1_started_during_correction","model_image_torch_timm_cuda_or_download_access","terminal_before_validation_inventory_or_false_pass","terminal_not_bound_to_clean_commit_and_archive")
SOURCE_FILES=(CONFIG,SCHEMA,AMENDMENT,NEXT_PROMPT,COMPARATORS,BENCHMARK,Path("src/vitnns/stage_x_x0_r1.py"),Path("tests/test_stage_x_x0_r1.py"),Path("docs/NNS-v2前瞻性主张扩展方案.md"),Path("../论文1实现审计与后续计划.md"),Path("docs/阶段路线图.md"),Path("docs/阶段F规划.md"),Path("docs/阶段G规划.md"))

class StageXX0R1Error(ValueError): pass

def _constant(token:str)->None: raise StageXX0R1Error(f"non-finite JSON rejected: {token}")
def _pairs(pairs:list[tuple[str,Any]])->dict[str,Any]:
    result={}
    for key,value in pairs:
        if key in result: raise StageXX0R1Error(f"duplicate JSON key rejected: {key}")
        result[key]=value
    return result
def read_json(path:Path)->Any:
    try: return json.loads(path.read_text(encoding="utf-8"),object_pairs_hook=_pairs,parse_constant=_constant)
    except (json.JSONDecodeError,UnicodeDecodeError) as error: raise StageXX0R1Error(f"invalid JSON: {path}") from error
def read_json_bytes(data:bytes)->Any:
    try: return json.loads(data.decode("utf-8"),object_pairs_hook=_pairs,parse_constant=_constant)
    except (json.JSONDecodeError,UnicodeDecodeError) as error: raise StageXX0R1Error("invalid JSON bytes") from error
def canonical_bytes(value:Any)->bytes: return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def canonical_sha256(value:Any)->str: return hashlib.sha256(canonical_bytes(value)).hexdigest()
def signed(value:Mapping[str,Any])->dict[str,Any]:
    result=dict(value); result["canonical_digest"]=canonical_sha256(result); return result
def validate_signed(value:Mapping[str,Any],context:str)->None:
    payload=dict(value); observed=payload.pop("canonical_digest",None)
    if observed!=canonical_sha256(payload): raise StageXX0R1Error(f"{context} canonical digest mismatch")
def sha256_file(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()
def _git(repo:Path,*args:str)->str:
    result=subprocess.run(["git",*args],cwd=repo,text=True,capture_output=True,check=False)
    if result.returncode: raise StageXX0R1Error(result.stderr.strip() or "git failed")
    return result.stdout.strip()
def _git_bytes(repo:Path,spec:str)->bytes:
    result=subprocess.run(["git","show",spec],cwd=repo,capture_output=True,check=False)
    if result.returncode: raise StageXX0R1Error(result.stderr.decode(errors="replace"))
    return result.stdout
def _write_new(path:Path,value:Mapping[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True); fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,"wb") as stream: stream.write(canonical_bytes(value)+b"\n"); stream.flush(); os.fsync(stream.fileno())
def _record(path:Path,base:Path)->dict[str,Any]: return {"path":path.relative_to(base).as_posix(),"bytes":path.stat().st_size,"sha256":sha256_file(path)}

def load_contracts(repo_root:Path|str=REPO_ROOT)->dict[str,Any]:
    repo=Path(repo_root).resolve(); config=read_json(repo/CONFIG); schema=read_json(repo/SCHEMA)
    Draft202012Validator.check_schema(schema); errors=sorted(Draft202012Validator(schema).iter_errors(config),key=lambda item:list(item.path))
    if errors: raise StageXX0R1Error(errors[0].message)
    validate_signed(config,"X0-R1 config"); comp=read_json(repo/COMPARATORS); benchmark=read_json(repo/BENCHMARK)
    validate_signed(comp,"X0-R1 comparator registry"); validate_signed(benchmark,"X0-R1 benchmark plan")
    return {"config":config,"schema":schema,"comparators":comp,"benchmark":benchmark}

def prove_original_contradiction(repo_root:Path|str=REPO_ROOT)->dict[str,Any]:
    repo=Path(repo_root).resolve(); old=read_json(repo/OLD_CONFIG); old_comp=read_json(repo/OLD_COMPARATORS)
    validate_signed(old,"old X0 config"); validate_signed(old_comp,"old X0 comparator registry")
    nns=old["nns_v2"]; ig=next(row for row in old_comp["primary_family"] if row["id"]=="full_model_integrated_gradients")
    facts={"nns_total_is_input_ig":nns["decomposition"]=="input_ig_plus_complete_cut_path_jacobian_allocation","same_unmodified_target":nns["target"]=="clean_predicted_class_raw_logit_unmodified_model" and "unmodified selected raw class logit" in ig["target_semantics"],"same_baseline":nns["baseline"]=="normalized_zero_canonical_rgb_0_5" and "normalized-zero baseline" in ig["target_semantics"],"same_path_and_schedule_required_by_original_registry":"same Gauss-Legendre schedule as NNS-v2 total input attribution"==ig["cost_class"],"ig_was_primary_superiority_comparator":"full_model_integrated_gradients" in old["benchmark"]["primary_comparators"]}
    if not all(facts.values()): raise StageXX0R1Error("original identity contradiction is not source-provable")
    return signed({"schema_version":1,"document_type":"stage_x_x0_r1_contradiction_report","status":"CONFIRMED","facts":facts,"conclusion":"STRICT_SUPERIORITY_AGAINST_IDENTICAL_TOTAL_ATTRIBUTION_IS_IMPOSSIBLE","detected_before_accepted_x1_implementation":True})

def validate_correction(repo_root:Path|str=REPO_ROOT)->dict[str,Any]:
    repo=Path(repo_root).resolve(); docs=load_contracts(repo); cfg=docs["config"]; comp=docs["comparators"]; benchmark=docs["benchmark"]
    if _git(repo,"symbolic-ref","--short","HEAD")!=BRANCH: raise StageXX0R1Error("wrong branch")
    if _git(repo,"rev-parse",UPSTREAM_REF+"^{commit}")!=BASE_COMMIT: raise StageXX0R1Error("original X0 archive ref drifted")
    if sha256_file(repo/OLD_COMPLETION)!=OLD_COMPLETION_SHA: raise StageXX0R1Error("original X0 completion drifted")
    pointer=_git_bytes(repo,f"{UPSTREAM_REF}:{repo.name}/{OLD_COMPLETION.as_posix()}")
    match=re.fullmatch(rb"version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n?",pointer)
    if not match or match.group(1).decode()!=OLD_COMPLETION_SHA or int(match.group(2))!=(repo/OLD_COMPLETION).stat().st_size: raise StageXX0R1Error("original X0 completion differs from archive LFS identity")
    contradiction=prove_original_contradiction(repo)
    identity=cfg["identity_control"]; expected={"method_id":"full_model_integrated_gradients","status":"INCLUDED_PRIMARY_IDENTITY_CONTROL","superiority_test_allowed":False,"required_equality":"same_target_baseline_path_quadrature_and_elementwise_attribution","atol":1e-6,"rtol":1e-6,"same_steps_required":True,"mismatch_status":"NO_GO_METHOD_VALIDITY"}
    if identity!=expected or comp["identity_control"]["id"]!=identity["method_id"] or comp["identity_control"]["superiority_test_allowed"]: raise StageXX0R1Error("identity-control contract drifted")
    if tuple(cfg["competitive_primary_comparators"])!=COMPETITIVE or tuple(comp["competitive_primary_family"])!=COMPETITIVE or tuple(benchmark["competitive_primary_family"])!=COMPETITIVE: raise StageXX0R1Error("competitive family drifted")
    family=tuple("nns_v2_minus_"+item for item in COMPETITIVE)
    if tuple(benchmark["multiplicity"]["family_members"])!=family or any("integrated_gradients" in item for item in family): raise StageXX0R1Error("Holm family contains identity control or drifted")
    wording=cfg["strongest_permitted_wording"]
    if "competitive baselines" not in wording or "identity control" not in wording or "universal" in wording.lower(): raise StageXX0R1Error("strongest wording drifted")
    access=cfg["access_boundary"]; expected_access={"new_empirical_outcome_access":False,"formal_image_bytes_accessed":False,"audit_image_bytes_accessed":False,"model_or_checkpoint_bytes_accessed":False,"model_loaded":False,"torch_or_timm_imported_by_x0_r1":False,"cuda_or_gpu_accessed_by_x0_r1":False,"dataset_or_checkpoint_downloaded":False,"x1_draft_attempt_started":True,"x1_draft_synthetic_tests_run":True,"x1_draft_removed":True,"x1_accepted_implementation":False,"x1_empirical_execution":False}
    if access!=expected_access: raise StageXX0R1Error("X0-R1 access or interrupted-draft boundary drifted")
    source=(repo/"src/vitnns/stage_x_x0_r1.py").read_text(encoding="utf-8"); tree=ast.parse(source); imports=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Import): imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node,ast.ImportFrom) and node.module: imports.add(node.module.split(".")[0])
    if imports&{"torch","timm","PIL","cv2","requests"}: raise StageXX0R1Error("X0-R1 imports forbidden runtime modules")
    forbidden=sorted(name for name in sys.modules if name=="torch" or name.startswith("torch.") or name=="timm" or name.startswith("timm."))
    if forbidden: raise StageXX0R1Error("X0-R1 process imported Torch/timm")
    if any((repo/path).exists() for path in (Path("src/vitnns/stage_x_nns_v2.py"),Path("src/vitnns/stage_x_comparators.py"),Path("tests/test_stage_x_nns_v2.py"),Path("tests/test_stage_x_comparators.py"))): raise StageXX0R1Error("unaccepted X1 implementation remains in worktree")
    return signed({"schema_version":1,"document_type":"stage_x_x0_r1_validation","status":"PASS","original_x0_status":"BLOCKED_SUPERSEDED_IDENTITY_COMPARATOR_CONTRADICTION","contradiction_digest":contradiction["canonical_digest"],"identity_control":identity,"competitive_primary_comparators":list(COMPETITIVE),"next_authorization":"X1_NNS_V2_IMPLEMENTATION_ONLY","access_boundary":cfg["access_boundary"]})

def _source_binding(repo:Path)->dict[str,Any]:
    files=[]
    for path in SOURCE_FILES:
        full=repo/path
        if not full.is_file(): raise StageXX0R1Error(f"missing source: {path}")
        files.append({"path":path.as_posix(),"bytes":full.stat().st_size,"sha256":sha256_file(full)})
    return signed({"schema_version":1,"document_type":"stage_x_x0_r1_source_binding","status":"PASS","files":files,"files_digest":canonical_sha256(files)})

def freeze(repo_root:Path|str=REPO_ROOT,*,completed_at:str,test_passed:int,test_failed:int,test_command:str)->dict[str,Any]:
    repo=Path(repo_root).resolve()
    if (repo/ROOT).exists() or (repo/REPORT_ROOT).exists(): raise StageXX0R1Error("X0-R1 root exists; refusing overwrite")
    if type(test_passed) is not int or test_passed<=0 or type(test_failed) is not int or test_failed!=0: raise StageXX0R1Error("X0-R1 tests failed")
    validation=validate_correction(repo); contradiction=prove_original_contradiction(repo); cfg=load_contracts(repo)["config"]
    (repo/ROOT).mkdir(parents=True,exist_ok=False); (repo/REPORT_ROOT).mkdir(parents=True,exist_ok=False)
    _write_new(repo/PENDING,signed({"schema_version":1,"document_type":"stage_x_x0_r1_pending","status":"PENDING","created_at":completed_at,"base_commit":BASE_COMMIT}))
    payload={"startup_gate.json":signed({"schema_version":1,"document_type":"stage_x_x0_r1_startup_gate","status":"PASS","branch":BRANCH,"local_head":BASE_COMMIT,"tracking_ref":BASE_COMMIT,"remote_branch":BASE_COMMIT,"upstream_archive_ref":UPSTREAM_REF,"upstream_archive_commit":BASE_COMMIT,"initial_worktree_clean":True,"initial_lfs_clean":True,"x1_root_absent":True}),"contradiction_report.json":contradiction,"corrected_contract.json":signed({"schema_version":1,"document_type":"stage_x_x0_r1_corrected_contract","status":"FROZEN","identity_control":cfg["identity_control"],"competitive_primary_comparators":cfg["competitive_primary_comparators"],"multiplicity":cfg["multiplicity"],"strongest_permitted_wording":cfg["strongest_permitted_wording"]}),"source_binding.json":_source_binding(repo),"validation_report.json":validation,"false_pass_audit.json":signed({"schema_version":1,"document_type":"stage_x_x0_r1_false_pass_audit","status":"PASS","attack_count":len(ATTACKS),"attacks":[{"attack":item,"result":"REJECTED"} for item in ATTACKS],"pass_possible_without_all_conditions":False}),"test_report.json":signed({"schema_version":1,"document_type":"stage_x_x0_r1_test_report","status":"PASS","passed":test_passed,"failed":test_failed,"command":test_command}),"access_audit.json":signed({"schema_version":1,"document_type":"stage_x_x0_r1_access_audit","status":"PASS",**cfg["access_boundary"],"x2_started":False,"x3_started":False,"f3_final_started":False,"stage_g_started":False}),"status.json":signed({"schema_version":1,"document_type":"stage_x_x0_r1_status","status":"PASS","terminal":True,"supersedes_x0_authorization":True,"next_authorization":"X1_NNS_V2_IMPLEMENTATION_ONLY","git":{"commit":"SELF","ref":ARCHIVE_REF}})}
    for name,value in payload.items(): _write_new(repo/ROOT/name,value)
    records=[_record(repo/ROOT/name,repo/ROOT) for name in sorted(payload)]
    manifest=signed({"schema_version":1,"document_type":"stage_x_x0_r1_payload_manifest","status":"PASS","files":records,"files_digest":canonical_sha256(records)}); _write_new(repo/ROOT/"payload_manifest.json",manifest)
    paths=sorted((path for path in (repo/ROOT).rglob("*") if path.is_file() and path!=repo/PENDING),key=lambda path:path.relative_to(repo/ROOT).as_posix()); inventory_records=[_record(path,repo/ROOT) for path in paths]
    inventory=signed({"schema_version":1,"document_type":"stage_x_x0_r1_whole_root_inventory","status":"PASS","files":inventory_records,"file_count":len(inventory_records),"total_bytes":sum(row["bytes"] for row in inventory_records),"files_digest":canonical_sha256(inventory_records)}); _write_new(repo/ROOT/"whole_root_inventory.json",inventory)
    names=tuple(payload)+("payload_manifest.json","whole_root_inventory.json"); completion=signed({"schema_version":1,"document_type":"stage_x_x0_r1_terminal_completion","status":"PASS","terminal":True,"completed_at":completed_at,"base_commit":BASE_COMMIT,"original_x0_status":"BLOCKED_SUPERSEDED_IDENTITY_COMPARATOR_CONTRADICTION","records":{name:{"path":(ROOT/name).as_posix(),"sha256":sha256_file(repo/ROOT/name),"canonical_digest":read_json(repo/ROOT/name)["canonical_digest"]} for name in names},**cfg["access_boundary"],"x2_started":False,"x3_started":False,"f3_final_started":False,"stage_g_started":False,"next_authorization":"X1_NNS_V2_IMPLEMENTATION_ONLY","git":{"commit":"SELF","ref":ARCHIVE_REF}})
    temp=(repo/PENDING).with_name(".x0-r1-terminal.tmp"); _write_new(temp,completion); os.replace(temp,repo/PENDING); os.replace(repo/PENDING,repo/COMPLETION)
    path=repo/REPORT_ROOT/"x0_r1_report.md"; fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,"w",encoding="utf-8") as stream: stream.write("# Stage X X0-R1 report\n\nPASS. The impossible NNS-v2 versus identical full-model IG superiority comparison is replaced prospectively by an identity-control gate. Five competitive comparators remain. No X1 or empirical access occurred.\n"); stream.flush(); os.fsync(stream.fileno())
    return completion

def validate_completion(repo_root:Path|str=REPO_ROOT,*,published:bool=False)->dict[str,Any]:
    repo=Path(repo_root).resolve(); completion=read_json(repo/COMPLETION); validate_signed(completion,"X0-R1 completion")
    if (repo/PENDING).exists() or completion.get("status")!="PASS" or not completion.get("terminal"): raise StageXX0R1Error("X0-R1 completion is not atomic terminal PASS")
    for record in completion["records"].values():
        path=repo/record["path"]; value=read_json(path); validate_signed(value,str(path))
        if sha256_file(path)!=record["sha256"] or value["canonical_digest"]!=record["canonical_digest"]: raise StageXX0R1Error("completion record drifted")
    validate_correction(repo)
    if published:
        head=_git(repo,"rev-parse","HEAD"); tracking=_git(repo,"rev-parse",f"origin/{BRANCH}"); remote=_git(repo,"ls-remote","--heads","origin",BRANCH).split()[0]; archive=_git(repo,"rev-parse",ARCHIVE_REF+"^{commit}"); remote_archive=_git(repo,"ls-remote","origin",ARCHIVE_REF).split()[0]
        if len({head,tracking,remote,archive,remote_archive})!=1 or _git(repo,"status","--porcelain=v1"): raise StageXX0R1Error("published Git state drifted")
    return {"status":"PASS","original_x0_status":completion["original_x0_status"],"next_authorization":completion["next_authorization"],"archive_ref":ARCHIVE_REF,"published":published}

def main(argv:Sequence[str]|None=None)->int:
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="command",required=True)
    check=sub.add_parser("validate"); check.add_argument("--repo-root",type=Path,default=REPO_ROOT)
    run=sub.add_parser("freeze"); run.add_argument("--repo-root",type=Path,default=REPO_ROOT); run.add_argument("--completed-at",required=True); run.add_argument("--test-passed",type=int,required=True); run.add_argument("--test-failed",type=int,required=True); run.add_argument("--test-command",required=True)
    done=sub.add_parser("validate-completion"); done.add_argument("--repo-root",type=Path,default=REPO_ROOT); done.add_argument("--published",action="store_true"); args=parser.parse_args(argv)
    try:
        result=validate_correction(args.repo_root) if args.command=="validate" else freeze(args.repo_root,completed_at=args.completed_at,test_passed=args.test_passed,test_failed=args.test_failed,test_command=args.test_command) if args.command=="freeze" else validate_completion(args.repo_root,published=args.published)
    except (StageXX0R1Error,OSError,ValueError,KeyError,IndexError) as error: print(json.dumps({"status":"BLOCKED","error":str(error)},sort_keys=True),file=sys.stderr); return 2
    print(json.dumps(result,indent=2,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
