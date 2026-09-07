"""CPU-only validator and freezer for the prospective F3C protocol."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from .constants import REPO_ROOT
from .stage_d_provenance import canonical_json_bytes, canonical_sha256

CONFIG_PATH="config/stage_f_f3c.json"
SCHEMA_PATH="config/stage_f_f3c.schema.json"
PROTOCOL_PATH="docs/stage_f_f3c_class_targeted_protocol_20260907.md"
SPLIT_PATH="manifests/stage_f_f3c_v1/sample_split.json"
ROOT_PATH="evidence/stage_f/f3c0_protocol_v1_20260907/formal"
REPORT_PATH="reports/stage_f_f3c0_protocol_v1_20260907"
PENDING_PATH=ROOT_PATH+"/completion/f3c0.pending.json"
COMPLETION_PATH=ROOT_PATH+"/completion/f3c0.json"
STATUS_PATH=ROOT_PATH+"/status.json"
VALIDATION_PATH=REPORT_PATH+"/validation_report.json"
FALSE_PASS_PATH=REPORT_PATH+"/false_pass_audit.json"
TEST_PATH=REPORT_PATH+"/test_report.json"
ARCHIVE_REF="refs/archive/stage-f-v1-f3c0-protocol-freeze"
F3H_REF="refs/archive/stage-f-v1-f3h-completion"
F3H_COMMIT="56afedd2e73a760dc5127ab4ee9c38d72623dbbc"
METHODS=("conditional_all_blocks_fixed_clean_attention_class_logit_ig","vit_grad_cam_block11_norm1","full_model_class_logit_ig","attention_rollout_mean_head_residual_all_blocks","direct_raw_cls_attention_last_block_head_mean")
EXCLUDED=(14,44,117,145,231,281,326,352,372,380,387)
ATTACKS=("f3h_not_terminal_pass","f3h_basis_result_ignored","f3h_head_maps_posthoc_weighted","method_set_reduced","class_agnostic_baseline_relabelled_class_targeted","target_reselected_per_method_or_perturbation","pilot_or_formal_sample_visual_selection","viewed_sample_in_formal_statistics","pilot_and_formal_overlap","method_specific_display_gain","signed_branches_collapsed_to_abs","nonnegative_negative_map_fabricated","cross_target_conservation_claim","object_localization_claim_without_ground_truth","failed_or_na_unit_dropped_or_imputed","pilot_gate_uses_map_appearance","budget_waiver_or_method_reduction","f3c_empirical_execution_during_f3c0","f4_started","terminal_pass_before_tests_and_validation")

class StageFF3CProtocolError(ValueError): pass

def _sha(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()

def _read(path:Path)->dict[str,Any]:
    value=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value,dict):raise StageFF3CProtocolError(f"not an object: {path}")
    return value

def _signed(value:Mapping[str,Any])->dict[str,Any]:
    result=dict(value);result["canonical_digest"]=canonical_sha256(result);return result

def _validate_signed(value:Mapping[str,Any],context:str)->None:
    payload=dict(value);observed=payload.pop("canonical_digest",None)
    if observed!=canonical_sha256(payload):raise StageFF3CProtocolError(f"{context} canonical digest mismatch")

def _git(repo:Path,*args:str)->str:
    result=subprocess.run(["git",*args],cwd=repo,text=True,capture_output=True,check=False)
    if result.returncode:raise StageFF3CProtocolError(result.stderr.strip() or "git failed")
    return result.stdout.strip()

def _write_new(path:Path,value:Mapping[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True);fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,"wb") as stream:stream.write(canonical_json_bytes(value)+b"\n");stream.flush();os.fsync(stream.fileno())

def load_config(repo_root:Path|str=REPO_ROOT)->dict[str,Any]:
    repo=Path(repo_root).resolve();config=_read(repo/CONFIG_PATH);schema=_read(repo/SCHEMA_PATH);errors=sorted(Draft202012Validator(schema).iter_errors(config),key=lambda e:list(e.path))
    if errors:raise StageFF3CProtocolError(errors[0].message)
    _validate_signed(config,"F3C config");return config

def validate_protocol(repo_root:Path|str=REPO_ROOT,*,require_clean:bool=False)->dict[str,Any]:
    repo=Path(repo_root).resolve();config=load_config(repo)
    if _git(repo,"symbolic-ref","--short","HEAD")!="stage-d-d3-pass":raise StageFF3CProtocolError("wrong branch")
    if require_clean and _git(repo,"status","--porcelain=v1"):raise StageFF3CProtocolError("F3C0 freeze requires clean worktree")
    if _git(repo,"rev-parse",F3H_REF+"^{commit}")!=F3H_COMMIT:raise StageFF3CProtocolError("F3H archive ref drifted")
    upstream=config["upstream"];completion_path=repo/upstream["f3h_completion_path"];completion=_read(completion_path);_validate_signed(completion,"F3H completion")
    if _sha(completion_path)!=upstream["f3h_completion_sha256"] or completion["canonical_digest"]!=upstream["f3h_completion_canonical_digest"] or completion.get("status")!="PASS" or completion.get("next_authorization")!="F3C0_PROTOCOL_FREEZE_ONLY" or completion.get("f4_started"):raise StageFF3CProtocolError("F3H completion binding drifted")
    basis_path=repo/upstream["basis_assessment_path"];basis=_read(basis_path);_validate_signed(basis,"F3H basis assessment")
    if _sha(basis_path)!=upstream["basis_assessment_sha256"] or basis["canonical_digest"]!=upstream["basis_assessment_canonical_digest"] or basis["status"]!=upstream["required_basis_status"] or basis["atlas_visual_appearance_used_to_define_weights"]:raise StageFF3CProtocolError("F3H basis assessment drifted")
    for context,path_key,digest_key in (("protocol","path","sha256"),("schema","schema_path","schema_sha256")):
        if _sha(repo/config["protocol"][path_key])!=config["protocol"][digest_key]:raise StageFF3CProtocolError(f"F3C {context} drifted")
    if _sha(repo/"requirements.lock.txt")!=config["protocol"]["requirements_lock_sha256"]:raise StageFF3CProtocolError("requirements lock drifted")
    for path,digest in config["protocol"]["existing_kernel_hashes"].items():
        if _sha(repo/path)!=digest:raise StageFF3CProtocolError(f"existing kernel drifted: {path}")
    split_path=repo/config["sample_split"]["path"];split=_read(split_path);_validate_signed(split,"F3C split")
    if _sha(split_path)!=config["sample_split"]["sha256"] or split["canonical_digest"]!=config["sample_split"]["canonical_digest"]:raise StageFF3CProtocolError("F3C sample split drifted")
    registry=_read(repo/split["registry"]["path"]);_validate_signed(registry,"D1 registry")
    if _sha(repo/split["registry"]["path"])!=split["registry"]["sha256"] or registry["canonical_digest"]!=split["registry"]["canonical_digest"] or len(registry["rows"])!=500:raise StageFF3CProtocolError("D1 registry binding drifted")
    eligible=[row for row in registry["rows"] if row["selection_order"] not in EXCLUDED]
    namespace=split["selection_rule"]["namespace"]
    eligible.sort(key=lambda row:(hashlib.sha256((namespace+row["sample_id"]).encode("utf-8")).hexdigest(),row["selection_order"]))
    keys=("selection_order","sample_id","relative_path","image_sha256","imagenet1k_index","canonical_digest")
    compact=lambda row:{key:row[key] for key in keys}
    if split["pilot"]!=[compact(row) for row in eligible[:5]] or split["formal"]!=[compact(row) for row in eligible[5:55]]:raise StageFF3CProtocolError("F3C sample split is not source-recomputable")
    pilot={x["selection_order"] for x in split["pilot"]};formal={x["selection_order"] for x in split["formal"]}
    if len(pilot)!=5 or len(formal)!=50 or pilot&formal or (pilot|formal)&set(EXCLUDED) or split["counts"]["statistical_n"]!=50 or split["visual_replacement_allowed"]:raise StageFF3CProtocolError("F3C sample split semantics drifted")
    methods=config["methods"]
    if tuple(item["id"] for item in methods)!=METHODS:raise StageFF3CProtocolError("F3C method set/labels drifted")
    expected_flags=((True,False,True,True),(True,False,False,False),(True,False,False,True),(False,True,False,False),(False,True,False,False))
    observed_flags=tuple((item["class_targeted"],item["class_agnostic"],item["conditional_functional"],item["signed"]) for item in methods)
    if observed_flags!=expected_flags:raise StageFF3CProtocolError("F3C method semantic labels drifted")
    nns=config["class_targeted_nns"]
    if nns["manual_head_weights"] or nns["f3h_head_map_weighted_sum"] or nns["attention_scope"]!="all_12_blocks_all_12_heads_full_attention_tensor" or not nns["attention_frozen_during_ig"]:raise StageFF3CProtocolError("class-targeted NNS definition drifted")
    if nns["ig_schedule"]!=[512,1024,2048,4096,8192] or nns["ig_step_chunk_size"]!=8 or nns["endpoint_logits_max_abs_error_lt"]!=1e-6:raise StageFF3CProtocolError("class-targeted NNS numeric contract drifted")
    if config["target"]["class_index_rule"]!="single_clean_input_argmax_then_frozen_for_all_methods_and_interventions" or not config["target"]["misclassified_samples_retained"]:raise StageFF3CProtocolError("F3C target contract drifted")
    if config["evaluation"]["primary_contrast"]!="class_targeted_nns_minus_vit_grad_cam" or config["evaluation"]["formal_statistical_unit"]!="image" or config["evaluation"]["formal_bootstrap_resamples"]!=10000 or config["evaluation"]["object_localization_claim_allowed"]:raise StageFF3CProtocolError("F3C evaluation contract drifted")
    display=config["display"];budget=config["budget"]
    if not display["shared_scale_within_sample_all_five_methods"] or display["method_specific_gain_gamma_alpha_allowed"] or display["percentile_clipping_allowed"] or display["negative_map_for_nonnegative_methods_allowed"]:raise StageFF3CProtocolError("F3C display contract drifted")
    if budget["waiver_allowed"] or not budget["pilot_reprojection_required_before_formal"] or budget["pilot_samples"]!=5 or budget["formal_samples"]!=50 or budget["methods_per_sample"]!=5:raise StageFF3CProtocolError("F3C budget contract drifted")
    authorization=config["authorization"]
    if authorization!={"current":"F3C0_PROTOCOL_FREEZE_ONLY","on_f3c0_pass":"F3C1_IMPLEMENTATION_AND_SYNTHETIC_PREFLIGHT_ONLY","empirical_f3c_started":False,"f4_status":"PAUSED_PENDING_F3C","f4_started":False}:raise StageFF3CProtocolError("F3C authorization drifted")
    if any(path.exists() for base in (repo/"evidence/stage_f",repo/"reports") for path in base.glob("f4*")):raise StageFF3CProtocolError("F4 output exists")
    forbidden=sorted(name for name in sys.modules if name=="torch" or name.startswith("torch.") or name=="timm" or name.startswith("timm."))
    if forbidden or os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise StageFF3CProtocolError("F3C0 runtime boundary failed")
    return _signed({"schema_version":1,"document_type":"stage_f_f3c0_validation","status":"PASS","protocol_id":config["protocol_id"],"config_digest":config["canonical_digest"],"methods":list(METHODS),"pilot_count":5,"formal_count":50,"statistical_n":50,"basis_status":basis["status"],"empirical_f3c_started":False,"f4_status":"PAUSED_PENDING_F3C","next_authorization":"F3C1_IMPLEMENTATION_AND_SYNTHETIC_PREFLIGHT_ONLY","runtime":{"model_loaded":False,"jpeg_decoded":False,"torch_imported":False,"timm_imported":False,"cuda_or_gpu_access":False,"attribution_executed":False}})

def freeze(repo_root:Path|str=REPO_ROOT,*,frozen_at:str,test_passed:int,test_failed:int,test_command:str)->dict[str,Any]:
    repo=Path(repo_root).resolve()
    if (repo/ROOT_PATH).exists() or (repo/REPORT_PATH).exists():raise StageFF3CProtocolError("F3C0 root exists")
    if test_passed<=0 or test_failed!=0:raise StageFF3CProtocolError("F3C0 tests failed")
    validation=validate_protocol(repo,require_clean=True);head=_git(repo,"rev-parse","HEAD");tracking=_git(repo,"rev-parse","origin/stage-d-d3-pass")
    if head!=tracking:raise StageFF3CProtocolError("F3C0 local/tracking drifted")
    pending=_signed({"schema_version":1,"document_type":"stage_f_f3c0_pending","status":"PENDING","frozen_at":frozen_at,"protocol_commit":head,"empirical_f3c_started":False,"f4_started":False});_write_new(repo/PENDING_PATH,pending)
    audit=_signed({"schema_version":1,"document_type":"stage_f_f3c0_false_pass_audit","status":"PASS","attacks":[{"attack":item,"result":"REJECTED"} for item in ATTACKS],"pass_possible_without_all_conditions":False})
    test=_signed({"schema_version":1,"document_type":"stage_f_f3c0_test_report","status":"PASS","passed":test_passed,"failed":test_failed,"command":test_command,"empirical_f3c_started":False})
    _write_new(repo/VALIDATION_PATH,validation);_write_new(repo/FALSE_PASS_PATH,audit);_write_new(repo/TEST_PATH,test)
    status=_signed({"schema_version":1,"document_type":"stage_f_f3c0_status","status":"PASS","terminal":True,"frozen_at":frozen_at,"method_count":5,"pilot_count":5,"formal_count":50,"empirical_f3c_started":False,"f4_status":"PAUSED_PENDING_F3C","f4_started":False,"next_authorization":"F3C1_IMPLEMENTATION_AND_SYNTHETIC_PREFLIGHT_ONLY","git":{"commit":"SELF","ref":ARCHIVE_REF}});_write_new(repo/STATUS_PATH,status)
    completion=_signed({"schema_version":1,"document_type":"stage_f_f3c0_terminal_completion","status":"PASS","frozen_at":frozen_at,"protocol_commit":head,"config":{"path":CONFIG_PATH,"sha256":_sha(repo/CONFIG_PATH),"canonical_digest":load_config(repo)["canonical_digest"]},"schema":{"path":SCHEMA_PATH,"sha256":_sha(repo/SCHEMA_PATH)},"protocol":{"path":PROTOCOL_PATH,"sha256":_sha(repo/PROTOCOL_PATH)},"sample_split":{"path":SPLIT_PATH,"sha256":_sha(repo/SPLIT_PATH),"canonical_digest":_read(repo/SPLIT_PATH)["canonical_digest"]},"validation_report":{"path":VALIDATION_PATH,"sha256":_sha(repo/VALIDATION_PATH),"canonical_digest":validation["canonical_digest"]},"false_pass_audit":{"path":FALSE_PASS_PATH,"sha256":_sha(repo/FALSE_PASS_PATH),"canonical_digest":audit["canonical_digest"]},"test_report":{"path":TEST_PATH,"sha256":_sha(repo/TEST_PATH),"canonical_digest":test["canonical_digest"]},"status_record":{"path":STATUS_PATH,"sha256":_sha(repo/STATUS_PATH),"canonical_digest":status["canonical_digest"]},"method_count":5,"pilot_count":5,"formal_count":50,"statistical_n":50,"class_targeted_nns_uses_manual_head_weights":False,"empirical_f3c_started":False,"model_loaded":False,"jpeg_decoded":False,"torch_imported":False,"timm_imported":False,"cuda_or_gpu_access":False,"attribution_executed":False,"f4_status":"PAUSED_PENDING_F3C","f4_started":False,"next_authorization":"F3C1_IMPLEMENTATION_AND_SYNTHETIC_PREFLIGHT_ONLY","git":{"commit":"SELF","ref":ARCHIVE_REF}})
    temporary=(repo/PENDING_PATH).with_name(".f3c0-terminal.tmp");_write_new(temporary,completion);os.replace(temporary,repo/PENDING_PATH);os.replace(repo/PENDING_PATH,repo/COMPLETION_PATH);return completion

def validate_completion(repo_root:Path|str=REPO_ROOT)->dict[str,Any]:
    repo=Path(repo_root).resolve();completion=_read(repo/COMPLETION_PATH);_validate_signed(completion,"F3C0 completion")
    if (repo/PENDING_PATH).exists() or completion["status"]!="PASS" or completion["method_count"]!=5 or completion["pilot_count"]!=5 or completion["formal_count"]!=50 or completion["class_targeted_nns_uses_manual_head_weights"] or completion["empirical_f3c_started"] or completion["model_loaded"] or completion["jpeg_decoded"] or completion["torch_imported"] or completion["timm_imported"] or completion["cuda_or_gpu_access"] or completion["attribution_executed"] or completion["f4_started"] or completion["next_authorization"]!="F3C1_IMPLEMENTATION_AND_SYNTHETIC_PREFLIGHT_ONLY":raise StageFF3CProtocolError("F3C0 completion state drifted")
    for field in ("validation_report","false_pass_audit","test_report","status_record"):
        record=completion[field];value=_read(repo/record["path"]);_validate_signed(value,field)
        if _sha(repo/record["path"])!=record["sha256"] or value["canonical_digest"]!=record["canonical_digest"]:raise StageFF3CProtocolError(f"F3C0 completion binding drifted: {field}")
    return {"status":"PASS","method_count":5,"pilot_count":5,"formal_count":50,"empirical_f3c_started":False,"f4_status":"PAUSED_PENDING_F3C","next_authorization":"F3C1_IMPLEMENTATION_AND_SYNTHETIC_PREFLIGHT_ONLY","archive_ref":ARCHIVE_REF}

def main(argv:Sequence[str]|None=None)->int:
    import argparse
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest="command",required=True)
    check=sub.add_parser("validate");check.add_argument("--repo-root",type=Path,default=REPO_ROOT);check.add_argument("--require-clean",action="store_true")
    freeze_parser=sub.add_parser("freeze");freeze_parser.add_argument("--repo-root",type=Path,default=REPO_ROOT);freeze_parser.add_argument("--frozen-at",required=True);freeze_parser.add_argument("--test-passed",type=int,required=True);freeze_parser.add_argument("--test-failed",type=int,required=True);freeze_parser.add_argument("--test-command",required=True)
    completion=sub.add_parser("validate-completion");completion.add_argument("--repo-root",type=Path,default=REPO_ROOT);args=parser.parse_args(argv)
    try:
        if args.command=="validate":result=validate_protocol(args.repo_root,require_clean=args.require_clean)
        elif args.command=="freeze":result=freeze(args.repo_root,frozen_at=args.frozen_at,test_passed=args.test_passed,test_failed=args.test_failed,test_command=args.test_command)
        else:result=validate_completion(args.repo_root)
    except (StageFF3CProtocolError,OSError,ValueError) as error:
        print(json.dumps({"status":"BLOCKED","error":str(error)},sort_keys=True),file=sys.stderr);return 2
    print(json.dumps(result,indent=2,sort_keys=True));return 0

if __name__=="__main__":raise SystemExit(main())
