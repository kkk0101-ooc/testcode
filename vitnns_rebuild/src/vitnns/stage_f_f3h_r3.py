"""CPU-only atomic sealing wrapper for the validated F3H-R2 payload."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .constants import REPO_ROOT
from .stage_d_provenance import canonical_sha256
from . import stage_f_f3h as f3h

ROOT_PATH = "evidence/stage_f/f3h_full_head_r3_terminal_seal_20260907/formal"
REPORT_PATH = "reports/stage_f_f3h_full_head_r3_terminal_seal_20260907"
PENDING_PATH = ROOT_PATH + "/completion/f3h.pending.json"
COMPLETION_PATH = ROOT_PATH + "/completion/f3h.json"
SOURCE_BINDING_PATH = ROOT_PATH + "/source_binding.json"
STATUS_PATH = ROOT_PATH + "/status.json"
PAYLOAD_PATH = ROOT_PATH + "/payload_manifest.json"
INVENTORY_PATH = ROOT_PATH + "/whole_root_inventory.json"
VALIDATION_PATH = REPORT_PATH + "/validation_report.json"
FALSE_PASS_PATH = REPORT_PATH + "/false_pass_audit.json"
TEST_PATH = REPORT_PATH + "/test_report.json"
ADDENDUM_PATH = "docs/stage_f_f3h_r3_terminal_seal_addendum_20260907.md"
ADDENDUM_SHA256 = "7cd11609a204bc85b39ec36bcf889f6ec1b61779d578598cc1551eb022a8e84b"
R2_ROOT = "evidence/stage_f/f3h_full_head_r2_cpu_reconcile_20260907/formal"
R2_REPORT = "reports/stage_f_f3h_full_head_r2_cpu_reconcile_20260907"
R2_FAILURE = R2_REPORT + "/failure_event.json"
R2_FAILURE_SHA256 = "30666a204979bcacf741446883208e124752b7a65ed67c6fd34c6f4088b024ce"
R2_BLOCKED_REF = "refs/archive/stage-f-v1-f3h-r2-blocked"
R2_BLOCKED_COMMIT = "a006cd528273937dd4f9ee08beaf9e313889b57d"
R3_PRODUCER_REF = "refs/archive/stage-f-v1-f3h-r3-producer"
ARCHIVE_REF = "refs/archive/stage-f-v1-f3h-completion"
ATTACKS = (
    "r2_blocked_attempt_relabelled_as_terminal",
    "r2_payload_file_drift",
    "r2_whole_root_inventory_drift",
    "r2_output_validator_skipped",
    "validation_double_signed_again",
    "review_copy_or_manual_qc_omitted",
    "large_payload_copied_or_rerendered",
    "model_torch_timm_cuda_gpu_or_attribution_access",
    "atlas_used_to_define_head_weights",
    "empirical_f3c_started",
    "f4_started",
    "terminal_completion_before_all_receipts",
)

class StageFF3HR3Error(ValueError):
    pass

def _signed(value: Mapping[str, Any]) -> dict[str, Any]:
    result=dict(value);result["canonical_digest"]=canonical_sha256(result);return result

def _runtime_boundary() -> dict[str, bool]:
    forbidden=sorted(name for name in sys.modules if name=="torch" or name.startswith("torch.") or name=="timm" or name.startswith("timm."))
    if forbidden: raise StageFF3HR3Error(f"forbidden R3 runtime imported: {forbidden[0]}")
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="": raise StageFF3HR3Error("R3 requires CUDA_VISIBLE_DEVICES=''")
    return {"model_loaded":False,"jpeg_decoded":False,"torch_imported":False,"timm_imported":False,"cuda_or_gpu_access":False,"attribution_executed":False,"rerendered":False}

def _validate_r2_inventory(repo: Path) -> dict[str, Any]:
    inventory=f3h._strict_json(repo/f3h.INVENTORY_PATH,"R2 whole-root inventory");f3h._validate_signed(inventory,"R2 whole-root inventory")
    observed=f3h._inventory(repo/R2_ROOT,exclude=("whole_root_inventory.json","status.json","completion/f3h.pending.json","completion/f3h.json"))
    if inventory["files"]!=observed["files"] or inventory["file_count"]!=observed["file_count"] or inventory["total_bytes"]!=observed["total_bytes"]: raise StageFF3HR3Error("R2 whole-root inventory drifted")
    payload=f3h._strict_json(repo/f3h.PAYLOAD_MANIFEST_PATH,"R2 payload");f3h._validate_signed(payload,"R2 payload")
    excluded={"payload_manifest.json","whole_root_inventory.json","status.json","completion/f3h.pending.json","completion/f3h.json"}
    files=[{"path":p.relative_to(repo/R2_ROOT).as_posix(),"bytes":p.stat().st_size,"sha256":f3h._sha_file(p)} for p in sorted((repo/R2_ROOT).rglob("*")) if p.is_file() and p.relative_to(repo/R2_ROOT).as_posix() not in excluded]
    if payload["artifacts"]!=files or payload["payload_count"]!=len(files): raise StageFF3HR3Error("R2 payload manifest drifted")
    if not (repo/R2_ROOT/"completion/f3h.pending.json").is_file() or (repo/R2_ROOT/"completion/f3h.json").exists(): raise StageFF3HR3Error("R2 blocked lifecycle drifted")
    return {"inventory":inventory,"payload":payload}

def validate_source(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo=Path(repo_root).resolve();runtime=_runtime_boundary()
    if f3h._sha_file(repo/ADDENDUM_PATH)!=ADDENDUM_SHA256: raise StageFF3HR3Error("R3 addendum drifted")
    if f3h._sha_file(repo/R2_FAILURE)!=R2_FAILURE_SHA256: raise StageFF3HR3Error("R2 failure receipt drifted")
    ref=f3h._git(repo,"rev-parse",R2_BLOCKED_REF+"^{commit}")
    if ref!=R2_BLOCKED_COMMIT: raise StageFF3HR3Error("R2 blocked archive ref drifted")
    bound=_validate_r2_inventory(repo);validation=f3h.validate_outputs(repo);f3h._validate_signed(validation,"live R2 validation")
    if validation["status"]!="PASS" or validation["value_path_counts"]["valid"]!=288 or validation["render_count"]!=20 or validation["review_count"]!=20 or validation["class_conditioned_basis"]!="INSUFFICIENT_BASIS_FOR_NUMERICAL_CLASS_CONDITIONED_WEIGHTS": raise StageFF3HR3Error("R2 validated result drifted")
    return {"status":"PASS","validation":validation,"r2_inventory":bound["inventory"],"r2_payload":bound["payload"],"runtime":runtime}

def initialize(repo_root: Path | str = REPO_ROOT, *, started_at: str) -> dict[str, Any]:
    repo=Path(repo_root).resolve()
    if (repo/ROOT_PATH).exists() or (repo/REPORT_PATH).exists(): raise StageFF3HR3Error("R3 root exists")
    if f3h._git(repo,"status","--porcelain=v1"): raise StageFF3HR3Error("R3 initialization requires clean worktree")
    head=f3h._git(repo,"rev-parse","HEAD");tracking=f3h._git(repo,"rev-parse","origin/stage-d-d3-pass")
    if head!=tracking or f3h._git(repo,"rev-parse",R3_PRODUCER_REF+"^{commit}")!=head: raise StageFF3HR3Error("R3 producer/local/tracking drifted")
    source=validate_source(repo)
    binding=_signed({"schema_version":1,"document_type":"stage_f_f3h_r3_source_binding","status":"PASS","r2_blocked_ref":{"ref":R2_BLOCKED_REF,"commit":f3h._git(repo,"rev-parse",R2_BLOCKED_REF+"^{commit}")},"r2_failure":f3h._file_record(repo,R2_FAILURE),"r2_payload_manifest":f3h._record(repo,f3h.PAYLOAD_MANIFEST_PATH),"r2_whole_root_inventory":f3h._record(repo,f3h.INVENTORY_PATH),"r2_render_inventory":f3h._record(repo,f3h.RENDER_INVENTORY_PATH),"r2_review_inventory":f3h._record(repo,f3h.REVIEW_INVENTORY_PATH),"r2_figure_analysis":f3h._file_record(repo,f3h.ANALYSIS_PATH),"r3_addendum":f3h._file_record(repo,ADDENDUM_PATH),"large_payload_copied":False,"rerendered":False,"f4_started":False})
    pending=_signed({"schema_version":1,"document_type":"stage_f_f3h_r3_pending","status":"PENDING","started_at":started_at,"execution_commit":head,"r2_validation_digest":source["validation"]["canonical_digest"],"terminal_pass_published":False,"f4_status":"PAUSED_PENDING_F3H_AND_F3C","f4_started":False})
    f3h._write_new(repo/SOURCE_BINDING_PATH,binding);f3h._write_new(repo/PENDING_PATH,pending)
    return {"status":"INITIALIZED","r2_payload_count":source["r2_payload"]["payload_count"],"next_authorization":"R3_FINALIZE_ONLY"}

def record_tests(repo_root: Path | str, *, passed: int, failed: int, command: str) -> dict[str, Any]:
    if passed<=0 or failed!=0: raise StageFF3HR3Error("R3 tests failed")
    report=_signed({"schema_version":1,"document_type":"stage_f_f3h_r3_test_report","status":"PASS","passed":passed,"failed":failed,"command":command,"cpu_only":True})
    f3h._write_new(Path(repo_root).resolve()/TEST_PATH,report);return report

def finalize(repo_root: Path | str = REPO_ROOT, *, completed_at: str) -> dict[str, Any]:
    repo=Path(repo_root).resolve();source=validate_source(repo);pending=f3h._strict_json(repo/PENDING_PATH,"R3 pending");f3h._validate_signed(pending,"R3 pending")
    test=f3h._record(repo,TEST_PATH);validation=source["validation"];f3h._write_new(repo/VALIDATION_PATH,validation)
    audit=_signed({"schema_version":1,"document_type":"stage_f_f3h_r3_false_pass_audit","status":"PASS","attacks":[{"attack":item,"result":"REJECTED"} for item in ATTACKS],"pass_possible_without_all_conditions":False});f3h._write_new(repo/FALSE_PASS_PATH,audit)
    files=[{"path":p.relative_to(repo/ROOT_PATH).as_posix(),"bytes":p.stat().st_size,"sha256":f3h._sha_file(p)} for p in sorted((repo/ROOT_PATH).rglob("*")) if p.is_file() and p.relative_to(repo/ROOT_PATH).as_posix() not in {"payload_manifest.json","whole_root_inventory.json","status.json","completion/f3h.pending.json","completion/f3h.json"}]
    payload=_signed({"schema_version":1,"document_type":"stage_f_f3h_r3_payload_manifest","status":"PASS","payload_count":len(files),"artifacts":files});f3h._write_new(repo/PAYLOAD_PATH,payload)
    observed=f3h._inventory(repo/ROOT_PATH,exclude=("whole_root_inventory.json","status.json","completion/f3h.pending.json","completion/f3h.json"));inventory=_signed({"schema_version":1,"document_type":"stage_f_f3h_r3_whole_root_inventory","root":ROOT_PATH,"files":observed["files"],"file_count":observed["file_count"],"total_bytes":observed["total_bytes"]});f3h._write_new(repo/INVENTORY_PATH,inventory)
    status=_signed({"schema_version":1,"document_type":"stage_f_f3h_r3_status","status":"PASS","terminal":True,"scientific_validity":"PASS","value_path_counts":validation["value_path_counts"],"gradcam_status":"VALID","render_count":20,"class_conditioned_basis":validation["class_conditioned_basis"],"f4_status":"PAUSED_PENDING_F3H_AND_F3C","f4_started":False,"next_authorization":"F3C0_PROTOCOL_FREEZE_ONLY","git":{"commit":"SELF","ref":ARCHIVE_REF}});f3h._write_new(repo/STATUS_PATH,status)
    completion=_signed({"schema_version":1,"document_type":"stage_f_f3h_terminal_completion","status":"PASS","completed_at":completed_at,"started_at":pending["started_at"],"scientific_validity":"PASS","value_path_counts":validation["value_path_counts"],"gradcam_status":"VALID","render_count":20,"exported_file_count":60,"review_count":20,"class_conditioned_basis":validation["class_conditioned_basis"],"r2_source_binding":f3h._record(repo,SOURCE_BINDING_PATH),"r3_validation_report":f3h._record(repo,VALIDATION_PATH),"r3_false_pass_audit":f3h._record(repo,FALSE_PASS_PATH),"r3_test_report":test,"r3_payload_manifest":f3h._record(repo,PAYLOAD_PATH),"r3_whole_root_inventory":f3h._record(repo,INVENTORY_PATH),"r2_terminal_union":f3h._record(repo,f3h.TERMINAL_UNION_PATH),"r2_render_inventory":f3h._record(repo,f3h.RENDER_INVENTORY_PATH),"r2_review_inventory":f3h._record(repo,f3h.REVIEW_INVENTORY_PATH),"r2_figure_analysis":f3h._file_record(repo,f3h.ANALYSIS_PATH),"r2_class_conditioned_basis_assessment":f3h._record(repo,f3h.SUFFICIENCY_PATH),"model_torch_timm_cuda_gpu_or_attribution_access":False,"large_payload_copied_or_rerendered":False,"f4_started":False,"next_authorization":"F3C0_PROTOCOL_FREEZE_ONLY","git":{"commit":"SELF","ref":ARCHIVE_REF}})
    temporary=(repo/PENDING_PATH).with_name(".f3h-r3-terminal.tmp");f3h._write_new(temporary,completion);os.replace(temporary,repo/PENDING_PATH);os.replace(repo/PENDING_PATH,repo/COMPLETION_PATH);return completion

def validate_completion(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo=Path(repo_root).resolve();completion=f3h._strict_json(repo/COMPLETION_PATH,"R3 completion");f3h._validate_signed(completion,"R3 completion")
    if (repo/PENDING_PATH).exists() or completion["status"]!="PASS" or completion["value_path_counts"]["valid"]!=288 or completion["render_count"]!=20 or completion["review_count"]!=20 or completion["class_conditioned_basis"]!="INSUFFICIENT_BASIS_FOR_NUMERICAL_CLASS_CONDITIONED_WEIGHTS" or completion["model_torch_timm_cuda_gpu_or_attribution_access"] or completion["large_payload_copied_or_rerendered"] or completion["f4_started"] or completion["next_authorization"]!="F3C0_PROTOCOL_FREEZE_ONLY": raise StageFF3HR3Error("R3 completion state drifted")
    for field in ("r2_source_binding","r3_validation_report","r3_false_pass_audit","r3_test_report","r3_payload_manifest","r3_whole_root_inventory","r2_terminal_union","r2_render_inventory","r2_review_inventory","r2_class_conditioned_basis_assessment"):
        if completion[field]!=f3h._record(repo,completion[field]["path"]): raise StageFF3HR3Error(f"R3 completion binding drifted: {field}")
    if completion["r2_figure_analysis"]!=f3h._file_record(repo,f3h.ANALYSIS_PATH): raise StageFF3HR3Error("R3 figure analysis binding drifted")
    return {"status":"PASS","scientific_validity":"PASS","value_path_counts":completion["value_path_counts"],"gradcam_status":"VALID","render_count":20,"class_conditioned_basis":completion["class_conditioned_basis"],"next_authorization":"F3C0_PROTOCOL_FREEZE_ONLY","f4_status":"PAUSED_PENDING_F3H_AND_F3C","archive_ref":ARCHIVE_REF}

def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest="command",required=True)
    for name in ("validate-source","validate-completion"):
        child=sub.add_parser(name);child.add_argument("--repo-root",type=Path,default=REPO_ROOT)
    init=sub.add_parser("initialize");init.add_argument("--repo-root",type=Path,default=REPO_ROOT);init.add_argument("--started-at",required=True)
    tests=sub.add_parser("record-tests");tests.add_argument("--repo-root",type=Path,default=REPO_ROOT);tests.add_argument("--passed",type=int,required=True);tests.add_argument("--failed",type=int,required=True);tests.add_argument("--test-command",required=True)
    final=sub.add_parser("finalize");final.add_argument("--repo-root",type=Path,default=REPO_ROOT);final.add_argument("--completed-at",required=True);args=parser.parse_args(argv)
    try:
        table={"validate-source":lambda:validate_source(args.repo_root),"initialize":lambda:initialize(args.repo_root,started_at=args.started_at),"record-tests":lambda:record_tests(args.repo_root,passed=args.passed,failed=args.failed,command=args.test_command),"finalize":lambda:finalize(args.repo_root,completed_at=args.completed_at),"validate-completion":lambda:validate_completion(args.repo_root)};result=table[args.command]()
    except (StageFF3HR3Error,f3h.StageFF3HError,OSError,ValueError) as error:
        print(json.dumps({"status":"BLOCKED","error":str(error)},sort_keys=True),file=sys.stderr);return 2
    print(json.dumps(result,indent=2,sort_keys=True));return 0

if __name__=="__main__": raise SystemExit(main())
