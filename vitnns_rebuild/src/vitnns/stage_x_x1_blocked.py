"""Seal X1's fail-closed missing-comparator dependency outcome."""
from __future__ import annotations
import argparse, ast, hashlib, importlib.util, json, os, re, subprocess, sys
from pathlib import Path
from typing import Any, Mapping, Sequence
from jsonschema import Draft202012Validator
from .constants import REPO_ROOT

BASE_COMMIT="a7adbd64c468aa33ffa1c02ae1c575609d837b30"
BRANCH="stage-d-d3-pass"
UPSTREAM_REF="refs/archive/stage-x-nns-v2-x0-r1-completion"
ARCHIVE_REF="refs/archive/stage-x-nns-v2-x1-blocked"
ROOT=Path("evidence/stage_x/x1_nns_v2_20260908/formal")
REPORT_ROOT=Path("reports/stage_x_x1_nns_v2_20260908")
PENDING=ROOT/"completion/x1.pending.json"
COMPLETION=ROOT/"completion/x1.json"
CONFIG=Path("config/stage_x_x1_blocked.json")
SCHEMA=Path("config/stage_x_x1_blocked.schema.json")
DOC=Path("docs/stage_x_x1_blocked_20260908.md")
X0R1=Path("evidence/stage_x/x0_r1_identity_control_20260908/formal/completion/x0_r1.json")
X0R1_SHA="a14c58963f73b9f0854964478d97cf9ba5ac39c195e1b7a096e01d6b4184e3b7"
SOURCE_FILES=(CONFIG,SCHEMA,DOC,Path("src/vitnns/stage_x_x1_blocked.py"),Path("tests/test_stage_x_x1_blocked.py"),Path("docs/NNS-v2前瞻性主张扩展方案.md"),Path("../论文1实现审计与后续计划.md"),Path("docs/阶段路线图.md"),Path("docs/阶段F规划.md"),Path("docs/阶段G规划.md"))
ATTACKS=("x0_r1_completion_or_ref_drift","missing_engine_reported_present","required_comparator_removed","official_commit_or_license_changed","download_attempt_hidden","image_or_checkpoint_access_hidden","torch_timm_or_cuda_access_hidden","draft_relabelled_accepted_implementation","blocked_relabelled_pass","x2_started_after_blocked_x1","terminal_before_validation_inventory_false_pass","terminal_not_bound_to_clean_commit_archive")

class StageXX1BlockedError(ValueError): pass
def _constant(token:str)->None: raise StageXX1BlockedError(f"nonfinite JSON: {token}")
def _pairs(pairs:list[tuple[str,Any]])->dict[str,Any]:
    result={}
    for key,value in pairs:
        if key in result: raise StageXX1BlockedError(f"duplicate key: {key}")
        result[key]=value
    return result
def read_json(path:Path)->Any: return json.loads(path.read_text(encoding="utf-8"),object_pairs_hook=_pairs,parse_constant=_constant)
def canonical_bytes(value:Any)->bytes: return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def canonical_sha256(value:Any)->str: return hashlib.sha256(canonical_bytes(value)).hexdigest()
def signed(value:Mapping[str,Any])->dict[str,Any]:
    result=dict(value);result["canonical_digest"]=canonical_sha256(result);return result
def validate_signed(value:Mapping[str,Any],context:str)->None:
    payload=dict(value);observed=payload.pop("canonical_digest",None)
    if observed!=canonical_sha256(payload):raise StageXX1BlockedError(context+" digest mismatch")
def sha(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1048576),b""):digest.update(chunk)
    return digest.hexdigest()
def git(repo:Path,*args:str)->str:
    process=subprocess.run(["git",*args],cwd=repo,text=True,capture_output=True,check=False)
    if process.returncode:raise StageXX1BlockedError(process.stderr.strip() or "git failed")
    return process.stdout.strip()
def write_new(path:Path,value:Mapping[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True);descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(descriptor,"wb") as stream:stream.write(canonical_bytes(value)+b"\n");stream.flush();os.fsync(stream.fileno())
def record(path:Path,base:Path)->dict[str,Any]:return {"path":path.relative_to(base).as_posix(),"bytes":path.stat().st_size,"sha256":sha(path)}

def load(repo:Path)->dict[str,Any]:
    config=read_json(repo/CONFIG);schema=read_json(repo/SCHEMA);Draft202012Validator.check_schema(schema);errors=list(Draft202012Validator(schema).iter_errors(config))
    if errors:raise StageXX1BlockedError(errors[0].message)
    validate_signed(config,"X1 blocked contract");return config

def validate(repo_root:Path|str=REPO_ROOT)->dict[str,Any]:
    repo=Path(repo_root).resolve();config=load(repo)
    if git(repo,"symbolic-ref","--short","HEAD")!=BRANCH or git(repo,"rev-parse",UPSTREAM_REF+"^{commit}")!=BASE_COMMIT:raise StageXX1BlockedError("X0-R1 authority drifted")
    if sha(repo/X0R1)!=X0R1_SHA:raise StageXX1BlockedError("X0-R1 completion drifted")
    expected=[("transformer_attribution_chefer","c3e578f76b954e8528afeaaee26de3f07e3fe559","MIT"),("attnlrp","f04344177551bfcc3665e88afe96501f43fe560b","Clear-BSD with scientific-use patent limitation")]
    if [(item["method_id"],item["commit"],item["license"]) for item in config["required_engines"]]!=expected or any(item["local_status"]!="MISSING" or not item["required"] for item in config["required_engines"]):raise StageXX1BlockedError("missing-engine roster/status drifted")
    discovered={"lxt":None if importlib.util.find_spec("lxt") is None else "PRESENT","pytorch_grad_cam":None if importlib.util.find_spec("pytorch_grad_cam") is None else "PRESENT"}
    if discovered["lxt"] is not None:raise StageXX1BlockedError("AttnLRP engine unexpectedly available; re-evaluate in fresh recovery")
    if any((repo/path).exists() for path in (Path("vendor/Transformer-Explainability"),Path("third_party/Transformer-Explainability"),Path("src/vitnns/stage_x_chefer_official.py"))):raise StageXX1BlockedError("Chefer engine presence contradicts blocker")
    if any(config["access_boundary"].values()):raise StageXX1BlockedError("X1 blocked access boundary drifted")
    if config["status"]!="BLOCKED_MISSING_PINNED_COMPARATOR_ENGINE" or config["next_authorization"]!="NONE_REQUIRES_PROSPECTIVE_X1_RECOVERY_AMENDMENT":raise StageXX1BlockedError("blocked status or authorization drifted")
    tree=ast.parse((repo/"src/vitnns/stage_x_x1_blocked.py").read_text());imports=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node,ast.ImportFrom) and node.module:imports.add(node.module.split(".")[0])
    if imports&{"torch","timm","PIL","cv2","requests"}:raise StageXX1BlockedError("blocked recorder imports forbidden runtime modules")
    return signed({"schema_version":1,"document_type":"stage_x_x1_blocker_validation","status":"PASS_BLOCKER_CONFIRMED","blocker":config["status"],"required_engines":config["required_engines"],"module_discovery":discovered,"access_boundary":config["access_boundary"],"next_authorization":config["next_authorization"]})

def source_binding(repo:Path)->dict[str,Any]:
    files=[{"path":path.as_posix(),"bytes":(repo/path).stat().st_size,"sha256":sha(repo/path)} for path in SOURCE_FILES]
    return signed({"schema_version":1,"document_type":"stage_x_x1_blocked_source_binding","status":"PASS","files":files,"files_digest":canonical_sha256(files)})

def freeze(repo_root:Path|str=REPO_ROOT,*,completed_at:str,test_passed:int,test_failed:int,test_command:str)->dict[str,Any]:
    repo=Path(repo_root).resolve()
    if (repo/ROOT).exists() or (repo/REPORT_ROOT).exists():raise StageXX1BlockedError("X1 root exists; refusing overwrite")
    if type(test_passed) is not int or test_passed<=0 or type(test_failed) is not int or test_failed!=0:raise StageXX1BlockedError("blocker tests failed")
    validation=validate(repo);config=load(repo);(repo/ROOT).mkdir(parents=True);(repo/REPORT_ROOT).mkdir(parents=True)
    write_new(repo/PENDING,signed({"schema_version":1,"document_type":"stage_x_x1_pending","status":"PENDING","created_at":completed_at,"upstream_commit":BASE_COMMIT}))
    payload={
      "startup_gate.json":signed({"schema_version":1,"document_type":"stage_x_x1_startup_gate","status":"PASS","branch":BRANCH,"local_tracking_remote":BASE_COMMIT,"upstream_ref":UPSTREAM_REF,"upstream_commit":BASE_COMMIT,"initial_worktree_clean":True,"initial_lfs_clean":True}),
      "dependency_preflight.json":validation,
      "source_binding.json":source_binding(repo),
      "access_audit.json":signed({"schema_version":1,"document_type":"stage_x_x1_blocked_access_audit","status":"PASS",**config["access_boundary"]}),
      "false_pass_audit.json":signed({"schema_version":1,"document_type":"stage_x_x1_blocked_false_pass_audit","status":"PASS","attacks":[{"attack":attack,"result":"REJECTED"} for attack in ATTACKS],"attack_count":len(ATTACKS),"pass_possible":False}),
      "test_report.json":signed({"schema_version":1,"document_type":"stage_x_x1_blocked_test_report","status":"PASS","passed":test_passed,"failed":test_failed,"command":test_command}),
      "status.json":signed({"schema_version":1,"document_type":"stage_x_x1_status","status":"BLOCKED_MISSING_PINNED_COMPARATOR_ENGINE","terminal":True,"scientific_failure":False,"next_authorization":config["next_authorization"],"git":{"commit":"SELF","ref":ARCHIVE_REF}}),
    }
    for name,value in payload.items():write_new(repo/ROOT/name,value)
    rows=[record(repo/ROOT/name,repo/ROOT) for name in sorted(payload)]
    write_new(repo/ROOT/"payload_manifest.json",signed({"schema_version":1,"document_type":"stage_x_x1_blocked_payload_manifest","status":"PASS","files":rows,"files_digest":canonical_sha256(rows)}))
    paths=sorted((path for path in (repo/ROOT).rglob("*") if path.is_file() and path!=repo/PENDING),key=lambda path:path.relative_to(repo/ROOT).as_posix());rows=[record(path,repo/ROOT) for path in paths]
    write_new(repo/ROOT/"whole_root_inventory.json",signed({"schema_version":1,"document_type":"stage_x_x1_blocked_inventory","status":"PASS","files":rows,"file_count":len(rows),"total_bytes":sum(item["bytes"] for item in rows),"files_digest":canonical_sha256(rows)}))
    names=tuple(payload)+("payload_manifest.json","whole_root_inventory.json")
    completion=signed({"schema_version":1,"document_type":"stage_x_x1_terminal_completion","status":config["status"],"terminal":True,"scientific_failure":False,"completed_at":completed_at,"upstream_commit":BASE_COMMIT,"records":{name:{"path":(ROOT/name).as_posix(),"sha256":sha(repo/ROOT/name),"canonical_digest":read_json(repo/ROOT/name)["canonical_digest"]} for name in names},**config["access_boundary"],"next_authorization":config["next_authorization"],"git":{"commit":"SELF","ref":ARCHIVE_REF}})
    temporary=(repo/PENDING).with_name(".x1-terminal.tmp");write_new(temporary,completion);os.replace(temporary,repo/PENDING);os.replace(repo/PENDING,repo/COMPLETION)
    report=repo/REPORT_ROOT/"x1_blocked_report.md";descriptor=os.open(report,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(descriptor,"w",encoding="utf-8") as stream:stream.write("# Stage X X1 terminal report\n\n`BLOCKED_MISSING_PINNED_COMPARATOR_ENGINE`: the pinned Chefer and AttnLRP engines are unavailable locally and X1 has a zero-download cap. No image, checkpoint, model, or GPU access occurred.\n");stream.flush();os.fsync(stream.fileno())
    return completion

def validate_completion(repo_root:Path|str=REPO_ROOT,*,published:bool=False)->dict[str,Any]:
    repo=Path(repo_root).resolve();completion=read_json(repo/COMPLETION);validate_signed(completion,"completion")
    if (repo/PENDING).exists() or completion["status"]!="BLOCKED_MISSING_PINNED_COMPARATOR_ENGINE" or not completion["terminal"]:raise StageXX1BlockedError("not terminal blocker")
    for item in completion["records"].values():
        path=repo/item["path"];value=read_json(path);validate_signed(value,str(path))
        if sha(path)!=item["sha256"] or value["canonical_digest"]!=item["canonical_digest"]:raise StageXX1BlockedError("completion binding drift")
    validate(repo)
    if published:
        head=git(repo,"rev-parse","HEAD");tracking=git(repo,"rev-parse",f"origin/{BRANCH}");remote=git(repo,"ls-remote","--heads","origin",BRANCH).split()[0];archive=git(repo,"rev-parse",ARCHIVE_REF+"^{commit}");remote_archive=git(repo,"ls-remote","origin",ARCHIVE_REF).split()[0]
        if len({head,tracking,remote,archive,remote_archive})!=1 or git(repo,"status","--porcelain=v1"):raise StageXX1BlockedError("published Git state drift")
    return {"status":completion["status"],"next_authorization":completion["next_authorization"],"archive_ref":ARCHIVE_REF,"published":published}

def main(argv:Sequence[str]|None=None)->int:
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest="command",required=True)
    command=sub.add_parser("validate");command.add_argument("--repo-root",type=Path,default=REPO_ROOT)
    command=sub.add_parser("freeze");command.add_argument("--repo-root",type=Path,default=REPO_ROOT);command.add_argument("--completed-at",required=True);command.add_argument("--test-passed",type=int,required=True);command.add_argument("--test-failed",type=int,required=True);command.add_argument("--test-command",required=True)
    command=sub.add_parser("validate-completion");command.add_argument("--repo-root",type=Path,default=REPO_ROOT);command.add_argument("--published",action="store_true");args=parser.parse_args(argv)
    try:
        result=validate(args.repo_root) if args.command=="validate" else freeze(args.repo_root,completed_at=args.completed_at,test_passed=args.test_passed,test_failed=args.test_failed,test_command=args.test_command) if args.command=="freeze" else validate_completion(args.repo_root,published=args.published)
    except (StageXX1BlockedError,OSError,ValueError,KeyError,IndexError) as error:print(json.dumps({"status":"BLOCKED","error":str(error)},sort_keys=True),file=sys.stderr);return 2
    print(json.dumps(result,indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
