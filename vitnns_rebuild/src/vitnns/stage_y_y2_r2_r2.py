"""CPU-only source validation and canonical resource accounting."""
from __future__ import annotations

import ast
import hashlib
import importlib.abc
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

STAGE = "Y2-R2-R2"
TOKEN = "STAGE_Y2_R2_R2_CPU_ONLY_RESOURCE_LEDGER_CANONICALIZATION_AND_COMPLETION_RECOVERY_ONLY"
ROOT = Path("evidence/stage_y/y2_r2_r2_clean_formal_20260912/formal")
REPORT = Path("reports/stage_y_y2_r2_r2_20260912/report.md")
SPEC = Path("config/stage_y/y2_r2_r2_run_spec.json")
SCHEMA = Path("config/stage_y/y2_r2_r2_artifact.schema.json")
SPEC_SCHEMA = Path("config/stage_y/y2_r2_r2_run_spec.schema.json")
AMENDMENT = Path("docs/stage_y_y2_r2_r2_cpu_only_ledger_recovery_amendment_20260912.md")
BASE = "dbb63a2d2c96243f05af5abdeffa45045590f58a"
PRODUCER = "refs/archive/stage-y-y2-r2-r2-resource-ledger-recovery-producer"
PASS_REF = "refs/archive/stage-y-y2-r2-r2-clean-cohort-evidence-completion"
BLOCKED_REF = "refs/archive/stage-y-y2-r2-r2-resource-ledger-recovery-blocked"
FILES = (AMENDMENT,SPEC,SCHEMA,SPEC_SCHEMA,Path("src/vitnns/stage_y_y2_r2_r2.py"),Path("scripts/run_stage_y_y2_r2_r2.py"),Path("scripts/validate_stage_y_y2_r2_r2.py"),Path("tests/test_stage_y_y2_r2_r2.py"))
FORBIDDEN = {"torch","torchvision","timm","PIL","cv2","cupy","numba","jax","tensorflow"}
_GUARDED = False

class RecoveryError(RuntimeError):
    code = "BLOCKED_PROVENANCE"
class AccessError(RecoveryError):
    code = "BLOCKED_ACCESS"
class NumericalError(RecoveryError):
    code = "FAILED_NUMERICAL"

def require(flag,message,error=RecoveryError):
    if not flag: raise error(message)

class ImportBoundary(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split(".")[0] in FORBIDDEN:raise AccessError("forbidden framework import: "+fullname)
        return None

def runtime_guard():
    global _GUARDED
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CUDA_VISIBLE_DEVICES must be empty",AccessError)
    require(not FORBIDDEN.intersection(k.split(".")[0] for k in sys.modules),"accelerator/decoder already imported",AccessError)
    if _GUARDED:return
    sys.meta_path.insert(0,ImportBoundary())
    def audit(event,args):
        if event in ("socket.connect","socket.bind","socket.getaddrinfo"):
            raise AccessError("external network access is not authorized")
        if event=="subprocess.Popen":
            command=args[1]
            require(isinstance(command,(list,tuple)) and Path(str(command[0])).name=="git","only Git subprocesses are authorized",AccessError)
        if event=="open" and isinstance(args[0],(str,bytes,os.PathLike)):
            name=os.fsdecode(args[0]);mode=args[1] or "";flags=args[2] or 0
            if name.startswith(str(old.y2.PROJECT_ROOT)+"/evidence/") and (any(c in mode for c in "wax+") or flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT)):
                require("/y2_r2_r2_clean_formal_20260912/formal/" in name and not name.endswith(".npy"),"write outside metadata-only new root",AccessError)
    sys.addaudithook(audit);_GUARDED=True

import numpy as np
from jsonschema import Draft202012Validator
from . import stage_y_y2_r2 as old
from .stage_y_y2_r2_schema import science_schema

NEXT=old.NEXT
STAGES=("Y2","Y2-R1","Y2-R2","Y2-R2-R1",STAGE)
ROOTS=(old.Y2_ROOT,old.Y2_R1_ROOT,old.Y2_R2_ROOT,old.ROOT)
REFS={**old.EXPECTED_REFS,old.PRODUCER_REF:"ce56ee5bdb92910d3ed4b0e8a0bbf6a1093c9e8b",old.BLOCKED_REF:BASE}
PROTECTED=old.PROTECTED+((old.ROOT,BASE),(old.REPORT_ROOT,BASE))
DEPENDENCIES=tuple(dict.fromkeys(old.SOURCE_FILES+old.DEPENDENCIES))
canonical=old.canonical_digest
read=old.read_json
sha=old.sha_file
record=old.file_record


def git(repo,*args):return old.y2.git(repo,*args)
def canonical_bytes(value):return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def checked(path):
    value=read(path);old.validate_canonical(value);return value

def obj(properties,optional=()):return {"type":"object","additionalProperties":False,"properties":properties,"required":[k for k in properties if k not in optional]}
def arr(item):return {"type":"array","items":item}
def nullable(item):return {"anyOf":[item,{"type":"null"}]}
S={"type":"string"};I={"type":"integer","minimum":0};N={"type":"number","minimum":0};B={"type":"boolean"};H={"type":"string","pattern":"^[0-9a-f]{64}$"};G={"type":"string","pattern":"^[0-9a-f]{40}$"}
F=obj({"path":S,"bytes":I,"sha256":H,"canonical_digest":H},("canonical_digest",))
COUNTS=obj({k:I for k in old.SOURCE_TYPES})


def artifact_schema():
    rebind=obj({"selection_order":I,"source_object_id":S,"class_id":I,"source_type":{"enum":list(old.SOURCE_TYPES)},"original_producing_stage":{"enum":["Y2-R1","Y2-R2-R1"]},"validation_stage":{"const":STAGE},"evidence_mode":{"const":"REBIND_ONLY"},"status":{"const":"VALID"},"source_terminal":F,"source_terminal_status":S,"scientific_result":nullable(F),"array_records_digest":H,"attempt_records_digest":H,"validated_array_count":{"const":262},"validation_digest":H,"gpu_reexecuted":{"const":False},"arrays_copied":{"const":False},"scientific_scope_changed":{"const":False},"intervention_payload_retained":B})
    observation=obj({"observation_id":S,"stage":{"enum":list(STAGES)},"kind":S,"owner":S,"metric":{"enum":["cpu_seconds","gpu_active_seconds","wall_seconds","worker_wall_seconds","stage_wall_seconds","retained_bytes","download_bytes","peak_reserved_gpu_bytes"]},"source_records":arr(F),"source_fields":arr(S),"raw_value":nullable(N),"included":B,"reason":S,"duplicate_of":nullable(S),"measurement_status":{"enum":["MEASURED","PROVEN_ZERO","NOT_RECORDED","EXCLUDED_DUPLICATE"]}})
    total=obj({"stage":S,"cpu_seconds":nullable(N),"cpu_hours":nullable(N),"gpu_active_seconds":nullable(N),"h20_gpu_hours":nullable(N),"measured_cpu_seconds":N,"measured_gpu_active_seconds":N,"cpu_total_is_lower_bound":B,"gpu_total_is_lower_bound":B,"wall_seconds_sum":nullable(N),"measured_wall_seconds_sum":N,"worker_wall_seconds":arr(N),"critical_path_wall_seconds":nullable(N),"stage_wall_seconds":nullable(N),"retained_bytes":I,"retained_storage_gib":N,"download_bytes":nullable(I),"measured_download_bytes":I,"peak_reserved_gpu_bytes":I,"missing_observation_ids":arr(S)})
    binding=obj({"root":S,"commit":G,"files":arr(F),"files_digest":H})
    counts=obj({"denominator":{"const":100},"valid_clean_count":I,"source_counts":COUNTS,"records_digest":H,"new_scientific_attempt_count":{"const":0},"gpu_execution":{"const":False},"formal_outcome_access":{"const":False},"next_authorization":{"enum":[NEXT,"NONE"]}})
    test=obj({"name":S,"status":{"const":"PASS"}})
    variants={
        "pending":obj({"started_unix_ns":I,"next_authorization":{"const":"NONE_WHILE_PENDING"}}),
        "implementation":obj({"producer_commit":G,"producer_ref":S,"source_files":arr(F),"files_digest":H}),
        "bindings":obj({"bindings":arr(binding),"bindings_digest":H}),
        "source_snapshot":obj({"source_files":arr(F),"files_digest":H}),
        "preflight":obj({"tests":arr(test),"cpu_seconds":N,"wall_seconds":N,"measurement_window":S,"startup_commit":G,"source_files":arr(F),"files_digest":H}),
        "rebind":rebind,
        "manifest":obj({"source_type":S,"records":arr(rebind),"record_count":I,"records_digest":H}),
        "validation":obj({"valid_clean_count":I,"source_counts":COUNTS,"records_digest":H,"checks":arr(S)}),
        "runtime":obj({"cpu_seconds":N,"wall_seconds":N,"gpu_active_seconds":{"const":0},"download_bytes":{"const":0},"measurement_window":S}),
        "observations":obj({"records":arr(observation),"record_count":I,"records_digest":H}),
        "ledger":obj({"stage_totals":arr(total),"cumulative":total,"observations_digest":H,"reporting_only":{"const":True},"reduction_contract":S}),
        "missingness":obj({"records":arr(observation),"cpu_total_is_lower_bound":{"const":True},"unmeasured_overhead":arr(S)}),
        "discrepancy":obj({"classification":S,"source_file":F,"rows":arr(obj({"metric":S,"stored_value":N,"shard_order_sum":N,"selection_order_sum":N,"canonical_value":N,"fsum_permutations_identical":B})),"source_identity_verified":B}),
        "overruns":obj({"records":arr(obj({"stage":S,"metric":S,"observed":N,"former_ceiling":N,"classification":{"const":"RESOURCE_OVERRUN_REPORTED"}})),"reporting_only":{"const":True}}),
        "access":obj({"cpu_only":{"const":True},"clean_array_validation_performed":B,"new_scientific_attempt_count":{"const":0},"gpu_execution":{"const":False},"arrays_copied":{"const":False},"external_accesses":{"const":[]},"later_stages_started":{"const":[]},"bootstrap_executed":{"const":False},"holm_executed":{"const":False},"claim_decision_executed":{"const":False},"forbidden_imports_observed":{"const":[]}}),
        "audit":obj({"checks":arr(test),"rejected_count":I}),
        "reconciliation":counts,
        "inventory":obj({"files":arr(F),"file_count":I,"total_bytes":I,"files_digest":H,"excluded":arr(S)}),
        "completion":obj({**counts["properties"],"workflow_status":S,"reconciliation":nullable(F),"validation_report":nullable(F),"whole_root_inventory":F,"payload_manifest":nullable(F),"report":F,"failure":nullable(F)}),
        "failure":obj({"phase":S,"failure_type":S,"message":S,"validated_unit_count":I,"source_counts":COUNTS,"next_authorization":{"const":"NONE"}}),
    }
    schema=obj({"$schema":{"const":SCHEMA.name},"schema_version":{"const":1},"stage_id":{"const":STAGE},"document_type":{"enum":list(variants)},"status":{"enum":["PENDING","VALID","PASS","BLOCKED_STARTUP_STATE","BLOCKED_PROVENANCE","BLOCKED_SOURCE_DRIFT","BLOCKED_ACCESS","BLOCKED_INFRASTRUCTURE","FAILED_NUMERICAL"]},"payload":{"anyOf":[{"$ref":"#/$defs/"+k} for k in variants]},"canonical_digest":H})
    schema.update({"$schema":"https://json-schema.org/draft/2020-12/schema","$defs":variants,"allOf":[{"if":{"properties":{"document_type":{"const":k}}},"then":{"properties":{"payload":{"$ref":"#/$defs/"+k}}}} for k in variants]})
    return schema

_SCHEMA_VALIDATOR=None

def validate_document(value):
    global _SCHEMA_VALIDATOR
    old.validate_canonical(value)
    if _SCHEMA_VALIDATOR is None:_SCHEMA_VALIDATOR=Draft202012Validator(artifact_schema())
    _SCHEMA_VALIDATOR.validate(value)

def signed(kind,payload,status="PASS"):
    value={"$schema":SCHEMA.name,"schema_version":1,"stage_id":STAGE,"document_type":kind,"status":status,"payload":payload}
    value["canonical_digest"]=canonical(value);validate_document(value);return value

def write(path,kind,payload,status="PASS"):
    document=signed(kind,payload,status);old.y2.write_json_new(path,document);return document

def terminal(pending,path,value):
    validate_document(value);old.y2.publish_terminal(pending,path,value)


def git_binding(repo,relative,commit):
    top=Path(git(repo,"rev-parse","--show-toplevel"));base=repo/relative
    tree_relative=base.relative_to(top)
    entries=[x.split(b"\t",1) for x in subprocess.check_output(["git","ls-tree","-rz",commit,"--",str(tree_relative)],cwd=top).split(b"\0") if x]
    paths=[top/name.decode() for _,name in entries]
    require(set(paths)=={p for p in base.rglob("*") if p.is_file()},"historical Git file roster drift: "+str(relative))
    data=subprocess.check_output(["git","cat-file","--batch"],cwd=repo,input=b"".join(meta.split()[2]+b"\n" for meta,_ in entries));offset=0;rows=[]
    for path in paths:
        end=data.index(b"\n",offset);size=int(data[offset:end].split()[2]);blob=data[end+1:end+1+size];offset=end+size+2
        require(not path.is_symlink(),"historical symlink source")
        row=record(path,repo)
        if blob.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
            lines=blob.decode().splitlines();require(row["sha256"]==lines[1].split("sha256:")[1] and row["bytes"]==int(lines[2].split()[1]),"historical LFS bytes drift: "+str(path))
        else:require(row["sha256"]==hashlib.sha256(blob).hexdigest(),"historical Git bytes drift: "+str(path))
        rows.append(row)
    return {"root":str(relative),"commit":commit,"files":rows,"files_digest":canonical({"files":rows})}


def source_snapshots(repo):
    rows=[record(repo/p,repo) for p in DEPENDENCIES]
    for row in rows:
        top=Path(git(repo,"rev-parse","--show-toplevel"));relative=(repo/row["path"]).relative_to(top)
        blob=subprocess.check_output(["git","show",f"{BASE}:{relative}"],cwd=repo)
        require(hashlib.sha256(blob).hexdigest()==row["sha256"],"frozen producer source drift")
    return {"source_files":rows,"files_digest":canonical({"files":rows})}


def check_refs(repo,remote=True):
    remote_rows={}
    if remote:remote_rows={line.split()[1]:line.split()[0] for line in git(repo,"ls-remote","origin").splitlines()}
    for ref,commit in REFS.items():
        require(git(repo,"rev-parse",ref)==commit and (not remote or remote_rows.get(ref)==commit),"archive ref drift: "+ref)
    require(not git(repo,"for-each-ref","--format=%(refname)",old.COMPLETION_REF) and old.COMPLETION_REF not in remote_rows,"old PASS ref exists")
    return remote_rows


def no_workers_or_later_stages(repo):
    for parent in (repo/"evidence/stage_y",repo/"reports"):
        require(not any(p.name.startswith(("y3","y4","stage_y_y3","stage_y_y4")) for p in parent.iterdir()),"later-stage root exists")
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:args=(proc/"cmdline").read_bytes().split(b"\0")
        except OSError:continue
        if not args or b"python" not in args[0]:continue
        require(not any(a.startswith(b"vitnns.stage_y_y") and a.endswith(b"worker") for a in args[1:3]),"residual Stage Y worker")


def startup(repo,allow_sources=False):
    runtime_guard();require(os.path.realpath(sys.executable)==os.path.realpath(old.PYTHON),"Python executable drift")
    remote=check_refs(repo);require(git(repo,"symbolic-ref","--short","HEAD")=="stage-d-d3-pass","branch drift")
    require(git(repo,"rev-parse","HEAD")==git(repo,"rev-parse","@{upstream}")==remote.get("refs/heads/stage-d-d3-pass")==BASE,"startup HEAD drift")
    status=git(repo,"status","--porcelain");paths=[line[2:].strip().removeprefix(repo.name+"/") for line in status.splitlines()]
    require(not paths or allow_sources and all(Path(p) in FILES for p in paths),"unexpected worktree changes")
    lfs=git(repo,"lfs","status");require("(LFS:" not in lfs and "(Git:" not in lfs,"pending LFS objects")
    for ref in (PRODUCER,PASS_REF,BLOCKED_REF):require(ref not in remote and not git(repo,"for-each-ref","--format=%(refname)",ref),"new ref exists")
    require(not (repo/ROOT).exists() and not (repo/REPORT.parent).exists(),"new root exists")
    no_workers_or_later_stages(repo)
    return validate_history(repo)


def validate_history(repo):
    bindings=[git_binding(repo,path,commit) for path,commit in PROTECTED]
    source_snapshots(repo)
    counts=[]
    for relative in ROOTS:
        root=repo/relative;inv=checked(root/"whole_root_inventory.json");p=inv["payload"]
        actual=[{"path":path.relative_to(root).as_posix(),"bytes":path.stat().st_size,"sha256":sha(path)} for path in sorted(root.rglob("*")) if path.is_file() and path.relative_to(root).as_posix() not in p["excluded"]]
        require(actual==p["files"] and canonical({"files":actual})==p["files_digest"],"predecessor root inventory drift")
        counts.append(len(actual))
    completion=checked(repo/old.ROOT/"completion/y2_r2_r1.json")
    rec=checked(repo/old.ROOT/"reconciliation.json")
    require(completion["status"]=="BLOCKED_SOURCE_DRIFT" and completion["payload"]["next_authorization"]=="NONE" and rec["status"]=="PASS" and rec["payload"]["valid_clean_count"]==100,"predecessor completion/reconciliation drift")
    return {"bindings":bindings,"bindings_digest":canonical(bindings)}


def validate_cohort(repo,raw_release=True):
    cohort=checked(repo/old.Y2_ROOT/"cohort_registry.json");rows=cohort["payload"]["records"]
    require(cohort["canonical_digest"]==old.COHORT_DIGEST and cohort["payload"]["selection_digest"]==old.SELECTION_DIGEST,"cohort digest drift")
    require([r["selection_order"] for r in rows]==list(range(100)) and len({r["class_id"] for r in rows})==100,"cohort cardinality/classes drift")
    audit=checked(repo/old.Y2_ROOT/"audit_subset_registry.json");perturb=checked(repo/old.Y2_ROOT/"perturbation_subset_registry.json")
    require(old.y2.subset_registry(rows,old.y2.AUDIT_NAMESPACE,10,"audit_10")==audit["payload"] and old.y2.subset_registry(rows,old.y2.PERTURBATION_NAMESPACE,20,"perturbation_20")==perturb["payload"],"subset identity drift")
    if raw_release:
        _,excluded=old.y2.exclusion_union(repo);candidates,_=old.y2.enumerate_candidates(repo);selected,proof=old.y2.select_cohort(candidates,excluded)
        require(selected==rows and proof["selection_digest"]==old.SELECTION_DIGEST,"raw-release selection drift")
    require(sha(repo/"assets/models/vit_base_patch16_224_weights.semantic_candidate.pth")==old.CHECKPOINT_SHA,"checkpoint byte identity drift")
    return rows


def source_terminal(repo,order):
    if order in old.MISSING_ORDERS:
        shard=old.MISSING_ORDERS.index(order)%4;base=old.ROOT;stage="Y2-R2-R1";kind=old.SOURCE_TYPES[2]
    else:
        shard=order%4;base=old.Y2_R1_ROOT;stage="Y2-R1";kind=old.SOURCE_TYPES[int(order in old.COMPLETED_BUDGET_ORDERS)]
    return repo/base/f"shards/shard-{shard:02d}/units/unit-{order:03d}/completion/unit.json",stage,kind


def validate_source(repo,order):
    require((repo/ROOT/"producer_implementation_seal.json").is_file(),"formal array validation requires producer seal")
    path,stage,kind=source_terminal(repo,order);t=checked(path);q=t["payload"];unit=path.parent.parent
    require(t["stage_id"]==stage and q["selection_order"]==order,"producing stage/order drift")
    completed=kind==old.SOURCE_TYPES[1]
    require(t["status"]==("BLOCKED_BUDGET" if completed else "VALID"),"source terminal status drift")
    if completed:require(q["failure_code"]=="per-shard resource cap exceeded by completed unit" and q["scientific_record"] is None,"completed-budget failure drift")
    old.validate_source_binding(repo,t,order);old.validate_attempts(repo,t,unit)
    gate_path=unit.parents[1].parent.parent/("startup_gate.json" if stage=="Y2-R2-R1" else "multi_gpu_startup_gate.json")
    gate=checked(gate_path);device=gate["payload"]["devices"][q["shard_id"]]
    worker=checked(unit.parents[1]/"worker_identity.json")["payload"]
    require(all(worker[k]==device[k] for k in ("physical_index","gpu_uuid","pci_bus_id","environment_digest")),"source worker/startup ownership drift")
    arrays=old.load_validated_arrays(repo,t,unit)
    final=q["quadrature_attempts"][-1]
    if final["identity_max_abs_error"]==0:require(old.array_digest(arrays["input_attribution"])==final["array_digest"],"identity witness digest drift")
    closure=validate_closure(arrays,float(final["input_delta"]))
    require(closure["status"]=="PASS","source closure failure",NumericalError)
    science=None
    if completed:
        expected={512:dict(zip(old.CALL_KEYS,(7359,352,6272,25088,128,3584))),1024:dict(zip(old.CALL_KEYS,(14015,736,12544,50176,256,7168)))}
        require(final["steps"] in expected and q["resource_telemetry"]["calls"]==expected[final["steps"]],"completed intervention call proof drift")
    else:
        old.validate_scientific_result(repo,t,unit,closure)
        science=q["scientific_result" if stage=="Y2-R2-R1" else "scientific_record"]
    expected_files={path,*[repo/r["path"] for r in q["array_records"]+q["attempt_records"]]}
    if science:expected_files.add(repo/science["path"])
    if stage=="Y2-R2-R1":
        for key in ("array_inventory","resource_record"):
            document=checked(old.checked_record(repo,q[key],unit));expected_files.add(repo/q[key]["path"])
            if key=="resource_record":require(document["payload"]==q["resource_telemetry"],"resource source link drift")
            else:require(document["payload"]["array_records"]==q["array_records"] and document["payload"]["records_digest"]==canonical(q["array_records"]),"array inventory drift")
        original=old.terminal_by_order(repo)[order][0]
        require(q["predecessor_terminal"]==record(original,repo),"computed retry predecessor drift")
    require({p for p in unit.rglob("*") if p.is_file()}==expected_files,"source unit payload not closed")
    return {"selection_order":order,"source_object_id":q["source_object_id"],"class_id":old.cohort_by_order(repo)[order]["class_id"],"source_type":kind,"original_producing_stage":stage,"validation_stage":STAGE,"evidence_mode":"REBIND_ONLY","status":"VALID","source_terminal":record(path,repo),"source_terminal_status":t["status"],"scientific_result":science,"array_records_digest":canonical(q["array_records"]),"attempt_records_digest":canonical(q["attempt_records"]),"validated_array_count":len(arrays),"validation_digest":canonical({"closure":closure,"arrays":q["array_records"],"target":q["target_identity"],"attempts":q["attempt_records"]}),"gpu_reexecuted":False,"arrays_copied":False,"scientific_scope_changed":False,"intervention_payload_retained":not completed}


def validate_scientific_array(value,shape,dtype=np.dtype("float32")):
    value=np.asarray(value)
    require(value.shape==tuple(shape) and value.dtype==dtype and value.flags.c_contiguous,"scientific array contract drift")
    require(np.isfinite(value).all(),"nonfinite scientific array",NumericalError)
    return value

BLOCKS=12
INPUT_ATOL,INPUT_RTOL=1e-4,0.005
CUT_PIXEL_ATOL,CUT_PIXEL_RTOL=2e-5,0.005
CUT_TOTAL_ATOL,CUT_TOTAL_RTOL=2e-4,0.01


def validate_closure(arrays,delta):
    input_attr=validate_scientific_array(arrays["input_attribution"],(3,224,224))
    input_cells=validate_scientific_array(arrays["input_cell_map"],(56,56))
    input_patches=input_attr.astype(np.float64).sum(0).reshape(14,16,14,16).sum((1,3)).reshape(196)
    maximum=float(np.max(np.abs(input_attr)))
    pixel_threshold=CUT_PIXEL_ATOL+CUT_PIXEL_RTOL*maximum
    total_threshold=CUT_TOTAL_ATOL+CUT_TOTAL_RTOL*abs(delta)
    input_error=abs(float(input_attr.astype(np.float64).sum())-delta)
    failures=[];per_cut={};head_max=0.0;branch_max=0.0;scalar_max=0.0
    if input_error>INPUT_ATOL+INPUT_RTOL*abs(delta):failures.append("input_completeness")
    def check(name,matrix,cls,cells):
        nonlocal scalar_max
        route=matrix.astype(np.float64).sum(1)+cls.astype(np.float64)
        patch_error=float(np.max(np.abs(route-input_patches)))
        cell_error=float(np.max(np.abs(cells.astype(np.float64)-input_cells.astype(np.float64))))
        total_error=abs(float(matrix.astype(np.float64).sum()+cls.astype(np.float64).sum())-delta)
        scalar_error=abs(float(cells.astype(np.float64).sum())-float(matrix.astype(np.float64).sum()+cls.astype(np.float64).sum()))
        scalar_max=max(scalar_max,scalar_error)
        passed=patch_error<=max(pixel_threshold*16*16*3,total_threshold) and cell_error<=max(pixel_threshold*16*3,total_threshold) and total_error<=total_threshold and scalar_error<=total_threshold
        per_cut[name]={"patch_row_max_abs_error":patch_error,"cell_max_abs_error":cell_error,"total_abs_error":total_error,"scalar_map_matrix_abs_error":scalar_error,"pass":passed}
        if not passed:failures.append(name)
    check("post_patch_embedding_tokens",arrays["post_embedding_matrix"],arrays["post_embedding_cls"],arrays["post_embedding_cell"])
    for block in range(BLOCKS):
        prefix=f"b{block:02d}"
        check(prefix+".input",arrays[prefix+"_input_matrix"],arrays[prefix+"_input_cls"],arrays[prefix+"_input_cell"])
        heads_matrix=arrays[prefix+"_first_heads_matrix"].astype(np.float64);heads_cls=arrays[prefix+"_first_heads_cls"].astype(np.float64);heads_cell=arrays[prefix+"_first_heads_cell"].astype(np.float64)
        attention_matrix=arrays[prefix+"_first_attention_matrix"].astype(np.float64);attention_cls=arrays[prefix+"_first_attention_cls"].astype(np.float64);attention_cell=arrays[prefix+"_first_attention_cell"].astype(np.float64)
        current_head=max(float(np.max(np.abs(heads_matrix.sum(0)-attention_matrix))),float(np.max(np.abs(heads_cls.sum(0)-attention_cls))),float(np.max(np.abs(heads_cell.sum(0)-attention_cell))))
        head_max=max(head_max,current_head)
        first_matrix=arrays[prefix+"_first_identity_matrix"].astype(np.float64)+attention_matrix
        first_cls=arrays[prefix+"_first_identity_cls"].astype(np.float64)+attention_cls
        first_cell=arrays[prefix+"_first_identity_cell"].astype(np.float64)+attention_cell
        check(prefix+".first_residual",first_matrix.astype(np.float32),first_cls.astype(np.float32),first_cell.astype(np.float32))
        second_matrix=arrays[prefix+"_second_residual_matrix"].astype(np.float64)+arrays[prefix+"_second_mlp_matrix"].astype(np.float64)
        second_cls=arrays[prefix+"_second_residual_cls"].astype(np.float64)+arrays[prefix+"_second_mlp_cls"].astype(np.float64)
        second_cell=arrays[prefix+"_second_residual_cell"].astype(np.float64)+arrays[prefix+"_second_mlp_cell"].astype(np.float64)
        check(prefix+".second_residual",second_matrix.astype(np.float32),second_cls.astype(np.float32),second_cell.astype(np.float32))
        check(prefix+".output",arrays[prefix+"_output_matrix"],arrays[prefix+"_output_cls"],arrays[prefix+"_output_cell"])
        branch_max=max(branch_max,float(np.max(np.abs(first_cell-input_cells))),float(np.max(np.abs(second_cell-input_cells))))
        if current_head>total_threshold:failures.append(prefix+".head_sum")
    check("final_norm",arrays["final_norm_matrix"],arrays["final_norm_cls"],arrays["final_norm_cell"])
    check("classifier_input",np.zeros((196,196),dtype=np.float32),arrays["classifier_input_cls"],arrays["classifier_input_cell"])
    return {"status":"PASS" if not failures else "FAILED_NUMERICAL","failures":failures,"input_abs_error":input_error,"input_threshold":INPUT_ATOL+INPUT_RTOL*abs(delta),"pixel_base_threshold":pixel_threshold,"total_threshold":total_threshold,"head_sum_max_abs_error":head_max,"branch_cell_max_abs_error":branch_max,"scalar_map_matrix_max_abs_error":scalar_max,"per_cut":per_cut}


def observation(stage,kind,owner,metric,value,sources,fields,included=True,reason="original nonoverlapping measurement",duplicate=None,status="MEASURED"):
    require(value is None or type(value) in (int,float) and math.isfinite(value) and value>=0,"invalid raw resource value")
    if metric in ("retained_bytes","download_bytes","peak_reserved_gpu_bytes") and value is not None:require(type(value) is int,"integer metric became floating")
    return {"observation_id":"/".join((stage,kind,str(owner),metric)),"stage":stage,"kind":kind,"owner":str(owner),"metric":metric,"source_records":sources,"source_fields":fields,"raw_value":value,"included":included,"reason":reason,"duplicate_of":duplicate,"measurement_status":status}


def sorted_observations(rows):
    require(len({r["observation_id"] for r in rows})==len(rows),"duplicate observation")
    return sorted(rows,key=lambda r:(STAGES.index(r["stage"]),r["kind"],r["owner"],r["metric"]))


def resource_observations(repo):
    rows=[]
    def add_metrics(stage,kind,owner,path,payload,field_prefix):
        source=record(path,repo)
        for metric in ("cpu_seconds","gpu_active_seconds","wall_seconds","peak_reserved_gpu_bytes","download_bytes"):
            value=payload.get(metric)
            status="MEASURED" if value is not None else "NOT_RECORDED"
            if metric=="download_bytes" and value is None:
                rows.append(observation(stage,kind,owner,metric,None,[source],[field_prefix+metric],reason="download telemetry not instrumented in this receipt",status=status))
            else:rows.append(observation(stage,kind,owner,metric,value,[source],[field_prefix+metric],status=status))
        if "retained_bytes" in payload:
            rows.append(observation(stage,kind,owner,"retained_bytes",payload["retained_bytes"],[source],[field_prefix+"retained_bytes"],False,"covered by unique-root retained storage",stage+"/storage/root/retained_bytes","EXCLUDED_DUPLICATE"))
    for stage,relative in zip(STAGES[:4],ROOTS):
        root=repo/relative
        for path in sorted(root.glob("shards/shard-*/units/unit-*/completion/unit.json")):
            q=checked(path)["payload"];owner=f"unit-{q['selection_order']:03d}"
            if stage=="Y2-R2-R1":
                actual=old.checked_record(repo,q["resource_record"],path.parent.parent)
                resource=checked(actual)["payload"];require(resource==q["resource_telemetry"],"raw unit resource copy drift")
                add_metrics(stage,"unit",owner,actual,resource,"payload.")
                for metric in ("cpu_seconds","gpu_active_seconds","wall_seconds"):
                    rows.append(observation(stage,"terminal_copy",owner,metric,resource[metric],[record(path,repo)],["payload.resource_telemetry."+metric],False,"terminal embeds existing resource receipt",f"{stage}/unit/{owner}/{metric}","EXCLUDED_DUPLICATE"))
            else:add_metrics(stage,"unit",owner,path,q["resource_telemetry"],"payload.resource_telemetry.")
        shards=[]
        for path in sorted(root.glob("shards/shard-*/completion/shard.json")):
            q=checked(path)["payload"];shards.append(q);sid=f"shard-{q['shard_id']:02d}"
            key="fixed_initialization" if stage=="Y2-R2-R1" else "fixed_initialization_resource"
            add_metrics(stage,"initialization",sid,path,q[key],"payload."+key+".")
            wall_key="worker_wall_seconds" if stage=="Y2-R2-R1" else "shard_elapsed_seconds"
            rows.append(observation(stage,"worker_elapsed",sid,"worker_wall_seconds",q[wall_key],[record(path,repo)],["payload."+wall_key]))
        if stage=="Y2-R2":
            source=record(root/"completion/y2_r2.json",repo)
            for metric in ("cpu_seconds","wall_seconds","stage_wall_seconds"):
                rows.append(observation(stage,"unrecorded","recovery",metric,None,[source],[],reason="historical control-plane measurement not retained",status="NOT_RECORDED"))
            for metric in ("gpu_active_seconds","peak_reserved_gpu_bytes","download_bytes"):
                rows.append(observation(stage,"absent_execution","recovery",metric,0,[record(root/"access_audit.json",repo)],[],reason="immutable access audit and empty scientific/worker root prove absence",status="PROVEN_ZERO"))
        if stage=="Y2-R2-R1":
            for name in ("rebind_validation_resource.json","reconciliation_resource.json","producer_preflight.json"):
                path=root/name;q=checked(path)["payload"]
                if name=="producer_preflight.json":q=q["resource_telemetry"]
                add_metrics(stage,"verification",name,path,q,"payload.resource_telemetry." if name=="producer_preflight.json" else "payload.")
            for path in sorted(root.glob("rebind_units/unit-*/completion/unit.json")):
                q=checked(path)["payload"];owner=f"unit-{q['selection_order']:03d}"
                for metric in ("cpu_seconds","gpu_active_seconds","wall_seconds"):
                    rows.append(observation(stage,"rebind_copy",owner,metric,q["resource_telemetry"][metric],[record(path,repo)],["payload.resource_telemetry."+metric],False,"original Y2-R1 measurement; validation is charged separately",f"Y2-R1/unit/{owner}/{metric}","EXCLUDED_DUPLICATE"))
        if shards:
            start_path=root/("execution_start.json" if stage=="Y2-R2-R1" else "multi_gpu_startup_gate.json")
            start=checked(start_path)["payload"]["started_unix_ns"];end=max(s["completed_unix_ns"] for s in shards)
            rows.append(observation(stage,"stage_elapsed","execution","stage_wall_seconds",(end-start)/1e9,[record(start_path,repo)]+[record(p,repo) for p in sorted(root.glob("shards/shard-*/completion/shard.json"))],["started_unix_ns","max(completed_unix_ns)"],reason="integer timestamp subtraction then seconds conversion"))
        files=[p for p in sorted(root.rglob("*")) if p.is_file()]
        sources=[record(root/"whole_root_inventory.json",repo)]+[record(p,repo) for p in sorted((root/"completion").glob("*.json"))]
        rows.append(observation(stage,"storage","root","retained_bytes",sum(p.stat().st_size for p in files),sources,["inventory-listed bytes plus excluded immutable inventory and completion bytes"],reason="unique full historical root files; no scientific array copies"))
    root=repo/ROOT
    for name in ("producer_preflight.json","verification_runtime.json"):
        path=root/name;q=checked(path)["payload"]
        add_metrics(STAGE,"verification",name,path,{"cpu_seconds":q["cpu_seconds"],"wall_seconds":q["wall_seconds"],"gpu_active_seconds":0,"peak_reserved_gpu_bytes":0,"download_bytes":0},"payload.")
    paths=[p for p in sorted(root.rglob("*")) if p.is_file() and (p.relative_to(root).parts[0]=="units" or p.name in ("producer_implementation_seal.json","producer_preflight.json","prospective_amendment_binding.json","historical_immutable_bindings.json","historical_source_snapshot.json","verification_runtime.json"))]
    rows.append(observation(STAGE,"storage","root","retained_bytes",sum(p.stat().st_size for p in paths),[record(p,repo) for p in paths],["sum of frozen pre-accounting new-root payload file bytes"],reason="excludes derived ledgers manifests report inventory and completion to prevent recursion"))
    rows.append(observation(STAGE,"unrecorded","post_window","wall_seconds",None,[record(root/"verification_runtime.json",repo)],[],reason="final seal/publication operations outside captured verification window",status="NOT_RECORDED"))
    rows.append(observation(STAGE,"unrecorded","post_window","cpu_seconds",None,[record(root/"verification_runtime.json",repo)],[],reason="final seal/publication operations outside captured verification window",status="NOT_RECORDED"))
    return sorted_observations(rows)


def canonical_ledger(observations):
    rows=sorted_observations(observations);ids={r["observation_id"] for r in rows}
    for row in rows:
        require(row["included"]==(row["duplicate_of"] is None),"included duplicate observation")
        if not row["included"]:require(row["duplicate_of"] in ids,"excluded duplicate points to absent source")
        if row["measurement_status"]=="NOT_RECORDED":require(row["raw_value"] is None,"missing measurement was zero-imputed")
        else:require(row["raw_value"] is not None,"measured value absent")
    def total(stage,current):
        included=[r for r in current if r["included"]]
        def measured(metric):return [r["raw_value"] for r in included if r["metric"]==metric and r["raw_value"] is not None]
        def missing(metric):return any(r["metric"]==metric and r["raw_value"] is None for r in included)
        def seconds(metric):return math.fsum(measured(metric))
        cpu=seconds("cpu_seconds");gpu=seconds("gpu_active_seconds");wall=seconds("wall_seconds");storage=sum(measured("retained_bytes"));download=sum(measured("download_bytes"));workers=measured("worker_wall_seconds");stage_wall=measured("stage_wall_seconds")
        return {"stage":stage,"cpu_seconds":None if missing("cpu_seconds") else cpu,"cpu_hours":None if missing("cpu_seconds") else cpu/3600,"gpu_active_seconds":None if missing("gpu_active_seconds") else gpu,"h20_gpu_hours":None if missing("gpu_active_seconds") else gpu/3600,"measured_cpu_seconds":cpu,"measured_gpu_active_seconds":gpu,"cpu_total_is_lower_bound":missing("cpu_seconds"),"gpu_total_is_lower_bound":missing("gpu_active_seconds"),"wall_seconds_sum":None if missing("wall_seconds") else wall,"measured_wall_seconds_sum":wall,"worker_wall_seconds":workers,"critical_path_wall_seconds":max(workers) if workers else None,"stage_wall_seconds":None if missing("stage_wall_seconds") or not stage_wall else math.fsum(stage_wall),"retained_bytes":storage,"retained_storage_gib":storage/1024**3,"download_bytes":None if missing("download_bytes") else download,"measured_download_bytes":download,"peak_reserved_gpu_bytes":max(measured("peak_reserved_gpu_bytes"),default=0),"missing_observation_ids":[r["observation_id"] for r in included if r["raw_value"] is None]}
    stages=[total(stage,[r for r in rows if r["stage"]==stage]) for stage in STAGES]
    return {"stage_totals":stages,"cumulative":total("CUMULATIVE",rows),"observations_digest":canonical(rows),"reporting_only":True,"reduction_contract":"stable(stage,kind,owner,metric); math.fsum(seconds); divide once; integer bytes; exclude linked duplicates"}


def validate_observations(repo,rows):
    expected=resource_observations(repo)
    require(sorted_observations(rows)==expected,"resource observation source/value/coverage drift")
    return canonical_ledger(rows)


def discrepancy(repo):
    root=repo/old.ROOT;terms=[checked(p)["payload"] for p in sorted(root.glob("shards/shard-*/units/unit-*/completion/unit.json"))]
    extra=[checked(p)["payload"]["fixed_initialization"] for p in sorted(root.glob("shards/shard-*/completion/shard.json"))]
    extra += [checked(root/p)["payload"] for p in ("rebind_validation_resource.json","reconciliation_resource.json")]
    extra.append(checked(root/"producer_preflight.json")["payload"]["resource_telemetry"])
    groups=([x["resource_telemetry"] for x in terms]+extra,[x["resource_telemetry"] for x in sorted(terms,key=lambda x:x["selection_order"])]+extra)
    stored=checked(root/"resource_ledger.json")["payload"]["stage_resources"]["Y2_R2_R1"];rows=[]
    for metric,key in (("cpu_seconds","cpu_hours"),("gpu_active_seconds","h20_gpu_hours")):
        sums=[sum(float(x.get(metric,0)) for x in group)/3600 for group in groups]
        canonical_values=[math.fsum(float(x.get(metric,0)) for x in group)/3600 for group in groups]
        require(sums[0]==stored[key] and canonical_values[0]==canonical_values[1],"old ledger failure not explained by frozen raw observations")
        rows.append({"metric":key,"stored_value":stored[key],"shard_order_sum":sums[0],"selection_order_sum":sums[1],"canonical_value":canonical_values[0],"fsum_permutations_identical":True})
    return {"classification":"RESOURCE_LEDGER_CANONICALIZATION_AND_SEALING_IMPLEMENTATION_FAILURE","source_file":record(root/"resource_ledger.json",repo),"rows":rows,"source_identity_verified":True}


def manifest_counts(records,complete=True):
    orders=[r["selection_order"] for r in records]
    require(len(orders)==len(set(orders)) and all(0<=o<100 for o in orders),"duplicate/invalid rebind order")
    if complete:require(orders==list(range(100)),"100-unit reconciliation incomplete")
    counts={k:0 for k in old.SOURCE_TYPES}
    for row in records:
        expected=old.SOURCE_TYPES[2] if row["selection_order"] in old.MISSING_ORDERS else old.SOURCE_TYPES[int(row["selection_order"] in old.COMPLETED_BUDGET_ORDERS)]
        stage="Y2-R2-R1" if expected==old.SOURCE_TYPES[2] else "Y2-R1"
        require(row["source_type"]==expected and row["original_producing_stage"]==stage and row["status"]=="VALID" and row["evidence_mode"]=="REBIND_ONLY" and row["validated_array_count"]==262 and not row["gpu_reexecuted"] and not row["arrays_copied"],"source classification/validity drift")
        counts[expected]+=1
    if complete:require(list(counts.values())==[84,3,13],"source partition drift")
    return counts


def verify_implementation(repo):
    seal=read(repo/ROOT/"producer_implementation_seal.json");validate_document(seal);p=seal["payload"]
    rows=[record(repo/path,repo) for path in FILES+DEPENDENCIES]
    require(rows==p["source_files"] and canonical({"files":rows})==p["files_digest"],"new implementation seal drift")
    require(git(repo,"rev-parse",PRODUCER)==p["producer_commit"],"producer ref drift")
    top=Path(git(repo,"rev-parse","--show-toplevel"))
    for row in rows:
        blob=subprocess.check_output(["git","show",p["producer_commit"]+":"+str((repo/row["path"]).relative_to(top))],cwd=repo)
        require(hashlib.sha256(blob).hexdigest()==row["sha256"],"producer snapshot mismatch")
    return p["producer_commit"]


def validate_schema_sources(repo):
    require(read(repo/SCHEMA)==artifact_schema(),"serialized schema drift")
    Draft202012Validator.check_schema(artifact_schema());spec=read(repo/SPEC);schema=read(repo/SPEC_SCHEMA)
    Draft202012Validator.check_schema(schema);Draft202012Validator(schema).validate(spec)
    require(spec["stage_id"]==STAGE and spec["authorization"]==TOKEN and spec["denominator"]==100 and spec["new_scientific_attempt_count"]==0 and spec["cpu_only"] is True,"run spec scope drift")


def payload_inventory(root,excluded):
    rows=[record(p,root) for p in sorted(root.rglob("*")) if p.is_file() and p.relative_to(root).as_posix() not in excluded]
    return {"files":rows,"file_count":len(rows),"total_bytes":sum(r["bytes"] for r in rows),"files_digest":canonical({"files":rows}),"excluded":sorted(excluded)}


def access_payload(validated):
    require(not FORBIDDEN.intersection(k.split(".")[0] for k in sys.modules),"forbidden import observed",AccessError)
    return {"cpu_only":True,"clean_array_validation_performed":validated,"new_scientific_attempt_count":0,"gpu_execution":False,"arrays_copied":False,"external_accesses":[],"later_stages_started":[],"bootstrap_executed":False,"holm_executed":False,"claim_decision_executed":False,"forbidden_imports_observed":[]}


def read_records(root):
    result=[]
    for path in sorted(root.glob("units/unit-*/validation.json")):
        doc=read(path);validate_document(doc);require(doc["status"]=="VALID","invalid rebind record");result.append(doc["payload"])
    return result


def reconcile(repo):
    root=repo/ROOT;verify_implementation(repo);validate_schema_sources(repo)
    bindings=validate_history(repo);require(bindings==read(root/"historical_immutable_bindings.json")["payload"],"immutable historical binding drift")
    require(source_snapshots(repo)==read(root/"historical_source_snapshot.json")["payload"],"source snapshot drift")
    validate_cohort(repo);records=[validate_source(repo,o) for o in range(100)]
    require(records==read_records(root),"independent source validation disagrees with rebind")
    counts=manifest_counts(records);rows=resource_observations(repo);ledger=canonical_ledger(rows)
    return records,rows,ledger,{"denominator":100,"valid_clean_count":100,"source_counts":counts,"records_digest":canonical(records),"new_scientific_attempt_count":0,"gpu_execution":False,"formal_outcome_access":False,"next_authorization":NEXT}


def derived_payloads(repo,records,rows,ledger,rec):
    root=repo/ROOT
    outputs={"unified_100_unit_manifest.json":("manifest",{"source_type":"ALL","records":records,"record_count":100,"records_digest":canonical(records)}),"raw_resource_observation_manifest.json":("observations",{"records":rows,"record_count":len(rows),"records_digest":canonical(rows)}),"resource_ledger.json":("ledger",ledger),"cumulative_resource_ledger.json":("ledger",ledger),"reconciliation.json":("reconciliation",rec),"source_recomputing_validation_report.json":("validation",{"valid_clean_count":100,"source_counts":rec["source_counts"],"records_digest":rec["records_digest"],"checks":["raw source hashes and canonical arrays","all 100 source validity contracts","84+3+13 provenance partition","immutable Git and LFS identities","order-invariant canonical resource reduction","CPU-only no empirical execution"]}),"historical_telemetry_missingness_report.json":("missingness",{"records":[r for r in rows if r["raw_value"] is None],"cpu_total_is_lower_bound":True,"unmeasured_overhead":["historical Y2-R2 CPU/wall not retained","unmeasured historical launcher/probe/validation overhead","new final seal and publication outside captured verification window"]}),"old_new_ledger_discrepancy.json":("discrepancy",discrepancy(repo)),"access_audit.json":("access",access_payload(True))}
    for kind in old.SOURCE_TYPES:
        group=[r for r in records if r["source_type"]==kind]
        outputs[kind.lower()+"_manifest.json"]=("manifest",{"source_type":kind,"records":group,"record_count":len(group),"records_digest":canonical(group)})
    overruns=[]
    for total in ledger["stage_totals"]+[ledger["cumulative"]]:
        metrics={"h20_gpu_hours":total["measured_gpu_active_seconds"]/3600,"cpu_hours":total["measured_cpu_seconds"]/3600,"retained_storage_gib":total["retained_storage_gib"],"peak_reserved_gpu_bytes":total["peak_reserved_gpu_bytes"]}
        for metric,ceiling in (("h20_gpu_hours",64.0),("cpu_hours",218.0),("retained_storage_gib",6.0),("peak_reserved_gpu_bytes",21*1024**3)):
            if metrics[metric]>ceiling:overruns.append({"stage":total["stage"],"metric":metric,"observed":metrics[metric],"former_ceiling":ceiling,"classification":"RESOURCE_OVERRUN_REPORTED"})
    outputs["resource_overrun_report.json"]=("overruns",{"records":overruns,"reporting_only":True})
    return outputs


def collect(repo):
    root=repo/ROOT;verify_implementation(repo);validate_cohort(repo)
    start=time.monotonic();cpu=time.process_time()
    for order in range(100):
        row=validate_source(repo,order);write(root/f"units/unit-{order:03d}/validation.json","rebind",row,"VALID")
        print(json.dumps({"event":"source_validated","selection_order":order,"source_type":row["source_type"]}),flush=True)
    write(root/"verification_runtime.json","runtime",{"cpu_seconds":time.process_time()-cpu,"wall_seconds":time.monotonic()-start,"gpu_active_seconds":0,"download_bytes":0,"measurement_window":"100 source validations and rebind writes; excludes final reconciliation/seal/publication"})
    records,rows,ledger,rec=reconcile(repo)
    for name,(kind,payload) in derived_payloads(repo,records,rows,ledger,rec).items():write(root/name,kind,payload)
    return rec


def validate_derived(repo):
    root=repo/ROOT;records,rows,ledger,rec=reconcile(repo)
    for name,(kind,payload) in derived_payloads(repo,records,rows,ledger,rec).items():
        document=read(root/name);validate_document(document);require(document==signed(kind,payload),"derived artifact drift: "+name)
    return rec


def seal(repo,audit):
    root=repo/ROOT;rec=validate_derived(repo)
    write(root/"false_pass_audit.json","audit",audit)
    for path in root.rglob("*.json"):validate_document(read(path))
    require(not list(root.rglob("*.npy")) and not list(root.rglob("attempt*")),"scientific output in CPU-only root")
    report=repo/REPORT;report.parent.mkdir(parents=True,exist_ok=True)
    text="# Y2-R2-R2 CPU-only completion\n\nPASS: 100/100 independently validated rebinds (84+3+13), no scientific attempts or GPU execution, and no array copying. All predecessor failures remain immutable.\n\nResource observations are source-recomputed and canonically ordered before math.fsum. Missing historical CPU/wall values remain null; measured cumulative CPU is a lower bound. Overruns are reporting-only. Final metadata inventory bytes are separate from the frozen pre-accounting storage window.\n\nThe three completed-budget sources preserve missing scalar intervention payloads; immutable execution/call evidence does not reconstruct those payloads.\n\nNo Y3/Y4 or statistical/claim analysis executed. Next authorization: `"+NEXT+"`.\n"
    with report.open("x") as stream:stream.write(text);stream.flush();os.fsync(stream.fileno())
    excluded={"completion/y2_r2_r2.pending.json","completion/y2_r2_r2.json","whole_root_inventory.json","payload_manifest.json"}
    write(root/"payload_manifest.json","inventory",payload_inventory(root,excluded))
    write(root/"whole_root_inventory.json","inventory",payload_inventory(root,excluded-{"payload_manifest.json"}))
    completion=signed("completion",{**rec,"workflow_status":"PASS","reconciliation":record(root/"reconciliation.json",repo),"validation_report":record(root/"source_recomputing_validation_report.json",repo),"whole_root_inventory":record(root/"whole_root_inventory.json",repo),"payload_manifest":record(root/"payload_manifest.json",repo),"report":record(report,repo),"failure":None})
    validate_package(repo,completion,pending=True)
    terminal(root/"completion/y2_r2_r2.pending.json",root/"completion/y2_r2_r2.json",completion)
    return completion


def validate_package(repo,completion,pending=False):
    root=repo/ROOT;validate_document(completion);p=completion["payload"]
    require(completion["status"]==p["workflow_status"]=="PASS" and p["valid_clean_count"]==100 and p["next_authorization"]==NEXT,"completion PASS semantics drift")
    require(p["source_counts"]==manifest_counts(read_records(root)),"completion counts drift")
    for key in ("reconciliation","validation_report","whole_root_inventory","payload_manifest","report"):old.checked_record(repo,p[key])
    for name in ("payload_manifest.json","whole_root_inventory.json"):
        doc=read(root/name);validate_document(doc);require(doc["payload"]==payload_inventory(root,set(doc["payload"]["excluded"])),"package inventory drift: "+name)
    pending_paths=list(root.rglob("*.pending.json"));require(pending_paths==[root/"completion/y2_r2_r2.pending.json"] if pending else not pending_paths,"pending transition incomplete")
    require(not list(root.rglob("*.npy")) and not list(root.rglob("*.tmp")),"forbidden or temporary payload")
    audit=read(root/"false_pass_audit.json");validate_document(audit);require(audit["payload"]["rejected_count"]>=24,"false-pass audit incomplete")
    for path in root.rglob("*.json"):validate_document(read(path))


def validate_completion(repo,publication=False):
    root=repo/ROOT;before=payload_inventory(root,set());completion=read(root/"completion/y2_r2_r2.json")
    validate_package(repo,completion);rec=validate_derived(repo)
    require(all(completion["payload"][k]==v for k,v in rec.items()),"completion source recomputation drift")
    if publication:validate_publication(repo)
    require(payload_inventory(root,set())==before,"read-only validation modified evidence")
    return {"status":"PASS","valid_clean_count":100,"source_counts":rec["source_counts"],"completion_sha256":sha(root/"completion/y2_r2_r2.json"),"completion_canonical_digest":completion["canonical_digest"],"inventory_sha256":sha(root/"whole_root_inventory.json"),"new_scientific_attempt_count":0,"gpu_execution":False,"next_authorization":NEXT}


def validate_publication(repo):
    head=git(repo,"rev-parse","HEAD");remote={line.split()[1]:line.split()[0] for line in git(repo,"ls-remote","origin").splitlines()}
    require(head==git(repo,"rev-parse","@{upstream}")==remote.get("refs/heads/stage-d-d3-pass")==git(repo,"rev-parse",PASS_REF)==remote.get(PASS_REF),"completion publication refs disagree")
    require(remote.get(PRODUCER)==verify_implementation(repo),"producer remote drift")
    check_refs(repo,True);require(not git(repo,"status","--porcelain"),"publication worktree dirty")
    lfs=git(repo,"lfs","status");require("(LFS:" not in lfs and "(Git:" not in lfs,"publication LFS pending")
    git_binding(repo,ROOT,head)


def seal_failure(repo,phase,error):
    root=repo/ROOT;rows=read_records(root);counts=manifest_counts(rows,complete=False)
    code=getattr(error,"code","BLOCKED_PROVENANCE")
    if code not in artifact_schema()["properties"]["status"]["enum"]:code="BLOCKED_PROVENANCE"
    write(root/"failure.json","failure",{"phase":phase,"failure_type":type(error).__name__,"message":str(error),"validated_unit_count":len(rows),"source_counts":counts,"next_authorization":"NONE"},code)
    report=repo/REPORT;report.parent.mkdir(parents=True,exist_ok=True)
    if not report.exists():
        with report.open("x") as stream:stream.write(f"# Y2-R2-R2 blocked\n\n{code}; validated units retained: {len(rows)}. Phase: {phase}.\n\n{error}\n\nNext authorization: NONE.\n");stream.flush();os.fsync(stream.fileno())
    inv=root/"whole_root_inventory.json"
    if not inv.exists():write(inv,"inventory",payload_inventory(root,{"completion/y2_r2_r2.pending.json","completion/y2_r2_r2.json","whole_root_inventory.json"}),code)
    recpath=root/"reconciliation.json";valpath=root/"source_recomputing_validation_report.json";payloadpath=root/"payload_manifest.json"
    value=signed("completion",{"denominator":100,"valid_clean_count":len(rows),"source_counts":counts,"records_digest":canonical(rows),"new_scientific_attempt_count":0,"gpu_execution":False,"formal_outcome_access":False,"next_authorization":"NONE","workflow_status":code,"reconciliation":record(recpath,repo) if recpath.exists() else None,"validation_report":record(valpath,repo) if valpath.exists() else None,"whole_root_inventory":record(inv,repo),"payload_manifest":record(payloadpath,repo) if payloadpath.exists() else None,"report":record(report,repo),"failure":record(root/"failure.json",repo)},code)
    terminal(root/"completion/y2_r2_r2.pending.json",root/"completion/y2_r2_r2.json",value)
