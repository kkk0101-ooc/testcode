"""Source-recomputing recovery without changing the frozen numerical engine."""
from __future__ import annotations

import hashlib
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from . import stage_y_y2 as y2

PROTOCOL_ID = y2.PROTOCOL_ID
STAGE_ID = "Y2-R2-R1"
AUTHORIZATION = "STAGE_Y2_R2_RESOURCE_REPORTING_RECOVERY_AND_MISSING_UNITS_ONLY"
ROOT = Path("evidence/stage_y/y2_r2_r1_clean_formal_20260912/formal")
REPORT_ROOT = Path("reports/stage_y_y2_r2_r1_20260912")
RUN_SPEC = Path("config/stage_y/y2_r2_r1_run_spec.json")
ARTIFACT_SCHEMA = Path("config/stage_y/y2_r2_r1_artifact.schema.json")
RUN_SCHEMA = Path("config/stage_y/y2_r2_r1_run_spec.schema.json")
AMENDMENT = Path("docs/stage_y_y2_r2_r1_producer_repair_amendment_20260912.md")
Y2_ROOT = y2.ORIGINAL_ROOT
Y2_R1_ROOT = y2.R1_ROOT
Y2_R2_ROOT = Path("evidence/stage_y/y2_r2_clean_formal_20260912/formal")
Y2_COMMIT = "1a621724b32b208036fb5f2a3993a48697e933c8"
Y2_R1_COMMIT = "7cf9b1f9a25695821b19e98cff4e2edbc369891a"
Y2_R2_PRODUCER = "d0023fa67d494e73422fd0e7df767f768a3c942a"
STARTUP_COMMIT = "1fd71d1b3e4d4c10fd3c65374169c734b64b8ff3"
PYTHON = Path("/data01/home/yeziqi.831/nxtrec_wksp/testcode/.conda/envs/vitnns/bin/python")
PRODUCER_REF = "refs/archive/stage-y-y2-r2-r1-resource-reporting-recovery-producer"
COMPLETION_REF = "refs/archive/stage-y-y2-r2-r1-clean-cohort-evidence-completion"
BLOCKED_REF = "refs/archive/stage-y-y2-r2-r1-resource-reporting-recovery-blocked"
MISSING_ORDERS = (77,81,84,85,88,89,90,92,93,94,96,97,98)
COMPLETED_BUDGET_ORDERS = (73,80,86)
SOURCE_TYPES = ("Y2_R1_VALID_REBIND","Y2_R1_COMPLETED_BUDGET_REBIND","Y2_R2_COMPUTED")
COHORT_DIGEST = "156eab4bd86ff9979abc6acf8b06af3aa5a90e90cd02edef7713539b147ce277"
SELECTION_DIGEST = "8b3faa6d437bda38e963985ac12f14f3b440a3e4021915081f1d5e1b1c80d2cf"
CHECKPOINT_SHA = "489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d"
NUMERIC_CONTRACT = "stage-y-y1-grouped-jvp-vjp-frozen-v1"
LADDER = (128,256,512,1024,2048,4096,8192)
CALL_KEYS = ("forward_calls","backward_calls","jvp_calls","jvp_direction_lanes","quadrature_alpha_batches","vjp_calls")
NEXT = "STAGE_Y3_PERTURBATION_MECHANISM_PRODUCTION_ONLY"
PREFIX = "stage_y_y2_r2_r1_"
EXPECTED_REFS = {
    "refs/archive/stage-y-y0-protocol-freeze":"a6efa1426902708d83de1d83ef257d524cacf95b",
    "refs/archive/stage-y-y1-r1-resource-telemetry-waiver":"70f70ca7290316531406791a6533d65590424a24",
    "refs/archive/stage-y-y2-p0-parallel-schedule-freeze":"66a0dfb889fdb0d45d9e566ed422ba9a40bd30a1",
    "refs/archive/stage-y-y2-p0-r1-elastic-schedule-freeze":"e6321a17a416b693f02f12bc0c7c329c21db5307",
    "refs/archive/stage-y-y2-clean-cohort-evidence-blocked":Y2_COMMIT,
    "refs/archive/stage-y-y2-r1-clean-cohort-evidence-blocked":Y2_R1_COMMIT,
    "refs/archive/stage-y-y2-r2-resource-reporting-recovery-producer":Y2_R2_PRODUCER,
    "refs/archive/stage-y-y2-r2-resource-reporting-recovery-blocked":STARTUP_COMMIT,
}
PROTECTED = ((Y2_ROOT,Y2_COMMIT),(y2.ORIGINAL_REPORT_ROOT,Y2_COMMIT),
             (Y2_R1_ROOT,Y2_R1_COMMIT),(y2.R1_REPORT_ROOT,Y2_R1_COMMIT),
             (Y2_R2_ROOT,STARTUP_COMMIT),(Path("reports/stage_y_y2_r2_20260912"),STARTUP_COMMIT))
SOURCE_FILES = (AMENDMENT,RUN_SPEC,RUN_SCHEMA,ARTIFACT_SCHEMA,
    Path("src/vitnns/stage_y_y2_r2.py"),Path("src/vitnns/stage_y_y2_r2_schema.py"),
    Path("src/vitnns/stage_y_y2_r2_worker.py"),Path("scripts/run_stage_y_y2_r2.py"),
    Path("scripts/validate_stage_y_y2_r2.py"),Path("tests/test_stage_y_y2_r2.py"))
DEPENDENCIES = tuple(Path(p) for p in (
    "src/vitnns/stage_y_y2.py","src/vitnns/stage_y_y2_worker.py","src/vitnns/stage_y_y1.py",
    "src/vitnns/stage_y_y1_worker.py","src/vitnns/stage_x_nns_v2.py","src/vitnns/modeling.py",
    "src/vitnns/constants.py","scripts/run_stage_y_y2.py","config/stage_y/y2_run_spec.json",
    "config/stage_y/y2_r1_run_spec.json"))
EXPECTED_ARRAY_NAMES = tuple(sorted(
    ["input_attribution","input_cell_map","post_embedding_matrix","post_embedding_cls","post_embedding_cell","final_norm_matrix","final_norm_cls","final_norm_cell","classifier_input_cls","classifier_input_cell"]
    + [f"b{b:02d}_{s}" for b in range(12) for s in ("first_identity_matrix","first_identity_cls","first_attention_matrix","first_attention_cls","second_residual_matrix","second_residual_cls","second_mlp_matrix","second_mlp_cls","input_matrix","input_cls","output_matrix","output_cls","first_heads_matrix","first_heads_cls","input_cell","first_identity_cell","first_heads_cell","first_attention_cell","second_residual_cell","second_mlp_cell","output_cell")]))

class R2Error(RuntimeError):
    code = "BLOCKED_PROVENANCE"
class StartupError(R2Error):
    code = "BLOCKED_STARTUP_STATE"
class SourceDriftError(R2Error):
    code = "BLOCKED_SOURCE_DRIFT"
class AccessError(R2Error):
    code = "BLOCKED_ACCESS"
class InfrastructureError(R2Error):
    code = "BLOCKED_INFRASTRUCTURE"
class NumericalError(R2Error):
    code = "FAILED_NUMERICAL"

def require(flag: bool, message: str, error_type: type[R2Error] = SourceDriftError) -> None:
    if not flag:
        raise error_type(message)

canonical_digest = y2.canonical_digest
read_json = y2.read_json
sha_file = y2.sha_file

def signed(kind: str, status: str, payload: Mapping[str,Any]) -> dict[str,Any]:
    value = {"$schema":ARTIFACT_SCHEMA.name,"schema_version":1,"document_type":PREFIX+kind,
             "protocol_id":PROTOCOL_ID,"stage_id":STAGE_ID,"status":status,"payload":dict(payload)}
    value["canonical_digest"] = canonical_digest(value)
    validate_signed(value)
    return value

def validate_canonical(value: Mapping[str,Any]) -> None:
    unsigned = dict(value)
    claimed = unsigned.pop("canonical_digest")
    require(claimed == canonical_digest(unsigned), "canonical digest mismatch")

def validate_signed(value: Mapping[str,Any]) -> None:
    from jsonschema import Draft202012Validator
    from .stage_y_y2_r2_schema import artifact_schema
    validate_canonical(value)
    Draft202012Validator(artifact_schema()).validate(value)

def write_json_new(path: Path, value: Mapping[str,Any]) -> None:
    validate_signed(value)
    y2.write_json_new(path,value)

def publish_terminal(pending: Path, terminal: Path, value: Mapping[str,Any]) -> None:
    validate_signed(value)
    y2.publish_terminal(pending,terminal,value)

def file_record(path: Path, repo: Path | None = None) -> dict[str,Any]:
    result = y2.file_record(path)
    if repo is not None:
        result["path"] = path.resolve().relative_to(repo.resolve()).as_posix()
    return result

def checked_record(repo: Path, record: Mapping[str,Any], parent: Path | None = None) -> Path:
    path = repo/str(record["path"])
    require(path.is_file() and not path.is_symlink(),f"missing or symlink source: {path}")
    if parent is not None:
        require(path.resolve().is_relative_to(parent.resolve()),f"source outside owner: {path}")
    require(file_record(path,repo)==record,f"file record drift: {path}")
    return path

def terminal_by_order(repo: Path) -> dict[int,tuple[Path,dict[str,Any]]]:
    result = {}
    for path in sorted((repo/Y2_R1_ROOT).glob("shards/shard-*/units/unit-*/completion/unit.json")):
        value=read_json(path); validate_canonical(value); order=value["payload"]["selection_order"]
        require(order not in result,"duplicate Y2-R1 order")
        require(path == repo/Y2_R1_ROOT/f"shards/shard-{order%4:02d}/units/unit-{order:03d}/completion/unit.json","old shard ownership drift")
        result[order]=(path,value)
    require(sorted(result)==list(range(100)),"Y2-R1 order coverage drift")
    return result

def cohort_by_order(repo: Path) -> dict[int,dict[str,Any]]:
    document=read_json(repo/Y2_ROOT/"cohort_registry.json");validate_canonical(document)
    require(document["canonical_digest"]==COHORT_DIGEST,"cohort digest drift")
    return {row["selection_order"]:dict(row) for row in document["payload"]["records"]}

def expected_shape(name: str) -> tuple[int,...]:
    require(name in EXPECTED_ARRAY_NAMES,"unexpected scientific array name")
    if name=="input_attribution": return (3,224,224)
    if name=="input_cell_map" or name.endswith("_cell"): shape=(56,56)
    elif name.endswith("_cls"): shape=(196,)
    else: shape=(196,196)
    return (12,)+shape if "_first_heads_" in name else shape

def expected_axes(name: str, shape: Sequence[int]) -> list[str]:
    axes={(3,224,224):["rgb","pixel_y","pixel_x"],(56,56):["cell_y","cell_x"],
          (12,56,56):["head","cell_y","cell_x"],(196,196):["source_patch","destination_patch"],
          (12,196,196):["head","source_patch","destination_patch"],(196,):["source_patch"],(12,196):["head","source_patch"]}
    require(tuple(shape) in axes,"unsupported scientific shape")
    return axes[tuple(shape)]

def array_digest(array: np.ndarray) -> str:
    array=np.ascontiguousarray(array);h=hashlib.sha256()
    h.update(str(array.dtype).encode());h.update(b"\0");h.update(str(tuple(array.shape)).encode());h.update(b"\0");h.update(array.tobytes(order="C"))
    return h.hexdigest()

def validate_array_record(repo: Path, record: Mapping[str,Any]) -> np.ndarray:
    path=repo/record["path"];require(path.is_file() and not path.is_symlink(),f"missing array {path}")
    require(path.stat().st_size==record["bytes"] and sha_file(path)==record["sha256"],f"array file digest drift {path}")
    array=np.load(path,allow_pickle=False)
    require(array.dtype==np.dtype("<f4") and record["dtype"]=="float32" and array.shape==expected_shape(path.stem)==tuple(record["shape"]),f"array dtype/shape drift {path}")
    require(record["axes"]==expected_axes(path.stem,array.shape),f"array axes drift {path}")
    require(array.flags.c_contiguous and np.isfinite(array).all(),f"nonfinite/layout array {path}",NumericalError)
    require(array_digest(array)==record["canonical_digest"],f"array canonical digest drift {path}")
    return array

def load_validated_arrays(repo: Path, terminal: Mapping[str,Any], unit_root: Path) -> dict[str,np.ndarray]:
    rows=terminal["payload"]["array_records"]
    require(len(rows)==262 and tuple(sorted(Path(r["path"]).stem for r in rows))==EXPECTED_ARRAY_NAMES,"exact array roster drift")
    paths={repo/r["path"] for r in rows}
    require(paths==set((unit_root/"arrays").glob("*.npy")) and all(p.parent==unit_root/"arrays" for p in paths),"array owner or unbound file drift")
    return {Path(row["path"]).stem:validate_array_record(repo,row) for row in rows}

def validate_attempts(repo: Path, terminal: Mapping[str,Any], unit_root: Path) -> None:
    q=terminal["payload"];documents=[]
    require({repo/r["path"] for r in q["attempt_records"]}==set((unit_root/"attempts").glob("*.json")),"attempt inventory drift")
    for row in q["attempt_records"]:
        path=checked_record(repo,row,unit_root/"attempts");doc=read_json(path);validate_canonical(doc);documents.append(doc)
    require(documents and documents[-1]["payload"]["attempt_id"]==q["attempt_id"],"attempt journal final identity drift")
    attempts=q["quadrature_attempts"]
    require(attempts and tuple(a["steps"] for a in attempts)==LADDER[:len(attempts)],"quadrature ladder drift")
    if terminal["stage_id"]==STAGE_ID:
        require(len(documents)==1,"new scientific attempt count drift");validate_signed(documents[0])
        require(documents[0]["payload"]["quadrature_attempts"]==attempts and documents[0]["payload"]["retry_index"]==1,"new attempt binding drift")
    else:
        require(len(documents)==len(attempts),"historical attempt count drift")
        for i,(doc,attempt) in enumerate(zip(documents,attempts)):
            require(doc["payload"]=={"attempt_id":f"q-{i:03d}","attempt_index":i,**attempt},"historical attempt journal drift")
    for i,attempt in enumerate(attempts):
        threshold=1e-4+0.005*abs(attempt["input_delta"])
        require(attempt["threshold"]==threshold and attempt["input_abs_error"]==abs(attempt["input_sum"]-attempt["input_delta"]),"quadrature threshold/arithmetic drift")
        passed=(attempt["input_abs_error"]<=threshold and attempt["closure_status"]=="PASS" and attempt["identity_max_abs_error"] is not None and attempt["identity_max_abs_error"]<=1e-5)
        require(attempt["all_contracts_pass"]==passed and passed==(i==len(attempts)-1),"not first passing frozen quadrature",NumericalError)

def validate_source_binding(repo: Path, terminal: Mapping[str,Any], order: int) -> None:
    row=cohort_by_order(repo)[order];q=terminal["payload"]
    require(q["selection_order"]==order and q["source_object_id"]==row["source_object_id"],"source/order drift")
    require(q["condition_id"]=="clean" and set(q["target_identity"])=={"class_id","rule"} and q["target_identity"]["rule"]=="clean predicted-class raw logit" and 0<=q["target_identity"]["class_id"]<1000,"target/condition drift")
    require(q["source_binding"]=={"cohort_digest":COHORT_DIGEST,"file_sha256":row["file_sha256"]},"cohort/JPEG binding drift")
    require(q["checkpoint_sha256"]==CHECKPOINT_SHA and q["numeric_contract_digest"]==NUMERIC_CONTRACT,"checkpoint/numeric contract drift")
    base=repo/(ROOT if terminal["stage_id"]==STAGE_ID else Y2_R1_ROOT)
    identity=read_json(base/f"shards/shard-{q['shard_id']:02d}/worker_identity.json");validate_canonical(identity)
    env=identity["payload"]["environment_identity"]
    require(set(env)==y2.ENVIRONMENT_IDENTITY_FIELDS and canonical_digest(env)==q["environment_digest"]==identity["payload"]["environment_digest"],"environment identity drift")
    require(env["python_executable_realpath"]==os.path.realpath(PYTHON) and env["cublas_workspace_config"]==":4096:8" and env["deterministic_algorithms_enabled"] and not env["deterministic_warn_only_enabled"],"deterministic executable environment drift")
    require(env["cudnn_deterministic"] and not any(env[k] for k in ("cudnn_benchmark","cuda_matmul_allow_tf32","cudnn_allow_tf32")),"deterministic flags drift")
    require(env["checkpoint_sha256"]==CHECKPOINT_SHA and env["numeric_contract_digest"]==NUMERIC_CONTRACT,"environment checkpoint drift")

def validate_quadrature_arrays(terminal: Mapping[str,Any], arrays: Mapping[str,np.ndarray]) -> dict[str,Any]:
    from . import stage_y_y1 as y1
    final=terminal["payload"]["quadrature_attempts"][-1];attr=arrays["input_attribution"]
    if final["identity_max_abs_error"]==0:
        require(array_digest(attr)==final["array_digest"],"input identity witness drift")
    require(abs(float(attr.astype(np.float64).sum())-final["input_delta"])<=final["threshold"],"input completeness recomputation failed",NumericalError)
    closure=y1.validate_closure(arrays,float(final["input_delta"]))
    require(closure["status"]=="PASS","recomputed closure failure",NumericalError)
    return closure

def validate_scientific_result(repo: Path, terminal: Mapping[str,Any], unit_root: Path, closure: Mapping[str,Any]) -> Mapping[str,Any]:
    q=terminal["payload"];key="scientific_result" if terminal["stage_id"]==STAGE_ID else "scientific_record"
    path=checked_record(repo,q[key],unit_root);require(path==unit_root/"scientific_result.json","scientific result owner drift")
    doc=read_json(path);validate_canonical(doc);science=doc["payload"]
    from jsonschema import Draft202012Validator
    from .stage_y_y2_r2_schema import science_schema
    Draft202012Validator(science_schema()).validate(science)
    require(science["closure"]==closure and science["delta"]==science["clean_logit"]-science["baseline_logit"]==q["quadrature_attempts"][-1]["input_delta"],"scientific closure/delta drift")
    require(science["target_identity"]==q["target_identity"] and science["selected_steps"]==q["quadrature_attempts"][-1]["steps"] and science["calls"]==q["resource_telemetry"]["calls"],"science target/quadrature/calls drift")
    target=q["target_identity"]["class_id"]
    require(science["ablations"]["status"]==science["deletion_operator"]["status"]=="PASS" and science["ablations"]["target"]==science["deletion_operator"]["target"]==target,"intervention endpoint drift")
    for endpoint in (science["ablations"],science["deletion_operator"]):
        require([b["block"] for b in endpoint["blocks"]]==list(range(12)),"intervention block coverage drift")
    return science

def _source_record(repo: Path, order: int, completed_budget: bool) -> dict[str,Any]:
    path,t=terminal_by_order(repo)[order];q=t["payload"];unit_root=path.parent.parent
    require(not list(unit_root.rglob("*.pending.json")) and not list(unit_root.rglob("*.tmp")),"partial historical unit")
    if completed_budget:
        require(order in COMPLETED_BUDGET_ORDERS and t["status"]=="BLOCKED_BUDGET" and q["failure_code"]=="per-shard resource cap exceeded by completed unit" and q["scientific_record"] is None,"completed-budget identity drift")
    else: require(t["status"]=="VALID" and q["failure_code"] is None,"source not VALID")
    validate_source_binding(repo,t,order);validate_attempts(repo,t,unit_root);arrays=load_validated_arrays(repo,t,unit_root);closure=validate_quadrature_arrays(t,arrays)
    if completed_budget:
        steps=q["quadrature_attempts"][-1]["steps"]
        expected={512:dict(zip(CALL_KEYS,(7359,352,6272,25088,128,3584))),1024:dict(zip(CALL_KEYS,(14015,736,12544,50176,256,7168)))}
        require(steps in expected and q["resource_telemetry"]["calls"]==expected[steps],"completed intervention execution accounting drift")
    else: validate_scientific_result(repo,t,unit_root,closure)
    kind=SOURCE_TYPES[int(completed_budget)]
    return {"selection_order":order,"source_object_id":q["source_object_id"],"class_id":cohort_by_order(repo)[order]["class_id"],
            "source_type":kind,"status":"VALID","source_terminal":file_record(path,repo),"source_terminal_status":t["status"],
            "source_scientific_result":q["scientific_record"],"array_count":262,"array_records_digest":canonical_digest(q["array_records"]),
            "attempt_records_digest":canonical_digest(q["attempt_records"]),"validation_digest":canonical_digest({"closure":closure,"target":q["target_identity"],"arrays":q["array_records"],"attempts":q["attempt_records"]}),
            "resource_telemetry":q["resource_telemetry"],"arrays_copied":False,"gpu_reexecuted":False,
            "intervention_payload_retained":not completed_budget}

def validate_valid_source(repo: Path, order: int) -> dict[str,Any]:
    return _source_record(repo,order,False)

def validate_completed_budget_source(repo: Path, order: int) -> dict[str,Any]:
    return _source_record(repo,order,True)

def validate_missing_sources(repo: Path) -> list[dict[str,Any]]:
    terms=terminal_by_order(repo)
    observed=tuple(sorted(o for o,(_,d) in terms.items() if d["status"]=="BLOCKED_BUDGET" and d["payload"]["failure_code"]=="per-shard resource cap already exhausted"))
    require(observed==MISSING_ORDERS,"missing roster drift")
    result=[]
    for order in observed:
        path,t=terms[order];q=t["payload"];unit=path.parent.parent
        require(q["scientific_record"] is None and not (unit/"scientific_result.json").exists() and not list(unit.rglob("*.npy")) and q["array_records"]==[] and q["quadrature_attempts"]==[],"missing unit contains scientific work")
        require(q["resource_telemetry"]["calls"]==dict.fromkeys(CALL_KEYS,0),"missing unit executed model calls")
        result.append({"selection_order":order,"source_object_id":q["source_object_id"],"source_terminal":file_record(path,repo)})
    return result

def shard_orders(shard: int, count: int) -> list[int]:
    require(1<=count<=8 and 0<=shard<count,"invalid locked shard count")
    return [order for i,order in enumerate(MISSING_ORDERS) if i%count==shard]

def validate_implementation(repo: Path) -> str:
    seal=read_json(repo/ROOT/"producer_implementation_seal.json");validate_signed(seal);q=seal["payload"]
    require(y2.git(repo,"rev-parse",PRODUCER_REF)==q["producer_commit"],"producer ref drift")
    observed=[file_record(repo/p,repo) for p in SOURCE_FILES+DEPENDENCIES]
    require(observed==q["source_files"] and canonical_digest({"files":observed})==q["files_digest"],"implementation seal drift")
    for row in observed:
        data=subprocess.check_output(["git","show",f"{q['producer_commit']}:{repo.name}/{row['path']}"],cwd=repo)
        require(hashlib.sha256(data).hexdigest()==row["sha256"],"producer Git source drift")
    return q["producer_commit"]

def resource_sum(rows: Sequence[Mapping[str,Any]], *, retained_bytes: int | None = None) -> dict[str,Any]:
    for row in rows:
        for key in ("gpu_active_seconds","cpu_seconds","wall_seconds","peak_reserved_gpu_bytes","retained_bytes","download_bytes"):
            value=row.get(key,0);require(math.isfinite(value) and value>=0,"invalid resource measurement")
    retained=sum(int(r.get("retained_bytes",0)) for r in rows) if retained_bytes is None else retained_bytes
    return {"h20_gpu_hours":sum(float(r.get("gpu_active_seconds",0)) for r in rows)/3600,
            "cpu_hours":sum(float(r.get("cpu_seconds",0)) for r in rows)/3600,
            "wall_seconds_sum":sum(float(r.get("wall_seconds",0)) for r in rows),
            "retained_bytes":retained,"retained_storage_gib":retained/1024**3,
            "peak_memory_bytes":max([int(r.get("peak_reserved_gpu_bytes",0)) for r in rows]+[0]),
            "download_bytes":sum(int(r.get("download_bytes",0)) for r in rows)}

def whole_root_inventory(root: Path, status: str) -> dict[str,Any]:
    excluded={"whole_root_inventory.json","completion/y2_r2_r1.pending.json","completion/y2_r2_r1.json"}
    rows=[{"path":p.relative_to(root).as_posix(),"bytes":p.stat().st_size,"sha256":sha_file(p)} for p in sorted(root.rglob("*")) if p.is_file() and p.relative_to(root).as_posix() not in excluded]
    return signed("inventory",status,{"root":ROOT.as_posix(),"excluded":sorted(excluded),"files":rows,"file_count":len(rows),"total_bytes":sum(r["bytes"] for r in rows),"files_digest":canonical_digest({"files":rows})})
