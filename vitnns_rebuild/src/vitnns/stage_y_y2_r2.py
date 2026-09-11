"""Stage Y2-R2 evidence primitives and source-recomputing validation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import stage_y_y2 as y2

PROTOCOL_ID = y2.PROTOCOL_ID
STAGE_ID = "Y2-R2"
AUTHORIZATION = "STAGE_Y2_R2_RESOURCE_REPORTING_RECOVERY_AND_MISSING_UNITS_ONLY"
ROOT = Path("evidence/stage_y/y2_r2_clean_formal_20260912/formal")
REPORT_ROOT = Path("reports/stage_y_y2_r2_20260912")
RUN_SPEC = Path("config/stage_y/y2_r2_run_spec.json")
Y2_ROOT = y2.ORIGINAL_ROOT
Y2_R1_ROOT = y2.R1_ROOT
Y2_COMMIT = "1a621724b32b208036fb5f2a3993a48697e933c8"
Y2_R1_COMMIT = "7cf9b1f9a25695821b19e98cff4e2edbc369891a"
PRODUCER_REF = "refs/archive/stage-y-y2-r2-resource-reporting-recovery-producer"
COMPLETION_REF = "refs/archive/stage-y-y2-r2-clean-cohort-evidence-completion"
MISSING_ORDERS = (77,81,84,85,88,89,90,92,93,94,96,97,98)
COMPLETED_BUDGET_ORDERS = (73,80,86)
SOURCE_TYPES = ("Y2_R1_VALID_REBIND","Y2_R1_COMPLETED_BUDGET_REBIND","Y2_R2_COMPUTED")
EXPECTED_ARRAY_NAMES = tuple(sorted(
    ["input_attribution","input_cell_map","post_embedding_matrix","post_embedding_cls","post_embedding_cell","final_norm_matrix","final_norm_cls","final_norm_cell","classifier_input_cls","classifier_input_cell"]
    + [f"b{block:02d}_{suffix}" for block in range(12) for suffix in ("first_identity_matrix","first_identity_cls","first_attention_matrix","first_attention_cls","second_residual_matrix","second_residual_cls","second_mlp_matrix","second_mlp_cls","input_matrix","input_cls","output_matrix","output_cls","first_heads_matrix","first_heads_cls","input_cell","first_identity_cell","first_heads_cell","first_attention_cell","second_residual_cell","second_mlp_cell","output_cell")]
))


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


def canonical_digest(value: Any) -> str:
    return y2.canonical_digest(value)


def read_json(path: Path) -> Any:
    return y2.read_json(path)


def sha_file(path: Path) -> str:
    return y2.sha_file(path)


def signed(document_type: str, status: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    value = {"$schema":"stage_y_y2_r2_artifact.schema.json","schema_version":1,"document_type":document_type,"protocol_id":PROTOCOL_ID,"stage_id":STAGE_ID,"status":status,"payload":dict(payload)}
    value["canonical_digest"] = canonical_digest(value)
    return value


def validate_signed(value: Mapping[str, Any]) -> None:
    require(set(value) == {"$schema","schema_version","document_type","protocol_id","stage_id","status","payload","canonical_digest"}, "R2 envelope is not closed")
    unsigned=dict(value); claimed=unsigned.pop("canonical_digest")
    require(claimed == canonical_digest(unsigned), "R2 canonical digest mismatch")
    require(value["$schema"] == "stage_y_y2_r2_artifact.schema.json" and value["stage_id"] == STAGE_ID and value["protocol_id"] == PROTOCOL_ID, "R2 envelope identity drift")


def write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    y2.write_json_new(path, value)


def publish_terminal(pending: Path, terminal: Path, value: Mapping[str, Any]) -> None:
    y2.publish_terminal(pending, terminal, value)


def file_record(path: Path, repo: Path | None = None) -> dict[str, Any]:
    result=y2.file_record(path)
    if repo is not None:
        result["path"] = path.resolve().relative_to(repo.resolve()).as_posix()
    return result


def terminal_by_order(repo: Path) -> dict[int, tuple[Path, dict[str, Any]]]:
    result={}
    for path in sorted((repo/Y2_R1_ROOT).glob("shards/shard-*/units/unit-*/completion/unit.json")):
        value=read_json(path); order=int(value["payload"]["selection_order"]); require(order not in result, "duplicate Y2-R1 order")
        result[order]=(path,value)
    require(sorted(result)==list(range(100)), "Y2-R1 order coverage drift")
    return result


def cohort_by_order(repo: Path) -> dict[int, dict[str, Any]]:
    cohort=read_json(repo/Y2_ROOT/"cohort_registry.json")
    return {int(row["selection_order"]):dict(row) for row in cohort["payload"]["records"]}


def expected_axes(name: str, shape: Sequence[int]) -> list[str]:
    if tuple(shape)==(3,224,224): return ["rgb","pixel_y","pixel_x"]
    if tuple(shape)==(56,56): return ["cell_y","cell_x"]
    if tuple(shape)==(12,56,56): return ["head","cell_y","cell_x"]
    if tuple(shape)==(196,196): return ["source_patch","destination_patch"]
    if tuple(shape)==(12,196,196): return ["head","source_patch","destination_patch"]
    if tuple(shape)==(196,): return ["source_patch"]
    if tuple(shape)==(12,196): return ["head","source_patch"]
    return [f"axis_{index}" for index in range(len(shape))]


def array_digest(array: np.ndarray) -> str:
    value=np.ascontiguousarray(array)
    digest=hashlib.sha256();digest.update(str(value.dtype).encode("utf-8"));digest.update(b"\0");digest.update(str(tuple(value.shape)).encode("utf-8"));digest.update(b"\0");digest.update(value.tobytes(order="C"));return digest.hexdigest()


def validate_array_record(repo: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    path=repo/str(record["path"])
    require(path.is_file(), f"array missing: {path}")
    require(path.stat().st_size==record["bytes"] and sha_file(path)==record["sha256"], f"array file digest drift: {path}")
    array=np.load(path,allow_pickle=False)
    require(str(array.dtype)==record["dtype"] and list(array.shape)==record["shape"], f"array dtype/shape drift: {path}")
    require(list(record["axes"])==expected_axes(path.stem,array.shape), f"array axes drift: {path}")
    require(array.flags.c_contiguous and np.isfinite(array).all(), f"array finite/layout drift: {path}", NumericalError)
    require(array_digest(array)==record["canonical_digest"], f"array canonical digest drift: {path}")
    return {"path":record["path"],"bytes":record["bytes"],"sha256":record["sha256"],"canonical_digest":record["canonical_digest"],"dtype":record["dtype"],"shape":record["shape"],"axes":record["axes"]}


def load_validated_arrays(repo: Path, terminal: Mapping[str, Any]) -> dict[str,np.ndarray]:
    records=terminal["payload"]["array_records"]
    require(len(records)==262, "array roster must contain 262 records")
    require(tuple(sorted(Path(str(row["path"])).stem for row in records))==EXPECTED_ARRAY_NAMES, "exact expected array set drift")
    arrays={}
    for row in records:
        validate_array_record(repo,row); path=repo/str(row["path"]); arrays[path.stem]=np.load(path,allow_pickle=False)
    return arrays


def validate_attempts(repo: Path, terminal: Mapping[str,Any]) -> None:
    payload=terminal["payload"]; observed=[]
    for row in payload["attempt_records"]:
        path=repo/str(row["path"]); require(file_record(path,repo)==row, "attempt record drift"); document=read_json(path)
        if document.get("stage_id")==STAGE_ID: validate_signed(document)
        else: y2.validate_signed(document,allow_historical_y2=True)
        observed.append(document["payload"])
    require(observed and observed[-1]["attempt_id"]==payload["attempt_id"], "attempt journal final identity drift")
    require(payload["quadrature_attempts"] and payload["quadrature_attempts"][-1]["all_contracts_pass"] is True, "quadrature did not finish on a passing attempt", NumericalError)
    require(all(row["all_contracts_pass"] is False for row in payload["quadrature_attempts"][:-1]), "not first passing quadrature")


def validate_source_binding(repo: Path, terminal: Mapping[str,Any], order: int) -> None:
    cohort=cohort_by_order(repo)[order]; payload=terminal["payload"]
    require(payload["selection_order"]==order and payload["source_object_id"]==cohort["source_object_id"], "source/order drift")
    require(payload["condition_id"]=="clean" and payload["target_identity"]["rule"]=="clean predicted-class raw logit", "target/condition drift")
    require(payload["source_binding"]=={"cohort_digest":"156eab4bd86ff9979abc6acf8b06af3aa5a90e90cd02edef7713539b147ce277","file_sha256":cohort["file_sha256"]}, "cohort/JPEG binding drift")
    require(payload["checkpoint_sha256"]=="489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d" and payload["numeric_contract_digest"]=="stage-y-y1-grouped-jvp-vjp-frozen-v1", "checkpoint/numeric contract drift")
    jpeg=y2.TAR_PATH
    # The original registry file hash was computed from the tar member bytes;
    # full release recomputation is performed separately before rebinding.
    require(len(cohort["file_sha256"])==64 and jpeg==Path("assets/imagenet/ILSVRC2012_img_val.tar"), "JPEG identity format drift")


def validate_valid_source(repo: Path, order: int) -> dict[str,Any]:
    from . import stage_y_y1 as y1
    path,terminal=terminal_by_order(repo)[order]; require(terminal["status"]=="VALID", "source unit is not VALID")
    validate_source_binding(repo,terminal,order); validate_attempts(repo,terminal); arrays=load_validated_arrays(repo,terminal)
    scientific_record=terminal["payload"]["scientific_record"]; require(isinstance(scientific_record,Mapping),"VALID source missing scientific result")
    scientific_path=repo/str(scientific_record["path"]); require(file_record(scientific_path,repo)==scientific_record,"scientific result file drift")
    scientific=read_json(scientific_path); y2.validate_signed(scientific,allow_historical_y2=True)
    result=scientific["payload"]; closure=y1.validate_closure(arrays,float(result["delta"]))
    require(closure["status"]=="PASS" and closure==result["closure"],"source closure recomputation drift",NumericalError)
    require(result["target_identity"]==terminal["payload"]["target_identity"] and result["selected_steps"]==terminal["payload"]["quadrature_attempts"][-1]["steps"],"source identity/quadrature drift")
    require(result["ablations"]["status"]=="PASS" and result["deletion_operator"]["status"]=="PASS","intervention endpoint drift",NumericalError)
    return {"selection_order":order,"source_object_id":terminal["payload"]["source_object_id"],"source_type":"Y2_R1_VALID_REBIND","status":"VALID","source_terminal":file_record(path,repo),"source_scientific_result":scientific_record,"array_count":262,"array_records_digest":canonical_digest(terminal["payload"]["array_records"]),"attempt_records_digest":canonical_digest(terminal["payload"]["attempt_records"]),"validation_digest":canonical_digest({"closure":closure,"target":result["target_identity"],"arrays":terminal["payload"]["array_records"]}),"resource_telemetry":terminal["payload"]["resource_telemetry"]}


def validate_completed_budget_source(repo: Path, order: int) -> dict[str,Any]:
    from . import stage_y_y1 as y1
    path,terminal=terminal_by_order(repo)[order]; payload=terminal["payload"]
    require(terminal["status"]=="BLOCKED_BUDGET" and payload["failure_code"]=="per-shard resource cap exceeded by completed unit","completed-budget source identity drift")
    require(payload["scientific_record"] is None and len(payload["array_records"])==262,"completed-budget source completeness drift")
    validate_source_binding(repo,terminal,order); validate_attempts(repo,terminal); arrays=load_validated_arrays(repo,terminal)
    final=payload["quadrature_attempts"][-1]; closure=y1.validate_closure(arrays,float(final["input_delta"]))
    require(closure["status"]=="PASS" and final["closure_status"]=="PASS" and final["identity_max_abs_error"]<=1e-5 and closure["input_abs_error"]<=closure["input_threshold"],"completed-budget scientific validation failed",NumericalError)
    calls=payload["resource_telemetry"]["calls"]; require(all(int(calls[k])>0 for k in ("forward_calls","backward_calls","jvp_calls","vjp_calls")),"completed-budget intervention/integration call coverage absent")
    selected=int(final["steps"]); expected={512:{"forward_calls":7359,"backward_calls":352,"jvp_calls":6272,"jvp_direction_lanes":25088,"quadrature_alpha_batches":128,"vjp_calls":3584},1024:{"forward_calls":14015,"backward_calls":736,"jvp_calls":12544,"jvp_direction_lanes":50176,"quadrature_alpha_batches":256,"vjp_calls":7168}}
    require(selected in expected and calls==expected[selected],"completed-budget call accounting does not prove full integration/intervention endpoint execution")
    return {"selection_order":order,"source_object_id":payload["source_object_id"],"source_type":"Y2_R1_COMPLETED_BUDGET_REBIND","status":"VALID","source_terminal":file_record(path,repo),"source_scientific_result":None,"array_count":262,"array_records_digest":canonical_digest(payload["array_records"]),"attempt_records_digest":canonical_digest(payload["attempt_records"]),"validation_digest":canonical_digest({"closure":closure,"target":payload["target_identity"],"arrays":payload["array_records"],"calls":calls}),"resource_telemetry":payload["resource_telemetry"]}


def validate_missing_sources(repo: Path) -> list[dict[str,Any]]:
    terminals=terminal_by_order(repo); result=[]
    observed=[]
    for order,(path,terminal) in terminals.items():
        payload=terminal["payload"]
        if terminal["status"]=="BLOCKED_BUDGET" and payload["failure_code"]=="per-shard resource cap already exhausted": observed.append(order)
    require(tuple(sorted(observed))==MISSING_ORDERS,"missing-unit roster drift")
    for order in MISSING_ORDERS:
        path,terminal=terminals[order]; payload=terminal["payload"]
        require(payload["scientific_record"] is None and payload["array_records"]==[] and payload["quadrature_attempts"]==[],"missing unit contains scientific result/arrays/attempts")
        require(payload["resource_telemetry"]["calls"]=={"forward_calls":0,"backward_calls":0,"jvp_calls":0,"jvp_direction_lanes":0,"quadrature_alpha_batches":0,"vjp_calls":0},"missing unit executed model calls")
        result.append({"selection_order":order,"source_object_id":payload["source_object_id"],"source_type":"Y2_R2_COMPUTED","status":"PENDING","source_terminal":file_record(path,repo),"source_scientific_result":None,"array_count":0,"array_records_digest":canonical_digest([]),"attempt_records_digest":canonical_digest(payload["attempt_records"]),"validation_digest":canonical_digest({"missing":True,"terminal":terminal["canonical_digest"]}),"resource_telemetry":payload["resource_telemetry"]})
    return result


def whole_root_inventory(root: Path,status: str) -> dict[str,Any]:
    excluded={"whole_root_inventory.json","completion/y2_r2.pending.json","completion/y2_r2.json"}; rows=[]
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.relative_to(root).as_posix() not in excluded:
            rows.append({"path":path.relative_to(root).as_posix(),"bytes":path.stat().st_size,"sha256":sha_file(path)})
    return signed("stage_y_y2_r2_whole_root_inventory",status,{"artifact_id":"whole_root_inventory","root":ROOT.as_posix(),"excluded":sorted(excluded),"files":rows,"file_count":len(rows),"total_bytes":sum(x["bytes"] for x in rows),"files_digest":canonical_digest({"files":rows})})
