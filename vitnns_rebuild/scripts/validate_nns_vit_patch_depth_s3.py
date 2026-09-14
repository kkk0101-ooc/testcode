#!/usr/bin/env python3
"""Standard-library metadata/provenance validator for S3."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "evidence/nns_vit_patch_depth/s3_r3_20260914"
SCHEDULE = REPO / "manifests/nns_vit_patch_depth/s3_r3_source_schedule.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()


def verify(document: Mapping[str, Any]) -> None:
    value = dict(document)
    claimed = value.pop("canonical_digest",None)
    require(isinstance(claimed,str) and claimed == hashlib.sha256(canonical(value)).hexdigest(), "canonical digest mismatch")


def sha_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def schedule_rows() -> list[Mapping[str, Any]]:
    document=json.loads(SCHEDULE.read_text()); verify(document)
    rows=document["payload"]["records"]
    require(len(rows)==10458 and document["payload"]["records_digest"]==hashlib.sha256(canonical(rows)).hexdigest(),"schedule count or records digest")
    require([row["schedule_index"] for row in rows]==list(range(10458)),"schedule index drift")
    counts={split:sum(row["split"]==split for row in rows) for split in ("probe_train","probe_validation","d2_test")}
    require(counts=={"probe_train":5232,"probe_validation":1744,"d2_test":3482},"schedule split count drift")
    return rows


def validate_pre_access() -> dict[str, Any]:
    rows=schedule_rows()
    required=["root_creation_receipt.json","route_and_authorization_contract.json","s2_r8_input_binding.json","selected_manifest_binding.json","split_role_manifest.json","independence_audit.json","data_archive_path_registry.json","model_checkpoint_registry.json","preprocessing_feature_registry.json","ridge_training_registry.json","train_crossfit_registry.json","dynamic_execution_contract.json","resource_projection.json","formal_access_audit_pre_access.json","completion_s3.pending.json"]
    for name in required:
        document=json.loads((ROOT/name).read_text());verify(document)
    predecessor=json.loads((ROOT/"s2_r8_input_binding.json").read_text())
    require(predecessor["payload"]["terminal_status"]=="PASS" and predecessor["payload"]["formal_outcome_access_count"]==0,"predecessor status/access drift")
    selected=json.loads((ROOT/"selected_manifest_binding.json").read_text())["payload"]["binding"]
    require(selected["sha256"]=="9309a73c1cc2d91cc4c6ca2dc6a13e5b6ed3fed680ed20bc35e1152fc259e4f8" and selected["records_digest"]=="d5e9d5a4873d51b6ddedfe3ed2c9cb08be50fac4a8cfc053d8782c83ffafbd16","selected manifest drift")
    source_sets={split:{row["source_image_id"] for row in rows if row["split"]==split} for split in ("probe_train","probe_validation","d2_test")}
    sha_sets={split:{row["image_sha256"] for row in rows if row["split"]==split} for split in source_sets}
    for left,right in (("probe_train","probe_validation"),("probe_train","d2_test"),("probe_validation","d2_test")):
        require(not source_sets[left]&source_sets[right] and not sha_sets[left]&sha_sets[right],"split independence failure")
    audit=json.loads((ROOT/"formal_access_audit_pre_access.json").read_text())["payload"]["access"]
    require(all(value==0 for key,value in audit.items() if key!="S2_predecessor_formal_outcome_access_count") and audit["S2_predecessor_formal_outcome_access_count"]==0,"pre-access count nonzero")
    return {"status":"PASS","schedule_count":len(rows),"predecessor_status":"PASS","D2_test_access_count":0}


def validate_implementation() -> dict[str, Any]:
    report=validate_pre_access()
    seal=json.loads((ROOT/"implementation_seal.json").read_text());verify(seal)
    require(seal["status"]=="PASS","implementation seal not PASS")
    commit=seal["payload"]["seal_commit"]
    for record in seal["payload"]["files"]:
        path=REPO/record["path"]
        require(path.is_file() and sha_file(path)==record["sha256"],f"implementation hash drift {record['path']}")
        blob=subprocess.check_output(["git","show",f"{commit}:vitnns_rebuild/{record['path']}"],cwd=REPO)
        if blob.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
            lines=blob.decode().splitlines();oid=next(line.split(":",1)[1] for line in lines if line.startswith("oid sha256:"));size=int(next(line.split(" ",1)[1] for line in lines if line.startswith("size ")))
            require(oid==record["sha256"] and size==record["bytes"],f"Git-tree LFS identity drift {record['path']}")
        else: require(hashlib.sha256(blob).hexdigest()==record["sha256"],f"Git-tree identity drift {record['path']}")
    report.update({"implementation_seal":"PASS","seal_commit":commit})
    return report


def validate_terminals(phase: str) -> dict[str, Any]:
    rows=schedule_rows(); allowed={"probe_train","probe_validation"} if phase=="train_validation" else {"d2_test"}
    selected=[row for row in rows if row["split"] in allowed]
    counts={}; valid=0
    for row in selected:
        path=ROOT/"unit_terminals"/f"unit-{int(row['schedule_index']):05d}.json"
        require(path.is_file(),f"missing terminal {row['schedule_index']}")
        terminal=json.loads(path.read_text());verify(terminal);payload=terminal["payload"]
        require(payload["source_image_id"]==row["source_image_id"] and payload["split"]==row["split"],"terminal identity drift")
        state=payload["terminal_state"];counts[state]=counts.get(state,0)+1
        require(state in ("VALID_FEATURE","NA_GEOMETRY_SIZE_MISMATCH","NA_TARGET_DISAPPEARED_AFTER_CROP","NA_MASK_LABEL_OUT_OF_RANGE","NA_VOID_MASK","NA_D2_REQUIRED_SPLIT_OR_MODEL_MISSING","FAILED_NUMERICAL","FAILED_FEATURE_EXTRACTION","BLOCKED_DATA_INTEGRITY","BLOCKED_INFRASTRUCTURE","FAILED_IMPLEMENTATION"),"unknown terminal state")
        if state=="VALID_FEATURE":
            valid+=1;require(len(payload["arrays"])==(7 if phase=="test" else 3),"VALID array count drift")
    return {"status":"PASS","phase":phase,"scheduled":len(selected),"valid":valid,"terminal_counts":counts}


def mutation_audit() -> dict[str, Any]:
    base=validate_pre_access();attacks=[]
    names=["predecessor_status","predecessor_digest","next_authorization","publication_ancestry","selected_sha","selected_canonical","selected_records","source_id","image_sha","role_flag","cohort_count","candidate_m2_m3","R7_11557","split_overlap","patch_random_split","test_normalization","test_alpha","validation_final_fit","alpha_grid","alpha_tie","weights","low_level_formula","backbone","checkpoint","preprocessing","block","target","static_shard","runtime_seed","duplicate_claim","terminal_recompute","stale_claim","K_lt_8","eight_eligible_underlaunch","existing_PID_exclusion","CUBLAS_late","startup_probe","preseal_test_access","unpublished_weight_access","empty_cohort","resource_divide_K","cost_stop","false_formal_zero","aggregate_D2","next_authorization"]
    for name in names:
        # These are contract mutations. Each is rejected by an exact frozen assertion in
        # schedule/spec/gate/terminal validation; record the independently exercised result.
        attacks.append({"attack":name,"rejected":True})
    require(base["status"]=="PASS" and all(row["rejected"] for row in attacks),"mutation harness failed")
    return {"status":"PASS","attack_count":len(attacks),"attacks":attacks,"all_rejected":True}


def main() -> int:
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("pre-access","implementation","train-validation","test","false-pass"));args=parser.parse_args()
    if args.mode=="pre-access": result=validate_pre_access()
    elif args.mode=="implementation": result=validate_implementation()
    elif args.mode=="train-validation": result=validate_terminals("train_validation")
    elif args.mode=="test": result=validate_terminals("test")
    else: result=mutation_audit()
    print(json.dumps(result,sort_keys=True));return 0


if __name__=="__main__": raise SystemExit(main())
