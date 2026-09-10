from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from vitnns import stage_x_x2_r1 as x2r1
from vitnns import stage_x_x2_failure as failure
from vitnns import stage_x_x2_r1_reconcile as reconcile
from vitnns import stage_x_x2_r1_worker as worker
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]


def test_recovery_config_is_closed_bound_and_self_digesting():
    value=x2r1.load_config(REPO)
    assert value["authorization"]=="X2_R1_NNS_V2_DEVELOPMENT_AND_FUTILITY_RECOVERY_ONLY"
    assert value["terminal"]["pass_next_authorization"]=="NONE_REQUIRES_OWNER_AUTHORIZATION"
    changed=copy.deepcopy(value);changed["roots"]["unknown"]=True
    errors=list(Draft202012Validator(x2r1.strict_read(REPO/x2r1.CONFIG_SCHEMA)).iter_errors(changed))
    assert errors


def test_failed_x2_receipt_source_recomputes_without_unit_terminals():
    observed=failure.read(REPO/failure.RECEIPT);failure.validate(observed,REPO)
    expected=failure.recompute(REPO);expected["canonical_digest"]=x2r1.canonical_sha256(expected)
    assert observed==expected
    assert observed["counts"]=={
        "development_images":40,"expected_routes":200,
        "complete_diagnostic_routes":11,"interrupted_attempts":4,
        "not_started_total":189,"never_started":185,
        "published_unit_terminals":0,"existing_file_count_before_receipt":331,
        "existing_bytes_before_receipt":62171593,
    }


def test_pre_outcome_mutations_are_all_rejected_and_bound_into_run_spec():
    receipt=x2r1.strict_read(REPO/x2r1.MUTATION_RECEIPT);x2r1.validate_artifact(receipt,REPO,verify_files=True)
    checks=receipt["payload"]["checks"]
    assert len(checks)==20 and all(row["status"]=="PASS" for row in checks)
    run_spec=x2r1.strict_read(REPO/x2r1.RUN_SPEC);x2r1.validate_artifact(run_spec,REPO,verify_files=True)
    state={row["name"]:row["value"] for row in run_spec["payload"]["state"]}
    assert state["pre_outcome_mutation_receipt_sha256"]==x2r1.sha_file(REPO/x2r1.MUTATION_RECEIPT)


def test_x2_r1_uses_exact_old_roster_but_new_unit_identities():
    samples=x2r1.load_samples(REPO);units=x2r1.load_units(REPO)
    old=x2r1.strict_read(REPO/x2r1.OLD_SAMPLE_MANIFEST)["payload"]["samples"]
    assert [row["source_object_id"] for row in samples]==[row["source_object_id"] for row in old]
    assert len(samples)==40 and len(units)==len({row["unit_digest"] for row in units})==200
    assert set(row["run_id"] for row in units)=={x2r1.RUN_ID}
    assert set(row["shard"] for row in units)=={0,1,2,3}


def test_artifact_payload_and_nested_objects_are_closed():
    base=x2r1.signed("stage_x_x2_r1_phase_manifest","PASS",x2r1.payload())
    x2r1.validate_artifact(base,REPO)
    for mutate in (
        lambda value:value.__setitem__("unknown",True),
        lambda value:value["payload"].__setitem__("unknown",True),
        lambda value:value["payload"]["resources"].__setitem__("unknown",True),
    ):
        changed=copy.deepcopy(base);mutate(changed);changed["canonical_digest"]=x2r1.canonical_sha256({key:item for key,item in changed.items() if key!="canonical_digest"})
        with pytest.raises(x2r1.X2R1Error):x2r1.validate_artifact(changed,REPO)


def test_strict_json_duplicate_and_nonfinite_rejection():
    with pytest.raises(x2r1.X2R1Error):x2r1.strict_loads('{"x":1,"x":2}')
    with pytest.raises(x2r1.X2R1Error):x2r1.strict_loads('{"x":NaN}')
    with pytest.raises(x2r1.X2R1Error):x2r1.strict_loads('{"x":Infinity}')


def test_pending_and_terminal_overwrite_are_rejected(tmp_path: Path):
    value=x2r1.signed("stage_x_x2_r1_phase_manifest","PASS",x2r1.payload())
    pending=tmp_path/"pending.json";terminal=tmp_path/"terminal.json"
    x2r1.publish_atomic(pending,terminal,value,REPO)
    assert terminal.is_file() and not pending.exists()
    with pytest.raises(x2r1.X2R1Error):x2r1.publish_atomic(pending,terminal,value,REPO)


def test_holm_family_rejects_identity_control_or_wrong_count():
    for state in ({"holm_family_count":5,"full_model_ig_in_holm":True},{"holm_family_count":6,"full_model_ig_in_holm":False}):
        value=x2r1.signed("stage_x_x2_r1_run_spec","PASS",x2r1.payload(state=state))
        with pytest.raises(x2r1.X2R1Error):x2r1.validate_artifact(value,REPO)


def test_margin_and_hierarchical_bootstrap_are_deterministic():
    median,mad,margin=reconcile._margin([-1.0,0.0,1.0],0.2,0.01,0.05)
    assert median==0.0 and mad==1.0 and margin==0.05
    rows=[{"model_id":"a","difference":str(value)} for value in (0.1,0.2)]+[{"model_id":"b","difference":str(value)} for value in (0.3,0.4)]
    assert reconcile._hierarchical_ci(rows,"test",100)==reconcile._hierarchical_ci(rows,"test",100)


def test_positive_and_negative_support_deletion_are_typed_and_deterministic():
    image=torch.ones((1,3,4,4));score=np.arange(16,dtype=np.float32).reshape(4,4)-8
    positive=worker._delete_support(image,score,positive=True);negative=worker._delete_support(image,score,positive=False)
    assert torch.equal(positive,worker._delete_support(image,score,positive=True))
    assert torch.equal(negative,worker._delete_support(image,score,positive=False))
    assert int((positive==0).all(1).sum())>0 and int((negative==0).all(1).sum())>0
