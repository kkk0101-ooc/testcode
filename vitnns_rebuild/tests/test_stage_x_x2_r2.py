from __future__ import annotations

import copy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from vitnns import stage_x_x2_r2 as r2

REPO=Path(__file__).resolve().parents[1]


def test_r2_contract_exact_coverage_and_closed_schema():
    value=r2.load_config(REPO)
    assert value["authorization"]=="X2_R2_COVERAGE_GATED_NUMERICAL_RECOVERY_ONLY"
    assert value["coverage"]["overall_valid_min"]==190
    assert value["coverage"]["per_model_valid_min"]==38
    changed=copy.deepcopy(value);changed["coverage"]["unknown"]=True
    assert list(Draft202012Validator(r2.strict_read(REPO/r2.CONFIG_SCHEMA)).iter_errors(changed))


def test_r2_artifact_schema_has_no_open_object_nodes():
    schema=r2.strict_read(REPO/r2.ARTIFACT_SCHEMA);bad=[]
    def walk(value,path):
        if isinstance(value,dict):
            if value.get("type")=="object" and value.get("additionalProperties") is not False:bad.append(path)
            for key,item in value.items():walk(item,path+"/"+key)
        elif isinstance(value,list):
            for index,item in enumerate(value):walk(item,path+f"/{index}")
    walk(schema,"root");assert bad==[]


def test_coverage_truth_table_is_exact():
    def gate(total,counts):return total>=190 and all(value>=38 for value in counts)
    assert gate(190,[38,38,38,38,38])
    assert gate(199,[40,40,40,40,39])
    assert not gate(189,[38,38,38,38,37])
    assert not gate(199,[40,40,40,40,37])
    models=[f"m{index}" for index in range(5)]
    rows=[{"unit_digest":r2.hashlib.sha256(f"{model}-{index}".encode()).hexdigest(),"model_id":model,"status":"VALID"} for model in models for index in range(40)]
    assert r2.validate_coverage_states(rows)["pass"]
    rows[0]["status"]="NOT_STARTED"
    with pytest.raises(r2.X2R2Error):r2.validate_coverage_states(rows)


def test_r2_new_identities_are_distinct_from_r1():
    old=r2.r1.load_units(REPO);rows=[]
    for item in old:
        identity={"run_id":r2.RUN_ID,"model_id":item["model_id"],"sample_id":item["sample_id"],"sample_sha256":item["sample_sha256"],"development_rank":item["development_rank"]};rows.append(r2.canonical_sha256(identity))
    assert len(rows)==len(set(rows))==200
    assert not set(rows)&{item["unit_digest"] for item in old}


def test_r1_failure_is_not_relabelled_valid():
    value=r2.r1.strict_read(REPO/r2.R1_STATUS)
    assert value["status"]=="NO_GO_CLASS_LOGIT_COMPLETENESS"
    failed=r2.r1.strict_read(REPO/r2.R1_ROOT/"units/f24e5fafada063b390d95a4aa3a62a645b14f1b34c26a0ed691dc89ba2d2609a/terminal.json")
    assert failed["status"]=="FAILED"
