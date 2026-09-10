from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
from jsonschema import Draft202012Validator

from vitnns import stage_x_x2_r2_r1 as r1
from vitnns import stage_x_x2_metrics as metrics

REPO=Path(__file__).resolve().parents[1]


def test_r2_r1_contract_exact_control_recovery():
    value=r1.load_config(REPO)
    assert value["authorization"]=="X2_R2_R1_CONTROL_INITIALIZATION_RECOVERY_ONLY"
    assert value["roots"]["formal"].endswith("x2_r2_r1_nns_v2_development_20260909/formal")
    assert value["coverage"]["overall_valid_min"]==190
    assert value["coverage"]["per_model_valid_min"]==38
    changed=copy.deepcopy(value);changed["downstream"]["x3_started"]=True
    assert list(Draft202012Validator(r1.strict_read(REPO/r1.CONFIG_SCHEMA)).iter_errors(changed))


def test_r2_r1_artifact_schema_has_no_open_object_nodes():
    schema=r1.strict_read(REPO/r1.ARTIFACT_SCHEMA);bad=[]
    def walk(value,path):
        if isinstance(value,dict):
            if value.get("type")=="object" and value.get("additionalProperties") is not False:bad.append(path)
            for key,item in value.items():walk(item,path+"/"+key)
        elif isinstance(value,list):
            for index,item in enumerate(value):walk(item,path+f"/{index}")
    walk(schema,"root");assert bad==[]


def test_r2_r1_unit_identities_are_fresh():
    old=r1.r1.load_units(REPO);rows=[]
    for item in old:
        identity={"run_id":r1.RUN_ID,"model_id":item["model_id"],"sample_id":item["sample_id"],"sample_sha256":item["sample_sha256"],"development_rank":item["development_rank"]};rows.append(r1.canonical_sha256(identity))
    assert len(rows)==len(set(rows))==200
    assert not set(rows)&{item["unit_digest"] for item in old}


def test_preflight_refuses_torch_or_timm_import(monkeypatch):
    monkeypatch.setitem(sys.modules,"torch",object())
    with pytest.raises(r1.X2R2R1Error):
        r1.preflight_worker(REPO,"clean")


def test_r2_blocked_binding_source_state():
    state=r1._recompute_r2_blocked(REPO)
    assert state["stage"]["status"]=="BLOCKED_INFRASTRUCTURE"
    assert state["seal"]["canonical_digest"]=="514df49130f4e7dd903468d20ad8e269717dd48061bdee12c73c568884184b7c"
    assert len(state["terminals"])==2
    assert all(row["status"]=="NA" for row in state["terminals"])


@pytest.mark.parametrize("metadata",[
    {"inverse":"rotate","angle":10.0,"valid_box":[20,204,20,204]},
    {"inverse":"horizontal_flip"},
    {"inverse":"resize_224_to_216_then_pad4"},
])
def test_inverse_map_accepts_2d_stability_maps(metadata):
    restored,mask=metrics.inverse_map(torch.zeros(224,224),metadata)
    assert restored.shape==(224,224)
    assert mask.shape==(224,224)
