"""E1 patch-scale contract, computation, and false-PASS tests."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

import vitnns.stage_e_e1 as e1
from vitnns.stage_d_provenance import canonical_sha256

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_frozen_e1_contract_schema_variants_budget_and_authorization() -> None:
    config=e1._validate_config(REPO_ROOT)
    assert config["canonical_digest"] == e1.EXPECTED_E1_CONFIG_DIGEST
    assert tuple(row["variant_id"] for row in config["scope"]["variants"]) == e1.VARIANTS
    assert config["scope"]["variants"][0]["compute"] == "FORBIDDEN_RECOMPUTE"
    assert config["scope"]["expected_image_groups"] == 500
    assert config["scope"]["expected_new_units"] == 1000
    assert config["authorization"]["on_pass"] == "E2_ONLY"
    assert all(config["authorization"][key] is False for key in ("e2_started","e3_started","e4_started","stage_f_started"))
    assert config["budget"] == {"gpu_hours_max":2.5,"cpu_hours_max":8.0,"retained_bytes_max":4294967296}


def test_registry_and_d10_anchor_order_are_exact() -> None:
    registry,_=e1._load_registry(REPO_ROOT)
    anchor,source=e1._anchor_rows(REPO_ROOT)
    assert len(registry)==len(anchor)==source["row_count"]==500
    assert [row["sample_id"] for row in registry] == [row["sample_id"] for row in anchor]
    assert [row["selection_order"] for row in anchor] == list(range(500))
    assert len({row["sample_id"] for row in registry})==500
    assert all(row["recompute_status"]=="FORBIDDEN_RECOMPUTE" for row in anchor)


def test_typed_plan_has_every_expected_patch_unit() -> None:
    config=e1._validate_config(REPO_ROOT)
    plan=e1._build_plan(REPO_ROOT,config)
    assert len(plan["groups"])==500
    assert len(plan["units"])==1000
    assert [row["selection_order"] for row in plan["groups"]]==list(range(500))
    assert {row["variant_id"] for row in plan["units"]}==set(e1.PATCH_VARIANTS)
    assert all(row["scale"]=="native_14x14_patch" for row in plan["units"])


def test_float64_aligned_patch_aggregation_and_mass_conservation() -> None:
    source=np.arange(56*56,dtype=np.float32).reshape(56,56)-1000.25
    observed=e1.aggregate_patch_map(source)
    expected=np.empty((14,14),dtype=np.float64)
    for row in range(14):
        for column in range(14):
            expected[row,column]=source[4*row:4*row+4,4*column:4*column+4].astype(np.float64).sum()
    assert observed.dtype==np.float64
    assert np.array_equal(observed,expected)
    assert observed.sum(dtype=np.float64)==source.sum(dtype=np.float64)


def test_matched_native_patch_recovery_fixture_and_cls_not_ranked() -> None:
    native=np.arange(196,dtype=np.float64).reshape(14,14)-50.5
    fine=np.repeat(np.repeat(native/16.0,4,axis=0),4,axis=1).astype(np.float32)
    recovered=e1.aggregate_patch_map(fine)
    assert np.allclose(recovered,native,rtol=0,atol=2e-5)
    cls=10**9
    assert e1.patch_ranking(recovered,sigma=1,seed=7).shape==(196,)
    assert cls not in e1.patch_ranking(recovered,sigma=1,seed=7)


def test_patch_counts_actual_fractions_and_auc_golden() -> None:
    expected=(0,10,20,30,40,49,59,69,79,89,98,108,118,128,138,147,157,167,177,187,196)
    assert e1.UNIT_COUNTS==expected
    assert e1.ACTUAL_FRACTIONS==tuple(value/196 for value in expected)
    assert e1.trapezoid_auc(e1.ACTUAL_FRACTIONS)==pytest.approx(0.5)


def test_common_seed_and_tie_priority_are_deterministic_and_method_independent() -> None:
    sample="ILSVRC2012_val/example.JPEG"
    seeds=[e1.e1_seed(sample,repeat) for repeat in range(5)]
    assert len(set(seeds))==5
    for seed in seeds:
        assert np.array_equal(e1.tie_priority(seed),e1.tie_priority(seed))
    tied=np.zeros((14,14),dtype=np.float64)
    for seed in seeds:
        ranking=e1.patch_ranking(tied,sigma=1,seed=seed)
        assert np.array_equal(ranking,np.argsort(e1.tie_priority(seed),kind="stable"))


def test_delta_fid_patch_synthetic_golden() -> None:
    nns_i,nns_d,matched_i,matched_d=0.8,0.2,0.6,0.4
    observed=0.5*((nns_i-matched_i)+(matched_d-nns_d))
    assert observed==pytest.approx(0.2)


@pytest.mark.parametrize("mutation",[
    lambda value:value["scope"]["variants"].pop(),
    lambda value:value["scope"].update({"expected_new_units":999}),
    lambda value:value["scientific_contract"].update({"unit_count":3136}),
    lambda value:value["scientific_contract"].update({"tie_seed_comparison_identity":"method_specific"}),
    lambda value:value["scientific_contract"].update({"p_values_emitted":True}),
    lambda value:value["authorization"].update({"on_pass":"E3_ONLY"}),
    lambda value:value["budget"].update({"gpu_hours_max":3.0}),
])
def test_scope_seed_decision_or_budget_drift_cannot_validate(tmp_path:Path,monkeypatch:pytest.MonkeyPatch,mutation) -> None:
    source=json.loads((REPO_ROOT/e1.CONFIG_PATH).read_text())
    mutation(source)
    source["canonical_digest"]=canonical_sha256({k:v for k,v in source.items() if k!="canonical_digest"})
    path=tmp_path/"stage_e_e1.json"; path.write_text(json.dumps(source)+"\n")
    monkeypatch.setattr(e1,"CONFIG_PATH",str(path))
    with pytest.raises(e1.StageEE1Error):
        e1._validate_config(REPO_ROOT)


def test_attempt_missing_curve_nonfinite_and_placeholder_cannot_pass(tmp_path:Path,monkeypatch:pytest.MonkeyPatch) -> None:
    unit={"canonical_digest":"a"*64,"variant_id":e1.PATCH_VARIANTS[0],"sample_id":"x"}
    monkeypatch.setattr(e1,"ROOT_PATH","formal")
    root=tmp_path/"formal/units"/("a"*64)/"attempts/attempt-000001"
    root.mkdir(parents=True)
    result=e1._signed({"unit_digest":"a"*64,"variant_id":e1.PATCH_VARIANTS[0],"sample_id":"x","validity_status":"VALID","curves":[]})
    raw=e1.canonical_json_bytes(result)+b"\n"; (root/"result.json").write_bytes(raw)
    status=e1._signed({"terminal":True,"attempt_id":"attempt-000001"}); sraw=e1.canonical_json_bytes(status)+b"\n"; (root/"status.json").write_bytes(sraw)
    manifest=e1._signed({"result_sha256":e1._sha(raw),"status_sha256":e1._sha(sraw)}); (root/"manifest.json").write_bytes(e1.canonical_json_bytes(manifest)+b"\n")
    with pytest.raises(e1.StageEE1Error):
        e1._validate_attempt(tmp_path,unit)
    with pytest.raises(e1.StageEE1Error):
        e1.trapezoid_auc([math.nan]*21)


def test_attempt_source_hash_pair_binding_and_inventory_tamper_cannot_pass(tmp_path:Path,monkeypatch:pytest.MonkeyPatch) -> None:
    unit={"canonical_digest":"b"*64,"variant_id":e1.PATCH_VARIANTS[0],"sample_id":"x"}
    monkeypatch.setattr(e1,"ROOT_PATH","formal")
    root=tmp_path/"formal/units"/("b"*64)/"attempts/attempt-000001"
    root.mkdir(parents=True)
    result=e1._signed({"unit_digest":"b"*64,"variant_id":e1.PATCH_VARIANTS[0],"sample_id":"x","validity_status":"VALID","curves":[],"pair_binding_digest":"0"*64})
    raw=e1.canonical_json_bytes(result)+b"\n"; (root/"result.json").write_bytes(raw)
    status=e1._signed({"terminal":True,"attempt_id":"attempt-000001"}); sraw=e1.canonical_json_bytes(status)+b"\n"; (root/"status.json").write_bytes(sraw)
    manifest=e1._signed({"result_sha256":"0"*64,"status_sha256":e1._sha(sraw)}); (root/"manifest.json").write_bytes(e1.canonical_json_bytes(manifest)+b"\n")
    with pytest.raises(e1.StageEE1Error,match="control binding"):
        e1._validate_attempt(tmp_path,unit)


def test_budget_and_missing_terminal_unit_cannot_pass(monkeypatch:pytest.MonkeyPatch) -> None:
    assert e1.budget_reasons({"gpu_hours":2.500001,"cpu_seconds":0},0)==("gpu_hours",)
    assert e1.budget_reasons({"gpu_hours":0,"cpu_seconds":8*3600+1},0)==("cpu_hours",)
    assert e1.budget_reasons({"gpu_hours":0,"cpu_seconds":0},4294967297)==("retained_bytes",)
    config=e1._validate_config(REPO_ROOT)
    plan=e1._build_plan(REPO_ROOT,config)
    original=e1._validate_attempt
    calls=0
    def missing(repo,unit):
        nonlocal calls
        calls+=1
        if calls==1:
            raise e1.StageEE1Error("attempt is absent or incomplete")
        return original(repo,unit)
    monkeypatch.setattr(e1,"_validate_attempt",missing)
    with pytest.raises(e1.StageEE1Error,match="absent or incomplete"):
        e1._collect_patch_rows(REPO_ROOT,plan)


def test_terminal_completion_is_atomic_and_authorizes_only_e2() -> None:
    completion=json.loads((REPO_ROOT/e1.COMPLETION_PATH).read_text())
    assert completion["status"]=="PASS"
    assert completion["next_authorization"]=="E2_ONLY"
    assert not (REPO_ROOT/e1.COMPLETION_PENDING_PATH).exists()
    assert completion["predecessor"]["canonical_digest"]==e1.E0_COMPLETION_DIGEST


def test_false_pass_guard_rejects_every_attack() -> None:
    valid={name:True for name in e1.FALSE_PASS_ATTACKS}
    assert e1.false_pass_reasons(valid)==()
    for name in e1.FALSE_PASS_ATTACKS:
        mutated=dict(valid); mutated[name]=False
        assert e1.false_pass_reasons(mutated)==(name,)


def test_output_schema_rejects_empty_placeholder_and_no_render_contract() -> None:
    schema=json.loads((REPO_ROOT/e1.OUTPUT_SCHEMA_PATH).read_text())
    Draft202012Validator.check_schema(schema)
    assert list(Draft202012Validator(schema).iter_errors({"schema_version":1}))
    config=e1._validate_config(REPO_ROOT)
    assert config["execution"]["renders_png"] is False
    assert config["execution"]["renders_pdf"] is False
    assert config["execution"]["renders_svg"] is False


def test_no_e2_e3_e4_or_stage_f_execution_surface() -> None:
    public={name for name in dir(e1) if name.startswith(("run_","start_","execute_"))}
    assert public=={"run_shard"}
    assert set(e1.VARIANTS)=={"d10_fine_4x4_primary_anchor","nns_native_14x14_patch","matched_native_14x14_patch"}
