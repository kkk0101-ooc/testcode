from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from vitnns import stage_x_x0 as x0

REPO = Path(__file__).resolve().parents[1]

def test_strict_json_rejects_duplicate_and_nonfinite() -> None:
    with pytest.raises(x0.StageXX0Error, match="duplicate"):
        x0.strict_json_bytes(b'{"a":1,"a":2}', context="fixture")
    for token in (b'NaN', b'Infinity', b'-Infinity'):
        with pytest.raises(x0.StageXX0Error, match="non-finite"):
            x0.strict_json_bytes(b'{"a":'+token+b'}', context="fixture")

def test_config_schema_is_closed_required_and_rejects_bool_as_int() -> None:
    config=x0.read_json(REPO/x0.CONFIG); schema=x0.read_json(REPO/x0.SCHEMA)
    Draft202012Validator.check_schema(schema)
    assert not list(Draft202012Validator(schema).iter_errors(config))
    extra=copy.deepcopy(config); extra["unknown"]=1
    assert list(Draft202012Validator(schema).iter_errors(extra))
    missing=copy.deepcopy(config); missing.pop("nns_v2")
    assert list(Draft202012Validator(schema).iter_errors(missing))
    boolean=copy.deepcopy(config); boolean["statistics"]["cell_bootstrap"]=True
    assert list(Draft202012Validator(schema).iter_errors(boolean))

def test_canonical_self_digest_and_round_trip() -> None:
    documents=x0.load_contracts(REPO)
    for name,value in documents.items():
        if name != "schema": x0.validate_signed(value,name)
    config=documents["config"]
    assert x0.strict_json_bytes(x0.canonical_bytes(config),context="roundtrip")==config

def test_historical_statuses_and_f3c_are_source_recomputed() -> None:
    validation=x0.validate_protocol(REPO)
    assert validation["history"]=={"c10":"UNCHANGED","d10":"UNCHANGED","e4":"UNCHANGED","f3c":"UNCHANGED"}
    f3c=validation["f3c_recomputed"]
    assert f3c["mean"]==-0.05585432814553341
    assert f3c["ci"]==[-0.07085058706573369,-0.04100117599983641]
    assert f3c["p_value"]==9.99990000099999e-06

def test_claim_registry_has_eight_falsifiable_claims_and_no_universal_status() -> None:
    registry=x0.claim_registry(); assert len(registry["claims"])==8
    assert tuple(row["claim_id"] for row in registry["claims"])==x0.CLAIM_IDS
    assert x0.FORBIDDEN_CLAIM not in x0.canonical_bytes(registry).decode()
    for row in registry["claims"]:
        for key in ("population","target","estimand","pass_condition","no_go_status","inconclusive_status","invalidity","permitted_wording"):
            assert row[key]

def test_primary_secondary_registry_and_required_metadata() -> None:
    documents=x0.load_contracts(REPO); counts=x0._validate_registry(documents)
    assert counts=={"included_primary_comparators":6,"included_secondary_references":5,"deferred":5,"excluded":0}
    primary=documents["comparators"]["primary_family"]
    assert all(row["class_targeted"] for row in primary)
    for row in primary[1:]:
        assert row["official_source"] and row["implementation_url"] and row["commit"] and row["license"] and row["target_semantics"]

def test_model_dataset_matrix_and_sample_isolation_are_exact() -> None:
    documents=x0.load_contracts(REPO); report=x0._validate_matrix_and_split(documents)
    assert report["required_cells"]==20 and report["prior_access_exclusions"]==520
    benchmark=documents["benchmark"]
    assert benchmark["aggregation"]["simple_image_pooling_across_cells"] is False
    assert all(cell["n"] in {740,919,1000} for cell in benchmark["required_cells"])
    assert documents["split"]["isolation"]["x2_excluded_from_x3_x5"] is True

def test_pixel_localization_metric_and_statistics_contracts_are_frozen() -> None:
    documents=x0.load_contracts(REPO); config=documents["config"]; benchmark=documents["benchmark"]
    assert config["nns_v2"]["native_shape"]==[3,224,224]
    assert config["nns_v2"]["display_isolated"] is True
    assert benchmark["endpoints"]["pixel_interventions"]==[1,2,4,8,16]
    assert benchmark["endpoints"]["localization_primary"]=="class_macro_pixel_AUPRC_positive_support"
    assert config["statistics"]["required_failure_complete_case_allowed"] is False
    assert config["statistics"]["margin_seal_stage"]=="X2_BEFORE_X3_OUTCOME_ACCESS"

def test_resource_caps_and_futility_are_not_result_adaptive() -> None:
    config=x0.load_contracts(REPO)["config"]
    assert config["resources"]["x1_caps"]["h20_gpu_hours"]==16
    assert config["resources"]["x2_caps"]["h20_gpu_hours"]==160
    assert config["resources"]["automatic_waiver"] is False
    assert config["resources"]["posthoc_workload_reduction"] is False
    assert "upper_ci_below_negative" in config["statistics"]["futility_algorithm"]

def test_x0_module_has_no_model_image_download_or_cuda_surface() -> None:
    source=(REPO/"src/vitnns/stage_x_x0.py").read_text(encoding="utf-8")
    tree=ast.parse(source); imports=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Import): imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node,ast.ImportFrom) and node.module: imports.add(node.module.split(".")[0])
    assert not imports.intersection({"torch","torchvision","timm","PIL","cv2"})
    assert "subprocess.run([\"nvidia-smi\"" not in source

@pytest.mark.parametrize("mutation",[
    "historical","universal","class_agnostic_primary","missing_license",
    "cell_removed","x2_count","prior_access","pixel_shape",
    "localization","multiplicity","margin_time","complete_case",
    "pooling","early_stage","automatic_waiver",
    "metric_contract","failure_na",
])
def test_false_pass_semantic_mutations_are_rejected(monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    docs=copy.deepcopy(x0.load_contracts(REPO))
    with pytest.raises(x0.StageXX0Error):
        if mutation=="historical":
            docs["config"]["history"]["c10"]["decision"]="GO"; x0._validate_protocol_semantics(REPO,docs)
        elif mutation=="universal":
            docs["comparators"]["secondary_family"][0]["target_semantics"]=x0.FORBIDDEN_CLAIM; x0._validate_registry(docs)
        elif mutation=="class_agnostic_primary":
            docs["comparators"]["primary_family"][1]["class_targeted"]=False; x0._validate_registry(docs)
        elif mutation=="missing_license":
            docs["literature"]["records"][0]["license"]=None; x0._validate_registry(docs)
        elif mutation=="cell_removed": docs["benchmark"]["required_cells"].pop(); x0._validate_matrix_and_split(docs)
        elif mutation=="x2_count": docs["split"]["x2"]["images_per_required_model"]=39; x0._validate_matrix_and_split(docs)
        elif mutation=="prior_access": docs["prior_access"]["unique_prior_access_exclusion_count"]=519; x0._validate_matrix_and_split(docs)
        elif mutation=="pixel_shape": docs["config"]["nns_v2"]["native_shape"]=[14,14]; x0._validate_protocol_semantics(REPO,docs)
        elif mutation=="localization": docs["model_data"]["datasets"][3]["annotation"]="none"; x0._validate_matrix_and_split(docs)
        elif mutation=="multiplicity": docs["config"]["statistics"]["multiplicity"]="none"; x0._validate_protocol_semantics(REPO,docs)
        elif mutation=="margin_time": docs["config"]["statistics"]["margin_seal_stage"]="AFTER_X3"; x0._validate_protocol_semantics(REPO,docs)
        elif mutation=="complete_case": docs["config"]["statistics"]["required_failure_complete_case_allowed"]=True; x0._validate_protocol_semantics(REPO,docs)
        elif mutation=="pooling": docs["benchmark"]["aggregation"]["simple_image_pooling_across_cells"]=True; x0._validate_matrix_and_split(docs)
        elif mutation=="early_stage": docs["config"]["authorization"]["x1_started"]=True; x0._validate_protocol_semantics(REPO,docs)
        elif mutation=="metric_contract": docs["benchmark"]["faithfulness_contract"]["primary_direction"]="lower_is_better"; x0._validate_matrix_and_split(docs)
        elif mutation=="failure_na": docs["benchmark"]["terminal_policy"]["drop_failed_or_na"]=True; x0._validate_matrix_and_split(docs)
        else: docs["config"]["resources"]["automatic_waiver"]=True; x0._validate_protocol_semantics(REPO,docs)

def test_false_pass_attack_registry_covers_all_requested_classes() -> None:
    assert len(x0.ATTACKS)==22 and len(set(x0.ATTACKS))==22
    assert x0._false_pass()["attack_count"]==22

def test_completion_is_atomic_and_fully_bound_if_present() -> None:
    if not (REPO/x0.COMPLETION).exists(): return
    result=x0.validate_completion(REPO)
    assert result["status"]=="PASS"
    assert result["next_authorization"]=="X1_NNS_V2_IMPLEMENTATION_ONLY"
    assert not (REPO/x0.PENDING).exists()

def test_cli_exposes_only_x0_lifecycle() -> None:
    source=(REPO/"src/vitnns/stage_x_x0.py").read_text(encoding="utf-8")
    for command in ("validate","freeze","validate-completion"): assert command in source
    for forbidden in ("run-x1","run-x2","run-x3","load-model","download-dataset"): assert forbidden not in source
