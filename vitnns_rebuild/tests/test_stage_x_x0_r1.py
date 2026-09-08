from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from vitnns import stage_x_x0_r1 as r1

REPO=Path(__file__).resolve().parents[1]

def test_original_identity_contradiction_is_source_proven() -> None:
    report=r1.prove_original_contradiction(REPO)
    assert report["status"]=="CONFIRMED" and all(report["facts"].values())
    assert report["conclusion"]=="STRICT_SUPERIORITY_AGAINST_IDENTICAL_TOTAL_ATTRIBUTION_IS_IMPOSSIBLE"

def test_corrected_config_is_closed_signed_and_round_trips() -> None:
    docs=r1.load_contracts(REPO); config=docs["config"]; schema=docs["schema"]
    assert not list(Draft202012Validator(schema).iter_errors(config)); r1.validate_signed(config,"config")
    assert r1.read_json_bytes(r1.canonical_bytes(config))==config

def test_validator_accepts_only_identity_control_plus_five_competitors() -> None:
    result=r1.validate_correction(REPO)
    assert result["identity_control"]["method_id"]=="full_model_integrated_gradients"
    assert result["identity_control"]["superiority_test_allowed"] is False
    assert tuple(result["competitive_primary_comparators"])==r1.COMPETITIVE

def test_identity_control_is_not_in_holm_family() -> None:
    benchmark=r1.load_contracts(REPO)["benchmark"]
    assert all("full_model_integrated_gradients" not in item for item in benchmark["multiplicity"]["family_members"])
    assert len(benchmark["multiplicity"]["family_members"])==5

@pytest.mark.parametrize("mutation",["identity_superiority","identity_removed","loose_tolerance","different_steps","wrong_failure","drop_competitor","holm_identity","weak_wording","x1_started"] )
def test_false_pass_mutations_are_rejected(monkeypatch:pytest.MonkeyPatch,mutation:str) -> None:
    docs=copy.deepcopy(r1.load_contracts(REPO)); cfg=docs["config"]; comp=docs["comparators"]; benchmark=docs["benchmark"]
    if mutation=="identity_superiority": cfg["identity_control"]["superiority_test_allowed"]=True
    elif mutation=="identity_removed": comp["identity_control"]["id"]="none"
    elif mutation=="loose_tolerance": cfg["identity_control"]["atol"]=1e-2
    elif mutation=="different_steps": cfg["identity_control"]["same_steps_required"]=False
    elif mutation=="wrong_failure": cfg["identity_control"]["mismatch_status"]="INCONCLUSIVE"
    elif mutation=="drop_competitor": benchmark["competitive_primary_family"].pop()
    elif mutation=="holm_identity": benchmark["multiplicity"]["family_members"][0]="nns_v2_minus_full_model_integrated_gradients"
    elif mutation=="weak_wording": cfg["strongest_permitted_wording"]="NNS-v2 wins universally"
    else: cfg["access_boundary"]["x1_accepted_implementation"]=True
    monkeypatch.setattr(r1,"load_contracts",lambda _repo:docs)
    with pytest.raises(r1.StageXX0R1Error): r1.validate_correction(REPO)

def test_x0_r1_source_has_no_model_image_network_torch_or_cuda_surface() -> None:
    source=(REPO/"src/vitnns/stage_x_x0_r1.py").read_text(); tree=ast.parse(source); imports=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Import): imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node,ast.ImportFrom) and node.module: imports.add(node.module.split(".")[0])
    assert not imports&{"torch","timm","torchvision","PIL","cv2","requests"}

def test_original_x0_files_remain_byte_identical_to_archive() -> None:
    assert r1.sha256_file(REPO/r1.OLD_COMPLETION)==r1.OLD_COMPLETION_SHA
    assert r1._git(REPO,"rev-parse",r1.UPSTREAM_REF+"^{commit}")==r1.BASE_COMMIT

def test_false_pass_registry_is_complete() -> None:
    assert len(r1.ATTACKS)==14 and len(set(r1.ATTACKS))==14

def test_completion_is_atomic_if_present() -> None:
    if not (REPO/r1.COMPLETION).exists(): return
    result=r1.validate_completion(REPO); assert result["status"]=="PASS"; assert result["next_authorization"]=="X1_NNS_V2_IMPLEMENTATION_ONLY"

def test_no_x1_implementation_draft_is_present() -> None:
    for path in ("src/vitnns/stage_x_nns_v2.py","src/vitnns/stage_x_comparators.py","tests/test_stage_x_nns_v2.py","tests/test_stage_x_comparators.py"):
        assert not (REPO/path).exists()
