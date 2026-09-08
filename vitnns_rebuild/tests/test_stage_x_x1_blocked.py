from __future__ import annotations
import ast,copy
from pathlib import Path
import pytest
from vitnns import stage_x_x1_blocked as x1
REPO=Path(__file__).resolve().parents[1]

def test_blocker_is_source_reproducible_and_has_no_next_authorization():
    result=x1.validate(REPO)
    assert result["status"]=="PASS_BLOCKER_CONFIRMED"
    assert result["blocker"]=="BLOCKED_MISSING_PINNED_COMPARATOR_ENGINE"
    assert result["next_authorization"]=="NONE_REQUIRES_PROSPECTIVE_X1_RECOVERY_AMENDMENT"

def test_required_pinned_engines_are_exact():
    config=x1.load(REPO)
    assert [(row["method_id"],row["commit"]) for row in config["required_engines"]]==[("transformer_attribution_chefer","c3e578f76b954e8528afeaaee26de3f07e3fe559"),("attnlrp","f04344177551bfcc3665e88afe96501f43fe560b")]
    assert all(row["local_status"]=="MISSING" for row in config["required_engines"])

def test_blocked_recorder_imports_no_runtime_stack():
    tree=ast.parse((REPO/"src/vitnns/stage_x_x1_blocked.py").read_text());imports=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node,ast.ImportFrom) and node.module:imports.add(node.module.split(".")[0])
    assert not imports&{"torch","timm","torchvision","PIL","cv2","requests"}

@pytest.mark.parametrize("field",["new_empirical_outcome_access","formal_image_bytes_accessed","audit_image_bytes_accessed","model_or_checkpoint_bytes_accessed","model_loaded","torch_or_timm_imported_by_preflight","cuda_or_gpu_accessed","dataset_or_checkpoint_downloaded","x1_accepted_implementation","x2_started","x3_started","f3_final_started","stage_g_started"] )
def test_hidden_access_or_early_stage_is_rejected(monkeypatch,field):
    original=x1.load(REPO);changed=copy.deepcopy(original);changed["access_boundary"][field]=True;monkeypatch.setattr(x1,"load",lambda _repo:changed)
    with pytest.raises(x1.StageXX1BlockedError):x1.validate(REPO)

def test_blocked_status_cannot_be_relabelled_pass(monkeypatch):
    original=x1.load(REPO);changed=copy.deepcopy(original);changed["status"]="PASS";monkeypatch.setattr(x1,"load",lambda _repo:changed)
    with pytest.raises(x1.StageXX1BlockedError):x1.validate(REPO)

def test_false_pass_attack_registry():
    assert len(x1.ATTACKS)==12 and len(set(x1.ATTACKS))==12

def test_completion_if_present():
    if (REPO/x1.COMPLETION).exists():assert x1.validate_completion(REPO)["status"]=="BLOCKED_MISSING_PINNED_COMPARATOR_ENGINE"
