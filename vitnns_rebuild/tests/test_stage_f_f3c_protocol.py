from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

from vitnns import stage_f_f3c_protocol as f3c

REPO=Path(__file__).resolve().parents[1]

def test_config_is_signed_and_has_all_five_methods() -> None:
    config=f3c.load_config(REPO)
    assert tuple(x["id"] for x in config["methods"])==f3c.METHODS
    assert config["class_targeted_nns"]["manual_head_weights"] is False
    assert config["class_targeted_nns"]["f3h_head_map_weighted_sum"] is False

def test_sample_split_is_prospective_disjoint_and_excludes_viewed() -> None:
    split=json.loads((REPO/f3c.SPLIT_PATH).read_text())
    pilot={x["selection_order"] for x in split["pilot"]};formal={x["selection_order"] for x in split["formal"]}
    assert len(pilot)==5 and len(formal)==50 and not pilot&formal
    assert not (pilot|formal)&set(f3c.EXCLUDED)
    assert split["selection_rule"]["jpeg_or_f3c_outcome_accessed"] is False

def test_protocol_text_defines_direct_class_target_not_posthoc_weights() -> None:
    text=(REPO/f3c.PROTOCOL_PATH).read_text(encoding="utf-8")
    for phrase in ("conditional_all_blocks_fixed_clean_attention_class_logit_ig","does **not** learn, tune, rank, or visually choose head weights","ViT Grad-CAM","full-model class-logit IG","attention_rollout_mean_head_residual_all_blocks","direct_raw_cls_attention_last_block_head_mean"):
        assert phrase in text

def test_f3c0_module_has_no_torch_timm_or_image_import() -> None:
    tree=ast.parse((REPO/"src/vitnns/stage_f_f3c_protocol.py").read_text())
    imports=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):imports.update(x.name.split(".")[0] for x in node.names)
        elif isinstance(node,ast.ImportFrom) and node.module:imports.add(node.module.split(".")[0])
    assert not imports.intersection({"torch","torchvision","timm","PIL","cv2"})

def test_false_pass_attack_set_is_complete() -> None:
    assert len(f3c.ATTACKS)==20 and len(set(f3c.ATTACKS))==20

@pytest.mark.parametrize("mutation",["drop_method","manual_weights","start_f4","overlap"] )
def test_semantic_false_pass_mutations_are_rejected(monkeypatch:pytest.MonkeyPatch,mutation:str) -> None:
    original=f3c.load_config(REPO);changed=copy.deepcopy(original)
    if mutation=="drop_method":changed["methods"].pop()
    elif mutation=="manual_weights":changed["class_targeted_nns"]["manual_head_weights"]=True
    elif mutation=="start_f4":changed["authorization"]["f4_started"]=True
    else:
        split=json.loads((REPO/f3c.SPLIT_PATH).read_text());split["formal"][0]=split["pilot"][0]
        monkeypatch.setattr(f3c,"_read",lambda path:split if path==REPO/f3c.SPLIT_PATH else (changed if path==REPO/f3c.CONFIG_PATH else json.loads(path.read_text())))
        with pytest.raises(f3c.StageFF3CProtocolError):f3c.validate_protocol(REPO)
        return
    monkeypatch.setattr(f3c,"load_config",lambda _repo:changed)
    with pytest.raises(f3c.StageFF3CProtocolError):f3c.validate_protocol(REPO)

def test_completion_is_atomic_if_present() -> None:
    if not (REPO/f3c.COMPLETION_PATH).exists():return
    result=f3c.validate_completion(REPO);assert result["status"]=="PASS";assert result["next_authorization"]=="F3C1_IMPLEMENTATION_AND_SYNTHETIC_PREFLIGHT_ONLY"

def test_cli_exposes_only_protocol_lifecycle() -> None:
    source=(REPO/"src/vitnns/stage_f_f3c_protocol.py").read_text(encoding="utf-8")
    for command in ("validate","freeze","validate-completion"):
        assert command in source
    for forbidden in ("run-pilot","run-shard","run-f3c","render-f3c"):
        assert forbidden not in source

def test_unified_stage_f_cli_exposes_f3c0_only() -> None:
    source=(REPO/"src/vitnns/stage_f_cli.py").read_text(encoding="utf-8")
    for command in ("validate-f3c0-protocol","freeze-f3c0-protocol","validate-f3c0-completion"):
        assert command in source
    assert "run-f3c-pilot" not in source and "run-f3c-formal" not in source
