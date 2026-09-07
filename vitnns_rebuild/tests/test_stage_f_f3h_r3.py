from __future__ import annotations

import ast
from pathlib import Path

from vitnns import stage_f_f3h_r3 as r3

REPO=Path(__file__).resolve().parents[1]

def test_r3_source_has_no_model_gpu_or_image_imports() -> None:
    source=(REPO/"src/vitnns/stage_f_f3h_r3.py").read_text(encoding="utf-8");tree=ast.parse(source);imports=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Import): imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node,ast.ImportFrom) and node.module: imports.add(node.module.split(".")[0])
    assert not imports.intersection({"torch","torchvision","timm","PIL","cv2"})

def test_r3_attack_list_and_authorization_are_exact() -> None:
    assert len(r3.ATTACKS)==12 and len(set(r3.ATTACKS))==12
    assert r3.ARCHIVE_REF=="refs/archive/stage-f-v1-f3h-completion"

def test_r3_live_source_validation_is_cpu_only() -> None:
    result=r3.validate_source(REPO)
    assert result["status"]=="PASS"
    assert result["validation"]["value_path_counts"]["valid"]==288
    assert result["validation"]["class_conditioned_basis"]=="INSUFFICIENT_BASIS_FOR_NUMERICAL_CLASS_CONDITIONED_WEIGHTS"
    assert all(value is False for value in result["runtime"].values())

def test_r3_completion_is_atomic_if_present() -> None:
    if not (REPO/r3.COMPLETION_PATH).exists(): return
    result=r3.validate_completion(REPO)
    assert result["status"]=="PASS" and result["next_authorization"]=="F3C0_PROTOCOL_FREEZE_ONLY"
