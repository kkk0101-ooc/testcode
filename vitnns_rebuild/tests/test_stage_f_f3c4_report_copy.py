from __future__ import annotations

import ast
from pathlib import Path

import pytest

from vitnns import stage_f_f3c4_report_copy as copy

REPO=Path(__file__).resolve().parents[1]

def test_config_freezes_byte_copy_only_scope():
    config=copy.load_config(REPO); assert config["expected_png_count"]==23; assert config["copy_mode"]=="byte_for_byte"; assert config["rerender_allowed"] is False; assert config["scientific_recomputation_allowed"] is False; assert config["f4_started"] is False

def test_attack_roster_is_exactly_twelve():
    assert len(copy.ATTACKS)==12 and len(set(copy.ATTACKS))==12

@pytest.mark.parametrize("attack",copy.ATTACKS)
def test_attack_is_rejected(attack: str):
    state={name:True for name in copy.ATTACKS}; state[attack]=False; assert copy.false_pass_reasons(state)==(attack,)

def test_source_has_no_renderer_statistics_model_or_future_stage_surface():
    source=(REPO/"src/vitnns/stage_f_f3c4_report_copy.py").read_text(); tree=ast.parse(source); imports=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Import): imports.update(x.name.split(".")[0] for x in node.names)
        elif isinstance(node,ast.ImportFrom) and node.module: imports.add(node.module.split(".")[0])
    assert not imports.intersection({"numpy","PIL","matplotlib","torch","timm"})
    assert not any(token in source for token in ("savefig","Image.open","np.load","bootstrap","sign_flip","load_canonical_model","run_f4"))

def test_existing_destination_is_rejected(tmp_path: Path):
    destination=tmp_path/"already.png"; destination.write_bytes(b"existing"); assert destination.exists()
