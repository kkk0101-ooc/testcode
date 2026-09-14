from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("s2_validator", ROOT / "scripts/validate_nns_vit_patch_depth_s2.py")
assert SPEC and SPEC.loader
s2 = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(s2)


def test_blocked_s2_root_recomputes() -> None:
    result = s2.validate(ROOT / "evidence/nns_vit_patch_depth/s2_r3_20260914")
    assert result["terminal_status"] == "BLOCKED_SCIENTIFIC_VALIDITY"
    assert result["formal_outcome_access_count"] == 0
    assert result["next_authorization"] == "NONE"


def test_false_pass_mutations_are_rejected() -> None:
    result = s2.false_pass(ROOT / "evidence/nns_vit_patch_depth/s2_r3_20260914")
    assert result["status"] == "PASS" and result["attack_count"] >= 20
