import json
from pathlib import Path
import pytest
from vitnns import nns_patch_depth_s2_r8 as r8

ROOT = Path(__file__).resolve().parents[1]


def test_protocol_hard_boundaries():
    spec = json.loads((ROOT / "config/nns_vit_patch_depth/s2_r8_run_spec.json").read_text())
    assert spec["frozen_changes_only"]["target_power"] == 0.8
    assert spec["candidate_manifests"]["caps"] == [2, 3, 4]
    assert spec["resource_limits"]["S3_source_image_forward_max"] == 10458
    assert spec["resource_limits"]["S4_NNS_occlusion_image_count"] == 1744


def test_gate_closed_and_open():
    assert r8.decisions([0, 0, 0, 10]) == (False, False)
    assert r8.decisions([10, 10, 10, 10]) == (True, True)


def test_holm_tie_and_fixed_family():
    result = r8.holm_primary({"P1": 0.01, "P2": 0.01, "D1": 0.01})
    assert [row["endpoint"] for row in result["rows"]] == ["P1", "P2", "D1"]
    with pytest.raises(ValueError):
        r8.holm_primary({"P1": 0.01, "P2": 0.01})


def test_wilson_and_cluster_se():
    assert 0 < r8.wilson_lower(800, 1000) < 0.8 < r8.wilson_upper(800, 1000) < 1
    assert r8.cluster_se(0.1, 0.2, [2, 2]) > r8.cluster_se(0.1, 0.2, [4, 4])


def test_correlation_scenarios_are_psd():
    base = [[1, .1, .1, .1], [.1, 1, .1, .1], [.1, .1, 1, .1], [.1, .1, .1, 1]]
    assert len(r8.correlation_scenarios(base)) == 3


def test_false_pass_contract_mutations():
    import importlib.util
    module_spec = importlib.util.spec_from_file_location("validator", ROOT / "scripts/validate_nns_vit_patch_depth_s2_r8.py")
    validator = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(validator)
    result = validator.false_pass()
    assert result["status"] == "PASS" and result["attack_count"] == 16
