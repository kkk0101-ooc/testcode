import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_r8_final_validation():
    path = ROOT / "scripts/validate_nns_vit_patch_depth_s2_r8_publication.py"
    spec = importlib.util.spec_from_file_location("r8_validator", path)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    result = validator.validate()
    assert result["terminal_status"] == "PASS"
    assert result["selected_cap"] == 4
    assert result["next_authorization"] == "S3_REPRESENTATION_AND_PROBE_TRAINING_ONLY"
