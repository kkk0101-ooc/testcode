from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/nns_vit_endpoint/endpoint_primary_track_amendment_r1.json"
VALIDATOR = ROOT / "scripts/validate_endpoint_primary_track_amendment.py"


def load_validator():
    spec = importlib.util.spec_from_file_location("endpoint_scope_validator", VALIDATOR)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_scope_split_preserves_path_failure_and_blocks_original_ep3():
    value = json.loads(CONFIG.read_text())
    assert value["status"] == {
        "endpoint_primary_track": "ENDPOINT_PRIMARY_TRACK_AUTHORIZED_BY_AMENDMENT",
        "path_dependent_track": "PATH_DEPENDENT_TRACK_BLOCKED",
        "ep2_r1": "EP2_R1_PATH_NUMERICAL_CALIBRATION_FAILED_REMAINS_IMMUTABLE",
        "original_ep3": "NOT_AUTHORIZED",
        "new_scientific_outcome_access": False,
    }
    assert value["access"]["next_authorization"] == "ENDPOINT_E2_ATLAS_REQUIRES_EXPLICIT_USER_REQUEST"
    assert value["access"]["path_dependent_authorization"] == "NONE"


def test_endpoint_e2_scope_excludes_path_and_utility_work():
    scope = json.loads(CONFIG.read_text())["endpoint_e2_scope"]
    assert scope["models"] == ["vit_b16", "deit_s16"]
    assert scope["methods_allowed"] == ["canonical_endpoint_x_times_gradient"]
    assert scope["claim_decisions_allowed"] == ["C2_REGISTERED_SCAN_COMPLETION_AND_RESOURCES"]
    assert not scope["path_ig_allowed"]
    assert not scope["input_fidelity_or_stability_allowed"]
    assert not scope["finite_intervention_allowed"]
    assert not scope["scientific_utility_claims_allowed"]


def test_amendment_closed_schema_and_semantics():
    validator = load_validator(); value = validator.load(CONFIG)
    validator.closed(value, validator.load(ROOT / "config/nns_vit_endpoint/endpoint_primary_track_amendment_r1.schema.json"))
    validator.semantic(value)
