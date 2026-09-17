from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_nns_vit_cut_intervention_p0_static",
    REPO / "scripts/validate_nns_vit_cut_intervention_p0_static.py",
)
assert SPEC and SPEC.loader
p0 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p0)


def test_strict_json_rejects_duplicate_nonfinite_and_bad_utf8() -> None:
    with pytest.raises(p0.StaticContractError, match="duplicate"):
        p0.strict_json_bytes(b'{"x":1,"x":2}', "duplicate")
    for token in (b"NaN", b"Infinity", b"-Infinity"):
        with pytest.raises(p0.StaticContractError, match="non-finite"):
            p0.strict_json_bytes(b'{"x":' + token + b"}", "nonfinite")
    with pytest.raises(p0.StaticContractError, match="UTF-8"):
        p0.strict_json_bytes(b'{"x":"\xff"}', "encoding")


def test_required_contracts_and_authority_are_exact() -> None:
    result = p0.validate_contracts(REPO)
    assert result == {
        "status": "PASS_STATIC",
        "contract_count": 10,
        "schema_count": 2,
        "closed_schema_errors": 0,
        "forbidden_imports": [],
        "next_authorization": p0.NEXT_AUTHORIZATION,
    }
    startup = p0.read_json(REPO, p0.ROOT / "startup_gate.json")
    assert startup["planning_authority_commit"] == p0.AUTHORITY_COMMIT
    assert startup["authority_planning_sha256"] == p0.PLANNING_SHA256
    assert startup["local_head"] == startup["tracking_ref"] == startup["remote_branch"]


def test_every_schema_object_is_closed_and_payload_is_not_open() -> None:
    for relative in p0.SCHEMA_PATHS:
        schema = p0.read_json(REPO, relative)
        assert p0._closed_schema_errors(schema) == []
        source = json.dumps(schema, sort_keys=True)
        assert '"payload": {"type": "object"}' not in source


def test_unknown_nested_field_is_rejected() -> None:
    contracts = p0.build_contracts()
    schema = p0.expected_schemas()[p0.CONFIG_DIR / "p0_static.schema.json"]
    value = copy.deepcopy(contracts[p0.MANIFEST_DIR / "preprocessing_registry.json"])
    value["spec"]["unknown"] = True
    with pytest.raises(p0.StaticContractError, match="schema oneOf mismatch|unknown keys"):
        p0.validate_schema_instance(value, schema, schema)


def test_math_edge_differential_and_interaction_contracts_are_exact() -> None:
    cuts = p0.read_json(REPO, p0.MANIFEST_DIR / "cut_intervention_registry.json")
    head = cuts["head_cut"]
    assert head["hook_position"] == "post_attention_dropout_pre_head_concat_pre_output_projection"
    assert head["query_token_outputs"] == 197 and head["heads"] == 12
    assert head["identity_edge_registered_separately"] is True and head["cls_only_is_main"] is False
    first, second = cuts["residual_cuts"]
    assert first["shared_output_adjoint"] == "grad_{h_prime}F"
    assert second["shared_output_adjoint"] == "grad_{h_next}F"
    assert not first["original_node_total_gradient_for_left"] and not second["original_node_total_gradient_for_left"]
    differential = cuts["attention_differential"]
    assert differential["mathematical_type"] == "differential_path_not_complete_edge_cut"
    assert differential["identity"] == "dO=(dA)V+A(dV)"
    assert differential["input_independent_bias_contribution"] == 0.0
    assert differential["bias_retained_in_forward_score"] is True
    assert cuts["interventions"]["interaction_formula"] == "I_RV=e_R+e_V-e_(R+V)"


def test_component_axes_dtype_map_and_scalar_roles_are_exact() -> None:
    cuts = p0.read_json(REPO, p0.MANIFEST_DIR / "cut_intervention_registry.json")
    conductance = cuts["component_conductance"]
    assert conductance["input_axes"] == ["channel", "row", "column"]
    assert conductance["aggregation_dtype"] == "float64"
    assert conductance["retained_map_level"] == "signed_4x4_cell_map_for_every_primary_component"
    scalars = cuts["attribution_scalars"]
    assert scalars["signed_predicts"] == "signed_effect"
    assert scalars["mass_predicts"] == "absolute_effect"
    assert scalars["zero_mass_code"] == "NA_ZERO_CUT_MASS"


def test_target_reference_and_cohort_are_frozen_not_executed() -> None:
    target = p0.read_json(REPO, p0.MANIFEST_DIR / "target_registry.json")
    preprocess = p0.read_json(REPO, p0.MANIFEST_DIR / "preprocessing_registry.json")
    assert target["primary_target"]["formula"] == "F(x)=logit_{y*}(x)"
    assert target["metadata_only_selection"]["executed_in_static_stage"] is False
    assert target["metadata_only_selection"]["cohort_size"] == 20
    assert set(target["membership_forbidden_fields"]) >= {"correctness", "prediction", "logit", "margin", "confidence"}
    spec = preprocess["spec"]
    assert spec == {"color_mode": "RGB", "raw_range": [0.0, 1.0], "resize_shorter_side": 248, "interpolation": "bicubic", "antialias": True, "center_crop": [224, 224], "channel_order": "RGB", "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
    assert preprocess["model_config_source"]["not_a_generic_vit_constant"] is True


def test_statistics_hypotheses_and_inference_are_complete() -> None:
    sap = p0.read_json(REPO, p0.MANIFEST_DIR / "statistical_analysis_plan.json")
    assert [row["id"] for row in sap["hypotheses"]] == [f"H{i}" for i in range(1, 6)] + [f"B{i}" for i in range(1, 11)] + [f"R{i}" for i in range(1, 4)]
    assert sap["independent_unit"] == "image"
    assert sap["aggregation"]["head_rows_are_independent_n"] is False
    assert sap["aggregation"]["head_min_valid_blocks"] == 8
    assert sap["validity"]["minimum_valid_images"] == 18
    assert sap["inference"]["bootstrap_replicates"] == 20000
    assert sap["inference"]["rng"] == "PCG64"
    assert sap["inference"]["sign_flip_max_enumerations"] == 2**20
    assert sap["thresholds"]["spearman_practical_min"] == 0.30
    assert sap["aggregate_decision"]["minimum_supported_families"] == 2
    assert sap["aggregate_decision"]["requires_head_or_routing_value"] is True


def test_baselines_have_no_random_or_per_image_oracle() -> None:
    registry = p0.read_json(REPO, p0.MANIFEST_DIR / "baseline_registry.json")
    assert registry["selection"]["per_image_oracle_allowed"] is False
    assert registry["selection"]["random_control_is_mandatory"] is False
    for family in registry["families"]:
        assert not any("random" in item for item in family["mandatory"])
        assert family["descriptive_sanity"]


def test_three_numerical_contracts_are_distinct_and_exact() -> None:
    cuts = p0.read_json(REPO, p0.MANIFEST_DIR / "cut_intervention_registry.json")
    numeric = cuts["numerical_contracts"]
    assert set(numeric) >= {"ALGEBRAIC_CUT_RECONSTRUCTION", "IG_COMPLETENESS_TO_TARGET_DELTA", "QUADRATURE_SUCCESSIVE_ORDER_CONVERGENCE"}
    assert numeric["quadrature"]["schedule"] == [128, 256, 512, 1024, 2048, 4096, 8192]
    assert numeric["ALGEBRAIC_CUT_RECONSTRUCTION"] == {"pixel_abs": 2e-5, "pixel_rel": 0.005, "total_abs": 2e-4, "total_rel": 0.01, "closures": ["heads_to_attention", "identity_plus_attention_to_first_residual", "carry_plus_mlp_to_second_residual", "value_plus_routing_to_attention", "query_plus_key_plus_bias_to_routing"]}
    assert numeric["IG_COMPLETENESS_TO_TARGET_DELTA"]["small_target_code"] == "NA_SMALL_TARGET_DELTA"
    assert numeric["QUADRATURE_SUCCESSIVE_ORDER_CONVERGENCE"]["map_l1_rel"] == 0.005
    assert numeric["low_order_algebraic_pass_implies_convergence"] is False


def test_p0_c_plan_has_caps_and_no_fabricated_projection() -> None:
    plan = p0.read_json(REPO, p0.MANIFEST_DIR / "p0_c_cost_pilot_plan.json")
    assert plan["status"] == "FROZEN_NOT_EXECUTED"
    assert plan["projection"]["formula"] == "P95_measured_cost*planned_calls*20_images*1.5"
    assert plan["projection"]["measured_projection"] is None
    assert plan["owner_ceilings"] == {"aggregate_h20_gpu_hours_max": 80.0, "eight_gpu_wall_hours_max": 24.0, "cpu_hours_max": 500.0, "peak_reserved_gpu_gib_per_process_max": 32.0, "fleet_concurrent_reserved_gpu_gib_max": 256.0, "retained_storage_gib_max": 32.0, "temporary_storage_gib_max": 64.0, "devices": 8}


@pytest.mark.parametrize("attack_id,description,expected", p0.FALSE_PASS_ATTACKS)
def test_each_false_pass_attack_is_independently_rejected(attack_id: str, description: str, expected: str) -> None:
    del description
    state = p0.base_attack_state()
    assert p0.attack_reasons(state) == ()
    p0.mutate_attack(state, attack_id)
    assert p0.attack_reasons(state) == (expected,)


def test_false_pass_audit_rejects_all_28() -> None:
    audit = p0.build_false_pass_audit()
    assert audit["status"] == "PASS"
    assert audit["attack_count"] == audit["rejected_count"] == 28
    assert all(row["result"] == "REJECTED" for row in audit["checks"])


def test_validator_and_tests_have_no_forbidden_imports_or_execution_surface() -> None:
    for relative in ("scripts/validate_nns_vit_cut_intervention_p0_static.py", "tests/test_nns_vit_cut_intervention_p0_static.py"):
        source = (REPO / relative).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        assert not imports.intersection(p0.FORBIDDEN_IMPORTS)
        for forbidden in ("Image" + ".open", "torch" + ".load", "np" + ".load", "numpy" + ".load", "nvidia" + "-smi", "cuda" + ".is_available"):
            assert forbidden not in source


def test_symlink_and_root_escape_are_rejected(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    (repository / "link.json").symlink_to(outside)
    with pytest.raises(p0.StaticContractError, match="symlink"):
        p0._safe_path(repository, Path("link.json"))
    with pytest.raises(p0.StaticContractError, match="root escape"):
        p0._safe_path(repository, Path("../outside.json"))


def test_lifecycle_exposes_only_static_commands_and_completion_is_scoped() -> None:
    source = (REPO / "scripts/validate_nns_vit_cut_intervention_p0_static.py").read_text(encoding="utf-8")
    for command in ("init", "draft", "false-pass-audit", "final", "seal", "validate-seal", "validate-publication"):
        assert f'"{command}"' in source
    for forbidden in ("run-p0-c", "run-p1", "run-p2", "run-p3", "run-p4"):
        assert forbidden not in source
    if (REPO / p0.COMPLETION).exists():
        result = p0.validate_seal(REPO)
        assert result["status"] == "PASS_STATIC"
        assert result["p0_terminal_pass"] is False
        assert result["next_authorization"] == p0.NEXT_AUTHORIZATION
