from __future__ import annotations

import ast
import copy
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("validate_stage_y_y0", REPO / "scripts/validate_stage_y_y0.py")
assert SPEC and SPEC.loader
y0 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(y0)


def test_strict_json_rejects_duplicates_and_nonfinite() -> None:
    with pytest.raises(y0.StageYY0Error, match="duplicate"):
        y0.strict_json_bytes(b'{"a":1,"a":2}', "fixture")
    for token in (b"NaN", b"Infinity", b"-Infinity"):
        with pytest.raises(y0.StageYY0Error, match="non-finite"):
            y0.strict_json_bytes(b'{"a":' + token + b"}", "fixture")


def test_required_contracts_parse_and_protocol_ids_match() -> None:
    documents = y0.load_contracts(REPO)
    assert len(documents) == 10
    assert all(value["protocol_id"] == y0.PROTOCOL_ID for value in documents.values())
    result = y0.validate_semantics(REPO, documents)
    assert result["status"] == "PASS"
    assert result["next_authorization"] == y0.NEXT_AUTHORIZATION


def test_contract_bundle_is_source_recomputed_and_sensitive() -> None:
    bundle = y0.contract_bundle(REPO)
    assert bundle["bundle_digest"] == y0.canonical_digest(bundle["files"])
    y0.validate_signed(bundle, "bundle")
    changed = copy.deepcopy(bundle["files"]); changed[0]["sha256"] = "0" * 64
    assert y0.canonical_digest(changed) != bundle["bundle_digest"]


def test_historical_facts_are_recomputed_from_sealed_sources() -> None:
    binding = y0._source_binding(REPO)
    assert binding["historical_facts"] == {
        "stage_d_decision": "GO_FIDELITY_SCOPE",
        "stage_e_scientific_validity": "MIXED",
        "c10_decision": "NO_GO_NOVELTY",
        "x2_decision": "NO_GO_BROAD_CLAIM_DEVELOPMENT",
        "x2_mech_m2_terminal": "PASS",
        "x2_mech_m2_interpretation": "MIXED_OR_INCONCLUSIVE",
    }
    assert binding["historical_sources_digest"] == y0.canonical_digest(binding["historical_sources"])
    assert len(binding["historical_sources"]) == 7


def test_six_claims_have_complete_independent_decision_schemas() -> None:
    claims = y0.read_json(REPO / "manifests/stage_y/claim_registry.json")
    rows = claims["claims"]
    assert tuple(row["claim_id"] for row in rows) == y0.CLAIM_IDS
    assert claims["no_omnibus_go"] is True
    for row in rows:
        assert set(row) == y0.REQUIRED_CLAIM_KEYS
        assert row["independent_statistical_unit"] == "image"
        assert all(row[key] for key in y0.REQUIRED_CLAIM_KEYS)


def test_scope_cohort_depth_head_patch_and_target_are_exact() -> None:
    config = y0.read_json(REPO / y0.CONFIG)
    cohort = y0.read_json(REPO / "manifests/stage_y/cohort_selection_and_exclusion_plan.json")
    model = config["model"]
    assert model["blocks"] == list(range(12))
    assert model["heads_per_block"] == 12 and model["block_head_coordinates"] == 144
    assert model["native_patches"] == 196
    assert model["depth_groups"] == {"shallow": [0,1,2,3], "middle": [4,5,6,7], "deep": [8,9,10,11]}
    assert config["target"]["primary"] == "clean predicted-class raw logit"
    assert cohort["dataset"]["count"] == cohort["dataset"]["distinct_true_classes"] == 100
    assert cohort["subsets"]["audit_10"]["algorithm"].startswith("take 10")
    assert cohort["subsets"]["perturbation_20"]["algorithm"].startswith("take 20")
    assert cohort["failure_rules"]["numerical_failure_replacement_allowed"] is False


def test_transform_panel_inverse_masks_and_fixed_target_are_exact() -> None:
    registry = y0.read_json(REPO / "manifests/stage_y/transform_registry.json")
    assert tuple(row["id"] for row in registry["conditions"]) == y0.TRANSFORMS
    assert len(registry["conditions"]) == 9
    for row in registry["conditions"]:
        assert {"forward_map", "inverse_map", "interpolation", "padding_fill", "full_domain"}.issubset(row)
    flip = next(row for row in registry["conditions"] if row["id"] == "horizontal_flip")
    assert flip["full_domain"] is True
    assert registry["target_handling"]["class_index_fixed_across_all_conditions"] is True
    assert registry["target_handling"]["prediction_flip_rows_retained"] is True


def test_complete_cuts_two_residuals_and_reconstruction_are_exact() -> None:
    cuts = y0.read_json(REPO / "manifests/stage_y/component_cut_registry.json")
    first, second = cuts["cuts"][2], cuts["cuts"][3]
    assert first["allocation_pair"] == ["pre_attention_identity", "attention_aggregate"]
    assert second["allocation_pair"] == ["post_attention_residual", "mlp_output"]
    assert len(cuts["reconstruction_contracts"]) == 10
    assert cuts["patch_routing"]["matrix_shape"] == [196,196]
    assert cuts["patch_routing"]["cls_shape"] == [196]
    assert "sum_decompositions_across_blocks" in cuts["forbidden_operations"]
    assert "three_way_identity_attention_mlp_allocation" in cuts["forbidden_operations"]


def test_metrics_statistics_retention_and_budget_have_no_open_choice() -> None:
    metrics = y0.read_json(REPO / "manifests/stage_y/metric_registry.json")
    sap = y0.read_json(REPO / "manifests/stage_y/statistical_analysis_plan.json")
    retention = y0.read_json(REPO / "manifests/stage_y/retention_plan.json")
    pilot = y0.read_json(REPO / "manifests/stage_y/y1_cost_pilot_plan.json")
    primary = metrics["within_patch"]["primary_operator"]
    assert primary["id"] == "hierarchical_positive_deletion_auc_gain"
    assert primary["primary_sign"] == "positive contribution first"
    assert primary["percentiles"] == list(range(0,101,5)) and primary["effect_threshold"] == 0.02
    assert sap["independent_unit"] == "image" and sap["bootstrap"]["replicates"] == 20000
    assert [family["family_id"] for family in sap["families"]] == ["A", "B", "C"]
    assert sap["multiplicity"]["method"] == "Holm step-down" and sap["no_best_selection"] is True
    assert retention["roots"]["scientific"] != retention["roots"]["display"]
    assert pilot["allowed_empirical_image"]["formal_stage_y_eligibility"] is False
    assert pilot["conditions"] == ["clean", "horizontal_flip"]
    assert pilot["pilot_hard_caps"]["h20_gpu_hours"] == 8.0
    assert pilot["projection"]["safety_multiplier"] == 1.5


@pytest.mark.parametrize("attack", y0.FALSE_PASS_ATTACKS)
def test_every_false_pass_attack_is_independently_rejected(attack: str) -> None:
    valid = {name: True for name in y0.FALSE_PASS_ATTACKS}
    assert y0.false_pass_reasons(valid) == ()
    valid[attack] = False
    assert y0.false_pass_reasons(valid) == (attack,)


def test_false_pass_report_has_all_required_attacks() -> None:
    audit = y0.build_false_pass_audit()
    assert audit["status"] == "PASS"
    assert audit["attack_count"] == len(y0.FALSE_PASS_ATTACKS) == 17
    assert {row["attack"] for row in audit["checks"]} == set(y0.FALSE_PASS_ATTACKS)
    assert all(row["result"] == "REJECTED" for row in audit["checks"])


def test_validator_has_no_model_image_checkpoint_or_accelerator_surface() -> None:
    source = (REPO / "scripts/validate_stage_y_y0.py").read_text(encoding="utf-8")
    tree = ast.parse(source); imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import): imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module: imports.add(node.module.split(".")[0])
    assert not imports.intersection(y0.FORBIDDEN_IMPORTS)
    for forbidden in ("nvidia-smi", "cuda.is_available", "Image.open", "torch.load", "read_image"):
        assert forbidden not in source


def test_route_docs_pause_f3_final_and_do_not_rewrite_history() -> None:
    protocol = (REPO / y0.PROTOCOL_DOC).read_text(encoding="utf-8")
    roadmap = (REPO / "docs/阶段路线图.md").read_text(encoding="utf-8")
    stage_f = (REPO / "docs/阶段F规划.md").read_text(encoding="utf-8")
    stage_g = (REPO / "docs/阶段G规划.md").read_text(encoding="utf-8")
    assert "F3_FINAL_PAUSED_PENDING_STAGE_Y" in protocol and "F3_FINAL_PAUSED_PENDING_STAGE_Y" in roadmap and "F3_FINAL_PAUSED_PENDING_STAGE_Y" in stage_f
    assert "Stage Y results-to-paper handoff" in stage_g
    for fact in ("GO_FIDELITY_SCOPE", "MIXED", "NO_GO_NOVELTY", "NO_GO_BROAD_CLAIM_DEVELOPMENT", "MIXED_OR_INCONCLUSIVE"):
        assert fact in protocol


def test_completion_is_atomic_and_bound_if_present() -> None:
    if not (REPO / y0.COMPLETION).exists():
        return
    result = y0.validate_completion(REPO)
    assert result["status"] in {"PASS", "BLOCKED"}
    assert not (REPO / y0.PENDING).exists()
    if result["status"] == "PASS":
        assert result["next_authorization"] == y0.NEXT_AUTHORIZATION


def test_cli_exposes_y0_lifecycle_only() -> None:
    source = (REPO / "scripts/validate_stage_y_y0.py").read_text(encoding="utf-8")
    for command in ("validate-contracts", "init-pending", "false-pass-audit", "freeze", "validate-completion", "validate-publication"):
        assert command in source
    for forbidden in ("run-y1", "run-y2", "run-y3", "run-y4", "run-y5", "load-model"):
        assert forbidden not in source
