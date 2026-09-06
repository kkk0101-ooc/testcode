"""E0 contract, golden, gate, and false-PASS tests."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

import vitnns.stage_e_e0 as e0
from vitnns.stage_d_provenance import canonical_sha256

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resign(value: dict) -> dict:
    value["canonical_digest"] = canonical_sha256({
        key: item for key, item in value.items()
        if key != "canonical_digest"
    })
    return value


def test_contracts_schema_typed_plan_and_output_columns_are_frozen() -> None:
    plan = e0.validate_e0_inputs(REPO_ROOT)
    config, _ = e0.load_stage_e_contract(REPO_ROOT)
    spec, _ = e0.load_stage_e_run_spec(REPO_ROOT)
    assert plan.sample_count == 500
    assert plan.strata == e0.EXPECTED_STRATA
    assert plan.coordinates == e0.EXPECTED_COORDINATES
    assert plan.e2_variants == e0.EXPECTED_E2_VARIANTS
    assert len(plan.variants) == 13
    assert [row["stage_id"] for row in spec["stages"]] == [
        "E1", "E2", "E3", "E4",
    ]
    assert config["outputs"]["numeric_source"] == "paper_data_only"
    assert config["outputs"]["renders_png"] is False
    output_schema = json.loads(
        (REPO_ROOT / "config/stage_e_output_table.schema.json").read_text()
    )
    Draft202012Validator.check_schema(output_schema)
    required = set(output_schema["$defs"]["row"]["required"])
    assert required == set(config["outputs"]["required_common_columns"])


def test_registry_order_and_prediction_strata_partition() -> None:
    value = e0.validate_registry_and_strata(REPO_ROOT)
    assert value["sample_count"] == 500
    assert value["overall"] == value["correct"] + value["misclassified"]
    assert (value["correct"], value["misclassified"]) == (418, 82)


def test_stage_c_coordinates_are_exact_and_pre_holdout() -> None:
    value = e0.validate_stage_c_coordinates(REPO_ROOT)
    assert value["coordinates"] == e0.EXPECTED_COORDINATES
    assert value["primary"] == (0, 9)
    assert value["holdout_accessed"] is False


def test_e2_coordinate_conservation_and_equivalence_golden() -> None:
    result = e0.validate_e2_golden()
    assert result["status"] == "PASS"
    assert result["uniform_nearest_operator_equal"] is True
    assert result["uniform_nearest_ranking_equal"] is True
    assert max(result["basis_conservation_max_abs_errors"].values()) <= 1e-12
    fixture = np.array([
        [(-1.0 if (row + col) % 3 == 0 else row * 14 + col + 0.25)
         for col in range(14)] for row in range(14)
    ], dtype=np.float64)
    uniform = e0.apply_upsampling(fixture, "uniform_mass_divide_by_16")
    nearest = e0.apply_upsampling(fixture, "mass_preserving_nearest")
    assert np.array_equal(uniform, nearest)
    assert np.array_equal(e0.ranking(uniform, 9), e0.ranking(nearest, 9))


def test_missing_upstream_binding_cannot_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    real = e0._git
    monkeypatch.setattr(
        e0, "_git",
        lambda repo, *args: "0" * 40
        if args[:2] == ("rev-parse", "refs/archive/stage-d-fidelity-v1-d10-completion^{commit}")
        else real(repo, *args),
    )
    with pytest.raises(e0.StageEE0Error, match="completion ref mismatch"):
        e0.verify_upstream(REPO_ROOT, d9_source_control={})


def test_missing_reuse_variant_and_untyped_not_run_cannot_pass() -> None:
    original = json.loads((REPO_ROOT / e0.REUSE_PATH).read_text())
    missing = copy.deepcopy(original)
    missing["items"] = missing["items"][:-1]
    _resign(missing)
    with pytest.raises(e0.StageEE0Error, match="variants are missing"):
        e0.validate_reuse_inventory_document(missing, REPO_ROOT)
    no_reason = copy.deepcopy(original)
    row = next(item for item in no_reason["items"] if item["status"] == "NOT_RUN_WITH_REASON")
    row["reason"] = ""
    _resign(no_reason)
    with pytest.raises(e0.StageEE0Error):
        e0.validate_reuse_inventory_document(no_reason, REPO_ROOT)


@pytest.mark.parametrize("mutation", [
    lambda value: value["firewall"].update(
        {"stage_e_new_outcome_accessed": True}
    ),
    lambda value: value["protocol"].update({"renders_png": True}),
    lambda value: value["protocol"].update(
        {"numeric_source": "analysis_memory"}
    ),
    lambda value: value["analyses"]["e3"]["coordinates"].append([3, 3]),
    lambda value: value["budget"]["e0"].update({"gpu_hours_max": 1.0}),
    lambda value: value["stage_gates"]["e1"].update(
        {"authorizes_on_pass": "E3"}
    ),
])
def test_outcome_access_render_scope_or_budget_drift_cannot_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation,
) -> None:
    source = json.loads((REPO_ROOT / e0.CONFIG_PATH).read_text())
    mutation(source)
    _resign(source)
    target = tmp_path / "stage_e_e0.json"
    target.write_text(json.dumps(source, sort_keys=True, separators=(",", ":")) + "\n")
    monkeypatch.setattr(e0, "CONFIG_PATH", str(target))
    with pytest.raises(e0.StageEE0Error):
        e0.load_stage_e_contract(REPO_ROOT)


def test_protected_d9_d10_trees_have_no_tracked_diff() -> None:
    result = e0.verify_upstream(
        REPO_ROOT,
        d9_source_control={
            "status": "PASS", "analysis_authorized": True,
            "root_inventory": {
                "canonical_digest": e0.EXPECTED_D9_INVENTORY,
            },
        },
    )
    assert result["protected_tracked_diff"] == []
    assert result["d10_output_root_inventory"]["canonical_digest"] == (
        e0.EXPECTED_D10_INVENTORY
    )


def test_completion_schema_rejects_empty_or_placeholder_pass() -> None:
    schema = json.loads(
        (REPO_ROOT / "config/stage_e_e0_completion.schema.json").read_text()
    )
    Draft202012Validator.check_schema(schema)
    Draft202012Validator.check_schema(schema)
    errors = list(Draft202012Validator(schema).iter_errors({
        "schema_version": 1, "document_type": "stage_e_e0_completion",
        "stage": "E0", "status": "PASS",
    }))
    assert errors


def test_figure_recipe_schema_requires_nonrendered_paper_data() -> None:
    schema = json.loads(
        (REPO_ROOT / "config/stage_e_figure_recipes.schema.json").read_text()
    )
    Draft202012Validator.check_schema(schema)
    valid = {
        "schema_version": 1, "document_type": "stage_e_figure_recipes",
        "numeric_source": "paper_data_only", "renders_png": False,
        "figures": [{
            "figure_id": "synthetic", "table_path": "paper_data/x.json",
            "x": "variant_id", "y": "median", "group": "stratum",
            "caption_contract": "synthetic schema fixture only",
            "numeric_source": "paper_data_only", "renders_png": False,
        }],
        "canonical_digest": "0" * 64,
    }
    assert not list(Draft202012Validator(schema).iter_errors(valid))
    invalid = copy.deepcopy(valid)
    invalid["renders_png"] = True
    assert list(Draft202012Validator(schema).iter_errors(invalid))


def test_e4_package_schema_is_valid() -> None:
    schema = json.loads(
        (REPO_ROOT / "config/stage_e_package.schema.json").read_text()
    )
    Draft202012Validator.check_schema(schema)
    completion_schema = json.loads(
        (REPO_ROOT / "config/stage_e_stage_completion.schema.json").read_text()
    )
    Draft202012Validator.check_schema(completion_schema)


def test_false_pass_guard_rejects_each_missing_precondition() -> None:
    valid = {name: True for name in e0.FALSE_PASS_PRECONDITIONS}
    assert e0.false_pass_reasons(valid) == ()
    for name in e0.FALSE_PASS_PRECONDITIONS:
        mutated = dict(valid)
        mutated[name] = False
        assert e0.false_pass_reasons(mutated) == (name,)


def test_completion_publication_is_pending_then_terminal_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(e0, "PENDING_PATH", "completion/e0.pending.json")
    monkeypatch.setattr(e0, "COMPLETION_PATH", "completion/e0.json")
    monkeypatch.setattr(
        e0, "validate_completion_document",
        lambda document, repo_root: document,
    )
    document = {"status": "PASS"}
    target = e0.publish_completion(document, tmp_path)
    assert target == tmp_path / "completion/e0.json"
    assert json.loads(target.read_text()) == document
    assert not (tmp_path / "completion/e0.pending.json").exists()
    with pytest.raises(e0.StageEE0Error, match="refusing to overwrite"):
        e0.publish_completion(document, tmp_path)


def test_no_new_stage_e_outcome_or_render_artifact_exists() -> None:
    root = REPO_ROOT / "evidence/stage_e/e0_recovery_v2_20260906"
    forbidden_suffixes = {".png", ".pdf", ".svg"}
    assert not any(
        path.suffix.lower() in forbidden_suffixes for path in root.rglob("*")
        if path.is_file()
    )
    assert not (root / "paper_data").exists()
    assert not any((root / name).exists() for name in ("e1", "e2", "e3", "e4"))
