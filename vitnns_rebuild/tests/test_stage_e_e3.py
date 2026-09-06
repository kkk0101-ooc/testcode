"""E3 frozen scope, computation, validation, and false-PASS tests."""

from __future__ import annotations

import copy
import inspect
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

import vitnns.stage_e_e3 as e3
from vitnns.stage_d_provenance import canonical_sha256

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_closed_config_schema_scope_budget_and_stage_boundary() -> None:
    config = e3._validate_config(REPO_ROOT)
    assert config["canonical_digest"] == e3.EXPECTED_CONFIG_DIGEST
    assert tuple((row["block"], row["head"]) for row in config["scope"]["coordinates"]) == e3.COORDINATES
    assert tuple(config["scope"]["methods"]) == e3.METHODS
    assert config["scope"]["stratum_counts"] == {"overall": 500, "correct": 418, "misclassified": 82}
    assert config["scope"]["expected_new_units"] == 3000
    assert config["budget"] == {"gpu_hours_max": 8.0, "cpu_hours_max": 16.0, "retained_bytes_max": 12884901888, "e2_waiver_applies": False}
    assert config["authorization"]["on_pass"] == "E4_ONLY"
    assert config["authorization"]["e4_started"] is False
    assert config["authorization"]["stage_f_started"] is False


def test_typed_plan_exact_registry_coordinates_methods_and_strata() -> None:
    config = e3._validate_config(REPO_ROOT); plan = e3._build_plan(REPO_ROOT, config)
    assert len(plan["groups"]) == 500 and len(plan["units"]) == 3000
    assert [row["selection_order"] for row in plan["groups"]] == list(range(500))
    assert len({row["sample_id"] for row in plan["groups"]}) == 500
    assert Counter(row["stratum"] for row in plan["groups"]) == Counter({"correct": 418, "misclassified": 82})
    assert Counter((row["block"], row["head"], row["method"]) for row in plan["units"]) == Counter({(block, head, method): 500 for block, head in e3.NEW_COORDINATES for method in e3.METHODS})
    assert all(group["primary_anchor"]["mode"] == "PRIMARY_ANCHOR_READ_ONLY" for group in plan["groups"])


def test_primary_has_no_e3_attempt_or_execution_surface() -> None:
    source = inspect.getsource(e3.run_shard)
    assert "NEW_COORDINATES" in source
    assert (0, 9) not in e3.NEW_COORDINATES
    with pytest.raises(e3.StageEE3Error): e3.comparison_identity(0, 9)


def test_seed_is_coordinate_comparison_level_and_method_direction_independent() -> None:
    sample = "ILSVRC2012_val/example.JPEG"
    for coordinate in e3.NEW_COORDINATES:
        seeds = [e3.e3_seed(sample, *coordinate, repeat) for repeat in range(5)]
        assert len(set(seeds)) == 5
        assert all(np.array_equal(e3.tie_priority(seed), e3.tie_priority(seed)) for seed in seeds)
    assert e3.e3_seed(sample, 4, 2, 0) != e3.e3_seed(sample, 6, 10, 0)


def test_fine_cell_counts_ranking_positive_part_and_auc_golden() -> None:
    assert e3.UNIT_COUNTS == (0,157,314,471,628,784,941,1098,1255,1412,1568,1725,1882,2039,2196,2352,2509,2666,2823,2980,3136)
    assert e3.ACTUAL_FRACTIONS == tuple(value / 3136 for value in e3.UNIT_COUNTS)
    signed = np.zeros((56, 56), dtype=np.float32); signed[0, 0] = -10; signed[0, 1] = 2
    ranked = e3.ranking(signed, sigma=1, seed=7)
    assert ranked[0] == 1 and ranked[-1] != 1
    assert e3.trapezoid_auc(e3.ACTUAL_FRACTIONS) == pytest.approx(0.5)


def test_delta_fid_synthetic_golden() -> None:
    nns = {"insertion_auc_mean": 0.8, "deletion_auc_mean": 0.2}
    matched = {"insertion_auc_mean": 0.6, "deletion_auc_mean": 0.4}
    assert e3.delta_fid(nns, matched) == pytest.approx(0.2)


def test_bootstrap_is_deterministic_image_level_10000_draws() -> None:
    values = np.asarray([1.0, 2.0, 3.0, 4.0])
    left = e3._bootstrap(values, "fixture", "correct"); right = e3._bootstrap(values, "fixture", "correct")
    assert left == right and left["resamples"] == 10000 and left["quantile_method"] == "linear"
    assert e3._median(np.asarray([1.0, 2.0, 3.0, 4.0])) == 2.5


@pytest.mark.parametrize("mutation", [
    lambda value: value["scope"]["coordinates"].pop(),
    lambda value: value["scope"].update({"expected_new_units": 2999}),
    lambda value: value["scientific_contract"].update({"unit_count": 196}),
    lambda value: value["scientific_contract"].update({"tie_seed_namespace": "stage-d-fidelity|20260821|"}),
    lambda value: value["scientific_contract"].update({"p_values_emitted": True}),
    lambda value: value["scientific_contract"].update({"clean_margin": {"status": "RUN", "reason": ""}}),
    lambda value: value["budget"].update({"gpu_hours_max": 9.0}),
    lambda value: value["budget"].update({"e2_waiver_applies": True}),
    lambda value: value["authorization"].update({"on_pass": "F0_ONLY"}),
])
def test_scope_seed_decision_margin_or_budget_drift_cannot_validate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation) -> None:
    source = json.loads((REPO_ROOT / e3.CONFIG_PATH).read_text()); mutation(source)
    source["canonical_digest"] = canonical_sha256({key: value for key, value in source.items() if key != "canonical_digest"})
    path = tmp_path / "stage_e_e3.json"; path.write_text(json.dumps(source) + "\n")
    monkeypatch.setattr(e3, "CONFIG_PATH", str(path))
    with pytest.raises(e3.StageEE3Error): e3._validate_config(REPO_ROOT)


def test_budget_gate_is_independent_of_e2_waiver() -> None:
    assert e3.budget_reasons({"gpu_hours": 8.000001, "cpu_seconds": 0}, 0) == ("gpu_hours",)
    assert e3.budget_reasons({"gpu_hours": 0, "cpu_seconds": 16 * 3600 + 1}, 0) == ("cpu_hours",)
    assert e3.budget_reasons({"gpu_hours": 0, "cpu_seconds": 0}, 12884901889) == ("retained_bytes",)


def test_attempt_missing_placeholder_nonfinite_or_wrong_geometry_cannot_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(e3, "ROOT_PATH", "formal")
    unit = {"canonical_digest": "a" * 64, "sample_id": "x", "selection_order": 0, "block": 4, "head": 2, "method": e3.NNS_METHOD, "stratum": "correct", "image_sha256": "b" * 64, "target": "fixed_clean_attention_cls_head_direction", "scale": "fine_4x4_cell", "coordinate_comparison_identity": e3.comparison_identity(4, 2)}
    with pytest.raises(e3.StageEE3Error, match="absent or incomplete"): e3._validate_attempt(tmp_path, unit)
    with pytest.raises(e3.StageEE3Error): e3.ranking(np.zeros((14, 14), dtype=np.float32), sigma=1, seed=0)
    with pytest.raises(e3.StageEE3Error): e3.trapezoid_auc([math.nan] * 21)


def test_false_pass_guard_rejects_every_attack() -> None:
    valid = {name: True for name in e3.FALSE_PASS_ATTACKS}; assert e3.false_pass_reasons(valid) == ()
    for name in e3.FALSE_PASS_ATTACKS:
        mutated = dict(valid); mutated[name] = False; assert e3.false_pass_reasons(mutated) == (name,)


def test_output_schemas_and_render_boundary() -> None:
    for name in (e3.CONFIG_SCHEMA_PATH, e3.OUTPUT_SCHEMA_PATH, "config/stage_e_stage_completion.schema.json"):
        Draft202012Validator.check_schema(json.loads((REPO_ROOT / name).read_text()))
    config = e3._validate_config(REPO_ROOT)
    assert all(config["execution"][key] is False for key in ("renders_png", "renders_pdf", "renders_svg", "manual_case_selection"))
    assert not any(path.suffix.lower() in e3.FORBIDDEN_SUFFIXES for path in (REPO_ROOT / e3.ROOT_PATH).rglob("*") if path.is_file())


def test_completion_if_present_is_atomic_and_fail_closed_or_authorizes_only_e4() -> None:
    path = REPO_ROOT / e3.COMPLETION_PATH
    if not path.exists(): pytest.skip("E3 terminal completion not yet available")
    result = e3.validate_completion(REPO_ROOT)
    assert result["status"] in {"PASS", "FAIL"}
    assert result["next_authorization"] == ("E4_ONLY" if result["status"] == "PASS" else "NONE")
    assert result["e4_started"] is False
    assert not (REPO_ROOT / e3.COMPLETION_PENDING_PATH).exists()


def test_no_e4_or_stage_f_execution_surface() -> None:
    public = {name for name in dir(e3) if name.startswith(("run_", "start_", "execute_"))}
    assert public == {"run_shard"}
