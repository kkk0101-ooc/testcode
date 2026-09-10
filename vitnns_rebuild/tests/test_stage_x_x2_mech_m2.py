from __future__ import annotations

import ast
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from vitnns import stage_x_x2_mech_analysis as analysis
from vitnns import stage_x_x2_mech_m2 as m2

REPO = Path(__file__).resolve().parents[1]


def synthetic_map(size: int = 224) -> np.ndarray:
    row, col = np.indices((size, size), dtype=np.float64)
    return (np.sin(row / 13.0) - np.cos(col / 19.0) + row / 1000.0).astype(np.float32)


def test_config_schema_digest_and_claim_boundaries() -> None:
    config = m2.load_config(REPO)
    schema = m2.strict_read(REPO / m2.CONFIG_SCHEMA)
    Draft202012Validator.check_schema(schema)
    assert not list(Draft202012Validator(schema).iter_errors(config))
    assert config["study_type"] == "POST_HOC_EXPLORATORY_MECHANISM_ANALYSIS"
    assert config["upstream_bindings"]["x2_decision"] == "NO_GO_BROAD_CLAIM_DEVELOPMENT"
    assert config["branch_rules"]["fallback_without_unique_thresholds"] == "MIXED_OR_INCONCLUSIVE"
    assert all(value is False for key, value in config["claims"].items() if key != "maximum_wording")
    mutated = copy.deepcopy(config)
    mutated["claims"]["spatial_stability_pass_allowed"] = True
    assert list(Draft202012Validator(schema).iter_errors(mutated))


def test_artifact_schema_allows_producer_and_completion_next_authorizations() -> None:
    schema = m2.strict_read(REPO / m2.ARTIFACT_SCHEMA)
    Draft202012Validator.check_schema(schema)
    for next_auth in (
        "X2_MECH_M2_CPU_ONLY_MULTISCALE_MECHANISM_ANALYSIS_ONLY",
        "X2_MECH_RESULTS_TO_PAPER_HANDOFF_ONLY",
    ):
        doc = m2.signed(
            "stage_x_x2_mech_m2_test",
            "PASS",
            m2._payload(terminal=m2._terminal(terminal=True, complete=True, next_authorization=next_auth)),
        )
        m2.validate_signed(doc, "synthetic", schema)


def test_output_schema_constants_match_config() -> None:
    config = m2.load_config(REPO)
    for name, fields in m2.REQUIRED_CSV_FIELDS.items():
        assert tuple(config["output_schemas"][name]) == tuple(fields)
    assert tuple(config["cohort"]["methods"]) == m2.METHODS
    assert tuple(config["cohort"]["transforms"]) == m2.TRANSFORMS


def test_strict_json_and_signed_digest_reject_mutation(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"a": 1, "a": 2}', encoding="utf-8")
    with pytest.raises(m2.X2MechM2Error):
        m2.strict_read(path)
    doc = m2.signed("stage_x_x2_mech_m2_test", "PASS", m2._payload(counts=m2._as_list({"a": 1})))
    m2.validate_signed(doc, "synthetic", m2._schema(REPO))
    doc["payload"]["counts"][0]["value"] = 2
    with pytest.raises(m2.X2MechM2Error, match="self-digest"):
        m2.validate_signed(doc, "synthetic", m2._schema(REPO))


def test_complete_block_aggregation_and_partial_exclusion() -> None:
    value = np.arange(64, dtype=np.float32).reshape(8, 8) - 20
    mask = np.ones((8, 8), dtype=bool)
    mask[0, 0] = False
    result = analysis.aggregate_blocks(value, mask, 4)
    assert not result["valid_blocks"][0, 0]
    assert result["valid_blocks"].sum() == 3
    assert np.isnan(result["signed_sum"][0, 0])
    assert result["signed_sum"][1, 1] == pytest.approx(value[4:8, 4:8].sum(dtype=np.float64))
    assert result["absolute_mass_sum"][1, 1] == pytest.approx(np.abs(value[4:8, 4:8]).sum(dtype=np.float64))


def test_top10_count_rule_is_ceil_not_floor() -> None:
    values = np.arange(95, dtype=np.float64).reshape(5, 19)
    valid = np.ones_like(values, dtype=bool)
    indices = analysis.top_fraction_indices(values, valid, 0.1)
    assert len(indices) == 10


def test_sse_identity_and_residual_semantics() -> None:
    clean = synthetic_map()
    transformed = clean.copy()
    transformed[:16, :16] += np.linspace(-1.0, 1.0, 256, dtype=np.float32).reshape(16, 16)
    mask = np.ones(clean.shape, dtype=bool)
    mask[7, 7] = False
    result = analysis.sse_decomposition(clean, transformed, mask, 16)
    assert result["sse_total"] == pytest.approx(result["sse_between"] + result["sse_within"], abs=1e-10, rel=1e-10)
    assert result["excluded_partial_block_count"] == 1
    assert "block_level_signed_rank_stability" in result
    assert "within_block_demeaned_residual_stability" in result


def test_transport_integer_translation_and_horizontal_flip_full_domain() -> None:
    value = synthetic_map()
    access = {"torch_imported_for_cpu_geometry": False}
    flipped, flip_mask = m2._transport_clean(value, "horizontal_flip", {"inverse": "horizontal_flip"}, access)
    assert np.array_equal(flipped, value)
    assert flip_mask.all()
    shifted, mask = m2._transport_clean(value, "translate_x_4", {"inverse": "translate", "axis": "x", "amount": 4}, access)
    assert np.array_equal(shifted[mask], value[mask])
    assert mask.sum() == 224 * 220
    assert access["torch_imported_for_cpu_geometry"] is False


def test_stability_metrics_include_signed_and_absolute_mass_views() -> None:
    clean = synthetic_map()
    transformed = clean + np.roll(clean, 2, axis=1).astype(np.float32) * np.float32(0.02)
    mask = np.ones(clean.shape, dtype=bool)
    for _, block in m2.SCALES:
        result = analysis.stability_metrics(clean, transformed, mask, block)
        assert set(result) == {
            "signed_spearman",
            "top10_unit_signed_value_jaccard",
            "absolute_mass_spearman",
            "top10_unit_absolute_mass_jaccard",
            "signed_ssim",
            "normalized_positive_negative_mass_displacement",
            "absolute_mass_center_of_mass_pixel_error",
            "valid_unit_count",
            "excluded_partial_unit_count",
        }
        assert all(np.isfinite(value) for value in result.values())


def test_association_bootstrap_is_synchronized_deterministic() -> None:
    records = []
    for method in ("nns_v2",):
        for model in ("m1", "m2"):
            for index in range(8):
                records.append({
                    "model_id": model,
                    "sample_id": f"s{index}",
                    "method_id": method,
                    "x": float(index),
                    "y": float(index + (0 if model == "m1" else 1)),
                })
    one = m2._assoc(records, "x", "y", draws=64, max_attempts=1000, namespace="TEST")
    two = m2._assoc(records, "x", "y", draws=64, max_attempts=1000, namespace="TEST")
    assert one == two
    assert one["effect"] == pytest.approx(1.0)
    assert one["confirmatory_p_value"] is None


def test_branch_falls_back_without_frozen_thresholds() -> None:
    stability = []
    for method in m2.METHODS:
        for scale, value in (("pixel", 0.1), ("fine_cell", 0.2), ("native_patch", 0.3)):
            stability.append({
                "group": "overall",
                "model_id": "ALL",
                "method_id": method,
                "mask_id": "x2_frozen_common_domain",
                "scale_id": scale,
                "block": {"pixel": 1, "fine_cell": 4, "native_patch": 16}[scale],
                "transform": "ALL",
                "n": 1,
                "failed_denominator_count": 1,
                "median_signed_spearman": value,
                "median_top10_unit_signed_value_jaccard": value,
                "median_absolute_mass_spearman": value,
                "median_top10_unit_absolute_mass_jaccard": value,
                "median_signed_ssim": value,
                "median_normalized_positive_negative_mass_displacement": value,
                "median_absolute_mass_center_of_mass_pixel_error": value,
                "median_valid_unit_count": 1,
                "sum_excluded_partial_unit_count": 0,
            })
    fidelity = []
    for method in m2.METHODS:
        for scale, block, value in (("fine_cell", 4, 0.5), ("native_patch", 16, 0.2)):
            fidelity.append({
                "group": "overall",
                "model_id": "ALL",
                "method_id": method,
                "scale_id": scale,
                "block": block,
                "n": 1,
                "failed_denominator_count": 1,
                "median_road_combined": value,
                "mean_road_combined": value,
                "median_road_morf": 0.0,
                "median_road_lerf": 1.0,
                "source": "synthetic",
            })
    within = [
        {
            "group": "overall",
            "model_id": "ALL",
            "method_id": "nns_v2",
            "mask_id": "x2_frozen_common_domain",
            "scale_id": scale,
            "block": block,
            "transform": "ALL",
            "n": 1,
            "failed_denominator_count": 1,
            "median_sse_total": 1.0,
            "median_between_error_fraction": 0.4,
            "median_within_error_fraction": 0.6,
            "max_identity_abs_diff": 0.0,
            "max_identity_rel_diff": 0.0,
            "median_block_level_signed_rank_stability": 0.1,
            "median_within_block_demeaned_residual_stability": 0.2,
            "median_valid_block_count": 1,
            "sum_excluded_partial_block_count": 0,
        }
        for scale, block in (("fine_cell", 4), ("native_patch", 16))
    ]
    transport = [{"transform": "horizontal_flip", "mask_id": "x2_frozen_common_domain", "signed_spearman": 1.0}]
    _, branch, interpretation = m2._branch_outputs(stability, fidelity, within, transport, m2.load_config(REPO))
    assert branch["mechanism_branch"] == "MIXED_OR_INCONCLUSIVE"
    assert interpretation["x2_decision"] == "NO_GO_BROAD_CLAIM_DEVELOPMENT"
    assert "No spatial-stability PASS is issued." in interpretation["boundaries"]


def test_atomic_publish_rejects_existing_completion(tmp_path: Path) -> None:
    pending = tmp_path / "pending.json"
    terminal = tmp_path / "terminal.json"
    doc = m2.signed("stage_x_x2_mech_m2_test", "PASS", m2._payload())
    m2._publish_atomic(pending, terminal, doc)
    assert terminal.exists()
    with pytest.raises(m2.X2MechM2Error):
        m2._publish_atomic(pending, terminal, doc)


def test_false_pass_attack_roster_covers_required_items() -> None:
    required = {
        "source_binding_mutation",
        "m1_completion_ref_drift",
        "m1_denominator_mutation",
        "m1_map_manifest_row_drop",
        "x2_pixel_numeric_drift",
        "partial_block_inclusion",
        "top10_count_floor_instead_of_ceil",
        "sse_identity_violation",
        "branch_threshold_invention",
        "confirmatory_p_value_insertion",
        "gpu_or_cuda_access",
        "model_checkpoint_or_jpeg_access",
        "png_or_jpeg_output",
        "atomic_completion_overwrite",
        "x3_x5_f3_final_stage_g_start",
    }
    assert required <= set(m2.FALSE_PASS_ATTACKS)
    assert all(m2._attack_mechanism(item) for item in m2.FALSE_PASS_ATTACKS)


def test_m2_module_forbidden_import_surface_and_clean_import() -> None:
    forbidden = {"timm", "PIL", "cv2"}
    tree = ast.parse((REPO / "src/vitnns/stage_x_x2_mech_m2.py").read_text(encoding="utf-8"))
    imports = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert not imports.intersection(forbidden)
    code = (
        "import json,sys; import vitnns.stage_x_x2_mech_m2; "
        "print(json.dumps({k:any(x==k or x.startswith(k+'.') for x in sys.modules) "
        "for k in ['timm','PIL']}))"
    )
    env = {**dict(os.environ), "PYTHONPATH": str(REPO / "src")}
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, text=True, capture_output=True, check=True)
    assert json.loads(result.stdout) == {"timm": False, "PIL": False}


def test_validate_config_cli_does_not_read_arrays_or_outcome_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("np.load reached")))
    config = m2.load_config(REPO)
    assert config["authorization"] == "X2_MECH_M2_CPU_ONLY_MULTISCALE_MECHANISM_ANALYSIS_ONLY"
