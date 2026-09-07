from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from vitnns.stage_d_provenance import canonical_sha256
from vitnns.stage_f_f2 import (
    ATTACKS, F2_COMPLETION_SCHEMA_PATH, F2_CONFIG_PATH,
    F2_CONFIG_SCHEMA_PATH, F2_FALSE_PASS_SCHEMA_PATH,
    F2_RUN_SPEC_PATH, F2_RUN_SPEC_SCHEMA_PATH, StageFF2Error,
    _load_config, _load_run_spec, assert_f2_source_boundary,
    assert_no_future_outputs, build_caption_data, build_false_pass_audit,
    build_plot_data, build_render_spec, load_source_bundle,
    validate_runtime_isolation, verify_startup_gates,
)

REPO = Path(__file__).resolve().parents[1]


def _read(path: str) -> dict:
    return json.loads((REPO / path).read_bytes())


def _schema(path: str) -> dict:
    document = _read(path)
    Draft202012Validator.check_schema(document)
    return document


@pytest.fixture(scope="module")
def source() -> dict:
    return load_source_bundle(REPO)


@pytest.fixture(scope="module")
def plots(source: dict) -> dict:
    return build_plot_data(source)


def test_f2_config_and_run_spec_accept_frozen_documents_and_reject_bad_examples():
    config = _load_config(REPO)
    run_spec = _load_run_spec(REPO)
    assert config["canonical_digest"] == canonical_sha256({key: value for key, value in config.items() if key != "canonical_digest"})
    assert run_spec["canonical_digest"] == canonical_sha256({key: value for key, value in run_spec.items() if key != "canonical_digest"})
    config_validator = Draft202012Validator(_schema(F2_CONFIG_SCHEMA_PATH))
    bad = copy.deepcopy(config); bad["authorization"]["on_pass"] = "F4_ONLY"
    assert list(config_validator.iter_errors(bad))
    run_validator = Draft202012Validator(_schema(F2_RUN_SPEC_SCHEMA_PATH))
    bad_run = copy.deepcopy(run_spec); del bad_run["operations"]
    assert list(run_validator.iter_errors(bad_run))


def test_f2_startup_gate_accepts_clean_successor_without_requiring_head_equals_f1():
    gate = verify_startup_gates(REPO, require_clean=False, require_remote=False)
    assert gate["status"] == "PASS"
    assert gate["f1"]["archive_commit"] == "7a516571eabf364b7bb5d46ceb8aa74b450b3ae3"
    assert gate["e4"]["archive_commit"] == "7c82eca4e7de488f4537eff2b3fdb59feb3413ab"
    assert gate["next_authorization"] == "F2_ONLY"


def test_f1_ref_completion_hash_inventory_and_terminal_tamper_rejected(monkeypatch: pytest.MonkeyPatch):
    import vitnns.stage_f_f2 as module
    original = module.sha256_file
    monkeypatch.setattr(module, "sha256_file", lambda path: "0" * 64 if str(path).endswith("completion/f1.json") else original(path))
    with pytest.raises(StageFF2Error, match="F1 completion or inventory identity"):
        module.verify_startup_gates(REPO, require_clean=False, require_remote=False)


@pytest.mark.parametrize("field", ["F1_COMMIT", "EXPECTED_F1_COMPLETION_DIGEST", "EXPECTED_F1_INVENTORY_DIGEST"])
def test_each_f1_identity_tamper_is_rejected(monkeypatch: pytest.MonkeyPatch, field: str):
    import vitnns.stage_f_f2 as module
    monkeypatch.setattr(module, field, "0" * (40 if field == "F1_COMMIT" else 64))
    with pytest.raises(StageFF2Error):
        module.verify_startup_gates(REPO, require_clean=False, require_remote=False)


def test_e4_ref_completion_hash_inventory_recipe_manifest_and_paper_data_tamper_rejected(monkeypatch: pytest.MonkeyPatch):
    import vitnns.stage_f_f2 as module
    original = module.sha256_file
    monkeypatch.setattr(module, "sha256_file", lambda path: "f" * 64 if str(path).endswith("patch_scale_sensitivity.csv") else original(path))
    with pytest.raises(StageFF2Error, match="patch_scale_sensitivity file hash"):
        module.load_source_bundle(REPO)


@pytest.mark.parametrize("field", ["E4_COMMIT", "EXPECTED_E4_COMPLETION_DIGEST", "EXPECTED_E4_INVENTORY_DIGEST", "EXPECTED_E4_RECIPE_SHA256"])
def test_each_e4_identity_tamper_is_rejected(monkeypatch: pytest.MonkeyPatch, field: str):
    import vitnns.stage_f_f2 as module
    monkeypatch.setattr(module, field, "0" * (40 if field == "E4_COMMIT" else 64))
    with pytest.raises(StageFF2Error):
        module.verify_startup_gates(REPO, require_clean=False, require_remote=False)


def test_e4_manifest_tamper_is_rejected(monkeypatch: pytest.MonkeyPatch):
    import vitnns.stage_f_f2 as module
    original = module._strict_json_object
    def tamper(path: Path, context: str):
        document, raw = original(path, context)
        if str(path).endswith("payload_manifest.json") and context == "E4 payload_manifest":
            document = copy.deepcopy(document)
            target = next(item for item in document["artifacts"] if item["path"] == "paper_data/patch_scale_sensitivity.csv")
            target["sha256"] = "0" * 64
            document["canonical_digest"] = canonical_sha256({key: value for key, value in document.items() if key != "canonical_digest"})
            raw = json.dumps(document, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        return document, raw
    monkeypatch.setattr(module, "_strict_json_object", tamper)
    with pytest.raises(StageFF2Error, match="manifest binding"):
        module.load_source_bundle(REPO)


def test_csv_json_parity_tamper_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import vitnns.stage_f_f2 as module
    source_root = REPO / "evidence/stage_e/e4_recovery_v2_20260907/formal/paper_data"
    target_root = tmp_path / "paper_data"; target_root.mkdir()
    for suffix in ("csv", "json"):
        (target_root / f"prediction_strata.{suffix}").write_bytes((source_root / f"prediction_strata.{suffix}").read_bytes())
    csv_path = target_root / "prediction_strata.csv"
    raw = csv_path.read_text(encoding="utf-8"); csv_path.write_text(raw.replace("stage_d_fidelity_v1_500", "tampered_cohort", 1), encoding="utf-8")
    monkeypatch.setattr(module, "E4_PAPER_DATA_ROOT_PATH", "paper_data")
    import hashlib
    expected = {"csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(), "json_sha256": hashlib.sha256((target_root / "prediction_strata.json").read_bytes()).hexdigest(), "json_canonical_digest": _read("evidence/stage_e/e4_recovery_v2_20260907/formal/paper_data/prediction_strata.json")["canonical_digest"]}
    with pytest.raises(StageFF2Error, match="CSV JSON parity mismatch"):
        module._read_table_pair(tmp_path, "prediction_strata", expected)


def test_all_e4_tables_have_exact_hash_canonical_and_csv_json_parity(source: dict):
    expected_rows = {"patch_scale_sensitivity": 1500, "upsampling_sensitivity": 2000, "layer_head_descriptive": 2000, "prediction_strata": 60, "resource_and_qc": 7000}
    assert {name: table["row_count"] for name, table in source["tables"].items()} == expected_rows
    for table in source["tables"].values():
        assert table["source"]["json"]["canonical_digest"]
        assert table["row_order_digest"] == canonical_sha256(table["rows"])


def test_f2_a_patch_scale_source_to_plotted_exactness(source: dict, plots: dict):
    rows = source["tables"]["patch_scale_sensitivity"]["rows"]
    payload = plots["F2-A"]["plot_payload"]
    assert payload["analysis_role"] == "POST_D10_SECONDARY_DESCRIPTIVE"
    for series in payload["delta_fid_series"]:
        expected = [row["delta_fid"] for row in sorted((row for row in rows if row["variant_id"] == series["variant_id"]), key=lambda row: row["selection_order"])]
        assert series["values"] == expected
    assert payload["d10_primary_replaced"] is False


def test_f2_b_upsampling_source_exactness_and_uniform_nearest_equivalence(source: dict, plots: dict):
    payload = plots["F2-B"]["plot_payload"]
    equivalent = payload["equivalence_class"]
    assert equivalent["members"] == ["uniform_mass_divide_by_16", "mass_preserving_nearest"]
    assert equivalent["values_byte_equivalent_by_field"] is True
    assert equivalent["winner_loser_split"] is False
    rows = source["tables"]["upsampling_sensitivity"]["rows"]
    uniform = sorted((row for row in rows if row["variant_id"] == equivalent["members"][0]), key=lambda row: row["selection_order"])
    nearest = sorted((row for row in rows if row["variant_id"] == equivalent["members"][1]), key=lambda row: row["selection_order"])
    assert [(row["insertion_auc_mean"], row["deletion_auc_mean"]) for row in uniform] == [(row["insertion_auc_mean"], row["deletion_auc_mean"]) for row in nearest]


def test_f2_c_layer_head_stratum_method_metric_counts_and_intervals_exact(source: dict, plots: dict):
    payload = plots["F2-C"]["plot_payload"]
    assert payload["coordinate_order"] == [[0, 9], [4, 2], [6, 10], [11, 2]]
    assert payload["stratum_order"] == ["overall", "correct", "misclassified"]
    assert payload["scientific_validity"] == "MIXED"
    assert len(payload["source_summary_rows"]) == 60
    for panel in payload["panels"]:
        assert len(panel["series"]) == 5
        for series in panel["series"]:
            assert len(series["y"]) == len(series["lower"]) == len(series["upper"]) == len(series["source_counts"]) == 4
            assert all(item["valid"] + item["failed"] + item["na"] == item["n"] for item in series["source_counts"])


def test_f2_d_counts_failures_and_owner_budget_waiver_are_exact(plots: dict):
    payload = plots["F2-D"]["plot_payload"]
    assert payload["e3_terminal_counts"] == {"expected": 3000, "valid": 2998, "failed": 2, "na": 0}
    assert [item["selection_order"] for item in payload["typed_failures"]] == [117, 352]
    assert all(item["numeric_result"] is None for item in payload["typed_failures"])
    assert payload["failure_numeric_values"] == [None, None]
    assert payload["line_through_failures"] is False
    assert payload["e2_owner_budget_waiver"]["recorded_cost_status"] == "FAIL_COST"
    assert payload["e2_owner_budget_waiver"]["completion_basis"] == "PASS_WITH_EXPLICIT_OWNER_BUDGET_WAIVER"


def test_no_p_value_winner_new_decision_or_endpoint_and_captions_disclose_role(plots: dict):
    for module, document in plots.items():
        assert document["p_value_computed"] is False
        assert document["winner_selection_performed"] is False
        assert document["new_go_no_go_issued"] is False
        assert document["new_endpoint_computed"] is False
        caption = build_caption_data(module, document)
        assert "POST_D10_SECONDARY_DESCRIPTIVE" in caption["caption_text"]
        assert caption["d10_primary_modified"] is False


def test_render_specs_fix_panel_order_method_styles_and_layouts(plots: dict):
    f2a = build_render_spec("F2-A", plots["F2-A"], "single_column")
    f2b = build_render_spec("F2-B", plots["F2-B"], "double_column")
    f2c = build_render_spec("F2-C", plots["F2-C"], "double_column")
    assert f2a["grid"] == {"rows": 2, "columns": 1}
    assert f2b["grid"] == {"rows": 1, "columns": 2}
    assert f2c["grid"] == {"rows": 1, "columns": 3}
    nns = next(item for item in f2c["panels"][0]["data"]["series"] if item["label"] == "NNS insertion")
    matched = next(item for item in f2c["panels"][0]["data"]["series"] if item["label"] == "matched insertion")
    assert nns["style"] == {"color": "#0072B2", "linestyle": "-", "marker": "o"}
    assert matched["style"] == {"color": "#D55E00", "linestyle": "--", "marker": "s"}
    for spec in (f2a, f2b, f2c):
        assert spec["caption_keys"] == [panel["caption_key"] for panel in spec["panels"]]


def test_runtime_source_and_future_stage_boundaries():
    assert validate_runtime_isolation()["status"] == "PASS"
    assert assert_f2_source_boundary(REPO)["status"] == "PASS"
    assert assert_no_future_outputs(REPO)["status"] == "PASS"


def test_runtime_rejects_nonempty_cuda_visibility(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(StageFF2Error, match="CUDA_VISIBLE_DEVICES"):
        validate_runtime_isolation()


def test_runtime_rejects_forbidden_import(monkeypatch: pytest.MonkeyPatch):
    import sys
    monkeypatch.setitem(sys.modules, "torch", object())
    with pytest.raises(StageFF2Error, match="forbidden runtime"):
        validate_runtime_isolation()


def test_false_pass_audit_and_completion_schema_are_exact():
    audit = build_false_pass_audit(REPO)
    assert [item["attack"] for item in audit["attacks"]] == list(ATTACKS)
    assert all(item["result"] == "REJECTED" for item in audit["attacks"])
    Draft202012Validator(_schema(F2_FALSE_PASS_SCHEMA_PATH)).validate(audit)
    completion_schema = _schema(F2_COMPLETION_SCHEMA_PATH)
    assert completion_schema["properties"]["next_authorization"]["const"] == "F3_ONLY"
    assert completion_schema["properties"]["f3_started"]["const"] is False
    assert completion_schema["properties"]["f4_started"]["const"] is False

