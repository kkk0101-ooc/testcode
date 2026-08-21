from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from vitnns.constants import REPO_ROOT
from vitnns.stage_c_config import (
    StageCConfig,
    StageCConfigError,
    load_stage_c_config,
)


CONFIG_PATH = REPO_ROOT / "config/stage_c.json"
SCHEMA_PATH = REPO_ROOT / "config/stage_c.schema.json"


def frozen_document() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def mutate(document: dict, path: tuple[object, ...], value: object) -> dict:
    changed = copy.deepcopy(document)
    target: object = changed
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    return changed


def test_frozen_config_is_typed_deeply_immutable_and_exact_roundtrip():
    original = frozen_document()
    config = load_stage_c_config(CONFIG_PATH)

    assert config.schema_version == 1
    assert config.protocol.version == "1.0.0"
    assert len(config.methods) == 7
    assert config.method_by_id["full_model_class_logit_ig"].comparison_line == "class_logit"
    assert config.targets["ig"]["steps_schedule"] == (512, 1024, 2048, 4096, 8192)
    assert config.to_dict() == original
    assert list(config.to_dict()) == list(original)
    assert all("method_version" not in method for method in config.to_dict()["methods"])

    with pytest.raises(TypeError):
        config.targets["baseline"] = {}
    with pytest.raises(TypeError):
        config.targets["baseline"]["channels"] = 4
    with pytest.raises((AttributeError, TypeError)):
        config.methods[0].id = "tampered"

    exported = config.to_dict()
    exported["targets"]["baseline"]["channels"] = 9
    assert config.targets["baseline"]["channels"] == 3


def test_from_dict_does_not_alias_mutable_caller_data():
    document = frozen_document()
    config = StageCConfig.from_dict(document, schema_path=SCHEMA_PATH)
    document["methods"][0]["id"] = "caller_mutated"
    assert config.methods[0].id == "direct_raw_cls_attention_per_head"


@pytest.mark.parametrize(
    "path,value",
    [
        (("schema_version",), True),
        (("stage",), "B"),
        (("registry", "expected_counts", "total"), 19),
        (("roles", "allowed_analysis_roles"), ["holdout", "calibration"]),
        (("methods", 0, "comparison_line"), "class_logit"),
        (("methods", 0, "explanation_target_ref"), "clean_predicted_class_raw_logit"),
        (("methods", 0, "fidelity_target_ref"), "missing_target"),
        (("methods", 0, "projection_ref"), "missing_projection"),
        (("targets", "ig", "steps_schedule"), [1024, 512, 2048, 4096, 8192]),
        (("targets", "ig", "step_chunk_size"), 0),
        (("targets", "ig", "atol"), 1),
        (("targets", "fixed_clean_attention_cls_head_direction", "direction_norm_relative_threshold"), float("nan")),
        (("layers_heads", "calibration_blocks"), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 11]),
        (("layers_heads", "depth_bands"), [[0, 1, 2], [3, 4, 5], [6, 7, 8], [8, 10, 11]]),
        (("projection", "metric_ready_dtype"), "float16"),
        (("metrics", "js_epsilon"), 1e-12),
        (("metrics", "hotspot_fractions"), [0.2, 0.1]),
        (("fidelity", "value_path_pair"), ["direct_raw_cls_attention_per_head", "matched_fixed_clean_attention_patch_token_contribution"]),
        (("fidelity", "curve_fractions"), [0.0, 1.0]),
        (("perturbations", "items", 0, "fully_valid_patch_rows"), list(range(12, -1, -1))),
        (("retention", "display_is_never_metric_input"), False),
        (("budget", "forward_calls_per_execution_unit_max"), 0),
        (("selection", "forbidden_role"), "calibration"),
        (("statistics", "confirmatory_n"), 14),
        (("decision_gates", "cost_caps_ref"), "unknown_budget"),
        (("decision_gates", "decision_priority"), list(reversed(frozen_document()["decision_gates"]["decision_priority"]))),
    ],
)
def test_strict_loader_rejects_value_type_order_range_reference_and_combo_drift(path, value):
    with pytest.raises(StageCConfigError):
        StageCConfig.from_dict(mutate(frozen_document(), path, value), schema_path=SCHEMA_PATH)


def test_unknown_and_missing_fields_fail_closed():
    unknown = frozen_document()
    unknown["silent_default"] = True
    with pytest.raises(StageCConfigError, match="unknown fields.*silent_default"):
        StageCConfig.from_dict(unknown, schema_path=SCHEMA_PATH)

    missing = frozen_document()
    del missing["metrics"]["js_log_base"]
    with pytest.raises(StageCConfigError, match="missing fields.*js_log_base"):
        StageCConfig.from_dict(missing, schema_path=SCHEMA_PATH)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_json_loader_rejects_nonfinite_tokens(tmp_path: Path, token: str):
    raw = CONFIG_PATH.read_text(encoding="utf-8")
    raw = raw.replace("0.0001", token, 1)
    path = tmp_path / "nonfinite.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(StageCConfigError, match="Non-finite JSON number"):
        load_stage_c_config(path)


def test_json_loader_rejects_duplicate_keys_at_any_depth(tmp_path: Path):
    raw = CONFIG_PATH.read_text(encoding="utf-8")
    raw = raw.replace(
        '"schema_version": 1,',
        '"schema_version": 1, "schema_version": 1,',
        1,
    )
    path = tmp_path / "duplicate.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(StageCConfigError, match="Duplicate JSON key: schema_version"):
        load_stage_c_config(path)


def test_loader_rejects_malformed_json_and_missing_path(tmp_path: Path):
    malformed = tmp_path / "bad.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(StageCConfigError, match="Invalid JSON"):
        load_stage_c_config(malformed)
    with pytest.raises(FileNotFoundError):
        load_stage_c_config(tmp_path / "missing.json")
