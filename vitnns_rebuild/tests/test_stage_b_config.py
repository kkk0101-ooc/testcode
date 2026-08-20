import copy
import json
from pathlib import Path

import pytest

from vitnns.constants import REPO_ROOT
from vitnns.stage_b_runner import (
    ALL_BLOCKS,
    STAGE_B_METHOD,
    ClassTargetConfig,
    StageBConfig,
    load_stage_b_config,
)


DEFAULT_CONFIG = REPO_ROOT / "config" / "stage_b.json"


def valid_config_dict() -> dict:
    return {
        "schema_version": 1,
        "method": STAGE_B_METHOD,
        "class_target": {"mode": "predicted"},
        "value_target": {"head": 2, "query_token": 0, "head_channel": 7},
        "baseline": "zero_normalized",
        "blocks": [0, 3, 6, 9, 11],
        "ig": {
            "quadrature": "gauss_legendre",
            "steps_schedule": [16, 32, 64],
            "step_chunk_size": 8,
            "atol": 1e-4,
            "rtol": 1e-2,
        },
        "profile": "paper",
        "budget_bytes": 8_000_000,
        "repeatability": {"enabled": True, "atol": 1e-6, "rtol": 1e-5},
        "sanity": {
            "enabled": True,
            "perturbation": "seeded_gaussian_noise",
            "seed": 123,
            "noise_std": 0.05,
            "clamp_to_valid_range": True,
            "min_relative_l1_change": 0.05,
            "epsilon": 1e-12,
        },
        "compute_budget": {
            "max_forward_calls": 30000,
            "max_backward_calls": 30000,
        },
        "formal": True,
    }


def test_default_stage_b_config_is_strict_and_round_trips():
    config = load_stage_b_config(DEFAULT_CONFIG)
    assert config.method == STAGE_B_METHOD
    assert config.profile == "paper"
    assert config.blocks == (0, 3, 6, 9, 11)
    assert config.ig.quadrature == "gauss_legendre"
    assert config.ig.steps_schedule == (512, 1024, 2048, 4096, 8192)
    assert config.compute_budget.max_forward_calls == 30_000
    assert config.value_target.to_dict() == {
        "head": 0,
        "query_token": 0,
        "head_channel": 0,
    }
    assert StageBConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("schema_version",), 2, "schema_version"),
        (("method",), "conditional_fixed_attention_model_ig", "method must be"),
        (("baseline",), "gray", "baseline"),
        (("profile",), "production", "profile"),
        (("budget_bytes",), 0, "budget_bytes"),
        (("ig", "quadrature"), "riemann", "gauss_legendre"),
        (("ig", "steps_schedule"), [32, 16], "strictly increasing"),
        (("ig", "step_chunk_size"), 65, "cannot exceed"),
        (("value_target", "head"), 12, "value_target.head"),
        (("value_target", "query_token"), 197, "value_target.query_token"),
        (("value_target", "head_channel"), 64, "value_target.head_channel"),
        (("sanity", "noise_std"), 0.0, "noise_std"),
        (("sanity", "min_relative_l1_change"), 0.0, "must be > 0"),
        (("compute_budget", "max_forward_calls"), 0, "max_forward_calls"),
    ],
)
def test_stage_b_config_rejects_invalid_values(path, value, match):
    document = valid_config_dict()
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=match):
        StageBConfig.from_dict(document)


def test_stage_b_config_rejects_unknown_and_duplicate_keys(tmp_path):
    document = valid_config_dict()
    document["silent_fallback"] = True
    with pytest.raises(ValueError, match="unknown keys.*silent_fallback"):
        StageBConfig.from_dict(document)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version": 1, "schema_version": 1}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Duplicate JSON key: schema_version"):
        load_stage_b_config(duplicate)


def test_class_target_modes_are_conditional():
    assert ClassTargetConfig.from_dict({"mode": "predicted"}).index is None
    assert ClassTargetConfig.from_dict({"mode": "ground_truth"}).index is None
    assert ClassTargetConfig.from_dict({"mode": "index", "index": 999}).index == 999
    with pytest.raises(ValueError, match="required"):
        ClassTargetConfig.from_dict({"mode": "index"})
    with pytest.raises(ValueError, match="not allowed"):
        ClassTargetConfig.from_dict({"mode": "predicted", "index": 1})


def test_blocks_are_preregistered_and_audit_requires_all_twelve():
    document = valid_config_dict()
    document["blocks"] = [0, 0, 11]
    with pytest.raises(ValueError, match="unique and strictly increasing"):
        StageBConfig.from_dict(document)

    document = valid_config_dict()
    document["profile"] = "audit"
    with pytest.raises(ValueError, match="requires every block"):
        StageBConfig.from_dict(document)
    document["blocks"] = list(ALL_BLOCKS)
    assert StageBConfig.from_dict(document).blocks == ALL_BLOCKS


@pytest.mark.parametrize("gate", ["repeatability", "sanity"])
def test_formal_config_requires_repeatability_and_sanity(gate):
    document = valid_config_dict()
    document[gate]["enabled"] = False
    with pytest.raises(ValueError, match="formal Stage B runs require"):
        StageBConfig.from_dict(document)

    document["formal"] = False
    assert StageBConfig.from_dict(document).formal is False


def test_config_rejects_nonfinite_json_numbers(tmp_path):
    document = copy.deepcopy(valid_config_dict())
    raw = json.dumps(document).replace('0.0001', 'NaN')
    path = tmp_path / "nonfinite.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError, match="Non-finite JSON number"):
        load_stage_b_config(path)
