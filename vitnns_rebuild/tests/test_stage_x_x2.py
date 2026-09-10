from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from vitnns import stage_x_x2 as x2

REPO = Path(__file__).resolve().parents[1]


def test_frozen_config_is_closed_and_self_digesting():
    value = x2.load_config(REPO)
    assert value["authorization"] == "X2_NNS_V2_DEVELOPMENT_AND_FUTILITY_ONLY"
    assert len(value["models"]) == 5
    assert value["methods"]["competitive_primary"] == [
        "vit_grad_cam", "transformer_attribution_chefer", "attnlrp",
        "rise", "occlusion",
    ]
    assert value["methods"]["identity_control"] not in value["methods"]["competitive_primary"]
    changed = copy.deepcopy(value); changed["extra"] = True
    with pytest.raises(x2.StageXX2Error):
        from jsonschema import Draft202012Validator
        errors = list(Draft202012Validator(x2.strict_read(REPO / x2.CONFIG_SCHEMA_PATH)).iter_errors(changed))
        if errors:
            raise x2.StageXX2Error(errors[0].message)


def test_selection_recomputes_from_only_five_allowed_fields():
    config = x2.load_config(REPO)
    samples, proof = x2._sample_rows(REPO, config)
    assert len(samples) == len({row["source_object_id"] for row in samples}) == 40
    assert len({row["class_id"] for row in samples}) == 40
    assert proof["outcome_fields_read"] == []
    assert set(proof["selection_fields_read"]) == set(config["sample_selection"]["allowed_fields"])
    first = samples[0]
    raw = "\0".join((config["sample_selection"]["namespace"], first["dataset_release_id"], first["class_id"], first["source_object_id"], first["file_sha256"]))
    assert first["selection_sha256"] == hashlib.sha256(raw.encode()).hexdigest()


def test_unit_manifest_is_200_unique_shared_routes_and_four_shards():
    config = x2.load_config(REPO); samples, _ = x2._sample_rows(REPO, config)
    manifest = x2._unit_manifest(config, samples)
    assert manifest["unit_count"] == manifest["unique_unit_count"] == 200
    assert set(manifest["per_model_counts"].values()) == {40}
    assert set(manifest["shard_counts"]) == {"0", "1", "2", "3"}
    expected = {row["source_object_id"] for row in samples}
    by_model = {}
    for unit in manifest["units"]:
        by_model.setdefault(unit["model_id"], set()).add(unit["sample_id"])
    assert all(value == expected for value in by_model.values())


def test_false_pass_roster_covers_user_required_attacks():
    attacks = set(x2.load_config(REPO)["false_pass_attacks"])
    required = {
        "identity_control_in_competitive_or_holm", "holm_not_exactly_five",
        "model_specific_sample_lists", "nns_ig_identity_mismatch",
        "fused_unfused_drift_misreported", "cross_layer_sum",
        "failed_or_na_dropped_replaced_or_imputed",
        "p99_replaced_by_mean_or_p95", "display_product_in_metric",
        "x3_candidate_early_access", "x2_pass_claimed_as_formal_superiority",
    }
    assert required <= attacks and len(attacks) == 22
