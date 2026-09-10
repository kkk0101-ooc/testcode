from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vitnns import stage_x_x2_mech_m1_reconcile as reconcile


def test_signed_document_recomputes_its_digest() -> None:
    value = reconcile._signed("stage_x_x2_mech_m1_test", "PASS", {"a": 1})
    reconcile._validate_signed(value, "synthetic")
    value["payload"]["a"] = 2
    with pytest.raises(reconcile.M1ReconcileError, match="self-digest"):
        reconcile._validate_signed(value, "synthetic")


def test_csv_contract_is_deterministic_and_closed(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    path.write_bytes(reconcile._csv_bytes(("a", "b"), [{"a": 1, "b": 2}]))
    assert reconcile._read_csv(path, ("a", "b")) == [{"a": "1", "b": "2"}]
    with pytest.raises(reconcile.M1ReconcileError, match="fields"):
        reconcile._read_csv(path, ("b", "a"))


def test_mask_digest_is_dtype_and_shape_bound() -> None:
    mask = np.ones((224, 224), dtype=bool)
    assert reconcile._mask_digest(mask) == reconcile._mask_digest(mask.copy())
    changed = mask.copy(); changed[0, 0] = False
    assert reconcile._mask_digest(mask) != reconcile._mask_digest(changed)


def test_false_pass_roster_covers_required_attacks() -> None:
    required = {
        "route_method_transform_scale_seed_reduction", "failed_route_deleted_replaced_or_retried",
        "source_clean_map_or_terminal_drift", "partial_block_perturbation",
        "resize_substituted_for_4x4_or_16x16_block_aggregation", "horizontal_flip_omitted",
        "transformed_map_omitted_or_inverse_only_retained", "budget_waiver_or_silent_cap_excess",
        "m2_analysis_executed", "x2_no_go_modified",
        "x3_x5_f3_final_or_stage_g_started", "pass_without_source_recomputing_validation",
    }
    assert required <= set(reconcile.FALSE_PASS_ATTACKS)
    document = reconcile._false_pass_document()
    assert document["status"] == "PASS"
    assert all(row["result"] == "REJECTED" for row in document["payload"]["attacks"])


def test_access_audit_is_empirical_and_m1_only() -> None:
    access = reconcile._access_document()["payload"]["access"]
    assert access["source_clean_map_values_read"] is True
    assert access["cuda_accessed"] is True
    assert access["source_x2_stability_rows_read"] is False
    assert access["m2_analysis_executed"] is False
    json.dumps(access, allow_nan=False)


def test_recovery_roots_do_not_overwrite_blocked_m1_root() -> None:
    assert reconcile.ROOT.as_posix().endswith("x2_mech_m1_r1_20260910/formal")
    assert reconcile.OLD_BLOCKED_ROOT.as_posix().endswith("x2_mech_m1_20260910/formal")
    assert reconcile.ROOT != reconcile.OLD_BLOCKED_ROOT
    assert reconcile.ARCHIVE_REF.endswith("stage-x-x2-mech-m1-r1-completion")


def test_typed_plan_source_expectations_recompute_one_route() -> None:
    repo = Path(__file__).resolve().parents[1]
    routes, failed = reconcile._load_plan(repo)
    route = next(row for row in routes if row["source_status"] == "VALID")
    source = reconcile._source_expectations(repo, route)
    assert source["target"] >= 0
    assert source["selected_steps"] in (64, 128, 256, 512)
    assert len(source["checkpoint_sha256"]) == 64
    assert failed["source_status"] == "FAILED_ROUTE_LOCAL_NUMERICAL"
