from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from vitnns import recovery_c9_launch as recovery
from vitnns import stage_c_c9 as c9
from vitnns.stage_c_provenance import canonical_sha256
from vitnns.recovery_v2 import load_recovery_v2_binding
from vitnns.stage_c_config import load_stage_c_config
from vitnns.stage_c_plan import (
    build_execution_plan, load_run_spec, load_stage_c_registry,
)


ROOT = Path(__file__).resolve().parents[1]


def test_recovery_c9_schema_is_valid_and_scope_is_narrow() -> None:
    schema = json.loads(recovery.DEFAULT_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    assert schema["additionalProperties"] is False
    assert schema["properties"]["status"]["const"] == recovery.STATUS


def test_salvage_selection_adapter_uses_exact_bound_file() -> None:
    path = recovery.DEFAULT_C8_ROOT / (
        "selection/stage_c_salvage_repair_selection.json"
    )
    record = SimpleNamespace(
        repair_record=SimpleNamespace(selection_path=path), repo_root=ROOT,
    )
    state = recovery.load_salvage_selection_state(record)
    assert state.sha256 == recovery.C8_SELECTION_FILE_SHA256
    assert tuple(
        (item.block_index, item.head_index)
        for item in state.depth_band_winners
    ) == recovery.WINNERS
    assert (state.primary.block_index, state.primary.head_index) == recovery.PRIMARY


def test_recovery_c9_real_plan_has_exact_method_counts() -> None:
    binding = load_recovery_v2_binding(verify_assets=False)
    record = SimpleNamespace(
        repair_record=SimpleNamespace(
            selection_path=(
                recovery.DEFAULT_C8_ROOT
                / "selection/stage_c_salvage_repair_selection.json"
            )
        ),
        repo_root=ROOT,
    )
    config = load_stage_c_config()
    selection = recovery.load_salvage_selection_state(record)
    registry = load_stage_c_registry(
        config, recovery_binding=binding, verify_live_assets=False,
    )
    plan = build_execution_plan(
        config, registry,
        load_run_spec(ROOT / "config/stage_c_c9_run_spec.json"),
        selection=selection,
    )

    result = c9.validate_c9_execution_plan(plan, config)
    counts = Counter(unit.key.method_id for unit in plan.units)

    assert result["valid"] is True
    assert len(plan.units) == 320
    assert counts == Counter({
        **{method: 80 for method in c9.VALUE_METHOD_IDS},
        **{method: 20 for method in c9.CLASS_METHOD_IDS},
    })


def test_recovery_handoff_rejects_resigned_digest_tamper() -> None:
    value = {
        "schema_version": 1,
        "document_type": "stage_c_c8_salvage_repair_to_c9_handoff",
        "status": "PASS", "integrity_valid": True,
        "c8_root": str(recovery.DEFAULT_C8_ROOT),
        "c8_report_path": "/c8/report.json",
        "c8_report_file_sha256": recovery.C8_COMPLETION_FILE_SHA256,
        "c8_report_canonical_digest": recovery.C8_COMPLETION_CANONICAL_DIGEST,
        "c8_recovery_selection_path": "/c8/selection.json",
        "c8_recovery_selection_file_sha256": recovery.C8_SELECTION_FILE_SHA256,
        "tracked_selection_path": "/c8/selection.json",
        "tracked_selection_file_sha256": recovery.C8_SELECTION_FILE_SHA256,
        "selection_canonical_digest": recovery.C8_SELECTION_CANONICAL_DIGEST,
        "c8_completion_commit": recovery.C8_ARCHIVED_CUSTODY_COMMIT,
        "c8_execution_plan_sha256": "1" * 64,
        "c8_run_spec_sha256": "2" * 64,
        "c8_source_tree_total_bytes": 1,
        "c8_retained_artifact_bytes": 1,
        "c9_projected_artifact_bytes": 1,
        "c8_c9_projected_artifact_bytes": 2,
        "c9_projected_h20_gpu_hours": 1.0,
        "c8_c9_projected_h20_gpu_hours": 2.0,
        "measured_slowest_unit_seconds": 1.0,
        "measured_peak_reserved_bytes_per_unit": 1,
        "stage_d_projected_artifact_bytes": 1,
        "stage_d_projected_h20_gpu_hours": 1.0,
        "stage_d_projected_h20_gpu_hours_per_image": 1.0,
        "salvage_binding": {}, "recovery_c9_authority": {},
    }
    value["canonical_digest"] = canonical_sha256(value)
    recovery.validate_recovery_handoff_document(value)
    value["c8_completion_commit"] = "0" * 40
    value["canonical_digest"] = canonical_sha256({
        key: item for key, item in value.items() if key != "canonical_digest"
    })
    with pytest.raises(recovery.RecoveryC9LaunchError, match="identity"):
        recovery.validate_recovery_handoff_document(value)


def test_c9_core_accepts_recovery_handoff_and_rejects_old_root_alias() -> None:
    value = {
        "schema_version": 1,
        "document_type": "stage_c_c8_salvage_repair_to_c9_handoff",
        "status": "PASS", "integrity_valid": True,
        "c8_root": str(recovery.DEFAULT_C8_ROOT),
        "c8_report_path": "/c8/report.json",
        "c8_report_file_sha256": recovery.C8_COMPLETION_FILE_SHA256,
        "c8_report_canonical_digest": recovery.C8_COMPLETION_CANONICAL_DIGEST,
        "c8_recovery_selection_path": "/c8/selection.json",
        "c8_recovery_selection_file_sha256": recovery.C8_SELECTION_FILE_SHA256,
        "tracked_selection_path": "/c8/selection.json",
        "tracked_selection_file_sha256": recovery.C8_SELECTION_FILE_SHA256,
        "selection_canonical_digest": recovery.C8_SELECTION_CANONICAL_DIGEST,
        "c8_completion_commit": recovery.C8_ARCHIVED_CUSTODY_COMMIT,
        "c8_execution_plan_sha256": "1" * 64,
        "c8_run_spec_sha256": "2" * 64,
        "c8_source_tree_total_bytes": 1, "c8_retained_artifact_bytes": 1,
        "c9_projected_artifact_bytes": 1,
        "c8_c9_projected_artifact_bytes": 2,
        "c9_projected_h20_gpu_hours": 1.0,
        "c8_c9_projected_h20_gpu_hours": 2.0,
        "measured_slowest_unit_seconds": 1.0,
        "measured_peak_reserved_bytes_per_unit": 1,
        "stage_d_projected_artifact_bytes": 1,
        "stage_d_projected_h20_gpu_hours": 1.0,
        "stage_d_projected_h20_gpu_hours_per_image": 1.0,
        "salvage_binding": {}, "recovery_c9_authority": {},
    }
    value["canonical_digest"] = canonical_sha256(value)
    c9._validate_handoff_document(value)
    value["c8_root"] = "/data00/old-interrupted-c8"
    value["canonical_digest"] = canonical_sha256({
        key: item for key, item in value.items() if key != "canonical_digest"
    })
    with pytest.raises(c9.StageCBenchmarkIntegrityError):
        c9._validate_handoff_document(value)
