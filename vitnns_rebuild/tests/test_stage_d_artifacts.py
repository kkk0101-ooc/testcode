from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from vitnns.stage_d_artifacts import (
    AUDIT_PATH,
    CURVES_PATH,
    FAULT_AFTER_MANIFEST,
    FAULT_AFTER_PAYLOAD,
    FAULT_AFTER_RUNNING,
    FAULT_BEFORE_TERMINAL_STATUS,
    MAP_PATH,
    PROVENANCE_PATH,
    RESULT_PATH,
    InjectedStageDArtifactCrash,
    StageDArtifactPolicyError,
    StageDArtifactWriter,
    StageDArtifactWriterStateError,
    StageDAttemptNotEmptyError,
    StageDUnitPayload,
    attempt_directories,
    next_attempt_directory,
    validate_stage_d_attempt,
)
from vitnns.stage_d_plan import (
    MATCHED_METHOD_ID,
    NNS_METHOD_ID,
    StageDImageGroup,
    StageDPairBinding,
    StageDUnit,
)
from vitnns.stage_d_provenance import (
    SCIENTIFIC_FAILURE_STATUS,
    SUCCESS_STATUS,
    SYNTHETIC_IMAGE_STATUS,
    SYNTHETIC_MODE,
    build_stage_d_provenance,
    canonical_json_bytes,
    canonical_json_file_bytes,
)


SHA = {str(index): str(index) * 64 for index in range(1, 10)}


def group_fixture(*, audit: bool = False) -> StageDImageGroup:
    pair = StageDPairBinding.create(
        sample_id="synthetic-sample-000", selection_order=0, image_sha256=SHA["1"],
        model_context_digest=SHA["2"], target_context_digest=SHA["3"],
        fidelity_context_digest=SHA["4"], plan_context_digest=SHA["5"],
    )
    overlay = "SYNTHETIC_AUDIT_00" if audit else None
    units = tuple(
        StageDUnit.create(
            sample_id=pair.sample_id, selection_order=pair.selection_order,
            method_index=index, method_id=method_id,
            pair_binding_digest=pair.canonical_digest, audit=audit,
            audit_overlay_id=overlay,
        )
        for index, method_id in enumerate((NNS_METHOD_ID, MATCHED_METHOD_ID))
    )
    return StageDImageGroup.create(
        selection_order=0, sample_id=pair.sample_id,
        relative_path="synthetic/input.bin", image_sha256=pair.image_sha256,
        registry_row_digest=SHA["6"], audit=audit, audit_overlay_id=overlay,
        pair_binding=pair, units=units,
    )


def payload_fixture(*, audit: bool = False) -> StageDUnitPayload:
    return StageDUnitPayload(
        l0_scalars={"endpoint_parity": 0.0, "five_repeat_aucs": [0.1] * 5},
        metric_ready_map=np.zeros((56, 56), dtype=np.float32),
        fidelity_curves=np.zeros((2, 5, 21), dtype=np.float64),
        audit_evidence=(
            {
                "raw_attribution": np.ones((2, 3), dtype=np.float32),
                "repeat_clean_metric_ready_map": np.zeros(
                    (56, 56), dtype=np.float32
                ),
            }
            if audit else None
        ),
    )


def provenance_for(
    writer: StageDArtifactWriter, *, failure: dict[str, object] | None = None,
    failure_l0_scalars: dict[str, object] | None = None,
):
    status = SCIENTIFIC_FAILURE_STATUS if failure else SUCCESS_STATUS
    return build_stage_d_provenance(
        mode=SYNTHETIC_MODE, producer={"commit": "a" * 40, "tree": "b" * 40, "clean": True},
        dependencies={"lock_sha256": SHA["1"]},
        model={"config_sha256": SHA["2"], "state_dict_sha256": SHA["3"]},
        source_bindings={
            "d0": {"canonical_digest": SHA["4"]},
            "d1": {"canonical_digest": SHA["5"]},
            "d2": {"canonical_digest": SHA["6"]},
            "c10": {"canonical_digest": SHA["7"]},
        },
        plan={"canonical_digest": SHA["9"]}, group=writer.group, unit=writer.unit,
        image_identity={"status": SYNTHETIC_IMAGE_STATUS, "fixture_id": "fixture-1"},
        method={"id": writer.unit.method_id, "version": "1.0.0"},
        target={"id": writer.unit.target_id, "version": "1.0.0"},
        operator={"id": "nonoverlapping_4x4_sum", "version": "1.0.0"},
        projection={"id": "signed_56x56_fine_cell_mass"},
        sign_semantics={"id": "target_oriented_signed"},
        normalization={"id": "signed_endpoint_delta_zero_to_one"},
        baseline={"id": "normalized_zero_canonical_rgb_0_5"},
        seed_contract={"namespace": "stage-d-fidelity|20260821|", "base_seed": 20260821},
        attempt={"number": writer.attempt_number}, retry={"status": "NOT_A_RETRY"},
        claim={"claim_id": "synthetic-claim"}, shard={"id": 0, "count": 1},
        terminal={"status": status, "terminal": True, "complete": True},
        failure=failure, resources={"forward_calls": 1, "backward_calls": 1},
        artifact_inventory=writer.artifacts,
        failure_l0_scalars=failure_l0_scalars,
    )


def write_complete(tmp_path: Path, *, audit: bool = False):
    group = group_fixture(audit=audit)
    attempt = next_attempt_directory(tmp_path / "unit" / "attempts")
    writer = StageDArtifactWriter(attempt, unit=group.units[0], group=group)
    writer.write_payload(payload_fixture(audit=audit))
    writer.succeed(provenance=provenance_for(writer))
    return attempt, writer, group


def test_closed_layout_success_validates_and_is_resumable(tmp_path: Path) -> None:
    attempt, writer, group = write_complete(tmp_path)
    assert sorted(
        path.relative_to(attempt).as_posix() for path in attempt.rglob("*") if path.is_file()
    ) == sorted({
        "status.json", "manifest.json", RESULT_PATH, PROVENANCE_PATH,
        MAP_PATH, CURVES_PATH,
    })
    verdict = validate_stage_d_attempt(
        attempt, expected_unit=group.units[0], expected_group=group,
        expected_plan_digest=SHA["9"], expected_mode=SYNTHETIC_MODE,
    )
    assert verdict.valid and verdict.terminal and verdict.complete and verdict.resumable
    assert verdict.status == SUCCESS_STATUS
    assert verdict.digest and verdict.manifest_sha256
    assert verdict.artifact_count == 4
    assert canonical_json_file_bytes(
        __import__("json").loads((attempt / "status.json").read_text(encoding="utf-8"))
    ) == (attempt / "status.json").read_bytes()
    with pytest.raises(StageDArtifactWriterStateError, match="immutable"):
        writer.write_payload(payload_fixture())
    with pytest.raises(StageDAttemptNotEmptyError):
        StageDArtifactWriter(attempt, unit=group.units[0], group=group)


def _trace_l0(mode: str) -> dict[str, object]:
    l0: dict[str, object] = {
        "endpoint_parity": 0.0, "trace_mode": mode,
        "provenance_fragment": {"ig_result": {"selected_steps": 512}},
        "map_execution_records": [{"status": "completed"}],
        "method_id": NNS_METHOD_ID, "method_version": "1.0.0",
        "comparison_line": "value_path",
        "target_ref": "fixed_clean_attention_cls_head_direction",
        "projection": "pixel_mass_to_patch_sum",
        "sign_semantics": "signed",
    }
    contract_keys = (
        "method_id", "method_version", "comparison_line",
        "target_ref", "projection", "sign_semantics",
    )
    l0["nns_contract"] = {name: l0[name] for name in contract_keys}
    l0["trace"] = {
        "provenance_fragment": l0["provenance_fragment"],
        "map_execution_records": l0["map_execution_records"],
    }
    return l0


def _trace_payload(mode: str, *, include_l2_trace: bool = True) -> StageDUnitPayload:
    audit: dict[str, np.ndarray] = {
        "repeat_clean_metric_ready_map": np.zeros((56, 56), dtype=np.float32),
        "raw_attribution": np.ones((2, 3), dtype=np.float32),
    }
    if include_l2_trace:
        audit["nns_trace"] = np.ones((3, 4, 4), dtype=np.float32)
    return StageDUnitPayload(
        l0_scalars=_trace_l0(mode),
        metric_ready_map=np.zeros((56, 56), dtype=np.float32),
        fidelity_curves=np.zeros((2, 5, 21), dtype=np.float64),
        audit_evidence=audit,
    )


def test_fast_trace_policy_filters_nns_trace_before_hashing(tmp_path: Path) -> None:
    group = group_fixture(audit=False)
    attempt = next_attempt_directory(tmp_path / "fast")
    writer = StageDArtifactWriter(attempt, unit=group.units[0], group=group)
    l0 = _trace_l0("fast")
    writer.write_payload(StageDUnitPayload(
        l0_scalars=l0,
        metric_ready_map=np.zeros((56, 56), dtype=np.float32),
        fidelity_curves=np.zeros((2, 5, 21), dtype=np.float64),
    ))
    manifest = writer.succeed(provenance=provenance_for(writer))

    result = json.loads((attempt / RESULT_PATH).read_text(encoding="utf-8"))
    assert result["trace_mode"] == "fast"
    assert not {
        "nns_contract", "trace", "provenance_fragment",
        "map_execution_records",
    } & set(result)
    assert not (attempt / AUDIT_PATH).exists()
    provenance = json.loads(
        (attempt / PROVENANCE_PATH).read_text(encoding="utf-8")
    )
    assert provenance["bindings"]["artifact_inventory"] == [
        item for item in manifest["artifacts"]
        if item["path"] != PROVENANCE_PATH
    ]
    result_record = next(
        item for item in manifest["artifacts"] if item["path"] == RESULT_PATH
    )
    assert result_record["sha256"] == hashlib.sha256(
        (attempt / RESULT_PATH).read_bytes()
    ).hexdigest()
    assert validate_stage_d_attempt(
        attempt, expected_unit=group.units[0], expected_group=group,
    ).valid


def test_full_trace_policy_requires_and_retains_all_trace_fields(tmp_path: Path) -> None:
    group = group_fixture(audit=True)
    missing_attempt = next_attempt_directory(tmp_path / "missing-full")
    missing = StageDArtifactWriter(
        missing_attempt, unit=group.units[0], group=group
    )
    with pytest.raises(StageDArtifactPolicyError, match="nns_trace"):
        missing.write_payload(_trace_payload("full", include_l2_trace=False))

    attempt = next_attempt_directory(tmp_path / "complete-full")
    writer = StageDArtifactWriter(attempt, unit=group.units[0], group=group)
    writer.write_payload(_trace_payload("full"))
    manifest = writer.succeed(provenance=provenance_for(writer))
    result = json.loads((attempt / RESULT_PATH).read_text(encoding="utf-8"))
    assert result["trace_mode"] == "full"
    assert {
        "nns_contract", "trace", "provenance_fragment",
        "map_execution_records",
    } <= set(result)
    with np.load(attempt / AUDIT_PATH, allow_pickle=False) as archive:
        assert "nns_trace" in archive.files
    provenance = json.loads(
        (attempt / PROVENANCE_PATH).read_text(encoding="utf-8")
    )
    assert provenance["bindings"]["artifact_inventory"] == [
        item for item in manifest["artifacts"]
        if item["path"] != PROVENANCE_PATH
    ]
    assert validate_stage_d_attempt(
        attempt, expected_unit=group.units[0], expected_group=group,
    ).valid


def test_trace_mode_must_match_the_typed_unit(tmp_path: Path) -> None:
    group = group_fixture(audit=False)
    writer = StageDArtifactWriter(
        next_attempt_directory(tmp_path / "mismatch"),
        unit=group.units[0], group=group,
    )
    payload = StageDUnitPayload(
        l0_scalars=_trace_l0("full"),
        metric_ready_map=np.zeros((56, 56), dtype=np.float32),
        fidelity_curves=np.zeros((2, 5, 21), dtype=np.float64),
    )
    with pytest.raises(StageDArtifactPolicyError, match="disagrees"):
        writer.write_payload(payload)


def test_failure_transition_uses_the_filtered_fast_payload(tmp_path: Path) -> None:
    group = group_fixture(audit=False)
    attempt = next_attempt_directory(tmp_path / "fast-failure")
    writer = StageDArtifactWriter(attempt, unit=group.units[0], group=group)
    payload = StageDUnitPayload(
        l0_scalars=_trace_l0("fast"),
        metric_ready_map=np.zeros((56, 56), dtype=np.float32),
        fidelity_curves=np.zeros((2, 5, 21), dtype=np.float64),
    )
    writer.write_payload(payload)
    retained_l0 = json.loads((attempt / RESULT_PATH).read_text(encoding="utf-8"))
    failure = {
        "code": "SCIENTIFIC_INVALIDITY", "phase": "fidelity",
        "source": "stage_d_scientific_execution",
        "produced": [RESULT_PATH, MAP_PATH, CURVES_PATH], "absent": [],
    }
    writer.fail_scientifically(
        failure=failure, provenance=provenance_for(
            writer, failure=failure,
            failure_l0_scalars=retained_l0,
        ),
    )

    failure_l0 = json.loads((attempt / "L0/failure.json").read_text())[
        "l0_scalars"
    ]
    assert failure_l0["trace_mode"] == "fast"
    assert not {
        "provenance_fragment", "map_execution_records",
        "nns_contract", "trace",
    } & set(failure_l0)


def test_audit_has_one_consolidated_l2_and_non_audit_rejects_it(tmp_path: Path) -> None:
    attempt, _writer, group = write_complete(tmp_path, audit=True)
    assert (attempt / AUDIT_PATH).is_file()
    with np.load(attempt / AUDIT_PATH, allow_pickle=False) as archive:
        assert "repeat_clean_metric_ready_map" in archive.files
        assert archive["repeat_clean_metric_ready_map"].dtype == np.float32
        assert archive["repeat_clean_metric_ready_map"].shape == (56, 56)
    assert validate_stage_d_attempt(
        attempt, expected_unit=group.units[0], expected_group=group,
    ).resumable
    non_audit = group_fixture(audit=False)
    writer = StageDArtifactWriter(
        next_attempt_directory(tmp_path / "plain"), unit=non_audit.units[0],
        group=non_audit,
    )
    with pytest.raises(StageDArtifactPolicyError, match="audit payload presence"):
        writer.write_payload(payload_fixture(audit=True))


def test_audit_payload_requires_repeat_clean_map_inside_consolidated_l2() -> None:
    with pytest.raises(StageDArtifactPolicyError, match="repeat_clean_metric_ready_map"):
        StageDUnitPayload(
            l0_scalars={"ok": True},
            metric_ready_map=np.zeros((56, 56), dtype=np.float32),
            fidelity_curves=np.zeros((2, 5, 21), dtype=np.float64),
            audit_evidence={"raw": np.ones((2, 3), dtype=np.float32)},
        )


def test_scientific_failure_is_complete_terminal_and_resume_skippable(tmp_path: Path) -> None:
    group = group_fixture()
    attempt = next_attempt_directory(tmp_path / "unit")
    writer = StageDArtifactWriter(attempt, unit=group.units[0], group=group)
    payload = payload_fixture()
    writer.write_payload(payload)
    failure = {
        "code": "SCIENTIFIC_INVALIDITY", "phase": "fidelity",
        "source": "stage_d_scientific_execution",
        "produced": [RESULT_PATH, MAP_PATH, CURVES_PATH], "absent": [],
    }
    writer.fail_scientifically(
        failure=failure, provenance=provenance_for(
            writer, failure=failure,
            failure_l0_scalars=dict(payload.l0_scalars),
        ),
    )
    verdict = validate_stage_d_attempt(attempt, expected_unit=group.units[0], expected_group=group)
    assert verdict.valid and verdict.terminal and verdict.complete and verdict.resumable
    assert verdict.status == SCIENTIFIC_FAILURE_STATUS
    assert (attempt / "L0/failure.json").is_file()


@pytest.mark.parametrize("audit", [False, True])
def test_zero_payload_scientific_failure_is_complete_terminal(
    tmp_path: Path, audit: bool,
) -> None:
    group = group_fixture(audit=audit)
    attempt = next_attempt_directory(tmp_path / ("audit" if audit else "plain"))
    writer = StageDArtifactWriter(attempt, unit=group.units[0], group=group)
    l0 = {"validity": "invalid", "target_resolved": False}
    failure = {"code": "TARGET_DEGENERATE", "phase": "target_resolution"}
    provenance = provenance_for(
        writer, failure=failure, failure_l0_scalars=l0,
    )
    manifest = writer.fail_scientifically_partial(
        failure=failure, provenance=provenance, l0_scalars=l0,
    )
    expected_absent = [RESULT_PATH, MAP_PATH, CURVES_PATH] + (
        [AUDIT_PATH] if audit else []
    )
    assert manifest["failure"]["produced"] == []
    assert manifest["failure"]["absent"] == expected_absent
    assert [item["path"] for item in manifest["artifacts"]] == [
        "L0/failure.json", PROVENANCE_PATH,
    ]
    verdict = validate_stage_d_attempt(
        attempt, expected_unit=group.units[0], expected_group=group,
    )
    assert verdict.valid and verdict.complete and verdict.terminal and verdict.resumable


def test_partial_map_failure_retains_only_actual_scientific_payload(tmp_path: Path) -> None:
    group = group_fixture()
    attempt = next_attempt_directory(tmp_path / "partial-map")
    writer = StageDArtifactWriter(attempt, unit=group.units[0], group=group)
    writer.write_metric_ready_map(np.ones((56, 56), dtype=np.float32))
    l0 = {"validity": "invalid", "map_completed": True}
    failure = {"code": "FIDELITY_INVALID", "phase": "fidelity"}
    provenance = provenance_for(
        writer, failure=failure, failure_l0_scalars=l0,
    )
    writer.fail_scientifically_partial(
        failure=failure, provenance=provenance, l0_scalars=l0,
    )
    verdict = validate_stage_d_attempt(
        attempt, expected_unit=group.units[0], expected_group=group,
    )
    assert verdict.resumable
    assert (attempt / MAP_PATH).is_file()
    assert not (attempt / RESULT_PATH).exists()
    assert not (attempt / CURVES_PATH).exists()


def test_partial_failure_declaration_tamper_fails_even_when_chain_is_resigned(
    tmp_path: Path,
) -> None:
    group = group_fixture()
    attempt = next_attempt_directory(tmp_path / "tampered-failure")
    writer = StageDArtifactWriter(attempt, unit=group.units[0], group=group)
    l0 = {"validity": "invalid"}
    failure = {"code": "TARGET_DEGENERATE", "phase": "target_resolution"}
    writer.fail_scientifically_partial(
        failure=failure,
        provenance=provenance_for(
            writer, failure=failure, failure_l0_scalars=l0,
        ),
        l0_scalars=l0,
    )
    import json

    status_path = attempt / "status.json"
    manifest_path = attempt / "manifest.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status["failure"]["absent"].remove(CURVES_PATH)
    manifest["failure"] = status["failure"]
    manifest_raw = canonical_json_file_bytes(manifest)
    status["manifest_sha256"] = hashlib.sha256(manifest_raw).hexdigest()
    manifest_path.write_bytes(manifest_raw)
    status_path.write_bytes(canonical_json_file_bytes(status))
    verdict = validate_stage_d_attempt(
        attempt, expected_unit=group.units[0], expected_group=group,
    )
    assert not verdict.valid and not verdict.resumable
    assert any("failure" in error for error in verdict.errors)


def test_partial_failure_rejects_mismatched_expected_source_bindings(
    tmp_path: Path,
) -> None:
    group = group_fixture()
    attempt = next_attempt_directory(tmp_path / "source-binding")
    writer = StageDArtifactWriter(attempt, unit=group.units[0], group=group)
    l0 = {"validity": "invalid"}
    failure = {"code": "TARGET_DEGENERATE", "phase": "target_resolution"}
    writer.fail_scientifically_partial(
        failure=failure,
        provenance=provenance_for(
            writer, failure=failure, failure_l0_scalars=l0,
        ),
        l0_scalars=l0,
    )
    verdict = validate_stage_d_attempt(
        attempt, expected_unit=group.units[0], expected_group=group,
        expected_provenance_bindings={
            "source_bindings": {
                "d0": {"canonical_digest": SHA["4"]},
                "d1": {"canonical_digest": SHA["5"]},
                "d2": {"canonical_digest": SHA["6"]},
                "c10": {"canonical_digest": SHA["8"]},
            }
        },
    )
    assert not verdict.valid and not verdict.resumable
    assert any("source_bindings" in error for error in verdict.errors)


def test_infrastructure_interruption_is_nonterminal_and_old_attempt_is_retained(tmp_path: Path) -> None:
    group = group_fixture()
    attempts_root = tmp_path / "unit"
    first = next_attempt_directory(attempts_root)
    writer = StageDArtifactWriter(first, unit=group.units[0], group=group)
    writer.mark_infrastructure_interruption(
        code="SCHEDULER_PREEMPTION", resources={"wall_time_seconds": 2.0},
    )
    verdict = validate_stage_d_attempt(first, expected_unit=group.units[0], expected_group=group)
    assert verdict.valid and not verdict.terminal and not verdict.complete and not verdict.resumable
    before = {path.relative_to(first): path.read_bytes() for path in first.rglob("*") if path.is_file()}
    second = next_attempt_directory(attempts_root)
    assert second.name == "attempt-000002"
    assert attempt_directories(attempts_root) == (first, second)
    assert before == {path.relative_to(first): path.read_bytes() for path in first.rglob("*") if path.is_file()}


@pytest.mark.parametrize(
    "fault",
    [FAULT_AFTER_RUNNING, FAULT_AFTER_PAYLOAD, FAULT_AFTER_MANIFEST, FAULT_BEFORE_TERMINAL_STATUS],
)
def test_fault_boundaries_never_create_resumable_attempt(tmp_path: Path, fault: str) -> None:
    group = group_fixture()
    attempt = next_attempt_directory(tmp_path / fault)
    with pytest.raises(InjectedStageDArtifactCrash):
        writer = StageDArtifactWriter(
            attempt, unit=group.units[0], group=group, fault_injector=fault,
        )
        writer.write_payload(payload_fixture())
        writer.succeed(provenance=provenance_for(writer))
    verdict = validate_stage_d_attempt(attempt, expected_unit=group.units[0], expected_group=group)
    assert not verdict.resumable
    assert not (verdict.valid and verdict.terminal and verdict.complete)


@pytest.mark.parametrize(
    "relative", [RESULT_PATH, PROVENANCE_PATH, MAP_PATH, CURVES_PATH, "manifest.json", "status.json"],
)
def test_any_byte_tamper_breaks_validation(tmp_path: Path, relative: str) -> None:
    attempt, _writer, group = write_complete(tmp_path)
    path = attempt / relative
    path.write_bytes(path.read_bytes() + b"x")
    verdict = validate_stage_d_attempt(attempt, expected_unit=group.units[0], expected_group=group)
    assert not verdict.valid and not verdict.resumable


def test_non_string_manifest_artifact_path_returns_invalid(tmp_path: Path) -> None:
    attempt, _writer, group = write_complete(tmp_path)
    import json

    manifest_path = attempt / "manifest.json"
    status_path = attempt / "status.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["path"] = ["L0", "result.json"]
    manifest_raw = canonical_json_file_bytes(manifest)
    status["manifest_sha256"] = hashlib.sha256(manifest_raw).hexdigest()
    manifest_path.write_bytes(manifest_raw)
    status_path.write_bytes(canonical_json_file_bytes(status))
    verdict = validate_stage_d_attempt(
        attempt, expected_unit=group.units[0], expected_group=group,
    )
    assert not verdict.valid and not verdict.resumable
    assert any("path" in error for error in verdict.errors)


def test_exact_inventory_rejects_extra_symlink_and_hardlink(tmp_path: Path) -> None:
    attempt, _writer, group = write_complete(tmp_path)
    (attempt / "L3").mkdir()
    (attempt / "L3" / "forbidden.dat").write_bytes(b"not-display-data")
    verdict = validate_stage_d_attempt(attempt, expected_unit=group.units[0], expected_group=group)
    assert not verdict.valid
    assert any(
        "extra files" in error or "forbidden directory" in error
        for error in verdict.errors
    )

    other, _writer, other_group = write_complete(tmp_path / "other")
    source = other / RESULT_PATH
    linked = other / "L0" / "linked.json"
    os.link(source, linked)
    verdict = validate_stage_d_attempt(other, expected_unit=other_group.units[0], expected_group=other_group)
    assert not verdict.valid
    assert any("hardlink" in error for error in verdict.errors)


def test_payload_rejects_wrong_dtype_shape_nonfinite_and_extra_scope() -> None:
    with pytest.raises(StageDArtifactPolicyError, match="float32"):
        StageDUnitPayload(
            l0_scalars={"ok": True},
            metric_ready_map=np.zeros((56, 56), dtype=np.float64),
            fidelity_curves=np.zeros((2, 5, 21), dtype=np.float64),
        )
    with pytest.raises(StageDArtifactPolicyError, match="shape"):
        StageDUnitPayload(
            l0_scalars={"ok": True},
            metric_ready_map=np.zeros((14, 14), dtype=np.float32),
            fidelity_curves=np.zeros((2, 5, 21), dtype=np.float64),
        )
    bad = np.zeros((2, 5, 21), dtype=np.float64)
    bad[0, 0, 0] = np.nan
    with pytest.raises(StageDArtifactPolicyError, match="finite"):
        StageDUnitPayload(
            l0_scalars={"ok": True},
            metric_ready_map=np.zeros((56, 56), dtype=np.float32),
            fidelity_curves=bad,
        )
    with pytest.raises(StageDArtifactPolicyError, match="JSON-compatible"):
        StageDUnitPayload(
            l0_scalars={"per_pixel": np.zeros(2)},
            metric_ready_map=np.zeros((56, 56), dtype=np.float32),
            fidelity_curves=np.zeros((2, 5, 21), dtype=np.float64),
        )


def test_guard_environment_is_synthetic_and_model_free() -> None:
    assert os.environ.get("VITNNS_D4_STRICT") == "1"
    assert os.environ.get("VITNNS_D4_GUARD_ACTIVE") == "1"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    import sys
    assert not any(
        name == prefix or name.startswith(prefix + ".")
        for name in sys.modules
        for prefix in ("PIL", "timm", "torchvision", "vitnns.modeling")
    )
