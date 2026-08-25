from __future__ import annotations

import copy
import hashlib

import pytest

from vitnns.stage_d_plan import (
    MATCHED_METHOD_ID,
    NNS_METHOD_ID,
    StageDImageGroup,
    StageDPairBinding,
    StageDUnit,
)
from vitnns.stage_d_provenance import (
    DOCUMENT_TYPE,
    SCIENTIFIC_FAILURE_STATUS,
    SUCCESS_STATUS,
    SYNTHETIC_IMAGE_STATUS,
    SYNTHETIC_MODE,
    StageDProvenanceError,
    build_stage_d_provenance,
    canonical_json_bytes,
    canonical_json_file_bytes,
    validate_stage_d_provenance,
)


SHA = {str(index): str(index) * 64 for index in range(1, 10)}


def synthetic_group(*, audit: bool = False) -> StageDImageGroup:
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


def payload_inventory(*, audit: bool = False) -> list[dict[str, object]]:
    records: list[dict[str, object]] = [
        {
            "level": "L0", "path": "L0/result.json", "kind": "result",
            "format": "json", "dtype": "json", "shape": None,
            "bytes": 3, "sha256": SHA["1"],
        },
        {
            "level": "L1", "path": "L1/metric_ready_map.npy",
            "kind": "metric_ready_map", "format": "npy",
            "dtype": "float32", "shape": [56, 56], "bytes": 4,
            "sha256": SHA["2"],
        },
        {
            "level": "L1", "path": "L1/fidelity_curves.npz",
            "kind": "fidelity_curves", "format": "npz",
            "dtype": "float64", "shape": [2, 5, 21], "bytes": 5,
            "sha256": SHA["3"],
        },
    ]
    if audit:
        records.append({
            "level": "L2", "path": "L2/audit_evidence.npz",
            "kind": "audit_evidence", "format": "npz",
            "dtype": {
                "raw": "float32",
                "repeat_clean_metric_ready_map": "float32",
            },
            "shape": {
                "raw": [2, 3],
                "repeat_clean_metric_ready_map": [56, 56],
            }, "bytes": 6,
            "sha256": SHA["4"],
        })
    return records


def build_provenance(
    *, audit: bool = False, failure: dict[str, object] | None = None,
    inventory: list[dict[str, object]] | None = None,
    failure_l0_scalars: dict[str, object] | None = None,
    source_bindings: dict[str, object] | None = None,
):
    group = synthetic_group(audit=audit)
    unit = group.units[0]
    terminal_status = SCIENTIFIC_FAILURE_STATUS if failure else SUCCESS_STATUS
    provenance = build_stage_d_provenance(
        mode=SYNTHETIC_MODE, producer={"commit": "a" * 40, "tree": "b" * 40, "clean": True},
        dependencies={"lock_sha256": SHA["1"]},
        model={"config_sha256": SHA["2"], "state_dict_sha256": SHA["3"]},
        source_bindings=source_bindings or {
            "d0": {"canonical_digest": SHA["4"]},
            "d1": {"canonical_digest": SHA["5"]},
            "d2": {"canonical_digest": SHA["6"]},
            "c10": {"canonical_digest": SHA["7"]},
        },
        plan={"canonical_digest": SHA["9"]}, group=group, unit=unit,
        image_identity={"status": SYNTHETIC_IMAGE_STATUS, "fixture_id": "fixture-1"},
        method={"id": unit.method_id, "version": "1.0.0"},
        target={"id": unit.target_id, "version": "1.0.0"},
        operator={"id": "nonoverlapping_4x4_sum", "version": "1.0.0"},
        projection={"id": "signed_56x56_fine_cell_mass"},
        sign_semantics={"id": "target_oriented_signed"},
        normalization={"id": "signed_endpoint_delta_zero_to_one"},
        baseline={"id": "normalized_zero_canonical_rgb_0_5"},
        seed_contract={"namespace": "stage-d-fidelity|20260821|", "base_seed": 20260821},
        attempt={"number": 1}, retry={"status": "NOT_A_RETRY"},
        claim={"claim_id": "synthetic-claim"}, shard={"id": 0, "count": 1},
        terminal={"status": terminal_status, "terminal": True, "complete": True},
        failure=failure, resources={"forward_calls": 1, "backward_calls": 1},
        artifact_inventory=(
            payload_inventory(audit=audit) if inventory is None else inventory
        ),
        failure_l0_scalars=failure_l0_scalars,
    )
    return provenance, group, unit


def resign(document: dict[str, object]) -> None:
    semantic = {key: value for key, value in document.items() if key != "canonical_digest"}
    document["canonical_digest"] = hashlib.sha256(
        canonical_json_bytes(semantic)
    ).hexdigest()


def test_builds_stage_d_only_synthetic_provenance_with_complete_coordinates() -> None:
    provenance, group, unit = build_provenance()
    document = provenance.to_dict()
    assert document["document_type"] == DOCUMENT_TYPE
    assert document["mode"] == SYNTHETIC_MODE
    assert document["bindings"]["d7_authorization"] == {
        "status": "NOT_ISSUED", "digest": None,
    }
    assert document["bindings"]["coordinates"]["work_item_count"] == 211
    assert document["bindings"]["coordinates"]["group_digest"] == group.canonical_digest
    assert document["unit_identity_sha256"] == unit.canonical_digest
    rendered = canonical_json_file_bytes(document)
    assert rendered.endswith(b"\n")
    assert b"stage_c" not in rendered.lower()
    assert "selection" not in document["bindings"]
    assert "configuration" not in document["bindings"]
    assert validate_stage_d_provenance(
        provenance, expected_group=group, expected_unit=unit,
        expected_plan_digest=SHA["9"],
    ).canonical_digest == provenance.canonical_digest


def test_audit_and_scientific_failure_are_complete_terminal_provenance() -> None:
    failure = {
        "code": "SCIENTIFIC_INVALIDITY", "phase": "fidelity",
        "source": "stage_d_scientific_execution",
        "produced": [
            "L0/result.json", "L1/metric_ready_map.npy",
            "L1/fidelity_curves.npz", "L2/audit_evidence.npz",
        ],
        "absent": [],
    }
    provenance, group, _unit = build_provenance(
        audit=True, failure=failure,
        failure_l0_scalars={"validity": "invalid"},
    )
    document = provenance.to_dict()
    assert document["bindings"]["terminal"] == {
        "status": SCIENTIFIC_FAILURE_STATUS, "terminal": True,
        "complete": True,
    }
    assert len(document["bindings"]["artifact_inventory"]) == 5
    assert document["bindings"]["coordinates"]["work_item_count"] == 212
    validate_stage_d_provenance(provenance, expected_group=group)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda d: d["bindings"]["d7_authorization"].update({"status": "AUTHORIZED_STAGE_D_FIDELITY_SCOPE", "digest": SHA["8"]}), "NOT_ISSUED"),
        (lambda d: d["bindings"]["coordinates"]["work_item_digests"].pop(), "coverage|count|sequence"),
        (lambda d: d["bindings"]["scientific_contract"].update({"block_index": 1}), "block/head"),
        (lambda d: d["bindings"].update({"stage_c_selection": {}}), "fields differ|foreign-stage"),
        (lambda d: d["bindings"]["scientific_contract"]["operator"].update({"id": "stage_c_operator"}), "foreign-stage"),
        (lambda d: d["bindings"]["source_bindings"]["d0"].update({"path": "stage_c_results/report.json"}), "foreign-stage"),
        (lambda d: d["bindings"]["model"].update({"identity": "stage_c_model"}), "foreign-stage"),
        (lambda d: d["bindings"]["artifact_inventory"][0].update({"path": "L0/per_pixel_1.json"}), "closed layout"),
    ],
)
def test_self_resigned_tampering_still_fails_closed(mutation, message: str) -> None:
    provenance, group, unit = build_provenance()
    document = provenance.to_dict()
    mutation(document)
    resign(document)
    with pytest.raises(StageDProvenanceError, match=message):
        validate_stage_d_provenance(
            document, expected_group=group, expected_unit=unit,
            expected_plan_digest=SHA["9"],
        )


def test_provenance_is_deeply_immutable_and_export_is_detached() -> None:
    provenance, _group, _unit = build_provenance()
    with pytest.raises(TypeError):
        provenance["bindings"]["producer"]["clean"] = False
    exported = provenance.to_dict()
    exported["bindings"]["producer"]["clean"] = False
    assert provenance["bindings"]["producer"]["clean"] is True


def test_canonical_json_is_utf8_compact_finite_and_deterministic() -> None:
    assert canonical_json_bytes({"z": "中文", "a": 1}) == '{"a":1,"z":"中文"}'.encode("utf-8")
    with pytest.raises(StageDProvenanceError, match="non-finite"):
        canonical_json_bytes({"value": float("nan")})


def test_c10_historical_stage_c_paths_are_legal_source_bindings() -> None:
    sources = {
        "d0": {"canonical_digest": SHA["4"]},
        "d1": {"canonical_digest": SHA["5"]},
        "d2": {"canonical_digest": SHA["6"]},
        "c10": {
            "canonical_digest": SHA["7"],
            "report_path": "stage_c_results/stage_c_report.json",
            "stage_c_acceptance": "BLOCKED",
        },
    }
    provenance, group, unit = build_provenance(source_bindings=sources)
    assert provenance["bindings"]["source_bindings"]["c10"] == sources["c10"]
    validate_stage_d_provenance(
        provenance, expected_group=group, expected_unit=unit,
        expected_bindings={"source_bindings": sources},
    )


def test_partial_failure_provenance_binds_absent_scientific_payloads() -> None:
    failure = {"code": "TARGET_DEGENERATE", "phase": "target_resolution"}
    provenance, group, unit = build_provenance(
        failure=failure, inventory=[],
        failure_l0_scalars={"validity": "invalid"},
    )
    normalized = provenance["bindings"]["failure"]
    assert normalized["produced"] == ()
    assert normalized["absent"] == (
        "L0/result.json", "L1/metric_ready_map.npy",
        "L1/fidelity_curves.npz",
    )
    assert [item["path"] for item in provenance["bindings"]["artifact_inventory"]] == [
        "L0/failure.json"
    ]
    validate_stage_d_provenance(provenance, expected_group=group, expected_unit=unit)


def test_partial_failure_rejects_forged_produced_absent_declaration() -> None:
    failure = {
        "code": "TARGET_DEGENERATE", "phase": "target_resolution",
        "produced": ["L1/metric_ready_map.npy"],
        "absent": ["L0/result.json", "L1/fidelity_curves.npz"],
    }
    with pytest.raises(StageDProvenanceError, match="produced declaration"):
        build_provenance(
            failure=failure, inventory=[],
            failure_l0_scalars={"validity": "invalid"},
        )


def test_provenance_omits_post_publication_control_byte_count() -> None:
    provenance, _group, _unit = build_provenance()
    assert "control_and_provenance_bytes" not in provenance["bindings"]["resources"]
    with pytest.raises(StageDProvenanceError, match="disk-derived"):
        group = synthetic_group()
        build_stage_d_provenance(
            mode=SYNTHETIC_MODE,
            producer={"commit": "a" * 40, "tree": "b" * 40, "clean": True},
            dependencies={"lock_sha256": SHA["1"]},
            model={"config_sha256": SHA["2"]},
            source_bindings={name: {"canonical_digest": SHA[str(index)]} for index, name in enumerate(("d0", "d1", "d2", "c10"), 4)},
            plan={"canonical_digest": SHA["9"]}, group=group, unit=group.units[0],
            image_identity={"status": SYNTHETIC_IMAGE_STATUS},
            method={"id": group.units[0].method_id}, target={"id": group.units[0].target_id},
            operator={"id": "op"}, projection={"id": "projection"},
            sign_semantics={"id": "signed"}, normalization={"id": "normalized"},
            baseline={"id": "baseline"}, seed_contract={"base_seed": 1},
            attempt=1, retry={"status": "NOT_A_RETRY"}, claim={"id": "claim"},
            shard={"id": 0, "count": 1},
            terminal={"status": SUCCESS_STATUS, "terminal": True, "complete": True},
            failure=None, resources={"forward_calls": 1, "control_and_provenance_bytes": 4},
            artifact_inventory=payload_inventory(),
        )
