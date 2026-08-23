from __future__ import annotations

import copy

import pytest

import vitnns.stage_c_benchmark as benchmark
from vitnns.stage_c_plan import CLASS_METHOD_IDS, VALUE_METHOD_IDS


def _c7_storage_model() -> dict:
    return {
        "estimate_model": {
            "l0_base_bytes_per_unit": 32_768,
            "l0_bytes_per_work_item": 192,
            "l1_float32_patch_map_bytes": 784,
            "l2_shared_attention_bytes_per_audit_sample": 22_353_984,
        },
        "L0": {"estimated_bytes": 6_316_544, "file_count": 80},
        "L1": {"estimated_bytes": 484_480, "file_count": 200},
        "L2": {"estimated_bytes": 64_218_544, "file_count": 41},
        "L3": {"estimated_bytes": 8_192_000, "file_count": 40},
        "control": {
            "estimated_bytes": 19_802_892,
            "file_count": 84,
            "unit_manifest_files": 40,
            "unit_status_files": 40,
            "batch_control_files": 4,
        },
        "payload_bytes_before_control_overhead": 79_211_568,
    }


def _record(method: str, *, work_items: int = 929, map_runs: int = 5) -> dict:
    return {
        "unit_key": {"method_id": method},
        "work_item_counts": {
            "work_items": work_items,
            "map_runs": map_runs,
        },
    }


def _c8_usable_records() -> list[dict]:
    # Exact C8 method multiplicities are 720 for each value method and five
    # for each class method.  The retained run has one failed NNS unit, so the
    # post-C8 budget receives 2,179 usable records and 9,587 map runs.
    records: list[dict] = []
    for method in VALUE_METHOD_IDS:
        work = 5 if method == "direct_raw_cls_attention_per_head" else 929
        audit_count = 287 if method == benchmark.C7_DIAGNOSTIC_METHOD_ID else 288
        records.extend(
            _record(method, work_items=work, map_runs=5)
            for _ in range(audit_count)
        )
        records.extend(
            _record(method, work_items=work - 1, map_runs=4)
            for _ in range(432)
        )
    for method in CLASS_METHOD_IDS:
        records.extend(_record(method, map_runs=5) for _ in range(2))
        records.extend(_record(method, work_items=928, map_runs=4) for _ in range(3))
    assert len(records) == 2_179
    assert sum(item["work_item_counts"]["map_runs"] for item in records) == 9_587
    return records


def test_c7_aggregate_is_rescaled_from_its_own_unit_and_map_counts():
    non_l2, l2, _, details = benchmark._modeled_storage_profiles(
        _c8_usable_records(),
        _c7_storage_model(),
        c6_storage_payload_bytes=79_211_568,
        source_unit_count=2_179,
    )

    assert details["source_execution_units"] == 40
    assert details["source_map_file_count"] == 200
    assert details["estimate_model"]["l1_float32_patch_map_bytes"] == 784
    assert details["l1_container_bytes_per_unit_ceiling"] == 8_192
    assert set(non_l2) == set((*VALUE_METHOD_IDS, *CLASS_METHOD_IDS))
    assert set(l2) == set((*VALUE_METHOD_IDS, *CLASS_METHOD_IDS))


def test_same_scale_c7_projection_keeps_historical_detail_shape():
    records = []
    for method in VALUE_METHOD_IDS:
        work = 5 if method == "direct_raw_cls_attention_per_head" else 929
        records.extend(_record(method, work_items=work) for _ in range(12))
    for method in CLASS_METHOD_IDS:
        records.append(_record(method))

    _, _, _, details = benchmark._modeled_storage_profiles(
        records, _c7_storage_model(),
        c6_storage_payload_bytes=79_211_568, source_unit_count=40,
    )

    assert "source_execution_units" not in details
    assert "source_map_file_count" not in details


def test_missing_failed_method_uses_frozen_c7_method_profile():
    records = [
        item for item in _c8_usable_records()
        if item["unit_key"]["method_id"] != benchmark.C7_DIAGNOSTIC_METHOD_ID
    ]

    non_l2, l2, _, _ = benchmark._modeled_storage_profiles(
        records,
        _c7_storage_model(),
        c6_storage_payload_bytes=79_211_568,
        source_unit_count=len(records),
    )

    assert non_l2[benchmark.C7_DIAGNOSTIC_METHOD_ID] > 0
    assert l2[benchmark.C7_DIAGNOSTIC_METHOD_ID] > 0


@pytest.mark.parametrize(
    "damage, message",
    [
        (lambda model: model["L1"].update({"file_count": 201}),
         "unit/map counts differ from the frozen C7 plan"),
        (lambda model: model["L1"].update({"estimated_bytes": 156_799}),
         "C6 L1 estimate is below its exact map payload"),
        (lambda model: model["L1"].update({"estimated_bytes": 484_481}),
         "C6 L1 container bytes do not divide across source units"),
        (lambda model: model["control"].update({"unit_status_files": 39}),
         "manifest/status counts must be equal and non-zero"),
    ],
)
def test_c7_storage_scale_tampering_fails_closed(damage, message):
    model = copy.deepcopy(_c7_storage_model())
    damage(model)

    with pytest.raises(benchmark.StageCBenchmarkIntegrityError, match=message):
        benchmark._modeled_storage_profiles(
            _c8_usable_records(), model,
            c6_storage_payload_bytes=79_211_568, source_unit_count=2_179,
        )
