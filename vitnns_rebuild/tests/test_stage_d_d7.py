"""Focused non-reserve tests for the recovery-v2 D7 gate."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from vitnns.stage_d_d7 import (
    AUTHORIZATION_PATH, BUDGET_PATH, D8_0_REBIND_FORMAL_ROOT,
    DEFAULT_SHARD_COUNT, FORMAL_ROOT, FULL_CHAIN_RECEIPT_PATH,
    StageDD7Error, _D7ValidatedExecutionContext,
    build_d8_0_fresh_root_authorization, build_d7_budget,
    build_execution_authorization, controls_from_authorization, load_d7_budget,
    load_recovery_pipeline,
    run_d8_initializer_synthetic_smoke, run_synthetic_smoke,
    validate_d8_initializer_synthetic_smoke, validate_execution_authorization, validate_smoke,
    validate_full_chain_synthetic_smoke,
)
from vitnns import stage_d_d7 as d7_module
from vitnns.stage_d_plan import (
    D7_VALIDATED_PROFILE_STATUS, canonical_sha256,
    validate_stage_d_shard_mapping,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_recovery_pipeline_rejects_stale_loader_but_preserves_frozen_plan() -> None:
    pipeline = load_recovery_pipeline(repo_root=REPO_ROOT)
    assert pipeline.plan.canonical_digest == (
        "5a40051ac17a7bfd6663640b1d2478c69d45eea07d3182653b90d2c7ca760443"
    )
    assert pipeline.plan.coordinate_digest == (
        "cc560e2a2d4e1220ccd812c073119a2b679de213f9770358f7decf51824661fb"
    )
    assert pipeline.plan.counts["image_groups"] == 500
    assert pipeline.plan.counts["execution_units"] == 1000


def test_archived_d0_d6_chain_is_source_recomputed() -> None:
    result = d7_module.validate_archived_recovery_chain(repo_root=REPO_ROOT)
    assert result["status"] == "PASS"
    assert result["source_recomputed_from_archive_blobs"] is True
    assert [item["stage"] for item in result["stages"]] == [
        "D0", "D1", "D2", "D3", "D4", "D5", "D6",
    ]
    assert all(item["producer_source_count"] > 0 for item in result["stages"])


def test_fresh_synthetic_smoke_is_422_items_and_source_recomputed(tmp_path: Path) -> None:
    receipt_path = tmp_path / "smoke_receipt.json"
    receipt = run_synthetic_smoke(
        repo_root=REPO_ROOT, output_root=tmp_path / "diagnostic",
        receipt_path=receipt_path,
    )
    assert receipt["counts"] == {
        "sample_count": 1, "unit_count": 2, "map_count": 2,
        "curve_count": 20, "curve_point_count": 420,
        "logical_work_item_count": 422,
    }
    assert set(receipt["access_attestation"].values()) == {False}
    assert validate_smoke(
        repo_root=REPO_ROOT, receipt_path=receipt_path,
    )["canonical_digest"] == receipt["canonical_digest"]
    with pytest.raises(StageDD7Error, match="fresh"):
        run_synthetic_smoke(
            repo_root=REPO_ROOT, output_root=tmp_path / "diagnostic",
            receipt_path=receipt_path,
        )


def test_r3_full_chain_smoke_calls_real_d4_d5_d6_apis() -> None:
    receipt = validate_full_chain_synthetic_smoke(repo_root=REPO_ROOT)
    assert receipt["status"] == "PASS"
    assert receipt["counts"] == {
        "image_groups": 500, "execution_units": 1000,
        "terminal_groups": 500, "terminal_units": 1000,
        "cache_releases": 500, "analysis_coordinates": 500,
    }
    assert set(receipt["real_api_chain"].values()) == {"PASS"}
    assert receipt["diagnostic_semantics"][
        "synthetic_scientific_decision_is_not_a_formal_result"
    ] is True
    assert set(receipt["access_attestation"].values()) == {False}
    assert (REPO_ROOT / FULL_CHAIN_RECEIPT_PATH).is_file()


def test_r4_paths_do_not_reuse_prior_terminal_or_authority() -> None:
    assert d7_module.D7_ID.endswith("_R4_20260904")
    assert d7_module.AUTHORIZATION_PATH.endswith("_r4.json")
    assert d7_module.COMPLETION_ROOT.endswith("_20260904_r4")
    assert d7_module.FULL_CHAIN_RECEIPT_PATH.startswith(
        d7_module.COMPLETION_ROOT + "/"
    )
    assert d7_module.PRODUCER_REF.endswith("d7-r4-producer")
    assert d7_module.AUTHORITY_REF.endswith("d7-r4-authority")


def test_r4_d8_initializer_smoke_is_source_recomputed(tmp_path: Path) -> None:
    receipt_path = tmp_path / "initializer_receipt.json"
    receipt = run_d8_initializer_synthetic_smoke(
        repo_root=REPO_ROOT, output_root=tmp_path / "synthetic-d8-init",
        receipt_path=receipt_path,
    )
    assert receipt["status"] == "PASS"
    assert receipt["state_sequence"] == [
        "ACCESS_STARTED", "INTERRUPTED", "RESUMED", "RUNNABLE",
    ]
    assert receipt["counts"] == {
        "expected_assets": 500, "validated_assets": 500,
        "interrupted_after_assets": 127, "journal_entries": 1002,
        "unit_attempt_directories": 0,
    }
    assert validate_d8_initializer_synthetic_smoke(
        repo_root=REPO_ROOT, receipt_path=receipt_path,
    )["canonical_digest"] == receipt["canonical_digest"]


def test_final_budget_uses_1000_unit_plan_and_passes_all_caps() -> None:
    document = build_d7_budget(repo_root=REPO_ROOT)
    estimate = document["estimate"]
    projection = estimate["projections"]
    assert document["plan"]["historical_8000_unit_projection_used"] is False
    assert document["plan"]["execution_units"] == 1000
    assert estimate["profile_status"] == D7_VALIDATED_PROFILE_STATUS
    assert estimate["budget_status"] == "PASS"
    assert document["preflight"]["status"] == "READY_FOR_D7_AUTHORIZATION"
    assert document["preflight"]["d7_final"] is True
    assert estimate["launch_violations"] == []
    assert estimate["completed_gate_violations"] == []
    assert projection["projected_h20_gpu_hours"] < 2000
    assert projection["projected_h20_gpu_hours_per_image_group_max"] < 4
    assert projection["projected_retained_regular_file_bytes"] < 68_719_476_736


def test_persisted_budget_is_source_recomputed() -> None:
    document, estimate, preflight = load_d7_budget(repo_root=REPO_ROOT)
    assert document["canonical_digest"] == build_d7_budget(
        repo_root=REPO_ROOT
    )["canonical_digest"]
    assert estimate.budget_status == "PASS"
    assert preflight.status == "READY_FOR_D7_AUTHORIZATION"


@pytest.mark.parametrize(
    "field,value",
    [
        ("seconds", 10_000_000.0),
        ("forward_calls", 30_001),
        ("cuda_peak_reserved_bytes", 96_636_764_161),
    ],
)
def test_d7_budget_rejects_over_cap_profiles(field: str, value: int | float) -> None:
    document = build_d7_budget(repo_root=REPO_ROOT)
    profiles = copy.deepcopy(document["estimate"]["profiles"])
    profiles[0][field] = value
    profiles[0]["canonical_digest"] = canonical_sha256({
        key: child for key, child in profiles[0].items()
        if key != "canonical_digest"
    })
    pipeline = load_recovery_pipeline(repo_root=REPO_ROOT)
    from vitnns.stage_d_plan import estimate_stage_d_run, preflight_stage_d

    estimate = estimate_stage_d_run(pipeline.contracts.config, pipeline.plan, profiles)
    preflight = preflight_stage_d(estimate)
    assert estimate.budget_status == "BLOCKED_NO_LAUNCH"
    assert preflight.status == "BLOCKED_NO_LAUNCH"
    assert preflight.violations


def test_sharding_is_exact_modulo_and_balanced() -> None:
    plan = load_recovery_pipeline(repo_root=REPO_ROOT).plan
    mapping = dict(validate_stage_d_shard_mapping(plan, DEFAULT_SHARD_COUNT))
    assert mapping == {index: index % DEFAULT_SHARD_COUNT for index in range(500)}
    assert [sum(value == shard for value in mapping.values()) for shard in range(8)] == [
        63, 63, 63, 63, 62, 62, 62, 62,
    ]


def test_authorization_is_fail_closed_on_dirty_producer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = d7_module._git

    def dirty(repo, *arguments):
        if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
            return " M synthetic-dirty-probe"
        return original(repo, *arguments)

    monkeypatch.setattr(d7_module, "_git", dirty)
    with pytest.raises(StageDD7Error, match="clean producer"):
        build_execution_authorization(repo_root=REPO_ROOT)


def test_authorization_validation_rejects_tamper_when_present() -> None:
    path = REPO_ROOT / AUTHORIZATION_PATH
    if not path.exists():
        pytest.skip("authority-only commit is intentionally not created yet")
    document = json.loads(path.read_text(encoding="utf-8"))
    checked = validate_execution_authorization(
        document, repo_root=REPO_ROOT, require_authority_ref=False,
    )
    assert isinstance(checked, _D7ValidatedExecutionContext)
    assert checked.formal is True
    tampered = copy.deepcopy(document)
    tampered["sharding"]["shard_count"] += 1
    with pytest.raises(StageDD7Error):
        validate_execution_authorization(
            tampered, repo_root=REPO_ROOT, require_authority_ref=False,
        )


def test_original_formal_root_is_not_reused_after_d8_0_blocked() -> None:
    root = REPO_ROOT / FORMAL_ROOT
    if root.exists():
        lifecycle = json.loads((root / "access/lifecycle.json").read_text())
        assert lifecycle["state"] == "BLOCKED"
        assert not list((root / "units").glob("*/attempts/attempt-*"))
    else:
        assert not root.exists()
    assert canonical_sha256([index % 8 for index in range(500)])
    assert (REPO_ROOT / BUDGET_PATH).is_file()


def test_d8_0_rebind_authorization_is_source_recomputed_when_present() -> None:
    path = REPO_ROOT / d7_module.D8_0_REBIND_AUTHORIZATION_PATH
    if not path.exists():
        pytest.skip("D8.0 rebind authorization is intentionally absent before rebind")
    document = json.loads(path.read_text(encoding="utf-8"))
    expected = build_d8_0_fresh_root_authorization(
        repo_root=REPO_ROOT, require_current_head=False,
        require_fresh_root=False, require_clean_worktree=False,
    )
    assert document == expected
    checked = validate_execution_authorization(
        document, repo_root=REPO_ROOT, require_authority_ref=False,
        require_fresh_root=False,
    )
    assert checked.formal is True
    assert checked.formal_root == D8_0_REBIND_FORMAL_ROOT
    assert checked.authorization_slot["authorization_canonical_digest"] == (
        document["canonical_digest"]
    )
    estimate, preflight = controls_from_authorization(document, repo_root=REPO_ROOT)
    assert estimate.canonical_digest == document["base_authorization"].get(
        "estimate_digest", estimate.canonical_digest,
    )
    assert preflight.analysis_authorization == "NOT_EVALUATED"


def test_production_backend_factory_is_lazy_before_d8_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Binding:
        model_path = REPO_ROOT / "assets/models/vit_base_patch16_224_weights.semantic_candidate.pth"

    class Backend:
        def __init__(self, **kwargs):
            events.append("constructed")
            self.sample_loader = kwargs["sample_loader"]

    monkeypatch.setattr(
        "vitnns.recovery_v2.load_recovery_v2_binding",
        lambda *args, **kwargs: Binding(),
    )
    monkeypatch.setattr("vitnns.stage_d_producer.ProductionStageDBackend", Backend)
    context = type("Context", (), {
        "code_binding": {"commit": "a" * 40, "tree": "b" * 40, "clean": True},
        "model_binding": {
            "identity": "RECOVERY_V2_NEW_ARTIFACT_NOT_HISTORICAL_2E718",
            "state_dict_sha256": "4" * 64,
        },
    })()
    backend = d7_module.production_backend_factory(context=context, device="cuda:0")
    assert events == ["constructed"]
    assert callable(backend.sample_loader)
