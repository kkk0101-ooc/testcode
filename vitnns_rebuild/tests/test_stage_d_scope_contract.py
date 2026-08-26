"""Strictly model-free/no-JPEG D0 scope and sealed-metadata tests."""

from __future__ import annotations

import copy
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from vitnns.stage_d_scope import (
    ALLOWED_CLAIM_IDS,
    C10_ROOT_INVENTORY_DIGEST,
    D0_SCOPE_CANONICAL_DIGEST,
    DEFAULT_SCOPE_DECISION,
    DEFAULT_SCOPE_DOCUMENT,
    DEFAULT_SCOPE_SCHEMA,
    DENIED_CLAIM_IDS,
    FUTURE_CONDITIONAL_CLAIM_IDS,
    MANDATORY_DISCLOSURE_IDS,
    StageDScopeError,
    load_stage_d_scope_decision,
    extract_scope_document_projection,
    required_scope_document_fragments,
    scope_decision_digest,
    scope_document_projection,
    strict_json_load,
    validate_stage_d_scope_decision,
    verify_c10_sealed_evidence,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
STEP_BY_STEP = REPO_ROOT / "docs/阶段D-step-by-step.md"
TOP_LEVEL_PLAN = REPO_ROOT.parent / "论文1实现审计与后续计划.md"
REBUILD_README = REPO_ROOT / "README.md"
TOP_LEVEL_README = REPO_ROOT.parent / "README.md"
FORBIDDEN_MODULE_PREFIXES = (
    "PIL", "cv2", "timm", "torch", "torchvision",
    "vitnns.modeling",
)


def _load() -> dict:
    return strict_json_load(DEFAULT_SCOPE_DECISION)


def _resign(document: dict) -> dict:
    document["canonical_digest"] = scope_decision_digest(document)
    return document


def test_scope_schema_is_closed_draft_2020_12() -> None:
    schema = strict_json_load(DEFAULT_SCOPE_SCHEMA)
    Draft202012Validator.check_schema(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])

    def visit(rule: object) -> None:
        if not isinstance(rule, dict):
            return
        if rule.get("type") == "object":
            assert rule.get("additionalProperties") is False
            properties = rule.get("properties", {})
            assert set(rule.get("required", ())) == set(properties)
        for value in rule.values():
            if isinstance(value, dict):
                visit(value)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

    visit(schema)


def test_checked_in_scope_decision_matches_schema_and_semantics() -> None:
    document = _load()
    schema = strict_json_load(DEFAULT_SCOPE_SCHEMA)
    Draft202012Validator(schema).validate(document)
    assert load_stage_d_scope_decision() == document
    assert document["canonical_digest"] == scope_decision_digest(document)
    assert document["canonical_digest"] == D0_SCOPE_CANONICAL_DIGEST


def test_scope_import_does_not_load_model_or_torch_stack() -> None:
    assert os.environ.get("VITNNS_D0_STRICT") == "1"
    assert os.environ.get("VITNNS_D0_GUARD_ACTIVE") == "1"
    assert os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD") == "1"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    loaded = set(sys.modules)
    bad = sorted(
        name for name in loaded
        if any(name == prefix or name.startswith(prefix + ".")
               for prefix in FORBIDDEN_MODULE_PREFIXES)
    )
    assert bad == []
    with pytest.raises(ImportError, match="D0 strict guard blocked import"):
        importlib.import_module("torch")
    with pytest.raises(PermissionError, match="blocked image/model path"):
        Path("d0_guard_probe.JPEG").stat()
    with pytest.raises(PermissionError, match="blocked reserve access"):
        (REPO_ROOT / "manifests/stage_c_plus_v2").stat()
    with pytest.raises(PermissionError, match="blocked subprocess"):
        subprocess.Popen(["true"])


def test_scope_document_and_machine_record_agree() -> None:
    document = _load()
    markdown = DEFAULT_SCOPE_DOCUMENT.read_text(encoding="utf-8")
    for fragment in required_scope_document_fragments(document):
        assert fragment in markdown, fragment
    assert extract_scope_document_projection(markdown) == scope_document_projection(document)
    assert "C10 `n=15` 与 Stage D planned `n=500` 分开报告" in markdown
    assert "run_stability=false" in markdown
    assert "fidelity 角色是 `PRIMARY_HARD_SCIENTIFIC_GATE`" in markdown
    assert "D1 才能在 metadata-only 边界内生成正式 registry" in markdown
    assert "D0_INITIAL_ATTEMPT_20260825" in markdown
    assert "D0_MODEL_FREE_RERUN_20260825" in markdown
    assert "历史失败没有被改写" in markdown


def test_scope_markdown_projection_tamper_is_rejected() -> None:
    document = _load()
    markdown = DEFAULT_SCOPE_DOCUMENT.read_text(encoding="utf-8")
    tampered = markdown.replace(
        '"execution_authorization": "NOT_ISSUED"',
        '"execution_authorization": "AUTHORIZED_STAGE_D_FIDELITY_SCOPE"',
        1,
    )
    assert extract_scope_document_projection(tampered) != scope_document_projection(document)


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda d: d.update({"unexpected": True}), "unknown"),
        (lambda d: d.pop("authority"), "missing"),
        (lambda d: d["authority"].update({"execution_authorization": "AUTHORIZED_STAGE_D_FIDELITY_SCOPE"}), "frozen"),
        (lambda d: d["access_attestation"].update({"reserve_image_bytes_accessed": True}), "frozen"),
        (lambda d: d["stage_d_scope"].update({"run_stability": True}), "frozen"),
    ],
)
def test_scope_rejects_schema_and_authorization_drift(mutate, expected: str) -> None:
    document = copy.deepcopy(_load())
    mutate(document)
    _resign(document)
    with pytest.raises(StageDScopeError, match=expected):
        validate_stage_d_scope_decision(document)


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (
            lambda d: d["attempt_history"]["records"][0]["access_facts"].update(
                {"model_inference_performed": False}
            ),
            "attempt history drifted",
        ),
        (
            lambda d: d["attempt_history"]["records"][0].update(
                {"status": "PASS"}
            ),
            "attempt history drifted",
        ),
        (
            lambda d: d["verification_policy"].update(
                {"full_regression_suite_allowed": True}
            ),
            "frozen value",
        ),
    ],
)
def test_scope_rejects_attempt_history_or_allowlist_tamper(
    mutate, expected: str,
) -> None:
    document = copy.deepcopy(_load())
    mutate(document)
    _resign(document)
    with pytest.raises(StageDScopeError, match=expected):
        validate_stage_d_scope_decision(document)


def test_scope_rejects_unsigned_tamper() -> None:
    document = copy.deepcopy(_load())
    document["scientific_basis"]["c10_effect"]["positive_count"] = 14
    with pytest.raises(StageDScopeError, match="canonical digest mismatch"):
        validate_stage_d_scope_decision(document)


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda d: d["source_c10"]["historical_status"].update({"primary_decision": "GO"}), "frozen value"),
        (lambda d: d["source_c10"]["historical_status"].update({"downstream_authorized": True}), "frozen value"),
        (lambda d: d["source_c10"]["whole_root_inventory"].update({"canonical_digest": "0" * 64}), "frozen value"),
        (lambda d: d["stage_d_scope"].update({"result_status": "PASS_FIDELITY"}), "frozen value"),
        (lambda d: d["claim_boundaries"]["allowed_claim_ids"].append("C10_GO"), "too many items"),
        (lambda d: d["mandatory_disclosures"][0].update({"fact": "tampered"}), "trusted D0 semantic digest"),
        (lambda d: d["stage_d_scope"]["model"].update({"state_dict_sha256": "0" * 64}), "frozen value"),
    ],
)
def test_scope_rejects_resigned_semantic_tamper(mutate, expected: str) -> None:
    document = copy.deepcopy(_load())
    mutate(document)
    _resign(document)
    with pytest.raises(StageDScopeError, match=expected):
        validate_stage_d_scope_decision(document)


def test_strict_loader_rejects_duplicate_and_nonfinite_json(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a": 1, "a": 2}', encoding="utf-8")
    with pytest.raises(StageDScopeError, match="duplicate JSON key"):
        strict_json_load(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a": NaN}', encoding="utf-8")
    with pytest.raises(StageDScopeError, match="non-finite JSON number"):
        strict_json_load(nonfinite)


def test_claim_sets_and_access_attestation_are_exact() -> None:
    document = _load()
    claims = document["claim_boundaries"]
    assert tuple(claims["allowed_claim_ids"]) == ALLOWED_CLAIM_IDS
    assert tuple(claims["denied_claim_ids"]) == DENIED_CLAIM_IDS
    assert tuple(claims["emitted_claim_ids"]) == ALLOWED_CLAIM_IDS
    assert tuple(claims["future_conditional_claim_ids"]) == FUTURE_CONDITIONAL_CLAIM_IDS
    assert set(ALLOWED_CLAIM_IDS).isdisjoint(DENIED_CLAIM_IDS)
    assert set(FUTURE_CONDITIONAL_CLAIM_IDS).isdisjoint(claims["emitted_claim_ids"])
    assert tuple(item["id"] for item in document["mandatory_disclosures"]) == MANDATORY_DISCLOSURE_IDS
    spatial_disclosure = next(
        item for item in document["mandatory_disclosures"]
        if item["id"] == "DISCLOSE_C10_SPATIAL_MIXED_RESULT"
    )
    assert spatial_disclosure["required_locations"] == [
        "study_design", "results", "limitations"
    ]

    access = document["access_attestation"]
    assert access["attestation_scope"] == "CURRENT_D0_MODEL_FREE_RERUN_ONLY"
    assert access["reserve_access_state"] == "METADATA_EXISTENCE_ONLY"
    assert access["reserve_metadata_contents_accessed"] is False
    assert access["reserve_image_bytes_accessed"] is False
    assert access["reserve_result_accessed"] is False
    assert access["model_or_gpu_used"] is False
    assert access["model_loaded"] is False
    assert access["model_inference_performed"] is False
    assert access["jpeg_decode_performed"] is False
    assert access["non_reserve_jpeg_bytes_accessed"] is False
    assert access["c9_evidence_read"] is False
    assert access["c10_scientific_results_recomputed"] is False

    attempts = document["attempt_history"]["records"]
    assert attempts[0]["status"] == "BLOCKED"
    assert attempts[0]["access_facts"]["model_inference_performed"] is True
    assert attempts[0]["access_facts"]["jpeg_decode_performed"] is True
    assert attempts[0]["access_facts"]["reserve_image_bytes_accessed"] is False
    assert attempts[1]["status"] == "PASS"
    assert attempts[1]["authoritative_for_current_authorization"] is True

    policy = document["verification_policy"]
    assert len(policy["allowed_pytest_node_ids"]) == 12
    assert all(
        item.startswith("tests/test_stage_d_scope_contract.py::test_")
        for item in policy["allowed_pytest_node_ids"]
    )
    assert policy["full_regression_suite_allowed"] is False
    assert policy["pytest_plugin_autoload"] is False
    assert policy["safety_guard_path"] == "tests/d0_guard/sitecustomize.py"
    assert "torch" in policy["model_and_image_stack_imports_forbidden"]


def test_sealed_c10_hash_manifest_status_archive_and_inventory_chain() -> None:
    result = verify_c10_sealed_evidence()
    assert result["status"] == "PASS"
    assert result["file_count"] == 12
    assert result["total_bytes"] == 524_796_892
    assert result["whole_root_inventory_digest"] == C10_ROOT_INVENTORY_DIGEST
    assert result["producer_commit"] == "92ad81342ff22e2710df495a9961a3ab8c015a3a"
    assert result["scientific_results_recomputed"] is False


def test_sealed_c10_tamper_is_rejected_without_reading_large_scientific_payloads(
    tmp_path: Path,
) -> None:
    source = Path(
        "/data00/home/yeziqi.831/testcode/vitnns_c10_formal_92ad813_20260824"
    )
    for name in ("status.json", "stage_c_report.json", "decision.json", "holm.json", "provenance.json", "manifest.json"):
        (tmp_path / name).write_bytes((source / name).read_bytes())
    for name in ("bootstrap.json", "per_sample.csv", "per_sample.schema.json", "per_sample_manifest.json", "reconciliation.json", "summaries.json"):
        (tmp_path / name).write_bytes(b"tampered")
    with pytest.raises(StageDScopeError, match="whole-root inventory drifted"):
        verify_c10_sealed_evidence(tmp_path)


def test_sealed_c10_rejects_root_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(StageDScopeError, match="symlink"):
        verify_c10_sealed_evidence(link)


def test_sealed_c10_rejects_file_symlink(tmp_path: Path) -> None:
    source = Path(
        "/data00/home/yeziqi.831/testcode/vitnns_c10_formal_92ad813_20260824"
    )
    for name in ("status.json", "stage_c_report.json", "decision.json", "holm.json", "provenance.json", "manifest.json"):
        (tmp_path / name).write_bytes((source / name).read_bytes())
    target = tmp_path.parent / f"{tmp_path.name}-symlink-target"
    target.write_bytes(b"not a sealed payload")
    for name in ("bootstrap.json", "per_sample.csv", "per_sample.schema.json", "per_sample_manifest.json", "reconciliation.json", "summaries.json"):
        (tmp_path / name).symlink_to(target)
    with pytest.raises(StageDScopeError, match="symlink"):
        verify_c10_sealed_evidence(tmp_path)


def test_sealed_c10_rejects_hardlink(tmp_path: Path) -> None:
    source = Path(
        "/data00/home/yeziqi.831/testcode/vitnns_c10_formal_92ad813_20260824"
    )
    for name in ("status.json", "stage_c_report.json", "decision.json", "holm.json", "provenance.json", "manifest.json"):
        (tmp_path / name).write_bytes((source / name).read_bytes())
    target = tmp_path.parent / f"{tmp_path.name}-hardlink-target"
    target.write_bytes(b"not a sealed payload")
    for name in ("bootstrap.json", "per_sample.csv", "per_sample.schema.json", "per_sample_manifest.json", "reconciliation.json", "summaries.json"):
        (tmp_path / name).hardlink_to(target)
    with pytest.raises(StageDScopeError, match="single-link regular file"):
        verify_c10_sealed_evidence(tmp_path)


def test_sealed_c10_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    source = Path(
        "/data00/home/yeziqi.831/testcode/vitnns_c10_formal_92ad813_20260824"
    )
    for name in ("status.json", "stage_c_report.json", "decision.json", "holm.json", "provenance.json", "manifest.json"):
        (tmp_path / name).write_bytes((source / name).read_bytes())
    for name in ("bootstrap.json", "per_sample.csv", "per_sample.schema.json", "per_sample_manifest.json", "reconciliation.json"):
        (tmp_path / name).write_bytes(b"tampered")
    os.mkfifo(tmp_path / "summaries.json")
    with pytest.raises(StageDScopeError, match="single-link regular file"):
        verify_c10_sealed_evidence(tmp_path)


def test_d0_static_files_keep_execution_and_reserve_closed() -> None:
    document = _load()
    assert document["authority"]["preparation_authorization"] == "APPROVED_TO_PREPARE_STAGE_D"
    assert document["authority"]["execution_authorization"] == "NOT_ISSUED"
    assert document["authority"]["analysis_authorization"] == "NOT_EVALUATED"
    assert document["stage_d_scope"]["result_status"] == "NOT_EVALUATED"
    assert document["stage_d_scope"]["run_stability"] is False
    assert document["blocking_reasons"] == []
    assert document["task_status"] == "PASS"

    roadmap = STEP_BY_STEP.read_text(encoding="utf-8")
    plan = TOP_LEVEL_PLAN.read_text(encoding="utf-8")
    rebuild_readme = REBUILD_README.read_text(encoding="utf-8")
    top_readme = TOP_LEVEL_README.read_text(encoding="utf-8")
    assert "D0 | PASS（首次尝试 BLOCKED；严格重试 PASS）" in roadmap
    assert "当前 next task：`D6`" in roadmap
    assert "| D1 | PASS |" in roadmap
    assert "| D2 | PASS |" in roadmap
    assert "| D5 | PASS |" in roadmap
    assert "D0_INITIAL_ATTEMPT_20260825" in roadmap
    assert "D0_MODEL_FREE_RERUN_20260825" in roadmap
    assert "D0_NO_MODEL_OR_INFERENCE_GATE_VIOLATED" in roadmap
    assert "D0_NO_JPEG_DECODE_GATE_VIOLATED" in roadmap
    assert D0_SCOPE_CANONICAL_DIGEST in roadmap
    current_handoff = roadmap.split(
        "#### 当前权威的下一对话完整提示词", 1
    )[1].split("### 2026-08-25 13:49 +0800 — D1 completion record", 1)[0]
    assert D0_SCOPE_CANONICAL_DIGEST in current_handoff
    assert "8fff3955c488a344262e5e0fde5dfc1e9178de6f25dda5b866010c8c0861c3ce" not in current_handoff
    assert "只执行 ViT-NNS fidelity-only Stage D 路线的 D1" in current_handoff
    assert "C10 v1 primary decision 为 `NO_GO_NOVELTY`" in roadmap
    assert "Stage D 不运行、不分析新的 spatial/stability endpoint" in roadmap
    assert "Stage C acceptance 为 `BLOCKED`" in plan
    assert "Stage D 不新增 spatial endpoint" in plan
    assert "D0 首次尝试" in top_readme
    assert "model-free/no-JPEG D0 重试" in top_readme
    assert "initial D0 attempt" in rebuild_readme
    assert "strict model-free/no-JPEG D0 rerun passed" in rebuild_readme
