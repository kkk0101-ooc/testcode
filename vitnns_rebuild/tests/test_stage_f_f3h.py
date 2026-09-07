from __future__ import annotations

import ast
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from vitnns import stage_f_f3h as f3h


REPO = Path(__file__).resolve().parents[1]


def test_plan_is_complete_block_major_and_source_recomputable() -> None:
    left = f3h.build_plan(REPO); right = f3h.build_plan(REPO)
    assert left == right
    assert left["coordinates"] == [[block, head] for block in range(12) for head in range(12)]
    assert left["counts"] == {"coordinates": 144, "value_path_terminals": 288, "inherited_valid": 8, "new_value_path": 280, "gradcam": 1}
    assert Counter((u["block"], u["head"], u["method"]) for u in left["value_path_units"]) == Counter((b, h, m) for b, h in f3h.COORDINATES for m in f3h.METHODS)


def test_exactly_four_coordinate_pairs_are_inherited() -> None:
    plan = f3h.build_plan(REPO)
    inherited = [unit for unit in plan["value_path_units"] if unit["mode"] == "INHERITED_READ_ONLY"]
    assert len(inherited) == 8
    assert {(u["block"], u["head"]) for u in inherited} == set(f3h.INHERITED_COORDINATES)
    assert all(unit["source"] is not None for unit in inherited)
    assert all(unit["source"]["source_path"] for unit in inherited)


def test_bulk_shards_are_disjoint_complete_and_pair_preserving() -> None:
    shards = {index: set(f3h._bulk_coordinates(index, 8)) for index in range(8)}
    expected = set(f3h.COORDINATES) - set(f3h.INHERITED_COORDINATES) - set(f3h.PILOT_COORDINATES)
    assert set().union(*shards.values()) == expected
    assert sum(map(len, shards.values())) == len(expected) == 136
    assert not any(shards[left] & shards[right] for left in range(8) for right in range(left + 1, 8))
    with pytest.raises(f3h.StageFF3HError): f3h._bulk_coordinates(8, 8)
    with pytest.raises(f3h.StageFF3HError): f3h._bulk_coordinates(0, 7)


def test_sign_split_is_exact_and_rejects_invalid_input() -> None:
    raw = np.zeros((56, 56), dtype=np.float32); raw[0, 0] = 2; raw[1, 1] = -3
    positive, negative = f3h.split_signed_map(raw)
    np.testing.assert_array_equal(positive - negative, raw)
    assert positive[0, 0] == 2 and negative[1, 1] == 3
    with pytest.raises(f3h.StageFF3HError): f3h.split_signed_map(raw.astype(np.float64))
    with pytest.raises(f3h.StageFF3HError): f3h.split_signed_map(np.zeros((14, 14), dtype=np.float32))


def test_pair_shared_log_transform_has_one_scale_and_no_gain() -> None:
    nns = np.zeros((56, 56), dtype=np.float32); matched = np.zeros_like(nns)
    nns[0, 0] = 10; nns[1, 1] = -5; matched[2, 2] = 1; matched[3, 3] = -2
    shown, record = f3h.pair_shared_log_display(nns, matched)
    assert record["pair_scale"] == 10
    assert record["method_specific_gain"] is False
    assert record["paired_magnitude_comparison_allowed"] is True
    assert all(array.shape == (224, 224) and array.dtype == np.float32 for array in shown.values())
    assert shown["nns_positive"].max() == pytest.approx(1.0)
    assert shown["matched_positive"].max() < shown["nns_positive"].max()


def test_pair_shared_log_is_sign_symmetric_and_zero_safe() -> None:
    nns = np.zeros((56, 56), dtype=np.float32); matched = np.zeros_like(nns)
    nns[20, 20] = 2; matched[20, 20] = -2
    shown, _ = f3h.pair_shared_log_display(nns, matched)
    np.testing.assert_array_equal(shown["nns_positive"], shown["matched_negative"])
    zero, record = f3h.pair_shared_log_display(np.zeros_like(nns), np.zeros_like(nns))
    assert record["scale_basis"] == "ZERO_PAIR_SCALE"
    assert all(not np.count_nonzero(value) for value in zero.values())


def test_implementation_validation_binds_existing_kernels_and_never_starts_f4() -> None:
    result = f3h.validate_implementation(REPO)
    assert result["status"] == "PASS"
    assert result["counts"]["new_value_path"] == 280
    assert result["render_count"] == 20
    assert result["formal_gpu_execution_started"] is False
    assert result["f4_status"] == "PAUSED_PENDING_F3H_AND_F3C"
    assert f3h.F3H_RENDER_COUNT == 20 and f3h.F3H_EXPORT_COUNT == 60


def test_implementation_false_pass_audit_is_complete() -> None:
    audit = f3h.implementation_false_pass_audit()
    assert len(audit["attacks"]) == len(f3h.IMPLEMENTATION_ATTACKS) == 12
    assert all(item["result"] == "REJECTED" for item in audit["attacks"])


def test_production_execution_is_map_only_by_source_inspection() -> None:
    source = (REPO / "src/vitnns/stage_f_f3h.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    execute = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_execute_coordinates")
    segment = ast.get_source_segment(source, execute) or ""
    assert "_curves_for_method" not in segment
    assert "evaluate_fidelity_curve" not in segment
    assert "resolve_target_response" not in segment
    assert "e3_r1_conditional_fixed_clean_attention_head_direction_value_path_ig" in segment
    assert "matched_fixed_clean_attention_patch_token_contribution" in segment


def test_scientific_invalidities_are_typed_but_unknown_errors_are_not_caught() -> None:
    source = (REPO / "src/vitnns/stage_f_f3h.py").read_text(encoding="utf-8")
    tree = ast.parse(source); execute = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_execute_coordinates")
    caught = {ast.unparse(handler.type) for node in ast.walk(execute) if isinstance(node, ast.Try) for handler in node.handlers if handler.type is not None}
    assert "StageCValuePathConvergenceError" in caught
    assert "StageCValuePathScientificInvalidity" in caught
    assert "Exception" not in caught and "BaseException" not in caught


def _synthetic_unit(digest: str) -> dict[str, object]:
    return {"canonical_digest": digest, "sample_id": "ILSVRC2012_val/example.JPEG", "selection_order": 44, "block": 0, "head": 0, "method": f3h.NNS, "target": "fixed_clean_attention_cls_head_direction", "protocol_id": "test"}


def _synthetic_result(unit: dict[str, object], status: str, map_digest: str | None) -> dict[str, object]:
    return {"unit_digest": unit["canonical_digest"], "sample_id": unit["sample_id"], "selection_order": 44, "block": 0, "head": 0, "method": f3h.NNS, "validity_status": status, "source_hashes": {"signed_map_sha256": map_digest}}


def test_atomic_valid_terminal_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(f3h, "ROOT_PATH", "formal")
    unit = _synthetic_unit("a" * 64); array = np.arange(56 * 56, dtype=np.float32).reshape(56, 56)
    f3h._publish_terminal(tmp_path, unit, _synthetic_result(unit, "VALID", f3h._array_digest(array)), array)
    observed = f3h.validate_attempt(tmp_path, unit)
    assert observed["result"]["validity_status"] == "VALID"
    assert observed["manifest"]["map"]["array_digest"] == f3h._array_digest(array)
    with pytest.raises(f3h.StageFF3HError, match="overwrite"):
        f3h._publish_terminal(tmp_path, unit, _synthetic_result(unit, "VALID", f3h._array_digest(array)), array)


def test_failed_terminal_has_no_imputed_map(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(f3h, "ROOT_PATH", "formal")
    unit = _synthetic_unit("b" * 64)
    f3h._publish_terminal(tmp_path, unit, _synthetic_result(unit, "FAILED", None), None)
    observed = f3h.validate_attempt(tmp_path, unit)
    assert observed["result"]["validity_status"] == "FAILED"
    assert observed["manifest"]["map"] is None
    assert not (observed["root"] / "map.npy").exists()


def test_consensus_is_equal_weight_and_fail_closed_on_missing() -> None:
    arrays = {}
    for head in range(12): arrays[(0, head, f3h.NNS, "positive")] = np.full((224,224), head, dtype=np.float32)
    result = f3h._consensus(arrays, [(0, head) for head in range(12)], f3h.NNS, "positive")
    assert result is not None and float(result[0, 0]) == pytest.approx(5.5)
    arrays[(0, 4, f3h.NNS, "positive")] = None
    assert f3h._consensus(arrays, [(0, head) for head in range(12)], f3h.NNS, "positive") is None


def test_consensus_diagnostics_do_not_select_heads() -> None:
    identical = [np.ones((4, 4), dtype=np.float32) for _ in range(12)]
    assert f3h._mean_pairwise_cosine(identical) == pytest.approx(1.0)
    assert f3h._mean_pairwise_cosine(identical[:-1] + [None]) is None
    entropy = f3h._normalized_entropy(np.ones((4, 4), dtype=np.float32))
    assert entropy == pytest.approx(1.0)


def test_class_conditioned_basis_rule_cannot_use_visual_head_selection() -> None:
    text = (REPO / f3h.F3H1_ADDENDUM).read_text(encoding="utf-8")
    for phrase in ("all 144 signed class-logit coupling coefficients", "predeclared conservation test", "cannot by itself authorize numerical class-conditioned weights", "attention rollout", "block-11 raw CLS attention head mean"):
        assert phrase in text
    assert "equal-head-weighted spatial-support consensus" in text
    assert "INSUFFICIENT_BASIS_FOR_NUMERICAL_CLASS_CONDITIONED_WEIGHTS" in text


def test_f3h_completion_if_present_is_atomic() -> None:
    if not (REPO / f3h.COMPLETION_PATH).exists(): return
    result=f3h.validate_completion(REPO);assert result["status"]=="PASS";assert result["next_authorization"]=="F3C0_PROTOCOL_FREEZE_ONLY"
