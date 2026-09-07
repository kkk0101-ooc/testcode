from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

from vitnns.stage_f_f3h_protocol import (
    ATTACKS, COMPLETION_PATH, CONFIG_PATH, PROTOCOL_PATH,
    StageFF3HProtocolError, build_false_pass_audit, load_and_validate_config,
    validate_completion, validate_contract_semantics, validate_protocol,
)


REPO = Path(__file__).resolve().parents[1]


def test_protocol_config_is_signed_and_exact() -> None:
    config = load_and_validate_config(REPO)
    assert config["protocol_id"] == "STAGE_F_F3H_FULL_HEAD_GRADCAM_V1_20260907"
    assert config["status"] == "FROZEN_PROTOCOL_EXECUTION_NOT_STARTED"
    assert config["authorization"]["f4_status"] == "PAUSED_PENDING_F3H"
    assert config["authorization"]["formal_gpu_execution_authorized_now"] is False


def test_full_head_plan_counts_and_order_are_frozen() -> None:
    value = load_and_validate_config(REPO)["value_path"]
    assert value["blocks"] == list(range(12))
    assert value["heads"] == list(range(12))
    assert value["coordinate_order"] == "BLOCK_MAJOR_THEN_HEAD_ASCENDING"
    assert value["coordinate_count"] == 144
    assert value["expected_terminal_count"] == 288
    assert value["inherited_valid_terminals"] == 8
    assert value["new_terminal_units"] == 280


def test_sample_model_and_gradcam_target_are_frozen() -> None:
    config = load_and_validate_config(REPO)
    assert config["sample"]["selection_order"] == 44
    assert config["sample"]["sealed_clean_top1_index"] == 713
    assert config["model"]["state_dict_sha256"] == "489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d"
    assert config["model"]["historical_2e718_checkpoint_allowed"] is False
    assert config["gradcam"]["frozen_class_index"] == 713
    assert config["gradcam"]["same_target_as_value_path"] is False
    assert config["gradcam"]["same_target_winner_comparison_allowed"] is False


def test_pair_shared_log_has_no_method_specific_advantage() -> None:
    display = load_and_validate_config(REPO)["display"]
    assert display["components"] == ["positive", "negative_magnitude"]
    assert display["pair_scale_formula"] == "max(max_abs_nns,max_abs_matched)"
    assert display["kappa"] == 999.0
    assert all(display[key] is False for key in ("method_specific_gain_allowed", "method_specific_gamma_allowed", "method_specific_alpha_allowed", "separate_pair_autoscale_allowed"))
    assert display["clipping"] is None and display["smoothing"] is None and display["max_pool"] is None


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("value_path", "heads"), list(range(11))),
        (("rendering", "best_head_selection_allowed"), True),
        (("display", "method_specific_gain_allowed"), True),
        (("display", "separate_pair_autoscale_allowed"), True),
        (("gradcam", "same_target_as_value_path"), True),
        (("gradcam", "algorithm_minmax_normalization"), True),
        (("budget", "waiver_allowed"), True),
        (("authorization", "f4_started"), True),
        (("value_path", "run_fidelity"), True),
    ],
)
def test_semantic_validator_rejects_false_pass_mutations(path: tuple[str, str], value: object) -> None:
    config = copy.deepcopy(load_and_validate_config(REPO))
    config[path[0]][path[1]] = value
    with pytest.raises(StageFF3HProtocolError):
        validate_contract_semantics(config)


def test_live_protocol_validation_is_cpu_only_and_f4_paused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    result = validate_protocol(REPO)
    assert result["status"] == "PASS"
    assert result["f4_status"] == "PAUSED_PENDING_F3H"
    assert result["new_value_path_terminals"] == 280
    assert result["runtime"]["model_file_identity_hash_read"] is True
    assert all(result["runtime"][key] is False for key in ("model_deserialized", "model_loaded", "torch_imported", "timm_imported", "cuda_or_gpu_access", "jpeg_decoded", "attribution_executed"))


def test_false_pass_audit_enumerates_every_attack(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    audit = build_false_pass_audit(REPO)
    assert [item["attack"] for item in audit["attacks"]] == list(ATTACKS)
    assert len(audit["attacks"]) == 20
    assert all(item["result"] == "REJECTED" for item in audit["attacks"])
    assert audit["pass_possible_without_all_conditions"] is False


def test_protocol_source_has_no_torch_timm_or_image_import() -> None:
    tree = ast.parse((REPO / "src/vitnns/stage_f_f3h_protocol.py").read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert not imports.intersection({"torch", "torchvision", "timm", "PIL", "cv2"})


def test_protocol_text_forbids_cherry_pick_and_same_target_claim() -> None:
    text = (REPO / PROTOCOL_PATH).read_text(encoding="utf-8")
    for phrase in ("12 个 blocks × 12 个 heads", "不得生成只包含“最好看 heads”的精选面板", "不同 target", "不得人为“调亮 NNS、调暗 matched”", "PAUSED_PENDING_F3H"):
        assert phrase in text


def test_completion_is_atomic_if_present() -> None:
    if not (REPO / COMPLETION_PATH).exists():
        return
    result = validate_completion(REPO)
    assert result["status"] == "PASS"
    assert result["f4_status"] == "PAUSED_PENDING_F3H"
    assert result["formal_gpu_execution_started"] is False
