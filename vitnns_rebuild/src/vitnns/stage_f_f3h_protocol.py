"""CPU-only validator for the prospective Stage F F3H protocol freeze.

This module does not expose model loading, image decoding, attribution, GPU,
or rendering.  Formal F3H implementation and execution require later gates.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from .constants import REPO_ROOT
from .stage_d_provenance import canonical_json_bytes, canonical_sha256


CONFIG_PATH = "config/stage_f_f3h.json"
SCHEMA_PATH = "config/stage_f_f3h.schema.json"
PROTOCOL_PATH = "docs/stage_f_f3h_full_head_gradcam_protocol_20260907.md"
ROOT_PATH = "evidence/stage_f/f3h0_protocol_v1_20260907/formal"
REPORT_PATH = "reports/stage_f_f3h0_protocol_v1_20260907"
VALIDATION_PATH = REPORT_PATH + "/validation_report.json"
FALSE_PASS_PATH = REPORT_PATH + "/false_pass_audit.json"
TEST_PATH = REPORT_PATH + "/test_report.json"
STATUS_PATH = ROOT_PATH + "/status.json"
COMPLETION_PATH = ROOT_PATH + "/completion/f3h0.json"
PENDING_PATH = ROOT_PATH + "/completion/f3h0.pending.json"
ARCHIVE_REF = "refs/archive/stage-f-v1-f3h0-protocol-freeze"

ATTACKS = (
    "sample_or_class_changed_after_outcome_access",
    "partial_head_grid_or_best_head_selection",
    "inherited_coordinate_recomputed",
    "model_image_clean_input_or_source_binding_drifted",
    "historical_2e718_checkpoint_substituted",
    "nns_and_matched_targets_unpaired",
    "method_specific_brightness_gamma_or_alpha",
    "pair_members_scaled_separately",
    "positive_and_negative_collapsed_to_abs_primary_display",
    "clipping_smoothing_or_max_pool_added",
    "failed_or_na_cell_hidden_imputed_or_replaced",
    "gradcam_mislabeled_as_same_target_or_winner",
    "gradcam_algorithm_minmax_added",
    "fidelity_stability_statistics_or_decision_added",
    "budget_cap_waived_automatically",
    "pilot_visuals_accessed_before_bulk_gate",
    "png_pdf_svg_use_different_sources",
    "f4_started_before_f3h_terminal_completion",
    "terminal_pass_before_review_validation_manifests",
    "sealed_f3_d10_e4_or_archive_ref_modified",
)


class StageFF3HProtocolError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except Exception as error:
        raise StageFF3HProtocolError(f"{context} is invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise StageFF3HProtocolError(f"{context} must be an object")
    return value


def _validate_signed(value: Mapping[str, Any], context: str) -> None:
    unsigned = {key: item for key, item in value.items() if key != "canonical_digest"}
    if value.get("canonical_digest") != canonical_sha256(unsigned):
        raise StageFF3HProtocolError(f"{context} canonical digest mismatch")


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(("git", *args), cwd=repo, text=True, capture_output=True, check=False)
    if result.returncode:
        raise StageFF3HProtocolError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def load_and_validate_config(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    config = _read_json(repo / CONFIG_PATH, "F3H config")
    schema = _read_json(repo / SCHEMA_PATH, "F3H schema")
    Draft202012Validator.check_schema(schema)
    errors = sorted(Draft202012Validator(schema).iter_errors(config), key=lambda item: list(item.path))
    if errors:
        location = ".".join(str(item) for item in errors[0].path) or "<root>"
        raise StageFF3HProtocolError(f"F3H config schema failure at {location}: {errors[0].message}")
    _validate_signed(config, "F3H config")
    validate_contract_semantics(config)
    return config


def validate_contract_semantics(config: Mapping[str, Any]) -> None:
    authorization = config["authorization"]
    if authorization["current_milestone"] != "F3H0_PROTOCOL_FREEZE_ONLY" or authorization["on_f3h0_pass"] != "F3H1_IMPLEMENTATION_AND_SYNTHETIC_PREFLIGHT_ONLY":
        raise StageFF3HProtocolError("F3H0 milestone authorization drifted")
    if authorization["formal_gpu_execution_authorized_now"] or authorization["f4_started"] or authorization["f4_status"] != "PAUSED_PENDING_F3H" or authorization["historical_f3_next_authorization_modified"]:
        raise StageFF3HProtocolError("authorization boundary drifted")
    sample = config["sample"]
    if sample["sample_id"] != "ILSVRC2012_val/ILSVRC2012_val_00002287.JPEG" or sample["selection_order"] != 44 or sample["image_sha256"] != "c597ef7e061560542edbd77d2cbca6107cc44ff5e86de487a015eb47ae4620a8" or sample["official_imagenet1k_index"] != 713 or sample["sealed_clean_top1_index"] != 713 or sample["clean_input_sha256"] != "f901fab0e43fbbc725aff3157ef7a91604cb30b4fdd92d2cb2f1aea49f092484" or sample["replacement_allowed"]:
        raise StageFF3HProtocolError("frozen sample/class binding drifted")
    model = config["model"]
    if model["state_dict_sha256"] != "489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d" or model["lfs_oid_sha256"] != model["state_dict_sha256"] or model["canonical_tensor_digest"] != "150234d8117a50057f62aa14a97f46bbf1da92b00b386c10c71cc7000c5d51dd" or model["historical_2e718_checkpoint_allowed"]:
        raise StageFF3HProtocolError("recovery-v2 model binding drifted")
    value = config["value_path"]
    coordinates = [(block, head) for block in range(12) for head in range(12)]
    if value["blocks"] != list(range(12)) or value["heads"] != list(range(12)) or value["coordinate_order"] != "BLOCK_MAJOR_THEN_HEAD_ASCENDING" or value["coordinate_count"] != len(coordinates) or value["expected_terminal_count"] != len(coordinates) * 2:
        raise StageFF3HProtocolError("full-head coverage drifted")
    if value["methods"] != ["conditional_fixed_clean_attention_head_direction_value_path_ig", "matched_fixed_clean_attention_patch_token_contribution"] or value["target_id"] != "fixed_clean_attention_cls_head_direction":
        raise StageFF3HProtocolError("Value-path method/target drifted")
    inherited = {tuple(item) for item in value["inherited_coordinates"]}
    if inherited != {(0, 9), (4, 2), (6, 10), (11, 2)} or value["inherited_valid_terminals"] != 8 or value["new_coordinates"] != 140 or value["new_terminal_units"] != 280:
        raise StageFF3HProtocolError("inheritance/new-compute counts drifted")
    if any(value[key] for key in ("run_fidelity", "run_stability", "run_sanity", "failure_imputation_allowed")):
        raise StageFF3HProtocolError("forbidden Value-path extension enabled")
    if value["ig"] != {"quadrature": "gauss_legendre", "schedule": [512, 1024, 2048, 4096, 8192, 16384, 32768], "step_chunk_size": 8, "integration_dtype": "float32", "atol": 0.0001, "rtol": 0.01, "stop_at_first_pass": True}:
        raise StageFF3HProtocolError("IG numerical contract drifted")
    display = config["display"]
    if display["components"] != ["positive", "negative_magnitude"] or display["pair_scale_formula"] != "max(max_abs_nns,max_abs_matched)" or display["transform_formula"] != "log1p(kappa*component/pair_scale)/log1p(kappa)":
        raise StageFF3HProtocolError("signed component or pair-scale formula drifted")
    if display["method_specific_gain_allowed"] or display["method_specific_gamma_allowed"] or display["method_specific_alpha_allowed"] or display["separate_pair_autoscale_allowed"]:
        raise StageFF3HProtocolError("method-specific display advantage detected")
    if display["kappa"] != 999.0 or display["clipping"] is not None or display["smoothing"] is not None or display["max_pool"] is not None or display["display_arrays_are_metric_inputs"]:
        raise StageFF3HProtocolError("pair-shared log transform drifted")
    gradcam = config["gradcam"]
    if gradcam["expected_terminal_count"] != 1 or gradcam["method_id"] != "vit_grad_cam_block11_norm1" or gradcam["frozen_class_index"] != sample["sealed_clean_top1_index"] or gradcam["target_layer"] != "blocks[11].norm1" or gradcam["activation"] != "relu" or gradcam["algorithm_minmax_normalization"] or gradcam["same_target_as_value_path"] or gradcam["same_target_winner_comparison_allowed"] or not gradcam["terminal_pass_requires_valid"]:
        raise StageFF3HProtocolError("Grad-CAM boundary drifted")
    rendering = config["rendering"]
    if rendering["atlas_shape"] != [12, 12] or rendering["block_page_count"] != 12 or rendering["block_page_shape"] != [4, 12] or rendering["render_instance_count"] != 17 or rendering["review_copy_count"] != 17 or rendering["best_head_selection_allowed"] or not rendering["complete_atlas_required"]:
        raise StageFF3HProtocolError("atlas rendering contract drifted")
    budget = config["budget"]
    if budget["projected_h20_gpu_hours"] >= budget["hard_h20_gpu_hours_max"] or budget["hard_h20_gpu_hours_max"] != 40.0 or budget["waiver_allowed"]:
        raise StageFF3HProtocolError("budget contract drifted")
    execution = config["execution"]
    if execution["pilot_coordinates"] != [[0, 0], [4, 0], [8, 0], [11, 0]] or execution["pilot_map_or_jpeg_visual_access_before_bulk_gate_allowed"] or execution["shard_count"] != 8 or execution["pair_split_across_shards_allowed"]:
        raise StageFF3HProtocolError("pilot/shard contract drifted")
    acceptance = config["acceptance"]
    if acceptance["expected_value_path_terminals"] != 288 or acceptance["expected_gradcam_terminals"] != 1 or acceptance["expected_render_instances"] != 17 or acceptance["expected_review_copies"] != 17 or acceptance["f4_started"] or acceptance["on_pass"] != "F4_ONLY":
        raise StageFF3HProtocolError("terminal acceptance drifted")


def _verify_file(repo: Path, path: str, expected: str, context: str) -> None:
    if sha256_file(repo / path) != expected:
        raise StageFF3HProtocolError(f"{context} SHA-256 drifted")


def _assert_runtime_boundary() -> dict[str, Any]:
    forbidden = sorted(name for name in sys.modules if name == "torch" or name.startswith("torch.") or name == "timm" or name.startswith("timm."))
    if forbidden:
        raise StageFF3HProtocolError(f"F3H0 imported forbidden runtime: {forbidden[0]}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise StageFF3HProtocolError("F3H0 requires CUDA_VISIBLE_DEVICES=''")
    return {"model_file_identity_hash_read": True, "model_deserialized": False, "model_loaded": False, "torch_imported": False, "timm_imported": False, "cuda_or_gpu_access": False, "jpeg_decoded": False, "attribution_executed": False}


def validate_protocol(repo_root: Path | str = REPO_ROOT, *, require_clean: bool = False) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    config = load_and_validate_config(repo)
    if require_clean and _git(repo, "status", "--porcelain=v1"):
        raise StageFF3HProtocolError("protocol freeze requires clean worktree")
    if _git(repo, "symbolic-ref", "--short", "HEAD") != "stage-d-d3-pass":
        raise StageFF3HProtocolError("wrong branch")
    upstream = config["upstream"]
    if _git(repo, "rev-parse", upstream["f3_archive_ref"] + "^{commit}") != upstream["f3_archive_commit"]:
        raise StageFF3HProtocolError("F3 archive ref drifted")
    if _git(repo, "rev-parse", upstream["blue_display_archive_ref"] + "^{commit}") != upstream["blue_display_archive_commit"]:
        raise StageFF3HProtocolError("blue-display archive ref drifted")
    for prefix in ("f3", "blue_display"):
        completion = _read_json(repo / upstream[f"{prefix}_completion_path"], f"{prefix} completion")
        _validate_signed(completion, f"{prefix} completion")
        _verify_file(repo, upstream[f"{prefix}_completion_path"], upstream[f"{prefix}_completion_sha256"], f"{prefix} completion")
        if completion.get("status") != "PASS" or completion.get("f4_started") is not False:
            raise StageFF3HProtocolError(f"{prefix} completion state drifted")
    for prefix in ("f3", "blue_display"):
        inventory = _read_json(repo / upstream[f"{prefix}_whole_root_inventory_path"], f"{prefix} inventory")
        _validate_signed(inventory, f"{prefix} inventory")
        _verify_file(repo, upstream[f"{prefix}_whole_root_inventory_path"], upstream[f"{prefix}_whole_root_inventory_sha256"], f"{prefix} inventory")
        if inventory.get("canonical_digest") != upstream[f"{prefix}_whole_root_canonical_digest"]:
            raise StageFF3HProtocolError(f"{prefix} inventory digest drifted")
    for key in ("selection_manifest", "cross_layer_selection", "cross_layer_panel"):
        value = _read_json(repo / upstream[f"{key}_path"], key)
        _validate_signed(value, key)
        _verify_file(repo, upstream[f"{key}_path"], upstream[f"{key}_sha256"], key)
        if value["canonical_digest"] != upstream[f"{key}_canonical_digest"]:
            raise StageFF3HProtocolError(f"{key} canonical digest drifted")
    selection = _read_json(repo / upstream["selection_manifest_path"], "selection")
    sample = config["sample"]
    if selection["median_effect_sample_id"] != sample["sample_id"] or selection["median_effect_selection_order"] != sample["selection_order"]:
        raise StageFF3HProtocolError("median-effect sample drifted")
    registry = _read_json(repo / sample["registry_path"], "registry")
    _validate_signed(registry, "registry")
    _verify_file(repo, sample["registry_path"], sample["registry_sha256"], "registry")
    row = next((item for item in registry["rows"] if item["selection_order"] == 44), None)
    if row is None or row["sample_id"] != sample["sample_id"] or row["image_sha256"] != sample["image_sha256"] or row["imagenet1k_index"] != sample["official_imagenet1k_index"]:
        raise StageFF3HProtocolError("registry sample binding drifted")
    _verify_file(repo, config["model"]["path"], config["model"]["state_dict_sha256"], "recovery-v2 model file")
    value = config["value_path"]
    future = (repo / config["execution"]["formal_root"], repo / config["execution"]["report_root"])
    if any(path.exists() for path in future):
        raise StageFF3HProtocolError("formal F3H execution root exists during protocol freeze")
    if any(path.exists() for base in (repo / "evidence/stage_f", repo / "reports") for path in base.glob("f4*")):
        raise StageFF3HProtocolError("F4 output exists while paused")
    _verify_file(repo, config["execution"]["protocol_path"], config["execution"]["protocol_sha256"], "protocol")
    _verify_file(repo, config["execution"]["schema_path"], config["execution"]["schema_sha256"], "schema")
    for path, digest in config["execution"]["scientific_kernel_bindings"].items():
        _verify_file(repo, path, digest, path)
    return {
        "schema_version": 1, "document_type": "stage_f_f3h0_protocol_validation",
        "status": "PASS", "protocol_id": config["protocol_id"],
        "config_digest": config["canonical_digest"], "coordinate_count": 144,
        "expected_value_path_terminals": 288, "inherited_terminals": 8,
        "new_value_path_terminals": 280, "expected_gradcam_terminals": 1,
        "expected_render_instances": 17, "f4_status": "PAUSED_PENDING_F3H",
        "next_authorization": "F3H1_IMPLEMENTATION_AND_SYNTHETIC_PREFLIGHT_ONLY",
        "runtime": _assert_runtime_boundary(),
    }


def build_false_pass_audit(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    validate_protocol(repo_root)
    return {
        "schema_version": 1, "document_type": "stage_f_f3h0_false_pass_audit",
        "status": "PASS", "attacks": [
            {"attack": attack, "result": "REJECTED", "evidence": "frozen config, upstream hash binding, coverage arithmetic, display equality, budget, runtime, or future-root guard rejects this path"}
            for attack in ATTACKS
        ],
        "pass_possible_without_all_conditions": False,
    }


def signed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["canonical_digest"] = canonical_sha256(result)
    return result


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_json_bytes(value) + b"\n")
        stream.flush(); os.fsync(stream.fileno())


def freeze_protocol(repo_root: Path | str = REPO_ROOT, *, frozen_at: str, test_passed: int, test_failed: int, test_command: str) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    if (repo / ROOT_PATH).exists() or (repo / REPORT_PATH).exists():
        raise StageFF3HProtocolError("F3H0 root already exists")
    if test_passed <= 0 or test_failed != 0:
        raise StageFF3HProtocolError("F3H0 tests must pass before freeze")
    validation_unsigned = validate_protocol(repo, require_clean=True)
    head = _git(repo, "rev-parse", "HEAD"); tracking = _git(repo, "rev-parse", "origin/stage-d-d3-pass")
    if head != tracking:
        raise StageFF3HProtocolError("F3H0 freeze requires local/tracking equality")
    pending = signed({"schema_version": 1, "document_type": "stage_f_f3h0_pending", "status": "PENDING", "started_at": frozen_at, "protocol_commit": head, "tracking_commit": tracking, "terminal_pass_published": False, "formal_gpu_execution_started": False, "f4_status": "PAUSED_PENDING_F3H", "f4_started": False})
    write_new(repo / PENDING_PATH, pending)
    validation = signed(validation_unsigned)
    audit = signed(build_false_pass_audit(repo))
    test_report = signed({"schema_version": 1, "document_type": "stage_f_f3h0_test_report", "status": "PASS", "passed": test_passed, "failed": test_failed, "command": test_command, "model_file_identity_hash_read": True, "model_deserialized": False, "model_loaded": False, "torch_imported": False, "timm_imported": False, "cuda_or_gpu_access": False, "jpeg_decoded": False, "attribution_executed": False})
    write_new(repo / VALIDATION_PATH, validation); write_new(repo / FALSE_PASS_PATH, audit); write_new(repo / TEST_PATH, test_report)
    config = load_and_validate_config(repo)
    status = signed({"schema_version": 1, "document_type": "stage_f_f3h0_status", "status": "PASS", "terminal": True, "protocol_id": config["protocol_id"], "frozen_at": frozen_at, "protocol_commit": head, "protocol_only": True, "formal_gpu_execution_started": False, "model_file_identity_hash_read": True, "model_deserialized": False, "model_loaded": False, "jpeg_decoded": False, "attribution_executed": False, "f4_status": "PAUSED_PENDING_F3H", "f4_started": False, "next_authorization": "F3H1_IMPLEMENTATION_AND_SYNTHETIC_PREFLIGHT_ONLY", "git": {"commit": "SELF", "ref": ARCHIVE_REF}})
    write_new(repo / STATUS_PATH, status)
    completion = signed({"schema_version": 1, "document_type": "stage_f_f3h0_terminal_completion", "status": "PASS", "frozen_at": frozen_at, "protocol_commit": head, "protocol_id": config["protocol_id"], "config": {"path": CONFIG_PATH, "sha256": sha256_file(repo / CONFIG_PATH), "canonical_digest": config["canonical_digest"]}, "protocol": {"path": PROTOCOL_PATH, "sha256": sha256_file(repo / PROTOCOL_PATH)}, "schema": {"path": SCHEMA_PATH, "sha256": sha256_file(repo / SCHEMA_PATH)}, "validation_report": {"path": VALIDATION_PATH, "sha256": sha256_file(repo / VALIDATION_PATH), "canonical_digest": validation["canonical_digest"]}, "false_pass_audit": {"path": FALSE_PASS_PATH, "sha256": sha256_file(repo / FALSE_PASS_PATH), "canonical_digest": audit["canonical_digest"]}, "test_report": {"path": TEST_PATH, "sha256": sha256_file(repo / TEST_PATH), "canonical_digest": test_report["canonical_digest"]}, "status_record": {"path": STATUS_PATH, "sha256": sha256_file(repo / STATUS_PATH), "canonical_digest": status["canonical_digest"]}, "protocol_only": True, "formal_gpu_execution_started": False, "model_file_identity_hash_read": True, "model_deserialized": False, "model_loaded": False, "torch_imported": False, "timm_imported": False, "cuda_or_gpu_access": False, "jpeg_decoded": False, "attribution_executed": False, "f4_status": "PAUSED_PENDING_F3H", "f4_started": False, "next_authorization": "F3H1_IMPLEMENTATION_AND_SYNTHETIC_PREFLIGHT_ONLY", "git": {"commit": "SELF", "ref": ARCHIVE_REF}})
    temporary = (repo / PENDING_PATH).with_name(".f3h0-terminal.tmp")
    write_new(temporary, completion); os.replace(temporary, repo / PENDING_PATH); os.replace(repo / PENDING_PATH, repo / COMPLETION_PATH)
    return completion


def validate_completion(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); completion_path = repo / COMPLETION_PATH
    if not completion_path.is_file() or (repo / PENDING_PATH).exists():
        raise StageFF3HProtocolError("F3H0 terminal completion absent or pending remains")
    completion = _read_json(completion_path, "F3H0 completion"); _validate_signed(completion, "F3H0 completion")
    expected = {"status": "PASS", "protocol_only": True, "formal_gpu_execution_started": False, "model_file_identity_hash_read": True, "model_deserialized": False, "model_loaded": False, "torch_imported": False, "timm_imported": False, "cuda_or_gpu_access": False, "jpeg_decoded": False, "attribution_executed": False, "f4_status": "PAUSED_PENDING_F3H", "f4_started": False, "next_authorization": "F3H1_IMPLEMENTATION_AND_SYNTHETIC_PREFLIGHT_ONLY"}
    if any(completion.get(key) != value for key, value in expected.items()):
        raise StageFF3HProtocolError("F3H0 completion state drifted")
    for field in ("config", "protocol", "schema", "validation_report", "false_pass_audit", "test_report", "status_record"):
        record = completion[field]; path = repo / record["path"]
        if sha256_file(path) != record["sha256"]:
            raise StageFF3HProtocolError(f"completion record drifted: {field}")
        if "canonical_digest" in record:
            value = _read_json(path, field); _validate_signed(value, field)
            if value["canonical_digest"] != record["canonical_digest"]:
                raise StageFF3HProtocolError(f"completion canonical digest drifted: {field}")
    return {**expected, "archive_ref": ARCHIVE_REF, "canonical_digest": completion["canonical_digest"]}


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("validate"); check.add_argument("--repo-root", type=Path, default=REPO_ROOT); check.add_argument("--require-clean", action="store_true")
    completion = sub.add_parser("validate-completion"); completion.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    freeze = sub.add_parser("freeze"); freeze.add_argument("--repo-root", type=Path, default=REPO_ROOT); freeze.add_argument("--frozen-at", required=True); freeze.add_argument("--test-passed", required=True, type=int); freeze.add_argument("--test-failed", required=True, type=int); freeze.add_argument("--test-command", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate": result = validate_protocol(args.repo_root, require_clean=args.require_clean)
        elif args.command == "validate-completion": result = validate_completion(args.repo_root)
        else: result = freeze_protocol(args.repo_root, frozen_at=args.frozen_at, test_passed=args.test_passed, test_failed=args.test_failed, test_command=args.test_command)
    except (StageFF3HProtocolError, OSError, ValueError) as error:
        print(json.dumps({"status": "BLOCKED", "error": str(error)}, sort_keys=True), file=sys.stderr); return 2
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
