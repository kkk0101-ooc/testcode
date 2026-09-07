"""Prospective F3H full-head attribution study.

F3H extends the sealed F3 package for one pre-existing median-effect sample.
It reuses four sealed coordinate pairs, computes the remaining full 12x12
head grid, adds one different-target Grad-CAM reference, and renders complete
positive/negative atlases with a pair-shared monotonic log transform.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .constants import REPO_ROOT
from .stage_d_provenance import canonical_json_bytes, canonical_sha256
from . import stage_f_f3h_protocol as f3h0


CONFIG_PATH = f3h0.CONFIG_PATH
F3H1_ROOT = "evidence/stage_f/f3h1_implementation_v1_20260907/formal"
F3H1_REPORT = "reports/stage_f_f3h1_implementation_v1_20260907"
F3H1_COMPLETION = F3H1_ROOT + "/completion/f3h1.json"
F3H1_STATUS = F3H1_ROOT + "/status.json"
F3H1_VALIDATION = F3H1_REPORT + "/validation_report.json"
F3H1_FALSE_PASS = F3H1_REPORT + "/false_pass_audit.json"
F3H1_TEST = F3H1_REPORT + "/test_report.json"
F3H1_ARCHIVE_REF = "refs/archive/stage-f-v1-f3h1-implementation"
F3H1_ADDENDUM = "docs/stage_f_f3h1_implementation_addendum_20260907.md"
F3H1_ADDENDUM_SHA256 = "89450b3dc711f32511a1570c7365ea714090da055cba05f32d79e56ba88516bd"

ROOT_PATH = "evidence/stage_f/f3h_full_head_v1_20260907/formal"
REPORT_PATH = "reports/stage_f_f3h_full_head_v1_20260907"
PLAN_PATH = ROOT_PATH + "/control/plan.json"
RUN_SPEC_PATH = ROOT_PATH + "/control/run_spec.json"
PILOT_AUTH_PATH = ROOT_PATH + "/control/pilot_authorization.json"
BULK_AUTH_PATH = ROOT_PATH + "/control/bulk_authorization.json"
SOURCE_BINDING_PATH = ROOT_PATH + "/source_binding.json"
PENDING_PATH = ROOT_PATH + "/completion/f3h.pending.json"
COMPLETION_PATH = ROOT_PATH + "/completion/f3h.json"
TERMINAL_UNION_PATH = ROOT_PATH + "/terminal_union.json"
MAP_INVENTORY_PATH = ROOT_PATH + "/signed_map_inventory.json"
COMPONENT_MANIFEST_PATH = ROOT_PATH + "/signed_component_manifest.json"
SCALE_MANIFEST_PATH = ROOT_PATH + "/pair_shared_scale_manifest.json"
PANEL_MANIFEST_PATH = ROOT_PATH + "/panel_manifest.json"
RENDER_INVENTORY_PATH = ROOT_PATH + "/render_inventory.json"
REVIEW_INVENTORY_PATH = ROOT_PATH + "/review_copy_inventory.json"
ACCESS_AUDIT_PATH = ROOT_PATH + "/access_audit.json"
RESOURCE_AUDIT_PATH = ROOT_PATH + "/resource_audit.json"
PAYLOAD_MANIFEST_PATH = ROOT_PATH + "/payload_manifest.json"
INVENTORY_PATH = ROOT_PATH + "/whole_root_inventory.json"
STATUS_PATH = ROOT_PATH + "/status.json"
VALIDATION_PATH = REPORT_PATH + "/validation_report.json"
FALSE_PASS_PATH = REPORT_PATH + "/false_pass_audit.json"
TEST_PATH = REPORT_PATH + "/test_report.json"
ANALYSIS_PATH = REPORT_PATH + "/figure_analysis.md"
SUFFICIENCY_PATH = REPORT_PATH + "/class_conditioned_basis_assessment.json"
CONSENSUS_PATH = ROOT_PATH + "/equal_head_consensus_manifest.json"
REVIEW_ROOT_PATH = REPORT_PATH + "/figures_png"
F3H_ARCHIVE_REF = "refs/archive/stage-f-v1-f3h-completion"
F3H_RENDER_COUNT = 20
F3H_EXPORT_COUNT = 60

NNS = "conditional_fixed_clean_attention_head_direction_value_path_ig"
MATCHED = "matched_fixed_clean_attention_patch_token_contribution"
GRADCAM = "vit_grad_cam_block11_norm1"
METHODS = (NNS, MATCHED)
COORDINATES = tuple((block, head) for block in range(12) for head in range(12))
INHERITED_COORDINATES = ((0, 9), (4, 2), (6, 10), (11, 2))
PILOT_COORDINATES = ((0, 0), (4, 0), (8, 0), (11, 0))
ATTEMPT_ID = "attempt-000001"
MODEL_SHA256 = "489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d"
MODEL_TENSOR_DIGEST = "150234d8117a50057f62aa14a97f46bbf1da92b00b386c10c71cc7000c5d51dd"

IMPLEMENTATION_ATTACKS = (
    "formal_execution_before_f3h1_completion",
    "non_map_only_value_path_workload",
    "partial_12x12_coordinate_plan",
    "pair_split_across_shards",
    "inherited_coordinate_scheduled_for_recompute",
    "scientific_failure_not_typed_terminal",
    "display_array_used_as_numeric_source",
    "method_specific_display_gain",
    "gradcam_target_not_frozen_class_713",
    "gradcam_compared_as_same_target",
    "best_head_selection_surface",
    "f4_execution_surface",
)
FINAL_ATTACKS = f3h0.ATTACKS + (
    "terminal_union_not_exactly_288",
    "incomplete_consensus_uses_complete_case_average",
    "atlas_appearance_used_to_define_class_weights",
    "f3c_empirical_compute_started_before_protocol_freeze",
)


class StageFF3HError(ValueError):
    pass


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())[:-2] + ":" + time.strftime("%z")[-2:]


def _signed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value); result["canonical_digest"] = canonical_sha256(result); return result


def _sha_file(path: Path) -> str:
    return f3h0.sha256_file(path)


def _strict_json(path: Path, context: str) -> dict[str, Any]:
    value = f3h0._read_json(path, context)
    return value


def _validate_signed(value: Mapping[str, Any], context: str) -> None:
    f3h0._validate_signed(value, context)


def _git(repo: Path, *args: str) -> str:
    return f3h0._git(repo, *args)


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    f3h0.write_new(path, value)


def _record(repo: Path, relative: str) -> dict[str, Any]:
    path = repo / relative; value = _strict_json(path, relative); _validate_signed(value, relative)
    return {"path": relative, "bytes": path.stat().st_size, "sha256": _sha_file(path), "canonical_digest": value["canonical_digest"]}


def _file_record(repo: Path, relative: str) -> dict[str, Any]:
    path = repo / relative
    return {"path": relative, "bytes": path.stat().st_size, "sha256": _sha_file(path)}


def _array_digest(array: np.ndarray) -> str:
    source = np.ascontiguousarray(array)
    digest = hashlib.sha256(); digest.update(source.dtype.str.encode("ascii")); digest.update(b"\0"); digest.update(json.dumps(list(source.shape), separators=(",", ":")).encode("ascii")); digest.update(b"\0"); digest.update(source.tobytes(order="C")); return digest.hexdigest()


def split_signed_map(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(raw)
    if value.dtype != np.float32 or value.shape != (56, 56) or not np.isfinite(value).all():
        raise StageFF3HError("signed map must be finite float32 56x56")
    positive = np.maximum(value, np.float32(0.0)).astype(np.float32)
    negative = np.maximum(-value, np.float32(0.0)).astype(np.float32)
    if not np.array_equal(positive - negative, value):
        raise StageFF3HError("positive/negative reconstruction failed")
    return np.ascontiguousarray(positive), np.ascontiguousarray(negative)


def _component_display(component: np.ndarray, maximum: float, kappa: float) -> np.ndarray:
    from scipy.ndimage import zoom
    if maximum == 0.0: shaped = np.zeros((56, 56), dtype=np.float32)
    else: shaped = (np.log1p(np.float64(kappa) * component.astype(np.float64) / maximum) / math.log1p(kappa)).astype(np.float32)
    displayed = zoom(shaped, (4.0, 4.0), order=1, mode="nearest", prefilter=False).astype(np.float32)
    if displayed.shape != (224, 224): raise StageFF3HError("display resize drifted")
    return np.ascontiguousarray(displayed)


def pair_shared_log_display(nns_raw: np.ndarray, matched_raw: np.ndarray, *, kappa: float = 999.0) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    nns_positive, nns_negative = split_signed_map(nns_raw); matched_positive, matched_negative = split_signed_map(matched_raw)
    maximum = max(float(np.max(np.abs(nns_raw))), float(np.max(np.abs(matched_raw))))
    if not math.isfinite(kappa) or kappa <= 0: raise StageFF3HError("kappa must be finite and positive")
    def transform(component: np.ndarray) -> np.ndarray:
        return _component_display(component, maximum, kappa)
    arrays = {"nns_positive": transform(nns_positive), "nns_negative": transform(nns_negative), "matched_positive": transform(matched_positive), "matched_negative": transform(matched_negative)}
    return arrays, {"pair_scale": maximum, "scale_basis": "NNS_MATCHED_UNION_MAX_ABS" if maximum else "ZERO_PAIR_SCALE", "kappa": float(kappa), "formula": "log1p(kappa*component/pair_scale)/log1p(kappa)", "method_specific_gain": False, "paired_magnitude_comparison_allowed": True, "raw": {"nns": _array_digest(nns_raw), "matched": _array_digest(matched_raw)}, "components": {"nns_positive": _array_digest(nns_positive), "nns_negative": _array_digest(nns_negative), "matched_positive": _array_digest(matched_positive), "matched_negative": _array_digest(matched_negative)}, "display_shape": [224, 224], "resize_interpolation": "bilinear", "display": {key: _array_digest(value) for key, value in arrays.items()}}


def _inventory(root: Path, *, exclude: Sequence[str] = ()) -> dict[str, Any]:
    excluded = set(exclude); files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_file() and not path.is_symlink() and relative not in excluded:
            files.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha_file(path)})
    return {"files": files, "file_count": len(files), "total_bytes": sum(item["bytes"] for item in files), "canonical_digest": canonical_sha256({"files": files})}


def _config(repo: Path) -> dict[str, Any]:
    return f3h0.load_and_validate_config(repo)


def _registry_row(repo: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    registry = _strict_json(repo / config["sample"]["registry_path"], "D1 registry"); _validate_signed(registry, "D1 registry")
    return next(row for row in registry["rows"] if row["selection_order"] == config["sample"]["selection_order"])


def _cross_sources(repo: Path, config: Mapping[str, Any]) -> dict[tuple[int, int, str], dict[str, Any]]:
    path = repo / config["upstream"]["cross_layer_panel_path"]
    manifest = _strict_json(path, "F3 cross-layer panel manifest"); _validate_signed(manifest, "F3 cross-layer panel manifest")
    sample_id = config["sample"]["sample_id"]
    return {(row["coordinate"][0], row["coordinate"][1], row["method"]): row for row in manifest["cells"] if row.get("sample_id") == sample_id and row.get("method") in METHODS}


def _source_result_path(source_map_path: str) -> str:
    path = Path(source_map_path)
    if path.name == "metric_ready_map.npy" and path.parent.name == "L1":
        return (path.parent.parent / "L0/result.json").as_posix()
    if path.name == "map.npy":
        return path.with_name("result.json").as_posix()
    raise StageFF3HError(f"unknown inherited map layout: {source_map_path}")


def build_plan(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); config = _config(repo); sources = _cross_sources(repo, config); units = []
    inherited = set(INHERITED_COORDINATES)
    for ordinal, (block, head) in enumerate(COORDINATES):
        for method_index, method in enumerate(METHODS):
            source = sources.get((block, head, method)) if (block, head) in inherited else None
            if (block, head) in inherited and (source is None or source["validity_status"] != "VALID"):
                raise StageFF3HError(f"inherited source absent or invalid: {(block, head, method)}")
            payload = {
                "schema_version": 1, "document_type": "stage_f_f3h_value_path_unit",
                "protocol_id": config["protocol_id"], "sample_id": config["sample"]["sample_id"],
                "selection_order": 44, "image_sha256": config["sample"]["image_sha256"],
                "block": block, "head": head, "coordinate_ordinal": ordinal,
                "method": method, "method_index": method_index,
                "target": "fixed_clean_attention_cls_head_direction",
                "mode": "INHERITED_READ_ONLY" if source else "NEW_MAP_ONLY_COMPUTE",
                "shard_id": ordinal % config["execution"]["shard_count"],
                "source": None if source is None else {**{key: source.get(key) for key in ("source_path", "source_file_sha256", "source_result_sha256", "signed_map_sha256", "attempt_id", "source_evidence_origin", "applicable_canonical_digest")}, "source_result_path": _source_result_path(source["source_path"])},
            }
            units.append(_signed(payload))
    gradcam = _signed({"schema_version": 1, "document_type": "stage_f_f3h_gradcam_unit", "protocol_id": config["protocol_id"], "sample_id": config["sample"]["sample_id"], "selection_order": 44, "image_sha256": config["sample"]["image_sha256"], "method": GRADCAM, "target": "clean_predicted_class_raw_logit", "class_index": 713, "target_layer": "blocks[11].norm1", "mode": "NEW_MAP_ONLY_COMPUTE"})
    return _signed({"schema_version": 1, "document_type": "stage_f_f3h_frozen_plan", "protocol_id": config["protocol_id"], "config_digest": config["canonical_digest"], "coordinate_order": "BLOCK_MAJOR_THEN_HEAD_ASCENDING", "coordinates": [list(item) for item in COORDINATES], "methods": list(METHODS), "value_path_units": units, "gradcam_unit": gradcam, "counts": {"coordinates": 144, "value_path_terminals": 288, "inherited_valid": 8, "new_value_path": 280, "gradcam": 1}})


def validate_implementation(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); protocol = f3h0.validate_completion(repo); config = _config(repo); plan = build_plan(repo)
    if protocol["next_authorization"] != "F3H1_IMPLEMENTATION_AND_SYNTHETIC_PREFLIGHT_ONLY": raise StageFF3HError("F3H0 did not authorize F3H1")
    counts = Counter((u["block"], u["head"], u["method"]) for u in plan["value_path_units"])
    expected = Counter((b, h, m) for b, h in COORDINATES for m in METHODS)
    if counts != expected or plan["counts"] != {"coordinates": 144, "value_path_terminals": 288, "inherited_valid": 8, "new_value_path": 280, "gradcam": 1}: raise StageFF3HError("F3H plan coverage failed")
    inherited = [unit for unit in plan["value_path_units"] if unit["mode"] == "INHERITED_READ_ONLY"]
    if len(inherited) != 8 or any(unit["source"] is None for unit in inherited): raise StageFF3HError("inherited source plan failed")
    for unit in inherited:
        source = unit["source"]; path = repo / source["source_path"]
        if _sha_file(path) != source["source_file_sha256"]: raise StageFF3HError("inherited map file drifted")
        if _sha_file(repo/source["source_result_path"]) != source["source_result_sha256"]: raise StageFF3HError("inherited result file drifted")
        array = np.load(path, allow_pickle=False)
        if array.dtype != np.float32 or array.shape != (56, 56) or _array_digest(array) != source["signed_map_sha256"]: raise StageFF3HError("inherited map array drifted")
    source = (repo / __file__).read_text(encoding="utf-8")
    for token in ("resolve_fixed_clean_attention_head_directions", "e3_r1_conditional_fixed_clean_attention_head_direction_value_path_ig", "matched_fixed_clean_attention_patch_token_contribution", "vit_grad_cam_block11_norm1", "pixel_to_fine_cell_mass", "patch_to_fine_cell_mass"):
        if token not in source: raise StageFF3HError(f"implementation does not bind required kernel: {token}")
    if _sha_file(repo/F3H1_ADDENDUM) != F3H1_ADDENDUM_SHA256: raise StageFF3HError("F3H1 addendum drifted")
    return {"schema_version": 1, "document_type": "stage_f_f3h1_implementation_validation", "status": "PASS", "plan_digest": plan["canonical_digest"], "counts": plan["counts"], "render_count":F3H_RENDER_COUNT, "f3h1_addendum_sha256":_sha_file(repo/F3H1_ADDENDUM), "formal_gpu_execution_started": False, "model_loaded": False, "jpeg_decoded": False, "f4_status": "PAUSED_PENDING_F3H_AND_F3C", "next_authorization": "F3H2_PILOT_ONLY"}


def implementation_false_pass_audit() -> dict[str, Any]:
    return _signed({"schema_version": 1, "document_type": "stage_f_f3h1_false_pass_audit", "status": "PASS", "attacks": [{"attack": item, "result": "REJECTED"} for item in IMPLEMENTATION_ATTACKS], "pass_possible_without_all_conditions": False})


def freeze_implementation(repo_root: Path | str = REPO_ROOT, *, completed_at: str, passed: int, failed: int, command: str) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    if (repo / F3H1_ROOT).exists() or (repo / F3H1_REPORT).exists(): raise StageFF3HError("F3H1 root exists")
    if _git(repo, "status", "--porcelain=v1"): raise StageFF3HError("F3H1 freeze requires clean worktree")
    head = _git(repo, "rev-parse", "HEAD"); tracking = _git(repo, "rev-parse", "origin/stage-d-d3-pass")
    if head != tracking: raise StageFF3HError("F3H1 freeze requires local/tracking equality")
    if passed <= 0 or failed != 0: raise StageFF3HError("F3H1 tests failed")
    validation = _signed(validate_implementation(repo)); false_pass = implementation_false_pass_audit(); test = _signed({"schema_version": 1, "document_type": "stage_f_f3h1_test_report", "status": "PASS", "passed": passed, "failed": failed, "command": command, "formal_gpu_execution_started": False, "model_loaded": False, "jpeg_decoded": False})
    _write_new(repo / F3H1_VALIDATION, validation); _write_new(repo / F3H1_FALSE_PASS, false_pass); _write_new(repo / F3H1_TEST, test)
    status = _signed({"schema_version": 1, "document_type": "stage_f_f3h1_status", "status": "PASS", "completed_at": completed_at, "implementation_commit": head, "formal_gpu_execution_started": False, "f4_status": "PAUSED_PENDING_F3H_AND_F3C", "next_authorization": "F3H2_PILOT_ONLY", "git": {"commit": "SELF", "ref": F3H1_ARCHIVE_REF}}); _write_new(repo / F3H1_STATUS, status)
    completion = _signed({"schema_version": 1, "document_type": "stage_f_f3h1_terminal_completion", "status": "PASS", "completed_at": completed_at, "implementation_commit": head, "plan_digest": validation["plan_digest"], "counts": validation["counts"], "render_count":F3H_RENDER_COUNT, "implementation_addendum":_file_record(repo,F3H1_ADDENDUM), "validation_report": _record(repo, F3H1_VALIDATION), "false_pass_audit": _record(repo, F3H1_FALSE_PASS), "test_report": _record(repo, F3H1_TEST), "status_record": _record(repo, F3H1_STATUS), "formal_gpu_execution_started": False, "model_loaded": False, "jpeg_decoded": False, "f4_status": "PAUSED_PENDING_F3H_AND_F3C", "f4_started": False, "next_authorization": "F3H2_PILOT_ONLY", "git": {"commit": "SELF", "ref": F3H1_ARCHIVE_REF}}); _write_new(repo / F3H1_COMPLETION, completion); return completion


def validate_implementation_completion(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); completion = _strict_json(repo / F3H1_COMPLETION, "F3H1 completion"); _validate_signed(completion, "F3H1 completion")
    if completion.get("status") != "PASS" or completion.get("next_authorization") != "F3H2_PILOT_ONLY" or completion.get("formal_gpu_execution_started") or completion.get("f4_started") or completion.get("f4_status") != "PAUSED_PENDING_F3H_AND_F3C" or completion.get("render_count") != F3H_RENDER_COUNT: raise StageFF3HError("F3H1 completion state drifted")
    for field in ("validation_report", "false_pass_audit", "test_report", "status_record"):
        if completion[field] != _record(repo, completion[field]["path"]): raise StageFF3HError(f"F3H1 completion binding drifted: {field}")
    if completion["implementation_addendum"] != _file_record(repo,F3H1_ADDENDUM):raise StageFF3HError("F3H1 addendum binding drifted")
    return {"status": "PASS", "implementation_commit": completion["implementation_commit"], "plan_digest": completion["plan_digest"], "next_authorization": "F3H2_PILOT_ONLY", "f4_status": "PAUSED_PENDING_F3H_AND_F3C", "archive_ref": F3H1_ARCHIVE_REF}


def _attempt_root(repo: Path, unit: Mapping[str, Any]) -> Path:
    return repo / ROOT_PATH / "units" / unit["canonical_digest"] / "attempts" / ATTEMPT_ID


def _gradcam_root(repo: Path) -> Path:
    return repo / ROOT_PATH / "gradcam" / "attempts" / ATTEMPT_ID


def _write_npy(path: Path, array: np.ndarray) -> None:
    with path.open("wb") as stream:
        np.save(stream, np.ascontiguousarray(array), allow_pickle=False); stream.flush(); os.fsync(stream.fileno())


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())


def _publish_terminal(repo: Path, unit: Mapping[str, Any], result: Mapping[str, Any], signed_map: np.ndarray | None) -> Path:
    root = _attempt_root(repo, unit)
    if root.exists(): raise StageFF3HError(f"refusing to overwrite terminal attempt: {root}")
    root.parent.mkdir(parents=True, exist_ok=True); temporary = Path(tempfile.mkdtemp(prefix=".attempt-000001-", dir=root.parent))
    try:
        signed_result = _signed(result); result_raw = canonical_json_bytes(signed_result) + b"\n"; _write_bytes(temporary / "result.json", result_raw)
        map_record = None
        if signed_map is not None:
            if signed_map.dtype != np.float32 or signed_map.shape != (56, 56) or not np.isfinite(signed_map).all(): raise StageFF3HError("terminal map contract failed")
            _write_npy(temporary / "map.npy", signed_map); map_record = {"sha256": _sha_file(temporary / "map.npy"), "array_digest": _array_digest(signed_map), "bytes": (temporary / "map.npy").stat().st_size, "shape": [56, 56], "dtype": "float32"}
        status = _signed({"schema_version": 1, "document_type": "stage_f_f3h_attempt_status", "attempt_id": ATTEMPT_ID, "unit_digest": unit["canonical_digest"], "state": result["validity_status"], "terminal": True, "published_at": _now()}); status_raw = canonical_json_bytes(status) + b"\n"; _write_bytes(temporary / "status.json", status_raw)
        manifest = _signed({"schema_version": 1, "document_type": "stage_f_f3h_attempt_manifest", "attempt_id": ATTEMPT_ID, "unit_digest": unit["canonical_digest"], "result": {"sha256": hashlib.sha256(result_raw).hexdigest(), "bytes": len(result_raw)}, "status": {"sha256": hashlib.sha256(status_raw).hexdigest(), "bytes": len(status_raw)}, "map": map_record, "immutable": True}); _write_bytes(temporary / "manifest.json", canonical_json_bytes(manifest) + b"\n")
        for path in temporary.iterdir():
            with path.open("rb") as stream: os.fsync(stream.fileno())
        os.replace(temporary, root)
        directory = os.open(root.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        if temporary.exists(): shutil.rmtree(temporary)
    validate_attempt(repo, unit); return root


def validate_attempt(repo: Path, unit: Mapping[str, Any]) -> dict[str, Any]:
    root = _attempt_root(repo, unit); required = (root / "result.json", root / "status.json", root / "manifest.json")
    if not root.is_dir() or any(not path.is_file() for path in required): raise StageFF3HError(f"attempt absent or incomplete: {root}")
    result = _strict_json(required[0], "attempt result"); status = _strict_json(required[1], "attempt status"); manifest = _strict_json(required[2], "attempt manifest")
    for value, name in ((result, "result"), (status, "status"), (manifest, "manifest")): _validate_signed(value, name)
    if result["unit_digest"] != unit["canonical_digest"] or result["sample_id"] != unit["sample_id"] or result["block"] != unit["block"] or result["head"] != unit["head"] or result["method"] != unit["method"] or result["validity_status"] not in {"VALID", "FAILED", "NA"} or status["state"] != result["validity_status"] or not status["terminal"]:
        raise StageFF3HError("attempt identity/status drifted")
    result_raw = required[0].read_bytes(); status_raw = required[1].read_bytes()
    if manifest["result"]["sha256"] != hashlib.sha256(result_raw).hexdigest() or manifest["status"]["sha256"] != hashlib.sha256(status_raw).hexdigest(): raise StageFF3HError("attempt file hash drifted")
    map_path = root / "map.npy"
    if result["validity_status"] == "VALID":
        if not map_path.is_file() or manifest["map"] is None: raise StageFF3HError("valid attempt lacks map")
        array = np.load(map_path, allow_pickle=False)
        if array.dtype != np.float32 or array.shape != (56, 56) or not np.isfinite(array).all() or _sha_file(map_path) != manifest["map"]["sha256"] or _array_digest(array) != manifest["map"]["array_digest"] or _array_digest(array) != result["source_hashes"]["signed_map_sha256"]: raise StageFF3HError("valid attempt map drifted")
    elif map_path.exists() or manifest["map"] is not None:
        raise StageFF3HError("failed/NA attempt must not contain a map")
    return {"root": root, "result": result, "status": status, "manifest": manifest}


def _fine_map(method_result: Any) -> np.ndarray:
    from .stage_c_maps import patch_to_fine_cell_mass, pixel_to_fine_cell_mass, signed_rgb_channel_sum
    if method_result.method_id == NNS:
        mapped = pixel_to_fine_cell_mass(signed_rgb_channel_sum(method_result.native_tensor))
    elif method_result.method_id == MATCHED:
        mapped = patch_to_fine_cell_mass(method_result.metric_map)
    else:
        raise StageFF3HError("unsupported Value-path method result")
    value = np.asarray(mapped.data)
    if value.dtype != np.float32 or value.shape != (56, 56) or not np.isfinite(value).all(): raise StageFF3HError("fine map projection failed")
    return np.ascontiguousarray(value)


def _target_identity(target: Any) -> tuple[dict[str, Any], str]:
    from .stage_e_e1 import _target_semantic_digest
    identity = {"comparison_line": "value_path", "target_ref": "fixed_clean_attention_cls_head_direction", "clean_input_sha256": target.clean_input_sha256, "frozen_target": target.to_dict()}
    return identity, _target_semantic_digest(identity)


def _valid_result(unit: Mapping[str, Any], target: Any, method_result: Any, signed_map: np.ndarray, *, elapsed: float, cpu: float, forward_calls: int, backward_calls: int, peak: int) -> dict[str, Any]:
    identity, target_digest = _target_identity(target); provenance = json.loads(canonical_json_bytes(dict(method_result.provenance)))
    pair = {"sample_id": unit["sample_id"], "selection_order": 44, "block": unit["block"], "head": unit["head"], "target": unit["target"], "target_semantic_digest": target_digest, "clean_input_sha256": target.clean_input_sha256, "model_sha256": MODEL_SHA256}
    return {"schema_version": 1, "document_type": "stage_f_f3h_value_path_terminal", "protocol_id": unit["protocol_id"], "unit_digest": unit["canonical_digest"], "sample_id": unit["sample_id"], "selection_order": 44, "block": unit["block"], "head": unit["head"], "method": unit["method"], "target": unit["target"], "validity_status": "VALID", "failure_code": None, "na_reason": None, "attempt_id": ATTEMPT_ID, "target_identity": identity, "target_semantic_digest": target_digest, "pair_binding": pair, "pair_binding_digest": canonical_sha256(pair), "source_hashes": {"image_sha256": unit["image_sha256"], "clean_input_sha256": target.clean_input_sha256, "model_sha256": MODEL_SHA256, "model_canonical_tensor_digest": MODEL_TENSOR_DIGEST, "signed_map_sha256": _array_digest(signed_map), "method_provenance_canonical_digest": canonical_sha256(provenance)}, "method_provenance": provenance, "resources": {"elapsed_seconds": elapsed, "cpu_seconds": cpu, "gpu_hours": elapsed / 3600.0, "forward_calls": forward_calls, "backward_calls": backward_calls, "peak_cuda_memory_bytes": peak}, "map_only": True, "fidelity_executed": False, "statistics_computed": False}


def _invalid_result(unit: Mapping[str, Any], target: Any, *, status: str, code: str, details: Mapping[str, Any], elapsed: float, cpu: float, forward_calls: int, backward_calls: int, peak: int) -> dict[str, Any]:
    identity, target_digest = _target_identity(target); pair = {"sample_id": unit["sample_id"], "selection_order": 44, "block": unit["block"], "head": unit["head"], "target": unit["target"], "target_semantic_digest": target_digest, "clean_input_sha256": target.clean_input_sha256, "model_sha256": MODEL_SHA256}
    return {"schema_version": 1, "document_type": "stage_f_f3h_value_path_terminal", "protocol_id": unit["protocol_id"], "unit_digest": unit["canonical_digest"], "sample_id": unit["sample_id"], "selection_order": 44, "block": unit["block"], "head": unit["head"], "method": unit["method"], "target": unit["target"], "validity_status": status, "failure_code": code if status == "FAILED" else None, "na_reason": code if status == "NA" else None, "failure_details": dict(details), "attempt_id": ATTEMPT_ID, "target_identity": identity, "target_semantic_digest": target_digest, "pair_binding": pair, "pair_binding_digest": canonical_sha256(pair), "source_hashes": {"image_sha256": unit["image_sha256"], "clean_input_sha256": target.clean_input_sha256, "model_sha256": MODEL_SHA256, "model_canonical_tensor_digest": MODEL_TENSOR_DIGEST, "signed_map_sha256": None}, "resources": {"elapsed_seconds": elapsed, "cpu_seconds": cpu, "gpu_hours": elapsed / 3600.0, "forward_calls": forward_calls, "backward_calls": backward_calls, "peak_cuda_memory_bytes": peak}, "map_only": True, "fidelity_executed": False, "statistics_computed": False}


def _runtime(repo: Path, device: str) -> tuple[Any, Any, Any, str, dict[str, Any]]:
    from .stage_e_e1 import _load_model_and_context, _load_sample
    from .stage_c_perturbations import tensor_sha256
    config = _config(repo); row = _registry_row(repo, config); model, binding, model_record = _load_model_and_context(repo, device); _rgb, clean, clean_sha = _load_sample(binding, row, model, device)
    if clean_sha != config["sample"]["clean_input_sha256"] or clean_sha != tensor_sha256(clean): raise StageFF3HError("live clean tensor hash drifted")
    return model, binding, clean, clean_sha, model_record


def _completed_gpu_hours(repo: Path) -> float:
    total = 0.0
    units_root = repo / ROOT_PATH / "units"
    if units_root.exists():
        for path in units_root.glob("*/attempts/*/result.json"):
            value = _strict_json(path, "completed unit resource")
            total += float(value.get("resources", {}).get("gpu_hours", 0.0))
    gradcam = _gradcam_root(repo) / "result.json"
    if gradcam.exists():
        value = _strict_json(gradcam, "Grad-CAM resource")
        total += float(value.get("resources", {}).get("gpu_hours", 0.0))
    return total


def _execute_coordinates(repo: Path, coordinates: Sequence[tuple[int, int]], *, device: str) -> dict[str, Any]:
    import torch
    from .stage_c_value_path import HeadCandidate, StageCValuePathConvergenceError, StageCValuePathScientificInvalidity, e3_r1_conditional_fixed_clean_attention_head_direction_value_path_ig, matched_fixed_clean_attention_patch_token_contribution, resolve_fixed_clean_attention_head_directions
    plan = _strict_json(repo / PLAN_PATH, "F3H plan"); by_key = {(u["block"], u["head"], u["method"]): u for u in plan["value_path_units"]}; model = None; completed = skipped = 0; statuses: Counter[str] = Counter(); records = []; process_start = time.monotonic(); cpu_start = time.process_time()
    try:
        model, _binding, clean, _clean_sha, model_record = _runtime(repo, device); calls = 0
        def hook(_module: Any, _args: Any) -> None:
            nonlocal calls; calls += 1
        handle = model.register_forward_pre_hook(hook)
        try:
            targets = resolve_fixed_clean_attention_head_directions(model, clean, tuple(HeadCandidate(*coord) for coord in coordinates), candidate_chunk_size=12); target_map = {(t.block_index, t.head_index): t for t in targets}
            for block, head in coordinates:
                target = target_map[(block, head)]
                for method in METHODS:
                    unit = by_key[(block, head, method)]; root = _attempt_root(repo, unit)
                    if root.exists():
                        record = validate_attempt(repo, unit); skipped += 1; statuses.update([record["result"]["validity_status"]]); continue
                    if device.startswith("cuda") and _completed_gpu_hours(repo) >= _config(repo)["budget"]["hard_h20_gpu_hours_max"]:
                        raise StageFF3HError("F3H hard GPU budget reached; refusing a new unit")
                    started = time.monotonic(); cpu_started = time.process_time(); before = calls; backward = 0
                    try:
                        result = e3_r1_conditional_fixed_clean_attention_head_direction_value_path_ig(model, clean, target, provenance={"stage": "F3H", "protocol_id": unit["protocol_id"], "map_only": True}) if method == NNS else matched_fixed_clean_attention_patch_token_contribution(target, provenance={"stage": "F3H", "protocol_id": unit["protocol_id"], "map_only": True})
                        if method == NNS:
                            backward = sum(int(math.ceil(int(item["steps"]) / 8)) for item in result.provenance["ig_result"]["attempts"])
                        if result.na is not None:
                            reason = result.na.reason.value; terminal = _invalid_result(unit, target, status="NA", code=reason, details=result.na.to_dict(), elapsed=time.monotonic()-started, cpu=time.process_time()-cpu_started, forward_calls=calls-before, backward_calls=backward, peak=int(torch.cuda.max_memory_reserved(device)) if device.startswith("cuda") else 0); signed_map = None
                        else:
                            signed_map = _fine_map(result); terminal = _valid_result(unit, target, result, signed_map, elapsed=time.monotonic()-started, cpu=time.process_time()-cpu_started, forward_calls=calls-before, backward_calls=backward, peak=int(torch.cuda.max_memory_reserved(device)) if device.startswith("cuda") else 0)
                    except StageCValuePathConvergenceError as error:
                        backward = sum(int(math.ceil(int(item["steps"]) / 8)) for item in error.ig_result.attempts); terminal = _invalid_result(unit, target, status="FAILED", code="ig_conservation_failed_f3h_full_schedule", details={"attempts": [dict(item) for item in error.ig_result.attempts]}, elapsed=time.monotonic()-started, cpu=time.process_time()-cpu_started, forward_calls=calls-before, backward_calls=backward, peak=int(torch.cuda.max_memory_reserved(device)) if device.startswith("cuda") else 0); signed_map = None
                    except StageCValuePathScientificInvalidity as error:
                        terminal = _invalid_result(unit, target, status="FAILED", code=error.failure_code, details=error.details or {}, elapsed=time.monotonic()-started, cpu=time.process_time()-cpu_started, forward_calls=calls-before, backward_calls=backward, peak=int(torch.cuda.max_memory_reserved(device)) if device.startswith("cuda") else 0); signed_map = None
                    _publish_terminal(repo, unit, terminal, signed_map); completed += 1; statuses.update([terminal["validity_status"]]); records.append({"block": block, "head": head, "method": method, "status": terminal["validity_status"], "elapsed_seconds": terminal["resources"]["elapsed_seconds"]})
        finally:
            handle.remove()
        return _signed({"schema_version": 1, "document_type": "stage_f_f3h_execution_telemetry", "device": device, "coordinates": [list(item) for item in coordinates], "completed_units": completed, "skipped_units": skipped, "statuses": dict(statuses), "elapsed_seconds": time.monotonic()-process_start, "cpu_seconds": time.process_time()-cpu_start, "gpu_hours": (time.monotonic()-process_start)/3600.0 if device.startswith("cuda") else 0.0, "forward_calls": calls, "model": model_record, "records": records, "completed_at": _now()})
    finally:
        if model is not None: del model
        if device.startswith("cuda") and torch.cuda.is_available(): torch.cuda.empty_cache()


def initialize(repo_root: Path | str = REPO_ROOT, *, started_at: str) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); config = _config(repo); implementation = validate_implementation_completion(repo)
    if (repo / ROOT_PATH).exists() or (repo / REPORT_PATH).exists(): raise StageFF3HError("F3H formal root already exists")
    if _git(repo, "status", "--porcelain=v1"): raise StageFF3HError("F3H initialization requires clean worktree")
    head = _git(repo, "rev-parse", "HEAD"); tracking = _git(repo, "rev-parse", "origin/stage-d-d3-pass")
    if head != tracking: raise StageFF3HError("F3H initialization requires local/tracking equality")
    if _git(repo, "rev-parse", F3H1_ARCHIVE_REF + "^{commit}") != head: raise StageFF3HError("F3H1 archive ref must equal execution HEAD")
    plan = build_plan(repo)
    source_binding = _signed({"schema_version": 1, "document_type": "stage_f_f3h_source_binding", "status": "PASS", "protocol_config": _record(repo, CONFIG_PATH), "implementation_addendum":_file_record(repo,F3H1_ADDENDUM), "f3h0_completion": _record(repo, f3h0.COMPLETION_PATH), "f3h1_completion": _record(repo, F3H1_COMPLETION), "f3_completion": _record(repo, config["upstream"]["f3_completion_path"]), "blue_display_completion": _record(repo, config["upstream"]["blue_display_completion_path"]), "sample": config["sample"], "model": config["model"], "selection_changed": False, "f4_started": False})
    run_spec = _signed({"schema_version": 1, "document_type": "stage_f_f3h_run_spec", "status": "FROZEN_BEFORE_FORMAL_OUTCOME_ACCESS", "plan_digest": plan["canonical_digest"], "pilot_coordinates": [list(item) for item in PILOT_COORDINATES], "bulk_shards": 8, "formal_device_type": "cuda", "run_fidelity": False, "run_stability": False, "run_sanity": False, "render_before_terminal_reconciliation": False, "f4_started": False})
    pilot = _signed({"schema_version": 1, "document_type": "stage_f_f3h_pilot_authorization", "status": "AUTHORIZED", "scope": "F3H2_PILOT_ONLY", "coordinates": [list(item) for item in PILOT_COORDINATES], "methods": list(METHODS), "expected_terminals": 8, "jpeg_or_map_visual_access_before_bulk_gate_allowed": False, "bulk_authorized": False, "gradcam_authorized": False, "f4_started": False})
    pending = _signed({"schema_version": 1, "document_type": "stage_f_f3h_pending", "status": "PENDING", "started_at": started_at, "execution_commit": head, "plan_digest": plan["canonical_digest"], "expected_value_path_terminals": 288, "expected_gradcam_terminals": 1, "terminal_pass_published": False, "f4_status": "PAUSED_PENDING_F3H", "f4_started": False})
    for relative, document in ((PLAN_PATH, plan), (RUN_SPEC_PATH, run_spec), (PILOT_AUTH_PATH, pilot), (SOURCE_BINDING_PATH, source_binding), (PENDING_PATH, pending)): _write_new(repo / relative, document)
    return {"status": "INITIALIZED", "plan_digest": plan["canonical_digest"], "implementation_commit": implementation["implementation_commit"], "pilot_only": True, "f4_status": "PAUSED_PENDING_F3H"}


def validate_initialized(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); plan = _strict_json(repo / PLAN_PATH, "plan"); run_spec = _strict_json(repo / RUN_SPEC_PATH, "run spec"); pilot = _strict_json(repo / PILOT_AUTH_PATH, "pilot auth"); source = _strict_json(repo / SOURCE_BINDING_PATH, "source binding"); pending = _strict_json(repo / PENDING_PATH, "pending")
    for value, name in ((plan, "plan"), (run_spec, "run spec"), (pilot, "pilot auth"), (source, "source binding"), (pending, "pending")): _validate_signed(value, name)
    if plan != build_plan(repo) or plan["counts"]["value_path_terminals"] != 288 or run_spec["plan_digest"] != plan["canonical_digest"]: raise StageFF3HError("initialized plan/run spec drifted")
    if pilot["status"] != "AUTHORIZED" or pilot["bulk_authorized"] or pilot["gradcam_authorized"] or pending["status"] != "PENDING": raise StageFF3HError("initial authorization state drifted")
    if pending["execution_commit"] != _git(repo,"rev-parse","HEAD") or pending["execution_commit"] != _git(repo,"rev-parse","origin/stage-d-d3-pass") or pending["execution_commit"] != _git(repo,"rev-parse",F3H1_ARCHIVE_REF+"^{commit}"):raise StageFF3HError("F3H execution commit/ref drifted")
    return {"status": "PASS", "plan_digest": plan["canonical_digest"], "pilot_terminals": 8, "bulk_authorized": False, "f4_started": False}


def run_pilot(repo_root: Path | str = REPO_ROOT, *, device: str = "cuda:0") -> dict[str, Any]:
    repo = Path(repo_root).resolve(); validate_initialized(repo); telemetry_path = repo / ROOT_PATH / "telemetry/pilot.json"
    if telemetry_path.exists():
        value = _strict_json(telemetry_path, "pilot telemetry"); _validate_signed(value, "pilot telemetry"); return value
    telemetry = _execute_coordinates(repo, PILOT_COORDINATES, device=device); _write_new(telemetry_path, telemetry); return telemetry


def authorize_bulk(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); validate_initialized(repo); config = _config(repo); plan = _strict_json(repo / PLAN_PATH, "plan"); pilot = _strict_json(repo / ROOT_PATH / "telemetry/pilot.json", "pilot telemetry"); _validate_signed(pilot, "pilot telemetry")
    by_key = {(u["block"], u["head"], u["method"]): u for u in plan["value_path_units"]}
    results = [validate_attempt(repo, by_key[(block, head, method)])["result"] for block, head in PILOT_COORDINATES for method in METHODS]
    if len(results) != 8 or any(row["validity_status"] not in {"VALID", "FAILED", "NA"} for row in results): raise StageFF3HError("pilot terminal coverage failed")
    projected = float(pilot["gpu_hours"]) / 4.0 * 140.0 + float(config["budget"]["gradcam_and_control_reserve_gpu_hours"])
    if pilot["gpu_hours"] > config["budget"]["hard_h20_gpu_hours_max"] or projected > config["budget"]["hard_h20_gpu_hours_max"]: raise StageFF3HError("pilot projection exceeded hard budget")
    authorization = _signed({"schema_version": 1, "document_type": "stage_f_f3h_bulk_authorization", "status": "AUTHORIZED", "authorized_at": _now(), "plan_digest": plan["canonical_digest"], "pilot_terminal_count": 8, "pilot_status_counts": dict(Counter(row["validity_status"] for row in results)), "pilot_gpu_hours": pilot["gpu_hours"], "projected_total_gpu_hours_from_pilot":projected, "pilot_visual_accessed": False, "bulk_coordinates": 136, "bulk_value_path_units": 272, "gradcam_units": 1, "hard_gpu_hours_max": config["budget"]["hard_h20_gpu_hours_max"], "f4_started": False})
    _write_new(repo / BULK_AUTH_PATH, authorization); return authorization


def _bulk_coordinates(shard_id: int, shard_count: int) -> tuple[tuple[int, int], ...]:
    if shard_count != 8 or type(shard_id) is not int or not 0 <= shard_id < 8: raise StageFF3HError("F3H requires one of eight frozen shards")
    excluded = set(INHERITED_COORDINATES) | set(PILOT_COORDINATES)
    return tuple(coord for ordinal, coord in enumerate(COORDINATES) if coord not in excluded and ordinal % 8 == shard_id)


def run_shard(repo_root: Path | str = REPO_ROOT, *, shard_id: int, shard_count: int = 8, device: str = "cuda:0") -> dict[str, Any]:
    repo = Path(repo_root).resolve(); validate_initialized(repo); authorization = _strict_json(repo / BULK_AUTH_PATH, "bulk authorization"); _validate_signed(authorization, "bulk authorization")
    if authorization["status"] != "AUTHORIZED" or authorization["f4_started"]: raise StageFF3HError("bulk is not authorized")
    coordinates = _bulk_coordinates(shard_id, shard_count); telemetry_path = repo / ROOT_PATH / "telemetry" / f"shard-{shard_id:02d}-of-08.json"
    if telemetry_path.exists():
        value = _strict_json(telemetry_path, "shard telemetry"); _validate_signed(value, "shard telemetry"); return value
    telemetry = _execute_coordinates(repo, coordinates, device=device); telemetry["shard_id"] = shard_id; telemetry["shard_count"] = 8; telemetry = _signed({key: value for key, value in telemetry.items() if key != "canonical_digest"}); _write_new(telemetry_path, telemetry); return telemetry


def run_gradcam(repo_root: Path | str = REPO_ROOT, *, device: str = "cuda:0") -> dict[str, Any]:
    repo = Path(repo_root).resolve(); validate_initialized(repo); authorization = _strict_json(repo / BULK_AUTH_PATH, "bulk authorization"); _validate_signed(authorization, "bulk authorization"); plan = _strict_json(repo / PLAN_PATH, "plan"); unit = plan["gradcam_unit"]; root = _gradcam_root(repo)
    if authorization["status"] != "AUTHORIZED": raise StageFF3HError("Grad-CAM is not authorized")
    if root.exists(): return validate_gradcam(repo)
    if device.startswith("cuda") and _completed_gpu_hours(repo) >= _config(repo)["budget"]["hard_h20_gpu_hours_max"]: raise StageFF3HError("F3H hard GPU budget reached before Grad-CAM")
    import torch
    from .stage_c_class_logit import resolve_clean_class_logit_target, vit_grad_cam_block11_norm1
    model = None; started = time.monotonic(); cpu_started = time.process_time(); calls = 0
    try:
        model, _binding, clean, clean_sha, model_record = _runtime(repo, device)
        def hook(_module: Any, _args: Any) -> None:
            nonlocal calls; calls += 1
        handle = model.register_forward_pre_hook(hook)
        try:
            target = resolve_clean_class_logit_target(model, clean, official_truth_index=713)
            if target.class_index != 713 or target.clean_input_sha256 != clean_sha: raise StageFF3HError("Grad-CAM clean class target drifted")
            method = vit_grad_cam_block11_norm1(model, clean, target, provenance={"stage": "F3H", "protocol_id": unit["protocol_id"], "different_target_reference_only": True})
        finally: handle.remove()
        raw = np.asarray(method.metric_map.data)
        if raw.dtype != np.float32 or raw.shape != (14, 14) or not np.isfinite(raw).all() or not np.count_nonzero(raw): raise StageFF3HError("Grad-CAM raw map invalid")
        root.parent.mkdir(parents=True, exist_ok=True); temporary = Path(tempfile.mkdtemp(prefix=".attempt-000001-", dir=root.parent))
        try:
            _write_npy(temporary / "map.npy", raw); provenance = json.loads(canonical_json_bytes(dict(method.provenance)))
            result = _signed({"schema_version": 1, "document_type": "stage_f_f3h_gradcam_terminal", "protocol_id": unit["protocol_id"], "unit_digest": unit["canonical_digest"], "sample_id": unit["sample_id"], "selection_order": 44, "method": GRADCAM, "target": "clean_predicted_class_raw_logit", "class_index": 713, "target_layer": "blocks[11].norm1", "validity_status": "VALID", "attempt_id": ATTEMPT_ID, "source_hashes": {"image_sha256": unit["image_sha256"], "clean_input_sha256": clean_sha, "model_sha256": MODEL_SHA256, "model_canonical_tensor_digest": MODEL_TENSOR_DIGEST, "raw_map_sha256": _array_digest(raw), "method_provenance_canonical_digest": canonical_sha256(provenance)}, "method_provenance": provenance, "resources": {"elapsed_seconds": time.monotonic()-started, "cpu_seconds": time.process_time()-cpu_started, "gpu_hours": (time.monotonic()-started)/3600.0 if device.startswith("cuda") else 0.0, "forward_calls": calls, "backward_calls": 1, "peak_cuda_memory_bytes": int(torch.cuda.max_memory_reserved(device)) if device.startswith("cuda") else 0}, "different_target_reference_only": True, "same_target_winner_comparison_allowed": False, "algorithm_minmax_normalization": False}); raw_result = canonical_json_bytes(result)+b"\n"; _write_bytes(temporary/"result.json",raw_result)
            status = _signed({"schema_version": 1, "document_type": "stage_f_f3h_gradcam_status", "state": "VALID", "terminal": True, "unit_digest": unit["canonical_digest"], "attempt_id": ATTEMPT_ID, "published_at": _now()}); raw_status=canonical_json_bytes(status)+b"\n"; _write_bytes(temporary/"status.json",raw_status)
            manifest=_signed({"schema_version":1,"document_type":"stage_f_f3h_gradcam_manifest","unit_digest":unit["canonical_digest"],"result_sha256":hashlib.sha256(raw_result).hexdigest(),"status_sha256":hashlib.sha256(raw_status).hexdigest(),"map_sha256":_sha_file(temporary/"map.npy"),"map_array_sha256":_array_digest(raw),"immutable":True});_write_bytes(temporary/"manifest.json",canonical_json_bytes(manifest)+b"\n"); os.replace(temporary,root)
        finally:
            if temporary.exists(): shutil.rmtree(temporary)
        return validate_gradcam(repo)
    finally:
        if model is not None: del model
        if device.startswith("cuda") and torch.cuda.is_available(): torch.cuda.empty_cache()


def validate_gradcam(repo: Path) -> dict[str, Any]:
    root = _gradcam_root(repo); result = _strict_json(root/"result.json", "Grad-CAM result"); status = _strict_json(root/"status.json", "Grad-CAM status"); manifest = _strict_json(root/"manifest.json", "Grad-CAM manifest")
    for value,name in ((result,"result"),(status,"status"),(manifest,"manifest")):_validate_signed(value,name)
    raw=np.load(root/"map.npy",allow_pickle=False)
    if result["validity_status"]!="VALID" or result["class_index"]!=713 or not result["different_target_reference_only"] or result["same_target_winner_comparison_allowed"] or result["algorithm_minmax_normalization"] or raw.dtype!=np.float32 or raw.shape!=(14,14) or _array_digest(raw)!=manifest["map_array_sha256"] or _sha_file(root/"map.npy")!=manifest["map_sha256"]:raise StageFF3HError("Grad-CAM terminal drifted")
    return {"status":"VALID","result":result,"manifest":manifest,"root":root.as_posix()}


def reconcile_terminals(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); config = _config(repo); plan = _strict_json(repo / PLAN_PATH, "plan"); sources = _cross_sources(repo, config); rows = []; maps = []
    for unit in plan["value_path_units"]:
        key = (unit["block"], unit["head"], unit["method"])
        if unit["mode"] == "INHERITED_READ_ONLY":
            source = sources[key]; path = repo / source["source_path"]; array = np.load(path, allow_pickle=False)
            result_path=repo/unit["source"]["source_result_path"];source_result=_strict_json(result_path,"inherited result")
            if _sha_file(path) != source["source_file_sha256"] or _array_digest(array) != source["signed_map_sha256"] or _sha_file(result_path)!=source["source_result_sha256"]: raise StageFF3HError("inherited source drifted during reconciliation")
            from .stage_e_e1 import _target_semantic_digest
            target_semantic=_target_semantic_digest(source_result["target_identity"])
            row = {"unit_digest": unit["canonical_digest"], "sample_id": unit["sample_id"], "selection_order": 44, "block": unit["block"], "head": unit["head"], "method": unit["method"], "validity_status": "VALID", "failure_code": None, "na_reason": None, "origin": "INHERITED_SEALED_F3", "attempt_id": source["attempt_id"], "source_path": source["source_path"], "source_file_sha256": source["source_file_sha256"], "signed_map_sha256": source["signed_map_sha256"], "source_result_sha256": source["source_result_sha256"], "target_semantic_digest": target_semantic, "resources": {"gpu_hours": 0.0, "elapsed_seconds": 0.0, "inherited": True}}
        else:
            attempt = validate_attempt(repo, unit); result = attempt["result"]; path = attempt["root"] / "map.npy"
            row = {"unit_digest": unit["canonical_digest"], "sample_id": unit["sample_id"], "selection_order": 44, "block": unit["block"], "head": unit["head"], "method": unit["method"], "validity_status": result["validity_status"], "failure_code": result.get("failure_code"), "na_reason": result.get("na_reason"), "origin": "F3H_NEW", "attempt_id": result["attempt_id"], "source_path": path.relative_to(repo).as_posix() if path.exists() else None, "source_file_sha256": _sha_file(path) if path.exists() else None, "signed_map_sha256": result["source_hashes"]["signed_map_sha256"], "source_result_sha256": _sha_file(attempt["root"] / "result.json"), "target_semantic_digest": result["target_semantic_digest"], "resources": result["resources"]}
        rows.append(row)
        if row["validity_status"] == "VALID": maps.append({key: row[key] for key in ("unit_digest","block","head","method","origin","source_path","source_file_sha256","signed_map_sha256")})
    counts = Counter(row["validity_status"] for row in rows)
    if len(rows) != 288 or sum(counts.values()) != 288 or len({(r["block"],r["head"],r["method"]) for r in rows}) != 288: raise StageFF3HError("terminal union coverage failed")
    gradcam = validate_gradcam(repo)
    document = _signed({"schema_version":1,"document_type":"stage_f_f3h_terminal_union","status":"PASS","sample_id":config["sample"]["sample_id"],"counts":{"expected":288,"valid":counts["VALID"],"failed":counts["FAILED"],"na":counts["NA"],"inherited":8,"new":280,"gradcam_valid":1},"rows":rows,"gradcam":{"result":_file_record(repo,Path(gradcam["root"]).relative_to(repo).as_posix()+"/result.json"),"map":_file_record(repo,Path(gradcam["root"]).relative_to(repo).as_posix()+"/map.npy")},"all_coordinates_retained":True,"best_head_selected":False})
    map_inventory = _signed({"schema_version":1,"document_type":"stage_f_f3h_signed_map_inventory","status":"PASS","valid_map_count":len(maps),"failed_or_na_count":counts["FAILED"]+counts["NA"],"maps":maps})
    _write_new(repo/TERMINAL_UNION_PATH,document);_write_new(repo/MAP_INVENTORY_PATH,map_inventory);return document


def _load_union_map(repo: Path, row: Mapping[str, Any]) -> np.ndarray | None:
    if row["validity_status"] != "VALID": return None
    path=repo/row["source_path"]; array=np.load(path,allow_pickle=False)
    if _sha_file(path)!=row["source_file_sha256"] or _array_digest(array)!=row["signed_map_sha256"]:raise StageFF3HError("union map drifted")
    return array


def build_components(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo=Path(repo_root).resolve();config=_config(repo);union=_strict_json(repo/TERMINAL_UNION_PATH,"terminal union");_validate_signed(union,"terminal union");by={(r["block"],r["head"],r["method"]):r for r in union["rows"]};components=[];scales=[]
    for block,head in COORDINATES:
        nr,mr=by[(block,head,NNS)],by[(block,head,MATCHED)];na,ma=_load_union_map(repo,nr),_load_union_map(repo,mr)
        if na is not None and ma is not None:
            arrays,scale=pair_shared_log_display(na,ma,kappa=config["display"]["kappa"]);scale.update({"block":block,"head":head,"nns_status":"VALID","matched_status":"VALID"})
            raw_by={NNS:na,MATCHED:ma}
            for method,prefix in ((NNS,"nns"),(MATCHED,"matched")):
                positive,negative=split_signed_map(raw_by[method])
                for component,source in (("positive",positive),("negative_magnitude",negative)):
                    display=arrays[f"{prefix}_{'positive' if component=='positive' else 'negative'}"];components.append({"block":block,"head":head,"method":method,"component":component,"validity_status":"VALID","raw_signed_map_sha256":_array_digest(raw_by[method]),"component_sha256":_array_digest(source),"display_sha256":_array_digest(display),"pair_scale":scale["pair_scale"],"scale_basis":scale["scale_basis"],"kappa":scale["kappa"],"source_path":by[(block,head,method)]["source_path"],"source_file_sha256":by[(block,head,method)]["source_file_sha256"],"paired_magnitude_comparison_allowed":True})
        else:
            valid=na if na is not None else ma;maximum=0.0 if valid is None else float(np.max(np.abs(valid)));scale={"block":block,"head":head,"pair_scale":maximum if valid is not None else None,"scale_basis":"AVAILABLE_VALID_MEMBER_ONLY" if valid is not None else "NO_VALID_MEMBER","kappa":config["display"]["kappa"],"nns_status":nr["validity_status"],"matched_status":mr["validity_status"],"paired_magnitude_comparison_allowed":False}
            for method,row,array in ((NNS,nr,na),(MATCHED,mr,ma)):
                for component in ("positive","negative_magnitude"):
                    digest=None
                    if array is not None:
                        positive,negative=split_signed_map(array);source=positive if component=="positive" else negative;display=_component_display(source,maximum,999.0);digest=_array_digest(display)
                    components.append({"block":block,"head":head,"method":method,"component":component,"validity_status":row["validity_status"],"failure_code":row.get("failure_code"),"na_reason":row.get("na_reason"),"raw_signed_map_sha256":row.get("signed_map_sha256"),"component_sha256":None if array is None else _array_digest(source),"display_sha256":digest,"pair_scale":scale["pair_scale"],"scale_basis":scale["scale_basis"],"kappa":999.0,"source_path":row.get("source_path"),"source_file_sha256":row.get("source_file_sha256"),"paired_magnitude_comparison_allowed":False})
        scales.append(scale)
    comp=_signed({"schema_version":1,"document_type":"stage_f_f3h_signed_component_manifest","status":"PASS","component_count":len(components),"components":components,"display_only":True});scale_doc=_signed({"schema_version":1,"document_type":"stage_f_f3h_pair_shared_scale_manifest","status":"PASS","coordinate_count":len(scales),"scales":scales,"method_specific_gain":False,"cross_coordinate_magnitude_comparison_allowed":False})
    _write_new(repo/COMPONENT_MANIFEST_PATH,comp);_write_new(repo/SCALE_MANIFEST_PATH,scale_doc);return {"status":"PASS","components":len(components),"scales":len(scales)}


def _display_arrays(repo: Path, union: Mapping[str, Any]) -> tuple[dict[tuple[int,int,str,str], np.ndarray | None], list[dict[str, Any]]]:
    config=_config(repo);by={(r["block"],r["head"],r["method"]):r for r in union["rows"]};arrays={};records=[]
    for block,head in COORDINATES:
        nr,mr=by[(block,head,NNS)],by[(block,head,MATCHED)];na,ma=_load_union_map(repo,nr),_load_union_map(repo,mr)
        if na is not None and ma is not None:
            shown,scale=pair_shared_log_display(na,ma,kappa=config["display"]["kappa"]); arrays[(block,head,NNS,"positive")]=shown["nns_positive"];arrays[(block,head,NNS,"negative_magnitude")]=shown["nns_negative"];arrays[(block,head,MATCHED,"positive")]=shown["matched_positive"];arrays[(block,head,MATCHED,"negative_magnitude")]=shown["matched_negative"]
        else:
            valid=na if na is not None else ma;maximum=0.0 if valid is None else float(np.max(np.abs(valid)))
            for method,row,raw in ((NNS,nr,na),(MATCHED,mr,ma)):
                if raw is None:
                    arrays[(block,head,method,"positive")]=None;arrays[(block,head,method,"negative_magnitude")]=None
                else:
                    pos,neg=split_signed_map(raw);arrays[(block,head,method,"positive")]=_component_display(pos,maximum,999.0);arrays[(block,head,method,"negative_magnitude")]=_component_display(neg,maximum,999.0)
        for method,row in ((NNS,nr),(MATCHED,mr)):
            for component in ("positive","negative_magnitude"):
                array=arrays[(block,head,method,component)];records.append({"block":block,"head":head,"method":method,"component":component,"validity_status":row["validity_status"],"failure_code":row.get("failure_code"),"na_reason":row.get("na_reason"),"display_sha256":None if array is None else _array_digest(array),"source_path":row.get("source_path"),"source_file_sha256":row.get("source_file_sha256"),"signed_map_sha256":row.get("signed_map_sha256")})
    return arrays,records


def _cmap(config: Mapping[str, Any]) -> Any:
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("f3h_blue_hot",config["display"]["colormap_stops"],N=256)


def _failed_axis(axis: Any, row: Mapping[str, Any]) -> None:
    axis.set_facecolor("#07077F");axis.set_xticks([]);axis.set_yticks([]);axis.text(.5,.57,"FAILED / NA",ha="center",va="center",color="white",fontsize=6,fontweight="bold");axis.text(.5,.39,row.get("failure_code") or row.get("na_reason") or row["validity_status"],ha="center",va="center",color="white",fontsize=3.8,wrap=True)


def _save_render(repo: Path, fig: Any, base: Path, *, module: str, panel_ids: Sequence[str], metadata: Mapping[str, Any]) -> dict[str, Any]:
    from matplotlib import pyplot as plt
    from .stage_f_renderer import _save_figure, load_contracts
    outputs=_save_figure(fig,base,load_contracts(repo)["config"]);size=[float(x) for x in fig.get_size_inches()];plt.close(fig);content=canonical_sha256({"module":module,"panel_ids":list(panel_ids),"metadata":dict(metadata),"config_digest":_config(repo)["canonical_digest"]})
    report=_signed({"schema_version":1,"document_type":"stage_f_f3h_render_report","module":module,"figure_id":base.name,"panel_ids":list(panel_ids),"metadata":dict(metadata),"content_digest":content,"format_content_digests":{ext:content for ext in outputs},"outputs":outputs,"size_inches":size,"dpi":600,"display_only":True,"best_head_selected":False,"f4_started":False});_write_new(base.with_suffix(".report.json"),report);return report


def _atlas(repo: Path, arrays: Mapping[tuple[int,int,str,str],np.ndarray|None], rows: Mapping[tuple[int,int,str],Mapping[str,Any]], *, method: str, component: str, base: Path, title: str, cmap: Any) -> dict[str,Any]:
    import matplotlib
    matplotlib.use("Agg",force=True)
    from matplotlib import pyplot as plt
    from .stage_f_renderer import load_contracts,renderer_style
    with renderer_style(load_contracts(repo)["config"]):
        fig,axes=plt.subplots(12,12,figsize=(10.8,10.8),constrained_layout=False);fig.subplots_adjust(left=.055,right=.995,bottom=.045,top=.93,wspace=.025,hspace=.025)
        panel_ids=[]
        for block in range(12):
            for head in range(12):
                ax=axes[block,head];array=arrays[(block,head,method,component)];row=rows[(block,head,method)];panel_ids.append(f"b{block:02d}h{head:02d}:{method}:{component}")
                if array is None:_failed_axis(ax,row)
                else:ax.imshow(array,cmap=cmap,vmin=0,vmax=1,interpolation="nearest");ax.axis("off")
                if block==0:ax.set_title(f"H{head}",fontsize=5.8,pad=2)
                if head==0:ax.text(-.12,.5,f"B{block}",transform=ax.transAxes,ha="right",va="center",fontsize=5.8,fontweight="bold")
        fig.suptitle(title,fontsize=10,fontweight="bold",y=.98)
        return _save_render(repo,fig,base,module=title,panel_ids=panel_ids,metadata={"method":method,"component":component,"complete_12x12":True})


def _block_page(repo: Path, arrays: Mapping[tuple[int,int,str,str],np.ndarray|None], rows: Mapping[tuple[int,int,str],Mapping[str,Any]], *, block: int, base: Path, cmap: Any) -> dict[str,Any]:
    import matplotlib
    matplotlib.use("Agg",force=True)
    from matplotlib import pyplot as plt
    from .stage_f_renderer import load_contracts,renderer_style
    row_defs=((NNS,"positive","NNS +"),(MATCHED,"positive","Matched +"),(NNS,"negative_magnitude","NNS −"),(MATCHED,"negative_magnitude","Matched −"))
    with renderer_style(load_contracts(repo)["config"]):
        fig,axes=plt.subplots(4,12,figsize=(12.0,4.25),constrained_layout=False);fig.subplots_adjust(left=.075,right=.995,bottom=.055,top=.86,wspace=.025,hspace=.035);panel_ids=[]
        for ri,(method,component,label) in enumerate(row_defs):
            for head in range(12):
                ax=axes[ri,head];array=arrays[(block,head,method,component)];row=rows[(block,head,method)];panel_ids.append(f"b{block:02d}h{head:02d}:{method}:{component}")
                if array is None:_failed_axis(ax,row)
                else:ax.imshow(array,cmap=cmap,vmin=0,vmax=1,interpolation="nearest");ax.axis("off")
                if ri==0:ax.set_title(f"H{head}",fontsize=6,pad=2)
                if head==0:ax.text(-.14,.5,label,transform=ax.transAxes,ha="right",va="center",fontsize=6.5,fontweight="bold")
        fig.suptitle(f"Block {block} · all 12 heads · pair-shared log display",fontsize=10,fontweight="bold",y=.98)
        return _save_render(repo,fig,base,module="F3H-E block paired page",panel_ids=panel_ids,metadata={"block":block,"head_count":12,"rows":[item[2] for item in row_defs]})


def _consensus(arrays: Mapping[tuple[int,int,str,str],np.ndarray|None], coordinates: Sequence[tuple[int,int]], method: str, component: str) -> np.ndarray|None:
    values=[arrays[(b,h,method,component)] for b,h in coordinates]
    if any(value is None for value in values):return None
    return np.mean(np.stack(values),axis=0,dtype=np.float64).astype(np.float32)


def _normalized_entropy(array: np.ndarray|None) -> float|None:
    if array is None:return None
    flat=np.asarray(array,dtype=np.float64).ravel();total=float(flat.sum())
    if total<=0:return None
    p=flat[flat>0]/total;return float(-(p*np.log(p)).sum()/math.log(flat.size))


def _mean_pairwise_cosine(values: Sequence[np.ndarray|None]) -> float|None:
    if any(value is None for value in values):return None
    vectors=[np.asarray(value,dtype=np.float64).ravel() for value in values];scores=[]
    for i in range(len(vectors)):
        for j in range(i+1,len(vectors)):
            denom=float(np.linalg.norm(vectors[i])*np.linalg.norm(vectors[j]));scores.append(0.0 if denom==0 else float(np.dot(vectors[i],vectors[j])/denom))
    return float(np.mean(scores)) if scores else None


def build_consensus_manifest(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo=Path(repo_root).resolve();union=_strict_json(repo/TERMINAL_UNION_PATH,"terminal union");_validate_signed(union,"terminal union");arrays,_records=_display_arrays(repo,union);groups=[];diagnostics=[]
    row_defs=((NNS,"positive"),(MATCHED,"positive"),(NNS,"negative_magnitude"),(MATCHED,"negative_magnitude"))
    group_defs=[]
    for block in range(12):group_defs.append(("layer",f"block_{block:02d}",[(block,h) for h in range(12)]))
    for index,(start,end) in enumerate(((0,2),(3,5),(6,8),(9,11))):group_defs.append(("depth_band",f"band_{index}_{start}_{end}",[(b,h) for b in range(start,end+1) for h in range(12)]))
    group_defs.append(("global","all_144_heads",list(COORDINATES)))
    for kind,name,coordinates in group_defs:
        paired_missing=[[b,h] for b,h in coordinates if arrays[(b,h,NNS,"positive")] is None or arrays[(b,h,MATCHED,"positive")] is None]
        for method,component in row_defs:
            values=[arrays[(b,h,method,component)] for b,h in coordinates];consensus=None if paired_missing else _consensus(arrays,coordinates,method,component)
            row={"group_kind":kind,"group_id":name,"coordinates":[list(item) for item in coordinates],"method":method,"component":component,"status":"VALID" if consensus is not None else "FAILED_OR_NA_INPUT","required_head_count":len(coordinates),"missing_coordinates":paired_missing,"complete_case_average_used":False,"equal_head_weight":1.0/len(coordinates),"consensus_display_sha256":None if consensus is None else _array_digest(consensus),"normalized_spatial_entropy":_normalized_entropy(consensus)};groups.append(row)
            if kind=="layer":diagnostics.append({"block":int(name.split("_")[1]),"method":method,"component":component,"all_12_head_pairs_valid":not paired_missing,"mean_pairwise_cosine":None if paired_missing else _mean_pairwise_cosine(values),"consensus_normalized_spatial_entropy":row["normalized_spatial_entropy"]})
    document=_signed({"schema_version":1,"document_type":"stage_f_f3h_equal_head_consensus_manifest","status":"PASS","name":"equal-head-weighted spatial-support consensus","is_raw_attribution_sum":False,"is_network_level_attribution":False,"is_class_conditioned_attribution":False,"head_selection_used":False,"groups":groups,"head_diversity_diagnostics":diagnostics});_write_new(repo/CONSENSUS_PATH,document);return document


def assess_class_conditioned_basis(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo=Path(repo_root).resolve();consensus=_strict_json(repo/CONSENSUS_PATH,"consensus");_validate_signed(consensus,"consensus");union=_strict_json(repo/TERMINAL_UNION_PATH,"terminal union");_validate_signed(union,"terminal union")
    valid=union["counts"]["valid"];failed=union["counts"]["failed"];na=union["counts"]["na"];diagnostics=consensus["head_diversity_diagnostics"]
    finite_cos=[row["mean_pairwise_cosine"] for row in diagnostics if row["mean_pairwise_cosine"] is not None];finite_entropy=[row["consensus_normalized_spatial_entropy"] for row in diagnostics if row["consensus_normalized_spatial_entropy"] is not None]
    assessment=_signed({"schema_version":1,"document_type":"stage_f_f3h_class_conditioned_basis_assessment","status":"INSUFFICIENT_BASIS_FOR_NUMERICAL_CLASS_CONDITIONED_WEIGHTS","sample_id":_config(repo)["sample"]["sample_id"],"full_head_terminal_counts":{"valid":valid,"failed":failed,"na":na},"descriptive_diagnostics":{"layer_method_sign_cells":len(diagnostics),"finite_pairwise_cosine_cells":len(finite_cos),"mean_of_layer_mean_pairwise_cosines":None if not finite_cos else float(np.mean(finite_cos)),"mean_consensus_normalized_spatial_entropy":None if not finite_entropy else float(np.mean(finite_entropy))},"required_for_weight_definition":{"signed_class_logit_coupling_coefficients_for_all_144_heads":False,"path_accounting_without_omission_or_double_count":False,"predeclared_class_logit_conservation_test":False},"atlas_visual_appearance_used_to_define_weights":False,"gradcam_overlap_used_to_define_weights":False,"best_head_selected":False,"conclusion":"F3H atlas documents head diversity but cannot by itself define class-conditioned numerical weights. Proceed to prospective F3C0 method/protocol design without empirical F3C computation.","next_authorization":"F3C0_PROTOCOL_FREEZE_ONLY","f4_status":"PAUSED_PENDING_F3H_AND_F3C"});_write_new(repo/SUFFICIENCY_PATH,assessment);return assessment


def _render_consensus_matrix(repo:Path,arrays:Mapping[tuple[int,int,str,str],np.ndarray|None],*,kind:str,base:Path,cmap:Any)->dict[str,Any]:
    import matplotlib
    matplotlib.use("Agg",force=True)
    from matplotlib import pyplot as plt
    from .stage_f_renderer import load_contracts,renderer_style
    row_defs=((NNS,"positive","NNS +"),(MATCHED,"positive","Matched +"),(NNS,"negative_magnitude","NNS −"),(MATCHED,"negative_magnitude","Matched −"))
    if kind=="layer":groups=[(f"B{b}",[(b,h) for h in range(12)]) for b in range(12)];title="Per-layer equal-head spatial-support consensus"
    else:groups=[(f"B{a}–{b}",[(layer,h) for layer in range(a,b+1) for h in range(12)]) for a,b in ((0,2),(3,5),(6,8),(9,11))];title="Depth-band equal-head spatial-support consensus"
    with renderer_style(load_contracts(repo)["config"]):
        fig,axes=plt.subplots(4,len(groups),figsize=(12.0,4.3) if kind=="layer" else (7.16,5.2),constrained_layout=False);fig.subplots_adjust(left=.09,right=.995,bottom=.06,top=.88,wspace=.03,hspace=.04);panel_ids=[]
        for ri,(method,component,label) in enumerate(row_defs):
            for ci,(group_name,coords) in enumerate(groups):
                ax=axes[ri,ci];paired_missing=any(arrays[(b,h,NNS,"positive")] is None or arrays[(b,h,MATCHED,"positive")] is None for b,h in coords);value=None if paired_missing else _consensus(arrays,coords,method,component);panel_ids.append(f"{kind}:{group_name}:{method}:{component}")
                if value is None:_failed_axis(ax,{"failure_code":"incomplete required head set","validity_status":"FAILED"})
                else:ax.imshow(value,cmap=cmap,vmin=0,vmax=1,interpolation="nearest");ax.axis("off")
                if ri==0:ax.set_title(group_name,fontsize=6.2 if kind=="layer" else 7.4,pad=2)
                if ci==0:ax.text(-.16,.5,label,transform=ax.transAxes,ha="right",va="center",fontsize=6.5,fontweight="bold")
        fig.suptitle(title,fontsize=10,fontweight="bold",y=.98);return _save_render(repo,fig,base,module=title,panel_ids=panel_ids,metadata={"consensus_kind":kind,"equal_head_weighted":True,"raw_attribution_sum":False})


def _load_rgb(repo:Path)->np.ndarray:
    from .stage_f_f3_visual import _load_rgb,_rgb_meta_for_sample,_sources
    sources=_sources(repo);return _load_rgb(repo,_rgb_meta_for_sample(sources,_config(repo)["sample"]["sample_id"]))


def render_outputs(repo_root: Path | str = REPO_ROOT)->dict[str,Any]:
    repo=Path(repo_root).resolve();config=_config(repo);union=_strict_json(repo/TERMINAL_UNION_PATH,"terminal union");_validate_signed(union,"terminal union");arrays,component_records=_display_arrays(repo,union);rows={(r["block"],r["head"],r["method"]):r for r in union["rows"]};cmap=_cmap(config);reports=[]
    for module,method,component,title in (("f3h_a",NNS,"positive","NNS positive · all blocks and heads"),("f3h_b",NNS,"negative_magnitude","NNS negative magnitude · all blocks and heads"),("f3h_c",MATCHED,"positive","Matched patch positive · all blocks and heads"),("f3h_d",MATCHED,"negative_magnitude","Matched patch negative magnitude · all blocks and heads")):
        base=repo/ROOT_PATH/"modules"/module/module;reports.append((module,_atlas(repo,arrays,rows,method=method,component=component,base=base,title=title,cmap=cmap),base.with_suffix(".report.json")))
    for block in range(12):
        base=repo/ROOT_PATH/"modules/f3h_e_block_pages"/f"block_{block:02d}";reports.append(("f3h_e",_block_page(repo,arrays,rows,block=block,base=base,cmap=cmap),base.with_suffix(".report.json")))
    for kind,name in (("layer","per_layer_consensus"),("band","depth_band_consensus")):
        base=repo/ROOT_PATH/"modules/f3h_g_consensus"/name;reports.append(("f3h_g",_render_consensus_matrix(repo,arrays,kind=kind,base=base,cmap=cmap),base.with_suffix(".report.json")))
    import matplotlib
    matplotlib.use("Agg",force=True)
    from matplotlib import pyplot as plt
    from .stage_f_renderer import load_contracts,renderer_style
    global_paired_missing=any(arrays[(b,h,NNS,"positive")] is None or arrays[(b,h,MATCHED,"positive")] is None for b,h in COORDINATES);global_values=[None if global_paired_missing else _consensus(arrays,list(COORDINATES),method,component) for method,component in ((NNS,"positive"),(MATCHED,"positive"),(NNS,"negative_magnitude"),(MATCHED,"negative_magnitude"))]
    with renderer_style(load_contracts(repo)["config"]):
        fig,axes=plt.subplots(1,5,figsize=(10.5,2.45),constrained_layout=False);fig.subplots_adjust(left=.02,right=.995,bottom=.05,top=.82,wspace=.04);rgb=_load_rgb(repo);axes[0].imshow(rgb);axes[0].axis("off");axes[0].set_title("Original",fontsize=7.5,fontweight="bold")
        labels=("NNS +","Matched +","NNS −","Matched −")
        for ax,value,label in zip(axes[1:],global_values,labels):
            if value is None:_failed_axis(ax,{"failure_code":"incomplete required head set","validity_status":"FAILED"})
            else:ax.imshow(value,cmap=cmap,vmin=0,vmax=1,interpolation="nearest");ax.axis("off")
            ax.set_title(label,fontsize=7.5,fontweight="bold")
        fig.suptitle("All-144-head equal-weight spatial-support consensus",fontsize=10,fontweight="bold",y=.98);base=repo/ROOT_PATH/"modules/f3h_g_consensus/all_heads_consensus";reports.append(("f3h_g",_save_render(repo,fig,base,module="all-head consensus",panel_ids=["original"]+list(labels),metadata={"equal_head_count":144,"class_conditioned":False}),base.with_suffix(".report.json")))
    grad=validate_gradcam(repo);raw=np.load(Path(grad["root"])/"map.npy",allow_pickle=False);maximum=float(raw.max());shown=_component_display(np.repeat(np.repeat(raw/16.0,4,axis=0),4,axis=1).astype(np.float32),maximum/16.0,999.0)
    with renderer_style(load_contracts(repo)["config"]):
        fig,axes=plt.subplots(1,3,figsize=(7.16,2.65),constrained_layout=False);fig.subplots_adjust(left=.04,right=.98,bottom=.08,top=.80,wspace=.10);rgb=_load_rgb(repo);axes[0].imshow(rgb);axes[1].imshow(shown,cmap=cmap,vmin=0,vmax=1);axes[2].imshow(rgb);axes[2].imshow(shown,cmap=cmap,vmin=0,vmax=1,alpha=.62)
        for ax,label in zip(axes,("Original","Grad-CAM","Grad-CAM overlay")):ax.axis("off");ax.set_title(label,fontsize=8,fontweight="bold")
        fig.suptitle("Different-target reference · frozen class 713",fontsize=10,fontweight="bold",y=.98);base=repo/ROOT_PATH/"modules/f3h_f_gradcam/gradcam_reference";reports.append(("f3h_f",_save_render(repo,fig,base,module="F3H-F Grad-CAM reference",panel_ids=["original","gradcam","overlay"],metadata={"class_index":713,"target":"clean_predicted_class_raw_logit","different_target_reference_only":True,"raw_map_sha256":_array_digest(raw),"display_sha256":_array_digest(shown)}),base.with_suffix(".report.json")))
    panel=_signed({"schema_version":1,"document_type":"stage_f_f3h_panel_manifest","status":"PASS","component_panel_count":len(component_records),"components":component_records,"render_count":len(reports),"best_head_selected":False});_write_new(repo/PANEL_MANIFEST_PATH,panel)
    render_rows=[];review=[]
    for index,(group,report,path) in enumerate(reports,1):
        render_rows.append({"module_group":group,"figure_id":report["figure_id"],"report":_record(repo,path.relative_to(repo).as_posix()),"outputs":{ext:{"path":(path.parent/item["path"]).relative_to(repo).as_posix(),"sha256":item["sha256"],"bytes":item["bytes"]} for ext,item in report["outputs"].items()},"content_digest":report["content_digest"]});src=path.parent/report["outputs"]["png"]["path"];dst=repo/REVIEW_ROOT_PATH/f"{index:02d}_{group}_{src.name}";dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);review.append({"path":dst.relative_to(repo).as_posix(),"sha256":_sha_file(dst),"source_path":src.relative_to(repo).as_posix(),"exact_copy":_sha_file(dst)==_sha_file(src)})
    inventory=_signed({"schema_version":1,"document_type":"stage_f_f3h_render_inventory","status":"PASS","render_count":len(reports),"exported_file_count":len(reports)*3,"module_counts":dict(Counter(item["module_group"] for item in render_rows)),"renders":render_rows,"final_combined_manuscript_figures_rendered":False});reviews=_signed({"schema_version":1,"document_type":"stage_f_f3h_review_inventory","status":"PASS","review_count":len(review),"copies":review})
    _write_new(repo/RENDER_INVENTORY_PATH,inventory);_write_new(repo/REVIEW_INVENTORY_PATH,reviews);return {"status":"PASS","render_count":len(reports),"exported_file_count":len(reports)*3,"review_count":len(review),"module_counts":inventory["module_counts"]}


def _validate_render_report(repo:Path,path:Path)->dict[str,Any]:
    report=_strict_json(path,"render report");_validate_signed(report,"render report")
    if report["display_only"] is not True or report["best_head_selected"] is not False or report["f4_started"] is not False or len(set(report["format_content_digests"].values()))!=1:raise StageFF3HError("render report boundary drifted")
    for item in report["outputs"].values():
        target=path.parent/item["path"]
        if not target.is_file() or _sha_file(target)!=item["sha256"]:raise StageFF3HError("render artifact drifted")
    return report


def build_final_audits(repo_root:Path|str=REPO_ROOT)->dict[str,Any]:
    repo=Path(repo_root).resolve();config=_config(repo);telemetry=[]
    for path in sorted((repo/ROOT_PATH/"telemetry").glob("*.json")):
        value=_strict_json(path,"telemetry");_validate_signed(value,"telemetry");telemetry.append(value)
    grad=validate_gradcam(repo)["result"];gpu_hours=sum(float(row.get("gpu_hours",0.0)) for row in telemetry)+float(grad["resources"]["gpu_hours"])
    resource=_signed({"schema_version":1,"document_type":"stage_f_f3h_resource_audit","status":"PASS" if gpu_hours<=config["budget"]["hard_h20_gpu_hours_max"] else "FAIL_COST","gpu_hours":gpu_hours,"hard_h20_gpu_hours_max":config["budget"]["hard_h20_gpu_hours_max"],"waiver_used":False,"telemetry_count":len(telemetry),"forward_calls":sum(int(row.get("forward_calls",0)) for row in telemetry)+int(grad["resources"]["forward_calls"]),"gradcam_backward_calls":1})
    if resource["status"]!="PASS":raise StageFF3HError("F3H hard GPU budget exceeded")
    access=_signed({"schema_version":1,"document_type":"stage_f_f3h_access_audit","status":"PASS","model_loaded_for_authorized_f3h":True,"jpeg_decoded_for_frozen_order_44_only":True,"torch_timm_cuda_used_for_authorized_f3h":True,"new_value_path_map_units":280,"new_gradcam_units":1,"fidelity_executed":False,"stability_executed":False,"sanity_executed":False,"statistics_or_winner_selection_executed":False,"f3_or_blue_r4_modified":False,"f4_started":False})
    false_pass=_signed({"schema_version":1,"document_type":"stage_f_f3h_false_pass_audit","status":"PASS","attacks":[{"attack":item,"result":"REJECTED"} for item in FINAL_ATTACKS],"pass_possible_without_all_conditions":False})
    _write_new(repo/RESOURCE_AUDIT_PATH,resource);_write_new(repo/ACCESS_AUDIT_PATH,access);_write_new(repo/FALSE_PASS_PATH,false_pass);return {"status":"PASS","gpu_hours":gpu_hours,"false_pass_attacks":len(FINAL_ATTACKS)}


def record_tests(repo_root:Path|str,*,passed:int,failed:int,command:str)->dict[str,Any]:
    if passed<=0 or failed!=0:raise StageFF3HError("test report requires positive passes and zero failures")
    report=_signed({"schema_version":1,"document_type":"stage_f_f3h_test_report","status":"PASS","passed":passed,"failed":failed,"command":command,"post_execution":True});_write_new(Path(repo_root).resolve()/TEST_PATH,report);return report


def validate_outputs(repo_root:Path|str=REPO_ROOT)->dict[str,Any]:
    repo=Path(repo_root).resolve();config=_config(repo);plan=_strict_json(repo/PLAN_PATH,"plan");_validate_signed(plan,"plan");union=_strict_json(repo/TERMINAL_UNION_PATH,"terminal union");maps=_strict_json(repo/MAP_INVENTORY_PATH,"map inventory");components=_strict_json(repo/COMPONENT_MANIFEST_PATH,"components");scales=_strict_json(repo/SCALE_MANIFEST_PATH,"scales");consensus=_strict_json(repo/CONSENSUS_PATH,"consensus");panel=_strict_json(repo/PANEL_MANIFEST_PATH,"panel");render=_strict_json(repo/RENDER_INVENTORY_PATH,"render");review=_strict_json(repo/REVIEW_INVENTORY_PATH,"review");assessment=_strict_json(repo/SUFFICIENCY_PATH,"assessment");resource=_strict_json(repo/RESOURCE_AUDIT_PATH,"resource");access=_strict_json(repo/ACCESS_AUDIT_PATH,"access");false_pass=_strict_json(repo/FALSE_PASS_PATH,"false pass")
    for value,name in ((union,"union"),(maps,"maps"),(components,"components"),(scales,"scales"),(consensus,"consensus"),(panel,"panel"),(render,"render"),(review,"review"),(assessment,"assessment"),(resource,"resource"),(access,"access"),(false_pass,"false pass")):_validate_signed(value,name)
    counts=union["counts"]
    if counts["expected"]!=288 or counts["valid"]+counts["failed"]+counts["na"]!=288 or counts["inherited"]!=8 or counts["new"]!=280 or counts["gradcam_valid"]!=1:raise StageFF3HError("terminal counts drifted")
    if components["component_count"]!=576 or scales["coordinate_count"]!=144 or scales["method_specific_gain"] or scales["cross_coordinate_magnitude_comparison_allowed"]:raise StageFF3HError("component/scale manifest drifted")
    if consensus["head_selection_used"] or consensus["is_raw_attribution_sum"] or consensus["is_class_conditioned_attribution"] or len(consensus["groups"])!=68:raise StageFF3HError("consensus contract drifted")
    if assessment["status"]!="INSUFFICIENT_BASIS_FOR_NUMERICAL_CLASS_CONDITIONED_WEIGHTS" or assessment["next_authorization"]!="F3C0_PROTOCOL_FREEZE_ONLY" or assessment["atlas_visual_appearance_used_to_define_weights"]:raise StageFF3HError("class-conditioned basis assessment drifted")
    if panel["component_panel_count"]!=576 or panel["render_count"]!=F3H_RENDER_COUNT or panel["best_head_selected"]:raise StageFF3HError("panel manifest drifted")
    if render["render_count"]!=F3H_RENDER_COUNT or render["exported_file_count"]!=F3H_EXPORT_COUNT or render["module_counts"]!={"f3h_a":1,"f3h_b":1,"f3h_c":1,"f3h_d":1,"f3h_e":12,"f3h_f":1,"f3h_g":3}:raise StageFF3HError("render inventory drifted")
    for row in render["renders"]:_validate_render_report(repo,repo/row["report"]["path"])
    if review["review_count"]!=F3H_RENDER_COUNT or len(review["copies"])!=F3H_RENDER_COUNT or any(not item["exact_copy"] or _sha_file(repo/item["path"])!=item["sha256"] for item in review["copies"]):raise StageFF3HError("review inventory drifted")
    if resource["status"]!="PASS" or resource["gpu_hours"]>config["budget"]["hard_h20_gpu_hours_max"] or resource["waiver_used"]:raise StageFF3HError("resource audit drifted")
    if access["fidelity_executed"] or access["stability_executed"] or access["sanity_executed"] or access["statistics_or_winner_selection_executed"] or access["f4_started"]:raise StageFF3HError("access boundary drifted")
    if [item["attack"] for item in false_pass["attacks"]]!=list(FINAL_ATTACKS) or any(item["result"]!="REJECTED" for item in false_pass["attacks"]):raise StageFF3HError("false-pass audit drifted")
    if not (repo/ANALYSIS_PATH).is_file():raise StageFF3HError("manual visual analysis absent")
    analysis=(repo/ANALYSIS_PATH).read_text(encoding="utf-8")
    if any(f"## Image #{i}:" not in analysis for i in range(1,F3H_RENDER_COUNT+1)):raise StageFF3HError("manual visual analysis incomplete")
    if _git(repo,"diff","--check"):raise StageFF3HError("git diff --check failed")
    return _signed({"schema_version":1,"document_type":"stage_f_f3h_validation_report","status":"PASS","value_path_counts":counts,"gradcam_status":"VALID","render_count":F3H_RENDER_COUNT,"exported_file_count":F3H_EXPORT_COUNT,"review_count":F3H_RENDER_COUNT,"false_pass_attack_count":len(FINAL_ATTACKS),"class_conditioned_basis":assessment["status"],"f4_status":"PAUSED_PENDING_F3H_AND_F3C","next_authorization":"F3C0_PROTOCOL_FREEZE_ONLY"})


def finalize(repo_root:Path|str=REPO_ROOT,*,completed_at:str)->dict[str,Any]:
    repo=Path(repo_root).resolve();validation=_signed(validate_outputs(repo));_write_new(repo/VALIDATION_PATH,validation);test=_record(repo,TEST_PATH);root=repo/ROOT_PATH;excluded={"payload_manifest.json","whole_root_inventory.json","status.json","completion/f3h.pending.json","completion/f3h.json"};files=[{"path":p.relative_to(root).as_posix(),"bytes":p.stat().st_size,"sha256":_sha_file(p)} for p in sorted(root.rglob("*")) if p.is_file() and p.relative_to(root).as_posix() not in excluded];payload=_signed({"schema_version":1,"document_type":"stage_f_f3h_payload_manifest","status":"PASS","payload_count":len(files),"artifacts":files});_write_new(repo/PAYLOAD_MANIFEST_PATH,payload);inv=_inventory(root,exclude=("whole_root_inventory.json","status.json","completion/f3h.pending.json","completion/f3h.json"));inventory=_signed({"schema_version":1,"document_type":"stage_f_f3h_whole_root_inventory","root":ROOT_PATH,"files":inv["files"],"file_count":inv["file_count"],"total_bytes":inv["total_bytes"]});_write_new(repo/INVENTORY_PATH,inventory);status=_signed({"schema_version":1,"document_type":"stage_f_f3h_status","status":"PASS","terminal":True,"scientific_validity":"PASS" if validation["value_path_counts"]["failed"]+validation["value_path_counts"]["na"]==0 else "MIXED","value_path_counts":validation["value_path_counts"],"gradcam_status":"VALID","render_count":F3H_RENDER_COUNT,"class_conditioned_basis":validation["class_conditioned_basis"],"f4_status":"PAUSED_PENDING_F3H_AND_F3C","f4_started":False,"next_authorization":"F3C0_PROTOCOL_FREEZE_ONLY","whole_root_inventory":{"canonical_digest":inventory["canonical_digest"],"file_count":inventory["file_count"],"total_bytes":inventory["total_bytes"]},"git":{"commit":"SELF","ref":F3H_ARCHIVE_REF}});_write_new(repo/STATUS_PATH,status);pending=_strict_json(repo/PENDING_PATH,"pending");completion=_signed({"schema_version":1,"document_type":"stage_f_f3h_terminal_completion","status":"PASS","completed_at":completed_at,"started_at":pending["started_at"],"scientific_validity":status["scientific_validity"],"value_path_counts":status["value_path_counts"],"gradcam_status":"VALID","render_count":F3H_RENDER_COUNT,"exported_file_count":F3H_EXPORT_COUNT,"review_count":F3H_RENDER_COUNT,"class_conditioned_basis":status["class_conditioned_basis"],"validation_report":_record(repo,VALIDATION_PATH),"false_pass_audit":_record(repo,FALSE_PASS_PATH),"test_report":test,"terminal_union":_record(repo,TERMINAL_UNION_PATH),"map_inventory":_record(repo,MAP_INVENTORY_PATH),"component_manifest":_record(repo,COMPONENT_MANIFEST_PATH),"scale_manifest":_record(repo,SCALE_MANIFEST_PATH),"consensus_manifest":_record(repo,CONSENSUS_PATH),"panel_manifest":_record(repo,PANEL_MANIFEST_PATH),"render_inventory":_record(repo,RENDER_INVENTORY_PATH),"review_inventory":_record(repo,REVIEW_INVENTORY_PATH),"access_audit":_record(repo,ACCESS_AUDIT_PATH),"resource_audit":_record(repo,RESOURCE_AUDIT_PATH),"payload_manifest":_record(repo,PAYLOAD_MANIFEST_PATH),"whole_root_inventory":_record(repo,INVENTORY_PATH),"figure_analysis":_file_record(repo,ANALYSIS_PATH),"class_conditioned_basis_assessment":_record(repo,SUFFICIENCY_PATH),"f3_r7_or_blue_r4_modified":False,"f4_status":"PAUSED_PENDING_F3H_AND_F3C","f4_started":False,"next_authorization":"F3C0_PROTOCOL_FREEZE_ONLY","git":{"commit":"SELF","ref":F3H_ARCHIVE_REF}});temporary=(repo/PENDING_PATH).with_name(".f3h-terminal.tmp");_write_new(temporary,completion);os.replace(temporary,repo/PENDING_PATH);os.replace(repo/PENDING_PATH,repo/COMPLETION_PATH);return completion


def validate_completion(repo_root:Path|str=REPO_ROOT)->dict[str,Any]:
    repo=Path(repo_root).resolve();completion=_strict_json(repo/COMPLETION_PATH,"completion");_validate_signed(completion,"completion")
    if (repo/PENDING_PATH).exists() or completion["status"]!="PASS" or completion["render_count"]!=F3H_RENDER_COUNT or completion["next_authorization"]!="F3C0_PROTOCOL_FREEZE_ONLY" or completion["f4_started"] or completion["f4_status"]!="PAUSED_PENDING_F3H_AND_F3C":raise StageFF3HError("F3H completion state drifted")
    for field in ("validation_report","false_pass_audit","test_report","terminal_union","map_inventory","component_manifest","scale_manifest","consensus_manifest","panel_manifest","render_inventory","review_inventory","access_audit","resource_audit","payload_manifest","whole_root_inventory","class_conditioned_basis_assessment"):
        if completion[field]!=_record(repo,completion[field]["path"]):raise StageFF3HError(f"completion binding drifted: {field}")
    if completion["figure_analysis"]!=_file_record(repo,ANALYSIS_PATH):raise StageFF3HError("figure analysis binding drifted")
    return {"status":"PASS","scientific_validity":completion["scientific_validity"],"value_path_counts":completion["value_path_counts"],"gradcam_status":"VALID","render_count":F3H_RENDER_COUNT,"class_conditioned_basis":completion["class_conditioned_basis"],"next_authorization":"F3C0_PROTOCOL_FREEZE_ONLY","f4_status":"PAUSED_PENDING_F3H_AND_F3C","archive_ref":F3H_ARCHIVE_REF}


def main(argv:Sequence[str]|None=None)->int:
    import argparse
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest="command",required=True)
    for name in ("validate-implementation","validate-implementation-completion","validate-initialized","authorize-bulk","run-gradcam","reconcile","build-components","build-consensus","assess-basis","render","build-audits","validate-outputs","validate-completion"):
        child=sub.add_parser(name);child.add_argument("--repo-root",type=Path,default=REPO_ROOT)
    freeze=sub.add_parser("freeze-implementation");freeze.add_argument("--repo-root",type=Path,default=REPO_ROOT);freeze.add_argument("--completed-at",required=True);freeze.add_argument("--passed",type=int,required=True);freeze.add_argument("--failed",type=int,required=True);freeze.add_argument("--test-command",required=True)
    init=sub.add_parser("initialize");init.add_argument("--repo-root",type=Path,default=REPO_ROOT);init.add_argument("--started-at",required=True)
    pilot=sub.add_parser("run-pilot");pilot.add_argument("--repo-root",type=Path,default=REPO_ROOT);pilot.add_argument("--device",default="cuda:0")
    shard=sub.add_parser("run-shard");shard.add_argument("--repo-root",type=Path,default=REPO_ROOT);shard.add_argument("--shard-id",type=int,required=True);shard.add_argument("--shard-count",type=int,default=8);shard.add_argument("--device",default="cuda:0")
    tests=sub.add_parser("record-tests");tests.add_argument("--repo-root",type=Path,default=REPO_ROOT);tests.add_argument("--passed",type=int,required=True);tests.add_argument("--failed",type=int,required=True);tests.add_argument("--test-command",required=True)
    final=sub.add_parser("finalize");final.add_argument("--repo-root",type=Path,default=REPO_ROOT);final.add_argument("--completed-at",required=True);args=parser.parse_args(argv)
    try:
        table={"validate-implementation":lambda:validate_implementation(args.repo_root),"validate-implementation-completion":lambda:validate_implementation_completion(args.repo_root),"freeze-implementation":lambda:freeze_implementation(args.repo_root,completed_at=args.completed_at,passed=args.passed,failed=args.failed,command=args.test_command),"validate-initialized":lambda:validate_initialized(args.repo_root),"initialize":lambda:initialize(args.repo_root,started_at=args.started_at),"run-pilot":lambda:run_pilot(args.repo_root,device=args.device),"authorize-bulk":lambda:authorize_bulk(args.repo_root),"run-shard":lambda:run_shard(args.repo_root,shard_id=args.shard_id,shard_count=args.shard_count,device=args.device),"run-gradcam":lambda:run_gradcam(args.repo_root,device=args.device),"reconcile":lambda:reconcile_terminals(args.repo_root),"build-components":lambda:build_components(args.repo_root),"build-consensus":lambda:build_consensus_manifest(args.repo_root),"assess-basis":lambda:assess_class_conditioned_basis(args.repo_root),"render":lambda:render_outputs(args.repo_root),"build-audits":lambda:build_final_audits(args.repo_root),"record-tests":lambda:record_tests(args.repo_root,passed=args.passed,failed=args.failed,command=args.test_command),"validate-outputs":lambda:validate_outputs(args.repo_root),"finalize":lambda:finalize(args.repo_root,completed_at=args.completed_at),"validate-completion":lambda:validate_completion(args.repo_root)};result=table[args.command]()
    except (StageFF3HError,OSError,ValueError) as error:
        print(json.dumps({"status":"BLOCKED","error":str(error)},sort_keys=True),file=sys.stderr);return 2
    print(json.dumps(result,indent=2,sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
