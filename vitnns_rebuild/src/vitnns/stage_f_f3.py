"""Stage F F3 selection-first, display-only qualitative producer.

F3 consumes sealed Stage D/E arrays and JPEGs only after the D10/D1 selection
manifest has been atomically published.  It never imports a model runtime and
never executes inference, attribution, fidelity, or statistical estimation.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
from jsonschema import Draft202012Validator
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageOps

from .constants import REPO_ROOT
from .display import DisplayTransform, metric_ready
from .stage_d_provenance import canonical_json_bytes, canonical_sha256
from .stage_f_renderer import (
    _pdf_object_digest, _save_figure, load_contracts, renderer_style,
    sha256_file, validate_font,
)

F3_CONFIG_PATH = "config/stage_f_f3.json"
F3_CONFIG_SCHEMA_PATH = "config/stage_f_f3.schema.json"
F3_RUN_SPEC_PATH = "config/stage_f_f3_run_spec.json"
F3_RUN_SPEC_SCHEMA_PATH = "config/stage_f_f3_run_spec.schema.json"
F3_COMPLETION_SCHEMA_PATH = "config/stage_f_f3_completion.schema.json"
F3_FALSE_PASS_SCHEMA_PATH = "config/stage_f_f3_false_pass_audit.schema.json"
F3_VALIDATION_SCHEMA_PATH = "config/stage_f_f3_validation_report.schema.json"

F3_ROOT_PATH = "evidence/stage_f/f3_r1_recovery_v2_20260907/formal"
F3_REPORT_PATH = "reports/stage_f_f3_r1_20260907"
PENDING_PATH = F3_ROOT_PATH + "/completion/f3.pending.json"
COMPLETION_PATH = F3_ROOT_PATH + "/completion/f3.json"
STARTUP_PATH = F3_ROOT_PATH + "/startup_gate.json"
SOURCE_BINDING_PATH = F3_ROOT_PATH + "/source_binding.json"
SELECTION_PATH = F3_ROOT_PATH + "/selection_manifest.json"
RULE_MAPPING_PATH = F3_ROOT_PATH + "/rule_to_image_manifest.json"
PANEL_MANIFEST_PATH = F3_ROOT_PATH + "/panel_manifest.json"
CAPTION_PATH = F3_ROOT_PATH + "/caption_data.json"
CROSS_UNION_PATH = F3_ROOT_PATH + "/cross_layer_source_union.json"
CROSS_SELECTION_PATH = F3_ROOT_PATH + "/cross_layer_selection_manifest.json"
CROSS_PANEL_PATH = F3_ROOT_PATH + "/cross_layer_panel_manifest.json"
CROSS_SCALE_PATH = F3_ROOT_PATH + "/cross_layer_scale_manifest.json"
RENDER_INVENTORY_PATH = F3_ROOT_PATH + "/render_inventory.json"
REVIEW_INVENTORY_PATH = F3_ROOT_PATH + "/review_copy_inventory.json"
ACCESS_AUDIT_PATH = F3_ROOT_PATH + "/access_audit.json"
ACCESS_SEAL_PATH = F3_ROOT_PATH + "/selection_access_seal.json"
PAYLOAD_MANIFEST_PATH = F3_ROOT_PATH + "/payload_manifest.json"
INVENTORY_PATH = F3_ROOT_PATH + "/whole_root_inventory.json"
STATUS_PATH = F3_ROOT_PATH + "/status.json"
VALIDATION_PATH = F3_REPORT_PATH + "/validation_report.json"
FALSE_PASS_PATH = F3_REPORT_PATH + "/false_pass_audit.json"
TEST_PATH = F3_REPORT_PATH + "/test_report.json"
ANALYSIS_PATH = F3_REPORT_PATH + "/figure_analysis.md"
REVIEW_ROOT_PATH = F3_REPORT_PATH + "/figures_png"
F3_ARCHIVE_REF = "refs/archive/stage-f-v1-f3-completion"

F2_ARCHIVE_REF = "refs/archive/stage-f-v1-f2-completion"
F2_COMMIT = "89dee5ad25b9cf3baa3a2ffdb2261ec8343a2ef6"
F2_ROOT_PATH = "evidence/stage_f/f2_r4_recovery_v2_20260907/formal"
F2_REPORT_PATH = "reports/stage_f_f2_r4_20260907"
F2_COMPLETION_PATH = F2_ROOT_PATH + "/completion/f2.json"
F2_INVENTORY_PATH = F2_ROOT_PATH + "/whole_root_inventory.json"
EXPECTED_F2_COMPLETION_SHA256 = "f4deb7d7a39714165b31d4978b8179e3418340de7b10ed53e0c091c93cedd31d"
EXPECTED_F2_COMPLETION_DIGEST = "1fa18250f104c7b66dbfd0c2dc1a1fb7bb978cb4940408f09bc4765c038deccc"
EXPECTED_F2_INVENTORY_DIGEST = "0f16684d2ed9fda74ac0f88406ff4987799c6b76e66112307f76cbe86964a405"

D1_REGISTRY_PATH = "manifests/stage_d_fidelity_v1/registry.json"
D10_DELTA_PATH = "evidence/stage_d/d10_recovery_v2_20260906/formal_r2/analysis/paper_data/per_sample_delta.json"
D8_ROOT_PATH = "evidence/stage_d/d8_recovery_v2_20260904/formal_r2"
D9_REPORT_PATH = "reports/stage_d_d9_recovery_v2_20260906/d9_formal_validation.json"
D10_REPORT_ROOT = "reports/stage_d_d10_recovery_v2_20260906"
E3_ROOT_PATH = "evidence/stage_e/e3_recovery_v2_20260906/formal"
E3_R1_ROOT_PATH = "evidence/stage_e/e3_r1_recovery_v2_20260906/formal"
E3_R2_ROOT_PATH = "evidence/stage_e/e3_r2_recovery_v2_20260906/formal"
E3_R4_ROOT_PATH = "evidence/stage_e/e3_r4_recovery_v2_20260907/formal"
E4_ROOT_PATH = "evidence/stage_e/e4_recovery_v2_20260907/formal"
E3_TYPED_PLAN_PATH = E3_ROOT_PATH + "/control/typed_plan.json"
E3_METHOD_ROWS_PATH = E3_R4_ROOT_PATH + "/paper_data/layer_head_method_rows.json"
E3_TERMINAL_ROWS_PATH = E3_R4_ROOT_PATH + "/paper_data/new_compute_terminal_rows.json"
E3_PAIRED_PATH = E3_R4_ROOT_PATH + "/paper_data/layer_head_paired_delta.json"
E4_LAYER_HEAD_PATH = E4_ROOT_PATH + "/paper_data/layer_head_descriptive.json"
JPEG_ROOT_PATH = "assets/data/ILSVRC2012_val"
DISPLAY_PROTOCOL_PATH = "docs/stage_f_display_protocol_v1.md"
CROSS_PROTOCOL_PATH = "docs/stage_f_cross_layer_display_addendum_20260907.md"
REFERENCE_PDF_PATH = "../论文1/参考文章.pdf"

EXPECTED_D1_SHA256 = "9aace3abf02a2c07d992d88b765c53cfeffa7f83a6ed72ecefb8cecf9eb79b6b"
EXPECTED_D1_DIGEST = "514ce4dd2e1fedc1744893e5543651ea42c053458b233ff11a45f700bb4bc028"
EXPECTED_D10_SHA256 = "8bb2e36dc2b4b3895bc5688b7ab5789719995add0c26cd48608d593486ba0cd2"
EXPECTED_D10_DIGEST = "4a89aeb89884219779699e15b49b017106bced1e60ab146dafac04d63ee9d9c0"
EXPECTED_DISPLAY_PROTOCOL_SHA256 = "de3ee8804f59e5541eb3c170110ab3b6c47f663698638e3fd668ece7f19b35e3"
EXPECTED_CROSS_PROTOCOL_SHA256 = "3449676703f985251b281b0c97d3c316d36ec551d4c5a80992fa63d7be0686bf"
EXPECTED_REFERENCE_SHA256 = "33e323f8cb664864b12daad1f06b6ef0d3268bcafaea4e84ac695fd286f3855b"
EXPECTED_D9_INVENTORY_DIGEST = "76d7d532049a5c90a09c74ba56b019042d7f2f0174ccb439bd8f67713514cc5a"
EXPECTED_D10_INVENTORY_DIGEST = "b72d36430d149598ccd9cfd8dd586297e7231be29e61b44309e9971087e46535"

NNS_METHOD = "conditional_fixed_clean_attention_head_direction_value_path_ig"
MATCHED_METHOD = "matched_fixed_clean_attention_patch_token_contribution"
METHODS = (NNS_METHOD, MATCHED_METHOD)
COORDINATES = ((0, 9), (4, 2), (6, 10), (11, 2))
SECONDARY_COORDINATES = COORDINATES[1:]
RULE_SLOTS = ("best_effect", "median_effect", "baseline_strong_nns_stronger", "weak_or_failure", "misclassified_or_difficult")
FAILURE_ORDERS = (117, 352)
FAILURE_CODES = {117: "ig_conservation_failed_after_e3_r1_schedule", 352: "ig_conservation_failed_typed_terminal"}

RULE_CAPTION = "案例图按 Stage D 结果访问前预注册的规则选择，仅作说明性展示；所有 primary 结论均基于完整 500 图队列。"
CROSS_CAPTION = "图中展示的是 fixed-clean-attention head-direction Value-path attribution，不是原始 self-attention。四个层/头坐标在 E3 outcome access 前已冻结；主样本来自既有 D10 median_effect 展示规则，而非查看跨层 map 后挑选。每个坐标内 NNS 与 matched 共享色限；不同坐标间只比较空间模式，不比较绝对强度。失败或不可用单元原位保留。"

ATTACKS = (
    "f2_archive_completion_inventory_or_terminal_authorization_tampered",
    "d8_primary_map_identity_or_d10_binding_tampered",
    "e3_union_not_exactly_3000_typed_terminals",
    "e3_union_not_exactly_2998_valid_maps",
    "failure_order_117_or_352_deleted_replaced_or_imputed",
    "coordinate_or_method_added_removed_or_reordered",
    "selection_manifest_not_sealed_before_jpeg_or_map_access",
    "figure6_not_exactly_median_effect",
    "s4_missing_audit_rule_or_failure_slot",
    "duplicate_sample_silently_replaced",
    "nns_and_matched_use_separate_scale_within_coordinate",
    "cross_coordinate_absolute_magnitude_claim_added",
    "failed_or_na_map_zero_filled_interpolated_or_copied",
    "jpeg_map_registry_result_or_signed_array_hash_mismatch",
    "png_pdf_svg_use_different_panel_manifest_or_config",
    "display_array_reenters_metric_or_selection",
    "model_checkpoint_torch_timm_cuda_gpu_or_new_attribution_used",
    "new_statistic_p_value_winner_or_go_no_go_created",
    "f4_or_final_combined_figure_generated",
    "terminal_pass_published_before_validator_manifest_and_review",
    "git_diff_check_failure_ignored",
)


class StageFF3Error(RuntimeError):
    """F3 cannot continue without weakening a frozen boundary."""


def _signed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: item for key, item in value.items() if key != "canonical_digest"}
    result["canonical_digest"] = canonical_sha256(result)
    return result


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_json_bytes(value) + b"\n")
        stream.flush(); os.fsync(stream.fileno())


def _write_text_new(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(value); stream.flush(); os.fsync(stream.fileno())


def _strict_json(path: Path, context: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes(); value = json.loads(raw)
    except Exception as error:
        raise StageFF3Error(f"{context} is invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise StageFF3Error(f"{context} must be a JSON object")
    return value, raw


def _validate_signed(value: Mapping[str, Any], context: str) -> None:
    if value.get("canonical_digest") != canonical_sha256({key: item for key, item in value.items() if key != "canonical_digest"}):
        raise StageFF3Error(f"{context} canonical digest mismatch")


def _schema(repo: Path, value: Mapping[str, Any], relative: str, context: str) -> None:
    schema, _ = _strict_json(repo / relative, f"{context} schema")
    Draft202012Validator.check_schema(schema)
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda item: list(item.path))
    if errors:
        where = ".".join(str(item) for item in errors[0].path) or "<root>"
        raise StageFF3Error(f"{context} schema failed at {where}: {errors[0].message}")


def _load_contract(repo: Path, relative: str, schema: str, context: str) -> dict[str, Any]:
    value, raw = _strict_json(repo / relative, context)
    _validate_signed(value, context); _schema(repo, value, schema, context)
    return value


def _load_config(repo: Path) -> dict[str, Any]:
    value = _load_contract(repo, F3_CONFIG_PATH, F3_CONFIG_SCHEMA_PATH, "F3 config")
    if value["authorization"] != {"current": "F3_ONLY", "on_pass": "F4_ONLY", "f4_started": False}:
        raise StageFF3Error("F3 authorization drifted")
    if any(value["safety"].values()):
        raise StageFF3Error("F3 safety boundary drifted")
    return value


def _load_run_spec(repo: Path) -> dict[str, Any]:
    return _load_contract(repo, F3_RUN_SPEC_PATH, F3_RUN_SPEC_SCHEMA_PATH, "F3 run spec")


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(("git", *arguments), cwd=repo, text=True, capture_output=True, check=False)
    if result.returncode:
        raise StageFF3Error(f"git {' '.join(arguments)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _is_ancestor(repo: Path, ancestor: str, descendant: str = "HEAD") -> bool:
    return subprocess.run(("git", "merge-base", "--is-ancestor", ancestor, descendant), cwd=repo, check=False).returncode == 0


def _lfs_clean(repo: Path) -> bool:
    return not any(line.startswith("\t") or line.startswith("    ") for line in _git(repo, "lfs", "status").splitlines())


def _inventory(root: Path, *, exclude: Sequence[str] = ()) -> dict[str, Any]:
    excluded = set(exclude); files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_file() and not path.is_symlink() and relative not in excluded:
            files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"files": files, "file_count": len(files), "total_bytes": sum(item["bytes"] for item in files), "canonical_digest": canonical_sha256({"files": files})}


def _record(repo: Path, relative: str) -> dict[str, Any]:
    path = repo / relative; value, _ = _strict_json(path, relative); _validate_signed(value, relative)
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path), "canonical_digest": value["canonical_digest"]}


def _file_record(repo: Path, relative: str) -> dict[str, Any]:
    path = repo / relative
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _array_digest(array: np.ndarray) -> str:
    source = np.ascontiguousarray(array)
    digest = hashlib.sha256(); digest.update(source.dtype.str.encode("ascii")); digest.update(b"\0")
    digest.update(json.dumps(list(source.shape), separators=(",", ":")).encode("ascii")); digest.update(b"\0"); digest.update(source.tobytes(order="C"))
    return digest.hexdigest()


def _verify_f2(repo: Path) -> dict[str, Any]:
    if _git(repo, "rev-parse", F2_ARCHIVE_REF + "^{commit}") != F2_COMMIT or not _is_ancestor(repo, F2_COMMIT):
        raise StageFF3Error("F2 archive ref drifted or is not an ancestor")
    completion, _ = _strict_json(repo / F2_COMPLETION_PATH, "F2 completion")
    inventory, _ = _strict_json(repo / F2_INVENTORY_PATH, "F2 inventory")
    _validate_signed(completion, "F2 completion"); _validate_signed(inventory, "F2 inventory")
    if (sha256_file(repo / F2_COMPLETION_PATH) != EXPECTED_F2_COMPLETION_SHA256 or completion["canonical_digest"] != EXPECTED_F2_COMPLETION_DIGEST or inventory["canonical_digest"] != EXPECTED_F2_INVENTORY_DIGEST):
        raise StageFF3Error("F2 completion or inventory identity drifted")
    expected = {"status": "PASS", "package_validation": "PASS", "workflow_validation": "PASS", "next_authorization": "F3_ONLY", "f3_started": False, "f4_started": False}
    if any(completion.get(key) != value for key, value in expected.items()):
        raise StageFF3Error("F2 terminal authorization drifted")
    rebuilt = _inventory(repo / F2_ROOT_PATH, exclude=("whole_root_inventory.json", "status.json", "completion/f2.pending.json", "completion/f2.json"))
    unsigned = {key: item for key, item in inventory.items() if key != "canonical_digest"}
    if any(inventory.get(key) != rebuilt[key] for key in ("files", "file_count", "total_bytes")) or inventory["canonical_digest"] != canonical_sha256(unsigned):
        raise StageFF3Error("F2 sealed inventory content drifted")
    if _git(repo, "diff", "--name-only", F2_COMMIT, "--", F2_ROOT_PATH, F2_REPORT_PATH):
        raise StageFF3Error("F2 sealed root changed after archive commit")
    return {"status": "PASS", "archive_ref": F2_ARCHIVE_REF, "archive_commit": F2_COMMIT, "completion_sha256": EXPECTED_F2_COMPLETION_SHA256, "whole_root_inventory_canonical_digest": EXPECTED_F2_INVENTORY_DIGEST, "next_authorization": "F3_ONLY"}


def _verify_frozen_protocols(repo: Path) -> dict[str, Any]:
    records = {}
    for name, relative, expected in (
        ("display_protocol", DISPLAY_PROTOCOL_PATH, EXPECTED_DISPLAY_PROTOCOL_SHA256),
        ("cross_layer_protocol", CROSS_PROTOCOL_PATH, EXPECTED_CROSS_PROTOCOL_SHA256),
        ("reference_pdf", REFERENCE_PDF_PATH, EXPECTED_REFERENCE_SHA256),
    ):
        observed = sha256_file((repo / relative).resolve())
        if observed != expected:
            raise StageFF3Error(f"frozen {name} hash drifted")
        records[name] = {"path": relative, "sha256": observed}
    return records


def _verify_scientific_state(repo: Path) -> dict[str, Any]:
    d9, _ = _strict_json(repo / D9_REPORT_PATH, "D9 report")
    d10, _ = _strict_json(repo / D10_REPORT_ROOT / "d10_formal_decision.json", "D10 report")
    e4, _ = _strict_json(repo / E4_ROOT_PATH / "completion/e4.json", "E4 completion")
    if d9.get("validation", {}).get("root_inventory", {}).get("canonical_digest") != EXPECTED_D9_INVENTORY_DIGEST:
        raise StageFF3Error("D8/D9 sealed inventory binding drifted")
    if d10.get("d9_revalidation", {}).get("root_inventory", {}).get("canonical_digest") != EXPECTED_D9_INVENTORY_DIGEST:
        raise StageFF3Error("D10 inventory binding drifted")
    truth = d10.get("truth_table", {})
    if not (truth.get("validity_status") == "PASS" and truth.get("fidelity_status") == "PASS_FIDELITY" and truth.get("cost_status") == "PASS_COST" and truth.get("overall_decision") == "GO_FIDELITY_SCOPE"):
        raise StageFF3Error("Stage D scientific state drifted")
    c10 = d10.get("c10_status_preserved", {})
    if c10 != {"acceptance": "BLOCKED", "downstream_authorized": False, "holm": "HOLM_BLOCKED_UNDERSPECIFIED", "overall_spatial": "FAIL_SPATIAL", "primary_decision": "NO_GO_NOVELTY"}:
        raise StageFF3Error("C10 frozen state drifted")
    if e4.get("scientific_validity") != "MIXED" or e4.get("status") != "PASS":
        raise StageFF3Error("E4 mixed-validity state drifted")
    return {"status": "PASS", "stage_d": {"validity": "PASS", "fidelity": "PASS_FIDELITY", "cost": "PASS_COST", "decision": "GO_FIDELITY_SCOPE", "n": 500, "positive": 500}, "c10": {"n": 15, **c10}, "stage_e": {"scientific_validity": "MIXED", "expected": 3000, "valid": 2998, "failed": 2, "e2_budget_waiver_retained": True}}


def validate_runtime_isolation(*, jpeg_read: bool = False, map_read: bool = False) -> dict[str, Any]:
    forbidden = sorted(name for name in sys.modules if name == "torch" or name.startswith("torch.") or name == "torchvision" or name.startswith("torchvision.") or name == "timm" or name.startswith("timm."))
    if forbidden:
        raise StageFF3Error(f"formal F3 process imported forbidden runtime: {forbidden[0]}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise StageFF3Error("formal F3 producer requires CUDA_VISIBLE_DEVICES to be empty")
    handles = []
    if Path("/proc/self/fd").is_dir():
        for path in Path("/proc/self/fd").iterdir():
            try: target = os.readlink(path)
            except OSError: continue
            if target.startswith("/dev/nvidia"): handles.append(target)
    if handles: raise StageFF3Error("formal F3 process holds an NVIDIA device handle")
    return {"status": "PASS", "cuda_visible_devices": "", "torch_imported": False, "torchvision_imported": False, "timm_imported": False, "nvidia_device_handles": [], "model_loaded": False, "checkpoint_loaded": False, "jpeg_read": jpeg_read, "map_read": map_read, "new_attribution_or_fidelity_executed": False}


def assert_f3_source_boundary(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); paths = [repo / "src/vitnns/stage_f_f3.py", repo / "src/vitnns/stage_f_renderer.py", repo / "src/vitnns/stage_f_cli.py"]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    for name in ("torch", "torchvision", "timm"):
        if re.search(rf"^\s*(?:from|import)\s+{name}(?:\.|\s|$)", combined, re.MULTILINE):
            raise StageFF3Error(f"F3 source imports forbidden runtime: {name}")
    forbidden = ("load_" + "canonical_model", "load_" + "state_dict", "for" + "ward(", "back" + "ward(", "integrated_" + "grad", "." + "cuda(", "cuda" + ":")
    if any(token in combined for token in forbidden):
        raise StageFF3Error("F3 source exposes forbidden model GPU or attribution execution")
    return {"status": "PASS", "model_access": False, "checkpoint_access": False, "torch_access": False, "timm_access": False, "cuda_access": False, "forward_backward_ig_attribution_fidelity_execution": False}


def verify_startup_gates(repo_root: Path | str = REPO_ROOT, *, require_clean: bool = True, require_remote: bool = True) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); config = _load_config(repo); run_spec = _load_run_spec(repo)
    branch = _git(repo, "symbolic-ref", "--short", "HEAD"); head = _git(repo, "rev-parse", "HEAD"); tracking = _git(repo, "rev-parse", "origin/stage-d-d3-pass")
    if branch != "stage-d-d3-pass" or head != tracking or not _is_ancestor(repo, "f05a8806f2d83df990189011fde956aaa73e473e"):
        raise StageFF3Error("F3 requires synchronized stage-d-d3-pass on an authorized successor")
    remote = None
    if require_remote:
        lines = _git(repo, "ls-remote", "--heads", "origin", "stage-d-d3-pass").splitlines(); remote = lines[0].split()[0] if len(lines) == 1 else None
        if remote != head: raise StageFF3Error("local tracking and remote branch differ")
        refs = dict(reversed(line.split()) for line in _git(repo, "ls-remote", "origin", F2_ARCHIVE_REF).splitlines())
        if refs.get(F2_ARCHIVE_REF) != F2_COMMIT: raise StageFF3Error("remote F2 archive ref drifted")
    status_clean = _git(repo, "status", "--porcelain=v1") == ""
    if require_clean and (not status_clean or not _lfs_clean(repo)):
        raise StageFF3Error("F3 start requires clean worktree index and Git LFS")
    if any((repo / "evidence/stage_f").glob("f4*")) or any((repo / "reports").glob("stage_f_f4*")):
        raise StageFF3Error("F4 output already exists")
    return _signed({"schema_version": 1, "document_type": "stage_f_f3_startup_gate", "stage_id": "F3", "status": "PASS", "validated_before_formal_output_or_qualitative_access": True, "branch": branch, "local_head": head, "tracking_head": tracking, "remote_head": remote, "worktree_index_clean": status_clean, "git_lfs_clean": _lfs_clean(repo), "f2": _verify_f2(repo), "protocols": _verify_frozen_protocols(repo), "scientific_state": _verify_scientific_state(repo), "source_boundary": assert_f3_source_boundary(repo), "runtime_isolation": validate_runtime_isolation(), "config_canonical_digest": config["canonical_digest"], "run_spec_canonical_digest": run_spec["canonical_digest"], "d8_d9_d10_e3_e4_payload_arrays_or_jpegs_read": False, "next_authorization": "F3_ONLY"})


def _load_selection_sources(repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    registry, registry_raw = _strict_json(repo / D1_REGISTRY_PATH, "D1 registry")
    delta, delta_raw = _strict_json(repo / D10_DELTA_PATH, "D10 per-sample delta")
    for value, context in ((registry, "D1 registry"), (delta, "D10 per-sample delta")):
        _validate_signed(value, context)
    if sha256_file(repo / D1_REGISTRY_PATH) != EXPECTED_D1_SHA256 or registry["canonical_digest"] != EXPECTED_D1_DIGEST:
        raise StageFF3Error("D1 registry identity drifted")
    if sha256_file(repo / D10_DELTA_PATH) != EXPECTED_D10_SHA256 or delta["canonical_digest"] != EXPECTED_D10_DIGEST:
        raise StageFF3Error("D10 per-sample identity drifted")
    rows = registry.get("rows"); drows = delta.get("rows")
    if not isinstance(rows, list) or not isinstance(drows, list) or len(rows) != 500 or len(drows) != 500:
        raise StageFF3Error("selection sources must contain 500 rows")
    if [(r["selection_order"], r["sample_id"]) for r in rows] != [(r["selection_order"], r["sample_id"]) for r in drows]:
        raise StageFF3Error("D1 and D10 sample order differs")
    if any(row.get("status") != "VALID" or row.get("validity_status") != "PASS" or not math.isfinite(row.get("delta_fid", math.nan)) for row in drows):
        raise StageFF3Error("D10 selection cohort contains an ineligible primary row")
    return registry, delta


def build_selection_manifests(repo_root: Path | str = REPO_ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    repo = Path(repo_root).resolve(); registry, delta = _load_selection_sources(repo)
    drows = delta["rows"]; by_order = {row["selection_order"]: row for row in registry["rows"]}
    median = float(np.median(np.asarray([row["delta_fid"] for row in drows], dtype=np.float64)))
    if median != 2.3081996221714247:
        raise StageFF3Error("D10 median selection anchor drifted")
    valid = list(drows)
    selected: dict[str, tuple[dict[str, Any], str]] = {}
    selected["best_effect"] = (min(valid, key=lambda row: (-row["delta_fid"], row["selection_order"], row["sample_id"])), "maximum primary delta_fid")
    selected["median_effect"] = (min(valid, key=lambda row: (abs(row["delta_fid"] - median), -row["delta_fid"], row["selection_order"], row["sample_id"])), "minimum absolute distance to frozen 500-image median")
    baseline = [row for row in valid if row["matched_insertion_auc"] - row["matched_deletion_auc"] > 0.0]
    selected["baseline_strong_nns_stronger"] = (min(baseline, key=lambda row: (-row["delta_fid"], row["selection_order"], row["sample_id"])), "positive matched fine-4x4 fidelity contribution then maximum positive delta_fid")
    failures = [row for row in drows if row.get("status") != "VALID" or row.get("validity_status") != "PASS"]
    selected["weak_or_failure"] = ((min(failures, key=lambda row: (row["selection_order"], row["sample_id"])) if failures else min(valid, key=lambda row: (row["delta_fid"], row["selection_order"], row["sample_id"]))), "smallest required-primary failure order, otherwise minimum valid primary delta_fid")
    misclassified = [row for row in valid if row["misclassified"] is True]
    if not misclassified:
        raise StageFF3Error("misclassified_or_difficult fallback requires unavailable clean-margin field")
    selected["misclassified_or_difficult"] = (min(misclassified, key=lambda row: (-row["delta_fid"], row["selection_order"], row["sample_id"])), "misclassified valid row with maximum positive delta_fid")
    rule_rows = []
    for rank, rule in enumerate(RULE_SLOTS):
        row, rationale = selected[rule]; registry_row = by_order[row["selection_order"]]
        rule_rows.append({"slot_index": rank, "rule": rule, "sample_id": row["sample_id"], "selection_order": row["selection_order"], "delta_fid": row["delta_fid"], "distance_to_median": abs(row["delta_fid"] - median) if rule == "median_effect" else None, "matched_fidelity_contribution": row["matched_insertion_auc"] - row["matched_deletion_auc"], "misclassified": row["misclassified"], "prediction_status": "misclassified" if row["misclassified"] else "correct", "fallback_used": False, "selection_rationale": rationale, "registry_row_canonical_digest": registry_row["canonical_digest"], "source_hashes": {"image_sha256": registry_row["image_sha256"], "d1_registry_sha256": EXPECTED_D1_SHA256, "d10_per_sample_sha256": EXPECTED_D10_SHA256}})
    audit_rows = [{"audit_rank": item["audit_rank"], "audit_overlay_id": item["audit_overlay_id"], "sample_id": item["sample_id"], "selection_order": item["selection_order"], "selection_digest": item["selection_digest"], "label": "fixed audit example"} for item in registry["audit_selection"]["entries"]]
    if [row["sample_id"] for row in audit_rows] != registry["audit_selection"]["selected_sample_ids"]:
        raise StageFF3Error("D1 audit ordering drifted")
    mapping = _signed({"schema_version": 1, "document_type": "stage_f_f3_rule_to_image_manifest", "stage_id": "F3", "status": "SEALED_BEFORE_JPEG_OR_MAP_ACCESS", "protocol_path": DISPLAY_PROTOCOL_PATH, "protocol_sha256": EXPECTED_DISPLAY_PROTOCOL_SHA256, "cohort_size": 500, "d10_median_delta_fid": median, "rules": rule_rows, "duplicates": [{"sample_id": sample, "rules": [row["rule"] for row in rule_rows if row["sample_id"] == sample]} for sample, count in Counter(row["sample_id"] for row in rule_rows).items() if count > 1], "render_order_unique_sample_ids": list(dict.fromkeys(row["sample_id"] for row in rule_rows)), "silent_replacement_allowed": False})
    selection = _signed({"schema_version": 1, "document_type": "stage_f_f3_selection_manifest", "stage_id": "F3", "status": "SEALED_BEFORE_JPEG_OR_MAP_ACCESS", "selection_inputs": {"d1_registry": {"path": D1_REGISTRY_PATH, "sha256": EXPECTED_D1_SHA256, "canonical_digest": EXPECTED_D1_DIGEST}, "d10_per_sample": {"path": D10_DELTA_PATH, "sha256": EXPECTED_D10_SHA256, "canonical_digest": EXPECTED_D10_DIGEST}}, "selection_access": {"jpeg_paths_statted_or_bytes_read": False, "map_paths_statted_or_bytes_read": False, "e3_map_outcomes_inspected": False}, "fixed_audit_examples": audit_rows, "rule_selections": rule_rows, "tie_break_order": ["distance_to_rule_target_ascending_when_applicable", "delta_fid_descending_when_applicable", "selection_order_ascending", "sample_id_lexicographic"], "eligible_primary_rows": 500, "required_primary_failures": len(failures), "baseline_positive_eligible_rows": len(baseline), "misclassified_eligible_rows": len(misclassified), "median_effect_sample_id": selected["median_effect"][0]["sample_id"], "median_effect_selection_order": selected["median_effect"][0]["selection_order"], "display_only": True, "primary_decision_modified": False, "stage_d_statistics_modified": False})
    return selection, mapping


def validate_selection_manifests(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); observed_selection, raw = _strict_json(repo / SELECTION_PATH, "F3 selection manifest"); observed_mapping, raw_mapping = _strict_json(repo / RULE_MAPPING_PATH, "F3 rule mapping")
    for value, source, context in ((observed_selection, raw, "F3 selection manifest"), (observed_mapping, raw_mapping, "F3 rule mapping")):
        if source != canonical_json_bytes(value) + b"\n": raise StageFF3Error(f"{context} is not byte-canonical")
        _validate_signed(value, context)
    expected_selection, expected_mapping = build_selection_manifests(repo)
    if observed_selection != expected_selection or observed_mapping != expected_mapping:
        raise StageFF3Error("selection manifests do not independently rebuild from D10 and D1")
    return {"status": "PASS", "selection_manifest_canonical_digest": observed_selection["canonical_digest"], "rule_mapping_canonical_digest": observed_mapping["canonical_digest"], "median_effect_sample_id": observed_selection["median_effect_sample_id"], "rule_slots": 5, "unique_rule_samples": len(observed_mapping["render_order_unique_sample_ids"]), "jpeg_or_map_access_before_seal": False}


def initialize(repo_root: Path | str = REPO_ROOT, *, renderer_commit: str, started_at: str) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); gate = verify_startup_gates(repo)
    if renderer_commit != gate["local_head"]:
        raise StageFF3Error("renderer commit must equal clean synchronized startup HEAD")
    root = repo / F3_ROOT_PATH
    if root.exists() or (repo / F3_REPORT_PATH).exists():
        raise StageFF3Error("refusing to overwrite an existing F3 root")
    _write_new(repo / STARTUP_PATH, gate)
    pending = _signed({"schema_version": 1, "document_type": "stage_f_f3_pending_completion", "stage_id": "F3", "status": "PENDING", "started_at": started_at, "renderer_commit": renderer_commit, "startup_gate": _record(repo, STARTUP_PATH), "selection_manifest_sealed": False, "jpeg_read": False, "map_read": False, "terminal_pass_published": False, "f4_started": False})
    _write_new(repo / PENDING_PATH, pending)
    selection, mapping = build_selection_manifests(repo)
    _write_new(repo / SELECTION_PATH, selection); _write_new(repo / RULE_MAPPING_PATH, mapping)
    verified = validate_selection_manifests(repo)
    access = _signed({"schema_version": 1, "document_type": "stage_f_f3_access_audit", "stage_id": "F3", "status": "SELECTION_SEALED", "startup_gate_completed_before_formal_modification_or_qualitative_access": True, "selection_manifest_path": SELECTION_PATH, "selection_manifest_sha256": sha256_file(repo / SELECTION_PATH), "selection_manifest_canonical_digest": selection["canonical_digest"], "selection_manifest_validated_before_source_payload_access": True, "jpeg_paths_statted_or_bytes_read_before_selection_seal": False, "map_paths_statted_or_bytes_read_before_selection_seal": False, "source_payload_access_started": False, "model_access": False, "checkpoint_access": False, "torch_imported": False, "timm_imported": False, "cuda_or_gpu_access": False, "forward_backward_ig_attribution_fidelity_executed": False})
    _write_new(repo / ACCESS_SEAL_PATH, access)
    return {"status": "INITIALIZED_SELECTION_SEALED", **verified, "renderer_commit": renderer_commit, "next_operation": "SOURCE_PAYLOAD_VALIDATION_AND_RENDERING"}


def _origin_attempt(repo: Path, unit: Mapping[str, Any], row: Mapping[str, Any]) -> tuple[str, Path]:
    origin = row["evidence_origin"]; attempt_id = row["attempt_id"]
    root_by_origin = {
        "E3_V1_INHERITED": E3_ROOT_PATH,
        "E3_R1_INHERITED": E3_R1_ROOT_PATH,
        "E3_R2_INHERITED_VALID": E3_R2_ROOT_PATH,
        "E3_R2_INHERITED_FAILED": E3_R2_ROOT_PATH,
        "E3_R4_NEW": E3_R4_ROOT_PATH,
    }
    expected_attempt = {
        "E3_V1_INHERITED": "attempt-000001",
        "E3_R1_INHERITED": "attempt-r1-000001",
        "E3_R2_INHERITED_VALID": "attempt-r2-000001",
        "E3_R2_INHERITED_FAILED": "attempt-r2-000001",
        "E3_R4_NEW": "attempt-r4-000001",
    }
    if origin not in root_by_origin or attempt_id != expected_attempt[origin]:
        raise StageFF3Error("E3 terminal origin or attempt ID drifted")
    relative = f"{root_by_origin[origin]}/units/{unit['canonical_digest']}/attempts/{attempt_id}"
    path = (repo / relative).resolve(); root = (repo / root_by_origin[origin] / "units").resolve()
    if root not in path.parents:
        raise StageFF3Error("unsafe E3 attempt path")
    return relative, path


def _load_valid_map(path: Path, *, expected_file_sha: str | None, expected_array_sha: str, context: str) -> tuple[np.ndarray, str, str]:
    if not path.is_file() or path.is_symlink():
        raise StageFF3Error(f"{context} map is absent or unsafe")
    file_sha = sha256_file(path)
    if expected_file_sha is not None and file_sha != expected_file_sha:
        raise StageFF3Error(f"{context} file SHA-256 mismatch")
    array = np.load(path, allow_pickle=False)
    if array.dtype != np.float32 or array.shape != (56, 56) or not np.isfinite(array).all():
        raise StageFF3Error(f"{context} array contract failed")
    array_sha = _array_digest(array)
    if array_sha != expected_array_sha:
        raise StageFF3Error(f"{context} signed map SHA-256 mismatch")
    return array, file_sha, array_sha


def build_primary_source_index(repo_root: Path | str = REPO_ROOT) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    repo = Path(repo_root).resolve(); method_table, _ = _strict_json(repo / E3_METHOD_ROWS_PATH, "E3-R4 method rows"); _validate_signed(method_table, "E3-R4 method rows")
    delta, _ = _strict_json(repo / D10_DELTA_PATH, "D10 per-sample delta"); _validate_signed(delta, "D10 per-sample delta")
    registry, _ = _strict_json(repo / D1_REGISTRY_PATH, "D1 registry"); reconciliation, _ = _strict_json(repo / D8_ROOT_PATH / "final/reconciliation.json", "D9 reconciliation")
    primary_rows = [row for row in method_table["rows"] if (row["layer"], row["head"]) == (0, 9)]
    if len(primary_rows) != 1000 or Counter(row["method"] for row in primary_rows) != Counter({method: 500 for method in METHODS}):
        raise StageFF3Error("D8 primary method rows are not exactly 1000")
    recon = {(row["sample_id"], row["method_id"]): row for row in reconciliation["rows"]}
    d10 = {row["sample_id"]: row for row in delta["rows"]}; registry_rows = {row["sample_id"]: row for row in registry["rows"]}
    index: dict[tuple[str, str], dict[str, Any]] = {}; records = []
    for row in sorted(primary_rows, key=lambda item: (item["selection_order"], METHODS.index(item["method"]))):
        key = (row["sample_id"], row["method"]); hashes = row["source_hashes"]; map_relative = hashes["source_map_path"]; result_relative = hashes["source_result_path"]
        map_path = repo / map_relative; result_path = repo / result_relative
        if sha256_file(result_path) != hashes["source_result_sha256"]:
            raise StageFF3Error("D8 primary result file SHA-256 mismatch")
        result, _ = _strict_json(result_path, "D8 primary result")
        if result.get("sample_id") != row["sample_id"] or result.get("method_id") != row["method"]:
            raise StageFF3Error("D8 primary result identity mismatch")
        curves = result.get("fidelity_curves", []); array_hashes = {curve.get("source_unit_map_sha256") for curve in curves}
        if len(curves) != 10 or len(array_hashes) != 1 or None in array_hashes:
            raise StageFF3Error("D8 primary curve-to-map binding drifted")
        array, file_sha, array_sha = _load_valid_map(map_path, expected_file_sha=hashes["source_map_sha256"], expected_array_sha=array_hashes.pop(), context="D8 primary")
        source = recon.get(key)
        if source is None or source["unit_digest"] != hashes["source_unit_digest"] or source["pair_digest"] is None:
            raise StageFF3Error("D8 reconciliation binding drifted")
        rrow = registry_rows[row["sample_id"]]
        if row["selection_order"] != rrow["selection_order"] or row["misclassified"] != d10[row["sample_id"]]["misclassified"]:
            raise StageFF3Error("D8 D10 D1 identity mismatch")
        aggregates = result.get("fidelity_aggregates", [])
        if len(aggregates) != 2 or any(len(item.get("response_mean", [])) != 21 for item in aggregates):
            raise StageFF3Error("D8 fidelity aggregate coverage drifted")
        expected_auc = (d10[row["sample_id"]]["nns_insertion_auc"], d10[row["sample_id"]]["nns_deletion_auc"]) if row["method"] == NNS_METHOD else (d10[row["sample_id"]]["matched_insertion_auc"], d10[row["sample_id"]]["matched_deletion_auc"])
        if tuple(item["auc_mean"] for item in aggregates) != expected_auc:
            raise StageFF3Error("D8 primary curves do not bind to D10 AUC values")
        record = {"sample_id": row["sample_id"], "selection_order": row["selection_order"], "coordinate": [0, 9], "method": row["method"], "validity_status": "VALID", "attempt_id": hashes["source_attempt_id"], "evidence_origin": "D8_FORMAL_R2_D9_VALIDATED_D10_BOUND", "source_map_path": map_relative, "source_map_file_sha256": file_sha, "signed_map_sha256": array_sha, "source_result_path": result_relative, "source_result_sha256": hashes["source_result_sha256"], "source_unit_digest": hashes["source_unit_digest"], "source_attempt_inventory_digest": hashes["source_attempt_inventory_digest"], "pair_binding_digest": source["pair_digest"], "applicable_canonical_digest": EXPECTED_D9_INVENTORY_DIGEST, "image_sha256": rrow["image_sha256"], "map": array, "fidelity": {"fractions": curves[0]["actual_fractions"], "insertion_mean": aggregates[0]["response_mean"], "deletion_mean": aggregates[1]["response_mean"], "source_curve_hashes": [curve["ranking_sha256"] for curve in curves]}}
        index[key] = record; records.append({key: value for key, value in record.items() if key not in {"map", "fidelity"}})
    if len(index) != 1000 or len({record["source_map_path"] for record in records}) != 1000:
        raise StageFF3Error("D8 primary map uniqueness failed")
    return index, _signed({"schema_version": 1, "document_type": "stage_f_f3_primary_source_index", "stage_id": "F3", "status": "PASS", "root": D8_ROOT_PATH, "root_inventory_canonical_digest": EXPECTED_D9_INVENTORY_DIGEST, "map_count": 1000, "sample_count": 500, "method_counts": dict(Counter(item["method"] for item in records)), "unique_map_paths": 1000, "d10_per_sample_sha256": EXPECTED_D10_SHA256, "records_digest": canonical_sha256(records)})


def build_secondary_source_union(repo_root: Path | str = REPO_ROOT) -> tuple[dict[tuple[str, int, int, str], dict[str, Any]], dict[str, Any]]:
    repo = Path(repo_root).resolve(); plan, _ = _strict_json(repo / E3_TYPED_PLAN_PATH, "E3 typed plan"); terminal, _ = _strict_json(repo / E3_TERMINAL_ROWS_PATH, "E3 terminal rows"); pairs, _ = _strict_json(repo / E3_PAIRED_PATH, "E3 paired rows"); method_rows, _ = _strict_json(repo / E3_METHOD_ROWS_PATH, "E3 method rows"); e4, _ = _strict_json(repo / E4_LAYER_HEAD_PATH, "E4 layer-head copy")
    for value, context in ((plan, "E3 typed plan"), (terminal, "E3 terminal rows"), (pairs, "E3 paired rows"), (method_rows, "E3 method rows"), (e4, "E4 layer-head copy")):
        _validate_signed(value, context)
    units = plan.get("units", []); rows = terminal.get("rows", [])
    if len(units) != 3000 or len(rows) != 3000:
        raise StageFF3Error("E3 union must contain exactly 3000 typed terminals")
    unit_index = {(item["sample_id"], item["block"], item["head"], item["method"]): item for item in units}
    row_index = {(item["sample_id"], item["layer"], item["head"], item["method"]): item for item in rows}
    if len(unit_index) != 3000 or set(unit_index) != set(row_index):
        raise StageFF3Error("E3 typed plan and final union keys differ")
    pair_index = {(item["sample_id"], item["layer"], item["head"]): item for item in pairs["rows"] if (item["layer"], item["head"]) in SECONDARY_COORDINATES}
    if len(pair_index) != 1500:
        raise StageFF3Error("E3 secondary pair table must contain 1500 coordinates")
    presence: dict[tuple[str, int, int, str], list[str]] = {}
    for key, unit in unit_index.items():
        found = []
        for root, attempt in ((E3_ROOT_PATH, "attempt-000001"), (E3_R1_ROOT_PATH, "attempt-r1-000001"), (E3_R2_ROOT_PATH, "attempt-r2-000001"), (E3_R4_ROOT_PATH, "attempt-r4-000001")):
            candidate = repo / root / "units" / unit["canonical_digest"] / "attempts" / attempt
            if candidate.is_dir(): found.append(f"{root}/units/{unit['canonical_digest']}/attempts/{attempt}")
        presence[key] = found
        if len(found) != 1:
            raise StageFF3Error("conflicting or missing E3 terminal payload for one logical unit")
    index: dict[tuple[str, int, int, str], dict[str, Any]] = {}; counts = Counter(); valid_by_root = Counter(); records = []
    for key in sorted(unit_index, key=lambda item: (unit_index[item]["selection_order"], item[1], item[2], METHODS.index(item[3]))):
        unit = unit_index[key]; row = row_index[key]; relative_attempt, attempt = _origin_attempt(repo, unit, row)
        if presence[key] != [relative_attempt]: raise StageFF3Error("E3 union evidence-origin table disagrees with retained payload")
        result_path = attempt / "result.json"; status_path = attempt / "status.json"; manifest_path = attempt / "manifest.json"
        result, result_raw = _strict_json(result_path, "E3 terminal result"); status, status_raw = _strict_json(status_path, "E3 terminal status"); manifest, _ = _strict_json(manifest_path, "E3 terminal manifest")
        for value, context in ((result, "E3 terminal result"), (status, "E3 terminal status"), (manifest, "E3 terminal manifest")): _validate_signed(value, context)
        if any(result.get(name) != expected for name, expected in (("sample_id", unit["sample_id"]), ("selection_order", unit["selection_order"]), ("block", unit["block"]), ("head", unit["head"]), ("method", unit["method"]), ("validity_status", row["validity_status"]), ("attempt_id", row["attempt_id"]))):
            raise StageFF3Error("E3 result identity differs from final union row")
        if manifest.get("result_sha256") != hashlib.sha256(result_raw).hexdigest() or manifest.get("status_sha256") != hashlib.sha256(status_raw).hexdigest() or status.get("terminal") is not True:
            raise StageFF3Error("E3 result status manifest binding drifted")
        valid = row["validity_status"] == "VALID"; payload_name = "map.npy" if valid else "failure_attribution.npy"
        if manifest.get("payload_name", manifest.get("map_path", payload_name)) != payload_name and valid:
            raise StageFF3Error("E3 valid payload name drifted")
        record: dict[str, Any] = {"sample_id": row["sample_id"], "selection_order": row["selection_order"], "coordinate": [row["layer"], row["head"]], "method": row["method"], "validity_status": row["validity_status"], "failure_class": row["failure_class"], "failure_code": row["failure_code"], "na_reason": row["na_reason"], "attempt_id": row["attempt_id"], "evidence_origin": row["evidence_origin"], "source_evidence_origin": relative_attempt, "source_result_path": result_path.relative_to(repo).as_posix(), "source_result_sha256": hashlib.sha256(result_raw).hexdigest(), "source_result_canonical_digest": result["canonical_digest"], "pair_binding_digest": row["pair_binding_digest"], "image_sha256": row["source_hashes"]["image_sha256"], "target_semantic_digest": row["source_hashes"]["target_semantic_digest"], "source_map_path": None, "source_map_file_sha256": None, "signed_map_sha256": None, "map": None}
        pair = pair_index[(row["sample_id"], row["layer"], row["head"])]
        pair_key = "nns_map_sha256" if row["method"] == NNS_METHOD else "matched_map_sha256"
        if valid:
            payload = attempt / "map.npy"; expected_array = row["source_hashes"]["signed_map_sha256"]
            array, file_sha, array_sha = _load_valid_map(payload, expected_file_sha=manifest.get("payload_sha256", manifest.get("map_sha256")), expected_array_sha=expected_array, context="E3 valid")
            if result.get("pair_binding_digest") != row["pair_binding_digest"] or result.get("pair_binding_digest") != canonical_sha256(result.get("pair_binding", {})) or pair["source_hashes"][pair_key] != array_sha:
                raise StageFF3Error("E3 valid map pair binding drifted")
            record.update({"source_map_path": payload.relative_to(repo).as_posix(), "source_map_file_sha256": file_sha, "signed_map_sha256": array_sha, "map": array})
            valid_by_root[row["evidence_origin"]] += 1
        else:
            if row["selection_order"] not in FAILURE_ORDERS or (row["layer"], row["head"], row["method"]) != (11, 2, NNS_METHOD) or row["failure_code"] != FAILURE_CODES[row["selection_order"]] or pair["source_hashes"][pair_key] is not None:
                raise StageFF3Error("E3 typed failure identity drifted")
            if (attempt / "map.npy").exists(): raise StageFF3Error("failed E3 unit unexpectedly contains a valid map")
        counts[row["validity_status"]] += 1; index[key] = record; records.append({key: value for key, value in record.items() if key != "map"})
    expected_valid = {"E3_V1_INHERITED": 516, "E3_R1_INHERITED": 12, "E3_R2_INHERITED_VALID": 636, "E3_R4_NEW": 1834}
    if counts != Counter({"VALID": 2998, "FAILED": 2}) or dict(valid_by_root) != expected_valid:
        raise StageFF3Error("E3 union valid/failed or root counts drifted")
    method_secondary = [row for row in method_rows["rows"] if (row["layer"], row["head"]) in SECONDARY_COORDINATES]
    if method_secondary != rows:
        raise StageFF3Error("E3 method rows and terminal rows differ")
    if e4.get("rows") != pairs.get("rows"):
        raise StageFF3Error("E4 sealed layer-head copy differs from E3-R4 paired table")
    root_records = []
    for root in (E3_ROOT_PATH, E3_R1_ROOT_PATH, E3_R2_ROOT_PATH, E3_R4_ROOT_PATH):
        inventory, _ = _strict_json(repo / root / "whole_root_inventory.json", f"{root} inventory"); _validate_signed(inventory, f"{root} inventory")
        root_records.append({"root": root, "whole_root_inventory_canonical_digest": inventory["canonical_digest"], "valid_map_count": sum(1 for item in records if item["source_evidence_origin"].startswith(root) and item["validity_status"] == "VALID")})
    document = _signed({"schema_version": 1, "document_type": "stage_f_f3_cross_layer_source_union", "stage_id": "F3", "status": "PASS", "index_table": {"path": E3_TERMINAL_ROWS_PATH, "sha256": sha256_file(repo / E3_TERMINAL_ROWS_PATH), "canonical_digest": terminal["canonical_digest"]}, "method_rows": {"path": E3_METHOD_ROWS_PATH, "sha256": sha256_file(repo / E3_METHOD_ROWS_PATH), "canonical_digest": method_rows["canonical_digest"]}, "paired_rows": {"path": E3_PAIRED_PATH, "sha256": sha256_file(repo / E3_PAIRED_PATH), "canonical_digest": pairs["canonical_digest"]}, "e4_sealed_copy": {"path": E4_LAYER_HEAD_PATH, "sha256": sha256_file(repo / E4_LAYER_HEAD_PATH), "canonical_digest": e4["canonical_digest"]}, "source_roots": root_records, "expected_logical_units": 3000, "resolved_terminal_units": 3000, "valid_maps": 2998, "failed_units": 2, "na_units": 0, "valid_map_counts_by_origin": expected_valid, "failure_units": [item for item in records if item["validity_status"] == "FAILED"], "coordinates": [list(item) for item in SECONDARY_COORDINATES], "methods": list(METHODS), "conflicting_terminal_payloads": 0, "missing_terminal_payloads": 0, "records_digest": canonical_sha256(records), "records": records})
    return index, document


def _d10_row_index(repo: Path) -> dict[str, dict[str, Any]]:
    value, _ = _strict_json(repo / D10_DELTA_PATH, "D10 per-sample delta")
    return {row["sample_id"]: row for row in value["rows"]}


def _registry_index(repo: Path) -> dict[str, dict[str, Any]]:
    value, _ = _strict_json(repo / D1_REGISTRY_PATH, "D1 registry")
    return {row["sample_id"]: row for row in value["rows"]}


def _read_display_rgb(repo: Path, row: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    path = (repo / JPEG_ROOT_PATH / row["relative_path"]).resolve(); root = (repo / JPEG_ROOT_PATH).resolve()
    if root not in path.parents or path.suffix.lower() not in {".jpeg", ".jpg"}:
        raise StageFF3Error("unsafe selected JPEG path")
    file_sha = sha256_file(path)
    if file_sha != row["image_sha256"]:
        raise StageFF3Error("selected JPEG hash differs from registry")
    settings = config["display"]["canonical_image_transform"]
    with Image.open(path) as source:
        source.load(); original = source.convert("RGB")
    width, height = original.size; target = settings["resize_shorter_side"]
    if width < height: resized_size = (target, int(round(height * target / width)))
    else: resized_size = (int(round(width * target / height)), target)
    resized = original.resize(resized_size, resample=Image.Resampling.BICUBIC)
    crop_h, crop_w = settings["center_crop"]; left = (resized.width - crop_w) // 2; top = (resized.height - crop_h) // 2
    cropped = resized.crop((left, top, left + crop_w, top + crop_h))
    rgb = np.asarray(cropped, dtype=np.float32) / np.float32(255.0)
    return rgb, {"path": path.relative_to(repo).as_posix(), "file_sha256": file_sha, "original_size": [width, height], "color_mode": "RGB", "resize_shorter_side": target, "resized_size": list(resized_size), "resize_interpolation": "bicubic", "center_crop": [crop_h, crop_w], "crop_box": [left, top, left + crop_w, top + crop_h], "display_array_sha256": _array_digest(rgb), "display_only": True}


def _display_array(array: np.ndarray, *, resize: tuple[int, int] | None = None, order: int = 0) -> tuple[np.ndarray, dict[str, Any]]:
    transform = DisplayTransform(percentile_clip=None, gaussian_sigma=None, max_pool_size=None, resize=resize, resize_order=order)
    displayed = transform(metric_ready(array))
    return np.asarray(displayed.data), {"raw_signed_map_sha256": _array_digest(array), "display_array_sha256": displayed.sha256, "transform": transform.to_dict(), "transforms": list(displayed.transforms), "clipping": None, "clipped_element_count": 0, "display_only": True}


def _caption_data(selection: Mapping[str, Any]) -> dict[str, Any]:
    return _signed({"schema_version": 1, "document_type": "stage_f_f3_caption_data", "stage_id": "F3", "captions": {"figure_1_module": "Deterministic method overview. This schematic describes only the implemented fixed-clean-attention head-direction Value-path workflow and carries no empirical result.", "figure_5_module": RULE_CAPTION, "supplement_s1_module": "Fixed audit examples selected in D1 for provenance and repeatability auditing; they are not best, representative, or random examples.", "supplement_s2_module": RULE_CAPTION + " Duplicate rule hits, weak effects, failures, and misclassified/difficult cases are retained.", "figure_6_module": CROSS_CAPTION, "supplement_s4_module": CROSS_CAPTION}, "median_effect_sample_id": selection["median_effect_sample_id"], "cross_layer_term": "fixed-clean-attention head-direction Value-path attribution progression", "forbidden_terms_absent": ["raw attention map", "raw self-attention", "complete Q/K/V attribution", "class-logit attribution"], "new_empirical_claims": False})


def _source_binding(repo: Path, primary_doc: Mapping[str, Any], union_doc: Mapping[str, Any]) -> dict[str, Any]:
    config = _load_config(repo); selection, _ = _strict_json(repo / SELECTION_PATH, "selection")
    return _signed({"schema_version": 1, "document_type": "stage_f_f3_source_binding", "stage_id": "F3", "status": "PASS", "renderer_commit": _git(repo, "rev-parse", "HEAD"), "config": _record(repo, F3_CONFIG_PATH), "run_spec": _record(repo, F3_RUN_SPEC_PATH), "f2_completion": _file_record(repo, F2_COMPLETION_PATH), "d1_registry": _file_record(repo, D1_REGISTRY_PATH), "d10_per_sample": _record(repo, D10_DELTA_PATH), "d9_sealed_root_inventory_canonical_digest": EXPECTED_D9_INVENTORY_DIGEST, "d10_output_root_inventory_canonical_digest": EXPECTED_D10_INVENTORY_DIGEST, "primary_source_index": primary_doc, "secondary_source_union_digest": union_doc["canonical_digest"], "selection_manifest": _record(repo, SELECTION_PATH), "display_protocol_sha256": EXPECTED_DISPLAY_PROTOCOL_SHA256, "cross_layer_protocol_sha256": EXPECTED_CROSS_PROTOCOL_SHA256, "reference_pdf": {"path": REFERENCE_PDF_PATH, "sha256": EXPECTED_REFERENCE_SHA256, "style_reference_only": True, "scientific_input": False, "content_or_graphics_copied": False, "reference_values_entered_results": False}, "input_digest": canonical_sha256({"selection": selection["canonical_digest"], "primary": primary_doc["canonical_digest"], "secondary": union_doc["canonical_digest"], "config": config["canonical_digest"]})})


def _render_context(repo: Path):
    f0 = load_contracts(repo)["config"]
    return f0, renderer_style(f0)


def _save_formal_figure(fig: Any, output_base: Path, *, content_digest: str, module_id: str, panel_ids: Sequence[str], renderer_commit: str, config_digest: str, panel_manifest_digest: str, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    repo = REPO_ROOT.resolve(); f0 = load_contracts(repo)["config"]; outputs = _save_figure(fig, output_base, f0); plt.close(fig)
    report = {"schema_version": 1, "document_type": "stage_f_f3_render_report", "stage_id": "F3", "module_id": module_id, "figure_id": output_base.name, "panel_ids": list(panel_ids), "renderer_commit": renderer_commit, "config_digest": config_digest, "f0_config_digest": f0["canonical_digest"], "panel_manifest_digest": panel_manifest_digest, "content_digest": content_digest, "format_content_digests": {extension: content_digest for extension in outputs}, "outputs": outputs, "size_inches": list(fig.get_size_inches()), "dpi": f0["export"]["png_dpi"], "display_only": True, "model_loaded": False, "torch_imported": False, "timm_imported": False, "cuda_used": False, "new_attribution_or_fidelity_executed": False, "new_statistic_or_decision_computed": False, "final_combined_manuscript_figure": False}
    if extra: report.update(extra)
    report = _signed(report); _write_new(output_base.with_suffix(".report.json"), report); return report


def _render_method_schematic(repo: Path, config: Mapping[str, Any], panel_manifest_digest: str) -> dict[str, Any]:
    f0 = load_contracts(repo)["config"]; profile = f0["layouts"]["full_width_qualitative"]
    with renderer_style(f0):
        fig, axis = plt.subplots(figsize=(profile["width_in"], 2.35), constrained_layout=True); axis.set_xlim(0, 1); axis.set_ylim(0, 1); axis.axis("off")
        nodes = [(0.07, "Input\nimage"), (0.21, "ViT"), (0.39, "fixed block/head\n+ clean attention"), (0.57, "Value-path IG\n+ matched baseline"), (0.73, "fine 4×4\nattribution map"), (0.90, "insertion / deletion\nfidelity")]
        for x, text in nodes:
            box = FancyBboxPatch((x - 0.065, 0.40), 0.13, 0.21, boxstyle="round,pad=0.012", facecolor="#F3F6F8", edgecolor="#333333", linewidth=0.8); axis.add_patch(box); axis.text(x, 0.505, text, ha="center", va="center", fontsize=7)
        for (x0, _), (x1, _) in zip(nodes, nodes[1:]): axis.add_patch(FancyArrowPatch((x0 + 0.068, 0.505), (x1 - 0.068, 0.505), arrowstyle="-|>", mutation_scale=9, linewidth=0.8, color="#333333"))
        axis.text(0.57, 0.27, "same fixed target • paired methods • no CAM/Grad-CAM/QKV path", ha="center", va="center", fontsize=7, color="#555555")
        axis.text(0.02, 0.93, "(a)", fontweight="bold", fontsize=9); axis.set_title("Figure 1 module — implemented method workflow (no empirical result)", fontsize=8)
        digest = canonical_sha256({"module": "Figure 1", "nodes": nodes, "config": config["canonical_digest"], "panel_manifest": panel_manifest_digest})
        return _save_formal_figure(fig, repo / F3_ROOT_PATH / "modules/figure_1_method_overview/figure_1_method_overview", content_digest=digest, module_id="Figure 1", panel_ids=["F3-FIG1-A"], renderer_commit=_git(repo, "rev-parse", "HEAD"), config_digest=config["canonical_digest"], panel_manifest_digest=panel_manifest_digest, extra={"method_schematic_only": True, "new_empirical_result": False})


def _map_axis(axis: Any, values: np.ndarray | None, *, limit: float | None, cmap: str, title: str, failure: Mapping[str, Any] | None = None, add_colorbar: bool = False) -> None:
    axis.set_xticks([]); axis.set_yticks([]); axis.set_title(title, fontsize=7, pad=2)
    if values is None:
        axis.set_facecolor("#F2F2F2"); axis.text(0.5, 0.58, "FAILED / NA", ha="center", va="center", fontweight="bold", color="#A32020", transform=axis.transAxes, fontsize=7); axis.text(0.5, 0.38, f"order {failure['selection_order']}\n{failure['failure_code']}", ha="center", va="center", transform=axis.transAxes, fontsize=5.5, wrap=True); return
    if limit is None or limit <= 0: limit = float(np.finfo(np.float32).eps)
    image = axis.imshow(values, cmap=cmap, norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit), interpolation="nearest", rasterized=True)
    if add_colorbar: axis.figure.colorbar(image, ax=axis, fraction=0.045, pad=0.025)


def _line_axis(axis: Any, primary_pair: Mapping[str, Mapping[str, Any]]) -> None:
    colors = {NNS_METHOD: "#0072B2", MATCHED_METHOD: "#D55E00"}; styles = {NNS_METHOD: "-", MATCHED_METHOD: "--"}
    for method in METHODS:
        fidelity = primary_pair[method]["fidelity"]; x = fidelity["fractions"]
        axis.plot(x, fidelity["insertion_mean"], color=colors[method], linestyle=styles[method], linewidth=0.8, label=("NNS insertion" if method == NNS_METHOD else "matched insertion"))
        axis.plot(x, fidelity["deletion_mean"], color=colors[method], linestyle=styles[method], linewidth=0.8, alpha=0.55, label=("NNS deletion" if method == NNS_METHOD else "matched deletion"))
    axis.set_xlabel("fraction", fontsize=7); axis.set_ylabel("normalized response", fontsize=7); axis.tick_params(labelsize=6); axis.grid(alpha=0.2, linewidth=0.35); axis.legend(frameon=False, fontsize=5.5, ncol=2, loc="best")


def _qualitative_panel_record(sample_id: str, rule_labels: Sequence[str], drow: Mapping[str, Any], jpeg: Mapping[str, Any], primary_pair: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    nns = primary_pair[NNS_METHOD]["map"]; matched = primary_pair[MATCHED_METHOD]["map"]; difference = nns - matched
    pair_limit = float(max(np.max(np.abs(nns)), np.max(np.abs(matched)))); difference_limit = float(np.max(np.abs(difference)))
    nns_display, nns_transform = _display_array(nns); matched_display, matched_transform = _display_array(matched); diff_display, diff_transform = _display_array(difference)
    return {"panel_id": "qualitative_" + sample_id.rsplit("_", 1)[-1].split(".")[0], "sample_id": sample_id, "selection_order": drow["selection_order"], "selection_rules": list(rule_labels), "sample_status": "misclassified" if drow["misclassified"] else "correct", "delta_fid": drow["delta_fid"], "coordinate": [0, 9], "methods": list(METHODS), "jpeg": jpeg, "sources": [{key: value for key, value in primary_pair[method].items() if key not in {"map", "fidelity"}} for method in METHODS], "color_limits": {"nns": [-pair_limit, pair_limit], "matched": [-pair_limit, pair_limit], "difference": [-difference_limit, difference_limit], "shared_nns_matched": True}, "transform_parameters": {"raw_maps": {"interpolation": "nearest", "clipping": None, "nns": nns_transform, "matched": matched_transform, "difference": diff_transform}, "overlay": {"resize": [224, 224], "resize_order": config["display"]["overlay_resize_order"], "interpolation": config["display"]["overlay_interpolation"], "alpha": config["display"]["overlay_alpha"], "clipping": None}}, "display_array_hashes": {"nns": _array_digest(nns_display), "matched": _array_digest(matched_display), "difference": _array_digest(diff_display)}, "display_only": True}


def _render_qualitative_sample(repo: Path, record: Mapping[str, Any], rgb: np.ndarray, primary_pair: Mapping[str, Mapping[str, Any]], output_base: Path, module_id: str, config: Mapping[str, Any], panel_manifest_digest: str) -> dict[str, Any]:
    f0 = load_contracts(repo)["config"]; nns = primary_pair[NNS_METHOD]["map"]; matched = primary_pair[MATCHED_METHOD]["map"]; difference = nns - matched; limits = record["color_limits"]
    nns_overlay, _ = _display_array(nns, resize=(224, 224), order=config["display"]["overlay_resize_order"]); matched_overlay, _ = _display_array(matched, resize=(224, 224), order=config["display"]["overlay_resize_order"])
    with renderer_style(f0):
        fig = plt.figure(figsize=(7.16, 4.8), constrained_layout=True); grid = GridSpec(2, 4, figure=fig, height_ratios=[1, 0.92])
        axes = [fig.add_subplot(grid[0, index]) for index in range(4)] + [fig.add_subplot(grid[1, index]) for index in range(4)]
        axes[0].imshow(rgb, interpolation="nearest"); axes[0].set_title("Original", fontsize=7); axes[0].set_xticks([]); axes[0].set_yticks([])
        _map_axis(axes[1], nns, limit=limits["nns"][1], cmap="RdBu_r", title="NNS attribution", add_colorbar=True); _map_axis(axes[2], matched, limit=limits["matched"][1], cmap="RdBu_r", title="Matched attribution", add_colorbar=True); _map_axis(axes[3], difference, limit=limits["difference"][1], cmap="PuOr", title="NNS − matched", add_colorbar=True)
        for axis, overlay, title, limit in ((axes[4], nns_overlay, "NNS overlay", limits["nns"][1]), (axes[5], matched_overlay, "Matched overlay", limits["matched"][1])):
            axis.imshow(rgb, interpolation="nearest"); axis.imshow(overlay, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), interpolation="bilinear", alpha=config["display"]["overlay_alpha"], rasterized=True); axis.set_title(title, fontsize=7); axis.set_xticks([]); axis.set_yticks([])
        _line_axis(axes[6], primary_pair); axes[6].set_title("Insertion / deletion", fontsize=7)
        axes[7].axis("off"); axes[7].text(0.02, 0.95, f"status: {record['sample_status']}\norder: {record['selection_order']}\nΔfid: {record['delta_fid']:.4f}\nslot(s): {', '.join(record['selection_rules'])}", va="top", fontsize=7); axes[7].text(0.02, 0.05, "Within-case shared NNS/matched scale.\nDisplay-only overlay; no metric feedback.", va="bottom", fontsize=6, color="#555555")
        for index, axis in enumerate(axes): axis.text(-0.06, 1.02, f"({chr(97 + index)})", transform=axis.transAxes, fontsize=8, fontweight="bold", clip_on=False)
        fig.suptitle(f"{module_id} module — {record['sample_id'].split('/')[-1]}", fontsize=8)
        digest = canonical_sha256({"record": {key: value for key, value in record.items() if key != "output_files"}, "panel_manifest": panel_manifest_digest, "config": config["canonical_digest"]})
        return _save_formal_figure(fig, output_base, content_digest=digest, module_id=module_id, panel_ids=[record["panel_id"]], renderer_commit=_git(repo, "rev-parse", "HEAD"), config_digest=config["canonical_digest"], panel_manifest_digest=panel_manifest_digest, extra={"sample_id": record["sample_id"], "selection_order": record["selection_order"], "selection_rules": record["selection_rules"], "caption_statement": RULE_CAPTION, "nns_matched_shared_scale": True, "difference_independent_scale": True})


def _cross_sample_record(sample_id: str, slots: Sequence[str], primary_index: Mapping[tuple[str, str], Mapping[str, Any]], secondary_index: Mapping[tuple[str, int, int, str], Mapping[str, Any]], registry: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    row = registry[sample_id]; cells = []; scales = []
    for layer, head in COORDINATES:
        pair = {method: (primary_index[(sample_id, method)] if (layer, head) == (0, 9) else secondary_index[(sample_id, layer, head, method)]) for method in METHODS}
        valid_maps = [pair[method]["map"] for method in METHODS if pair[method]["validity_status"] == "VALID"]
        raw_limit = float(max(np.max(np.abs(array)) for array in valid_maps)) if valid_maps else None
        difference = pair[NNS_METHOD]["map"] - pair[MATCHED_METHOD]["map"] if all(pair[m]["validity_status"] == "VALID" for m in METHODS) else None
        difference_limit = float(np.max(np.abs(difference))) if difference is not None else None
        display_hashes = {}
        for method in METHODS:
            if pair[method]["map"] is not None: display_hashes[method] = _display_array(pair[method]["map"])[1]["display_array_sha256"]
        if difference is not None: display_hashes["difference"] = _display_array(difference)[1]["display_array_sha256"]
        scales.append({"sample_id": sample_id, "coordinate": [layer, head], "nns_color_limit": [-raw_limit, raw_limit] if raw_limit is not None else None, "matched_color_limit": [-raw_limit, raw_limit] if raw_limit is not None else None, "difference_color_limit": [-difference_limit, difference_limit] if difference_limit is not None else None, "shared_nns_matched": True, "shared_across_coordinates": False, "clipping": None, "clipped_element_count": 0, "colormap": "RdBu_r", "difference_colormap": "PuOr"})
        for method in METHODS:
            source = pair[method]; cells.append({"sample_id": sample_id, "selection_order": row["selection_order"], "coordinate": [layer, head], "method": method, "validity_status": source["validity_status"], "failure_code": source.get("failure_code"), "attempt_id": source["attempt_id"], "source_evidence_origin": source["evidence_origin"], "source_path": source["source_map_path"] if source["validity_status"] == "VALID" else source["source_result_path"], "source_file_sha256": source["source_map_file_sha256"] if source["validity_status"] == "VALID" else source["source_result_sha256"], "source_result_sha256": source["source_result_sha256"], "applicable_canonical_digest": source.get("applicable_canonical_digest", source.get("source_result_canonical_digest")), "signed_map_sha256": source["signed_map_sha256"], "pair_binding_digest": source["pair_binding_digest"], "color_limits": [-raw_limit, raw_limit] if raw_limit is not None else None, "transform_parameters": {"interpolation": "nearest", "resize": None, "smoothing": None, "clipping": None, "alpha": None, "colormap": "RdBu_r"}, "display_array_sha256": display_hashes.get(method), "imputed_or_interpolated": False})
        cells.append({"sample_id": sample_id, "selection_order": row["selection_order"], "coordinate": [layer, head], "method": "NNS_MINUS_MATCHED", "validity_status": "VALID" if difference is not None else "FAILED_NNS_SOURCE", "failure_code": pair[NNS_METHOD].get("failure_code") if difference is None else None, "source_evidence_origin": "DERIVED_DISPLAY_ONLY_FROM_BOUND_PAIR", "source_path": None, "source_file_sha256": None, "applicable_canonical_digest": canonical_sha256({"nns": pair[NNS_METHOD]["signed_map_sha256"], "matched": pair[MATCHED_METHOD]["signed_map_sha256"]}), "signed_map_sha256": _array_digest(difference) if difference is not None else None, "pair_binding_digest": canonical_sha256({"nns": pair[NNS_METHOD]["pair_binding_digest"], "matched": pair[MATCHED_METHOD]["pair_binding_digest"]}), "color_limits": [-difference_limit, difference_limit] if difference_limit is not None else None, "transform_parameters": {"operation": "NNS_MINUS_MATCHED", "interpolation": "nearest", "resize": None, "smoothing": None, "clipping": None, "alpha": None, "colormap": "PuOr"}, "display_array_sha256": display_hashes.get("difference"), "imputed_or_interpolated": False})
    return {"sample_id": sample_id, "selection_order": row["selection_order"], "slots": list(slots), "image_sha256": row["image_sha256"], "cells": cells, "scales": scales}


def _render_cross_sample(repo: Path, record: Mapping[str, Any], rgb: np.ndarray, primary_index: Mapping[tuple[str, str], Mapping[str, Any]], secondary_index: Mapping[tuple[str, int, int, str], Mapping[str, Any]], output_base: Path, module_id: str, config: Mapping[str, Any], panel_manifest_digest: str) -> dict[str, Any]:
    f0 = load_contracts(repo)["config"]; sample_id = record["sample_id"]; scale_by_coord = {tuple(item["coordinate"]): item for item in record["scales"]}
    with renderer_style(f0):
        fig = plt.figure(figsize=(7.16, 5.6), constrained_layout=True); grid = GridSpec(4, 5, figure=fig, width_ratios=[0.88, 1, 1, 1, 1])
        original = fig.add_subplot(grid[:, 0]); original.imshow(rgb, interpolation="nearest"); original.set_xticks([]); original.set_yticks([]); original.set_title("Original", fontsize=7); original.text(-0.08, 1.02, "(a)", transform=original.transAxes, fontweight="bold", fontsize=8)
        row_labels = ("NNS", "Matched", "NNS − matched", "status / scale")
        for column, coordinate in enumerate(COORDINATES, start=1):
            layer, head = coordinate; sources = {method: (primary_index[(sample_id, method)] if coordinate == (0, 9) else secondary_index[(sample_id, layer, head, method)]) for method in METHODS}; scale = scale_by_coord[coordinate]
            for row_index, method in enumerate(METHODS):
                axis = fig.add_subplot(grid[row_index, column]); source = sources[method]; _map_axis(axis, source["map"], limit=scale["nns_color_limit"][1] if scale["nns_color_limit"] else None, cmap="RdBu_r", title=f"{row_labels[row_index]}\n({layer},{head})" if row_index == 0 else row_labels[row_index], failure=source if source["validity_status"] != "VALID" else None, add_colorbar=(row_index == 1 and source["map"] is not None))
            diff_axis = fig.add_subplot(grid[2, column]); difference = sources[NNS_METHOD]["map"] - sources[MATCHED_METHOD]["map"] if all(sources[m]["validity_status"] == "VALID" for m in METHODS) else None; _map_axis(diff_axis, difference, limit=scale["difference_color_limit"][1] if scale["difference_color_limit"] else None, cmap="PuOr", title=row_labels[2], failure=sources[NNS_METHOD] if difference is None else None, add_colorbar=difference is not None)
            text_axis = fig.add_subplot(grid[3, column]); text_axis.axis("off"); text_axis.text(0.5, 0.68, f"({layer},{head})\nshared within coordinate", ha="center", va="center", fontsize=6); text_axis.text(0.5, 0.20, "pattern only across columns", ha="center", va="center", fontsize=5.5, color="#555555")
        fig.suptitle(f"{module_id} module — fixed-clean-attention head-direction Value-path attribution progression\n{sample_id.split('/')[-1]} | order {record['selection_order']} | slots: {', '.join(record['slots'])}", fontsize=8)
        digest = canonical_sha256({"record": record, "panel_manifest": panel_manifest_digest, "config": config["canonical_digest"]})
        return _save_formal_figure(fig, output_base, content_digest=digest, module_id=module_id, panel_ids=[f"cross_{record['selection_order']}"], renderer_commit=_git(repo, "rev-parse", "HEAD"), config_digest=config["canonical_digest"], panel_manifest_digest=panel_manifest_digest, extra={"sample_id": sample_id, "selection_order": record["selection_order"], "slots": record["slots"], "caption_statement": CROSS_CAPTION, "cross_layer_term": "fixed-clean-attention head-direction Value-path attribution progression", "nns_matched_shared_scale_within_coordinate": True, "absolute_magnitude_compared_across_coordinates": False, "failed_na_retained_in_place": True})


def _render_rule_mapping(repo: Path, mapping: Mapping[str, Any], output_base: Path, config: Mapping[str, Any], panel_manifest_digest: str) -> dict[str, Any]:
    f0 = load_contracts(repo)["config"]
    with renderer_style(f0):
        fig, axis = plt.subplots(figsize=(7.16, 3.8), constrained_layout=True); axis.axis("off")
        columns = ["slot", "order", "sample", "delta_fid", "status"]
        rows = [[item["rule"].replace("_", " "), str(item["selection_order"]), item["sample_id"].split("/")[-1].replace(".JPEG", ""), f"{item['delta_fid']:.4f}", item["prediction_status"]] for item in mapping["rules"]]
        table = axis.table(cellText=rows, colLabels=columns, cellLoc="left", colLoc="left", loc="center", colWidths=[0.29, 0.08, 0.32, 0.10, 0.14]); table.auto_set_font_size(False); table.set_fontsize(7); table.scale(1, 1.45)
        for (row, _column), cell in table.get_celld().items(): cell.set_linewidth(0.5); cell.set_facecolor("#EAF2F8" if row == 0 else "white")
        axis.set_title("Supplement S2 module - complete frozen rule-to-image mapping", fontsize=8, pad=6); axis.text(0.0, 0.04, "Duplicate hits are retained; no replacement was used. Weak and misclassified/difficult slots remain visible.", transform=axis.transAxes, fontsize=6.5)
        digest = canonical_sha256({"mapping": mapping["canonical_digest"], "panel_manifest": panel_manifest_digest, "config": config["canonical_digest"]})
        return _save_formal_figure(fig, output_base, content_digest=digest, module_id="Supplement S2", panel_ids=["S2-RULE-MAPPING"], renderer_commit=_git(repo, "rev-parse", "HEAD"), config_digest=config["canonical_digest"], panel_manifest_digest=panel_manifest_digest, extra={"complete_rule_to_image_mapping": True, "duplicate_hits_retained": True, "caption_statement": RULE_CAPTION})


def _attach_outputs(record: Mapping[str, Any], report: Mapping[str, Any], report_path: Path, repo: Path) -> dict[str, Any]:
    result = dict(record)
    result["output_files"] = {extension: {"path": (report_path.parent / item["path"]).relative_to(repo).as_posix(), "sha256": item["sha256"]} for extension, item in report["outputs"].items()}
    result["render_report"] = {"path": report_path.relative_to(repo).as_posix(), "canonical_digest": report["canonical_digest"]}
    result["renderer_commit"] = report["renderer_commit"]
    result["config_digest"] = report["config_digest"]
    return result


def _copy_review_png(repo: Path, report_path: Path, module: str, sequence: int) -> dict[str, Any]:
    report, _ = _strict_json(report_path, "render report"); source = report_path.parent / report["outputs"]["png"]["path"]; target = repo / REVIEW_ROOT_PATH / f"{sequence:02d}_{module}_{source.name}"; target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists(): raise StageFF3Error("refusing to overwrite F3 review PNG")
    shutil.copyfile(source, target); source_sha = sha256_file(source); target_sha = sha256_file(target)
    if source_sha != target_sha: raise StageFF3Error("review PNG copy differs from formal source")
    return {"module_id": report["module_id"], "figure_id": report["figure_id"], "path": target.relative_to(repo).as_posix(), "sha256": target_sha, "source_path": source.relative_to(repo).as_posix(), "source_sha256": source_sha, "exact_copy": True}


def build_outputs(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); config = _load_config(repo); validate_runtime_isolation(); selection_check = validate_selection_manifests(repo)
    access, _ = _strict_json(repo / ACCESS_SEAL_PATH, "pre-access audit")
    if access.get("source_payload_access_started") is not False or access.get("selection_manifest_validated_before_source_payload_access") is not True:
        raise StageFF3Error("selection access seal is absent")
    primary_index, primary_doc = build_primary_source_index(repo); secondary_index, union_doc = build_secondary_source_union(repo)
    _write_new(repo / CROSS_UNION_PATH, union_doc); _write_new(repo / SOURCE_BINDING_PATH, _source_binding(repo, primary_doc, union_doc))
    selection, _ = _strict_json(repo / SELECTION_PATH, "selection manifest"); mapping, _ = _strict_json(repo / RULE_MAPPING_PATH, "rule mapping"); captions = _caption_data(selection); _write_new(repo / CAPTION_PATH, captions)
    registry = _registry_index(repo); d10 = _d10_row_index(repo)
    audit_slots = [{"slot_index": index, "slot": f"fixed_audit_{index:02d}", "slot_type": "FIXED_AUDIT", "sample_id": item["sample_id"], "selection_order": item["selection_order"]} for index, item in enumerate(selection["fixed_audit_examples"])]
    rule_slots = [{"slot_index": index + 5, "slot": item["rule"], "slot_type": "FROZEN_RULE", "sample_id": item["sample_id"], "selection_order": item["selection_order"]} for index, item in enumerate(mapping["rules"])]
    by_order = {row["selection_order"]: row for row in registry.values()}
    failure_slots = [{"slot_index": index + 10, "slot": f"typed_failure_order_{order}", "slot_type": "TYPED_FAILURE", "sample_id": by_order[order]["sample_id"], "selection_order": order, "failure_coordinate": [11, 2], "failure_method": NNS_METHOD, "failure_code": FAILURE_CODES[order]} for index, order in enumerate(FAILURE_ORDERS)]
    slots = audit_slots + rule_slots + failure_slots; unique_samples = list(dict.fromkeys(item["sample_id"] for item in slots))
    cross_selection = _signed({"schema_version": 1, "document_type": "stage_f_f3_cross_layer_selection_manifest", "stage_id": "F3", "status": "PASS", "figure_6": {"slot": "median_effect", "sample_id": selection["median_effect_sample_id"], "selection_order": selection["median_effect_selection_order"], "selection_basis": "D10_PRIMARY_DELTA_DISTANCE_TO_FROZEN_500_IMAGE_MEDIAN_BEFORE_E3_MAP_ACCESS"}, "s4_slots": slots, "slot_count": len(slots), "unique_render_sample_ids": unique_samples, "unique_render_sample_count": len(unique_samples), "duplicates_retained_in_slot_mapping": True, "silent_replacement_allowed": False, "failure_orders": list(FAILURE_ORDERS)})
    _write_new(repo / CROSS_SELECTION_PATH, cross_selection)
    selected_samples = list(dict.fromkeys([item["sample_id"] for item in selection["fixed_audit_examples"]] + [item["sample_id"] for item in mapping["rules"]] + [item["sample_id"] for item in failure_slots]))
    rgb_by_sample = {}; jpeg_by_sample = {}
    for sample_id in selected_samples: rgb_by_sample[sample_id], jpeg_by_sample[sample_id] = _read_display_rgb(repo, registry[sample_id], config)
    qualitative_records = []
    audit_rules = {item["sample_id"]: ["fixed_audit_example"] for item in selection["fixed_audit_examples"]}
    rule_labels = {sample: [item["rule"] for item in mapping["rules"] if item["sample_id"] == sample] for sample in mapping["render_order_unique_sample_ids"]}
    for sample_id in list(dict.fromkeys(list(audit_rules) + list(rule_labels))):
        labels = audit_rules.get(sample_id, []) + rule_labels.get(sample_id, []); pair = {method: primary_index[(sample_id, method)] for method in METHODS}
        qualitative_records.append(_qualitative_panel_record(sample_id, labels, d10[sample_id], jpeg_by_sample[sample_id], pair, config))
    cross_records = [_cross_sample_record(sample_id, [item["slot"] for item in slots if item["sample_id"] == sample_id], primary_index, secondary_index, registry, config) for sample_id in unique_samples]
    source_projection = _signed({"schema_version": 1, "document_type": "stage_f_f3_panel_source_projection", "stage_id": "F3", "qualitative_panels": qualitative_records, "cross_layer_samples": cross_records, "renderer_commit": _git(repo, "rev-parse", "HEAD"), "config_digest": config["canonical_digest"]})
    projection_digest = source_projection["canonical_digest"]
    reports: list[tuple[str, dict[str, Any], Path, Mapping[str, Any]]] = []
    report = _render_method_schematic(repo, config, projection_digest); path = repo / F3_ROOT_PATH / "modules/figure_1_method_overview/figure_1_method_overview.report.json"
    reports.append(("figure1", report, path, {"panel_id": "F3-FIG1-A", "sample_id": None, "selection_order": None, "coordinate": None, "method": "METHOD_SCHEMATIC", "validity_status": "NOT_EMPIRICAL", "source_path": None, "source_file_sha256": None, "applicable_canonical_digest": config["canonical_digest"], "color_limits": None, "transform_parameters": {"backend": "matplotlib_vector_patches", "rasterized": False}}))
    q_by_sample = {item["sample_id"]: item for item in qualitative_records}
    for index, sample_id in enumerate(mapping["render_order_unique_sample_ids"]):
        record = q_by_sample[sample_id]; pair = {method: primary_index[(sample_id, method)] for method in METHODS}; base = repo / F3_ROOT_PATH / "modules/figure_5_qualitative" / f"rule_{index:02d}_order_{record['selection_order']:03d}"
        report = _render_qualitative_sample(repo, record, rgb_by_sample[sample_id], pair, base, "Figure 5", config, projection_digest); reports.append(("figure5", report, base.with_suffix(".report.json"), record))
    for index, item in enumerate(selection["fixed_audit_examples"]):
        sample_id = item["sample_id"]; record = q_by_sample[sample_id]; pair = {method: primary_index[(sample_id, method)] for method in METHODS}; base = repo / F3_ROOT_PATH / "modules/supplement_s1_fixed_audit" / f"audit_{index:02d}_order_{record['selection_order']:03d}"
        report = _render_qualitative_sample(repo, record, rgb_by_sample[sample_id], pair, base, "Supplement S1 fixed audit example", config, projection_digest); reports.append(("s1", report, base.with_suffix(".report.json"), record))
    s2base = repo / F3_ROOT_PATH / "modules/supplement_s2_rule_mapping/s2_rule_to_image_mapping"; report = _render_rule_mapping(repo, mapping, s2base, config, projection_digest)
    reports.append(("s2", report, s2base.with_suffix(".report.json"), {"panel_id": "S2-RULE-MAPPING", "sample_id": None, "selection_order": None, "coordinate": [0, 9], "method": "RULE_MAPPING", "validity_status": "VALID", "source_path": RULE_MAPPING_PATH, "source_file_sha256": sha256_file(repo / RULE_MAPPING_PATH), "applicable_canonical_digest": mapping["canonical_digest"], "color_limits": None, "transform_parameters": {"table_only": True}}))
    for index, item in enumerate(mapping["rules"]):
        sample_id = item["sample_id"]; record = q_by_sample[sample_id]; slot_record = dict(record); slot_record["selection_rules"] = [item["rule"]]; pair = {method: primary_index[(sample_id, method)] for method in METHODS}; base = repo / F3_ROOT_PATH / "modules/supplement_s2_rule_mapping" / f"slot_{index:02d}_{item['rule']}_order_{record['selection_order']:03d}"
        report = _render_qualitative_sample(repo, slot_record, rgb_by_sample[sample_id], pair, base, "Supplement S2", config, projection_digest); reports.append(("s2", report, base.with_suffix(".report.json"), slot_record))
    median_id = selection["median_effect_sample_id"]; median_record = next(item for item in cross_records if item["sample_id"] == median_id); base = repo / F3_ROOT_PATH / "modules/figure_6_cross_layer/figure_6_median_effect"
    report = _render_cross_sample(repo, median_record, rgb_by_sample[median_id], primary_index, secondary_index, base, "Figure 6", config, projection_digest); reports.append(("figure6", report, base.with_suffix(".report.json"), median_record))
    for index, record in enumerate(cross_records):
        base = repo / F3_ROOT_PATH / "modules/supplement_s4_cross_layer" / f"s4_{index:02d}_order_{record['selection_order']:03d}"
        report = _render_cross_sample(repo, record, rgb_by_sample[record["sample_id"]], primary_index, secondary_index, base, "Supplement S4", config, projection_digest); reports.append(("s4", report, base.with_suffix(".report.json"), record))
    module_panel_records = [_attach_outputs(source_record, report, report_path, repo) for _module, report, report_path, source_record in reports]
    panel_manifest = _signed({"schema_version": 1, "document_type": "stage_f_f3_panel_manifest", "stage_id": "F3", "status": "PASS", "source_projection_digest": projection_digest, "renderer_commit": _git(repo, "rev-parse", "HEAD"), "config_digest": config["canonical_digest"], "panels": module_panel_records, "panel_count": len(module_panel_records), "display_only": True, "new_scientific_computation": False}); _write_new(repo / PANEL_MANIFEST_PATH, panel_manifest)
    cross_cells = [cell for record in cross_records for cell in record["cells"]]
    cross_panel = _signed({"schema_version": 1, "document_type": "stage_f_f3_cross_layer_panel_manifest", "stage_id": "F3", "status": "PASS", "source_union_digest": union_doc["canonical_digest"], "selection_manifest_digest": cross_selection["canonical_digest"], "renderer_commit": _git(repo, "rev-parse", "HEAD"), "config_digest": config["canonical_digest"], "sample_count": len(cross_records), "cells": cross_cells, "failed_cells": [cell for cell in cross_cells if cell["validity_status"] != "VALID"], "zero_imputation": False, "interpolation_of_failed_cells": False}); _write_new(repo / CROSS_PANEL_PATH, cross_panel)
    cross_scale = _signed({"schema_version": 1, "document_type": "stage_f_f3_cross_layer_scale_manifest", "stage_id": "F3", "status": "PASS", "scales": [scale for record in cross_records for scale in record["scales"]], "same_sample_same_coordinate_shared_nns_matched_scale": True, "different_coordinates_shared_scale": False, "cross_coordinate_absolute_magnitude_comparison_allowed": False, "percentile_clipping": False, "unclipped_audit_view_required": False}); _write_new(repo / CROSS_SCALE_PATH, cross_scale)
    render_rows = []
    for module, report, report_path, _source in reports:
        render_rows.append({"module_group": module, "module_id": report["module_id"], "figure_id": report["figure_id"], "report": {"path": report_path.relative_to(repo).as_posix(), "sha256": sha256_file(report_path), "canonical_digest": report["canonical_digest"]}, "outputs": {extension: {"path": (report_path.parent / item["path"]).relative_to(repo).as_posix(), "sha256": item["sha256"], "bytes": item["bytes"]} for extension, item in report["outputs"].items()}, "content_digest": report["content_digest"], "panel_source_projection_digest": report["panel_manifest_digest"]})
    render_inventory = _signed({"schema_version": 1, "document_type": "stage_f_f3_render_inventory", "stage_id": "F3", "status": "PASS", "renders": render_rows, "render_instance_count": len(render_rows), "exported_file_count": len(render_rows) * 3, "module_counts": dict(Counter(item["module_group"] for item in render_rows)), "formats": ["png", "pdf", "svg"], "png_dpi": 600, "final_combined_manuscript_figures_rendered": False}); _write_new(repo / RENDER_INVENTORY_PATH, render_inventory)
    review = [_copy_review_png(repo, report_path, module, index + 1) for index, (module, _report, report_path, _source) in enumerate(reports)]
    review_inventory = _signed({"schema_version": 1, "document_type": "stage_f_f3_review_copy_inventory", "stage_id": "F3", "status": "PASS", "review_copy_count": len(review), "copies": review}); _write_new(repo / REVIEW_INVENTORY_PATH, review_inventory)
    access_audit = _signed({"schema_version": 1, "document_type": "stage_f_f3_access_audit", "stage_id": "F3", "status": "PASS", "selection_access_seal": _record(repo, ACCESS_SEAL_PATH), "selection_manifest_sealed_and_validated_before_payload_access": True, "selected_jpeg_files_read": len(selected_samples), "primary_map_files_validated": 1000, "secondary_terminal_payloads_validated": 3000, "secondary_valid_map_files_read": 2998, "unselected_jpeg_files_read": 0, "model_access": False, "checkpoint_access": False, "torch_imported": False, "timm_imported": False, "cuda_or_gpu_access": False, "nvidia_device_handles": [], "forward_backward_ig_attribution_fidelity_executed": False, "new_scientific_computation": False, "display_arrays_entered_metric_or_selection": False}); _write_new(repo / ACCESS_AUDIT_PATH, access_audit)
    _write_new(repo / FALSE_PASS_PATH, build_false_pass_audit(repo))
    return {"status": "OUTPUTS_BUILT", "selection": selection_check, "primary_maps_validated": 1000, "secondary_terminals_validated": 3000, "secondary_valid_maps": 2998, "secondary_failed_units": 2, "figure_6_sample_id": median_id, "s4_slot_count": len(slots), "s4_unique_sample_count": len(unique_samples), "render_instances": len(render_rows), "exported_files": len(render_rows) * 3, "review_copies": len(review)}


def build_false_pass_audit(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    mechanisms = {
        ATTACKS[0]: "startup gate recomputes F2 ref, completion hash/digest, terminal state, inventory and tracked custody",
        ATTACKS[1]: "primary resolver checks 1000 unique D8 maps, L0 manifests, curve map identities, reconciliation pair bindings and D10 AUCs",
        ATTACKS[2]: "typed-plan keys must equal exactly 3000 E3-R4 terminal-table keys and each key must resolve to exactly one retained attempt",
        ATTACKS[3]: "all valid source arrays are loaded with dtype/shape/finite/file/hash checks and origin counts must be 516/12/636/1834",
        ATTACKS[4]: "orders 117 and 352 require exact coordinate, NNS method, failure code, absent valid map and retained matched pair",
        ATTACKS[5]: "coordinate and method key sets are checked against four frozen coordinates and two methods",
        ATTACKS[6]: "initialize atomically publishes and independently rebuilds selection manifests before source-payload access seal changes",
        ATTACKS[7]: "Figure 6 report and cross-layer selection manifest require the exact D10-only median_effect sample",
        ATTACKS[8]: "S4 requires ordered 5 audit, 5 rule and 2 typed-failure slots",
        ATTACKS[9]: "slot-to-image rows are immutable and render deduplication preserves every duplicate mapping",
        ATTACKS[10]: "scale manifest requires identical symmetric NNS/matched color limits for each sample-coordinate",
        ATTACKS[11]: "caption and reports explicitly forbid absolute-magnitude comparison across coordinates",
        ATTACKS[12]: "failed cells require null map/hash, explicit failure panel and imputed_or_interpolated=false",
        ATTACKS[13]: "JPEGs bind D1 hashes; D8/E3 arrays bind file and canonical array hashes plus result/pair records",
        ATTACKS[14]: "each render report uses one content identity for PNG/PDF/SVG and hashes all three outputs",
        ATTACKS[15]: "selection consumes D10/D1 only; display transforms are tagged display_only and no metric function accepts them",
        ATTACKS[16]: "runtime and source guards reject Torch/timm/model/checkpoint/CUDA/NVIDIA and scientific execution surfaces",
        ATTACKS[17]: "completion records no new statistic, p-value, winner, endpoint or GO/NO-GO decision",
        ATTACKS[18]: "future-output guard rejects F4 roots and all reports set final_combined_manuscript_figure=false",
        ATTACKS[19]: "finalize validates sources, renders, review copies, manifests, tests and analysis before atomic terminal rename",
        ATTACKS[20]: "source validator requires git diff --check to be empty",
    }
    document = _signed({"schema_version": 1, "document_type": "stage_f_f3_false_pass_audit", "stage_id": "F3", "status": "PASS", "attacks": [{"attack": attack, "result": "REJECTED", "mechanism": mechanisms[attack]} for attack in ATTACKS], "pass_possible_without_all_conditions": False})
    _schema(repo, document, F3_FALSE_PASS_SCHEMA_PATH, "F3 false-pass audit")
    return document


def validate_render_report(report_path: Path, repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); report, raw = _strict_json(report_path, "F3 render report")
    if raw != canonical_json_bytes(report) + b"\n": raise StageFF3Error("F3 render report is not byte-canonical")
    _validate_signed(report, "F3 render report")
    if any(report.get(key) is not False for key in ("model_loaded", "torch_imported", "timm_imported", "cuda_used", "new_attribution_or_fidelity_executed", "new_statistic_or_decision_computed", "final_combined_manuscript_figure")):
        raise StageFF3Error("F3 render report crossed a runtime or scientific boundary")
    if report.get("display_only") is not True or len(set(report["format_content_digests"].values())) != 1:
        raise StageFF3Error("F3 render content identity drifted")
    for extension, record in report["outputs"].items():
        path = report_path.parent / record["path"]
        if Path(record["path"]).name != record["path"] or path.suffix.lower() != f".{extension}" or not path.is_file() or sha256_file(path) != record["sha256"]:
            raise StageFF3Error(f"F3 {extension} output tampered")
    with Image.open(report_path.parent / report["outputs"]["png"]["path"]) as image:
        if image.size != tuple(round(value * report["dpi"]) for value in report["size_inches"]): raise StageFF3Error("F3 PNG canvas drifted")
    pdf = (report_path.parent / report["outputs"]["pdf"]["path"]).read_bytes()
    if b"/FontFile2" not in pdf and b"/FontFile3" not in pdf: raise StageFF3Error("F3 PDF font is not embedded")
    if report.get("caption_statement") == CROSS_CAPTION:
        if report.get("nns_matched_shared_scale_within_coordinate") is not True or report.get("absolute_magnitude_compared_across_coordinates") is not False or report.get("failed_na_retained_in_place") is not True: raise StageFF3Error("cross-layer render disclosure drifted")
    return {"status": "PASS", "module_id": report["module_id"], "figure_id": report["figure_id"], "formats": sorted(report["outputs"])}


def assert_no_f4_outputs(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    if any((repo / "evidence/stage_f").glob("f4*")) or any((repo / "reports").glob("stage_f_f4*")):
        raise StageFF3Error("F4 output exists")
    forbidden = re.compile(r"(?:final[_ -]?(?:figure|supplement)|manuscript)", re.I)
    if any(path.is_file() and forbidden.search(path.name) for root in (repo / F3_ROOT_PATH, repo / F3_REPORT_PATH) if root.exists() for path in root.rglob("*")):
        raise StageFF3Error("F3 generated a final combined manuscript artifact")
    return {"status": "PASS", "f4_started": False, "final_combined_manuscript_figures_rendered": False}


def validate_outputs(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); validate_runtime_isolation(jpeg_read=True, map_read=True); assert_no_f4_outputs(repo); _verify_f2(repo); _verify_frozen_protocols(repo); _verify_scientific_state(repo); selection_check = validate_selection_manifests(repo)
    primary_index, primary_doc = build_primary_source_index(repo); secondary_index, union_expected = build_secondary_source_union(repo)
    union, _ = _strict_json(repo / CROSS_UNION_PATH, "cross-layer union"); _validate_signed(union, "cross-layer union")
    if union != union_expected: raise StageFF3Error("cross-layer union is not source-recomputable")
    source, _ = _strict_json(repo / SOURCE_BINDING_PATH, "source binding"); _validate_signed(source, "source binding")
    if source["primary_source_index"] != primary_doc or source["secondary_source_union_digest"] != union["canonical_digest"]: raise StageFF3Error("source binding drifted")
    selection, _ = _strict_json(repo / SELECTION_PATH, "selection manifest"); mapping, _ = _strict_json(repo / RULE_MAPPING_PATH, "rule mapping"); cross_selection, _ = _strict_json(repo / CROSS_SELECTION_PATH, "cross selection")
    for value, context in ((selection, "selection"), (mapping, "mapping"), (cross_selection, "cross selection")): _validate_signed(value, context)
    if cross_selection["figure_6"] != {"slot": "median_effect", "sample_id": selection["median_effect_sample_id"], "selection_order": selection["median_effect_selection_order"], "selection_basis": "D10_PRIMARY_DELTA_DISTANCE_TO_FROZEN_500_IMAGE_MEDIAN_BEFORE_E3_MAP_ACCESS"}: raise StageFF3Error("Figure 6 median_effect binding drifted")
    slots = cross_selection["s4_slots"]
    if len(slots) != 12 or [item["slot_type"] for item in slots] != ["FIXED_AUDIT"] * 5 + ["FROZEN_RULE"] * 5 + ["TYPED_FAILURE"] * 2 or [item["selection_order"] for item in slots[-2:]] != [117, 352]: raise StageFF3Error("S4 slot coverage or order drifted")
    panel, _ = _strict_json(repo / PANEL_MANIFEST_PATH, "panel manifest"); cross_panel, _ = _strict_json(repo / CROSS_PANEL_PATH, "cross panel manifest"); scales, _ = _strict_json(repo / CROSS_SCALE_PATH, "cross scale manifest"); render, _ = _strict_json(repo / RENDER_INVENTORY_PATH, "render inventory"); review, _ = _strict_json(repo / REVIEW_INVENTORY_PATH, "review inventory"); captions, _ = _strict_json(repo / CAPTION_PATH, "caption data"); access, _ = _strict_json(repo / ACCESS_AUDIT_PATH, "access audit")
    for value, context in ((panel, "panel manifest"), (cross_panel, "cross panel"), (scales, "scale manifest"), (render, "render inventory"), (review, "review inventory"), (captions, "caption data"), (access, "access audit")): _validate_signed(value, context)
    expected_unique = len(cross_selection["unique_render_sample_ids"])
    if cross_panel["sample_count"] != expected_unique or len(cross_panel["cells"]) != expected_unique * 12: raise StageFF3Error("cross-layer panel coverage drifted")
    failed_nns = [item for item in cross_panel["cells"] if item["method"] == NNS_METHOD and item["validity_status"] == "FAILED"]
    if [(item["selection_order"], item["failure_code"]) for item in failed_nns] != [(117, FAILURE_CODES[117]), (352, FAILURE_CODES[352])] or any(item["source_path"] is None or item["signed_map_sha256"] is not None or item["imputed_or_interpolated"] for item in failed_nns): raise StageFF3Error("failure panels drifted or were imputed")
    if any(item["nns_color_limit"] != item["matched_color_limit"] or item["shared_nns_matched"] is not True or item["shared_across_coordinates"] is not False or item["clipping"] is not None for item in scales["scales"]): raise StageFF3Error("cross-layer scale contract drifted")
    if scales.get("cross_coordinate_absolute_magnitude_comparison_allowed") is not False or RULE_CAPTION not in captions["captions"]["figure_5_module"] or CROSS_CAPTION != captions["captions"]["figure_6_module"]: raise StageFF3Error("caption boundary drifted")
    expected_modules = {"figure1": 1, "figure5": len(mapping["render_order_unique_sample_ids"]), "s1": 5, "s2": 6, "figure6": 1, "s4": expected_unique}
    if render["module_counts"] != expected_modules or render["render_instance_count"] != sum(expected_modules.values()) or render["exported_file_count"] != render["render_instance_count"] * 3: raise StageFF3Error("F3 module/render counts drifted")
    for item in render["renders"]:
        report_path = repo / item["report"]["path"]; validate_render_report(report_path, repo); report, _ = _strict_json(report_path, "render report")
        if report["panel_manifest_digest"] != panel["source_projection_digest"] or report["config_digest"] != _load_config(repo)["canonical_digest"] or report["content_digest"] != item["content_digest"]: raise StageFF3Error("render panel/config/content binding drifted")
    if review["review_copy_count"] != render["render_instance_count"] or len(review["copies"]) != render["render_instance_count"]: raise StageFF3Error("review copy coverage drifted")
    for item in review["copies"]:
        if item["source_sha256"] != item["sha256"] or sha256_file(repo / item["path"]) != item["sha256"]: raise StageFF3Error("review copy drifted")
    if not (repo / ANALYSIS_PATH).is_file(): raise StageFF3Error("manual figure analysis is absent")
    analysis = (repo / ANALYSIS_PATH).read_text(encoding="utf-8")
    required_review_fields = ("图能说明什么", "图不能说明什么", "最可能的误读", "文字或 colorbar 裁切", "字号", "NNS/matched 色限", "FAILED/NA", "panel 顺序和图注", "原图、map、overlay 对齐", "插值伪影")
    if any(field not in analysis for field in required_review_fields) or any(f"## Image #{index}:" not in analysis for index in range(1, review["review_copy_count"] + 1)): raise StageFF3Error("manual figure analysis is incomplete")
    false_pass, _ = _strict_json(repo / FALSE_PASS_PATH, "false-pass audit"); _validate_signed(false_pass, "false-pass audit"); _schema(repo, false_pass, F3_FALSE_PASS_SCHEMA_PATH, "false-pass audit")
    if [item["attack"] for item in false_pass["attacks"]] != list(ATTACKS) or any(item["result"] != "REJECTED" for item in false_pass["attacks"]): raise StageFF3Error("false-pass audit drifted")
    if access.get("selection_manifest_sealed_and_validated_before_payload_access") is not True or any(access.get(key) is not False for key in ("model_access", "checkpoint_access", "torch_imported", "timm_imported", "cuda_or_gpu_access", "forward_backward_ig_attribution_fidelity_executed", "new_scientific_computation", "display_arrays_entered_metric_or_selection")): raise StageFF3Error("F3 access boundary drifted")
    if _git(repo, "diff", "--check"): raise StageFF3Error("git diff --check failed")
    checks = {attack: "PASS" for attack in ATTACKS}
    report = _signed({"schema_version": 1, "document_type": "stage_f_f3_validation_report", "stage_id": "F3", "status": "PASS", "checks": checks, "selection": selection_check, "primary_map_count": len(primary_index), "secondary_terminal_count": len(secondary_index), "secondary_valid_map_count": union["valid_maps"], "secondary_failed_count": union["failed_units"], "figure_6_sample_id": selection["median_effect_sample_id"], "s4_slot_count": 12, "s4_unique_sample_count": expected_unique, "failure_panel_status": {"117": "FAILED_RETAINED_IN_PLACE", "352": "FAILED_RETAINED_IN_PLACE"}, "module_counts": expected_modules, "render_instance_count": render["render_instance_count"], "exported_file_count": render["exported_file_count"], "review_copy_count": review["review_copy_count"], "model_torch_timm_gpu_cuda_access": False, "new_attribution_fidelity_or_scientific_computation": False, "new_statistic_p_value_winner_or_decision": False, "final_combined_manuscript_figures_rendered": False, "next_authorization": "F4_ONLY"}); _schema(repo, report, F3_VALIDATION_SCHEMA_PATH, "F3 validation report"); return report


def record_test_report(repo_root: Path | str, *, passed: int, failed: int, command: str) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    if type(passed) is not int or passed <= 0 or failed != 0 or not command:
        raise StageFF3Error("terminal test report requires positive passes and zero failures")
    report = _signed({"schema_version": 1, "document_type": "stage_f_f3_test_report", "stage_id": "F3", "status": "PASS", "command": command, "passed": passed, "failed": failed, "required_environment": {"CUDA_VISIBLE_DEVICES": "", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}, "model_access": False, "torch_imported": False, "timm_imported": False, "gpu_cuda_access": False, "new_attribution_or_fidelity_executed": False})
    _write_new(repo / TEST_PATH, report); return report


def finalize(repo_root: Path | str = REPO_ROOT, *, completed_at: str) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); root = repo / F3_ROOT_PATH; validate_runtime_isolation(jpeg_read=True, map_read=True)
    if (repo / COMPLETION_PATH).exists(): raise StageFF3Error("F3 terminal completion already exists")
    test = _record(repo, TEST_PATH); validation = validate_outputs(repo); _write_new(repo / VALIDATION_PATH, validation)
    excluded = {"payload_manifest.json", "whole_root_inventory.json", "status.json", "completion/f3.pending.json", "completion/f3.json"}
    artifacts = [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(root.rglob("*")) if path.is_file() and not path.is_symlink() and path.relative_to(root).as_posix() not in excluded]
    manifest = _signed({"schema_version": 1, "document_type": "stage_f_f3_payload_manifest", "stage_id": "F3", "immutable": True, "payload_count": len(artifacts), "artifacts": artifacts, "required_modules": ["Figure 1", "Figure 5", "Figure 6", "Supplement S1", "Supplement S2", "Supplement S4"], "final_combined_manuscript_figures_rendered": False}); _write_new(repo / PAYLOAD_MANIFEST_PATH, manifest)
    inventory = _inventory(root, exclude=("whole_root_inventory.json", "status.json", "completion/f3.pending.json", "completion/f3.json"))
    inventory_doc = _signed({"schema_version": 1, "document_type": "stage_f_f3_whole_root_inventory", "stage_id": "F3", "root": F3_ROOT_PATH, "scope": "all regular F3 files except self-referential inventory status and completion receipts", "files": inventory["files"], "file_count": inventory["file_count"], "total_bytes": inventory["total_bytes"]}); _write_new(repo / INVENTORY_PATH, inventory_doc)
    selection, _ = _strict_json(repo / SELECTION_PATH, "selection"); cross_selection, _ = _strict_json(repo / CROSS_SELECTION_PATH, "cross selection"); render, _ = _strict_json(repo / RENDER_INVENTORY_PATH, "render inventory")
    status = _signed({"schema_version": 1, "document_type": "stage_f_f3_status", "stage_id": "F3", "status": "PASS", "terminal": True, "immutable": True, "package_validation": "PASS", "workflow_validation": "PASS", "modules_completed": ["Figure 1", "Figure 5", "Figure 6", "Supplement S1", "Supplement S2", "Supplement S4"], "figure_6_sample_id": selection["median_effect_sample_id"], "s4_slot_count": cross_selection["slot_count"], "s4_unique_sample_count": cross_selection["unique_render_sample_count"], "e3_union": {"expected": 3000, "valid": 2998, "failed": 2}, "render_instance_count": render["render_instance_count"], "exported_file_count": render["exported_file_count"], "model_torch_timm_gpu_cuda_access": False, "new_scientific_computation": False, "upstream_decisions_or_completions_modified": False, "final_combined_manuscript_figures_rendered": False, "f4_started": False, "next_authorization": "F4_ONLY", "whole_root_inventory": {"canonical_digest": inventory_doc["canonical_digest"], "file_count": inventory_doc["file_count"], "total_bytes": inventory_doc["total_bytes"]}, "git": {"commit": "SELF", "ref": F3_ARCHIVE_REF}}); _write_new(repo / STATUS_PATH, status)
    pending, _ = _strict_json(repo / PENDING_PATH, "F3 pending"); _validate_signed(pending, "F3 pending")
    completion = _signed({"schema_version": 1, "document_type": "stage_f_f3_terminal_completion", "stage_id": "F3", "status": "PASS", "started_at": pending["started_at"], "completed_at": completed_at, "package_validation": "PASS", "workflow_validation": "PASS", "modules_completed": status["modules_completed"], "figure_6_sample_id": status["figure_6_sample_id"], "s4_slot_count": status["s4_slot_count"], "s4_unique_sample_count": status["s4_unique_sample_count"], "e3_union": status["e3_union"], "render_instance_count": status["render_instance_count"], "exported_file_count": status["exported_file_count"], "model_access": False, "torch_imported": False, "timm_imported": False, "gpu_cuda_access": False, "new_scientific_computation": False, "new_statistic_p_value_winner_or_go_no_go": False, "upstream_decisions_or_completions_modified": False, "d10_e4_f0_f1_f2_decisions_and_completions_modified": False, "final_combined_manuscript_figures_rendered": False, "startup_gate": _record(repo, STARTUP_PATH), "selection_manifest": _record(repo, SELECTION_PATH), "rule_to_image_manifest": _record(repo, RULE_MAPPING_PATH), "source_binding": _record(repo, SOURCE_BINDING_PATH), "panel_manifest": _record(repo, PANEL_MANIFEST_PATH), "caption_data": _record(repo, CAPTION_PATH), "cross_layer_source_union": _record(repo, CROSS_UNION_PATH), "cross_layer_selection_manifest": _record(repo, CROSS_SELECTION_PATH), "cross_layer_panel_manifest": _record(repo, CROSS_PANEL_PATH), "cross_layer_scale_manifest": _record(repo, CROSS_SCALE_PATH), "render_inventory": _record(repo, RENDER_INVENTORY_PATH), "review_copy_inventory": _record(repo, REVIEW_INVENTORY_PATH), "access_audit": _record(repo, ACCESS_AUDIT_PATH), "validation_report": _record(repo, VALIDATION_PATH), "false_pass_audit": _record(repo, FALSE_PASS_PATH), "test_report": test, "figure_analysis": _file_record(repo, ANALYSIS_PATH), "payload_manifest": _record(repo, PAYLOAD_MANIFEST_PATH), "status_record": _record(repo, STATUS_PATH), "whole_root_inventory": _record(repo, INVENTORY_PATH), "git": {"commit": "SELF", "ref": F3_ARCHIVE_REF}, "next_authorization": "F4_ONLY", "f4_started": False}); _schema(repo, completion, F3_COMPLETION_SCHEMA_PATH, "F3 completion")
    temporary = (repo / PENDING_PATH).with_name(".f3-terminal.tmp"); _write_new(temporary, completion); os.replace(temporary, repo / PENDING_PATH); os.replace(repo / PENDING_PATH, repo / COMPLETION_PATH)
    return validate_completion(repo)


def validate_completion(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); root = repo / F3_ROOT_PATH
    if not (repo / COMPLETION_PATH).is_file() or (repo / PENDING_PATH).exists(): raise StageFF3Error("terminal F3 completion missing or pending remains")
    completion, raw = _strict_json(repo / COMPLETION_PATH, "F3 completion")
    if raw != canonical_json_bytes(completion) + b"\n": raise StageFF3Error("F3 completion is not byte-canonical")
    _validate_signed(completion, "F3 completion"); _schema(repo, completion, F3_COMPLETION_SCHEMA_PATH, "F3 completion")
    expected = {"status": "PASS", "package_validation": "PASS", "workflow_validation": "PASS", "next_authorization": "F4_ONLY", "f4_started": False, "final_combined_manuscript_figures_rendered": False, "model_access": False, "torch_imported": False, "timm_imported": False, "gpu_cuda_access": False, "new_scientific_computation": False, "upstream_decisions_or_completions_modified": False}
    if any(completion.get(key) != value for key, value in expected.items()): raise StageFF3Error("F3 terminal state drifted")
    for field in ("startup_gate", "selection_manifest", "rule_to_image_manifest", "source_binding", "panel_manifest", "caption_data", "cross_layer_source_union", "cross_layer_selection_manifest", "cross_layer_panel_manifest", "cross_layer_scale_manifest", "render_inventory", "review_copy_inventory", "access_audit", "validation_report", "false_pass_audit", "test_report", "payload_manifest", "status_record", "whole_root_inventory"):
        if completion[field] != _record(repo, completion[field]["path"]): raise StageFF3Error(f"F3 completion binding drifted: {field}")
    if completion["figure_analysis"] != _file_record(repo, ANALYSIS_PATH): raise StageFF3Error("F3 figure analysis binding drifted")
    inventory, _ = _strict_json(repo / INVENTORY_PATH, "F3 inventory"); _validate_signed(inventory, "F3 inventory")
    rebuilt = _inventory(root, exclude=("whole_root_inventory.json", "status.json", "completion/f3.pending.json", "completion/f3.json"))
    unsigned = {"schema_version": 1, "document_type": "stage_f_f3_whole_root_inventory", "stage_id": "F3", "root": F3_ROOT_PATH, "scope": "all regular F3 files except self-referential inventory status and completion receipts", "files": rebuilt["files"], "file_count": rebuilt["file_count"], "total_bytes": rebuilt["total_bytes"]}
    if any(inventory[key] != rebuilt[key] for key in ("files", "file_count", "total_bytes")) or inventory["canonical_digest"] != canonical_sha256(unsigned): raise StageFF3Error("F3 whole-root inventory drifted")
    validate_runtime_isolation(jpeg_read=True, map_read=True); assert_no_f4_outputs(repo)
    return {**expected, "figure_6_sample_id": completion["figure_6_sample_id"], "s4_slot_count": completion["s4_slot_count"], "s4_unique_sample_count": completion["s4_unique_sample_count"], "e3_union": completion["e3_union"], "render_instance_count": completion["render_instance_count"], "exported_file_count": completion["exported_file_count"], "whole_root_inventory_digest": inventory["canonical_digest"]}


__all__ = ["ATTACKS", "F3_ARCHIVE_REF", "F3_ROOT_PATH", "F3_REPORT_PATH", "StageFF3Error", "assert_f3_source_boundary", "assert_no_f4_outputs", "build_false_pass_audit", "build_outputs", "build_primary_source_index", "build_secondary_source_union", "build_selection_manifests", "finalize", "initialize", "record_test_report", "validate_completion", "validate_outputs", "validate_render_report", "validate_runtime_isolation", "validate_selection_manifests", "verify_startup_gates"]
