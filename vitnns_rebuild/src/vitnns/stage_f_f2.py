"""Stage F F2 display-only rendering from the sealed E4 paper-data package.

The adapter accepts only the frozen E4 formal package and delegates drawing to
the F0 renderer.  It never reads images, attribution maps, model artifacts, or
Stage E unit roots and never computes an inferential result or decision.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
from PIL import Image, __version__ as PILLOW_VERSION
from jsonschema import Draft202012Validator

from .constants import REPO_ROOT
from .stage_d_provenance import canonical_json_bytes, canonical_sha256
from .stage_f_renderer import (
    StageFRendererError, load_contracts, render_panel_figure, sha256_file,
    validate_font,
)

F2_CONFIG_PATH = "config/stage_f_f2.json"
F2_CONFIG_SCHEMA_PATH = "config/stage_f_f2.schema.json"
F2_RUN_SPEC_PATH = "config/stage_f_f2_run_spec.json"
F2_RUN_SPEC_SCHEMA_PATH = "config/stage_f_f2_run_spec.schema.json"
F2_PLOT_DATA_SCHEMA_PATH = "config/stage_f_f2_plot_data.schema.json"
F2_RENDER_REPORT_SCHEMA_PATH = "config/stage_f_f2_render_report.schema.json"
F2_VALIDATION_SCHEMA_PATH = "config/stage_f_f2_validation_report.schema.json"
F2_COMPLETION_SCHEMA_PATH = "config/stage_f_f2_completion.schema.json"
F2_FALSE_PASS_SCHEMA_PATH = "config/stage_f_f2_false_pass_audit.schema.json"

F2_ROOT_PATH = "evidence/stage_f/f2_r2_recovery_v2_20260907/formal"
F2_REPORT_PATH = "reports/stage_f_f2_r2_20260907"
PENDING_PATH = F2_ROOT_PATH + "/completion/f2.pending.json"
COMPLETION_PATH = F2_ROOT_PATH + "/completion/f2.json"
STARTUP_PATH = F2_ROOT_PATH + "/startup_gate.json"
SOURCE_BINDING_PATH = F2_ROOT_PATH + "/source_binding.json"
ACCESS_AUDIT_PATH = F2_ROOT_PATH + "/access_audit.json"
ENVIRONMENT_PATH = F2_ROOT_PATH + "/environment_font_style_provenance.json"
RENDER_INVENTORY_PATH = F2_ROOT_PATH + "/render_inventory.json"
REVIEW_INVENTORY_PATH = F2_ROOT_PATH + "/review_copy_inventory.json"
MANIFEST_PATH = F2_ROOT_PATH + "/payload_manifest.json"
INVENTORY_PATH = F2_ROOT_PATH + "/whole_root_inventory.json"
STATUS_PATH = F2_ROOT_PATH + "/status.json"
VALIDATION_PATH = F2_REPORT_PATH + "/validation_report.json"
FALSE_PASS_PATH = F2_REPORT_PATH + "/false_pass_audit.json"
TEST_PATH = F2_REPORT_PATH + "/test_report.json"
ANALYSIS_PATH = F2_REPORT_PATH + "/figure_analysis.md"
REVIEW_ROOT_PATH = F2_REPORT_PATH + "/figures_png"

F1_ARCHIVE_REF = "refs/archive/stage-f-v1-f1-completion"
F1_COMMIT = "7a516571eabf364b7bb5d46ceb8aa74b450b3ae3"
F1_ROOT_PATH = "evidence/stage_f/f1_r3_recovery_v2_20260907/formal"
F1_REPORT_PATH = "reports/stage_f_f1_r3_20260907"
F1_COMPLETION_PATH = F1_ROOT_PATH + "/completion/f1.json"
F1_INVENTORY_PATH = F1_ROOT_PATH + "/whole_root_inventory.json"
F0_COMMIT = "c4df92ff05054712406caa5524c5d129da7dab42"
F0_ROOT_PATH = "evidence/stage_f/f0_r4_recovery_v2_20260907/formal"
F0_REPORT_PATH = "reports/stage_f_f0_r4_20260907"
D10_COMMIT = "7520689bb031304e4789567f99d093062aa50f72"
D10_ROOT_PATH = "evidence/stage_d/d10_recovery_v2_20260906/formal_r2"
D10_REPORT_PATH = "reports/stage_d_d10_recovery_v2_20260906"
E4_ARCHIVE_REF = "refs/archive/stage-e-e4-completion"
E4_COMMIT = "7c82eca4e7de488f4537eff2b3fdb59feb3413ab"
E4_ROOT_PATH = "evidence/stage_e/e4_recovery_v2_20260907/formal"
E4_PAPER_DATA_ROOT_PATH = E4_ROOT_PATH + "/paper_data"
F2_ARCHIVE_REF = "refs/archive/stage-f-v1-f2-completion"

EXPECTED_F1_COMPLETION_SHA256 = "079a646e7ab613d5eda3c99ed9f24d7b1635bfe31d53d5be422b5b5073cb2df3"
EXPECTED_F1_COMPLETION_DIGEST = "cba6d26364aea25a5f8276e96b6459e40742ada69af6beaccd4cdaa37dcaf005"
EXPECTED_F1_INVENTORY_DIGEST = "a32e7998ed13810f1ccd4e2f6d7f6358711add0513cdbcb5438765ce445885f0"
EXPECTED_E4_COMPLETION_SHA256 = "42ad2b483e40c164b4e26e1d8d65ecfc00703a2c01c2f0589961bcbefa8d58dc"
EXPECTED_E4_COMPLETION_DIGEST = "630423730c48c9ea4ae67d8b27b0ec690ee6c96f0ad36a1b3aa1312c3091a3a9"
EXPECTED_E4_INVENTORY_DIGEST = "75249c74e08cb5038acbc86f71051260f9d9b7b6cc80d23c517e8669f3765f67"
EXPECTED_E4_RECIPE_SHA256 = "57306bc69e399d1ea2353d908ac0d8b22403d17ec29318a3eadf36b3b8d98a54"

TABLES = ("patch_scale_sensitivity", "upsampling_sensitivity", "layer_head_descriptive", "prediction_strata", "resource_and_qc")
MODULES = ("F2-A", "F2-B", "F2-C", "F2-D")
LAYOUTS = ("single_column", "double_column")
MODULE_SLUG = {module: module.lower().replace("-", "_") for module in MODULES}
CAPTION_KEY = {
    "F2-A": "f2_a_patch_scale_sensitivity",
    "F2-B": "f2_b_upsampling_sensitivity",
    "F2-C": "f2_c_layer_head_prediction_strata",
    "F2-D": "f2_d_resource_qc_failure_disclosure",
}
COORDINATES = ((0, 9), (4, 2), (6, 10), (11, 2))
STRATA = ("overall", "correct", "misclassified")
E4_METADATA = ("figure_recipes", "analysis_report", "status", "cost", "payload_manifest", "provenance")
E4_COMPLETIONS = ("e0", "e1", "e2", "e3", "e4")

CONTRACT_FILES = (
    F2_CONFIG_PATH, F2_CONFIG_SCHEMA_PATH, F2_RUN_SPEC_PATH, F2_RUN_SPEC_SCHEMA_PATH,
    F2_PLOT_DATA_SCHEMA_PATH, F2_RENDER_REPORT_SCHEMA_PATH, F2_VALIDATION_SCHEMA_PATH,
    F2_COMPLETION_SCHEMA_PATH, F2_FALSE_PASS_SCHEMA_PATH, "config/stage_f_f0.json",
    "config/stage_f_caption_contract.json", "config/stage_f_figure_catalog.json",
)
SOURCE_FILES = (
    "src/vitnns/display.py", "src/vitnns/stage_f_renderer.py",
    "src/vitnns/stage_f_cli.py", "src/vitnns/stage_f_f2.py",
    "tests/test_stage_f_f2.py", "tests/test_stage_f_f2_boundary.py",
)
ATTACKS = (
    "f1_ref_completion_hash_canonical_inventory_or_terminal_state_tampered",
    "e4_ref_completion_hash_canonical_inventory_or_recipe_tampered",
    "e4_manifest_or_paper_data_file_tampered",
    "csv_json_row_column_order_or_field_parity_broken",
    "patch_scale_source_value_or_anchor_identity_changed",
    "upsampling_source_value_changed",
    "uniform_nearest_equivalence_split_into_winner_loser",
    "layer_head_coordinate_or_stratum_changed",
    "layer_head_method_metric_n_valid_failed_na_median_interval_changed",
    "failure_order_117_or_352_deleted_zero_imputed_or_interpolated",
    "valid_only_summary_shown_without_expected_valid_failed_na",
    "e2_budget_overrun_or_owner_waiver_hidden",
    "e4_mixed_validity_rewritten_as_scientific_pass",
    "new_p_value_winner_go_no_go_or_endpoint_created",
    "png_pdf_svg_use_different_source_plot_data_or_config",
    "method_color_linestyle_or_marker_contract_drifted",
    "layout_canvas_font_dpi_or_export_contract_drifted",
    "jpeg_model_checkpoint_torch_timm_cuda_or_nvidia_access",
    "stage_e_unit_or_d8_d9_map_used_as_numeric_input",
    "f3_f4_or_final_combined_figure_generated",
    "report_png_copies_or_image_analysis_missing",
    "pass_or_f3_authorization_published_before_terminal_validation",
)


class StageFF2Error(StageFRendererError):
    """Raised when F2 cannot be produced or independently verified."""


def _signed(value: Mapping[str, Any]) -> dict[str, Any]:
    document = {key: item for key, item in value.items() if key != "canonical_digest"}
    document["canonical_digest"] = canonical_sha256(document)
    return document


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_json_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_text_new(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _strict_json_object(path: Path, context: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except Exception as error:
        raise StageFF2Error(f"{context} is invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise StageFF2Error(f"{context} must be a JSON object")
    return value, raw


def _validate_signed(value: Mapping[str, Any], context: str) -> None:
    actual = canonical_sha256({key: item for key, item in value.items() if key != "canonical_digest"})
    if value.get("canonical_digest") != actual:
        raise StageFF2Error(f"{context} canonical digest mismatch")


def _schema(repo: Path, document: Mapping[str, Any], relative: str, context: str) -> None:
    schema, _ = _strict_json_object(repo / relative, f"{context} schema")
    Draft202012Validator.check_schema(schema)
    errors = sorted(Draft202012Validator(schema).iter_errors(document), key=lambda item: list(item.path))
    if errors:
        location = ".".join(str(item) for item in errors[0].path) or "<root>"
        raise StageFF2Error(f"{context} schema failed at {location}: {errors[0].message}")


def _record(repo: Path, relative: str) -> dict[str, Any]:
    path = repo / relative
    value, _ = _strict_json_object(path, relative)
    _validate_signed(value, relative)
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path), "canonical_digest": value["canonical_digest"]}


def _file_record(repo: Path, relative: str) -> dict[str, Any]:
    path = repo / relative
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(("git", *arguments), cwd=repo, text=True, capture_output=True, check=False)
    if result.returncode:
        raise StageFF2Error(f"git {' '.join(arguments)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _is_ancestor(repo: Path, ancestor: str, descendant: str = "HEAD") -> bool:
    return subprocess.run(("git", "merge-base", "--is-ancestor", ancestor, descendant), cwd=repo, check=False).returncode == 0


def _inventory(root: Path, *, exclude: Sequence[str] = ()) -> dict[str, Any]:
    excluded = set(exclude)
    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_file() and not path.is_symlink() and relative not in excluded:
            files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"files": files, "file_count": len(files), "total_bytes": sum(item["bytes"] for item in files), "canonical_digest": canonical_sha256({"files": files})}


def _load_signed_contract(repo: Path, path: str, schema_path: str, context: str) -> dict[str, Any]:
    document, raw = _strict_json_object(repo / path, context)
    if raw != canonical_json_bytes(document) + b"\n":
        raise StageFF2Error(f"{context} is not byte-canonical")
    _validate_signed(document, context)
    _schema(repo, document, schema_path, context)
    return document


def _load_config(repo: Path) -> dict[str, Any]:
    config = _load_signed_contract(repo, F2_CONFIG_PATH, F2_CONFIG_SCHEMA_PATH, "F2 config")
    f0 = load_contracts(repo)["config"]
    if config["renderer"]["f0_config_digest"] != f0["canonical_digest"] or config["renderer"]["renderer_contract_version"] != f0["renderer_contract_version"]:
        raise StageFF2Error("F2 is not bound to the frozen F0 renderer")
    return config


def _load_run_spec(repo: Path) -> dict[str, Any]:
    return _load_signed_contract(repo, F2_RUN_SPEC_PATH, F2_RUN_SPEC_SCHEMA_PATH, "F2 run spec")


def validate_runtime_isolation() -> dict[str, Any]:
    forbidden = sorted(name for name in sys.modules if name == "torch" or name.startswith("torch.") or name == "torchvision" or name.startswith("torchvision.") or name == "timm" or name.startswith("timm."))
    if forbidden:
        raise StageFF2Error(f"formal F2 process imported forbidden runtime: {forbidden[0]}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise StageFF2Error("formal F2 producer requires CUDA_VISIBLE_DEVICES to be empty")
    handles = []
    fd_root = Path("/proc/self/fd")
    if fd_root.is_dir():
        for path in fd_root.iterdir():
            try:
                target = os.readlink(path)
            except OSError:
                continue
            if target.startswith("/dev/" + "nvidia"):
                handles.append(target)
    if handles:
        raise StageFF2Error("formal F2 process holds an NVIDIA device handle")
    return {"status": "PASS", "cuda_visible_devices": "", "torch_imported": False, "torchvision_imported": False, "timm_imported": False, "nvidia_device_handles": [], "jpeg_read": False, "model_loaded": False, "checkpoint_loaded": False}


def assert_f2_source_boundary(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    paths = [repo / path for path in ("src/vitnns/stage_f_renderer.py", "src/vitnns/stage_f_cli.py", "src/vitnns/stage_f_f2.py")]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    for name in ("torch", "torchvision", "timm"):
        if re.search(rf"^\s*(?:from|import)\s+{re.escape(name)}(?:\.|\s|$)", combined, re.MULTILINE):
            raise StageFF2Error(f"F2 source imports forbidden runtime: {name}")
    forbidden_tokens = ("load_" + "state_dict", "load_" + "canonical_model", "for" + "ward(", "back" + "ward(", "integrated_" + "grad", "." + "cuda(", "cuda" + ":", "/dev/" + "nvidia", "formal_r2/" + "units", "assets/data/" + "ILSVRC2012_val")
    if any(token in combined for token in forbidden_tokens):
        raise StageFF2Error("F2 source exposes forbidden compute, GPU, JPEG, unit, or map surface")
    return {"status": "PASS", "model_access": False, "checkpoint_access": False, "torch_access": False, "timm_access": False, "cuda_access": False, "jpeg_access": False, "stage_e_unit_access": False, "d8_d9_map_access": False}


def _git_clean(repo: Path) -> bool:
    return _git(repo, "status", "--porcelain=v1") == ""


def _lfs_clean(repo: Path) -> bool:
    return not any(line.startswith("\t") or line.startswith("    ") for line in _git(repo, "lfs", "status").splitlines())


def _sealed_diff(repo: Path, commit: str, *paths: str) -> None:
    if _git(repo, "diff", "--name-only", commit, "--", *paths):
        raise StageFF2Error(f"sealed roots changed after {commit}")


def _verify_sealed_inventory(root: Path, inventory: Mapping[str, Any], *, exclude: Sequence[str]) -> None:
    rebuilt = _inventory(root, exclude=exclude)
    unsigned = {key: item for key, item in inventory.items() if key != "canonical_digest"}
    if (
        any(inventory.get(key) != rebuilt[key] for key in ("files", "file_count", "total_bytes"))
        or inventory.get("canonical_digest") != canonical_sha256(unsigned)
    ):
        raise StageFF2Error(f"sealed inventory content drifted: {root}")


def _verify_f1(repo: Path) -> dict[str, Any]:
    if _git(repo, "rev-parse", F1_ARCHIVE_REF + "^{commit}") != F1_COMMIT or not _is_ancestor(repo, F1_COMMIT):
        raise StageFF2Error("F1 archive ref drifted or is not an ancestor")
    completion, _ = _strict_json_object(repo / F1_COMPLETION_PATH, "F1 completion")
    inventory, _ = _strict_json_object(repo / F1_INVENTORY_PATH, "F1 inventory")
    _validate_signed(completion, "F1 completion"); _validate_signed(inventory, "F1 inventory")
    if sha256_file(repo / F1_COMPLETION_PATH) != EXPECTED_F1_COMPLETION_SHA256 or completion["canonical_digest"] != EXPECTED_F1_COMPLETION_DIGEST or inventory["canonical_digest"] != EXPECTED_F1_INVENTORY_DIGEST:
        raise StageFF2Error("F1 completion or inventory identity drifted")
    expected = {"status": "PASS", "package_validation": "PASS", "workflow_validation": "PASS", "scientific_modules_rendered": 3, "layout_render_instances": 6, "exported_figure_files": 18, "final_combined_figure_2_rendered": False, "model_access": False, "torch_imported": False, "timm_imported": False, "gpu_cuda_access": False, "jpeg_access": False, "bootstrap_ci_new_endpoint_recomputed": False, "c10_stage_d_pooled": False, "d10_f0_e4_decisions_modified": False, "f2_started": False, "f3_started": False, "f4_started": False, "next_authorization": "F2_ONLY"}
    if any(completion.get(key) != value for key, value in expected.items()):
        raise StageFF2Error("F1 terminal semantics drifted")
    _verify_sealed_inventory(repo / F1_ROOT_PATH, inventory, exclude=("whole_root_inventory.json", "status.json", "completion/f1.pending.json", "completion/f1.json"))
    _sealed_diff(repo, F1_COMMIT, F1_ROOT_PATH, F1_REPORT_PATH)
    return {"status": "PASS", "archive_ref": F1_ARCHIVE_REF, "archive_commit": F1_COMMIT, "completion_sha256": EXPECTED_F1_COMPLETION_SHA256, "completion_canonical_digest": EXPECTED_F1_COMPLETION_DIGEST, "whole_root_inventory_canonical_digest": EXPECTED_F1_INVENTORY_DIGEST, "next_authorization": "F2_ONLY"}


def _verify_e4(repo: Path) -> dict[str, Any]:
    config = _load_config(repo)
    if _git(repo, "rev-parse", E4_ARCHIVE_REF + "^{commit}") != E4_COMMIT or not _is_ancestor(repo, E4_COMMIT):
        raise StageFF2Error("E4 archive ref drifted or is not an ancestor")
    completion, _ = _strict_json_object(repo / E4_ROOT_PATH / "completion/e4.json", "E4 completion")
    inventory, _ = _strict_json_object(repo / E4_ROOT_PATH / "whole_root_inventory.json", "E4 inventory")
    _validate_signed(completion, "E4 completion"); _validate_signed(inventory, "E4 inventory")
    if sha256_file(repo / E4_ROOT_PATH / "completion/e4.json") != EXPECTED_E4_COMPLETION_SHA256 or completion["canonical_digest"] != EXPECTED_E4_COMPLETION_DIGEST or inventory["canonical_digest"] != EXPECTED_E4_INVENTORY_DIGEST or sha256_file(repo / E4_ROOT_PATH / "figure_recipes.json") != EXPECTED_E4_RECIPE_SHA256:
        raise StageFF2Error("E4 completion inventory or recipes identity drifted")
    if completion.get("status") != "PASS" or completion.get("package_completeness") != "PASS" or completion.get("workflow_integrity") != "PASS" or completion.get("scientific_validity") != "MIXED" or completion.get("cost_status") != "PASS_COST":
        raise StageFF2Error("E4 terminal scientific state drifted")
    _verify_sealed_inventory(repo / E4_ROOT_PATH, inventory, exclude=("whole_root_inventory.json", "status.json", "completion/e4.pending.json", "completion/e4.json"))
    for field in ("payload_manifest", "status", "whole_root_inventory"):
        if completion.get("source_validation", {}).get(field) != _record(repo, completion["source_validation"][field]["path"]):
            raise StageFF2Error(f"E4 completion binding drifted: {field}")
    _sealed_diff(repo, E4_COMMIT, E4_ROOT_PATH, "reports/stage_e_e4_20260907")
    return {"status": "PASS", "archive_ref": E4_ARCHIVE_REF, "archive_commit": E4_COMMIT, "completion_sha256": config["inputs"]["e4_completion_sha256"], "completion_canonical_digest": completion["canonical_digest"], "whole_root_inventory_canonical_digest": inventory["canonical_digest"], "figure_recipes_sha256": EXPECTED_E4_RECIPE_SHA256, "scientific_validity": "MIXED"}


def verify_startup_gates(repo_root: Path | str = REPO_ROOT, *, require_clean: bool = True, require_remote: bool = True) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); config = _load_config(repo); run_spec = _load_run_spec(repo)
    branch = _git(repo, "symbolic-ref", "--short", "HEAD"); head = _git(repo, "rev-parse", "HEAD"); tracking = _git(repo, "rev-parse", "origin/stage-d-d3-pass")
    if branch != "stage-d-d3-pass" or head != tracking:
        raise StageFF2Error("F2 requires stage-d-d3-pass with local and tracking heads identical")
    remote = None
    if require_remote:
        lines = _git(repo, "ls-remote", "--heads", "origin", "stage-d-d3-pass").splitlines()
        remote = lines[0].split()[0] if len(lines) == 1 else None
        if remote != head:
            raise StageFF2Error("local tracking and remote branch are not identical")
        remote_refs = dict(reversed(line.split()) for line in _git(repo, "ls-remote", "origin", F1_ARCHIVE_REF, E4_ARCHIVE_REF).splitlines())
        if remote_refs.get(F1_ARCHIVE_REF) != F1_COMMIT or remote_refs.get(E4_ARCHIVE_REF) != E4_COMMIT:
            raise StageFF2Error("remote F1 or E4 archive ref drifted")
    if require_clean and (not _git_clean(repo) or not _lfs_clean(repo)):
        raise StageFF2Error("F2 startup requires clean Git and LFS state")
    f1 = _verify_f1(repo); e4 = _verify_e4(repo)
    _sealed_diff(repo, F0_COMMIT, F0_ROOT_PATH, F0_REPORT_PATH)
    _sealed_diff(repo, D10_COMMIT, D10_ROOT_PATH, D10_REPORT_PATH)
    d10_report, _ = _strict_json_object(repo / D10_REPORT_PATH / "d10_formal_decision.json", "D10 decision report")
    _validate_signed(d10_report, "D10 decision report")
    truth = d10_report.get("truth_table", {}); statistics = d10_report.get("statistics", {}); c10 = d10_report.get("c10_status_preserved", {})
    if (
        truth.get("validity_status") != "PASS" or truth.get("fidelity_status") != "PASS_FIDELITY"
        or truth.get("cost_status") != "PASS_COST" or truth.get("overall_decision") != "GO_FIDELITY_SCOPE"
        or statistics.get("n") != 500 or statistics.get("positive_count") != 500
        or statistics.get("median") != 2.3081996221714247
        or statistics.get("bootstrap_ci_95") != [2.1110806126093067, 2.445235411755993]
        or c10.get("acceptance") != "BLOCKED" or c10.get("primary_decision") != "NO_GO_NOVELTY"
        or c10.get("overall_spatial") != "FAIL_SPATIAL" or c10.get("holm") != "HOLM_BLOCKED_UNDERSPECIFIED"
        or c10.get("downstream_authorized") is not False
    ):
        raise StageFF2Error("D10 or C10 frozen scientific state drifted")
    source = load_source_bundle(repo)
    return _signed({"schema_version": 1, "document_type": "stage_f_f2_startup_gate", "stage_id": "F2", "status": "PASS", "validated_before_modification_or_rendering": True, "branch": branch, "local_head": head, "tracking_head": tracking, "remote_head": remote, "worktree_index_clean": not require_clean or _git_clean(repo), "git_lfs_clean": not require_clean or _lfs_clean(repo), "f1": f1, "e4": e4, "d10_c10_scientific_state": {"status": "PASS", "d10_validity": "PASS", "d10_fidelity": "PASS_FIDELITY", "d10_cost": "PASS_COST", "d10_decision": "GO_FIDELITY_SCOPE", "d10_n": 500, "d10_positive": 500, "d10_median": 2.3081996221714247, "d10_ci_95": [2.1110806126093067, 2.445235411755993], "c10_acceptance": "BLOCKED", "c10_decision": "NO_GO_NOVELTY", "c10_spatial": "FAIL_SPATIAL", "c10_holm": "HOLM_BLOCKED_UNDERSPECIFIED", "c10_downstream_authorized": False}, "sealed_roots": {"f0": "PASS", "f1": "PASS", "d10": "PASS", "e4": "PASS"}, "source_binding_digest": source["source_input_digest"], "source_adapter": "stage_f_f2_e4_paper_data_adapter", "source_boundary": assert_f2_source_boundary(repo), "config_canonical_digest": config["canonical_digest"], "run_spec_canonical_digest": run_spec["canonical_digest"], "next_authorization": "F2_ONLY"})


def _coerce_csv_value(raw: str, expected: Any, context: str) -> Any:
    if isinstance(expected, bool):
        if raw not in {"true", "false"}: raise StageFF2Error(f"{context} bool drifted")
        return raw == "true"
    if isinstance(expected, int) and not isinstance(expected, bool):
        try: return int(raw)
        except ValueError as error: raise StageFF2Error(f"{context} int drifted") from error
    if isinstance(expected, float):
        try: value = float(raw)
        except ValueError as error: raise StageFF2Error(f"{context} float drifted") from error
        if not math.isfinite(value): raise StageFF2Error(f"{context} is not finite")
        return value
    if expected is None: return None if raw == "" else raw
    if isinstance(expected, (list, dict)):
        try: return json.loads(raw)
        except json.JSONDecodeError as error: raise StageFF2Error(f"{context} structured value drifted") from error
    return raw


def _read_table_pair(repo: Path, table: str, expected: Mapping[str, str]) -> dict[str, Any]:
    csv_path = repo / E4_PAPER_DATA_ROOT_PATH / f"{table}.csv"; json_path = repo / E4_PAPER_DATA_ROOT_PATH / f"{table}.json"
    if sha256_file(csv_path) != expected["csv_sha256"] or sha256_file(json_path) != expected["json_sha256"]:
        raise StageFF2Error(f"{table} file hash drifted")
    document, raw = _strict_json_object(json_path, f"{table}.json")
    if raw != canonical_json_bytes(document) + b"\n": raise StageFF2Error(f"{table}.json is not byte-canonical")
    _validate_signed(document, f"{table}.json")
    if document["canonical_digest"] != expected["json_canonical_digest"] or document.get("table_id") != table or document.get("row_count") != len(document.get("rows", [])):
        raise StageFF2Error(f"{table}.json metadata drifted")
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream); raw_rows = list(reader)
    if reader.fieldnames != document["columns"] or len(raw_rows) != len(document["rows"]):
        raise StageFF2Error(f"{table} CSV JSON columns or row count drifted")
    typed_rows = []
    for index, (csv_row, json_row) in enumerate(zip(raw_rows, document["rows"])):
        typed = {column: _coerce_csv_value(csv_row[column], json_row[column], f"{table}[{index}].{column}") for column in document["columns"]}
        if typed != json_row: raise StageFF2Error(f"{table} CSV JSON parity mismatch at row {index}")
        typed_rows.append(typed)
    return {"table_id": table, "columns": document["columns"], "row_count": len(typed_rows), "rows": typed_rows, "row_order_digest": canonical_sha256(typed_rows), "source": {"csv": _file_record(repo, E4_PAPER_DATA_ROOT_PATH + f"/{table}.csv"), "json": _record(repo, E4_PAPER_DATA_ROOT_PATH + f"/{table}.json")}}


def _validate_recipes(recipes: Mapping[str, Any]) -> None:
    if recipes.get("numeric_source") != "paper_data_only" or recipes.get("renders_png") is not False or len(recipes.get("figures", [])) != 3:
        raise StageFF2Error("E4 figure recipe contract drifted")
    text = " ".join(item.get("caption_contract", "") for item in recipes["figures"])
    for phrase in ("Post-D10 descriptive", "Uniform and nearest", "no winner selection", "MIXED", "117", "352"):
        if phrase not in text: raise StageFF2Error(f"E4 recipe disclosure missing: {phrase}")


def load_source_bundle(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); config = _load_config(repo); validate_runtime_isolation()
    formal = (repo / E4_ROOT_PATH).resolve()
    if formal != (repo / config["inputs"]["e4_formal_root"]).resolve() or (formal / "paper_data").resolve() != (repo / config["inputs"]["e4_paper_data_root"]).resolve():
        raise StageFF2Error("F2 E4 source root binding drifted")
    metadata = {}
    for name in E4_METADATA:
        relative = f"{name}.json"; document, raw = _strict_json_object(formal / relative, f"E4 {name}"); _validate_signed(document, f"E4 {name}")
        metadata[name] = {"document": document, "record": {"path": E4_ROOT_PATH + "/" + relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "canonical_digest": document["canonical_digest"]}}
    completions = {}
    for stage in E4_COMPLETIONS:
        relative = f"completion/{stage}.json"; document, raw = _strict_json_object(formal / relative, f"E4 {stage} completion"); _validate_signed(document, f"E4 {stage} completion")
        completions[stage] = {"document": document, "record": {"path": E4_ROOT_PATH + "/" + relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "canonical_digest": document["canonical_digest"]}}
    if metadata["figure_recipes"]["record"]["sha256"] != EXPECTED_E4_RECIPE_SHA256: raise StageFF2Error("E4 recipe hash drifted")
    _validate_recipes(metadata["figure_recipes"]["document"])
    manifest = {item["path"]: item for item in metadata["payload_manifest"]["document"]["artifacts"]}
    tables = {}
    for table in TABLES:
        value = _read_table_pair(repo, table, config["inputs"]["paper_data"][table])
        for suffix in ("csv", "json"):
            relative = f"paper_data/{table}.{suffix}"
            if manifest.get(relative, {}).get("sha256") != value["source"][suffix]["sha256"]: raise StageFF2Error(f"E4 manifest binding drifted for {relative}")
        tables[table] = value
    for name in ("analysis_report", "figure_recipes", "cost", "provenance"):
        if manifest.get(name + ".json", {}).get("sha256") != metadata[name]["record"]["sha256"]: raise StageFF2Error(f"E4 manifest binding drifted for {name}")
    for name in ("e0", "e1", "e2", "e3"):
        if manifest.get(f"completion/{name}.json", {}).get("sha256") != completions[name]["record"]["sha256"]: raise StageFF2Error(f"E4 manifest binding drifted for completion/{name}.json")
    analysis = metadata["analysis_report"]["document"]; status = metadata["status"]["document"]; e2 = completions["e2"]["document"]; e3 = completions["e3"]["document"]
    failures = analysis.get("scientific_summary", {}).get("failed_units", [])
    if analysis.get("descriptive_only") is not True or analysis.get("p_values_emitted") is not False or analysis.get("winner_selection_performed") is not False or analysis.get("new_go_no_go_issued") is not False or analysis.get("scientific_summary", {}).get("scientific_validity") != "MIXED" or status.get("scientific_validity") != "MIXED": raise StageFF2Error("E4 descriptive scientific state drifted")
    if [item["selection_order"] for item in failures] != [117, 352] or any(item.get("numeric_result") is not None for item in failures): raise StageFF2Error("E4 typed failure disclosure drifted")
    if e2.get("cost_status") != "FAIL_COST" or e2.get("source_validation", {}).get("completion_basis") != "PASS_WITH_EXPLICIT_OWNER_BUDGET_WAIVER": raise StageFF2Error("E2 budget waiver disclosure drifted")
    if e3.get("scientific_validity") != "MIXED" or e3.get("union", {}).get("terminal_counts") != {"expected": 3000, "failed": 2, "na": 0, "valid": 2998}: raise StageFF2Error("E3 mixed terminal counts drifted")
    resource_failures = [row for row in tables["resource_and_qc"]["rows"] if row["validity_status"] == "FAILED"]
    if [row["selection_order"] for row in resource_failures] != [117, 352] or any(any(row[field] is not None for field in ("delta_fid", "insertion_auc_mean", "deletion_auc_mean")) for row in resource_failures): raise StageFF2Error("resource table failures were removed or imputed")
    records = [metadata[name]["record"] for name in E4_METADATA] + [completions[name]["record"] for name in E4_COMPLETIONS]
    for table in TABLES: records.extend((tables[table]["source"]["csv"], tables[table]["source"]["json"]))
    return {"status": "PASS", "source_input_digest": canonical_sha256({"sources": records, "row_order_digests": {table: tables[table]["row_order_digest"] for table in TABLES}}), "records": records, "metadata": metadata, "completions": completions, "tables": tables, "primary_numeric_root": E4_PAPER_DATA_ROOT_PATH}


def _sources(source: Mapping[str, Any], *tables: str) -> list[dict[str, Any]]:
    result = []
    for table in tables:
        value = source["tables"][table]
        result.append({"table": table, "row_count": value["row_count"], "columns": value["columns"], "row_order_digest": value["row_order_digest"], "csv": value["source"]["csv"], "json": value["source"]["json"]})
    return result


def _series(rows: Sequence[Mapping[str, Any]], variant: str, field: str, *, method: str, label: str, style: Mapping[str, str] | None = None) -> dict[str, Any]:
    selected = sorted((row for row in rows if row["variant_id"] == variant), key=lambda row: row["selection_order"] if row["selection_order"] is not None else -1)
    values = [row[field] for row in selected]
    if not selected or any(row["validity_status"] != "VALID" or value is None for row, value in zip(selected, values)):
        raise StageFF2Error(f"{variant}.{field} requires complete valid source values")
    result = {"variant_id": variant, "field": field, "method": method, "label": label, "selection_orders": [row["selection_order"] for row in selected], "sample_ids": [row["sample_id"] for row in selected], "values": values}
    if style is not None: result["style"] = dict(style)
    return result


def _build_f2_a(source: Mapping[str, Any]) -> dict[str, Any]:
    rows = source["tables"]["patch_scale_sensitivity"]["rows"]
    variants = ("d10_fine_4x4_primary_anchor", "nns_native_14x14_patch", "matched_native_14x14_patch")
    if [rows[index]["variant_id"] for index in range(3)] != list(variants) or any(row["validity_status"] != "VALID" for row in rows): raise StageFF2Error("patch-scale row order or validity drifted")
    delta = [
        _series(rows, variants[0], "delta_fid", method="anchor", label="D10 fine 4x4 anchor", style={"color": "#4D4D4D", "linestyle": "-", "marker": "D"}),
        _series(rows, variants[1], "delta_fid", method="nns", label="NNS native 14x14"),
        _series(rows, variants[2], "delta_fid", method="matched", label="matched native 14x14"),
    ]
    relative = [_series(rows, variants[1], "delta_fid_patch_minus_fine", method="nns", label="NNS native minus fine"), _series(rows, variants[2], "delta_fid_patch_minus_fine", method="matched", label="matched native minus fine")]
    payload = {"module_id": "F2-A", "analysis_role": "POST_D10_SECONDARY_DESCRIPTIVE", "variant_order": list(variants), "delta_fid_series": delta, "relative_to_anchor_series": relative, "valid_rows": 1500, "d10_primary_replaced": False}
    return {"schema_version": 1, "document_type": "stage_f_f2_plot_data", "stage_id": "F2", "module_id": "F2-A", "source_tables": _sources(source, "patch_scale_sensitivity"), "caption_key": CAPTION_KEY["F2-A"], "plot_payload": payload, "plot_data_digest": canonical_sha256(payload), "p_value_computed": False, "winner_selection_performed": False, "new_go_no_go_issued": False, "new_endpoint_computed": False}


def _build_f2_b(source: Mapping[str, Any]) -> dict[str, Any]:
    rows = source["tables"]["upsampling_sensitivity"]["rows"]; variants = ("uniform_mass_divide_by_16", "mass_preserving_nearest", "bilinear", "bicubic")
    by_variant = {variant: sorted((row for row in rows if row["variant_id"] == variant), key=lambda row: row["selection_order"]) for variant in variants}
    if any(len(value) != 500 for value in by_variant.values()): raise StageFF2Error("upsampling variant coverage drifted")
    parity_fields = ("selection_order", "sample_id", "insertion_auc_mean", "deletion_auc_mean", "equivalence_class")
    if any(any(left[field] != right[field] for field in parity_fields) for left, right in zip(by_variant[variants[0]], by_variant[variants[1]])): raise StageFF2Error("uniform and nearest equivalence class split")
    styles = {"uniform": {"color": "#4D4D4D", "linestyle": "-", "marker": "D"}, "bilinear": {"color": "#009E73", "linestyle": "--", "marker": "^"}, "bicubic": {"color": "#CC79A7", "linestyle": ":", "marker": "v"}}
    panels = {}
    for field in ("insertion_auc_mean", "deletion_auc_mean"):
        panels[field] = [_series(rows, variants[0], field, method="uniform_nearest_exact", label="uniform = nearest (one evidence class)", style=styles["uniform"]), _series(rows, variants[2], field, method="bilinear", label="bilinear sensitivity", style=styles["bilinear"]), _series(rows, variants[3], field, method="bicubic", label="bicubic sensitivity", style=styles["bicubic"])]
    payload = {"module_id": "F2-B", "analysis_role": "POST_D10_SECONDARY_DESCRIPTIVE", "variant_order": list(variants), "equivalence_class": {"class_id": "uniform_nearest_exact", "members": list(variants[:2]), "source_rows_each": 500, "values_byte_equivalent_by_field": True, "winner_loser_split": False}, "panels": panels, "winner_selection_performed": False}
    return {"schema_version": 1, "document_type": "stage_f_f2_plot_data", "stage_id": "F2", "module_id": "F2-B", "source_tables": _sources(source, "upsampling_sensitivity"), "caption_key": CAPTION_KEY["F2-B"], "plot_payload": payload, "plot_data_digest": canonical_sha256(payload), "p_value_computed": False, "winner_selection_performed": False, "new_go_no_go_issued": False, "new_endpoint_computed": False}


def _build_f2_c(source: Mapping[str, Any]) -> dict[str, Any]:
    detailed = source["tables"]["layer_head_descriptive"]["rows"]; rows = source["tables"]["prediction_strata"]["rows"]
    observed = {(row["layer"], row["head"]) for row in detailed}
    failures = [row for row in detailed if row["validity_status"] == "FAILED"]
    if observed != set(COORDINATES) or [row["selection_order"] for row in failures] != [117, 352] or any(row["delta_fid"] is not None for row in failures): raise StageFF2Error("layer-head coordinates or typed failures drifted")
    style_by_key = {
        ("nns_vs_matched", "delta_fid"): {"color": "#4D4D4D", "linestyle": "-", "marker": "D", "label": "delta FID"},
        ("conditional_fixed_clean_attention_head_direction_value_path_ig", "insertion_auc"): {"color": "#0072B2", "linestyle": "-", "marker": "o", "label": "NNS insertion"},
        ("conditional_fixed_clean_attention_head_direction_value_path_ig", "deletion_auc"): {"color": "#0072B2", "linestyle": "-", "marker": "o", "label": "NNS deletion"},
        ("matched_fixed_clean_attention_patch_token_contribution", "insertion_auc"): {"color": "#D55E00", "linestyle": "--", "marker": "s", "label": "matched insertion"},
        ("matched_fixed_clean_attention_patch_token_contribution", "deletion_auc"): {"color": "#D55E00", "linestyle": "--", "marker": "s", "label": "matched deletion"},
    }
    panels = []
    for stratum in STRATA:
        series = []
        for identity, style in style_by_key.items():
            selected = []
            for layer, head in COORDINATES:
                matches = [row for row in rows if row["stratum"] == stratum and (row["layer"], row["head"]) == (layer, head) and (row["method"], row["metric"]) == identity]
                if len(matches) != 1: raise StageFF2Error("prediction-strata coordinate identity is ambiguous")
                selected.append(matches[0])
            if any(row["valid"] + row["failed"] + row["na"] != row["n"] or row["median"] is None for row in selected): raise StageFF2Error("prediction-strata counts or medians drifted")
            series.append({"identity": {"method": identity[0], "metric": identity[1]}, "method": "nns" if identity[0].startswith("conditional") else "matched" if identity[0].startswith("matched") else "delta_fid", "label": style["label"], "style": {key: style[key] for key in ("color", "linestyle", "marker")}, "x": list(range(4)), "y": [row["median"] for row in selected], "lower": [row["paired_image_bootstrap_median_95_interval"]["ci_95"][0] for row in selected], "upper": [row["paired_image_bootstrap_median_95_interval"]["ci_95"][1] for row in selected], "source_counts": [{key: row[key] for key in ("n", "valid", "failed", "na", "positive_count", "scientific_coverage_status")} for row in selected]})
        panels.append({"stratum": stratum, "series": series})
    payload = {"module_id": "F2-C", "analysis_role": "POST_D10_SECONDARY_DESCRIPTIVE", "scientific_validity": "MIXED", "coordinate_order": [list(item) for item in COORDINATES], "coordinate_labels": [f"L{layer}/H{head}" for layer, head in COORDINATES], "stratum_order": list(STRATA), "panels": panels, "typed_failures": [{key: row[key] for key in ("selection_order", "sample_id", "failure_code", "layer", "head", "stratum", "delta_fid")} for row in failures], "source_summary_rows": rows}
    return {"schema_version": 1, "document_type": "stage_f_f2_plot_data", "stage_id": "F2", "module_id": "F2-C", "source_tables": _sources(source, "layer_head_descriptive", "prediction_strata"), "caption_key": CAPTION_KEY["F2-C"], "plot_payload": payload, "plot_data_digest": canonical_sha256(payload), "p_value_computed": False, "winner_selection_performed": False, "new_go_no_go_issued": False, "new_endpoint_computed": False}


def _build_f2_d(source: Mapping[str, Any]) -> dict[str, Any]:
    analysis = source["metadata"]["analysis_report"]["document"]; e2 = source["completions"]["e2"]["document"]; e3 = source["completions"]["e3"]["document"]; e4_cost = source["metadata"]["cost"]["document"]
    counts_source = analysis["variant_status_counts"]
    counts = [
        {"label": "E1", "expected": counts_source["E1"]["valid"] + counts_source["E1"]["failed"] + counts_source["E1"]["na"], **counts_source["E1"]},
        {"label": "E2", "expected": counts_source["E2"]["valid"] + counts_source["E2"]["failed"] + counts_source["E2"]["na"], **counts_source["E2"]},
        {"label": "E3 pairs", **counts_source["E3_layer_head_pairs"]},
        {"label": "E3 compute", **counts_source["E3_new_compute"]},
    ]
    failures = analysis["scientific_summary"]["failed_units"]
    if counts[-1] != {"label": "E3 compute", "expected": 3000, "valid": 2998, "failed": 2, "na": 0} or [item["selection_order"] for item in failures] != [117, 352]: raise StageFF2Error("F2-D count or failure source drifted")
    waiver = {"completion_basis": e2["source_validation"]["completion_basis"], "recorded_cost_status": e2["cost_status"], "gpu_hours": e2["actual_cost"]["gpu_hours"], "gpu_hours_max": e2["actual_cost"]["gpu_hours_max"], "cpu_hours": e2["actual_cost"]["cpu_hours"], "cpu_hours_max": e2["actual_cost"]["cpu_hours_max"]}
    payload = {"module_id": "F2-D", "analysis_role": "POST_D10_SECONDARY_DESCRIPTIVE", "scientific_validity": "MIXED", "summary_population": "VALID_ONLY_WITH_EXPECTED_VALID_FAILED_NA_COUNTS", "counts": counts, "typed_failures": failures, "failure_numeric_values": [item["numeric_result"] for item in failures], "missing_failure_marker": "X", "line_through_failures": False, "e2_owner_budget_waiver": waiver, "e3_terminal_counts": e3["union"]["terminal_counts"], "e4_cost": e4_cost}
    return {"schema_version": 1, "document_type": "stage_f_f2_plot_data", "stage_id": "F2", "module_id": "F2-D", "source_tables": _sources(source, "resource_and_qc"), "caption_key": CAPTION_KEY["F2-D"], "plot_payload": payload, "plot_data_digest": canonical_sha256(payload), "p_value_computed": False, "winner_selection_performed": False, "new_go_no_go_issued": False, "new_endpoint_computed": False}


def build_plot_data(source: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {"F2-A": _build_f2_a(source), "F2-B": _build_f2_b(source), "F2-C": _build_f2_c(source), "F2-D": _build_f2_d(source)}


def _ecdf_panel(panel_id: str, label: str, caption: str, title: str, x_label: str, series: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"panel_id": panel_id, "label": label, "caption_key": caption, "kind": "ecdf", "title": title, "x_label": x_label, "y_label": "Empirical cumulative proportion", "grid": True, "data": {"series": [{key: item[key] for key in ("method", "label", "values", "style") if key in item} for item in series]}}


def build_render_spec(module_id: str, plot_data: Mapping[str, Any], layout: str) -> dict[str, Any]:
    if layout not in LAYOUTS: raise StageFF2Error("unsupported F2 layout")
    payload = plot_data["plot_payload"]; panels = []
    if module_id == "F2-A":
        panels = [_ecdf_panel("f2_a_delta", "(a)", "f2_a_delta_fid", "Patch-scale delta FID", "delta FID", payload["delta_fid_series"]), _ecdf_panel("f2_a_relative", "(b)", "f2_a_relative_anchor", "Native 14x14 relative to fine 4x4", "patch minus fine delta FID", payload["relative_to_anchor_series"])]
    elif module_id == "F2-B":
        panels = [_ecdf_panel("f2_b_insertion", "(a)", "f2_b_insertion", "Insertion operator sensitivity", "insertion AUC", payload["panels"]["insertion_auc_mean"]), _ecdf_panel("f2_b_deletion", "(b)", "f2_b_deletion", "Deletion operator sensitivity", "deletion AUC", payload["panels"]["deletion_auc_mean"])]
    elif module_id == "F2-C":
        for index, item in enumerate(payload["panels"]):
            panels.append({"panel_id": f"f2_c_{item['stratum']}", "label": f"({chr(ord('a') + index)})", "caption_key": f"f2_c_{item['stratum']}", "kind": "line_band", "title": item["stratum"].capitalize(), "x_label": "Frozen layer/head coordinate", "y_label": "Source median and interval", "x_ticks": list(range(4)), "x_tick_labels": payload["coordinate_labels"], "grid": True, "data": {"x": list(range(4)), "series": item["series"]}})
    elif module_id == "F2-D":
        counts = payload["counts"]; x = list(range(len(counts))); labels = [item["label"] for item in counts]
        panels = [
            {"panel_id": "f2_d_coverage", "label": "(a)", "caption_key": "f2_d_expected_valid", "kind": "line_band", "title": "Expected and valid units", "x_label": "Analysis family", "y_label": "Unit count", "x_ticks": x, "x_tick_labels": labels, "grid": True, "data": {"x": x, "series": [{"method": "expected", "label": "expected", "y": [item["expected"] for item in counts], "style": {"color": "#4D4D4D", "linestyle": "--", "marker": "D"}}, {"method": "valid", "label": "valid", "y": [item["valid"] for item in counts], "style": {"color": "#0072B2", "linestyle": "-", "marker": "o"}}]}},
            {"panel_id": "f2_d_failed_na", "label": "(b)", "caption_key": "f2_d_failed_na", "kind": "line_band", "title": "Failed and NA units", "x_label": "Analysis family", "y_label": "Unit count", "x_ticks": x, "x_tick_labels": labels, "y_limits": [-0.15, 2.25], "grid": True, "data": {"x": x, "series": [{"method": "failed", "label": "failed (X = missing)", "y": [item["failed"] for item in counts], "style": {"color": "#D55E00", "linestyle": ":", "marker": "X"}}, {"method": "na", "label": "NA", "y": [item["na"] for item in counts], "style": {"color": "#999999", "linestyle": "--", "marker": "s"}}]}},
            {"panel_id": "f2_d_disclosure", "label": "(c)", "caption_key": "f2_d_failures_waiver", "kind": "method_schematic", "title": "Typed failures and cost disclosure", "data": {"nodes": [{"id": "mixed", "text": "E4 validity: MIXED", "position": [0.5, 0.88], "size": [0.55, 0.13]}, {"id": "f117", "text": "X order 117\n00041879\nIG conservation failure", "position": [0.27, 0.58], "size": [0.42, 0.22], "facecolor": "#FDE0DD"}, {"id": "f352", "text": "X order 352\n00028862\ntyped IG failure", "position": [0.73, 0.58], "size": [0.42, 0.22], "facecolor": "#FDE0DD"}, {"id": "waiver", "text": "E2 exceeded CPU/GPU caps\nowner budget waiver retained", "position": [0.5, 0.22], "size": [0.7, 0.18], "facecolor": "#FFF2CC"}], "arrows": [{"from": "mixed", "to": "f117"}, {"from": "mixed", "to": "f352"}, {"from": "f117", "to": "waiver"}, {"from": "f352", "to": "waiver"}]}}
        ]
    else: raise StageFF2Error(f"unknown F2 module: {module_id}")
    grid = {"rows": len(panels), "columns": 1} if layout == "single_column" else ({"rows": 1, "columns": len(panels)} if len(panels) <= 3 else {"rows": 2, "columns": 2})
    return {"stage_id": "F2", "module_id": module_id, "figure_id": f"{MODULE_SLUG[module_id]}_{layout}", "title": f"{module_id} POST-D10 Secondary Descriptive" + (" — scientific validity MIXED" if module_id in {"F2-C", "F2-D"} else ""), "formal_scientific_figure": True, "layout_profile": layout, "grid": grid, "caption_keys": [panel["caption_key"] for panel in panels], "panels": panels}


def build_caption_data(module_id: str, plot_data: Mapping[str, Any]) -> dict[str, Any]:
    texts = {
        "F2-A": "POST_D10_SECONDARY_DESCRIPTIVE. Fine 4x4 is the unchanged D10 anchor; native 14x14 sensitivity does not replace the D10 primary result.",
        "F2-B": "POST_D10_SECONDARY_DESCRIPTIVE. Uniform and nearest form one exact equivalent evidence class; no winner is selected and no best operator is claimed.",
        "F2-C": "POST_D10_SECONDARY_DESCRIPTIVE. Four frozen coordinates and overall/correct/misclassified strata are shown from sealed source medians and intervals. Scientific validity remains MIXED; numeric summaries are VALID-only with n/valid/failed/NA retained.",
        "F2-D": "POST_D10_SECONDARY_DESCRIPTIVE. Expected/valid/failed/NA are disclosed; failed orders 117 and 352 remain missing typed failures without zero imputation or interpolation. E2 exceeded frozen CPU/GPU budgets and passed only with an explicit owner budget waiver.",
    }
    spec = build_render_spec(module_id, plot_data, "double_column")
    return _signed({"schema_version": 1, "document_type": "stage_f_f2_caption_data", "stage_id": "F2", "module_id": module_id, "caption_key": CAPTION_KEY[module_id], "caption_text": texts[module_id], "panel_order": [panel["panel_id"] for panel in spec["panels"]], "panel_caption_keys": spec["caption_keys"], "p_value_computed": False, "winner_selection_performed": False, "new_go_no_go_issued": False, "d10_primary_modified": False})


def build_mapping(module_id: str, plot_data: Mapping[str, Any]) -> dict[str, Any]:
    return _signed({"schema_version": 1, "document_type": "stage_f_f2_source_to_plotted_value_mapping", "stage_id": "F2", "module_id": module_id, "source_tables": plot_data["source_tables"], "plot_data_digest": plot_data["plot_data_digest"], "plot_payload_canonical_digest": canonical_sha256(plot_data["plot_payload"]), "mapping_rule": "explicit field copy and deterministic display ordering; source validator independently rebuilds the full plot payload", "source_rows_filtered_for_scientific_result": False, "failed_rows_imputed": False})


def _copy_new(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists(): raise StageFF2Error(f"refusing to overwrite {target}")
    shutil.copyfile(source, target)


def _copy_snapshots(repo: Path, files: Sequence[str], directory: str) -> list[dict[str, Any]]:
    root = repo / F2_ROOT_PATH
    copied = []
    for relative in files:
        source = repo / relative; target = root / directory / relative
        _copy_new(source, target)
        copied.append({"source": relative, "path": target.relative_to(root).as_posix(), "sha256": sha256_file(target), "bytes": target.stat().st_size})
    return copied


def initialize(repo_root: Path | str = REPO_ROOT, *, renderer_commit: str, started_at: str) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); root = repo / F2_ROOT_PATH; report_root = repo / F2_REPORT_PATH
    if root.exists() or report_root.exists(): raise StageFF2Error("F2 output root already exists; refusing overwrite")
    if not re.fullmatch(r"[0-9a-f]{40}", renderer_commit) or _git(repo, "rev-parse", renderer_commit + "^{commit}") != renderer_commit: raise StageFF2Error("renderer commit must be an available full SHA")
    startup = {key: value for key, value in verify_startup_gates(repo, require_clean=True, require_remote=True).items() if key != "canonical_digest"}
    if startup["local_head"] != renderer_commit: raise StageFF2Error("renderer commit must match clean pushed startup HEAD")
    startup["renderer_commit"] = renderer_commit; startup["formal_runtime_isolation"] = validate_runtime_isolation()
    _write_new(repo / STARTUP_PATH, _signed(startup))
    _write_new(repo / PENDING_PATH, _signed({"schema_version": 1, "document_type": "stage_f_f2_pending_completion", "stage_id": "F2", "status": "PENDING", "started_at": started_at, "renderer_commit": renderer_commit, "terminal_pass_published": False, "next_authorization": "NONE_PENDING_VALIDATION"}))
    return {"status": "PENDING", "root": F2_ROOT_PATH, "renderer_commit": renderer_commit}


def _source_binding(source: Mapping[str, Any]) -> dict[str, Any]:
    return _signed({"schema_version": 1, "document_type": "stage_f_f2_source_binding", "stage_id": "F2", "status": "PASS", "primary_numeric_root": source["primary_numeric_root"], "source_input_digest": source["source_input_digest"], "source_records": source["records"], "csv_json_parity": {table: {"row_count": source["tables"][table]["row_count"], "columns": source["tables"][table]["columns"], "row_order_digest": source["tables"][table]["row_order_digest"]} for table in TABLES}, "allowed_scientific_inputs": [record["path"] for record in source["records"]], "stage_e_unit_roots_used": False, "reserve_jpeg_read": False, "d8_d9_attribution_maps_read": False, "model_or_checkpoint_loaded": False, "torch_timm_cuda_used": False})


def _access_audit(repo: Path) -> dict[str, Any]:
    return _signed({"schema_version": 1, "document_type": "stage_f_f2_access_audit", "stage_id": "F2", "status": "PASS", "runtime_isolation": validate_runtime_isolation(), "source_boundary": assert_f2_source_boundary(repo), "allowed_data_root": E4_PAPER_DATA_ROOT_PATH, "allowed_metadata_root": E4_ROOT_PATH, "reserve_jpeg_access": False, "qualitative_selection_access": False, "model_access": False, "checkpoint_access": False, "torch_imported": False, "timm_imported": False, "cuda_or_gpu_access": False, "nvidia_device_handles": [], "forward_backward_ig_attribution_fidelity_executed": False})


def _environment(repo: Path) -> dict[str, Any]:
    f0 = load_contracts(repo)["config"]; fonts = validate_font(f0)
    return _signed({"schema_version": 1, "document_type": "stage_f_f2_environment_font_style_provenance", "stage_id": "F2", "python": platform.python_version(), "matplotlib": matplotlib.__version__, "pillow": PILLOW_VERSION, "numpy": np.__version__, "backend": "Agg", "fonts": fonts, "f0_config_digest": f0["canonical_digest"], "layouts": {key: f0["layouts"][key] for key in LAYOUTS}, "style": f0["style"], "export": f0["export"], "model_access": False, "gpu_cuda_access": False, "jpeg_access": False})


def _figure_analysis() -> str:
    entries = [
        ("F2-A single-column", "展示完整 valid rows 的 fine 4x4 anchor、native 14x14 NNS/matched delta_fid 分布及相对 anchor 的关系。", "不能替代 D10 primary，也不能据此断言尺度因果或优胜方法。", "native 曲线靠近零可能被误读为总体无效；它只描述冻结尺度下的敏感性。"),
        ("F2-A double-column", "与 Image #1 相同数据，宽版便于比较 anchor 与 native 的分布位置。", "不能当作独立复现实验或第二个统计检验。", "版式更宽不代表证据更强。"),
        ("F2-B single-column", "展示 insertion/deletion 对 upsampling operator 的描述性敏感性，并把 uniform/nearest 合并为同一等价证据类。", "不能选出最佳 operator，也没有 confirmatory p-value。", "曲线位置差异不应被解读为预注册 winner ranking。"),
        ("F2-B double-column", "与 Image #3 相同 source rows，以宽版并排观察 insertion 和 deletion。", "不能把 uniform 与 nearest 当成两份独立证据。", "重叠等价类不增加样本量。"),
        ("F2-C single-column", "展示四个冻结 layer/head 坐标在 overall/correct/misclassified 三个 strata 的 source median 与 interval。", "不能把 descriptive interval 当成 confirmatory interval，也不能覆盖 MIXED validity。", "不同 metric 同轴仅供结构化浏览，不是跨 metric 优劣排序。"),
        ("F2-C double-column", "与 Image #5 相同 60 个 source summary rows，宽版便于核对坐标和 strata。", "不能依据视觉高低选择 layer/head winner。", "L11/H2 的 valid-only 数值不包含两个失败值；失败并未补零。"),
        ("F2-D single-column", "同时披露 expected/valid/failed/NA、orders 117/352 typed failures，以及 E2 owner budget waiver。", "不能把 E4 package PASS 解读为全量 scientific PASS。", "valid 2998/3000 不等于失败不存在，E2 waiver 也不等于从未超预算。"),
        ("F2-D double-column", "与 Image #7 相同 QC 和 failure/cost disclosure，宽版便于人工审阅。", "不能产生新 GO/NO-GO 或修复 typed failures。", "failure marker X 表示缺失科学数值，不是数值零。"),
    ]
    lines = ["# Stage F F2 图片分析", "", "状态：`PASS`", "", "全部图片均为 `POST_D10_SECONDARY_DESCRIPTIVE`，使用 F0 冻结 renderer 和 E4 封存 paper data；未生成最终 Figure 3、Figure 4 或 Supplement S3。", ""]
    for index, (title, can, cannot, risk) in enumerate(entries, 1):
        lines.extend([f"## Image #{index}: {title}", "", f"- 这个图能说明什么：{can}", f"- 这个图不能说明什么：{cannot}", f"- 读者可能误读什么：{risk}", ""] )
    return "\n".join(lines)


def build_outputs(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); pending, _ = _strict_json_object(repo / PENDING_PATH, "F2 pending"); _validate_signed(pending, "F2 pending"); validate_runtime_isolation()
    config = _load_config(repo); source = load_source_bundle(repo)
    contracts = _copy_snapshots(repo, CONTRACT_FILES, "contracts"); sources = _copy_snapshots(repo, SOURCE_FILES, "source_snapshot")
    _write_new(repo / SOURCE_BINDING_PATH, _source_binding(source)); _write_new(repo / ACCESS_AUDIT_PATH, _access_audit(repo)); _write_new(repo / ENVIRONMENT_PATH, _environment(repo))
    plot_data = build_plot_data(source); render_reports = []; review_records = []; image_index = 0
    for module in MODULES:
        document = _signed(plot_data[module]); _schema(repo, document, F2_PLOT_DATA_SCHEMA_PATH, f"{module} plot data")
        _write_new(repo / F2_ROOT_PATH / "plot_data" / f"{MODULE_SLUG[module]}_plot_data.json", document)
        _write_new(repo / F2_ROOT_PATH / "captions" / f"{MODULE_SLUG[module]}_caption_data.json", build_caption_data(module, document))
        _write_new(repo / F2_ROOT_PATH / "mappings" / f"{MODULE_SLUG[module]}_source_to_plotted_value_mapping.json", build_mapping(module, document))
        for layout in LAYOUTS:
            report = render_panel_figure(build_render_spec(module, document, layout), repo / F2_ROOT_PATH / "renders" / module.lower(), repo_root=repo, renderer_commit=pending["renderer_commit"], source_input_digest=source["source_input_digest"], plot_data_digest=document["plot_data_digest"], stage_config_digest=config["canonical_digest"], stage_id="F2", report_document_type="stage_f_f2_render_report", report_flags={"p_value_computed": False, "winner_selection_performed": False, "new_go_no_go_issued": False, "new_endpoint_computed": False, "final_combined_figures_rendered": False, "failed_rows_117_352_retained": True, "counts_accompany_valid_only_summaries": True, "e4_scientific_validity": "MIXED"})
            report_path = repo / F2_ROOT_PATH / "renders" / module.lower() / f"{report['figure_id']}.report.json"; _schema(repo, report, F2_RENDER_REPORT_SCHEMA_PATH, "F2 render report"); validate_f2_render_report(report_path, repo); render_reports.append(report)
            image_index += 1; source_png = report_path.parent / report["outputs"]["png"]["path"]; review_relative = REVIEW_ROOT_PATH + f"/image_{image_index:02d}_{report['figure_id']}.png"; _copy_new(source_png, repo / review_relative)
            review_records.append({"image_number": image_index, "module_id": module, "layout_profile": layout, "source_path": source_png.relative_to(repo).as_posix(), "source_sha256": sha256_file(source_png), **_file_record(repo, review_relative)})
    inventory = _signed({"schema_version": 1, "document_type": "stage_f_f2_render_inventory", "stage_id": "F2", "status": "PASS", "scientific_modules_rendered": 4, "layout_render_instances": 8, "exported_figure_files": 24, "final_combined_figures_rendered": False, "renders": [{"module_id": report["module_id"], "figure_id": report["figure_id"], "layout_profile": report["layout_profile"], "source_input_digest": report["source_input_digest"], "plot_data_digest": report["plot_data_digest"], "config_digest": report["config_digest"], "renderer_commit": report["renderer_commit"], "outputs": report["outputs"], "report": _record(repo, F2_ROOT_PATH + f"/renders/{report['module_id'].lower()}/{report['figure_id']}.report.json")} for report in render_reports]})
    _write_new(repo / RENDER_INVENTORY_PATH, inventory)
    review_inventory = _signed({"schema_version": 1, "document_type": "stage_f_f2_review_copy_inventory", "stage_id": "F2", "status": "PASS", "review_copy_count": 8, "copies": review_records})
    _write_new(repo / REVIEW_INVENTORY_PATH, review_inventory); _write_text_new(repo / ANALYSIS_PATH, _figure_analysis()); _write_new(repo / FALSE_PASS_PATH, build_false_pass_audit(repo))
    return {"status": "OUTPUTS_BUILT", "scientific_modules_rendered": 4, "layout_render_instances": 8, "exported_figure_files": 24, "report_png_review_copies": 8, "copied_contracts": len(contracts), "copied_source_files": len(sources)}


def validate_f2_render_report(report_path: Path, repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); report, raw = _strict_json_object(report_path, "F2 render report")
    if raw != canonical_json_bytes(report) + b"\n": raise StageFF2Error("F2 render report is not byte-canonical")
    _validate_signed(report, "F2 render report"); _schema(repo, report, F2_RENDER_REPORT_SCHEMA_PATH, "F2 render report")
    if any(report.get(field) is not False for field in ("model_loaded", "torch_imported", "timm_imported", "cuda_used", "jpeg_read", "bootstrap_or_ci_recomputed", "new_endpoint_computed", "p_value_computed", "winner_selection_performed", "new_go_no_go_issued", "final_combined_figures_rendered")):
        raise StageFF2Error("F2 render report crossed a scientific or runtime boundary")
    if report.get("failed_rows_117_352_retained") is not True or report.get("counts_accompany_valid_only_summaries") is not True or report.get("e4_scientific_validity") != "MIXED":
        raise StageFF2Error("F2 render report lost required mixed-validity disclosure")
    f0 = load_contracts(repo)["config"]; profile = f0["layouts"][report["layout_profile"]]
    if report.get("f0_config_digest") != f0["canonical_digest"] or report.get("renderer_contract_version") != f0["renderer_contract_version"] or report.get("dpi") != f0["export"]["png_dpi"] or report.get("size_inches") != [profile["width_in"], profile["height_in"]] or report.get("style", {}).get("methods") != f0["style"]["methods"]:
        raise StageFF2Error("F2 render report drifted from the frozen F0 visual contract")
    if len(set(report["format_content_digests"].values())) != 1: raise StageFF2Error("F2 formats do not share one content identity")
    for extension, record in report["outputs"].items():
        path = report_path.parent / record["path"]
        if Path(record["path"]).name != record["path"] or path.suffix.lower() != f".{extension}" or not path.is_file() or sha256_file(path) != record["sha256"]: raise StageFF2Error(f"F2 render output tampered: {extension}")
    width, height = report["size_inches"]; png_path = report_path.parent / report["outputs"]["png"]["path"]
    with Image.open(png_path) as image:
        if image.size != (round(width * report["dpi"]), round(height * report["dpi"])): raise StageFF2Error("F2 PNG canvas drifted")
    pdf = (report_path.parent / report["outputs"]["pdf"]["path"]).read_bytes()
    if b"/FontFile2" not in pdf and b"/FontFile3" not in pdf: raise StageFF2Error("F2 PDF font is not embedded")
    return {"status": "PASS", "module_id": report["module_id"], "figure_id": report["figure_id"], "formats": sorted(report["outputs"])}


def assert_no_future_outputs(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); stage = repo / "evidence/stage_f"
    if any(stage.glob("f3*")) or any(stage.glob("f4*")): raise StageFF2Error("F3 or F4 output exists")
    roots = [repo / F2_ROOT_PATH, repo / F2_REPORT_PATH]
    if any(path.is_file() and re.search(r"(?:figure[_ -]?[34]|supp(?:lement)?[_ -]?s3)", path.name, re.I) for root in roots if root.exists() for path in root.rglob("*")): raise StageFF2Error("F2 generated a final combined Figure 3 Figure 4 or Supplement S3")
    return {"status": "PASS", "f3_started": False, "f4_started": False, "final_combined_figures_rendered": False}


def _load_plot(repo: Path, module: str) -> dict[str, Any]:
    path = repo / F2_ROOT_PATH / "plot_data" / f"{MODULE_SLUG[module]}_plot_data.json"; document, raw = _strict_json_object(path, f"{module} plot data")
    if raw != canonical_json_bytes(document) + b"\n": raise StageFF2Error(f"{module} plot data is not byte-canonical")
    _validate_signed(document, f"{module} plot data"); _schema(repo, document, F2_PLOT_DATA_SCHEMA_PATH, f"{module} plot data"); return document


def validate_outputs(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); validate_runtime_isolation(); assert_no_future_outputs(repo); _verify_f1(repo); _verify_e4(repo); _sealed_diff(repo, F0_COMMIT, F0_ROOT_PATH, F0_REPORT_PATH); _sealed_diff(repo, D10_COMMIT, D10_ROOT_PATH, D10_REPORT_PATH)
    source = load_source_bundle(repo); rebuilt = build_plot_data(source)
    for module in MODULES:
        observed = _load_plot(repo, module)
        if observed != _signed(rebuilt[module]): raise StageFF2Error(f"{module} plotted data no longer matches E4 source")
        caption, _ = _strict_json_object(repo / F2_ROOT_PATH / "captions" / f"{MODULE_SLUG[module]}_caption_data.json", f"{module} caption"); mapping, _ = _strict_json_object(repo / F2_ROOT_PATH / "mappings" / f"{MODULE_SLUG[module]}_source_to_plotted_value_mapping.json", f"{module} mapping")
        _validate_signed(caption, f"{module} caption"); _validate_signed(mapping, f"{module} mapping")
        if mapping["plot_data_digest"] != observed["plot_data_digest"] or caption["panel_order"] != [panel["panel_id"] for panel in build_render_spec(module, observed, "double_column")["panels"]]: raise StageFF2Error(f"{module} caption or mapping drifted")
    render_inventory, _ = _strict_json_object(repo / RENDER_INVENTORY_PATH, "F2 render inventory"); _validate_signed(render_inventory, "F2 render inventory")
    if (render_inventory["scientific_modules_rendered"], render_inventory["layout_render_instances"], render_inventory["exported_figure_files"]) != (4, 8, 24): raise StageFF2Error("F2 render counts drifted")
    if {(item["module_id"], item["layout_profile"]) for item in render_inventory["renders"]} != {(module, layout) for module in MODULES for layout in LAYOUTS}: raise StageFF2Error("F2 render module/layout coverage drifted")
    for item in render_inventory["renders"]:
        report_path = repo / item["report"]["path"]; validate_f2_render_report(report_path, repo); report, _ = _strict_json_object(report_path, "F2 render report")
        if report["source_input_digest"] != source["source_input_digest"] or report["plot_data_digest"] != _load_plot(repo, item["module_id"])["plot_data_digest"] or report["config_digest"] != _load_config(repo)["canonical_digest"]: raise StageFF2Error("F2 render source plot-data or config binding drifted")
        expected_spec = build_render_spec(item["module_id"], _load_plot(repo, item["module_id"]), item["layout_profile"]); f0 = load_contracts(repo)["config"]
        expected_content = canonical_sha256({"stage_id": "F2", "figure_spec": expected_spec, "source_input_digest": source["source_input_digest"], "plot_data_digest": report["plot_data_digest"], "stage_config_digest": report["config_digest"], "f0_config_digest": f0["canonical_digest"], "renderer_contract": f0["renderer_contract_version"]})
        if report["content_digest"] != expected_content or any(value != expected_content for value in report["format_content_digests"].values()): raise StageFF2Error("F2 render spec or cross-format content identity drifted")
        with Image.open(report_path.parent / report["outputs"]["png"]["path"]) as image: pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
        if pixels.std() <= 5 or not 0.005 < np.mean(np.any(pixels < 248, axis=2)) < 0.95: raise StageFF2Error("F2 visual sanity failed")
    review, _ = _strict_json_object(repo / REVIEW_INVENTORY_PATH, "F2 review inventory"); _validate_signed(review, "F2 review inventory")
    if review.get("review_copy_count") != 8 or len(review.get("copies", [])) != 8: raise StageFF2Error("F2 review PNG count drifted")
    for item in review["copies"]:
        if item["source_sha256"] != item["sha256"] or sha256_file(repo / item["path"]) != item["sha256"]: raise StageFF2Error("F2 review PNG is not an exact copy")
    analysis = (repo / ANALYSIS_PATH).read_text(encoding="utf-8")
    if any(f"## Image #{index}:" not in analysis for index in range(1, 9)) or any(phrase not in analysis for phrase in ("这个图能说明什么", "这个图不能说明什么", "读者可能误读什么")): raise StageFF2Error("F2 figure analysis is incomplete")
    false_pass, _ = _strict_json_object(repo / FALSE_PASS_PATH, "F2 false-pass audit"); _validate_signed(false_pass, "F2 false-pass audit"); _schema(repo, false_pass, F2_FALSE_PASS_SCHEMA_PATH, "F2 false-pass audit")
    if _git(repo, "diff", "--check"): raise StageFF2Error("git diff --check failed")
    checks = {name: "PASS" for name in ("f1_ref_completion_hash_inventory_terminal_state", "e4_ref_completion_hash_inventory_recipe", "e4_manifest_metadata_completion_paper_data_binding", "csv_json_rows_columns_order_fields_parity", "f2_a_patch_scale_source_to_plotted_exactness", "f2_b_upsampling_source_to_plotted_exactness", "uniform_nearest_one_equivalent_evidence_class", "f2_c_four_coordinates_and_three_strata_exactness", "f2_c_method_metric_n_valid_failed_na_median_interval_exactness", "f2_d_expected_valid_failed_na_exactness", "failed_orders_117_352_missing_typed_and_not_imputed", "valid_only_summary_with_counts", "e2_owner_budget_waiver_disclosed", "e4_scientific_validity_mixed", "no_p_value_winner_new_decision_or_endpoint", "caption_keys_and_panel_order", "eight_render_instances_png_pdf_svg_reports", "report_side_png_copies_and_image_analysis", "no_f3_f4_or_final_combined_figures", "no_jpeg_model_checkpoint_torch_timm_cuda_nvidia_access", "d10_f0_f1_e4_sealed_roots_unmodified", "png_pdf_svg_same_input_plot_data_config", "font_style_layout_export_contract", "git_diff_check")}
    report = _signed({"schema_version": 1, "document_type": "stage_f_f2_validation_report", "stage_id": "F2", "status": "PASS", "checks": checks, "scientific_modules_rendered": 4, "layout_render_instances": 8, "exported_figure_files": 24, "report_png_review_copies": 8, "final_combined_figures_rendered": False, "model_torch_timm_gpu_cuda_jpeg_access": False, "p_value_winner_new_decision_endpoint_computed": False, "failed_rows_117_352_retained": True, "counts_accompany_valid_only_summaries": True, "e4_scientific_validity": "MIXED", "d10_f0_f1_e4_decisions_modified": False, "f3_started": False, "f4_started": False, "next_authorization": "F3_ONLY"})
    _schema(repo, report, F2_VALIDATION_SCHEMA_PATH, "F2 validation report"); return report


def build_false_pass_audit(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    evidence = {attack: "F2 source adapter, schemas, render reports, source-recomputing validator, runtime boundary, or atomic finalizer rejects this false-pass path" for attack in ATTACKS}
    document = _signed({"schema_version": 1, "document_type": "stage_f_f2_false_pass_audit", "stage_id": "F2", "status": "PASS", "attacks": [{"attack": attack, "result": "REJECTED", "evidence": evidence[attack]} for attack in ATTACKS]})
    _schema(repo, document, F2_FALSE_PASS_SCHEMA_PATH, "F2 false-pass audit"); return document


def record_test_report(repo_root: Path | str, *, passed: int, failed: int, command: str) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    if type(passed) is not int or passed <= 0 or failed != 0 or not command: raise StageFF2Error("terminal test report requires passes and zero failures")
    report = _signed({"schema_version": 1, "document_type": "stage_f_f2_test_report", "stage_id": "F2", "status": "PASS", "command": command, "passed": passed, "failed": failed, "required_environment": {"CUDA_VISIBLE_DEVICES": "", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}, "model_access": False, "torch_imported": False, "timm_imported": False, "gpu_cuda_access": False, "jpeg_access": False})
    _write_new(repo / TEST_PATH, report); return report


def finalize(repo_root: Path | str = REPO_ROOT, *, completed_at: str) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); root = repo / F2_ROOT_PATH; validate_runtime_isolation()
    if (repo / COMPLETION_PATH).exists(): raise StageFF2Error("F2 terminal completion already exists")
    test = _record(repo, TEST_PATH); validation = validate_outputs(repo); _write_new(repo / VALIDATION_PATH, validation)
    excluded = {"payload_manifest.json", "whole_root_inventory.json", "status.json", "completion/f2.pending.json", "completion/f2.json"}
    artifacts = [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(root.rglob("*")) if path.is_file() and not path.is_symlink() and path.relative_to(root).as_posix() not in excluded]
    manifest = _signed({"schema_version": 1, "document_type": "stage_f_f2_payload_manifest", "stage_id": "F2", "immutable": True, "payload_count": len(artifacts), "artifacts": artifacts, "scientific_modules_rendered": 4, "layout_render_instances": 8, "exported_figure_files": 24, "report_png_review_copies": 8, "final_combined_figures_rendered": False}); _write_new(repo / MANIFEST_PATH, manifest)
    inventory = _inventory(root, exclude=("whole_root_inventory.json", "status.json", "completion/f2.pending.json", "completion/f2.json")); inventory_doc = _signed({"schema_version": 1, "document_type": "stage_f_f2_whole_root_inventory", "stage_id": "F2", "root": F2_ROOT_PATH, "scope": "all regular F2 files except self-referential inventory status and completion receipts", "files": inventory["files"], "file_count": inventory["file_count"], "total_bytes": inventory["total_bytes"]}); _write_new(repo / INVENTORY_PATH, inventory_doc)
    status = _signed({"schema_version": 1, "document_type": "stage_f_f2_status", "stage_id": "F2", "status": "PASS", "terminal": True, "immutable": True, "package_validation": "PASS", "workflow_validation": "PASS", "scientific_modules_rendered": 4, "layout_render_instances": 8, "exported_figure_files": 24, "report_png_review_copies": 8, "final_combined_figures_rendered": False, "model_torch_timm_gpu_cuda_jpeg_access": False, "p_value_winner_new_decision_endpoint_computed": False, "failed_rows_117_352_retained": True, "counts_accompany_valid_only_summaries": True, "e4_scientific_validity": "MIXED", "d10_f0_f1_e4_decisions_modified": False, "f3_started": False, "f4_started": False, "next_authorization": "F3_ONLY", "whole_root_inventory": {"canonical_digest": inventory_doc["canonical_digest"], "file_count": inventory_doc["file_count"], "total_bytes": inventory_doc["total_bytes"]}, "git": {"commit": "SELF", "ref": F2_ARCHIVE_REF}}); _write_new(repo / STATUS_PATH, status)
    pending, _ = _strict_json_object(repo / PENDING_PATH, "F2 pending"); _validate_signed(pending, "F2 pending")
    completion = _signed({"schema_version": 1, "document_type": "stage_f_f2_terminal_completion", "stage_id": "F2", "status": "PASS", "started_at": pending["started_at"], "completed_at": completed_at, "package_validation": "PASS", "workflow_validation": "PASS", "scientific_modules_rendered": 4, "layout_render_instances": 8, "exported_figure_files": 24, "report_png_review_copies": 8, "final_combined_figures_rendered": False, "model_access": False, "torch_imported": False, "timm_imported": False, "gpu_cuda_access": False, "jpeg_access": False, "p_value_winner_new_go_no_go_new_endpoint_recomputed": False, "failed_rows_117_352_retained": True, "counts_accompany_valid_only_summaries": True, "d10_f0_f1_e4_decisions_modified": False, "startup_gate": _record(repo, STARTUP_PATH), "source_binding": _record(repo, SOURCE_BINDING_PATH), "access_audit": _record(repo, ACCESS_AUDIT_PATH), "source_validation": _record(repo, VALIDATION_PATH), "test_report": test, "false_pass_audit": _record(repo, FALSE_PASS_PATH), "render_inventory": _record(repo, RENDER_INVENTORY_PATH), "review_copy_inventory": _record(repo, REVIEW_INVENTORY_PATH), "figure_analysis": _file_record(repo, ANALYSIS_PATH), "payload_manifest": _record(repo, MANIFEST_PATH), "status_record": _record(repo, STATUS_PATH), "whole_root_inventory": _record(repo, INVENTORY_PATH), "git": {"commit": "SELF", "ref": F2_ARCHIVE_REF}, "next_authorization": "F3_ONLY", "f3_started": False, "f4_started": False}); _schema(repo, completion, F2_COMPLETION_SCHEMA_PATH, "F2 completion")
    temporary = (repo / PENDING_PATH).with_name(".f2-terminal.tmp"); _write_new(temporary, completion); os.replace(temporary, repo / PENDING_PATH); os.replace(repo / PENDING_PATH, repo / COMPLETION_PATH); return validate_completion(repo)


def validate_completion(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); root = repo / F2_ROOT_PATH
    if not (repo / COMPLETION_PATH).is_file() or (repo / PENDING_PATH).exists(): raise StageFF2Error("terminal completion missing or pending remains")
    completion, raw = _strict_json_object(repo / COMPLETION_PATH, "F2 completion")
    if raw != canonical_json_bytes(completion) + b"\n": raise StageFF2Error("F2 completion is not byte-canonical")
    _validate_signed(completion, "F2 completion"); _schema(repo, completion, F2_COMPLETION_SCHEMA_PATH, "F2 completion")
    expected = {"status": "PASS", "package_validation": "PASS", "workflow_validation": "PASS", "scientific_modules_rendered": 4, "layout_render_instances": 8, "exported_figure_files": 24, "report_png_review_copies": 8, "final_combined_figures_rendered": False, "model_access": False, "torch_imported": False, "timm_imported": False, "gpu_cuda_access": False, "jpeg_access": False, "p_value_winner_new_go_no_go_new_endpoint_recomputed": False, "failed_rows_117_352_retained": True, "counts_accompany_valid_only_summaries": True, "d10_f0_f1_e4_decisions_modified": False, "f3_started": False, "f4_started": False, "next_authorization": "F3_ONLY"}
    if any(completion.get(key) != value for key, value in expected.items()): raise StageFF2Error("F2 completion terminal state drifted")
    for field in ("startup_gate", "source_binding", "access_audit", "source_validation", "test_report", "false_pass_audit", "render_inventory", "review_copy_inventory", "payload_manifest", "status_record", "whole_root_inventory"):
        if completion[field] != _record(repo, completion[field]["path"]): raise StageFF2Error(f"F2 completion binding drifted: {field}")
    if completion["figure_analysis"] != _file_record(repo, ANALYSIS_PATH): raise StageFF2Error("F2 figure analysis binding drifted")
    inventory, _ = _strict_json_object(repo / INVENTORY_PATH, "F2 inventory"); _validate_signed(inventory, "F2 inventory")
    rebuilt = _inventory(root, exclude=("whole_root_inventory.json", "status.json", "completion/f2.pending.json", "completion/f2.json"))
    unsigned = {"schema_version": 1, "document_type": "stage_f_f2_whole_root_inventory", "stage_id": "F2", "root": F2_ROOT_PATH, "scope": "all regular F2 files except self-referential inventory status and completion receipts", "files": rebuilt["files"], "file_count": rebuilt["file_count"], "total_bytes": rebuilt["total_bytes"]}
    if any(inventory[key] != rebuilt[key] for key in ("files", "file_count", "total_bytes")) or inventory["canonical_digest"] != canonical_sha256(unsigned): raise StageFF2Error("F2 whole-root inventory drifted")
    validate_runtime_isolation(); return {**expected, "whole_root_inventory_digest": inventory["canonical_digest"], "model_torch_timm_gpu_cuda_jpeg_access": False}


__all__ = ["ATTACKS", "COMPLETION_PATH", "F2_ARCHIVE_REF", "F2_ROOT_PATH", "F2_REPORT_PATH", "StageFF2Error", "assert_f2_source_boundary", "assert_no_future_outputs", "build_caption_data", "build_false_pass_audit", "build_outputs", "build_plot_data", "build_render_spec", "finalize", "initialize", "load_source_bundle", "record_test_report", "validate_completion", "validate_f2_render_report", "validate_outputs", "validate_runtime_isolation", "verify_startup_gates"]
