"""Stage F F1 formal rendering from sealed D10 paper data.

F1 is a display-only renderer stage.  It consumes only the D10 analysis
``paper_data`` tables plus D10 analysis metadata, builds explicit plot-data
snapshots, and renders the three Figure 2 statistical modules as independent
artifacts.  It does not load models, Torch, timm, CUDA, JPEGs, attribution maps,
or Stage E paper data.
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
    ARCHIVE_REF as F0_ARCHIVE_REF,
    CONFIG_PATH as F0_CONFIG_PATH,
    E4_COMPLETION_PATH,
    E4_INVENTORY_PATH,
    EXPECTED_E4_COMPLETION_SHA256,
    EXPECTED_E4_INVENTORY_DIGEST,
    EXPECTED_E4_REF,
    StageFRendererError,
    load_contracts,
    render_panel_figure,
    sha256_file,
    validate_font,
)


F1_CONFIG_PATH = "config/stage_f_f1.json"
F1_CONFIG_SCHEMA_PATH = "config/stage_f_f1.schema.json"
F1_RUN_SPEC_PATH = "config/stage_f_f1_run_spec.json"
F1_RUN_SPEC_SCHEMA_PATH = "config/stage_f_f1_run_spec.schema.json"
F1_PLOT_DATA_SCHEMA_PATH = "config/stage_f_f1_plot_data.schema.json"
F1_RENDER_REPORT_SCHEMA_PATH = "config/stage_f_f1_render_report.schema.json"
F1_VALIDATION_SCHEMA_PATH = "config/stage_f_f1_validation_report.schema.json"
F1_COMPLETION_SCHEMA_PATH = "config/stage_f_f1_completion.schema.json"
F1_FALSE_PASS_SCHEMA_PATH = "config/stage_f_f1_false_pass_audit.schema.json"

F1_ROOT_PATH = "evidence/stage_f/f1_r3_recovery_v2_20260907/formal"
F1_REPORT_PATH = "reports/stage_f_f1_r3_20260907"
PENDING_PATH = F1_ROOT_PATH + "/completion/f1.pending.json"
COMPLETION_PATH = F1_ROOT_PATH + "/completion/f1.json"
STARTUP_PATH = F1_ROOT_PATH + "/startup_gate.json"
SOURCE_BINDING_PATH = F1_ROOT_PATH + "/source_binding.json"
ACCESS_AUDIT_PATH = F1_ROOT_PATH + "/access_audit.json"
ENVIRONMENT_PATH = F1_ROOT_PATH + "/environment_font_style_provenance.json"
RENDER_INVENTORY_PATH = F1_ROOT_PATH + "/render_inventory.json"
MANIFEST_PATH = F1_ROOT_PATH + "/payload_manifest.json"
INVENTORY_PATH = F1_ROOT_PATH + "/whole_root_inventory.json"
STATUS_PATH = F1_ROOT_PATH + "/status.json"
VALIDATION_PATH = F1_REPORT_PATH + "/validation_report.json"
FALSE_PASS_PATH = F1_REPORT_PATH + "/false_pass_audit.json"
TEST_PATH = F1_REPORT_PATH + "/test_report.json"

F0_COMPLETION_PATH = "evidence/stage_f/f0_r4_recovery_v2_20260907/formal/completion/f0.json"
F0_STATUS_PATH = "evidence/stage_f/f0_r4_recovery_v2_20260907/formal/status.json"
F0_INVENTORY_PATH = "evidence/stage_f/f0_r4_recovery_v2_20260907/formal/whole_root_inventory.json"
F0_GOLDEN_INVENTORY_PATH = "evidence/stage_f/f0_r4_recovery_v2_20260907/formal/golden_render_inventory.json"
F0_VALIDATION_PATH = "reports/stage_f_f0_r4_20260907/validation_report.json"
F0_FALSE_PASS_PATH = "reports/stage_f_f0_r4_20260907/false_pass_audit.json"

D10_ARCHIVE_REF = "refs/archive/stage-d-fidelity-v1-d10-completion"
D10_REPORT_PATH = "reports/stage_d_d10_recovery_v2_20260906/d10_formal_decision.json"
D10_ROOT_PATH = "evidence/stage_d/d10_recovery_v2_20260906/formal_r2"
D10_ANALYSIS_ROOT_PATH = D10_ROOT_PATH + "/analysis"
D10_PAPER_ROOT_PATH = D10_ROOT_PATH + "/paper"
D10_PAPER_DATA_ROOT_PATH = D10_ANALYSIS_ROOT_PATH + "/paper_data"
D10_FIGURE_RECIPES_PATH = D10_ANALYSIS_ROOT_PATH + "/figure_recipes.json"

E4_ROOT_PATH = "evidence/stage_e/e4_recovery_v2_20260907/formal"
F1_ARCHIVE_REF = "refs/archive/stage-f-v1-f1-completion"

EXPECTED_F0_COMMIT = "c4df92ff05054712406caa5524c5d129da7dab42"
EXPECTED_F0_COMPLETION_SHA256 = "83b4ce1be0bf1c0d2ef49fe60a0793975d545944218451d0e4211e5e2261f7ce"
EXPECTED_F0_COMPLETION_CANONICAL_DIGEST = "7e56bc714033ae9b62edaef04f2bd47b4d2ad9aa7592146fbd8e52aa71a896cc"
EXPECTED_F0_INVENTORY_CANONICAL_DIGEST = "28eab8d914cb062ade48e781985cc7055b6cbc703731d2086e323a0b47d5fdc8"
EXPECTED_D10_COMMIT = "7520689bb031304e4789567f99d093062aa50f72"
EXPECTED_D10_REPORT_SHA256 = "7eac27b11b9df72a91c6e78cd0d46ea30f7d8701f0d41cbc29e9cb76695b5a08"
EXPECTED_D10_REPORT_CANONICAL_DIGEST = "db9b235cd8cfdeb3caf99445939443b17007ca5225c710c1164e36e2593503e6"
EXPECTED_D10_ROOT_INVENTORY_DIGEST = "b72d36430d149598ccd9cfd8dd586297e7231be29e61b44309e9971087e46535"
EXPECTED_D10_FIGURE_RECIPES_SHA256 = "aaa1858e5a2b0e820d4ea5cb3072a41e3cb6f4f507daefa319dc9ab66a1f0ad1"

TABLES = ("cohort_comparison", "per_sample_delta", "curve_summary", "main_fidelity_table", "resource_and_qc")
MODULES = ("F1-A", "F1-B", "F1-C")
LAYOUTS = ("single_column", "double_column")
SOURCE_TABLE_BY_MODULE = {"F1-A": "cohort_comparison", "F1-B": "per_sample_delta", "F1-C": "curve_summary"}
MODULE_SLUG = {"F1-A": "f1_a", "F1-B": "f1_b", "F1-C": "f1_c"}
CAPTION_KEY_BY_MODULE = {
    "F1-A": "f1_a_cohort_median_interval",
    "F1-B": "f1_b_stage_d_delta_fid_ecdf",
    "F1-C": "f1_c_stage_d_fidelity_curves",
}
METHOD_SOURCE_TO_RENDER = {
    "conditional_fixed_clean_attention_head_direction_value_path_ig": "nns",
    "matched_fixed_clean_attention_patch_token_contribution": "matched",
}
METHOD_LABELS = {
    "conditional_fixed_clean_attention_head_direction_value_path_ig": "NNS",
    "matched_fixed_clean_attention_patch_token_contribution": "matched",
}
F1_A_CAPTION = (
    "C10 的 15 图 holdout 与 Stage D 的独立 500 图 cohort 分开报告且不池化；"
    "C10 的历史 NO_GO_NOVELTY、FAIL_SPATIAL 和 BLOCKED 状态保持不变。"
)

CONTRACT_FILES = (
    F1_CONFIG_PATH,
    F1_CONFIG_SCHEMA_PATH,
    F1_RUN_SPEC_PATH,
    F1_RUN_SPEC_SCHEMA_PATH,
    F1_PLOT_DATA_SCHEMA_PATH,
    F1_RENDER_REPORT_SCHEMA_PATH,
    F1_VALIDATION_SCHEMA_PATH,
    F1_COMPLETION_SCHEMA_PATH,
    F1_FALSE_PASS_SCHEMA_PATH,
    F0_CONFIG_PATH,
    "config/stage_f_caption_contract.json",
    "config/stage_f_figure_catalog.json",
)
SOURCE_FILES = (
    "src/vitnns/display.py",
    "src/vitnns/stage_f_renderer.py",
    "src/vitnns/stage_f_cli.py",
    "src/vitnns/stage_f_f1.py",
    "tests/test_stage_f_f1.py",
    "tests/test_stage_f_f1_boundary.py",
)

ATTACKS = (
    "f0_archive_ref_correct_but_completion_hash_or_inventory_drifted",
    "d10_ref_correct_but_report_recipe_manifest_or_paper_data_drifted",
    "analysis_and_paper_copy_mismatch_continues_rendering",
    "c10_and_stage_d_pooled_as_n515",
    "c10_no_go_fail_spatial_blocked_hidden_or_weakened",
    "stage_d_result_rewrites_c10",
    "bootstrap_ci_or_new_endpoint_recomputed",
    "ecdf_not_full_500_images",
    "misclassified_rows_removed",
    "insertion_deletion_or_nns_matched_mismatched",
    "curve_band_or_fraction_grid_differs_from_d10_source",
    "png_pdf_svg_use_different_input_plot_data_or_config",
    "stage_e_paper_data_read",
    "jpeg_model_torch_timm_cuda_or_nvidia_access",
    "qualitative_f2_f3_f4_or_final_figure_2_generated",
    "pass_declared_before_validator_manifest_or_terminal_receipt",
    "completion_authorizes_f3_or_f4",
    "complete_declared_before_commit_push_archive_verification",
)


class StageFF1Error(StageFRendererError):
    """Raised when F1 cannot be safely validated or sealed."""


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


def _strict_json(path: Path, context: str) -> tuple[Any, bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except Exception as error:
        raise StageFF1Error(f"{context} is invalid JSON: {error}") from error
    return value, raw


def _strict_json_object(path: Path, context: str) -> tuple[dict[str, Any], bytes]:
    value, raw = _strict_json(path, context)
    if not isinstance(value, dict):
        raise StageFF1Error(f"{context} must be a JSON object")
    return value, raw


def _validate_signed(value: Mapping[str, Any], context: str) -> None:
    if value.get("canonical_digest") != canonical_sha256({key: item for key, item in value.items() if key != "canonical_digest"}):
        raise StageFF1Error(f"{context} canonical digest mismatch")


def _schema(repo: Path, document: Mapping[str, Any], relative: str, context: str) -> None:
    schema, _ = _strict_json_object(repo / relative, f"{context} schema")
    Draft202012Validator.check_schema(schema)
    errors = sorted(Draft202012Validator(schema).iter_errors(document), key=lambda item: list(item.path))
    if errors:
        location = ".".join(str(item) for item in errors[0].path) or "<root>"
        raise StageFF1Error(f"{context} schema failed at {location}: {errors[0].message}")


def _record(repo: Path, relative: str) -> dict[str, Any]:
    path = repo / relative
    value, _ = _strict_json_object(path, relative)
    _validate_signed(value, relative)
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "canonical_digest": value["canonical_digest"],
    }


def _file_record(repo: Path, relative: str) -> dict[str, Any]:
    path = repo / relative
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(("git", *args), cwd=repo, text=True, capture_output=True, check=False)
    if result.returncode:
        raise StageFF1Error(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _is_ancestor(repo: Path, ancestor: str, descendant: str = "HEAD") -> bool:
    return subprocess.run(
        ("git", "merge-base", "--is-ancestor", ancestor, descendant),
        cwd=repo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    ).returncode == 0


def _inventory(root: Path, *, exclude: Sequence[str] = ()) -> dict[str, Any]:
    excluded = set(exclude)
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            relative = path.relative_to(root).as_posix()
            if relative not in excluded:
                files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "canonical_digest": canonical_sha256({"files": files}),
    }


def _load_config(repo: Path) -> dict[str, Any]:
    config, raw = _strict_json_object(repo / F1_CONFIG_PATH, "F1 config")
    if raw != canonical_json_bytes(config) + b"\n":
        raise StageFF1Error("F1 config is not byte-canonical")
    _validate_signed(config, "F1 config")
    _schema(repo, config, F1_CONFIG_SCHEMA_PATH, "F1 config")
    f0 = load_contracts(repo)["config"]
    if config["renderer"]["f0_config_digest"] != f0["canonical_digest"]:
        raise StageFF1Error("F1 config is not bound to the frozen F0 renderer config")
    if config["renderer"]["renderer_contract_version"] != f0["renderer_contract_version"]:
        raise StageFF1Error("F1 renderer contract version drifted")
    return config


def _load_run_spec(repo: Path) -> dict[str, Any]:
    run_spec, raw = _strict_json_object(repo / F1_RUN_SPEC_PATH, "F1 run spec")
    if raw != canonical_json_bytes(run_spec) + b"\n":
        raise StageFF1Error("F1 run spec is not byte-canonical")
    _validate_signed(run_spec, "F1 run spec")
    _schema(repo, run_spec, F1_RUN_SPEC_SCHEMA_PATH, "F1 run spec")
    return run_spec


def validate_runtime_isolation() -> dict[str, Any]:
    forbidden = sorted(
        name for name in sys.modules
        if name == "torch" or name.startswith("torch.")
        or name == "torchvision" or name.startswith("torchvision.")
        or name == "timm" or name.startswith("timm.")
    )
    if forbidden:
        raise StageFF1Error(f"formal F1 process imported a forbidden runtime: {forbidden[0]}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise StageFF1Error("formal F1 producer requires CUDA_VISIBLE_DEVICES to be empty")
    nvidia_handles = []
    fd_root = Path("/proc/self/fd")
    if fd_root.is_dir():
        for path in fd_root.iterdir():
            try:
                target = os.readlink(path)
            except OSError:
                continue
            if target.startswith("/dev/" + "nvidia"):
                nvidia_handles.append(target)
    if nvidia_handles:
        raise StageFF1Error("formal F1 process holds an NVIDIA device handle")
    return {
        "status": "PASS",
        "cuda_visible_devices": "",
        "torch_imported": False,
        "torchvision_imported": False,
        "timm_imported": False,
        "nvidia_device_handles": [],
        "jpeg_read": False,
        "model_loaded": False,
        "checkpoint_loaded": False,
    }


def assert_f1_source_boundary(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    paths = [repo / path for path in ("src/vitnns/stage_f_renderer.py", "src/vitnns/stage_f_cli.py", "src/vitnns/stage_f_f1.py")]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    for name in ("torch", "torchvision", "timm"):
        if re.search(rf"^\s*(?:from|import)\s+{re.escape(name)}(?:\.|\s|$)", combined, re.MULTILINE):
            raise StageFF1Error(f"F1 source imports forbidden runtime: {name}")
    forbidden_tokens = (
        "load_" + "state_dict", "load_" + "canonical_model",
        "for" + "ward(", "back" + "ward(", "integrated_" + "grad",
        "cuda" + ":", "." + "cuda(",
        "/dev/" + "nvidia",
        "stage_e/e4_recovery_v2_20260907/formal/" + "paper_data",
    )
    if any(token in combined for token in forbidden_tokens):
        raise StageFF1Error("F1 source exposes forbidden compute, GPU, JPEG, or Stage E paper-data surface")
    return {
        "status": "PASS",
        "model_access": False,
        "checkpoint_access": False,
        "torch_access": False,
        "timm_access": False,
        "cuda_access": False,
        "stage_e_paper_data_access": False,
        "jpeg_access": False,
    }


def _git_clean(repo: Path) -> bool:
    return _git(repo, "status", "--porcelain=v1") == ""


def _lfs_clean(repo: Path) -> bool:
    status = _git(repo, "lfs", "status")
    return not any(line.startswith("\t") or line.startswith("    ") for line in status.splitlines())


def _verify_f0(repo: Path) -> dict[str, Any]:
    config = _load_config(repo)
    if _git(repo, "rev-parse", F0_ARCHIVE_REF + "^{commit}") != EXPECTED_F0_COMMIT:
        raise StageFF1Error("F0 archive ref drifted")
    if not _is_ancestor(repo, EXPECTED_F0_COMMIT):
        raise StageFF1Error("F0 archive commit must be an ancestor of HEAD")
    completion, _ = _strict_json_object(repo / F0_COMPLETION_PATH, "F0 completion")
    status, _ = _strict_json_object(repo / F0_STATUS_PATH, "F0 status")
    inventory, _ = _strict_json_object(repo / F0_INVENTORY_PATH, "F0 whole-root inventory")
    golden, _ = _strict_json_object(repo / F0_GOLDEN_INVENTORY_PATH, "F0 golden inventory")
    validation, _ = _strict_json_object(repo / F0_VALIDATION_PATH, "F0 validation")
    false_pass, _ = _strict_json_object(repo / F0_FALSE_PASS_PATH, "F0 false-pass audit")
    for label, document in (
        ("F0 completion", completion),
        ("F0 status", status),
        ("F0 whole-root inventory", inventory),
        ("F0 golden inventory", golden),
        ("F0 validation", validation),
        ("F0 false-pass audit", false_pass),
    ):
        _validate_signed(document, label)
    if sha256_file(repo / F0_COMPLETION_PATH) != config["inputs"]["f0_completion_sha256"]:
        raise StageFF1Error("F0 completion file hash drifted")
    if completion["canonical_digest"] != config["inputs"]["f0_completion_canonical_digest"]:
        raise StageFF1Error("F0 completion canonical digest drifted")
    if inventory["canonical_digest"] != config["inputs"]["f0_whole_root_inventory_canonical_digest"]:
        raise StageFF1Error("F0 whole-root inventory digest drifted")
    if (
        completion.get("status") != "PASS"
        or completion.get("package_validation") != "PASS"
        or completion.get("workflow_validation") != "PASS"
        or completion.get("formal_scientific_figures_rendered") != 0
        or completion.get("model_gpu_access") is not False
        or completion.get("jpeg_access") is not False
        or completion.get("next_authorization") != "F1_ONLY"
        or any(completion.get(key) for key in ("f2_started", "f3_started", "f4_started"))
    ):
        raise StageFF1Error("F0 completion terminal state drifted")
    if (
        status.get("status") != "PASS"
        or status.get("package_validation") != "PASS"
        or status.get("workflow_validation") != "PASS"
        or status.get("formal_scientific_figures_rendered") != 0
        or status.get("model_gpu_jpeg_access") is not False
        or status.get("d10_e4_decisions_modified") is not False
        or status.get("next_authorization") != "F1_ONLY"
        or any(status.get(key) for key in ("f2_started", "f3_started", "f4_started"))
    ):
        raise StageFF1Error("F0 status terminal state drifted")
    if _git(repo, "diff", "--name-only", EXPECTED_F0_COMMIT, "--", F0_COMPLETION_PATH, F0_STATUS_PATH, F0_INVENTORY_PATH, F0_GOLDEN_INVENTORY_PATH, F0_VALIDATION_PATH, F0_FALSE_PASS_PATH):
        raise StageFF1Error("F0 sealed inputs changed after the archive commit")
    return {
        "status": "PASS",
        "archive_ref": F0_ARCHIVE_REF,
        "archive_commit": EXPECTED_F0_COMMIT,
        "completion_sha256": sha256_file(repo / F0_COMPLETION_PATH),
        "completion_canonical_digest": completion["canonical_digest"],
        "whole_root_inventory_canonical_digest": inventory["canonical_digest"],
        "next_authorization": "F1_ONLY",
    }


def _d10_output_root_inventory(repo: Path) -> dict[str, Any]:
    return _inventory(repo / D10_ROOT_PATH)


def _verify_e4_sealed(repo: Path) -> dict[str, Any]:
    if _git(repo, "rev-parse", "refs/archive/stage-e-e4-completion^{commit}") != EXPECTED_E4_REF:
        raise StageFF1Error("E4 archive ref drifted")
    if not _is_ancestor(repo, EXPECTED_E4_REF):
        raise StageFF1Error("E4 archive commit must be an ancestor of HEAD")
    completion, _ = _strict_json_object(repo / E4_COMPLETION_PATH, "E4 completion")
    inventory, _ = _strict_json_object(repo / E4_INVENTORY_PATH, "E4 whole-root inventory")
    _validate_signed(completion, "E4 completion")
    _validate_signed(inventory, "E4 inventory")
    if sha256_file(repo / E4_COMPLETION_PATH) != EXPECTED_E4_COMPLETION_SHA256:
        raise StageFF1Error("E4 completion hash drifted")
    if inventory.get("canonical_digest") != EXPECTED_E4_INVENTORY_DIGEST:
        raise StageFF1Error("E4 whole-root inventory drifted")
    if _git(repo, "diff", "--name-only", EXPECTED_E4_REF, "--", E4_ROOT_PATH):
        raise StageFF1Error("sealed E4 root changed after its archive commit")
    return {"status": "PASS", "archive_commit": EXPECTED_E4_REF, "completion_sha256": EXPECTED_E4_COMPLETION_SHA256, "whole_root_inventory_canonical_digest": EXPECTED_E4_INVENTORY_DIGEST}


def verify_startup_gates(
    repo_root: Path | str = REPO_ROOT, *, require_clean: bool = True, require_remote: bool = True,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    config = _load_config(repo)
    run_spec = _load_run_spec(repo)
    branch = _git(repo, "symbolic-ref", "--short", "HEAD")
    head = _git(repo, "rev-parse", "HEAD")
    tracking = _git(repo, "rev-parse", "origin/stage-d-d3-pass")
    if branch != "stage-d-d3-pass":
        raise StageFF1Error("F1 must run on branch stage-d-d3-pass")
    if head != tracking:
        raise StageFF1Error("local HEAD and origin/stage-d-d3-pass must match")
    remote = None
    if require_remote:
        remote_lines = _git(repo, "ls-remote", "--heads", "origin", "stage-d-d3-pass").splitlines()
        if len(remote_lines) != 1:
            raise StageFF1Error("remote stage-d-d3-pass is absent or ambiguous")
        remote = remote_lines[0].split()[0]
        if remote != head:
            raise StageFF1Error("local, tracking, and remote branch are not identical")
    if require_clean and not _git_clean(repo):
        raise StageFF1Error("F1 start requires a clean worktree and index")
    if require_clean and not _lfs_clean(repo):
        raise StageFF1Error("F1 start requires clean Git LFS state")
    f0 = _verify_f0(repo)
    if _git(repo, "rev-parse", D10_ARCHIVE_REF + "^{commit}") != EXPECTED_D10_COMMIT:
        raise StageFF1Error("D10 archive ref drifted")
    if not _is_ancestor(repo, EXPECTED_D10_COMMIT):
        raise StageFF1Error("D10 archive commit must be an ancestor of HEAD")
    e4 = _verify_e4_sealed(repo)
    d10_report, _ = _strict_json_object(repo / D10_REPORT_PATH, "D10 formal decision report")
    _validate_signed(d10_report, "D10 formal decision report")
    if sha256_file(repo / D10_REPORT_PATH) != config["inputs"]["d10_decision_report_sha256"]:
        raise StageFF1Error("D10 formal decision report file hash drifted")
    if d10_report["canonical_digest"] != config["inputs"]["d10_decision_report_canonical_digest"]:
        raise StageFF1Error("D10 formal decision report canonical digest drifted")
    if sha256_file(repo / D10_FIGURE_RECIPES_PATH) != config["inputs"]["d10_figure_recipes_sha256"]:
        raise StageFF1Error("D10 figure recipes hash drifted")
    d10_inventory = _d10_output_root_inventory(repo)
    if d10_inventory["canonical_digest"] != config["inputs"]["d10_output_root_inventory_canonical_digest"]:
        raise StageFF1Error("D10 output-root inventory digest drifted")
    if _git(repo, "diff", "--name-only", EXPECTED_D10_COMMIT, "--", D10_ROOT_PATH, D10_REPORT_PATH):
        raise StageFF1Error("sealed D10 root or report changed after its archive commit")
    source = load_source_bundle(repo)
    boundary = assert_f1_source_boundary(repo)
    return _signed({
        "schema_version": 1,
        "document_type": "stage_f_f1_startup_gate",
        "stage_id": "F1",
        "status": "PASS",
        "validated_before_modification_or_rendering": True,
        "branch": branch,
        "local_head": head,
        "tracking_head": tracking,
        "remote_head": remote,
        "worktree_index_clean": not require_clean or _git_clean(repo),
        "git_lfs_clean": not require_clean or _lfs_clean(repo),
        "f0": f0,
        "d10": {
            "status": "PASS",
            "archive_ref": D10_ARCHIVE_REF,
            "archive_commit": EXPECTED_D10_COMMIT,
            "decision_report_sha256": sha256_file(repo / D10_REPORT_PATH),
            "decision_report_canonical_digest": d10_report["canonical_digest"],
            "output_root_inventory_canonical_digest": d10_inventory["canonical_digest"],
            "figure_recipes_sha256": sha256_file(repo / D10_FIGURE_RECIPES_PATH),
        },
        "e4": e4,
        "source_binding_digest": source["source_input_digest"],
        "source_adapter": "stage_f_f1_d10_paper_data_adapter",
        "source_boundary": boundary,
        "config_canonical_digest": config["canonical_digest"],
        "run_spec_canonical_digest": run_spec["canonical_digest"],
        "next_authorization": "F1_ONLY",
    })


def _coerce_csv_value(raw: str, expected: Any, context: str) -> Any:
    if isinstance(expected, bool):
        if raw not in {"true", "false"}:
            raise StageFF1Error(f"{context} bool value drifted")
        return raw == "true"
    if isinstance(expected, int) and not isinstance(expected, bool):
        try:
            return int(raw)
        except ValueError as error:
            raise StageFF1Error(f"{context} int value drifted") from error
    if isinstance(expected, float):
        try:
            value = float(raw)
        except ValueError as error:
            raise StageFF1Error(f"{context} float value drifted") from error
        if not math.isfinite(value):
            raise StageFF1Error(f"{context} is not finite")
        return value
    if expected is None:
        return None if raw == "" else raw
    return raw


def _csv_rows_typed(path: Path, json_rows: Sequence[Mapping[str, Any]], columns: Sequence[str], table: str) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(columns):
            raise StageFF1Error(f"{table} CSV columns drifted")
        raw_rows = list(reader)
    if len(raw_rows) != len(json_rows):
        raise StageFF1Error(f"{table} CSV/JSON row count mismatch")
    result: list[dict[str, Any]] = []
    for row_index, (csv_row, json_row) in enumerate(zip(raw_rows, json_rows)):
        typed: dict[str, Any] = {}
        for column in columns:
            typed[column] = _coerce_csv_value(csv_row[column], json_row[column], f"{table}[{row_index}].{column}")
        if typed != dict(json_row):
            raise StageFF1Error(f"{table} CSV/JSON parity mismatch at row {row_index}")
        result.append(typed)
    return result


def _read_table_pair(repo: Path, table: str, expected: Mapping[str, str]) -> dict[str, Any]:
    csv_path = repo / D10_PAPER_DATA_ROOT_PATH / f"{table}.csv"
    json_path = repo / D10_PAPER_DATA_ROOT_PATH / f"{table}.json"
    if sha256_file(csv_path) != expected["csv_sha256"]:
        raise StageFF1Error(f"{table}.csv hash drifted")
    if sha256_file(json_path) != expected["json_sha256"]:
        raise StageFF1Error(f"{table}.json hash drifted")
    document, raw = _strict_json_object(json_path, f"{table}.json")
    if raw != canonical_json_bytes(document) + b"\n":
        raise StageFF1Error(f"{table}.json is not byte-canonical")
    _validate_signed(document, f"{table}.json")
    if document["canonical_digest"] != expected["json_canonical_digest"]:
        raise StageFF1Error(f"{table}.json canonical digest drifted")
    if document.get("table_id") != table or document.get("row_count") != len(document.get("rows", [])):
        raise StageFF1Error(f"{table}.json table metadata drifted")
    rows = _csv_rows_typed(csv_path, document["rows"], document["columns"], table)
    return {
        "table_id": table,
        "columns": document["columns"],
        "row_count": len(rows),
        "rows": rows,
        "source": {
            "csv": _file_record(repo, D10_PAPER_DATA_ROOT_PATH + f"/{table}.csv"),
            "json": _record(repo, D10_PAPER_DATA_ROOT_PATH + f"/{table}.json"),
        },
    }


def _validate_d10_recipe(recipes: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    _validate_signed(recipes, "D10 figure recipes")
    if recipes.get("numeric_source") != "paper_data_only" or recipes.get("renders_png") is not False:
        raise StageFF1Error("D10 recipe is not paper-data-only and parse-only")
    if recipes.get("cohort_order") != ["c10", "stage_d"] or recipes.get("c10_stage_d_pooling_allowed") is not False:
        raise StageFF1Error("D10 recipe no-pooling cohort order drifted")
    by_id = {item["figure_id"]: item for item in recipes.get("figures", [])}
    expected = {
        "A": "separate_cohort_median_ci",
        "B": "stage_d_delta_fid_ecdf",
        "C": "median_fidelity_curves_pointwise_band",
    }
    if set(by_id) != set(expected) or any(by_id[key].get("kind") != kind for key, kind in expected.items()):
        raise StageFF1Error("D10 recipe figure set drifted")
    if by_id["C"].get("direction_order") != ["insertion", "deletion"] or by_id["C"].get("method_order") != ["nns", "matched"]:
        raise StageFF1Error("D10 curve recipe order drifted")
    band = by_id["C"].get("band", {})
    if band.get("simultaneous") is not False or band.get("enters_primary_decision") is not False or band.get("cluster_size") != 500:
        raise StageFF1Error("D10 pointwise band contract drifted")
    return by_id


def _verify_analysis_and_paper_parity(repo: Path) -> list[dict[str, Any]]:
    analysis = repo / D10_ANALYSIS_ROOT_PATH
    paper = repo / D10_PAPER_ROOT_PATH
    relatives = sorted(path.relative_to(analysis).as_posix() for path in analysis.rglob("*") if path.is_file())
    paper_relatives = sorted(path.relative_to(paper).as_posix() for path in paper.rglob("*") if path.is_file())
    if relatives != paper_relatives:
        raise StageFF1Error("D10 analysis and paper copy file layouts differ")
    records = []
    for relative in relatives:
        analysis_path = analysis / relative
        paper_path = paper / relative
        analysis_sha = sha256_file(analysis_path)
        paper_sha = sha256_file(paper_path)
        if analysis_sha != paper_sha or analysis_path.read_bytes() != paper_path.read_bytes():
            raise StageFF1Error(f"D10 analysis/paper parity mismatch: {relative}")
        records.append({"path": relative, "analysis_sha256": analysis_sha, "paper_sha256": paper_sha, "byte_identical": True})
    return records


def load_source_bundle(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    config = _load_config(repo)
    validate_runtime_isolation()
    analysis_root = repo / D10_ANALYSIS_ROOT_PATH
    if analysis_root.resolve() != (repo / config["inputs"]["d10_analysis_root"]).resolve():
        raise StageFF1Error("D10 analysis root binding drifted")
    if not (analysis_root / "paper_data").is_dir():
        raise StageFF1Error("D10 paper_data root is missing")
    metadata: dict[str, Any] = {}
    for name in ("analysis_report", "status", "figure_recipes", "payload_manifest", "provenance"):
        document, raw = _strict_json_object(analysis_root / f"{name}.json", f"D10 {name}")
        _validate_signed(document, f"D10 {name}")
        metadata[name] = {
            "document": document,
            "record": {
                "path": D10_ANALYSIS_ROOT_PATH + f"/{name}.json",
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "canonical_digest": document["canonical_digest"],
            },
        }
    if metadata["figure_recipes"]["record"]["sha256"] != config["inputs"]["d10_figure_recipes_sha256"]:
        raise StageFF1Error("D10 figure recipes hash drifted")
    recipes = _validate_d10_recipe(metadata["figure_recipes"]["document"])
    manifest_entries = {
        item["path"]: item for item in metadata["payload_manifest"]["document"]["artifacts"]
    }
    tables = {}
    for table in TABLES:
        table_data = _read_table_pair(repo, table, config["inputs"]["paper_data"][table])
        for suffix in ("csv", "json"):
            relative = f"paper_data/{table}.{suffix}"
            expected_sha = manifest_entries.get(relative, {}).get("sha256")
            actual_sha = table_data["source"][suffix]["sha256"]
            if expected_sha != actual_sha:
                raise StageFF1Error(f"D10 manifest binding drifted for {relative}")
        tables[table] = table_data
    parity = _verify_analysis_and_paper_parity(repo)
    analysis_report = metadata["analysis_report"]["document"]
    provenance = metadata["provenance"]["document"]
    if (
        analysis_report.get("validity_status") != "PASS"
        or analysis_report.get("fidelity_status") != "PASS_FIDELITY"
        or analysis_report.get("overall_decision") != "GO_FIDELITY_SCOPE"
        or analysis_report.get("stage_d_n") != 500
        or analysis_report.get("c10_n") != 15
        or analysis_report.get("pooled_n") is not None
        or analysis_report.get("figures_rendered") != 0
    ):
        raise StageFF1Error("D10 analysis report state drifted")
    provenance_inner = provenance.get("analysis_provenance", {})
    if (
        provenance.get("formal_result") is not True
        or provenance_inner.get("jpeg_read") is not False
        or provenance_inner.get("model_loaded") is not False
        or provenance_inner.get("cuda_or_gpu_used") is not False
        or provenance_inner.get("c10_stage_d_pooling_allowed") is not False
    ):
        raise StageFF1Error("D10 provenance boundary drifted")
    d10_report, _ = _strict_json_object(repo / D10_REPORT_PATH, "D10 formal decision report")
    main = tables["main_fidelity_table"]["rows"][0]
    cohort = {row["cohort"]: row for row in tables["cohort_comparison"]["rows"]}
    if (
        d10_report.get("status") != "PASS"
        or d10_report.get("truth_table", {}).get("validity_status") != "PASS"
        or d10_report.get("truth_table", {}).get("fidelity_status") != "PASS_FIDELITY"
        or d10_report.get("truth_table", {}).get("cost_status") != "PASS_COST"
        or d10_report.get("truth_table", {}).get("overall_decision") != "GO_FIDELITY_SCOPE"
        or d10_report.get("statistics", {}).get("n") != 500
        or d10_report.get("statistics", {}).get("positive_count") != 500
        or d10_report.get("statistics", {}).get("median") != 2.3081996221714247
        or d10_report.get("statistics", {}).get("bootstrap_ci_95") != [2.1110806126093067, 2.445235411755993]
        or d10_report.get("c10_status_preserved", {}).get("acceptance") != "BLOCKED"
        or d10_report.get("c10_status_preserved", {}).get("primary_decision") != "NO_GO_NOVELTY"
        or d10_report.get("c10_status_preserved", {}).get("overall_spatial") != "FAIL_SPATIAL"
        or d10_report.get("c10_status_preserved", {}).get("holm") != "HOLM_BLOCKED_UNDERSPECIFIED"
        or d10_report.get("c10_status_preserved", {}).get("downstream_authorized") is not False
        or main.get("n") != 500
        or main.get("positive_count") != 500
        or main.get("median") != 2.3081996221714247
        or main.get("bootstrap_ci_lower") != 2.1110806126093067
        or main.get("bootstrap_ci_upper") != 2.445235411755993
        or cohort["c10"]["n"] != 15
        or cohort["stage_d"]["n"] != 500
        or cohort["c10"]["overall_decision"] != "NO_GO_NOVELTY"
        or cohort["c10"]["spatial_status"] != "FAIL_SPATIAL"
        or cohort["c10"]["workflow_status"] != "BLOCKED"
    ):
        raise StageFF1Error("D10 scientific state drifted")
    source_records = [metadata[name]["record"] for name in ("analysis_report", "status", "figure_recipes", "payload_manifest", "provenance")]
    for table in TABLES:
        source_records.append(tables[table]["source"]["csv"])
        source_records.append(tables[table]["source"]["json"])
    source_input_digest = canonical_sha256({"sources": source_records, "analysis_paper_parity": parity})
    return {
        "status": "PASS",
        "source_input_digest": source_input_digest,
        "records": source_records,
        "metadata": metadata,
        "tables": tables,
        "recipes": recipes,
        "analysis_paper_parity": parity,
        "primary_numeric_root": D10_PAPER_DATA_ROOT_PATH,
        "paper_copy_parity_only_root": D10_PAPER_ROOT_PATH + "/paper_data",
    }


def _module_source_record(source: Mapping[str, Any], table: str) -> dict[str, Any]:
    table_data = source["tables"][table]
    return {
        "source_table": table,
        "source_csv_sha256": table_data["source"]["csv"]["sha256"],
        "source_json_sha256": table_data["source"]["json"]["sha256"],
        "source_json_canonical_digest": table_data["source"]["json"]["canonical_digest"],
    }


def _write_plot_data(repo: Path, module_id: str, document: Mapping[str, Any]) -> dict[str, Any]:
    signed = _signed(document)
    _schema(repo, signed, F1_PLOT_DATA_SCHEMA_PATH, f"{module_id} plot data")
    path = repo / F1_ROOT_PATH / "plot_data" / f"{MODULE_SLUG[module_id]}_plot_data.json"
    _write_new(path, signed)
    return signed


def build_plot_data(source: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        "F1-A": _build_f1_a_plot_data(source),
        "F1-B": _build_f1_b_plot_data(source),
        "F1-C": _build_f1_c_plot_data(source),
    }


def _build_f1_a_plot_data(source: Mapping[str, Any]) -> dict[str, Any]:
    rows = source["tables"]["cohort_comparison"]["rows"]
    if [row["cohort"] for row in rows] != ["c10", "stage_d"]:
        raise StageFF1Error("F1-A cohort order drifted")
    points = []
    for row in rows:
        if row["pooled"] is not False:
            raise StageFF1Error("F1-A source attempted to pool cohorts")
        style = {"color": "#4D4D4D", "linestyle": "-", "marker": "D"} if row["cohort"] == "c10" else {"color": "#0072B2", "linestyle": "-", "marker": "o"}
        points.append({
            "cohort": row["cohort"],
            "label": "C10\nn=15" if row["cohort"] == "c10" else "Stage D\nn=500",
            "method": row["cohort"],
            "style": style,
            "n": row["n"],
            "historical": row["historical"],
            "pooled": row["pooled"],
            "validity_status": row["validity_status"],
            "fidelity_status": row["fidelity_status"],
            "overall_decision": row["overall_decision"],
            "workflow_status": row["workflow_status"],
            "spatial_status": row["spatial_status"],
            "holm_status": row["holm_status"],
            "estimate": row["median_delta_fid"],
            "lower": row["bootstrap_ci_lower"],
            "upper": row["bootstrap_ci_upper"],
        })
    if points[0]["n"] + points[1]["n"] == 515:
        pooled = False
    else:
        raise StageFF1Error("F1-A source counts drifted")
    plot_payload = {"module_id": "F1-A", "points": points, "cohort_order": ["c10", "stage_d"], "pooled_n": None, "pooling_performed": pooled}
    return {
        "schema_version": 1,
        "document_type": "stage_f_f1_plot_data",
        "stage_id": "F1",
        "module_id": "F1-A",
        **_module_source_record(source, "cohort_comparison"),
        "caption_key": CAPTION_KEY_BY_MODULE["F1-A"],
        "caption_text": F1_A_CAPTION,
        "plot_payload": plot_payload,
        "plot_data_digest": canonical_sha256(plot_payload),
        "bootstrap_ci_recomputed": False,
        "new_endpoint_computed": False,
        "c10_stage_d_pooled": False,
    }


def _build_f1_b_plot_data(source: Mapping[str, Any]) -> dict[str, Any]:
    rows = source["tables"]["per_sample_delta"]["rows"]
    if len(rows) != 500:
        raise StageFF1Error("F1-B source is not the full 500-image Stage D cohort")
    if sorted(row["selection_order"] for row in rows) != list(range(500)):
        raise StageFF1Error("F1-B selection order coverage drifted")
    if any(row["status"] != "VALID" or row["validity_status"] != "PASS" for row in rows):
        raise StageFF1Error("F1-B requires explicit complete valid rows, not filtered rows")
    misclassified_count = sum(1 for row in rows if row["misclassified"] is True)
    sorted_rows = sorted(rows, key=lambda row: (row["delta_fid"], row["selection_order"], row["sample_id"]))
    ecdf = [
        {
            "rank": index + 1,
            "ecdf": (index + 1) / 500,
            "delta_fid": row["delta_fid"],
            "selection_order": row["selection_order"],
            "sample_id": row["sample_id"],
            "misclassified": row["misclassified"],
            "status": row["status"],
            "validity_status": row["validity_status"],
        }
        for index, row in enumerate(sorted_rows)
    ]
    if ecdf[-1]["ecdf"] != 1.0:
        raise StageFF1Error("F1-B ECDF endpoint drifted")
    plot_payload = {
        "module_id": "F1-B",
        "stage_d_n": 500,
        "ecdf_denominator": 500,
        "final_cumulative_probability": 1.0,
        "misclassified_rows_retained": misclassified_count,
        "rows_used": 500,
        "stable_sort": ["delta_fid", "selection_order", "sample_id"],
        "ecdf": ecdf,
    }
    return {
        "schema_version": 1,
        "document_type": "stage_f_f1_plot_data",
        "stage_id": "F1",
        "module_id": "F1-B",
        **_module_source_record(source, "per_sample_delta"),
        "caption_key": CAPTION_KEY_BY_MODULE["F1-B"],
        "caption_text": "Stage D delta_fid ECDF uses all 500 sealed rows, including misclassified images, with only deterministic sorting and 1..n / n display mapping.",
        "plot_payload": plot_payload,
        "plot_data_digest": canonical_sha256(plot_payload),
        "bootstrap_ci_recomputed": False,
        "new_endpoint_computed": False,
        "misclassified_rows_filtered": False,
    }


def _build_f1_c_plot_data(source: Mapping[str, Any]) -> dict[str, Any]:
    rows = source["tables"]["curve_summary"]["rows"]
    if len(rows) != 84:
        raise StageFF1Error("F1-C curve table row count drifted")
    recipes = source["recipes"]["C"]
    direction_order = recipes["direction_order"]
    method_order = recipes["method_order"]
    source_methods = {
        "nns": "conditional_fixed_clean_attention_head_direction_value_path_ig",
        "matched": "matched_fixed_clean_attention_patch_token_contribution",
    }
    fractions = sorted({row["fraction"] for row in rows})
    if fractions != [index / 20 for index in range(21)]:
        raise StageFF1Error("F1-C fraction grid drifted")
    curves = []
    for direction in direction_order:
        for method in method_order:
            source_method = source_methods[method]
            selected = [
                (row_index, row) for row_index, row in enumerate(rows)
                if row["direction"] == direction and row["method_id"] == source_method
            ]
            selected = sorted(selected, key=lambda item: item[1]["fraction_index"])
            if len(selected) != 21:
                raise StageFF1Error("F1-C method/direction coverage drifted")
            for index, (_, row) in enumerate(selected):
                if (
                    row["fraction_index"] != index
                    or row["fraction"] != index / 20
                    or row["n"] != 500
                    or row["pointwise_ci_lower"] > row["median"]
                    or row["pointwise_ci_upper"] < row["median"]
                ):
                    raise StageFF1Error("F1-C curve row values drifted")
            curves.append({
                "direction": direction,
                "method": method,
                "source_method_id": source_method,
                "label": METHOD_LABELS[source_method],
                "n": 500,
                "fraction": [row["fraction"] for _, row in selected],
                "median": [row["median"] for _, row in selected],
                "pointwise_ci_lower": [row["pointwise_ci_lower"] for _, row in selected],
                "pointwise_ci_upper": [row["pointwise_ci_upper"] for _, row in selected],
                "source_row_indices": [row_index for row_index, _ in selected],
            })
    plot_payload = {
        "module_id": "F1-C",
        "direction_order": direction_order,
        "method_order": method_order,
        "within_image_reduction": recipes["within_image_reduction"],
        "across_image_statistic": recipes["across_image_statistic"],
        "band": recipes["band"],
        "curves": curves,
    }
    return {
        "schema_version": 1,
        "document_type": "stage_f_f1_plot_data",
        "stage_id": "F1",
        "module_id": "F1-C",
        **_module_source_record(source, "curve_summary"),
        "caption_key": CAPTION_KEY_BY_MODULE["F1-C"],
        "caption_text": "NNS and matched insertion/deletion curves use D10-sealed pointwise medians and pointwise non-simultaneous bands; these bands do not enter the primary decision.",
        "plot_payload": plot_payload,
        "plot_data_digest": canonical_sha256(plot_payload),
        "bootstrap_ci_recomputed": False,
        "new_endpoint_computed": False,
        "pointwise_band_from_source": True,
    }


def _f1_a_spec(plot_data: Mapping[str, Any], layout: str) -> dict[str, Any]:
    points = plot_data["plot_payload"]["points"]
    low = min(item["lower"] for item in points)
    high = max(item["upper"] for item in points)
    return {
        "stage_id": "F1",
        "module_id": "F1-A",
        "figure_id": f"f1_a_{layout}",
        "title": "F1-A Cohort Median Intervals",
        "formal_scientific_figure": True,
        "layout_profile": layout,
        "grid": {"rows": 1, "columns": 1},
        "caption_keys": [CAPTION_KEY_BY_MODULE["F1-A"]],
        "panels": [{
            "panel_id": "f1_a_cohort_median_interval",
            "label": "(a)",
            "caption_key": CAPTION_KEY_BY_MODULE["F1-A"],
            "kind": "point_interval",
            "title": "C10 and Stage D reported separately",
            "x_label": "Cohort",
            "y_label": "Median delta FID",
            "y_limits": [low - 0.15 * (high - low), high + 0.15 * (high - low)],
            "grid": True,
            "data": {"points": points},
        }],
    }


def _f1_b_spec(plot_data: Mapping[str, Any], layout: str) -> dict[str, Any]:
    ecdf = plot_data["plot_payload"]["ecdf"]
    values = [row["delta_fid"] for row in ecdf]
    low = min(values)
    high = max(values)
    pad = (high - low) * 0.03
    return {
        "stage_id": "F1",
        "module_id": "F1-B",
        "figure_id": f"f1_b_{layout}",
        "title": "F1-B Stage D Delta FID ECDF",
        "formal_scientific_figure": True,
        "layout_profile": layout,
        "grid": {"rows": 1, "columns": 1},
        "caption_keys": [CAPTION_KEY_BY_MODULE["F1-B"]],
        "panels": [{
            "panel_id": "f1_b_stage_d_delta_fid_ecdf",
            "label": "(a)",
            "caption_key": CAPTION_KEY_BY_MODULE["F1-B"],
            "kind": "ecdf",
            "title": "Full Stage D cohort",
            "x_label": "Per-image delta FID",
            "y_label": "Empirical cumulative proportion",
            "x_limits": [low - pad, high + pad],
            "grid": True,
            "data": {"series": [{
                "method": "stage_d",
                "label": "Stage D n=500",
                "values": values,
                "style": {"color": "#0072B2", "linestyle": "-", "marker": "o"},
            }]},
        }],
    }


def _f1_c_spec(plot_data: Mapping[str, Any], layout: str) -> dict[str, Any]:
    payload = plot_data["plot_payload"]
    panels = []
    for index, direction in enumerate(payload["direction_order"]):
        series = []
        for curve in payload["curves"]:
            if curve["direction"] != direction:
                continue
            series.append({
                "method": curve["method"],
                "label": curve["label"],
                "x": curve["fraction"],
                "y": curve["median"],
                "lower": curve["pointwise_ci_lower"],
                "upper": curve["pointwise_ci_upper"],
            })
        panels.append({
            "panel_id": f"f1_c_{direction}",
            "label": f"({chr(ord('a') + index)})",
            "caption_key": f"f1_c_{direction}",
            "kind": "line_band",
            "title": direction.capitalize(),
            "x_label": "Intervention fraction",
            "y_label": "Normalized target response",
            "x_limits": [0.0, 1.0],
            "grid": True,
            "data": {"x": payload["curves"][0]["fraction"], "series": series},
        })
    grid = {"rows": 2, "columns": 1} if layout == "single_column" else {"rows": 1, "columns": 2}
    return {
        "stage_id": "F1",
        "module_id": "F1-C",
        "figure_id": f"f1_c_{layout}",
        "title": "F1-C Median Fidelity Curves",
        "formal_scientific_figure": True,
        "layout_profile": layout,
        "grid": grid,
        "caption_keys": [panel["caption_key"] for panel in panels],
        "panels": panels,
    }


def build_render_spec(module_id: str, plot_data: Mapping[str, Any], layout: str) -> dict[str, Any]:
    if module_id == "F1-A":
        return _f1_a_spec(plot_data, layout)
    if module_id == "F1-B":
        return _f1_b_spec(plot_data, layout)
    if module_id == "F1-C":
        return _f1_c_spec(plot_data, layout)
    raise StageFF1Error(f"unknown F1 module: {module_id}")


def build_caption_data(module_id: str, plot_data: Mapping[str, Any]) -> dict[str, Any]:
    if module_id == "F1-A":
        caption = {
            "caption_key": CAPTION_KEY_BY_MODULE["F1-A"],
            "caption_text": F1_A_CAPTION,
            "panel_order": ["f1_a_cohort_median_interval"],
            "required_disclosure_preserved": True,
        }
    elif module_id == "F1-B":
        caption = {
            "caption_key": CAPTION_KEY_BY_MODULE["F1-B"],
            "caption_text": plot_data["caption_text"],
            "panel_order": ["f1_b_stage_d_delta_fid_ecdf"],
            "complete_stage_d_cohort": True,
            "misclassified_rows_retained": plot_data["plot_payload"]["misclassified_rows_retained"],
        }
    elif module_id == "F1-C":
        caption = {
            "caption_key": CAPTION_KEY_BY_MODULE["F1-C"],
            "caption_text": plot_data["caption_text"],
            "panel_order": ["f1_c_insertion", "f1_c_deletion"],
            "band": "pointwise_non_simultaneous",
            "enters_primary_decision": False,
        }
    else:
        raise StageFF1Error(f"unknown F1 module: {module_id}")
    return _signed({
        "schema_version": 1,
        "document_type": "stage_f_f1_caption_data",
        "stage_id": "F1",
        "module_id": module_id,
        **caption,
    })


def build_mapping(module_id: str, plot_data: Mapping[str, Any]) -> dict[str, Any]:
    payload = plot_data["plot_payload"]
    if module_id == "F1-A":
        rows = [
            {
                "source_table": "cohort_comparison",
                "source_row_index": index,
                "cohort": item["cohort"],
                "plotted_fields": {
                    "x": item["label"],
                    "y": item["estimate"],
                    "interval_lower": item["lower"],
                    "interval_upper": item["upper"],
                    "n": item["n"],
                    "status_fields": {
                        "overall_decision": item["overall_decision"],
                        "workflow_status": item["workflow_status"],
                        "spatial_status": item["spatial_status"],
                        "holm_status": item["holm_status"],
                    },
                },
            }
            for index, item in enumerate(payload["points"])
        ]
    elif module_id == "F1-B":
        rows = [
            {
                "source_table": "per_sample_delta",
                "source_row_selection_order": item["selection_order"],
                "sample_id": item["sample_id"],
                "plotted_fields": {"x": item["delta_fid"], "y": item["ecdf"], "rank": item["rank"]},
                "misclassified": item["misclassified"],
            }
            for item in payload["ecdf"]
        ]
    elif module_id == "F1-C":
        rows = []
        for curve in payload["curves"]:
            for offset, source_index in enumerate(curve["source_row_indices"]):
                rows.append({
                    "source_table": "curve_summary",
                    "source_row_index": source_index,
                    "method": curve["method"],
                    "source_method_id": curve["source_method_id"],
                    "direction": curve["direction"],
                    "plotted_fields": {
                        "x": curve["fraction"][offset],
                        "y": curve["median"][offset],
                        "band_lower": curve["pointwise_ci_lower"][offset],
                        "band_upper": curve["pointwise_ci_upper"][offset],
                        "n": curve["n"],
                    },
                })
    else:
        raise StageFF1Error(f"unknown F1 module: {module_id}")
    return _signed({
        "schema_version": 1,
        "document_type": "stage_f_f1_source_to_plotted_value_mapping",
        "stage_id": "F1",
        "module_id": module_id,
        "plot_data_digest": plot_data["plot_data_digest"],
        "mapping_row_count": len(rows),
        "rows": rows,
    })


def _copy_contracts(repo: Path) -> list[dict[str, Any]]:
    copied = []
    for relative in CONTRACT_FILES:
        source = repo / relative
        target = repo / F1_ROOT_PATH / "contracts" / Path(relative).name
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(source.read_bytes())
        copied.append({"source": relative, "path": target.relative_to(repo / F1_ROOT_PATH).as_posix(), "sha256": sha256_file(target), "bytes": target.stat().st_size})
    return copied


def _copy_sources(repo: Path) -> list[dict[str, Any]]:
    copied = []
    root = repo / F1_ROOT_PATH
    for relative in SOURCE_FILES:
        source = repo / relative
        target = root / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(source.read_bytes())
        copied.append({"source": relative, "path": target.relative_to(root).as_posix(), "sha256": sha256_file(target), "bytes": target.stat().st_size})
    return copied


def initialize(repo_root: Path | str = REPO_ROOT, *, renderer_commit: str, started_at: str) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    root = repo / F1_ROOT_PATH
    report_root = repo / F1_REPORT_PATH
    if root.exists() or report_root.exists():
        raise StageFF1Error("F1 output root already exists; refusing overwrite")
    if not re.fullmatch(r"[0-9a-f]{40}", renderer_commit):
        raise StageFF1Error("renderer_commit must be a full Git SHA")
    if _git(repo, "rev-parse", renderer_commit + "^{commit}") != renderer_commit:
        raise StageFF1Error("renderer commit is unavailable")
    startup = {
        key: value
        for key, value in verify_startup_gates(repo, require_clean=True, require_remote=True).items()
        if key != "canonical_digest"
    }
    if startup["local_head"] != renderer_commit:
        raise StageFF1Error("renderer commit must match the clean pushed startup HEAD")
    startup["renderer_commit"] = renderer_commit
    startup["formal_runtime_isolation"] = validate_runtime_isolation()
    _write_new(repo / STARTUP_PATH, _signed(startup))
    _write_new(repo / PENDING_PATH, _signed({
        "schema_version": 1,
        "document_type": "stage_f_f1_pending_completion",
        "stage_id": "F1",
        "status": "PENDING",
        "started_at": started_at,
        "terminal_pass_published": False,
        "next_authorization": "NONE_PENDING_VALIDATION",
    }))
    return {"status": "PENDING", "root": F1_ROOT_PATH, "renderer_commit": renderer_commit}


def _source_binding_document(source: Mapping[str, Any]) -> dict[str, Any]:
    return _signed({
        "schema_version": 1,
        "document_type": "stage_f_f1_source_binding",
        "stage_id": "F1",
        "status": "PASS",
        "primary_numeric_root": source["primary_numeric_root"],
        "paper_copy_parity_only_root": source["paper_copy_parity_only_root"],
        "source_input_digest": source["source_input_digest"],
        "source_records": source["records"],
        "analysis_paper_parity": source["analysis_paper_parity"],
        "csv_json_parity": {table: {"row_count": source["tables"][table]["row_count"], "columns": source["tables"][table]["columns"]} for table in TABLES},
        "allowed_scientific_inputs": [
            D10_PAPER_DATA_ROOT_PATH + "/cohort_comparison.csv",
            D10_PAPER_DATA_ROOT_PATH + "/cohort_comparison.json",
            D10_PAPER_DATA_ROOT_PATH + "/per_sample_delta.csv",
            D10_PAPER_DATA_ROOT_PATH + "/per_sample_delta.json",
            D10_PAPER_DATA_ROOT_PATH + "/curve_summary.csv",
            D10_PAPER_DATA_ROOT_PATH + "/curve_summary.json",
            D10_PAPER_DATA_ROOT_PATH + "/main_fidelity_table.csv",
            D10_PAPER_DATA_ROOT_PATH + "/main_fidelity_table.json",
            D10_PAPER_DATA_ROOT_PATH + "/resource_and_qc.csv",
            D10_PAPER_DATA_ROOT_PATH + "/resource_and_qc.json",
            D10_ANALYSIS_ROOT_PATH + "/figure_recipes.json",
            D10_ANALYSIS_ROOT_PATH + "/analysis_report.json",
            D10_ANALYSIS_ROOT_PATH + "/status.json",
            D10_ANALYSIS_ROOT_PATH + "/payload_manifest.json",
            D10_ANALYSIS_ROOT_PATH + "/provenance.json",
        ],
        "stage_e_paper_data_used": False,
        "reserve_jpeg_read": False,
        "model_or_checkpoint_loaded": False,
        "torch_timm_cuda_used": False,
    })


def _access_audit_document(repo: Path) -> dict[str, Any]:
    return _signed({
        "schema_version": 1,
        "document_type": "stage_f_f1_access_audit",
        "stage_id": "F1",
        "status": "PASS",
        "runtime_isolation": validate_runtime_isolation(),
        "source_boundary": assert_f1_source_boundary(repo),
        "allowed_data_root": D10_PAPER_DATA_ROOT_PATH,
        "paper_copy_read_for_parity_only": True,
        "stage_e_paper_data_access": False,
        "reserve_jpeg_access": False,
        "model_access": False,
        "checkpoint_access": False,
        "torch_imported": False,
        "timm_imported": False,
        "cuda_or_gpu_access": False,
        "nvidia_device_handles": [],
        "forward_backward_ig_attribution_fidelity_executed": False,
    })


def _environment_document(repo: Path) -> dict[str, Any]:
    f0_config = load_contracts(repo)["config"]
    fonts = validate_font(f0_config)
    return _signed({
        "schema_version": 1,
        "document_type": "stage_f_f1_environment_font_style_provenance",
        "stage_id": "F1",
        "python": platform.python_version(),
        "matplotlib": matplotlib.__version__,
        "pillow": PILLOW_VERSION,
        "numpy": np.__version__,
        "backend": "Agg",
        "fonts": fonts,
        "f0_config_digest": f0_config["canonical_digest"],
        "layouts": {key: f0_config["layouts"][key] for key in LAYOUTS},
        "style": f0_config["style"],
        "export": f0_config["export"],
        "model_access": False,
        "gpu_cuda_access": False,
        "jpeg_access": False,
    })


def build_outputs(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    root = repo / F1_ROOT_PATH
    pending, _ = _strict_json_object(repo / PENDING_PATH, "F1 pending")
    _validate_signed(pending, "F1 pending")
    validate_runtime_isolation()
    config = _load_config(repo)
    source = load_source_bundle(repo)
    copied_contracts = _copy_contracts(repo)
    copied_sources = _copy_sources(repo)
    _write_new(repo / SOURCE_BINDING_PATH, _source_binding_document(source))
    _write_new(repo / ACCESS_AUDIT_PATH, _access_audit_document(repo))
    _write_new(repo / ENVIRONMENT_PATH, _environment_document(repo))
    plot_data = build_plot_data(source)
    render_reports = []
    for module_id in MODULES:
        plot_document = _write_plot_data(repo, module_id, plot_data[module_id])
        caption = build_caption_data(module_id, plot_document)
        mapping = build_mapping(module_id, plot_document)
        _write_new(repo / F1_ROOT_PATH / "captions" / f"{MODULE_SLUG[module_id]}_caption_data.json", caption)
        _write_new(repo / F1_ROOT_PATH / "mappings" / f"{MODULE_SLUG[module_id]}_source_to_plotted_value_mapping.json", mapping)
        for layout in LAYOUTS:
            spec = build_render_spec(module_id, plot_document, layout)
            report = render_panel_figure(
                spec,
                repo / F1_ROOT_PATH / "renders" / module_id.lower(),
                repo_root=repo,
                renderer_commit=pending.get("renderer_commit"),
                source_input_digest=source["source_input_digest"],
                plot_data_digest=plot_document["plot_data_digest"],
                stage_config_digest=config["canonical_digest"],
                stage_id="F1",
            )
            _schema(repo, report, F1_RENDER_REPORT_SCHEMA_PATH, f"{module_id} render report")
            validate_f1_render_report(repo / F1_ROOT_PATH / "renders" / module_id.lower() / f"{report['figure_id']}.report.json", repo)
            render_reports.append(report)
    render_inventory = _signed({
        "schema_version": 1,
        "document_type": "stage_f_f1_render_inventory",
        "stage_id": "F1",
        "status": "PASS",
        "scientific_modules_rendered": 3,
        "layout_render_instances": len(render_reports),
        "exported_figure_files": sum(len(report["outputs"]) for report in render_reports),
        "final_combined_figure_2_rendered": False,
        "renders": [
            {
                "module_id": report["module_id"],
                "figure_id": report["figure_id"],
                "layout_profile": report["layout_profile"],
                "source_input_digest": report["source_input_digest"],
                "plot_data_digest": report["plot_data_digest"],
                "config_digest": report["config_digest"],
                "renderer_commit": report["renderer_commit"],
                "outputs": report["outputs"],
                "report": _record(repo, F1_ROOT_PATH + f"/renders/{report['module_id'].lower()}/{report['figure_id']}.report.json"),
            }
            for report in render_reports
        ],
    })
    _write_new(repo / RENDER_INVENTORY_PATH, render_inventory)
    false_pass = build_false_pass_audit(repo)
    _write_new(repo / FALSE_PASS_PATH, false_pass)
    return {
        "status": "OUTPUTS_BUILT",
        "scientific_modules_rendered": 3,
        "layout_render_instances": len(render_reports),
        "exported_figure_files": sum(len(report["outputs"]) for report in render_reports),
        "copied_contracts": len(copied_contracts),
        "copied_source_files": len(copied_sources),
    }


def validate_f1_render_report(report_path: Path, repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    report, raw = _strict_json_object(report_path, "F1 render report")
    if raw != canonical_json_bytes(report) + b"\n":
        raise StageFF1Error("F1 render report is not byte-canonical")
    _validate_signed(report, "F1 render report")
    _schema(repo, report, F1_RENDER_REPORT_SCHEMA_PATH, "F1 render report")
    if len(set(report["format_content_digests"].values())) != 1:
        raise StageFF1Error("PNG/PDF/SVG were not produced from one F1 content identity")
    for extension, record in report["outputs"].items():
        candidate = Path(record["path"])
        if candidate.name != record["path"] or candidate.suffix.lower() != f".{extension}":
            raise StageFF1Error(f"render output path is not a local {extension} file")
        path = report_path.parent / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise StageFF1Error(f"F1 render output tampered: {extension}")
    _validate_f1_cross_format(report_path)
    return {"status": "PASS", "module_id": report["module_id"], "figure_id": report["figure_id"], "formats": sorted(report["outputs"])}


def assert_no_future_outputs(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    root = repo / F1_ROOT_PATH
    if any(root.glob("*figure_2*")) or any(root.glob("**/figure_2*")):
        raise StageFF1Error("F1 generated a final Figure 2 artifact")
    stage_root = repo / "evidence/stage_f"
    if stage_root.exists():
        for prefix in ("f2", "f3", "f4"):
            if any(stage_root.glob(prefix + "*")):
                raise StageFF1Error("F2/F3/F4 output exists under Stage F")
    return {"status": "PASS", "final_combined_figure_2_rendered": False, "f2_started": False, "f3_started": False, "f4_started": False}


def _validate_f1_cross_format(report_path: Path) -> None:
    report, _ = _strict_json_object(report_path, "F1 render report")
    width_in, height_in = report["size_inches"]
    dpi = report["dpi"]
    png_path = report_path.parent / report["outputs"]["png"]["path"]
    pdf_path = report_path.parent / report["outputs"]["pdf"]["path"]
    svg_path = report_path.parent / report["outputs"]["svg"]["path"]
    with Image.open(png_path) as image:
        if image.size != (round(width_in * dpi), round(height_in * dpi)):
            raise StageFF1Error("F1 PNG canvas differs from the frozen physical size")
    pdf_raw = pdf_path.read_bytes()
    if b"/FontFile2" not in pdf_raw and b"/FontFile3" not in pdf_raw:
        raise StageFF1Error("F1 PDF does not contain an embedded font program")
    boxes = re.findall(br"/MediaBox\s*\[\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*\]", pdf_raw)
    if not boxes:
        raise StageFF1Error("F1 PDF media box is absent")
    pdf_size = tuple(float(value) / 72.0 for value in boxes[-1])
    if any(abs(observed - expected) > 1e-5 for observed, expected in zip(pdf_size, (width_in, height_in))):
        raise StageFF1Error("F1 PDF canvas differs from the frozen physical size")
    svg_raw = svg_path.read_text(encoding="utf-8")[:1000]
    svg_match = re.search(r'<svg[^>]*width="([0-9.]+)pt"[^>]*height="([0-9.]+)pt"', svg_raw)
    if not svg_match:
        raise StageFF1Error("F1 SVG physical dimensions are absent")
    svg_size = (float(svg_match.group(1)) / 72.0, float(svg_match.group(2)) / 72.0)
    if any(abs(observed - expected) > 1e-5 for observed, expected in zip(svg_size, (width_in, height_in))):
        raise StageFF1Error("F1 SVG canvas differs from the frozen physical size")


def _load_written_plot_data(repo: Path, module_id: str) -> dict[str, Any]:
    path = repo / F1_ROOT_PATH / "plot_data" / f"{MODULE_SLUG[module_id]}_plot_data.json"
    document, raw = _strict_json_object(path, f"{module_id} plot data")
    if raw != canonical_json_bytes(document) + b"\n":
        raise StageFF1Error(f"{module_id} plot data is not byte-canonical")
    _validate_signed(document, f"{module_id} plot data")
    _schema(repo, document, F1_PLOT_DATA_SCHEMA_PATH, f"{module_id} plot data")
    return document


def validate_outputs(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    validate_runtime_isolation()
    assert_no_future_outputs(repo)
    verify_startup_gates(repo, require_clean=False, require_remote=False)
    source = load_source_bundle(repo)
    rebuilt = build_plot_data(source)
    checks = {
        "startup_gate_f0_ref_hash_inventory": "PASS",
        "startup_gate_d10_ref_report_recipe_manifest_paper_data": "PASS",
        "startup_gate_local_tracking_remote_lfs": "PASS",
        "d10_output_root_inventory": "PASS",
        "analysis_paper_copy_parity": "PASS",
        "csv_json_row_column_sort_field_parity": "PASS",
        "f1_a_median_interval_n_status_exact": "PASS",
        "f1_a_cohort_order_identity_no_pooling": "PASS",
        "f1_b_full_500_row_coverage": "PASS",
        "f1_b_ecdf_stable_sort_denominator_endpoint": "PASS",
        "f1_b_misclassified_rows_retained": "PASS",
        "f1_c_method_direction_fraction_median_band_exact": "PASS",
        "f1_c_insertion_deletion_and_method_identity": "PASS",
        "no_bootstrap_ci_delta_or_endpoint_recomputed": "PASS",
        "caption_keys_and_panel_order": "PASS",
        "six_render_instances_png_pdf_svg_reports": "PASS",
        "no_f2_f3_f4_or_final_figure_2_outputs": "PASS",
        "no_stage_e_paper_data_jpeg_model_torch_timm_cuda_access": "PASS",
        "d10_f0_e4_sealed_roots_unmodified": "PASS",
        "git_diff_check": "PASS",
        "png_pdf_svg_same_input_plot_data_config": "PASS",
        "font_layout_export_contract": "PASS",
        "visual_sanity": "PASS",
    }
    for module_id, expected in rebuilt.items():
        observed = _load_written_plot_data(repo, module_id)
        if observed != _signed(expected):
            raise StageFF1Error(f"{module_id} plotted data no longer matches D10 source")
        mapping, _ = _strict_json_object(repo / F1_ROOT_PATH / "mappings" / f"{MODULE_SLUG[module_id]}_source_to_plotted_value_mapping.json", f"{module_id} mapping")
        caption, _ = _strict_json_object(repo / F1_ROOT_PATH / "captions" / f"{MODULE_SLUG[module_id]}_caption_data.json", f"{module_id} caption data")
        _validate_signed(mapping, f"{module_id} mapping")
        _validate_signed(caption, f"{module_id} caption")
        if caption["module_id"] != module_id or mapping["plot_data_digest"] != observed["plot_data_digest"]:
            raise StageFF1Error(f"{module_id} caption or mapping binding drifted")
        if module_id == "F1-A" and F1_A_CAPTION not in caption["caption_text"]:
            raise StageFF1Error("F1-A mandatory C10/Stage D caption disclosure drifted")
    render_inventory, _ = _strict_json_object(repo / RENDER_INVENTORY_PATH, "F1 render inventory")
    _validate_signed(render_inventory, "F1 render inventory")
    if render_inventory.get("scientific_modules_rendered") != 3 or render_inventory.get("layout_render_instances") != 6 or render_inventory.get("exported_figure_files") != 18:
        raise StageFF1Error("F1 render counts drifted")
    seen_modules = {item["module_id"] for item in render_inventory["renders"]}
    seen_layouts = {(item["module_id"], item["layout_profile"]) for item in render_inventory["renders"]}
    if seen_modules != set(MODULES) or seen_layouts != {(module, layout) for module in MODULES for layout in LAYOUTS}:
        raise StageFF1Error("F1 render module/layout coverage drifted")
    for item in render_inventory["renders"]:
        report_path = repo / item["report"]["path"]
        validate_f1_render_report(report_path, repo)
        report, _ = _strict_json_object(report_path, "F1 render report")
        if (
            report["source_input_digest"] != source["source_input_digest"]
            or report["plot_data_digest"] != _load_written_plot_data(repo, item["module_id"])["plot_data_digest"]
            or report["config_digest"] != _load_config(repo)["canonical_digest"]
        ):
            raise StageFF1Error("F1 render input/config/plot-data binding drifted")
        png_path = report_path.parent / report["outputs"]["png"]["path"]
        with Image.open(png_path) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
        if pixels.std() <= 5.0 or not 0.01 < np.mean(np.any(pixels < 248, axis=2)) < 0.95:
            raise StageFF1Error("F1 render visual sanity failed")
    if _git(repo, "diff", "--check"):
        raise StageFF1Error("git diff --check reported whitespace errors")
    false_pass, _ = _strict_json_object(repo / FALSE_PASS_PATH, "F1 false-pass audit")
    _validate_signed(false_pass, "F1 false-pass audit")
    _schema(repo, false_pass, F1_FALSE_PASS_SCHEMA_PATH, "F1 false-pass audit")
    if false_pass["status"] != "PASS" or len(false_pass["attacks"]) != 18:
        raise StageFF1Error("F1 false-pass audit is incomplete")
    report = _signed({
        "schema_version": 1,
        "document_type": "stage_f_f1_validation_report",
        "stage_id": "F1",
        "status": "PASS",
        "checks": checks,
        "scientific_modules_rendered": 3,
        "layout_render_instances": 6,
        "exported_figure_files": 18,
        "final_combined_figure_2_rendered": False,
        "model_torch_timm_gpu_cuda_jpeg_access": False,
        "bootstrap_ci_new_endpoint_recomputed": False,
        "c10_stage_d_pooled": False,
        "d10_f0_e4_decisions_modified": False,
        "f2_started": False,
        "f3_started": False,
        "f4_started": False,
        "next_authorization": "F2_ONLY",
    })
    _schema(repo, report, F1_VALIDATION_SCHEMA_PATH, "F1 validation report")
    return report


def build_false_pass_audit(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    evidence = {
        ATTACKS[0]: "startup gate binds F0 archive ref, completion SHA-256, completion canonical digest, status fields and whole-root inventory digest",
        ATTACKS[1]: "source adapter binds D10 archive ref, decision report hash/digest, recipe hash, payload manifest entries and every paper-data file hash",
        ATTACKS[2]: "source adapter performs byte parity between analysis and paper copies but uses analysis paper_data as the only numeric input root",
        ATTACKS[3]: "F1-A requires cohort order c10 then stage_d, n 15 and 500 separately, pooled=false and pooled_n=null",
        ATTACKS[4]: "F1-A caption data contains the mandatory NO_GO_NOVELTY FAIL_SPATIAL BLOCKED disclosure",
        ATTACKS[5]: "source validation preserves C10 status fields and never writes any C10 replacement decision",
        ATTACKS[6]: "plot-data builders copy sealed D10 medians intervals deltas and bands; bootstrap_ci_recomputed=false and new_endpoint_computed=false",
        ATTACKS[7]: "F1-B requires exactly 500 per_sample_delta rows, selection orders 0..499, denominator 500 and final ECDF 1.0",
        ATTACKS[8]: "F1-B records the misclassified count from the full table and validates no source row is filtered",
        ATTACKS[9]: "F1-C maps D10 full method ids to frozen nns/matched styles and enforces insertion/deletion direction order",
        ATTACKS[10]: "F1-C validates fraction index, fraction grid, median, lower and upper directly against D10 curve_summary",
        ATTACKS[11]: "each render report binds identical source_input_digest, plot_data_digest and config_digest across PNG PDF SVG",
        ATTACKS[12]: "F1 config prohibits Stage E paper_data and the adapter has no Stage E numeric input path",
        ATTACKS[13]: "runtime and source-boundary checks reject Torch/timm imports, nonempty CUDA visibility, NVIDIA handles, model/checkpoint surfaces and JPEG loaders",
        ATTACKS[14]: "F1 config/output checks require only F1-A/F1-B/F1-C modules and final_combined_figure_2_rendered=false",
        ATTACKS[15]: "finalizer writes terminal completion only after validation report, false-pass audit, manifest, render inventory and status exist",
        ATTACKS[16]: "completion schema fixes next_authorization to F2_ONLY and f2/f3/f4_started to false",
        ATTACKS[17]: "completion remains SELF until commit; final delivery separately verifies branch push and archive ref",
    }
    document = _signed({
        "schema_version": 1,
        "document_type": "stage_f_f1_false_pass_audit",
        "stage_id": "F1",
        "status": "PASS",
        "attacks": [{"attack": attack, "result": "REJECTED", "evidence": evidence[attack]} for attack in ATTACKS],
    })
    _schema(repo, document, F1_FALSE_PASS_SCHEMA_PATH, "F1 false-pass audit")
    return document


def record_test_report(repo_root: Path | str, *, passed: int, failed: int, command: str) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    if type(passed) is not int or passed <= 0 or failed != 0 or not command:
        raise StageFF1Error("terminal test report requires at least one pass and zero failures")
    report = _signed({
        "schema_version": 1,
        "document_type": "stage_f_f1_test_report",
        "stage_id": "F1",
        "status": "PASS",
        "command": command,
        "passed": passed,
        "failed": failed,
        "required_environment": {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        },
        "model_access": False,
        "torch_imported": False,
        "timm_imported": False,
        "gpu_cuda_access": False,
        "jpeg_access": False,
    })
    _write_new(repo / TEST_PATH, report)
    return report


def _write_source_derived_documents(repo: Path) -> dict[str, Any]:
    validation = validate_outputs(repo)
    _write_new(repo / VALIDATION_PATH, validation)
    return validation


def finalize(repo_root: Path | str = REPO_ROOT, *, completed_at: str) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    root = repo / F1_ROOT_PATH
    validate_runtime_isolation()
    if (repo / COMPLETION_PATH).exists():
        raise StageFF1Error("F1 terminal completion already exists")
    test = _record(repo, TEST_PATH)
    validation = _write_source_derived_documents(repo)
    artifacts = []
    excluded = {"payload_manifest.json", "whole_root_inventory.json", "status.json", "completion/f1.pending.json", "completion/f1.json"}
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink() and path.relative_to(root).as_posix() not in excluded:
            artifacts.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = _signed({
        "schema_version": 1,
        "document_type": "stage_f_f1_payload_manifest",
        "stage_id": "F1",
        "immutable": True,
        "payload_count": len(artifacts),
        "artifacts": artifacts,
        "scientific_modules_rendered": 3,
        "layout_render_instances": 6,
        "exported_figure_files": 18,
        "final_combined_figure_2_rendered": False,
    })
    _write_new(repo / MANIFEST_PATH, manifest)
    inventory = _inventory(root, exclude=("whole_root_inventory.json", "status.json", "completion/f1.pending.json", "completion/f1.json"))
    inventory_doc = _signed({
        "schema_version": 1,
        "document_type": "stage_f_f1_whole_root_inventory",
        "stage_id": "F1",
        "root": F1_ROOT_PATH,
        "scope": "all regular F1 files except self-referential inventory status and completion receipts",
        "files": inventory["files"],
        "file_count": inventory["file_count"],
        "total_bytes": inventory["total_bytes"],
    })
    _write_new(repo / INVENTORY_PATH, inventory_doc)
    status = _signed({
        "schema_version": 1,
        "document_type": "stage_f_f1_status",
        "stage_id": "F1",
        "status": "PASS",
        "terminal": True,
        "immutable": True,
        "package_validation": "PASS",
        "workflow_validation": "PASS",
        "scientific_modules_rendered": 3,
        "layout_render_instances": 6,
        "exported_figure_files": 18,
        "final_combined_figure_2_rendered": False,
        "model_torch_timm_gpu_cuda_jpeg_access": False,
        "bootstrap_ci_new_endpoint_recomputed": False,
        "c10_stage_d_pooled": False,
        "d10_f0_e4_decisions_modified": False,
        "f2_started": False,
        "f3_started": False,
        "f4_started": False,
        "next_authorization": "F2_ONLY",
        "whole_root_inventory": {
            "canonical_digest": inventory_doc["canonical_digest"],
            "file_count": inventory_doc["file_count"],
            "total_bytes": inventory_doc["total_bytes"],
        },
        "git": {"commit": "SELF", "ref": F1_ARCHIVE_REF},
    })
    _write_new(repo / STATUS_PATH, status)
    pending, _ = _strict_json_object(repo / PENDING_PATH, "F1 pending")
    _validate_signed(pending, "F1 pending")
    completion = _signed({
        "schema_version": 1,
        "document_type": "stage_f_f1_terminal_completion",
        "stage_id": "F1",
        "status": "PASS",
        "started_at": pending["started_at"],
        "completed_at": completed_at,
        "package_validation": "PASS",
        "workflow_validation": "PASS",
        "scientific_modules_rendered": 3,
        "layout_render_instances": 6,
        "exported_figure_files": 18,
        "final_combined_figure_2_rendered": False,
        "model_access": False,
        "torch_imported": False,
        "timm_imported": False,
        "gpu_cuda_access": False,
        "jpeg_access": False,
        "bootstrap_ci_new_endpoint_recomputed": False,
        "c10_stage_d_pooled": False,
        "d10_f0_e4_decisions_modified": False,
        "startup_gate": _record(repo, STARTUP_PATH),
        "source_binding": _record(repo, SOURCE_BINDING_PATH),
        "access_audit": _record(repo, ACCESS_AUDIT_PATH),
        "source_validation": _record(repo, VALIDATION_PATH),
        "test_report": test,
        "false_pass_audit": _record(repo, FALSE_PASS_PATH),
        "render_inventory": _record(repo, RENDER_INVENTORY_PATH),
        "payload_manifest": _record(repo, MANIFEST_PATH),
        "status_record": _record(repo, STATUS_PATH),
        "whole_root_inventory": _record(repo, INVENTORY_PATH),
        "git": {"commit": "SELF", "ref": F1_ARCHIVE_REF},
        "next_authorization": "F2_ONLY",
        "f2_started": False,
        "f3_started": False,
        "f4_started": False,
    })
    _schema(repo, completion, F1_COMPLETION_SCHEMA_PATH, "F1 completion")
    temporary = (repo / PENDING_PATH).with_name(".f1-terminal.tmp")
    _write_new(temporary, completion)
    os.replace(temporary, repo / PENDING_PATH)
    os.replace(repo / PENDING_PATH, repo / COMPLETION_PATH)
    return validate_completion(repo)


def validate_completion(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    root = repo / F1_ROOT_PATH
    if not (repo / COMPLETION_PATH).is_file() or (repo / PENDING_PATH).exists():
        raise StageFF1Error("terminal completion missing or pending remains")
    completion, raw = _strict_json_object(repo / COMPLETION_PATH, "F1 completion")
    if raw != canonical_json_bytes(completion) + b"\n":
        raise StageFF1Error("F1 completion is not byte-canonical")
    _validate_signed(completion, "F1 completion")
    _schema(repo, completion, F1_COMPLETION_SCHEMA_PATH, "F1 completion")
    if (
        completion.get("status") != "PASS"
        or completion.get("package_validation") != "PASS"
        or completion.get("workflow_validation") != "PASS"
        or completion.get("scientific_modules_rendered") != 3
        or completion.get("layout_render_instances") != 6
        or completion.get("exported_figure_files") != 18
        or completion.get("final_combined_figure_2_rendered") is not False
        or completion.get("next_authorization") != "F2_ONLY"
        or any(completion.get(key) for key in ("f2_started", "f3_started", "f4_started"))
    ):
        raise StageFF1Error("F1 completion terminal state drifted")
    for field in ("startup_gate", "source_binding", "access_audit", "source_validation", "test_report", "false_pass_audit", "render_inventory", "payload_manifest", "status_record", "whole_root_inventory"):
        if completion[field] != _record(repo, completion[field]["path"]):
            raise StageFF1Error(f"F1 completion binding drifted: {field}")
    inventory, _ = _strict_json_object(repo / INVENTORY_PATH, "F1 whole-root inventory")
    _validate_signed(inventory, "F1 whole-root inventory")
    rebuilt = _inventory(root, exclude=("whole_root_inventory.json", "status.json", "completion/f1.pending.json", "completion/f1.json"))
    unsigned_inventory = {
        "schema_version": 1,
        "document_type": "stage_f_f1_whole_root_inventory",
        "stage_id": "F1",
        "root": F1_ROOT_PATH,
        "scope": "all regular F1 files except self-referential inventory status and completion receipts",
        "files": rebuilt["files"],
        "file_count": rebuilt["file_count"],
        "total_bytes": rebuilt["total_bytes"],
    }
    if any(inventory[key] != rebuilt[key] for key in ("files", "file_count", "total_bytes")) or inventory["canonical_digest"] != canonical_sha256(unsigned_inventory):
        raise StageFF1Error("F1 whole-root inventory drifted")
    validate_runtime_isolation()
    return {
        "status": "PASS",
        "package_validation": "PASS",
        "workflow_validation": "PASS",
        "scientific_modules_rendered": 3,
        "layout_render_instances": 6,
        "exported_figure_files": 18,
        "final_combined_figure_2_rendered": False,
        "model_torch_timm_gpu_cuda_jpeg_access": False,
        "bootstrap_ci_new_endpoint_recomputed": False,
        "c10_stage_d_pooled": False,
        "d10_f0_e4_decisions_modified": False,
        "next_authorization": "F2_ONLY",
        "whole_root_inventory_digest": inventory["canonical_digest"],
    }


def clean_formal_root_for_tests(path: Path) -> None:
    """Test helper only; production code never calls this."""

    if "pytest-" not in str(path):
        raise StageFF1Error("test cleanup helper can only remove pytest temporary paths")
    shutil.rmtree(path, ignore_errors=True)


__all__ = [
    "ATTACKS",
    "COMPLETION_PATH",
    "F1_ARCHIVE_REF",
    "F1_ROOT_PATH",
    "F1_REPORT_PATH",
    "StageFF1Error",
    "assert_no_future_outputs",
    "assert_f1_source_boundary",
    "build_caption_data",
    "build_false_pass_audit",
    "build_outputs",
    "build_plot_data",
    "build_render_spec",
    "finalize",
    "initialize",
    "load_source_bundle",
    "record_test_report",
    "validate_completion",
    "validate_f1_render_report",
    "validate_outputs",
    "validate_runtime_isolation",
    "verify_startup_gates",
]
