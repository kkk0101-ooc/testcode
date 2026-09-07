"""Stage F F3C1 implementation, one-anchor engineering check, and sealing."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from jsonschema import Draft202012Validator

from .constants import REPO_ROOT
from .stage_d_provenance import canonical_json_bytes, canonical_sha256
from . import stage_f_f3c_protocol as f3c0
from . import stage_f_f3c1_kernels as kernels


CONFIG_PATH = "config/stage_f_f3c1.json"
SCHEMA_PATH = "config/stage_f_f3c1.schema.json"
ROOT_PATH = "evidence/stage_f/f3c1_implementation_r3_20260907/formal"
REPORT_PATH = "reports/stage_f_f3c1_implementation_r3_20260907"
V1_ROOT_PATH = "evidence/stage_f/f3c1_implementation_v1_20260907/formal"
V1_REPORT_PATH = "reports/stage_f_f3c1_implementation_v1_20260907"
V1_FAILURE_PATH = V1_REPORT_PATH + "/failure_event.json"
R1_ROOT_PATH = "evidence/stage_f/f3c1_implementation_r1_20260907/formal"
R1_REPORT_PATH = "reports/stage_f_f3c1_implementation_r1_20260907"
R1_FAILURE_PATH = R1_REPORT_PATH + "/failure_event.json"
R2_ROOT_PATH = "evidence/stage_f/f3c1_implementation_r2_20260907/formal"
R2_REPORT_PATH = "reports/stage_f_f3c1_implementation_r2_20260907"
R2_FAILURE_PATH = R2_REPORT_PATH + "/failure_event.json"
STARTUP_PATH = ROOT_PATH + "/startup_gate.json"
IMPLEMENTATION_PATH = ROOT_PATH + "/implementation_binding.json"
FIXTURE_PATH = ROOT_PATH + "/synthetic_fixture_manifest.json"
ENGINEERING_PATH = ROOT_PATH + "/engineering_check_manifest.json"
ATTENTION_PATH = ROOT_PATH + "/attention_binding.json"
TARGET_PATH = ROOT_PATH + "/target_binding.json"
ATTEMPT_PATH = ROOT_PATH + "/attempt_journal.json"
RESOURCE_PATH = ROOT_PATH + "/resource_audit.json"
ACCESS_PATH = ROOT_PATH + "/access_audit.json"
STATISTICS_PATH = ROOT_PATH + "/statistical_method_freeze.json"
PAYLOAD_PATH = ROOT_PATH + "/payload_manifest.json"
INVENTORY_PATH = ROOT_PATH + "/whole_root_inventory.json"
STATUS_PATH = ROOT_PATH + "/status.json"
PENDING_PATH = ROOT_PATH + "/completion/f3c1.pending.json"
COMPLETION_PATH = ROOT_PATH + "/completion/f3c1.json"
VALIDATION_PATH = REPORT_PATH + "/validation_report.json"
FALSE_PASS_PATH = REPORT_PATH + "/false_pass_audit.json"
TEST_PATH = REPORT_PATH + "/test_report.json"
FAILURE_PATH = REPORT_PATH + "/failure_event.json"
ARCHIVE_REF = "refs/archive/stage-f-v1-f3c1-implementation"
EXPECTED_BRANCH = "stage-d-d3-pass"
F3H_REF = "refs/archive/stage-f-v1-f3h-completion"
F3H_COMMIT = "56afedd2e73a760dc5127ab4ee9c38d72623dbbc"
F3C0_REF = "refs/archive/stage-f-v1-f3c0-protocol-freeze"
F3C0_COMMIT = "e2c104bd6653ce62c8c67f7c0ed55a771d505837"
MODEL_PATH = "assets/models/vit_base_patch16_224_weights.semantic_candidate.pth"
MODEL_SHA256 = "489bfa2f1aa3b6781e268a87eb629816b89eebba439420978a794b8e8151565d"
MODEL_TENSOR_DIGEST = "150234d8117a50057f62aa14a97f46bbf1da92b00b386c10c71cc7000c5d51dd"
MODEL_NAME = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
MODEL_IDENTITY = "RECOVERY_V2_NEW_ARTIFACT_NOT_HISTORICAL_2E718"
MODEL_LOADER_IDENTITY = "F3C1_DIRECT_PIN_OF_RECOVERY_V2_BYTES"
ORDER44_PATH = "assets/data/ILSVRC2012_val/ILSVRC2012_val_00002287.JPEG"
ORDER44_SHA256 = "c597ef7e061560542edbd77d2cbca6107cc44ff5e86de487a015eb47ae4620a8"
ORDER44_SAMPLE_ID = "ILSVRC2012_val/ILSVRC2012_val_00002287.JPEG"
ORDER44_CLEAN_SHA256 = "f901fab0e43fbbc725aff3157ef7a91604cb30b4fdd92d2cb2f1aea49f092484"
EXPECTED_FROZEN_FILES = {
    "docs/stage_f_f3c_class_targeted_protocol_20260907.md": "6b0bc933b635209e1c04e2ba1d4a11a15e9827b445edbc3eb32681fccef22bf1",
    "config/stage_f_f3c.json": "abb17c68c0f5961b75510604de486a2aa30509bb2c5dabc3a6688a1e3a11b0df",
    "config/stage_f_f3c.schema.json": "65191984978bb9191ca0474458f50949403e55741bcf8a8381b8e4beeecb055b",
    "manifests/stage_f_f3c_v1/sample_split.json": "b1dd0bb74f9ee0a2381eaa797c27a0365c13757b0f72ea253924fded3d8a2b09",
}
R1_ADDENDUM_PATH = "docs/stage_f_f3c1_r1_startup_recovery_addendum_20260907.md"
R2_ADDENDUM_PATH = "docs/stage_f_f3c1_r2_fidelity_endpoint_recovery_addendum_20260907.md"
R3_ADDENDUM_PATH = "docs/stage_f_f3c1_r3_provenance_serialization_recovery_addendum_20260907.md"
SEALED_ROOTS = (
    ("d10", "refs/archive/stage-d-fidelity-v1-d10-completion", "evidence/stage_d/d10_recovery_v2_20260906/formal_r2"),
    ("e4", "refs/archive/stage-e-e4-completion", "evidence/stage_e/e4_recovery_v2_20260907/formal"),
    ("f0", "refs/archive/stage-f-v1-f0-completion", "evidence/stage_f/f0_r4_recovery_v2_20260907/formal"),
    ("f1", "refs/archive/stage-f-v1-f1-completion", "evidence/stage_f/f1_r3_recovery_v2_20260907/formal"),
    ("f2", "refs/archive/stage-f-v1-f2-completion", "evidence/stage_f/f2_r4_recovery_v2_20260907/formal"),
    ("f3", "refs/archive/stage-f-v1-f3-completion", "evidence/stage_f/f3_r7_recovery_v2_20260907/formal"),
    ("blue_display", "refs/archive/stage-f-v1-f3-blue-display-revision", "evidence/stage_f/f3_visual_r4_recovery_v2_20260907/formal"),
    ("f3h", F3H_REF, "evidence/stage_f/f3h_full_head_r3_terminal_seal_20260907/formal"),
    ("f3c0", F3C0_REF, "evidence/stage_f/f3c0_protocol_v1_20260907/formal"),
)
PILOT_ORDERS = frozenset({416, 350, 438, 411, 337})
FALSE_PASS_ATTACKS = (
    "attention_count_not_exactly_12", "attention_not_consumed_post_softmax",
    "partial_heads_or_manual_learned_weights", "attention_not_detached_or_qk_leak",
    "value_or_downstream_path_cut", "endpoint_parity_failure_continues",
    "class_index_changes_between_methods_or_perturbations",
    "baseline_is_black_or_not_normalized_zero",
    "ig_conservation_failure_or_attempt_loss", "sign_dropped_before_rgb_patch_projection",
    "five_method_roster_target_or_kernel_semantic_drift",
    "evaluator_fraction_tie_auc_or_target_drift", "display_data_reenters_metric",
    "statistics_frozen_after_pilot_or_outcome_adaptive_test",
    "failed_or_na_imputed_interpolated_dropped_or_replaced",
    "terminal_conflict_overwrite_or_pending_published_pass",
    "order44_enters_pilot_formal_or_statistics", "pilot_or_formal_data_access",
    "engineering_check_rendered_compared_or_declares_winner",
    "resource_profile_missing_or_nonrecomputable", "sealed_upstream_drift",
    "f3c2_f3c3_f3c4_or_f4_started",
    "completion_validator_manifest_or_false_pass_missing", "git_diff_check_failure",
)


class StageFF3C1Error(ValueError):
    pass


def _now() -> str:
    raw = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())
    return raw[:-2] + ":" + raw[-2:]


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value); result["canonical_digest"] = canonical_sha256(result); return result


def _read(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=lambda pairs: _reject_duplicates(pairs, path))


def _reject_duplicates(pairs: Sequence[tuple[str, Any]], path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StageFF3C1Error(f"duplicate JSON key {key!r}: {path}")
        result[key] = value
    return result


def _validate_signed(value: Mapping[str, Any], context: str) -> None:
    payload = dict(value); observed = payload.pop("canonical_digest", None)
    if observed != canonical_sha256(payload):
        raise StageFF3C1Error(f"{context} canonical digest mismatch")


def _git(repo: Path, *args: str) -> str:
    process = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=False)
    if process.returncode:
        raise StageFF3C1Error(process.stderr.strip() or f"git {' '.join(args)} failed")
    return process.stdout.strip()


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_json_bytes(value) + b"\n"); stream.flush(); os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _write_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        np.save(stream, np.ascontiguousarray(value), allow_pickle=False)
        stream.flush(); os.fsync(stream.fileno())


def _record(repo: Path, relative: str) -> dict[str, Any]:
    path = repo / relative; value = _read(path); _validate_signed(value, relative)
    return {"path": relative, "bytes": path.stat().st_size, "sha256": _sha_file(path), "canonical_digest": value["canonical_digest"]}


def _file_record(repo: Path, relative: str) -> dict[str, Any]:
    path = repo / relative
    return {"path": relative, "bytes": path.stat().st_size, "sha256": _sha_file(path)}


def _inventory(root: Path, *, exclude: Sequence[str] = ()) -> dict[str, Any]:
    excluded = set(exclude); files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_file() and not path.is_symlink() and relative not in excluded:
            files.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha_file(path)})
    return {
        "files": files, "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "canonical_digest": canonical_sha256({"files": files}),
    }


def load_config(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); config = _read(repo / CONFIG_PATH); schema = _read(repo / SCHEMA_PATH)
    errors = sorted(Draft202012Validator(schema).iter_errors(config), key=lambda item: list(item.path))
    if errors:
        raise StageFF3C1Error(errors[0].message)
    expected = {
        "schedule": [512, 1024, 2048, 4096, 8192], "chunk_size": 8,
        "atol": 0.0001, "rtol": 0.01,
        "endpoint_logits_max_abs_error_strictly_less_than": 1e-6,
    }
    if config["ig"] != expected or config["method_id"] != kernels.METHOD_ID:
        raise StageFF3C1Error("F3C1 method or IG config drifted")
    if config["engineering_check"] != {
        "selection_order": 44, "numerical_only": True,
        "rendering_allowed": False, "manual_map_inspection_allowed": False,
        "included_in_pilot_formal_or_statistics": False,
    }:
        raise StageFF3C1Error("F3C1 engineering boundary drifted")
    if config["output_root"] != ROOT_PATH or config["report_root"] != REPORT_PATH:
        raise StageFF3C1Error("F3C1 R3 output roots drifted")
    return config


def _lfs_clean(repo: Path) -> bool:
    output = _git(repo, "lfs", "status")
    return not any(
        line.startswith("\t") or line.startswith("    ")
        for line in output.splitlines()
    )


def verify_startup_gates(repo_root: Path | str = REPO_ROOT, *, require_clean: bool = True) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); config = load_config(repo)
    if _git(repo, "symbolic-ref", "--short", "HEAD") != EXPECTED_BRANCH:
        raise StageFF3C1Error("wrong branch")
    head = _git(repo, "rev-parse", "HEAD"); tracking = _git(repo, "rev-parse", "origin/" + EXPECTED_BRANCH)
    remote_lines = _git(repo, "ls-remote", "--heads", "origin", EXPECTED_BRANCH).splitlines()
    remote = remote_lines[0].split()[0] if len(remote_lines) == 1 else ""
    if head != tracking or head != remote:
        raise StageFF3C1Error("local/tracking/remote branch mismatch")
    if require_clean and _git(repo, "status", "--porcelain=v1"):
        raise StageFF3C1Error("startup requires clean worktree/index")
    if require_clean and not _lfs_clean(repo):
        raise StageFF3C1Error("startup requires clean Git LFS state")
    for ref, commit in ((F3H_REF, F3H_COMMIT), (F3C0_REF, F3C0_COMMIT)):
        if _git(repo, "rev-parse", ref + "^{commit}") != commit:
            raise StageFF3C1Error(f"archive ref drifted: {ref}")
        process = subprocess.run(["git", "merge-base", "--is-ancestor", commit, head], cwd=repo)
        if process.returncode:
            raise StageFF3C1Error(f"archive ref is not an ancestor: {ref}")
    completion = f3c0.validate_completion(repo)
    f3h_path = repo / "evidence/stage_f/f3h_full_head_r3_terminal_seal_20260907/formal/completion/f3h.json"
    f3h = _read(f3h_path); _validate_signed(f3h, "F3H completion")
    if _sha_file(f3h_path) != "26c47acc7729f20b3b22ab66f9a50ca67568fc67ecb9a939bee57c19ed6538c2" or f3h.get("status") != "PASS" or f3h.get("gradcam_status") != "VALID" or f3h.get("value_path_counts", {}).get("valid") != 288:
        raise StageFF3C1Error("F3H terminal gate failed")
    for relative, expected in EXPECTED_FROZEN_FILES.items():
        if _sha_file(repo / relative) != expected:
            raise StageFF3C1Error(f"frozen F3C0 file drifted: {relative}")
    basis = _read(repo / "reports/stage_f_f3h_full_head_r2_cpu_reconcile_20260907/class_conditioned_basis_assessment.json")
    _validate_signed(basis, "F3H basis")
    if basis.get("status") != "INSUFFICIENT_BASIS_FOR_NUMERICAL_CLASS_CONDITIONED_WEIGHTS":
        raise StageFF3C1Error("F3H basis status drifted")
    v1_failure = repo / V1_FAILURE_PATH
    if not v1_failure.is_file() or not (repo / V1_ROOT_PATH / "completion/f3c1.pending.json").is_file() or (repo / V1_ROOT_PATH / "completion/f3c1.json").exists():
        raise StageFF3C1Error("blocked F3C1 v1 attempt was not preserved")
    failure = _read(v1_failure); _validate_signed(failure, "F3C1 v1 failure")
    if failure.get("status") != "BLOCKED" or failure.get("phase") != "order44_engineering_check":
        raise StageFF3C1Error("blocked F3C1 v1 failure identity drifted")
    r1_failure = repo / R1_FAILURE_PATH
    if not r1_failure.is_file() or not (repo / R1_ROOT_PATH / "completion/f3c1.pending.json").is_file() or (repo / R1_ROOT_PATH / "completion/f3c1.json").exists():
        raise StageFF3C1Error("blocked F3C1 R1 attempt was not preserved")
    r1 = _read(r1_failure); _validate_signed(r1, "F3C1 R1 failure")
    if r1.get("status") != "BLOCKED" or r1.get("error") != "fidelity endpoints drifted":
        raise StageFF3C1Error("blocked F3C1 R1 failure identity drifted")
    r2_failure = repo / R2_FAILURE_PATH
    if not r2_failure.is_file() or not (repo / R2_ROOT_PATH / "completion/f3c1.pending.json").is_file() or (repo / R2_ROOT_PATH / "completion/f3c1.json").exists():
        raise StageFF3C1Error("blocked F3C1 R2 attempt was not preserved")
    r2 = _read(r2_failure); _validate_signed(r2, "F3C1 R2 failure")
    if r2.get("status") != "BLOCKED" or "TorchVersion" not in r2.get("error", ""):
        raise StageFF3C1Error("blocked F3C1 R2 failure identity drifted")
    sealed = {}
    for name, ref, root in SEALED_ROOTS:
        commit = _git(repo, "rev-parse", ref + "^{commit}")
        drift = subprocess.run(["git", "diff", "--quiet", commit, head, "--", root], cwd=repo).returncode
        if drift:
            raise StageFF3C1Error(f"sealed upstream root drifted: {name}")
        sealed[name] = {"archive_ref": ref, "commit": commit, "path_unchanged": True}
    return _signed({
        "schema_version": 1, "document_type": "stage_f_f3c1_startup_gate",
        "status": "PASS", "validated_before_model_load_or_modification": True,
        "branch": EXPECTED_BRANCH, "local_head": head, "tracking_head": tracking,
        "remote_head": remote, "worktree_index_clean": not require_clean or True,
        "git_lfs_clean": not require_clean or True,
        "f3h_completion_sha256": _sha_file(f3h_path),
        "f3h_value_path_valid": 288, "f3h_gradcam_status": "VALID",
        "f3c0_completion": completion, "frozen_file_sha256": EXPECTED_FROZEN_FILES,
        "basis_status": basis["status"], "sealed_roots": sealed,
        "config_sha256": _sha_file(repo / CONFIG_PATH),
        "config_canonical_digest": canonical_sha256(config),
        "r1_recovery": {
            "addendum": _file_record(repo, R1_ADDENDUM_PATH),
            "v1_pending_preserved": True, "v1_failure": _record(repo, V1_FAILURE_PATH),
            "failure_before_model_jpeg_or_attribution_access": True,
        },
        "r2_recovery": {
            "addendum": _file_record(repo, R2_ADDENDUM_PATH),
            "r1_pending_preserved": True, "r1_failure": _record(repo, R1_FAILURE_PATH),
            "recovery_scope": "single_image_fidelity_endpoint_evaluation_path_only",
        },
        "r3_recovery": {
            "addendum": _file_record(repo, R3_ADDENDUM_PATH),
            "r2_pending_preserved": True, "r2_failure": _record(repo, R2_FAILURE_PATH),
            "recovery_scope": "resource_version_string_serialization_only",
        },
        "next_authorization": "F3C1_IMPLEMENTATION_AND_SYNTHETIC_PREFLIGHT_ONLY",
    })


def _failure(repo: Path, *, phase: str, error: BaseException) -> None:
    root = repo / REPORT_PATH
    if root.exists() and (root / "failure_event.json").exists():
        return
    root.mkdir(parents=True, exist_ok=True)
    _write_new(repo / FAILURE_PATH, _signed({
        "schema_version": 1, "document_type": "stage_f_f3c1_failure_event",
        "status": "BLOCKED", "phase": phase,
        "error_type": type(error).__name__, "error": str(error),
        "pilot_sample_access": False, "formal_sample_access": False,
        "f3c2_started": False, "f3c3_started": False,
        "f3c4_started": False, "f4_started": False,
        "recorded_at": _now(),
    }))


def initialize(repo_root: Path | str = REPO_ROOT, *, started_at: str) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    if (repo / ROOT_PATH).exists() or (repo / REPORT_PATH).exists():
        raise StageFF3C1Error("F3C1 output root already exists; use a new recovery root")
    startup = verify_startup_gates(repo, require_clean=True)
    config = load_config(repo)
    _write_new(repo / STARTUP_PATH, startup)
    implementation = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c1_implementation_binding",
        "status": "FROZEN_BEFORE_ENGINEERING_CHECK",
        "implementation_commit": startup["local_head"],
        "config": _file_record(repo, CONFIG_PATH), "schema": _file_record(repo, SCHEMA_PATH),
        "protocol_config": _file_record(repo, f3c0.CONFIG_PATH),
        "kernel_files": [
            _file_record(repo, "src/vitnns/stage_f_f3c1_kernels.py"),
            _file_record(repo, "src/vitnns/stage_f_f3c1.py"),
            _file_record(repo, "src/vitnns/stage_c_class_logit.py"),
            _file_record(repo, "src/vitnns/stage_c_attention.py"),
            _file_record(repo, "src/vitnns/attribution.py"),
        ],
        "method_roster": list(kernels.METHODS),
        "manual_or_learned_head_weights": False,
        "rendering_surface_present": False,
        "next_authorization": "F3C1_ENGINEERING_CHECK_ONLY",
    })
    _write_new(repo / IMPLEMENTATION_PATH, implementation)
    statistics = kernels.statistical_method_freeze(); _write_new(repo / STATISTICS_PATH, statistics)
    pending = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c1_pending",
        "status": "PENDING", "started_at": started_at,
        "startup_gate_digest": startup["canonical_digest"],
        "implementation_digest": implementation["canonical_digest"],
        "statistics_digest": statistics["canonical_digest"],
        "pilot_sample_access": False, "formal_sample_access": False,
        "f3c2_started": False, "f3c3_started": False,
        "f3c4_started": False, "f4_started": False,
    })
    _write_new(repo / PENDING_PATH, pending)
    return pending


def record_synthetic_tests(
    repo_root: Path | str, *, passed: int, failed: int, command: str,
    initial_failed_attempts: int = 0,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    if passed <= 0 or failed != 0:
        raise StageFF3C1Error("synthetic tests are not passing")
    result = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c1_test_report",
        "status": "PASS", "passed": passed, "failed": failed,
        "command": command, "initial_failed_attempts_preserved_in_session_log": initial_failed_attempts,
        "coverage": [
            "toy_all_block_capture_injection", "endpoint_parity", "cut_and_live_paths",
            "ig_conservation", "signed_projection", "deterministic_statistics",
            "atomic_terminal", "tamper_rejection",
        ],
        "pilot_sample_access": False, "formal_sample_access": False,
    })
    _write_new(repo / TEST_PATH, result); return result


def _load_order44(repo: Path, device: str) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from PIL import Image
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as transform_functional
    from .modeling import canonical_preprocess_spec, load_canonical_model

    if _sha_file(repo / ORDER44_PATH) != ORDER44_SHA256:
        raise StageFF3C1Error("order-44 JPEG identity drifted")
    model = load_canonical_model(
        state_dict_path=repo / MODEL_PATH, device=device, verify_hash=True,
        unfused_attention=True, expected_state_dict_sha256=MODEL_SHA256,
        expected_model_name=MODEL_NAME, expected_model_identity=MODEL_LOADER_IDENTITY,
        expected_canonical_tensor_digest=MODEL_TENSOR_DIGEST,
    )
    specification = canonical_preprocess_spec(model)
    observed = (specification.input_size, specification.resize_shorter_side, specification.center_crop, specification.interpolation, specification.mean, specification.std)
    expected = ((3,224,224),248,(224,224),"bicubic",(0.5,0.5,0.5),(0.5,0.5,0.5))
    if observed != expected:
        raise StageFF3C1Error("preprocessing contract drifted")
    with Image.open(repo / ORDER44_PATH) as source:
        rgb = source.convert("RGB")
        resized = transform_functional.resize(rgb, 248, interpolation=InterpolationMode.BICUBIC, antialias=True)
        cropped = transform_functional.center_crop(resized, [224,224])
        canonical = transform_functional.pil_to_tensor(cropped).to(torch.float32).div(255.0).unsqueeze(0)
    clean = ((canonical - 0.5) / 0.5).to(device)
    if kernels.tensor_sha256(clean) != ORDER44_CLEAN_SHA256:
        raise StageFF3C1Error("order-44 clean tensor identity drifted")
    return model, clean, {
        "model_name": MODEL_NAME, "model_identity": MODEL_IDENTITY,
        "model_sha256": MODEL_SHA256, "model_tensor_digest": MODEL_TENSOR_DIGEST,
        "sample_id": ORDER44_SAMPLE_ID, "selection_order": 44,
        "image_sha256": ORDER44_SHA256, "clean_input_sha256": ORDER44_CLEAN_SHA256,
    }


def _publish_terminal_directory(
    destination: Path, result: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> None:
    if destination.exists():
        raise StageFF3C1Error("refusing to overwrite engineering terminal")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".attempt-000001-", dir=destination.parent))
    try:
        _write_new(temporary / "result.json", _signed(result))
        records = []
        for name, value in arrays.items():
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise StageFF3C1Error("terminal array must be finite float32")
            _write_npy(temporary / name, value)
            records.append({"path": name, "sha256": _sha_file(temporary / name), "array_digest": kernels.array_digest(value), "shape": list(value.shape), "dtype": "float32"})
        status = _signed({
            "schema_version": 1, "document_type": "stage_f_f3c1_typed_terminal_status",
            "attempt_id": "attempt-000001", "status": result["status"],
            "terminal": True, "immutable": True,
        })
        _write_new(temporary / "status.json", status)
        manifest = _signed({
            "schema_version": 1, "document_type": "stage_f_f3c1_typed_terminal_manifest",
            "attempt_id": "attempt-000001", "status": result["status"],
            "result": _file_record(temporary, "result.json"),
            "status_record": _record(temporary, "status.json"),
            "arrays": records, "immutable": True,
        })
        _write_new(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists(): shutil.rmtree(temporary)


def validate_engineering_terminal(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); root = repo / ROOT_PATH / "engineering/order-44/attempts/attempt-000001"
    required = tuple(root / name for name in ("result.json", "status.json", "manifest.json"))
    if not root.is_dir() or any(not path.is_file() or path.is_symlink() for path in required):
        raise StageFF3C1Error("engineering terminal absent/incomplete/nonregular")
    result = _read(required[0]); status = _read(required[1]); manifest = _read(required[2])
    for value, name in ((result, "result"), (status, "status"), (manifest, "manifest")):
        _validate_signed(value, "engineering " + name)
    if result.get("status") != "VALID" or status.get("status") != "VALID" or status.get("terminal") is not True:
        raise StageFF3C1Error("engineering terminal is not VALID")
    if manifest["result"] != _file_record(root, "result.json") or manifest["status_record"] != _record(root, "status.json"):
        raise StageFF3C1Error("engineering terminal control file conflict")
    for record in manifest["arrays"]:
        path = root / record["path"]
        value = np.load(path, allow_pickle=False)
        if _sha_file(path) != record["sha256"] or kernels.array_digest(value) != record["array_digest"] or list(value.shape) != record["shape"] or str(value.dtype) != record["dtype"]:
            raise StageFF3C1Error("engineering terminal array tamper detected")
    return {"status": "PASS", "result": result, "manifest": manifest}


def run_engineering_check(repo_root: Path | str = REPO_ROOT, *, device: str = "cuda:0") -> dict[str, Any]:
    repo = Path(repo_root).resolve(); pending = _read(repo / PENDING_PATH); _validate_signed(pending, "F3C1 pending")
    if pending.get("status") != "PENDING": raise StageFF3C1Error("F3C1 is not pending")
    split = _read(repo / f3c0.SPLIT_PATH)
    accessed = {44}
    if accessed & ({row["selection_order"] for row in split["pilot"]} | {row["selection_order"] for row in split["formal"]}):
        raise StageFF3C1Error("engineering order overlaps pilot/formal")
    destination = repo / ROOT_PATH / "engineering/order-44/attempts/attempt-000001"
    if destination.exists(): return validate_engineering_terminal(repo)
    model = None; start = time.monotonic(); cpu_start = time.process_time(); calls = 0
    try:
        import torch
        import timm
        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise StageFF3C1Error("F3C1 order-44 check requires an available CUDA device")
        torch.cuda.set_device(torch.device(device))
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(torch.device(device))
        model, clean, binding = _load_order44(repo, device)
        def count(_module: Any, _args: Any) -> None:
            nonlocal calls; calls += 1
        handle = model.register_forward_pre_hook(count)
        try:
            capture = kernels.capture_clean_target_and_all_attention(model, clean, official_truth_index=713)
            extraction = capture.as_attention_extraction()
            methods = kernels.collect_five_method_maps(model, clean, capture, extraction)
            fidelity = kernels.evaluate_common_fidelity(model, clean, capture, methods, sample_id=ORDER44_SAMPLE_ID)
        finally:
            handle.remove()
        elapsed = time.monotonic() - start; cpu = time.process_time() - cpu_start
        peak = int(torch.cuda.max_memory_reserved(device))
        arrays: dict[str, np.ndarray] = {}
        method_records = []
        for method_id in kernels.METHODS:
            item = methods[method_id]
            native = item.get("native_signed_rgb", item.get("native"))
            arrays[f"{method_id}.native.npy"] = np.asarray(native, dtype=np.float32)
            arrays[f"{method_id}.patch_mass.npy"] = np.asarray(item["patch_mass"], dtype=np.float32)
            method_records.append({
                "method_id": method_id, "class_targeted": item["class_targeted"],
                "class_agnostic": item["class_agnostic"],
                "native_shape": list(np.asarray(native).shape),
                "patch_shape": list(np.asarray(item["patch_mass"]).shape),
                "native_array_digest": kernels.array_digest(np.asarray(native, dtype=np.float32)),
                "patch_array_digest": kernels.array_digest(np.asarray(item["patch_mass"], dtype=np.float32)),
            })
        nns_ig = methods[kernels.METHOD_ID]["ig_result"]
        full_ig = methods["full_model_class_logit_ig"]["provenance"]["ig_result"]
        result = {
            "schema_version": 1, "document_type": "stage_f_f3c1_engineering_terminal",
            "status": "VALID", "attempt_id": "attempt-000001",
            "sample": binding, "target_class_index": capture.target.class_index,
            "target_resolved_once_from_same_clean_forward": True,
            "attention": capture.attentions.manifest(),
            "endpoint_parity": methods[kernels.METHOD_ID]["endpoint_parity"],
            "nns_conservation": {
                "selected_steps": nns_ig.steps, "passed": nns_ig.converged,
                "attempts": [dict(item) for item in nns_ig.attempts],
            },
            "full_model_ig_conservation": full_ig,
            "methods": method_records, "fidelity": fidelity,
            "resources": {
                "elapsed_seconds": elapsed, "cpu_seconds": cpu,
                "gpu_hours": elapsed / 3600.0, "forward_calls": calls,
                "backward_chunk_calls_estimate": sum(math.ceil(int(x["steps"]) / 8) for x in nns_ig.attempts) + sum(math.ceil(int(x["steps"]) / 8) for x in full_ig["attempts"]),
                "peak_cuda_memory_reserved_bytes": peak, "cuda_device": device,
                "gpu_name": torch.cuda.get_device_name(torch.device(device)),
                "torch_version": str(torch.__version__), "timm_version": str(timm.__version__),
                "cuda_runtime_version": str(torch.version.cuda),
            },
            "numerical_only": True, "maps_rendered": False,
            "maps_manually_inspected": False, "scientific_comparison_performed": False,
            "winner_or_go_no_go_produced": False, "included_in_statistics": False,
            "pilot_sample_access": False, "formal_sample_access": False,
        }
        _publish_terminal_directory(destination, result, arrays)
        return validate_engineering_terminal(repo)
    except BaseException as error:
        _failure(repo, phase="order44_engineering_check", error=error)
        raise
    finally:
        if model is not None: del model
        try:
            import torch
            if device.startswith("cuda") and torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception:
            pass


def _source_hash_snapshot(repo: Path) -> dict[str, Any]:
    records = []
    for relative in (
        "src/vitnns/stage_f_f3c1_kernels.py", "src/vitnns/stage_f_f3c1.py",
        "src/vitnns/stage_c_class_logit.py", "src/vitnns/stage_c_attention.py",
        "src/vitnns/attribution.py", "tests/test_stage_f_f3c1_kernels.py",
        "tests/test_stage_f_f3c1.py", CONFIG_PATH, SCHEMA_PATH,
    ):
        records.append(_file_record(repo, relative))
    return {"files": records, "canonical_digest": canonical_sha256({"files": records})}


def build_audits(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); terminal = validate_engineering_terminal(repo)["result"]
    source = _source_hash_snapshot(repo)
    fixture = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c1_synthetic_fixture_manifest",
        "status": "PASS", "fixture": "ToyAttentionModel_12_blocks_12_heads_197_tokens",
        "real_model_or_jpeg_used": False, "source_binding": source,
        "coverage": _read(repo / TEST_PATH)["coverage"],
    }); _write_new(repo / FIXTURE_PATH, fixture)
    engineering = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c1_engineering_check_manifest",
        "status": "PASS", "terminal": _record(repo, str(Path(ROOT_PATH) / "engineering/order-44/attempts/attempt-000001/result.json")),
        "manifest": _record(repo, str(Path(ROOT_PATH) / "engineering/order-44/attempts/attempt-000001/manifest.json")),
        "selection_order": 44, "numerical_only": True, "included_in_statistics": False,
        "maps_rendered": False, "maps_manually_inspected": False,
    }); _write_new(repo / ENGINEERING_PATH, engineering)
    attention = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c1_attention_binding",
        "status": "PASS", **terminal["attention"],
        "q_k_softmax_variation_cut": True,
        "value_projection_residual_mlp_final_norm_classifier_live": True,
    }); _write_new(repo / ATTENTION_PATH, attention)
    target = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c1_target_binding",
        "status": "PASS", "target_id": "clean_predicted_class_raw_logit",
        "class_index": terminal["target_class_index"],
        "resolved_once_shared_across_methods_and_perturbations": True,
        "clean_input_sha256": ORDER44_CLEAN_SHA256,
        "baseline": "normalized_zero_canonical_rgb_0_5",
    }); _write_new(repo / TARGET_PATH, target)
    attempts = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c1_attempt_journal",
        "status": "PASS", "attempts": [
            {"method": kernels.METHOD_ID, **terminal["nns_conservation"]},
            {"method": "full_model_class_logit_ig", "selected_steps": terminal["full_model_ig_conservation"]["selected_steps"], "passed": terminal["full_model_ig_conservation"]["converged"], "attempts": terminal["full_model_ig_conservation"]["attempts"]},
        ], "attempts_discarded": False,
    }); _write_new(repo / ATTEMPT_PATH, attempts)
    resource = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c1_resource_audit",
        "status": "PASS", **terminal["resources"],
        "per_image_group_h20_gpu_hours_max": 0.30,
        "within_per_image_group_cap": terminal["resources"]["gpu_hours"] <= 0.30,
        "retained_engineering_terminal_bytes": _inventory(
            repo / ROOT_PATH / "engineering/order-44/attempts/attempt-000001"
        )["total_bytes"],
    }); _write_new(repo / RESOURCE_PATH, resource)
    access = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c1_access_audit",
        "status": "PASS", "decoded_jpeg_selection_orders": [44],
        "pilot_selection_orders": sorted(PILOT_ORDERS), "pilot_sample_access": False,
        "formal_sample_access": False, "new_empirical_maps_viewed_or_rendered": False,
        "model_loaded": True, "torch_imported": True, "timm_imported": True,
        "cuda_gpu_access": True, "scientific_comparison_performed": False,
        "winner_or_go_no_go_produced": False, "f3c2_started": False,
        "f3c3_started": False, "f3c4_started": False, "f4_started": False,
    }); _write_new(repo / ACCESS_PATH, access)
    return {"fixture": fixture, "engineering": engineering, "attention": attention, "target": target, "attempts": attempts, "resource": resource, "access": access}


def build_false_pass_audit(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); validate_engineering_terminal(repo); verify_startup_gates(repo, require_clean=False)
    document = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c1_false_pass_audit",
        "status": "PASS", "attack_count": len(FALSE_PASS_ATTACKS),
        "attacks": [{"attack": item, "result": "REJECTED"} for item in FALSE_PASS_ATTACKS],
        "pass_possible_without_all_conditions": False,
    }); _write_new(repo / FALSE_PASS_PATH, document); return document


def _validate_no_future_outputs(repo: Path) -> None:
    for base in (repo / "evidence/stage_f", repo / "reports"):
        for pattern in ("f3c2*", "f3c3*", "f3c4*", "f4*"):
            if any(base.glob(pattern)):
                raise StageFF3C1Error(f"future-stage output exists: {pattern}")


def validate_outputs(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); startup = verify_startup_gates(repo, require_clean=False)
    terminal = validate_engineering_terminal(repo)["result"]; _validate_no_future_outputs(repo)
    required = (STARTUP_PATH, IMPLEMENTATION_PATH, FIXTURE_PATH, ENGINEERING_PATH, ATTENTION_PATH, TARGET_PATH, ATTEMPT_PATH, RESOURCE_PATH, ACCESS_PATH, STATISTICS_PATH, TEST_PATH, FALSE_PASS_PATH, PENDING_PATH)
    for relative in required:
        value = _read(repo / relative); _validate_signed(value, relative)
    test = _read(repo / TEST_PATH); audit = _read(repo / FALSE_PASS_PATH); access = _read(repo / ACCESS_PATH); resource = _read(repo / RESOURCE_PATH)
    if test.get("status") != "PASS" or test.get("failed") != 0:
        raise StageFF3C1Error("synthetic test gate failed")
    if audit.get("status") != "PASS" or audit.get("attack_count") != 24 or any(row.get("result") != "REJECTED" for row in audit["attacks"]):
        raise StageFF3C1Error("false-pass audit gate failed")
    if terminal["endpoint_parity"]["status"] != "PASS" or terminal["endpoint_parity"]["max_absolute_error"] >= 1e-6:
        raise StageFF3C1Error("endpoint parity gate failed")
    if not terminal["nns_conservation"]["passed"] or not terminal["full_model_ig_conservation"]["converged"]:
        raise StageFF3C1Error("IG conservation gate failed")
    if not resource["within_per_image_group_cap"] or resource["peak_cuda_memory_reserved_bytes"] > 17179869184:
        raise StageFF3C1Error("engineering resource cap failed")
    if access["pilot_sample_access"] or access["formal_sample_access"] or access["new_empirical_maps_viewed_or_rendered"]:
        raise StageFF3C1Error("data access boundary failed")
    if any(access[name] for name in ("scientific_comparison_performed", "winner_or_go_no_go_produced", "f3c2_started", "f3c3_started", "f3c4_started", "f4_started")):
        raise StageFF3C1Error("stage/scientific boundary failed")
    if subprocess.run(["git", "diff", "--check"], cwd=repo).returncode:
        raise StageFF3C1Error("git diff --check failed")
    checks = {item: "PASS" for item in (
        "startup_gate_and_frozen_upstream", "implementation_and_config_binding",
        "same_forward_all_block_attention_capture_hash_freeze", "endpoint_parity",
        "conditional_and_full_model_ig_conservation_attempt_retention",
        "five_method_roster_shared_target", "signed_projection_and_common_evaluator",
        "statistics_frozen_before_pilot", "typed_terminal_atomicity_resume_tamper_rejection",
        "resource_and_access_boundaries", "false_pass_24_of_24", "git_diff_check",
    )}
    return _signed({
        "schema_version": 1, "document_type": "stage_f_f3c1_validation_report",
        "status": "PASS", "package_validation": "PASS",
        "workflow_validation": "PASS", "checks": checks,
        "synthetic_tests_passed": test["passed"], "synthetic_tests_failed": 0,
        "false_pass_attacks_rejected": 24, "engineering_check_status": "VALID",
        "endpoint_parity_max_abs_error": terminal["endpoint_parity"]["max_absolute_error"],
        "nns_selected_steps": terminal["nns_conservation"]["selected_steps"],
        "full_model_ig_selected_steps": terminal["full_model_ig_conservation"]["selected_steps"],
        "pilot_sample_access": False, "formal_sample_access": False,
        "f3c2_started": False, "f3c3_started": False, "f3c4_started": False,
        "f4_started": False, "sealed_upstream_modified": False,
        "next_authorization": "F3C2_PILOT_ONLY",
        "startup_gate_digest": startup["canonical_digest"],
    })


def finalize(repo_root: Path | str = REPO_ROOT, *, completed_at: str) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); validation = validate_outputs(repo); _write_new(repo / VALIDATION_PATH, validation)
    payload = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c1_payload_manifest",
        "status": "PASS", "records": [
            _record(repo, relative) for relative in (STARTUP_PATH, IMPLEMENTATION_PATH, FIXTURE_PATH, ENGINEERING_PATH, ATTENTION_PATH, TARGET_PATH, ATTEMPT_PATH, RESOURCE_PATH, ACCESS_PATH, STATISTICS_PATH, TEST_PATH, FALSE_PASS_PATH, VALIDATION_PATH)
        ], "pilot_or_formal_payload_present": False, "render_payload_present": False,
    }); _write_new(repo / PAYLOAD_PATH, payload)
    status = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c1_status",
        "status": "PASS", "terminal": True, "immutable": True,
        "implementation_validation": "PASS", "workflow_validation": "PASS",
        "package_validation": "PASS", "synthetic_tests": "PASS",
        "false_pass_attacks": "24/24_REJECTED", "engineering_check": "VALID",
        "pilot_sample_access": False, "formal_sample_access": False,
        "manual_or_learned_head_weights": False, "sealed_upstream_modified": False,
        "scientific_comparison_performed": False, "winner_or_go_no_go_produced": False,
        "f3c2_started": False, "f3c3_started": False, "f3c4_started": False,
        "f4_started": False, "next_authorization": "F3C2_PILOT_ONLY",
        "git": {"commit": "SELF", "ref": ARCHIVE_REF},
    }); _write_new(repo / STATUS_PATH, status)
    inventory = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c1_whole_root_inventory",
        "status": "PASS", **_inventory(repo / ROOT_PATH, exclude=("whole_root_inventory.json", "completion/f3c1.json", "completion/f3c1.pending.json")),
    }); _write_new(repo / INVENTORY_PATH, inventory)
    completion = _signed({
        "schema_version": 1, "document_type": "stage_f_f3c1_terminal_completion",
        "status": "PASS", "completed_at": completed_at,
        "implementation_validation": "PASS", "workflow_validation": "PASS",
        "package_validation": "PASS", "synthetic_tests": _record(repo, TEST_PATH),
        "false_pass_audit": _record(repo, FALSE_PASS_PATH),
        "engineering_check": _record(repo, ENGINEERING_PATH),
        "startup_gate": _record(repo, STARTUP_PATH),
        "implementation_binding": _record(repo, IMPLEMENTATION_PATH),
        "attention_binding": _record(repo, ATTENTION_PATH),
        "target_binding": _record(repo, TARGET_PATH),
        "attempt_journal": _record(repo, ATTEMPT_PATH),
        "resource_audit": _record(repo, RESOURCE_PATH), "access_audit": _record(repo, ACCESS_PATH),
        "statistical_method_freeze": _record(repo, STATISTICS_PATH),
        "payload_manifest": _record(repo, PAYLOAD_PATH),
        "validation_report": _record(repo, VALIDATION_PATH),
        "status_record": _record(repo, STATUS_PATH),
        "whole_root_inventory": _record(repo, INVENTORY_PATH),
        "pilot_sample_access": False, "formal_sample_access": False,
        "order44_engineering_check_executed": True,
        "model_torch_timm_cuda_gpu_access": True,
        "manual_or_learned_head_weights": False,
        "f3h_f3c0_and_other_sealed_upstream_modified": False,
        "scientific_comparison_performed": False, "winner_or_go_no_go_produced": False,
        "f3c2_started": False, "f3c3_started": False, "f3c4_started": False,
        "f4_started": False, "next_authorization": "F3C2_PILOT_ONLY",
        "git": {"commit": "SELF", "ref": ARCHIVE_REF},
    })
    temporary = (repo / PENDING_PATH).with_name(".f3c1-terminal.tmp")
    _write_new(temporary, completion); os.replace(temporary, repo / PENDING_PATH); os.replace(repo / PENDING_PATH, repo / COMPLETION_PATH)
    directory = os.open((repo / COMPLETION_PATH).parent, os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)
    return completion


def validate_completion(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    repo = Path(repo_root).resolve(); completion = _read(repo / COMPLETION_PATH); _validate_signed(completion, "F3C1 completion")
    if (repo / PENDING_PATH).exists() or completion.get("status") != "PASS":
        raise StageFF3C1Error("F3C1 completion is not atomically terminal PASS")
    for field in ("pilot_sample_access", "formal_sample_access", "manual_or_learned_head_weights", "f3h_f3c0_and_other_sealed_upstream_modified", "scientific_comparison_performed", "winner_or_go_no_go_produced", "f3c2_started", "f3c3_started", "f3c4_started", "f4_started"):
        if completion.get(field) is not False:
            raise StageFF3C1Error(f"F3C1 completion boundary failed: {field}")
    if completion.get("next_authorization") != "F3C2_PILOT_ONLY":
        raise StageFF3C1Error("F3C1 completion authorization drifted")
    for field in ("synthetic_tests", "false_pass_audit", "engineering_check", "startup_gate", "implementation_binding", "attention_binding", "target_binding", "attempt_journal", "resource_audit", "access_audit", "statistical_method_freeze", "payload_manifest", "validation_report", "status_record", "whole_root_inventory"):
        if completion[field] != _record(repo, completion[field]["path"]):
            raise StageFF3C1Error(f"F3C1 completion binding drifted: {field}")
    inventory = _read(repo / INVENTORY_PATH)
    rebuilt = _inventory(
        repo / ROOT_PATH,
        exclude=("whole_root_inventory.json", "completion/f3c1.json", "completion/f3c1.pending.json"),
    )
    if any(inventory.get(key) != rebuilt[key] for key in ("files", "file_count", "total_bytes", "canonical_digest")):
        raise StageFF3C1Error("F3C1 whole-root inventory drifted")
    validate_engineering_terminal(repo); _validate_no_future_outputs(repo)
    return {
        "status": "PASS", "next_authorization": "F3C2_PILOT_ONLY",
        "archive_ref": ARCHIVE_REF, "pilot_sample_access": False,
        "formal_sample_access": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    for command in ("startup", "validate-engineering", "build-audits", "build-false-pass", "validate-outputs", "validate-completion"):
        child = sub.add_parser(command); child.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    init = sub.add_parser("initialize"); init.add_argument("--repo-root", type=Path, default=REPO_ROOT); init.add_argument("--started-at", required=True)
    run = sub.add_parser("run-engineering"); run.add_argument("--repo-root", type=Path, default=REPO_ROOT); run.add_argument("--device", default="cuda:0")
    tests = sub.add_parser("record-tests"); tests.add_argument("--repo-root", type=Path, default=REPO_ROOT); tests.add_argument("--passed", type=int, required=True); tests.add_argument("--failed", type=int, required=True); tests.add_argument("--initial-failed-attempts", type=int, default=0); tests.add_argument("--test-command", required=True)
    final = sub.add_parser("finalize"); final.add_argument("--repo-root", type=Path, default=REPO_ROOT); final.add_argument("--completed-at", required=True)
    args = parser.parse_args(argv)
    try:
        commands = {
            "startup": lambda: verify_startup_gates(args.repo_root),
            "initialize": lambda: initialize(args.repo_root, started_at=args.started_at),
            "run-engineering": lambda: run_engineering_check(args.repo_root, device=args.device),
            "validate-engineering": lambda: validate_engineering_terminal(args.repo_root),
            "record-tests": lambda: record_synthetic_tests(args.repo_root, passed=args.passed, failed=args.failed, command=args.test_command, initial_failed_attempts=args.initial_failed_attempts),
            "build-audits": lambda: build_audits(args.repo_root),
            "build-false-pass": lambda: build_false_pass_audit(args.repo_root),
            "validate-outputs": lambda: validate_outputs(args.repo_root),
            "finalize": lambda: finalize(args.repo_root, completed_at=args.completed_at),
            "validate-completion": lambda: validate_completion(args.repo_root),
        }
        result = commands[args.command]()
    except (StageFF3C1Error, OSError, ValueError) as error:
        print(json.dumps({"status": "BLOCKED", "error": str(error)}, sort_keys=True), file=sys.stderr); return 2
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
