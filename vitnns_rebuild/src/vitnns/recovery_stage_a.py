"""Isolated recovery-v2 Stage A verification.

This module deliberately does not call the historical Stage A runner or
provenance builder.  Those modules are bound to the missing historical model
artifact.  Recovery-v2 instead derives every identity from the closed machine
binding and keeps the historical pickle path unreachable.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
import PIL
import timm
import torch
import torchvision
from PIL import Image
from timm.layers import set_fused_attn

from .extractor import NativeViTExtractor, capture_block_io
from .io_utils import sha256_file
from .modeling import (
    _assert_model_contract,
    canonical_preprocess_spec,
    canonical_transform,
)
from .recovery_v2 import (
    RecoveryV2Binding,
    load_bound_stage_a_manifest_row,
    load_recovery_v2_binding,
    verify_recovery_v2_lfs_identity,
    verify_recovery_v2_state_dict,
)


EXPECTED_BRANCH = "stage-d-d3-pass"
REMOTE_TRACKING_REF = "refs/remotes/origin/stage-d-d3-pass"
LOGITS_THRESHOLD = 1e-5
ATTENTION_THRESHOLD = 1e-6
ARTIFACT_FILENAMES = (
    "extraction.pt",
    "provenance.json",
    "validation.json",
    "stage_a_verification.json",
)
NONTERMINAL_FILENAMES = ARTIFACT_FILENAMES[:-1]
TERMINAL_FILENAME = ARTIFACT_FILENAMES[-1]
PENDING_TERMINAL_FILENAME = ".stage_a_verification.pending"
UNSEALED_TERMINAL_FILENAME = "stage_a_verification.unsealed.json"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_GIT_TIMEOUT_SECONDS = 30


class RecoveryStageAError(RuntimeError):
    """Raised when recovery-v2 Stage A cannot establish a PASS result."""


def _git_process(
    repo_root: Path, arguments: Sequence[str], *, check: bool = True, text: bool = True,
) -> subprocess.CompletedProcess[Any]:
    environment = os.environ.copy()
    environment.update({
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_LFS_SKIP_SMUDGE": "1",
    })
    try:
        return subprocess.run(
            ["git", *arguments], cwd=repo_root, check=check,
            capture_output=True, text=text, timeout=_GIT_TIMEOUT_SECONDS,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RecoveryStageAError(
            f"Cannot inspect local Git state: git {' '.join(arguments)}"
        ) from error


def _run_git(repo_root: Path, *arguments: str) -> str:
    return _git_process(repo_root, arguments).stdout.strip()


def _repository_identity(repo_root: Path) -> tuple[Path, str]:
    try:
        root = Path(repo_root).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise RecoveryStageAError("repo_root must be an existing directory") from error
    if not root.is_dir() or root.is_symlink():
        raise RecoveryStageAError("repo_root must be an ordinary directory")
    top_text = _run_git(root, "rev-parse", "--show-toplevel")
    prefix = _run_git(root, "rev-parse", "--show-prefix")
    try:
        top = Path(top_text).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise RecoveryStageAError("Cannot resolve the Git top-level directory") from error
    if prefix:
        parsed = PurePosixPath(prefix)
        if (
            not prefix.endswith("/")
            or parsed.is_absolute()
            or "\\" in prefix
            or any(part in {"", ".", ".."} for part in parsed.parts)
        ):
            raise RecoveryStageAError("Git repository prefix is unsafe")
        project = top.joinpath(*parsed.parts).resolve(strict=True)
    else:
        project = top
    if project != root:
        raise RecoveryStageAError(
            "repo_root does not match the local Git top-level/prefix identity"
        )
    return top, prefix


def _git_status_entries(repo_root: Path) -> list[dict[str, str]]:
    result = _git_process(
        repo_root,
        (
            "-c", "status.relativePaths=false", "status",
            "--porcelain=v1", "-z", "--untracked-files=all",
        ),
        text=False,
    )
    try:
        records = result.stdout.decode("utf-8", errors="strict").split("\0")
    except UnicodeDecodeError as error:
        raise RecoveryStageAError("Git status is not valid UTF-8") from error
    entries: list[dict[str, str]] = []
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        if len(record) < 4 or record[2] != " ":
            raise RecoveryStageAError("Unexpected Git porcelain status record")
        status = record[:2]
        entry = {"status": status, "path": record[3:]}
        if status[0] in "RC" or status[1] in "RC":
            if index >= len(records) or not records[index]:
                raise RecoveryStageAError("Incomplete Git rename/copy status record")
            entry["source_path"] = records[index]
            index += 1
        entries.append(entry)
    return entries


def capture_source_git_state(repo_root: Path) -> dict[str, Any]:
    """Capture the local-only source identity used by a recovery run."""

    root = Path(repo_root).resolve(strict=True)
    _repository_identity(root)
    commit = _run_git(root, "rev-parse", "--verify", "HEAD^{commit}")
    tree = _run_git(root, "rev-parse", "--verify", "HEAD^{tree}")
    branch = _run_git(root, "symbolic-ref", "--quiet", "--short", "HEAD")
    remote_commit = _run_git(
        root, "rev-parse", "--verify", f"{REMOTE_TRACKING_REF}^{{commit}}",
    )
    ancestry = _git_process(
        root, ("merge-base", "--is-ancestor", REMOTE_TRACKING_REF, "HEAD"),
        check=False,
    )
    if ancestry.returncode not in (0, 1):
        raise RecoveryStageAError("Cannot determine local/remote-tracking ancestry")
    counts = _run_git(
        root, "rev-list", "--left-right", "--count",
        f"{REMOTE_TRACKING_REF}...HEAD",
    ).split()
    if len(counts) != 2 or any(not item.isdigit() for item in counts):
        raise RecoveryStageAError("Invalid local/remote-tracking ahead/behind counts")
    behind_count, ahead_count = (int(item) for item in counts)
    for value, context in ((commit, "HEAD"), (tree, "HEAD tree"),
                           (remote_commit, REMOTE_TRACKING_REF)):
        if _HEX40.fullmatch(value) is None:
            raise RecoveryStageAError(f"Invalid local Git {context} identity")
    entries = _git_status_entries(root)
    return {
        "commit": commit,
        "tree": tree,
        "branch": branch,
        "remote_tracking_ref": REMOTE_TRACKING_REF,
        "remote_tracking_commit": remote_commit,
        "head_matches_remote_tracking": commit == remote_commit,
        "remote_tracking_is_ancestor": ancestry.returncode == 0,
        "ahead_count": ahead_count,
        "behind_count": behind_count,
        "dirty": bool(entries),
        "status_entries": entries,
    }


def _require_clean_source_git(state: Mapping[str, Any]) -> None:
    required = {
        "commit", "tree", "branch", "remote_tracking_ref",
        "remote_tracking_commit", "head_matches_remote_tracking",
        "remote_tracking_is_ancestor", "ahead_count", "behind_count",
        "dirty", "status_entries",
    }
    if set(state) != required:
        raise RecoveryStageAError("Recovery Stage A Git state is incomplete")
    if (
        _HEX40.fullmatch(str(state["commit"])) is None
        or _HEX40.fullmatch(str(state["tree"])) is None
        or _HEX40.fullmatch(str(state["remote_tracking_commit"])) is None
        or state["branch"] != EXPECTED_BRANCH
        or state["remote_tracking_ref"] != REMOTE_TRACKING_REF
        or type(state["head_matches_remote_tracking"]) is not bool
        or state["remote_tracking_is_ancestor"] is not True
        or type(state["ahead_count"]) is not int
        or state["ahead_count"] < 0
        or state["behind_count"] != 0
        or state["head_matches_remote_tracking"] != (state["ahead_count"] == 0)
        or state["dirty"] is not False
        or state["status_entries"] != []
    ):
        raise RecoveryStageAError(
            f"Recovery Stage A requires a clean Git tree on {EXPECTED_BRANCH}"
        )
def _require_same_clean_git(
    expected: Mapping[str, Any], observed: Mapping[str, Any], *, context: str,
) -> None:
    _require_clean_source_git(observed)
    if dict(observed) != dict(expected):
        raise RecoveryStageAError(f"Git state changed {context}")


def _path_has_symlink_component(root: Path, relative: PurePosixPath) -> bool:
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _validate_output_dir(repo_root: Path, output_dir: Path) -> tuple[Path, str]:
    root = Path(repo_root).resolve(strict=True)
    raw = Path(output_dir)
    candidate = raw if raw.is_absolute() else root / raw
    candidate = Path(os.path.abspath(os.fspath(candidate)))
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise RecoveryStageAError(
            "Recovery Stage A output must be repository-relative under evidence/stage_a"
        ) from error
    if (
        len(relative.parts) < 3
        or relative.parts[:2] != ("evidence", "stage_a")
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise RecoveryStageAError(
            "Recovery Stage A output must be a named fresh root under evidence/stage_a"
        )
    if _path_has_symlink_component(root, PurePosixPath(relative.as_posix())):
        raise RecoveryStageAError("Recovery Stage A output path must not traverse symlinks")
    if candidate.exists() or candidate.is_symlink():
        raise RecoveryStageAError("Recovery Stage A output root is not fresh")

    top, _prefix = _repository_identity(root)
    policy_filenames = (
        *ARTIFACT_FILENAMES, PENDING_TERMINAL_FILENAME, UNSEALED_TERMINAL_FILENAME,
    )
    artifact_paths = [candidate / name for name in policy_filenames]
    top_relative = [path.relative_to(top).as_posix() for path in artifact_paths]
    project_relative = [path.relative_to(root).as_posix() for path in artifact_paths]
    attributes = _run_git(root, "check-attr", "filter", "--", *project_relative)
    lines = attributes.splitlines()
    if len(lines) != len(project_relative) or any(
        line != f"{path}: filter: lfs" for line, path in zip(lines, project_relative)
    ):
        raise RecoveryStageAError(
            "Recovery Stage A artifacts are not covered by the Git LFS filter"
        )
    for project_path, top_path in zip(project_relative, top_relative):
        ignored = _git_process(
            root,
            ("check-ignore", "-v", "--no-index", "--", project_path),
            check=False,
        )
        if ignored.returncode == 0:
            # With --no-index, check-ignore also reports a matching negation
            # pattern.  A leading `!` in the effective pattern means the path
            # is deliberately unignored and therefore safe to publish.
            rule = ignored.stdout.rstrip("\n").partition("\t")[0].rsplit(":", 1)[-1]
            if not rule.startswith("!"):
                raise RecoveryStageAError(
                    f"Recovery Stage A artifact would be Git-ignored: {top_path}"
                )
        elif ignored.returncode != 1:
            raise RecoveryStageAError("Cannot determine Git ignore policy for Stage A output")
    return candidate, relative.as_posix()


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _with_canonical_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    result = _plain(value)
    result["canonical_digest"] = _canonical_sha256(result)
    return result


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            _plain(value), indent=2, sort_keys=True, ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _repo_relative(repo_root: Path, path: Path, context: str) -> str:
    try:
        resolved = Path(path).resolve(strict=True)
        relative = resolved.relative_to(repo_root)
    except (OSError, RuntimeError, ValueError) as error:
        raise RecoveryStageAError(f"{context} must stay inside the active repository") from error
    if "reserve" in {part.lower() for part in relative.parts}:
        raise RecoveryStageAError(f"{context} must not access reserve data")
    return relative.as_posix()


def _validate_binding_preflight(
    repo_root: Path, binding: RecoveryV2Binding, binding_path: Path,
) -> tuple[Any, Path, dict[str, Any]]:
    if binding.path != Path(binding_path).absolute().resolve():
        raise RecoveryStageAError("Loaded recovery binding path differs from the requested path")
    binding_relative = _repo_relative(repo_root, binding.path, "Recovery binding")
    schema_relative = _repo_relative(repo_root, binding.schema_path, "Recovery binding schema")
    model_relative = _repo_relative(repo_root, binding.model_path, "Recovery model")
    data_relative = _repo_relative(repo_root, binding.data_root, "Recovery Stage C data root")
    _repo_relative(repo_root, binding.devkit_path, "Recovery official devkit")
    _repo_relative(repo_root, binding.source_manifest_path, "Recovery source manifest")
    if not model_relative.startswith("assets/models/"):
        raise RecoveryStageAError("Recovery model must remain under assets/models")
    if data_relative != "assets/data/ILSVRC2012_val":
        raise RecoveryStageAError("Recovery Stage A data root drifted")
    document = binding.document
    scope = document["scope"]
    if (
        document["legacy_project_allowed"] is not False
        or scope["stage_a_verification_authorized"] is not True
        or scope["formal_c7_c8_c9_c10_authorized"] is not False
        or scope["stage_d_d7_d8_authorized"] is not False
        or scope["stage_d_reserve_access_authorized"] is not False
    ):
        raise RecoveryStageAError("Recovery binding authorization boundary is unsafe")

    row = load_bound_stage_a_manifest_row(binding)
    image_path = (binding.data_root / row.relative_path).resolve(strict=True)
    image_relative = _repo_relative(repo_root, image_path, "Bound Stage A image")
    inputs = binding.stage_c_input
    if (
        image_relative != f"{data_relative}/{inputs['sample_relative_path']}"
        or row.image_sha256 != inputs["sample_sha256"]
        or sha256_file(image_path) != row.image_sha256
    ):
        raise RecoveryStageAError("Bound Stage A image/manifest identity drifted")
    metadata = {
        "binding_path": binding_relative,
        "binding_file_sha256": sha256_file(binding.path),
        "binding_schema_path": schema_relative,
        "binding_schema_sha256": sha256_file(binding.schema_path),
    }
    return row, image_path, metadata


def _validate_conda_environment(
    repo_root: Path, binding: RecoveryV2Binding,
) -> dict[str, Any]:
    declared = binding.document["environment"][
        "conda_environment_path_from_workspace"
    ]
    workspace = repo_root.parent
    expected = (workspace / declared).resolve(strict=True)
    observed = Path(sys.prefix).resolve(strict=True)
    executable = Path(sys.executable).resolve(strict=True)
    if not expected.is_dir():
        raise RecoveryStageAError("Bound recovery-v2 Conda environment is not a directory")
    if observed != expected or not executable.is_relative_to(expected):
        raise RecoveryStageAError(
            "Recovery Stage A must run inside the bound vitnns Conda environment"
        )
    environment = binding.document["environment"]
    required_versions = {
        "python": environment["python_version"],
        "torch": environment["torch_version"],
        "torchvision": environment["torchvision_version"],
        "timm": environment["timm_version"],
    }
    observed_versions = {
        "python": platform.python_version(),
        "torch": importlib.metadata.version("torch"),
        "torchvision": importlib.metadata.version("torchvision"),
        "timm": importlib.metadata.version("timm"),
    }
    if observed_versions != required_versions:
        raise RecoveryStageAError("Recovery Stage A runtime versions drifted")
    return {
        "declared_path_from_workspace": declared,
        "expected_prefix": str(expected),
        "observed_prefix": str(observed),
        "python_executable": str(executable),
        "prefix_match": True,
        "executable_under_prefix": True,
        "required_versions": required_versions,
        "observed_versions": observed_versions,
        "versions_match": True,
    }


def _seed_everything(seed: int, device: torch.device) -> None:
    if type(seed) is not int or seed < 0:
        raise RecoveryStageAError("Stage A seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def _resolve_device(device: str, *, required_type: str = "cuda") -> torch.device:
    try:
        resolved = torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise RecoveryStageAError(f"Invalid Stage A device: {device!r}") from error
    if required_type != "cuda" or resolved.type != required_type:
        raise RecoveryStageAError(
            "Recovery-v2 machine-bound Stage A requires an explicit CUDA device"
        )
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RecoveryStageAError("CUDA was requested but is unavailable")
        index = torch.cuda.current_device() if resolved.index is None else resolved.index
        if index < 0 or index >= torch.cuda.device_count():
            raise RecoveryStageAError(f"CUDA device index is unavailable: {index}")
        resolved = torch.device("cuda", index)
    return resolved


def _build_direct_model(
    *, state: Mapping[str, Any], model_name: str, device: torch.device,
    unfused_attention: bool,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    # Attention instances snapshot the global fused-attention setting during
    # construction, so this call must precede timm.create_model.
    set_fused_attn(not unfused_attention)
    model = timm.create_model(model_name, pretrained=False)
    loaded = model.load_state_dict(state, strict=True)
    missing = list(getattr(loaded, "missing_keys", ()))
    unexpected = list(getattr(loaded, "unexpected_keys", ()))
    if missing or unexpected:
        raise RecoveryStageAError(
            f"Strict state-dict load reported missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    if unfused_attention:
        for block in model.blocks:
            block.attn.fused_attn = False
    elif any(not block.attn.fused_attn for block in model.blocks):
        raise RecoveryStageAError("Fused reference model did not enable fused attention")
    _assert_model_contract(model)
    model.to(device)
    return model, {
        "pretrained": False,
        "strict": True,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "unfused_attention": unfused_attention,
    }


def _max_abs_error(left: torch.Tensor, right: torch.Tensor, context: str) -> float:
    if left.shape != right.shape or left.dtype != right.dtype:
        raise RecoveryStageAError(f"Tensor contract drifted for {context}")
    if left.numel() == 0 or not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise RecoveryStageAError(f"Non-finite or empty tensor encountered for {context}")
    value = float((left - right).abs().max().item())
    if not math.isfinite(value):
        raise RecoveryStageAError(f"Non-finite numerical error for {context}")
    return value


def _run_numeric_gates(
    reference: torch.nn.Module, observable: torch.nn.Module, image_tensor: torch.Tensor,
) -> tuple[Any, dict[str, Any], torch.Tensor]:
    with torch.inference_mode():
        fused_logits = reference(image_tensor).detach().clone()
    unfused_logits, plain_inputs, plain_outputs = capture_block_io(
        observable, image_tensor,
    )
    extraction = NativeViTExtractor(observable).extract(image_tensor)
    if (
        len(plain_inputs) != len(observable.blocks)
        or len(plain_outputs) != len(observable.blocks)
        or len(extraction.layers) != len(observable.blocks)
    ):
        raise RecoveryStageAError("Stage A extractor layer count drifted")

    fused_unfused_error = _max_abs_error(
        fused_logits, unfused_logits, "fused versus unfused logits",
    )
    reference_extractor_error = _max_abs_error(
        fused_logits, extraction.logits, "fused versus extractor logits",
    )
    hook_logits_error = _max_abs_error(
        unfused_logits, extraction.logits, "plain versus hooked logits",
    )
    block_input_errors = [
        _max_abs_error(expected, actual.block_input, f"block {index} input")
        for index, (expected, actual) in enumerate(zip(plain_inputs, extraction.layers))
    ]
    block_output_errors = [
        _max_abs_error(expected, actual.block_output, f"block {index} output")
        for index, (expected, actual) in enumerate(zip(plain_outputs, extraction.layers))
    ]
    attention_errors = [
        _max_abs_error(
            layer.attention, layer.attention_recomputed,
            f"block {index} attention recomputation",
        )
        for index, layer in enumerate(extraction.layers)
    ]
    attention_used_errors = [
        _max_abs_error(
            layer.attention, layer.attention_used,
            f"block {index} attention dropout identity",
        )
        for index, layer in enumerate(extraction.layers)
    ]
    block_input_error = max(block_input_errors)
    block_output_error = max(block_output_errors)
    attention_error = max(attention_errors)
    attention_used_error = max(attention_used_errors)
    fused_top5 = [int(value) for value in fused_logits[0].topk(5).indices.cpu().tolist()]
    extracted_top5 = [
        int(value) for value in extraction.logits[0].topk(5).indices.cpu().tolist()
    ]
    logits_pass = (
        fused_unfused_error < LOGITS_THRESHOLD
        and reference_extractor_error < LOGITS_THRESHOLD
    )
    hook_pass = (
        hook_logits_error == 0.0
        and block_input_error == 0.0
        and block_output_error == 0.0
    )
    attention_pass = (
        attention_error < ATTENTION_THRESHOLD and attention_used_error == 0.0
    )
    top5_pass = fused_top5 == extracted_top5
    validation = {
        "schema_version": 1,
        "document_type": "recovery_v2_stage_a_numeric_validation",
        "status": "PASS" if logits_pass and hook_pass and attention_pass and top5_pass else "FAIL",
        "fused_vs_unfused_logits_max_abs_error": fused_unfused_error,
        "reference_vs_extractor_logits_max_abs_error": reference_extractor_error,
        "unfused_plain_vs_hooked_logits_max_abs_error": hook_logits_error,
        "reference_vs_extractor_block_input_max_abs_error": block_input_error,
        "reference_vs_extractor_block_output_max_abs_error": block_output_error,
        "attention_recomputation_max_abs_error": attention_error,
        "attention_actual_vs_used_max_abs_error": attention_used_error,
        "per_block_attention_recomputation_max_abs_error": attention_errors,
        "logits_threshold": LOGITS_THRESHOLD,
        "attention_threshold": ATTENTION_THRESHOLD,
        "logits_pass": logits_pass,
        "hook_noninvasive_pass": hook_pass,
        "attention_pass": attention_pass,
        "top5_match_pass": top5_pass,
        "all_pass": logits_pass and hook_pass and attention_pass and top5_pass,
        "legacy_pickle_enabled": False,
        "legacy_pickle_used": False,
    }
    validation = _with_canonical_digest(validation)
    if validation["status"] != "PASS":
        raise RecoveryStageAError(f"Recovery Stage A numerical gates failed: {validation}")
    return extraction, validation, fused_logits


def _runtime_environment(device: torch.device, seed: int) -> dict[str, Any]:
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    else:
        device_name = "CPU"
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "device_name": device_name,
        "seed": seed,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _artifact_binding(payload: bytes) -> dict[str, Any]:
    return {"sha256": _sha256_bytes(payload), "bytes": len(payload)}


def _serialize_extraction(tensors: Mapping[str, torch.Tensor]) -> bytes:
    buffer = io.BytesIO()
    torch.save(dict(tensors), buffer)
    return buffer.getvalue()


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _publish_nonterminal(
    output_dir: Path, payloads: Mapping[str, bytes],
) -> None:
    if set(payloads) != set(NONTERMINAL_FILENAMES):
        raise RecoveryStageAError("Stage A nonterminal artifact set is incomplete")
    output_dir.mkdir(parents=True, exist_ok=False)
    for name in NONTERMINAL_FILENAMES:
        _write_exclusive(output_dir / name, payloads[name])
    for name in NONTERMINAL_FILENAMES:
        payload = payloads[name]
        path = output_dir / name
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_nlink != 1
            or path.stat().st_size != len(payload)
            or sha256_file(path) != _sha256_bytes(payload)
        ):
            raise RecoveryStageAError(f"Published Stage A artifact drifted: {name}")


def _write_pending_terminal(output_dir: Path, payload: bytes) -> None:
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RecoveryStageAError("Stage A terminal receipt is not valid UTF-8 JSON") from error
    if (
        not isinstance(document, dict)
        or document.get("document_type") != "recovery_v2_stage_a_verification"
        or document.get("status") != "PASS"
    ):
        raise RecoveryStageAError("Stage A terminal receipt is not a PASS verification")
    recorded_digest = document.pop("canonical_digest", None)
    if recorded_digest != _canonical_sha256(document):
        raise RecoveryStageAError("Stage A terminal receipt canonical digest drifted")
    _write_exclusive(output_dir / PENDING_TERMINAL_FILENAME, payload)
    pending = output_dir / PENDING_TERMINAL_FILENAME
    if (
        pending.is_symlink()
        or not pending.is_file()
        or pending.stat().st_nlink != 1
        or pending.stat().st_size != len(payload)
        or sha256_file(pending) != _sha256_bytes(payload)
    ):
        raise RecoveryStageAError("Pending Stage A terminal receipt drifted")


def _require_final_git_state(
    repo_root: Path, source: Mapping[str, Any], output_dir: Path,
    expected_filenames: Sequence[str],
) -> dict[str, Any]:
    observed = capture_source_git_state(repo_root)
    identity_keys = (
        "commit", "tree", "branch", "remote_tracking_ref",
        "remote_tracking_commit", "head_matches_remote_tracking",
        "remote_tracking_is_ancestor", "ahead_count", "behind_count",
    )
    if any(observed.get(key) != source.get(key) for key in identity_keys):
        raise RecoveryStageAError("Git identity changed during recovery Stage A")
    top, _prefix = _repository_identity(repo_root)
    expected = {
        (output_dir / name).relative_to(top).as_posix() for name in expected_filenames
    }
    actual_entries = observed.get("status_entries")
    if (
        not isinstance(actual_entries, list)
        or len(actual_entries) != len(expected)
        or any(
            set(entry) != {"status", "path"}
            or entry["status"] != "??"
            or entry["path"] not in expected
            for entry in actual_entries
        )
        or {entry["path"] for entry in actual_entries} != expected
    ):
        raise RecoveryStageAError(
            "Git state changed during recovery Stage A outside its fresh output root"
        )
    return observed


def _seal_terminal_receipt(output_dir: Path, payload: bytes) -> None:
    pending = output_dir / PENDING_TERMINAL_FILENAME
    terminal = output_dir / TERMINAL_FILENAME
    os.replace(pending, terminal)


def _finalize_terminal(
    repo_root: Path, source_git_state: Mapping[str, Any], output_dir: Path, payload: bytes,
) -> None:
    _write_pending_terminal(output_dir, payload)
    _require_final_git_state(
        repo_root, source_git_state, output_dir,
        (*NONTERMINAL_FILENAMES, PENDING_TERMINAL_FILENAME),
    )
    # Atomic rename is the last operation: no terminal PASS name can exist if
    # payload validation or the final Git-state gate fails.
    _seal_terminal_receipt(output_dir, payload)


def run_recovery_v2_stage_a(
    *,
    repo_root: Path,
    binding_path: Path,
    output_dir: Path,
    device: str = "cuda:0",
    seed: int = 20260820,
) -> dict[str, Any]:
    """Run the isolated recovery-v2 Stage A verifier.

    Every filesystem and identity precondition is checked before model loading,
    CUDA inspection, or output creation.  The returned mapping is byte-for-byte
    the document written as ``stage_a_verification.json``.
    """

    repo = Path(repo_root).resolve(strict=True)

    # Preflight phase: no output creation, torch.load, model construction, image
    # decode, or CUDA API call is permitted above the second Git check.
    source_git_state = capture_source_git_state(repo)
    _require_clean_source_git(source_git_state)
    output, output_relative = _validate_output_dir(repo, output_dir)
    binding = load_recovery_v2_binding(
        binding_path, repo_root=repo, verify_assets=True,
    )
    amendment_commit = str(binding.document["amendment"]["commit"])
    amendment_ancestry = _git_process(
        repo, ("merge-base", "--is-ancestor", amendment_commit, "HEAD"),
        check=False,
    )
    if amendment_ancestry.returncode != 0:
        if amendment_ancestry.returncode == 1:
            raise RecoveryStageAError("Bound recovery amendment commit is not an ancestor of HEAD")
        raise RecoveryStageAError("Cannot verify recovery amendment Git ancestry")
    lfs_identity = verify_recovery_v2_lfs_identity(binding, repo_root=repo)
    manifest_row, image_path, binding_files = _validate_binding_preflight(
        repo, binding, binding_path,
    )
    conda_identity = _validate_conda_environment(repo, binding)
    after_preflight_git = capture_source_git_state(repo)
    _require_same_clean_git(
        source_git_state, after_preflight_git, context="during Stage A preflight",
    )

    # Safe, direct state-dict loading.  Legacy pickle loading is intentionally
    # absent from this module and cannot be enabled by an argument.
    state = torch.load(binding.model_path, map_location="cpu", weights_only=True)
    model_identity = verify_recovery_v2_state_dict(binding, state=state)
    torch_device = _resolve_device(
        device, required_type=binding.document["environment"]["required_device_type"],
    )
    _seed_everything(seed, torch_device)
    reference, fused_load = _build_direct_model(
        state=state, model_name=binding.model["name"], device=torch_device,
        unfused_attention=False,
    )
    observable, unfused_load = _build_direct_model(
        state=state, model_name=binding.model["name"], device=torch_device,
        unfused_attention=True,
    )
    transform = canonical_transform(reference)
    with Image.open(image_path) as source:
        image_tensor = transform(source.convert("RGB")).unsqueeze(0).to(torch_device)
    extraction, validation, fused_logits = _run_numeric_gates(
        reference, observable, image_tensor,
    )

    tensors = extraction.tensor_dict(cpu=True)
    expected_tensor_names = {"logits"} | {
        f"blocks.{index}.{name}"
        for index in range(len(extraction.layers))
        for name in ("block_input", "block_output", "q", "k", "v", "attention")
    }
    if set(tensors) != expected_tensor_names:
        raise RecoveryStageAError("Recovery Stage A extraction tensor set drifted")
    extraction_bytes = _serialize_extraction(tensors)
    validation_bytes = _json_bytes(validation)
    preliminary_artifacts = {
        "extraction.pt": _artifact_binding(extraction_bytes),
        "validation.json": _artifact_binding(validation_bytes),
    }

    inputs = binding.stage_c_input
    binding_document = binding.document
    top5 = [int(value) for value in extraction.logits[0].topk(5).indices.cpu().tolist()]
    fused_top5 = [int(value) for value in fused_logits[0].topk(5).indices.cpu().tolist()]
    runtime = _runtime_environment(torch_device, seed)
    dependencies = binding_document["environment"]
    lock_path = repo / dependencies["requirements_lock_path"]
    model_record = {
        "name": binding.model["name"],
        "identity": binding.model["identity"],
        "state_dict_path": _repo_relative(repo, binding.model_path, "Recovery model"),
        "state_dict_sha256": model_identity["state_dict_sha256"],
        "state_dict_bytes": binding.model["state_dict_bytes"],
        "canonical_tensor_digest": model_identity["canonical_tensor_digest"],
        "canonical_tensor_digest_algorithm": binding.model["canonical_tensor_digest_algorithm"],
        "key_count": model_identity["key_count"],
        "parameter_count": model_identity["parameter_count"],
        "historical_artifact_recovered": binding.model["historical_artifact_recovered"],
        "historical_state_dict_sha256": binding.model["historical_state_dict_sha256"],
        "lfs": _plain(lfs_identity),
        "loader": {
            "format": "torch_state_dict",
            "torch_load_map_location": "cpu",
            "torch_load_weights_only": True,
            "fused_model": fused_load,
            "unfused_model": unfused_load,
            "legacy_pickle_enabled": False,
            "legacy_pickle_used": False,
        },
        "architecture": {
            "blocks": len(observable.blocks),
            "embed_dim": int(observable.embed_dim),
            "heads_per_block": [int(block.attn.num_heads) for block in observable.blocks],
            "head_dim_per_block": [int(block.attn.head_dim) for block in observable.blocks],
            "tokens": int(extraction.layers[0].q.shape[-2]),
        },
    }
    binding_record = {
        **binding_files,
        "binding_id": binding_document["binding_id"],
        "status": binding_document["status"],
        "canonical_digest": binding.canonical_digest,
    }
    input_record = {
        "sample_id": manifest_row.sample_id,
        "path": _repo_relative(repo, image_path, "Bound Stage A image"),
        "sha256": manifest_row.image_sha256,
        "bytes": manifest_row.size_bytes,
        "width": manifest_row.width,
        "height": manifest_row.height,
        "ground_truth_wnid": manifest_row.wnid,
        "ground_truth_index": manifest_row.imagenet1k_index,
        "label_source": manifest_row.label_source,
        "source_manifest_path": inputs["source_manifest_path"],
        "source_manifest_sha256": inputs["source_manifest_sha256"],
        "official_devkit_path": inputs["official_devkit_path"],
        "official_devkit_sha256": inputs["official_devkit_sha256"],
    }
    environment_record = {
        "binding": _plain(dependencies),
        "conda": conda_identity,
        "requirements_lock": {
            "path": dependencies["requirements_lock_path"],
            "sha256": sha256_file(lock_path),
            "bytes": lock_path.stat().st_size,
        },
        "runtime": runtime,
    }
    provenance = _with_canonical_digest({
        "schema_version": 1,
        "document_type": "recovery_v2_stage_a_provenance",
        "status": "PASS",
        "binding": binding_record,
        "amendment": _plain(binding_document["amendment"]),
        "scope": _plain(binding_document["scope"]),
        "model": model_record,
        "input": input_record,
        "preprocessing": canonical_preprocess_spec(reference).to_dict(),
        "prediction": {
            "top1_index": top5[0],
            "top5_indices": top5,
            "fused_top5_indices": fused_top5,
            "top1_correct": top5[0] == manifest_row.imagenet1k_index,
            "top5_correct": manifest_row.imagenet1k_index in top5,
            "logits_artifact": "extraction.pt",
        },
        "extractor": {
            "method": "native_timm_forward_hooks_unfused_attention",
            "captured_layers": len(extraction.layers),
            "tensor_count": len(tensors),
        },
        "numeric_gates": validation,
        "environment": environment_record,
        "source_git_state": _plain(source_git_state),
        "artifacts": preliminary_artifacts,
        "legacy_pickle": {"enabled": False, "used": False},
    })
    provenance_bytes = _json_bytes(provenance)
    artifacts = {
        "extraction.pt": preliminary_artifacts["extraction.pt"],
        "provenance.json": _artifact_binding(provenance_bytes),
        "validation.json": preliminary_artifacts["validation.json"],
    }
    verification = _with_canonical_digest({
        "schema_version": 1,
        "document_type": "recovery_v2_stage_a_verification",
        "status": "PASS",
        "binding": binding_record,
        "amendment": _plain(binding_document["amendment"]),
        "scope": _plain(binding_document["scope"]),
        "model": model_record,
        "input": input_record,
        "environment": environment_record,
        "source_git_state": _plain(source_git_state),
        "numeric_gates": validation,
        "artifacts": artifacts,
        "output": {
            "root": output_relative,
            "fresh_at_start": True,
            "repository_relative": True,
            "git_lfs_filter": "lfs",
        },
        "git_end_state_policy": {
            "identity_must_equal_source": True,
            "only_fresh_output_files_may_be_untracked": True,
            "expected_output_files": list(ARTIFACT_FILENAMES),
            "checked_before_terminal_seal": True,
            "terminal_receipt_published_last": True,
        },
        "legacy_pickle": {"enabled": False, "used": False},
    })
    # Recheck the pristine source immediately before the first output mkdir.
    before_publish_git = capture_source_git_state(repo)
    _require_same_clean_git(
        source_git_state, before_publish_git, context="before Stage A publication",
    )
    payloads = {
        "extraction.pt": extraction_bytes,
        "provenance.json": provenance_bytes,
        "validation.json": validation_bytes,
    }
    _publish_nonterminal(output, payloads)
    observed_end_state = _require_final_git_state(
        repo, source_git_state, output, NONTERMINAL_FILENAMES,
    )
    unsigned_verification = dict(verification)
    unsigned_verification.pop("canonical_digest")
    unsigned_verification["observed_git_state_before_terminal_seal"] = _plain(
        observed_end_state
    )
    verification = _with_canonical_digest(unsigned_verification)
    verification_bytes = _json_bytes(verification)
    _finalize_terminal(repo, source_git_state, output, verification_bytes)
    return verification


__all__ = [
    "RecoveryStageAError",
    "capture_source_git_state",
    "run_recovery_v2_stage_a",
]
