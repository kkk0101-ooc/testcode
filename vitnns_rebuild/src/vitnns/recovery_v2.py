from __future__ import annotations

import hashlib
import csv
import json
import os
import sys
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from .io_utils import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINDING_PATH = REPO_ROOT / "config/recovery_v2_binding.json"
DEFAULT_BINDING_SCHEMA_PATH = REPO_ROOT / "config/recovery_v2_binding.schema.json"
EXPECTED_BINDING_ID = "RECOVERY_V2_STAGE_A_20260901"
EXPECTED_AMENDMENT_ID = "RECOVERY-C7-RERUN-20260901"
EXPECTED_AMENDMENT_COMMIT = "4a87e4d9cb71fe9c394bb2fdd5d4e71bbfe8aaed"
HISTORICAL_STATE_DICT_SHA256 = (
    "2e718037429f18ec77aa629471beb2e99993d7383375188bab777aba04d5e4d0"
)


class RecoveryV2BindingError(ValueError):
    """Raised when the recovery-v2 machine binding is invalid or drifted."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RecoveryV2BindingError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> None:
    raise RecoveryV2BindingError(f"Non-finite JSON value is forbidden: {token}")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except OSError as error:
        raise RecoveryV2BindingError(f"Cannot read recovery-v2 JSON: {path}") from error
    except UnicodeDecodeError as error:
        raise RecoveryV2BindingError(f"Binding is not UTF-8: {path}") from error
    except json.JSONDecodeError as error:
        raise RecoveryV2BindingError(f"Invalid JSON binding: {path}: {error}") from error


def _regular_single_link_file(path: Path, context: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise RecoveryV2BindingError(f"{context} must be an ordinary file")
    if path.stat().st_nlink != 1:
        raise RecoveryV2BindingError(f"{context} must have exactly one hard link")


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_json_file_sha256(path: Path) -> str:
    return canonical_json_sha256(_read_json(path))


def _self_digest(document: Mapping[str, Any]) -> str:
    payload = dict(document)
    payload.pop("canonical_digest", None)
    return canonical_json_sha256(payload)


def _resolve_project_path(repo_root: Path, raw: str, context: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise RecoveryV2BindingError(f"{context} must be a non-empty string")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise RecoveryV2BindingError(f"{context} must stay under the active project")
    candidate = repo_root.joinpath(*relative.parts)
    current = repo_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise RecoveryV2BindingError(f"{context} must not traverse a symlink")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(repo_root):
        raise RecoveryV2BindingError(f"{context} escapes the active project")
    return resolved


def _resolve_declared_path(base: Path, raw: str, context: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise RecoveryV2BindingError(f"{context} must be a non-empty string")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise RecoveryV2BindingError(f"{context} must stay under its declared root")
    current = base
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise RecoveryV2BindingError(f"{context} must not traverse a symlink")
    resolved = current.resolve()
    if not resolved.is_relative_to(base):
        raise RecoveryV2BindingError(f"{context} escapes its declared root")
    return resolved


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class RecoveryV2Binding:
    path: Path
    schema_path: Path
    document: Mapping[str, Any]
    model_path: Path
    data_root: Path
    devkit_path: Path
    source_manifest_path: Path

    @property
    def model(self) -> Mapping[str, Any]:
        return self.document["model"]

    @property
    def stage_c_input(self) -> Mapping[str, Any]:
        return self.document["stage_c_input"]

    @property
    def canonical_digest(self) -> str:
        return self.document["canonical_digest"]


def load_recovery_v2_binding(
    path: Path | str = DEFAULT_BINDING_PATH,
    *,
    schema_path: Path | str = DEFAULT_BINDING_SCHEMA_PATH,
    repo_root: Path | str = REPO_ROOT,
    verify_assets: bool = False,
) -> RecoveryV2Binding:
    repo = Path(repo_root).resolve()
    raw_binding_path = Path(path).absolute()
    raw_schema_path = Path(schema_path).absolute()
    if raw_binding_path.is_symlink() or raw_schema_path.is_symlink():
        raise RecoveryV2BindingError("Recovery-v2 binding files must not be symlinks")
    binding_path = raw_binding_path.resolve()
    binding_schema_path = raw_schema_path.resolve()
    if not binding_path.is_relative_to(repo) or not binding_schema_path.is_relative_to(repo):
        raise RecoveryV2BindingError(
            "Recovery-v2 binding and schema must stay under the active project"
        )
    _regular_single_link_file(binding_path, "Recovery-v2 binding")
    _regular_single_link_file(binding_schema_path, "Recovery-v2 binding schema")
    document = _read_json(binding_path)
    schema = _read_json(binding_schema_path)
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(document),
        key=lambda error: tuple(str(item) for item in error.absolute_path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(item) for item in error.absolute_path) or "root"
        raise RecoveryV2BindingError(f"Recovery-v2 schema failure at {location}: {error.message}")

    if document["binding_id"] != EXPECTED_BINDING_ID:
        raise RecoveryV2BindingError("Unexpected recovery-v2 binding ID")
    if document["canonical_digest"] != _self_digest(document):
        raise RecoveryV2BindingError("Recovery-v2 binding canonical digest mismatch")
    amendment = document["amendment"]
    if amendment["id"] != EXPECTED_AMENDMENT_ID or amendment["commit"] != EXPECTED_AMENDMENT_COMMIT:
        raise RecoveryV2BindingError("Recovery-v2 amendment identity drifted")
    if document["legacy_project_allowed"] is not False:
        raise RecoveryV2BindingError("Legacy project use must remain disabled")
    scope = document["scope"]
    if (
        scope["stage_a_verification_authorized"] is not True
        or scope["formal_c7_c8_c9_c10_authorized"] is not False
        or scope["stage_d_d7_d8_authorized"] is not False
        or scope["stage_d_reserve_access_authorized"] is not False
    ):
        raise RecoveryV2BindingError("Recovery-v2 authorization boundary drifted")

    model = document["model"]
    if document["active_project"] != "vitnns_rebuild":
        raise RecoveryV2BindingError("Recovery-v2 active project drifted")
    if model["state_dict_sha256"] == HISTORICAL_STATE_DICT_SHA256:
        raise RecoveryV2BindingError("Recovery-v2 must not alias the historical artifact")
    if model["historical_artifact_recovered"] is not False:
        raise RecoveryV2BindingError("Historical artifact must remain recorded as missing")
    if model["lfs_oid_sha256"] != model["state_dict_sha256"]:
        raise RecoveryV2BindingError("Recovery-v2 LFS OID and file SHA-256 disagree")
    if model["identity"] != "RECOVERY_V2_NEW_ARTIFACT_NOT_HISTORICAL_2E718":
        raise RecoveryV2BindingError("Recovery-v2 model identity drifted")
    expected_algorithm = (
        "sha256_sorted_key_utf8_plus_torch_dtype_utf8_plus_python_shape_tuple_utf8_"
        "plus_cpu_contiguous_native_little_endian_numpy_c_order_bytes_no_delimiters"
    )
    if model["canonical_tensor_digest_algorithm"] != expected_algorithm:
        raise RecoveryV2BindingError("Recovery-v2 tensor-digest algorithm drifted")
    if sys.byteorder != "little":
        raise RecoveryV2BindingError(
            "The recorded tensor-digest algorithm requires a little-endian host"
        )

    amendment_path = _resolve_project_path(repo, amendment["path"], "amendment.path")
    model_path = _resolve_project_path(repo, model["state_dict_path"], "model.state_dict_path")
    inputs = document["stage_c_input"]
    data_root = _resolve_project_path(repo, inputs["data_root"], "stage_c_input.data_root")
    devkit_path = _resolve_project_path(repo, inputs["official_devkit_path"], "stage_c_input.official_devkit_path")
    source_manifest_path = _resolve_project_path(repo, inputs["source_manifest_path"], "stage_c_input.source_manifest_path")
    additional_input_paths = {
        "source_manifest_metadata_sha256": _resolve_project_path(
            repo, inputs["source_manifest_metadata_path"],
            "stage_c_input.source_manifest_metadata_path",
        ),
        "registry_sha256": _resolve_project_path(
            repo, inputs["registry_path"], "stage_c_input.registry_path"
        ),
        "registry_metadata_sha256": _resolve_project_path(
            repo, inputs["registry_metadata_path"],
            "stage_c_input.registry_metadata_path",
        ),
    }
    lock_path = _resolve_project_path(
        repo, document["environment"]["requirements_lock_path"],
        "environment.requirements_lock_path",
    )
    sample_path = _resolve_declared_path(
        data_root, inputs["sample_relative_path"],
        "stage_c_input.sample_relative_path",
    )

    if sha256_file(amendment_path) != amendment["sha256"]:
        raise RecoveryV2BindingError("Recovery amendment SHA-256 drifted")
    c7 = document["c7_contracts"]
    c7_paths = {
        "stage_c_config_sha256": _resolve_project_path(
            repo, c7["stage_c_config_path"], "c7_contracts.stage_c_config_path"
        ),
        "protocol_sha256": _resolve_project_path(
            repo, c7["protocol_path"], "c7_contracts.protocol_path"
        ),
        "acceptance_sha256": _resolve_project_path(
            repo, c7["acceptance_path"], "c7_contracts.acceptance_path"
        ),
        "diagnostic_run_spec_sha256": _resolve_project_path(
            repo, c7["diagnostic_run_spec_path"],
            "c7_contracts.diagnostic_run_spec_path",
        ),
        "formal_run_spec_sha256": _resolve_project_path(
            repo, c7["formal_run_spec_path"], "c7_contracts.formal_run_spec_path"
        ),
    }
    drifted_contracts = [
        key for key, contract_path in c7_paths.items()
        if sha256_file(contract_path) != c7[key]
    ]
    if drifted_contracts:
        raise RecoveryV2BindingError(
            "Recovery-v2 C7 contract drifted: " + ", ".join(drifted_contracts)
        )
    drifted_inputs = [
        key for key, asset_path in additional_input_paths.items()
        if sha256_file(asset_path) != inputs[key]
    ]
    if drifted_inputs:
        raise RecoveryV2BindingError(
            "Recovery-v2 Stage C input contract drifted: "
            + ", ".join(drifted_inputs)
        )
    for prefix in ("diagnostic", "formal"):
        observed_canonical = _canonical_json_file_sha256(
            c7_paths[f"{prefix}_run_spec_sha256"]
        )
        if observed_canonical != c7[f"{prefix}_run_spec_canonical_sha256"]:
            raise RecoveryV2BindingError(
                f"Recovery-v2 {prefix} run-spec canonical digest drifted"
            )
    if verify_assets:
        for asset_path, context in (
            (amendment_path, "Recovery amendment"),
            (model_path, "Recovery-v2 state dict"),
            (devkit_path, "Official devkit"),
            (source_manifest_path, "Stage C source manifest"),
            (sample_path, "Stage A sample"),
            (lock_path, "Dependency lock"),
            *((path, f"Recovery-v2 input {key}") for key, path in additional_input_paths.items()),
            *((path, f"Recovery-v2 C7 contract {key}") for key, path in c7_paths.items()),
        ):
            _regular_single_link_file(asset_path, context)
        observed = {
            "model.state_dict_sha256": sha256_file(model_path),
            "stage_c_input.official_devkit_sha256": sha256_file(devkit_path),
            "stage_c_input.source_manifest_sha256": sha256_file(source_manifest_path),
            "stage_c_input.sample_sha256": sha256_file(sample_path),
            "environment.requirements_lock_sha256": sha256_file(lock_path),
        }
        expected = {
            "model.state_dict_sha256": model["state_dict_sha256"],
            "stage_c_input.official_devkit_sha256": inputs["official_devkit_sha256"],
            "stage_c_input.source_manifest_sha256": inputs["source_manifest_sha256"],
            "stage_c_input.sample_sha256": inputs["sample_sha256"],
            "environment.requirements_lock_sha256": document["environment"][
                "requirements_lock_sha256"
            ],
        }
        drift = [key for key, value in expected.items() if observed[key] != value]
        if drift:
            raise RecoveryV2BindingError(
                "Recovery-v2 asset identity drifted: " + ", ".join(drift)
            )
        if model_path.stat().st_size != model["state_dict_bytes"]:
            raise RecoveryV2BindingError("Recovery-v2 state-dict byte count mismatch")
    return RecoveryV2Binding(
        path=binding_path, schema_path=binding_schema_path,
        document=_freeze(document), model_path=model_path,
        data_root=data_root, devkit_path=devkit_path,
        source_manifest_path=source_manifest_path,
    )


def canonical_state_dict_tensor_digest(state: Mapping[str, Any]) -> str:
    """Reproduce the x86_64 recovery digest recorded by the amendment."""

    import numpy as np
    import torch

    if not isinstance(state, Mapping):
        raise RecoveryV2BindingError("State dict must be a mapping")
    digest = hashlib.sha256()
    for key in sorted(state):
        if not isinstance(key, str):
            raise RecoveryV2BindingError("State-dict keys must be strings")
        tensor = state[key]
        if not isinstance(tensor, torch.Tensor):
            raise RecoveryV2BindingError(f"State-dict value is not a tensor: {key}")
        value = tensor.detach().cpu().contiguous()
        array = value.numpy()
        if array.dtype.byteorder == ">" or (
            array.dtype.byteorder == "=" and sys.byteorder != "little"
        ):
            array = array.astype(array.dtype.newbyteorder("<"), copy=False)
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def verify_recovery_v2_state_dict(
    binding: RecoveryV2Binding, *, state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    import torch

    actual_file_sha256 = sha256_file(binding.model_path)
    if actual_file_sha256 != binding.model["state_dict_sha256"]:
        raise RecoveryV2BindingError("Recovery-v2 state-dict SHA-256 mismatch")
    if state is None:
        state = torch.load(binding.model_path, map_location="cpu", weights_only=True)
    if len(state) != binding.model["expected_key_count"]:
        raise RecoveryV2BindingError("Recovery-v2 state-dict key count mismatch")
    tensor_digest = canonical_state_dict_tensor_digest(state)
    if tensor_digest != binding.model["canonical_tensor_digest"]:
        raise RecoveryV2BindingError("Recovery-v2 canonical tensor digest mismatch")
    parameter_count = sum(int(value.numel()) for value in state.values())
    if parameter_count != binding.model["expected_parameter_count"]:
        raise RecoveryV2BindingError("Recovery-v2 state-dict parameter count mismatch")
    return {
        "state_dict_sha256": actual_file_sha256,
        "canonical_tensor_digest": tensor_digest,
        "key_count": len(state),
        "parameter_count": parameter_count,
    }


def verify_recovery_v2_lfs_identity(
    binding: RecoveryV2Binding, *, repo_root: Path | str = REPO_ROOT,
) -> dict[str, Any]:
    """Bind the hydrated model bytes to the tracked Git LFS pointer."""

    repo = Path(repo_root).resolve()
    environment = os.environ.copy()
    environment.update({
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_LFS_SKIP_SMUDGE": "1",
    })
    try:
        top = Path(subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], cwd=repo, check=True,
            capture_output=True, text=True, timeout=30, env=environment,
        ).stdout.strip()).resolve()
    except (OSError, subprocess.SubprocessError) as error:
        raise RecoveryV2BindingError("Cannot resolve the Git LFS repository") from error
    relative = binding.model_path.relative_to(top).as_posix()
    try:
        result = subprocess.run(
            ["git", "lfs", "ls-files", "-l"],
            cwd=top, check=True, capture_output=True, text=True, timeout=30,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RecoveryV2BindingError("Cannot inspect the Git LFS model identity") from error
    lines = [
        line.strip() for line in result.stdout.splitlines()
        if line.strip().endswith(" " + relative)
    ]
    if len(lines) != 1:
        raise RecoveryV2BindingError("Recovery-v2 model has no unique Git LFS entry")
    fields = lines[0].split(maxsplit=2)
    if len(fields) != 3 or fields[0] != binding.model["lfs_oid_sha256"]:
        raise RecoveryV2BindingError("Recovery-v2 model Git LFS OID mismatch")
    if fields[2] != relative:
        raise RecoveryV2BindingError("Recovery-v2 model Git LFS path mismatch")
    hydrated = fields[1] == "*"
    if not hydrated:
        raise RecoveryV2BindingError("Recovery-v2 model Git LFS object is not hydrated")
    return {"path": relative, "oid_sha256": fields[0], "hydrated": hydrated}


def load_bound_stage_a_manifest_row(binding: RecoveryV2Binding):
    """Load the single bound Stage A row without scanning the 500 JPEGs."""

    from .dataset import ManifestRow
    from .labels import VAL_PATTERN, validation_wnid_map

    inputs = binding.stage_c_input
    required = tuple(ManifestRow.__dataclass_fields__)
    with binding.source_manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != required:
            raise RecoveryV2BindingError("Stage C source manifest columns drifted")
        matches = [
            row for row in reader
            if row["relative_path"] == inputs["sample_relative_path"]
        ]
    if len(matches) != 1:
        raise RecoveryV2BindingError("Bound Stage A sample is not unique in the manifest")
    row = matches[0]
    if (
        row["sample_id"] != f"ILSVRC2012_val/{inputs['sample_relative_path']}"
        or row["image_sha256"] != inputs["sample_sha256"]
        or row["wnid"] != inputs["sample_wnid"]
        or int(row["imagenet1k_index"]) != inputs["sample_imagenet1k_index"]
        or row["label_source"] != "official_devkit_val_gt"
    ):
        raise RecoveryV2BindingError("Bound Stage A manifest row drifted")
    match = VAL_PATTERN.fullmatch(inputs["sample_relative_path"])
    if match is None:
        raise RecoveryV2BindingError("Bound Stage A sample name is invalid")
    official = validation_wnid_map(binding.devkit_path)
    if official[int(match.group(1))] != inputs["sample_wnid"]:
        raise RecoveryV2BindingError("Bound Stage A official label drifted")
    return ManifestRow(
        sample_id=row["sample_id"], relative_path=row["relative_path"],
        image_sha256=row["image_sha256"], size_bytes=int(row["size_bytes"]),
        width=int(row["width"]), height=int(row["height"]),
        wnid=row["wnid"], imagenet1k_index=int(row["imagenet1k_index"]),
        label_source=row["label_source"],
    )
