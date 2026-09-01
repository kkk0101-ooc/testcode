"""Strict, acyclic provenance for one Stage C execution unit.

The provenance document never contains its own file hash or the hash of the
manifest which contains it.  The retention layer owns that outer hash chain:
manifest -> provenance artifact, then status -> manifest.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .constants import MODEL_NAME, MODEL_SHA256


STAGE_C_PROVENANCE_SCHEMA_VERSION = 1
STAGE_C_PROVENANCE_DOCUMENT_TYPE = "stage_c_execution_unit_provenance"
SELECTION_NOT_APPLICABLE = "not_applicable_pre_c8"
SELECTION_PRESENT = "present"


class StageCProvenanceError(ValueError):
    """Raised when a Stage C provenance document violates its contract."""


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return copy.deepcopy(value)


def _json_safe(value: Any, context: str = "provenance") -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise StageCProvenanceError(f"{context} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"{context} keys must be non-empty strings")
            _json_safe(item, f"{context}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _json_safe(item, f"{context}[{index}]")
        return
    raise TypeError(
        f"{context} contains non-JSON-safe {type(value).__name__}"
    )


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Encode a mapping with the canonical UnitKey JSON convention."""

    if not isinstance(value, Mapping):
        raise TypeError("canonical JSON input must be a mapping")
    plain = _thaw(value)
    _json_safe(plain, "canonical JSON input")
    return json.dumps(
        plain,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def capture_git_state(repo_root: Path | str) -> dict[str, Any]:
    """Capture the real commit and dirty flag without claiming cleanliness."""

    root = Path(repo_root).resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout
    return {"commit": commit, "dirty": bool(status.strip())}


@dataclass(frozen=True)
class StageCProvenance(Mapping[str, Any]):
    """Deeply read-only provenance with a detached plain-dict export."""

    unit_identity: Mapping[str, Any]
    bindings: Mapping[str, Any]
    schema_version: int = STAGE_C_PROVENANCE_SCHEMA_VERSION
    document_type: str = STAGE_C_PROVENANCE_DOCUMENT_TYPE
    trusted_recovery_context: InitVar[Any | None] = field(
        default=None, kw_only=True, repr=False,
    )
    require_trusted_recovery_context: InitVar[bool] = field(
        default=False, kw_only=True, repr=False,
    )

    def __post_init__(
        self,
        trusted_recovery_context: Any | None,
        require_trusted_recovery_context: bool,
    ) -> None:
        document = {
            "schema_version": self.schema_version,
            "document_type": self.document_type,
            "unit_identity": _thaw(self.unit_identity),
            "unit_identity_sha256": unit_identity_digest(self.unit_identity),
            "bindings": _thaw(self.bindings),
        }
        validate_stage_c_provenance(
            document,
            trusted_recovery_context=trusted_recovery_context,
            require_trusted_recovery_context=require_trusted_recovery_context,
        )
        object.__setattr__(self, "unit_identity", _freeze(document["unit_identity"]))
        object.__setattr__(self, "bindings", _freeze(document["bindings"]))

    @property
    def unit_identity_sha256(self) -> str:
        return unit_identity_digest(self.unit_identity)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "document_type": self.document_type,
            "unit_identity": _thaw(self.unit_identity),
            "unit_identity_sha256": self.unit_identity_sha256,
            "bindings": _thaw(self.bindings),
        }

    def __getitem__(self, key: str) -> Any:
        if key in {"schema_version", "document_type", "unit_identity", "bindings"}:
            return getattr(self, key)
        if key == "unit_identity_sha256":
            return self.unit_identity_sha256
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter((
            "schema_version", "document_type", "unit_identity",
            "unit_identity_sha256", "bindings",
        ))

    def __len__(self) -> int:
        return 5


# The implementation below stays Mapping-driven so stage_c_plan can be added
# without a provenance -> plan import cycle.

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_SEMVER = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.-]+)?")
_SAMPLE_ID = re.compile(r"ILSVRC2012_val/ILSVRC2012_val_[0-9]{8}\.JPEG")

_HISTORICAL_MODEL_FIELDS = {
    "name", "state_dict_path", "state_dict_sha256",
}
_RECOVERY_MODEL_FIELDS = _HISTORICAL_MODEL_FIELDS | {
    "identity",
    "canonical_tensor_digest",
    "historical_artifact_recovered",
    "historical_state_dict_sha256",
    "recovery_binding_path",
    "recovery_binding_file_sha256",
    "recovery_binding_canonical_digest",
    "recovery_launch_record_path",
    "recovery_launch_record_file_sha256",
    "recovery_launch_record_canonical_digest",
}
_RECOVERY_MODEL_MARKERS = _RECOVERY_MODEL_FIELDS - _HISTORICAL_MODEL_FIELDS
_RECOVERY_MODEL_IDENTITY_FIELDS = (
    "name",
    "state_dict_path",
    "state_dict_sha256",
    "identity",
    "canonical_tensor_digest",
    "historical_artifact_recovered",
    "historical_state_dict_sha256",
)
_RECOVERY_V2_MODEL_IDENTITY = (
    "RECOVERY_V2_NEW_ARTIFACT_NOT_HISTORICAL_2E718"
)

_UNIT_FIELDS = {
    "schema_version", "sample_id", "method_id", "config_id", "block_index",
    "head_index", "target_ref", "canonical_digest",
}
_BINDING_FIELDS = {
    "code", "model", "input", "dependencies", "registry", "protocol",
    "acceptance", "configuration", "selection", "method", "target", "map",
    "baseline", "ig_schedule", "seeds", "perturbations", "status", "failure",
    "resources", "retention", "artifact_inventory",
}
_METHOD_FIELDS = {
    "id", "exact_name", "version", "comparison_line",
    "explanation_target_ref", "fidelity_target_ref", "class_agnostic",
    "fidelity_eligible", "structural_reference_only", "native_map",
}

# exact_name, comparison line, explanation target, fidelity target,
# class agnostic, fidelity eligible, structural only, native map, projection,
# sign, metric-map normalization
_METHODS: dict[
    str,
    tuple[
        str, str, str | None, str | None, bool, bool, bool, str, str, str, str
    ],
] = {
    "direct_raw_cls_attention_per_head": (
        "direct raw CLS attention (matched block/head)", "value_path", None,
        None, True, False, True, "nonnegative_14x14_patch_mass",
        "raw_cls_attention_to_patch", "nonnegative",
        "patch_mass_renormalized_to_one_after_cls_exclusion",
    ),
    "conditional_fixed_clean_attention_head_direction_value_path_ig": (
        "NNS fixed-clean-attention head-direction Value-path IG", "value_path",
        "fixed_clean_attention_cls_head_direction",
        "fixed_clean_attention_cls_head_direction", True, True, False,
        "signed_float32_224x224_rgb_channel_sum", "pixel_mass_to_patch_sum",
        "signed", "none_signed_mass_preserving",
    ),
    "matched_fixed_clean_attention_patch_token_contribution": (
        "matched fixed-clean-attention patch-token contribution", "value_path",
        "fixed_clean_attention_cls_head_direction",
        "fixed_clean_attention_cls_head_direction", True, True, False,
        "signed_float32_14x14_patch_mass_plus_separate_cls_scalar",
        "patch_mass_to_pixel_uniform", "signed", "none_signed_mass_preserving",
    ),
    "direct_raw_cls_attention_last_block_head_mean": (
        "direct raw CLS attention (block 11 head mean)", "class_logit", None,
        "clean_predicted_class_raw_logit", True, True, False,
        "nonnegative_14x14_patch_mass", "raw_cls_attention_to_patch", "nonnegative",
        "patch_mass_renormalized_to_one_after_cls_exclusion",
    ),
    "attention_rollout_mean_head_residual_all_blocks": (
        "attention rollout (mean-head residual, blocks 0-11)", "class_logit",
        None, "clean_predicted_class_raw_logit", True, True, False,
        "nonnegative_14x14_patch_mass", "rollout_cls_to_patch", "nonnegative",
        "patch_mass_renormalized_to_one_after_cls_exclusion",
    ),
    "vit_grad_cam_block11_norm1": (
        "ViT Grad-CAM (blocks[11].norm1, clean predicted-class raw logit)",
        "class_logit", "clean_predicted_class_raw_logit",
        "clean_predicted_class_raw_logit", False, True, False,
        "nonnegative_float32_14x14", "identity_14x14", "nonnegative", "none",
    ),
    "full_model_class_logit_ig": (
        "full-model class-logit IG", "class_logit",
        "clean_predicted_class_raw_logit", "clean_predicted_class_raw_logit",
        False, True, False, "signed_float32_224x224_rgb_channel_sum",
        "pixel_mass_to_patch_sum", "signed",
        "none_signed_mass_preserving",
    ),
}
_PER_HEAD_METHODS = frozenset(tuple(_METHODS)[:3])
_PERTURBATION_IDS = (
    "translate_right_down_4",
    "center_crop_216_resize_224",
    "gaussian_rgb_std_0_01",
)
_PERTURBATION_PARAMETER_FIELDS = {
    "translate_right_down_4": {
        "kind", "dx_pixels", "dy_pixels", "padding", "interpolation",
        "inverse_map", "valid_pixel_box_half_open", "fully_valid_patch_rows",
        "fully_valid_patch_cols",
    },
    "center_crop_216_resize_224": {
        "kind", "crop_box_half_open", "crop_height", "crop_width",
        "resize_height", "resize_width", "image_interpolation",
        "image_antialias", "inverse_map_interpolation",
        "inverse_map_antialias", "inverse_mass_jacobian_multiplier",
        "valid_pixel_box_half_open", "fully_valid_patch_rows",
        "fully_valid_patch_cols",
    },
    "gaussian_rgb_std_0_01": {
        "kind", "mean", "std_rgb_0_to_1", "generator_device",
        "generator_dtype", "base_seed", "sample_seed_rule", "seed",
        "clamp_min", "clamp_max", "inverse_map", "valid_pixel_box_half_open",
        "fully_valid_patch_rows", "fully_valid_patch_cols",
    },
}
_TRANSLATION_PARAMETERS = {
    "kind": "integer_translation",
    "dx_pixels": 4,
    "dy_pixels": 4,
    "padding": "reflection",
    "interpolation": "none",
    "inverse_map": (
        "copy_perturbed_rows_4_223_cols_4_223_to_clean_rows_0_219_cols_0_219"
    ),
    "valid_pixel_box_half_open": [0, 220, 0, 220],
    "fully_valid_patch_rows": list(range(13)),
    "fully_valid_patch_cols": list(range(13)),
}
_SCALE_PARAMETERS = {
    "kind": "center_crop_scale",
    "crop_box_half_open": [4, 220, 4, 220],
    "crop_height": 216,
    "crop_width": 216,
    "resize_height": 224,
    "resize_width": 224,
    "image_interpolation": "bicubic",
    "image_antialias": True,
    "inverse_map_interpolation": "bilinear",
    "inverse_map_antialias": True,
    "inverse_mass_jacobian_multiplier": 1.075445816186557,
    "valid_pixel_box_half_open": [4, 220, 4, 220],
    "fully_valid_patch_rows": list(range(1, 13)),
    "fully_valid_patch_cols": list(range(1, 13)),
}
CANONICAL_RGB_HASH_DOMAIN = (
    "canonical_224x224_rgb_float32_0_to_1_before_normalization"
)
NORMALIZED_INPUT_HASH_DOMAIN = "normalized_model_input_float32"
PRE_TARGET_FAILURE_PHASES = frozenset({
    "setup",
    "input_validation",
    "model_loading",
    "target_resolution",
    # The runner uses this umbrella phase when the backend raises before it
    # can return a payload.  The unavailable target/input invariants below
    # prevent this name from weakening successful or resolved records.
    "backend_or_persistence",
})


def _exact(value: Any, fields: set[str], context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) and callable(getattr(value, "to_dict", None)):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    actual = set(value)
    missing, unknown = fields - actual, actual - fields
    if missing:
        raise StageCProvenanceError(
            f"{context} is missing fields: {sorted(missing)}"
        )
    if unknown:
        raise StageCProvenanceError(
            f"{context} has unknown fields: {sorted(unknown)}"
        )
    return value


def _nonempty(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{context} must be a non-empty string")
    return value


def _sha(value: Any, context: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise StageCProvenanceError(
            f"{context} must be a lowercase SHA-256"
        )
    return value


def _strict_int(value: Any, context: str, minimum: int = 0) -> int:
    if type(value) is not int:
        raise TypeError(f"{context} must be an integer (not bool)")
    if value < minimum:
        raise StageCProvenanceError(f"{context} must be >= {minimum}")
    return value


def _relative_path(value: Any, context: str) -> str:
    raw = _nonempty(value, context)
    if raw.startswith("/") or "\\" in raw:
        raise StageCProvenanceError(
            f"{context} must be a normalized relative POSIX path"
        )
    if any(part in {"", ".", ".."} for part in raw.split("/")):
        raise StageCProvenanceError(
            f"{context} must be a normalized relative POSIX path"
        )
    return raw


def unit_identity_digest(identity: Mapping[str, Any]) -> str:
    """Hash UnitKey's six semantic fields, excluding the digest itself."""

    unit = _exact(identity, _UNIT_FIELDS, "unit_identity")
    semantic = {
        key: unit[key]
        for key in (
            "schema_version", "sample_id", "method_id", "config_id",
            "block_index", "head_index", "target_ref",
        )
    }
    return canonical_sha256(semantic)


def build_unit_identity(
    *, sample_id: str, method_id: str, config_id: str, block_index: int | None,
    head_index: int | None, target_ref: str | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 1,
        "sample_id": sample_id,
        "method_id": method_id,
        "config_id": config_id,
        "block_index": block_index,
        "head_index": head_index,
        "target_ref": target_ref,
    }
    result["canonical_digest"] = canonical_sha256(result)
    _validate_unit_identity(result)
    return result


def _validate_unit_identity(value: Any) -> Mapping[str, Any]:
    unit = _exact(value, _UNIT_FIELDS, "unit_identity")
    if type(unit["schema_version"]) is not int or unit["schema_version"] != 1:
        raise StageCProvenanceError("unit_identity.schema_version must be 1")
    sample_id = unit["sample_id"]
    if not isinstance(sample_id, str) or not _SAMPLE_ID.fullmatch(sample_id):
        raise StageCProvenanceError("unit_identity.sample_id is invalid")
    method_id = _nonempty(unit["method_id"], "unit_identity.method_id")
    if method_id not in _METHODS:
        raise StageCProvenanceError("unit_identity.method_id is unregistered")
    block, head = unit["block_index"], unit["head_index"]
    if (block is None) != (head is None):
        raise StageCProvenanceError(
            "unit_identity block_index/head_index must both be set or both null"
        )
    if method_id in _PER_HEAD_METHODS:
        for name, index in (("block_index", block), ("head_index", head)):
            _strict_int(index, f"unit_identity.{name}")
            if index >= 12:
                raise StageCProvenanceError(
                    f"unit_identity.{name} must be in [0,11]"
                )
        if unit["config_id"] != f"block_{block:02d}_head_{head:02d}":
            raise StageCProvenanceError(
                "per-head unit config_id disagrees with block/head"
            )
    elif block is not None:
        raise StageCProvenanceError(
            "global methods must not be assigned fake block/head indices"
        )
    elif unit["config_id"] != f"global_{method_id}":
        raise StageCProvenanceError(
            "global unit config_id must identify its singleton method config"
        )
    if unit["target_ref"] != _METHODS[method_id][3]:
        raise StageCProvenanceError(
            "unit_identity.target_ref disagrees with the method"
        )
    if unit["canonical_digest"] != unit_identity_digest(unit):
        raise StageCProvenanceError(
            "unit_identity.canonical_digest does not match its semantic fields"
        )
    return unit


def _validate_code(value: Any, observed: Mapping[str, Any] | None) -> None:
    code = _exact(value, {"commit", "dirty"}, "bindings.code")
    if not isinstance(code["commit"], str) or not _COMMIT.fullmatch(code["commit"]):
        raise StageCProvenanceError(
            "bindings.code.commit must be a lowercase 40-hex commit"
        )
    if type(code["dirty"]) is not bool:
        raise TypeError("bindings.code.dirty must be a boolean")
    if observed is not None:
        actual = _exact(
            observed, {"commit", "dirty"}, "observed_git_state"
        )
        if dict(actual) != dict(code):
            raise StageCProvenanceError(
                "recorded code state does not match observed Git state"
            )


def _validate_historical_model(value: Any) -> None:
    model = _exact(value, _HISTORICAL_MODEL_FIELDS, "bindings.model")
    if model["name"] != MODEL_NAME:
        raise StageCProvenanceError("bindings.model.name is not the frozen model")
    _relative_path(model["state_dict_path"], "bindings.model.state_dict_path")
    _sha(model["state_dict_sha256"], "bindings.model.state_dict_sha256")
    if model["state_dict_sha256"] != MODEL_SHA256:
        raise StageCProvenanceError(
            "bindings.model.state_dict_sha256 is not the frozen model hash"
        )


def _validate_recovery_model(value: Any) -> Mapping[str, Any]:
    model = _exact(value, _RECOVERY_MODEL_FIELDS, "bindings.model")
    if model["name"] != MODEL_NAME:
        raise StageCProvenanceError("bindings.model.name is not the frozen model")
    _relative_path(model["state_dict_path"], "bindings.model.state_dict_path")
    if model["identity"] != _RECOVERY_V2_MODEL_IDENTITY:
        raise StageCProvenanceError(
            "bindings.model.identity is not the recovery-v2 model identity "
            "required by the trusted recovery context"
        )
    for field_name in (
        "state_dict_sha256",
        "canonical_tensor_digest",
        "historical_state_dict_sha256",
        "recovery_binding_file_sha256",
        "recovery_binding_canonical_digest",
        "recovery_launch_record_file_sha256",
        "recovery_launch_record_canonical_digest",
    ):
        _sha(model[field_name], f"bindings.model.{field_name}")
    for field_name in (
        "recovery_binding_path", "recovery_launch_record_path",
    ):
        _relative_path(model[field_name], f"bindings.model.{field_name}")
    if model["historical_artifact_recovered"] is not False:
        raise StageCProvenanceError(
            "bindings.model.historical_artifact_recovered must be false"
        )
    if model["historical_state_dict_sha256"] != MODEL_SHA256:
        raise StageCProvenanceError(
            "bindings.model.historical_state_dict_sha256 is not the frozen "
            "historical model hash"
        )
    if model["state_dict_sha256"] == MODEL_SHA256:
        raise StageCProvenanceError(
            "recovery state_dict_sha256 must not alias the historical model hash"
        )
    return model


def _sha256_path(path: Path, context: str) -> str:
    if not path.is_file():
        raise StageCProvenanceError(f"{context} path is not a readable file")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise StageCProvenanceError(f"cannot hash {context} path") from error
    return digest.hexdigest()


def _split_recovery_context(
    value: Any,
) -> tuple[Any, Any] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return value[0], value[1]
    if isinstance(value, Mapping):
        fields = {"recovery_binding", "recovery_launch_record"}
        if set(value) != fields:
            raise StageCProvenanceError(
                "trusted recovery context mapping must contain exactly the "
                "recovery binding and recovery launch record"
            )
        return value["recovery_binding"], value["recovery_launch_record"]
    return None


def _reload_recovery_authority(
    recovery_binding: Any, recovery_launch_record: Any,
) -> tuple[Any, Any]:
    """Reload the two recovery authorities instead of trusting dataclass fields."""

    # These imports are intentionally local: this module remains usable for the
    # historical three-field contract without importing recovery-only code.
    from .recovery_c7_launch import (
        REPO_ROOT as RECOVERY_REPO_ROOT,
        RecoveryC7LaunchRecord,
        load_recovery_c7_launch_record,
    )
    from .recovery_v2 import RecoveryV2Binding

    if type(recovery_binding) is not RecoveryV2Binding:
        raise StageCProvenanceError(
            "trusted recovery context requires a validated RecoveryV2Binding"
        )
    if type(recovery_launch_record) is not RecoveryC7LaunchRecord:
        raise StageCProvenanceError(
            "trusted recovery context requires a validated "
            "RecoveryC7LaunchRecord"
        )
    if type(recovery_launch_record.binding) is not RecoveryV2Binding:
        raise StageCProvenanceError(
            "trusted recovery launch record lacks its RecoveryV2Binding"
        )
    if recovery_launch_record.binding != recovery_binding:
        raise StageCProvenanceError(
            "trusted recovery launch record is bound to a different "
            "RecoveryV2Binding"
        )

    authority_root = RECOVERY_REPO_ROOT.resolve(strict=True)
    if Path(recovery_launch_record.repo_root).resolve(strict=True) != authority_root:
        raise StageCProvenanceError(
            "trusted recovery context is outside the active recovery repository"
        )

    try:
        authoritative_launch = load_recovery_c7_launch_record(
            recovery_launch_record.path,
            repo_root=authority_root,
            schema_path=recovery_launch_record.schema_path,
            require_fresh_roots=False,
            # Recovery provenance is consumed only after the launch authority
            # has been committed.  A merely well-shaped public dataclass is not
            # authority.
            require_tracked=True,
        )
    except Exception as error:
        raise StageCProvenanceError(
            "trusted recovery binding and launch record could not be reloaded "
            "and validated"
        ) from error
    if type(authoritative_launch) is not RecoveryC7LaunchRecord:
        raise StageCProvenanceError(
            "recovery launch validation did not return trusted authority"
        )
    authoritative_binding = authoritative_launch.binding
    if type(authoritative_binding) is not RecoveryV2Binding:
        raise StageCProvenanceError(
            "reloaded recovery launch record lacks its RecoveryV2Binding"
        )
    if (
        recovery_binding != authoritative_binding
        or recovery_launch_record != authoritative_launch
    ):
        raise StageCProvenanceError(
            "trusted recovery context does not match reloaded authority"
        )
    return authoritative_binding, authoritative_launch


def _authority_relative_path(path: Any, repo_root: Path, context: str) -> str:
    try:
        relative = Path(path).resolve(strict=True).relative_to(repo_root)
    except (OSError, TypeError, ValueError) as error:
        raise StageCProvenanceError(
            f"trusted {context} must be a file under the recovery repository"
        ) from error
    return _relative_path(relative.as_posix(), context)


def normalize_recovery_model_context(
    recovery_binding: Any, recovery_launch_record: Any | None = None,
) -> dict[str, Any]:
    """Normalize trusted recovery inputs to the closed provenance shape."""

    if recovery_launch_record is None:
        # A genuine launch record includes its genuine binding; retain that
        # convenient single-object form without accepting arbitrary objects.
        from .recovery_c7_launch import RecoveryC7LaunchRecord

        if isinstance(recovery_binding, RecoveryC7LaunchRecord):
            recovery_launch_record = recovery_binding
            recovery_binding = recovery_launch_record.binding
        else:
            pair = _split_recovery_context(recovery_binding)
            if pair is None:
                raise StageCProvenanceError(
                    "trusted recovery context requires a RecoveryV2Binding and "
                    "RecoveryC7LaunchRecord"
                )
            recovery_binding, recovery_launch_record = pair

    binding, launch = _reload_recovery_authority(
        recovery_binding, recovery_launch_record,
    )
    repo_root = Path(launch.repo_root).resolve(strict=True)
    binding_path = _authority_relative_path(
        binding.path, repo_root, "recovery_binding_path",
    )
    launch_path = _authority_relative_path(
        launch.path, repo_root, "recovery_launch_record_path",
    )
    binding_file_sha256 = _sha256_path(binding.path, "recovery binding")
    launch_file_sha256 = _sha256_path(launch.path, "recovery launch record")
    binding_record = launch.document.get("recovery_binding")
    if (
        not isinstance(binding_record, Mapping)
        or binding_record.get("path") != binding_path
        or binding_record.get("file_sha256") != binding_file_sha256
        or binding_record.get("canonical_digest") != binding.canonical_digest
    ):
        raise StageCProvenanceError(
            "reloaded recovery launch record is bound to different binding bytes"
        )
    if launch.file_sha256 != launch_file_sha256:
        raise StageCProvenanceError(
            "reloaded recovery launch record SHA-256 disagrees with its bytes"
        )

    model_source = binding.model
    if not isinstance(model_source, Mapping) or not isinstance(
        launch.model_identity, Mapping
    ):
        raise TypeError(
            "reloaded recovery authority model identities must be mappings"
        )
    model_drift = [
        field_name for field_name in _RECOVERY_MODEL_IDENTITY_FIELDS
        if launch.model_identity.get(field_name) != model_source.get(field_name)
    ]
    if model_drift:
        raise StageCProvenanceError(
            "trusted recovery binding and launch model identities disagree: "
            f"{model_drift}"
        )

    result = {
        **{field_name: model_source.get(field_name)
           for field_name in _RECOVERY_MODEL_IDENTITY_FIELDS},
        "recovery_binding_path": binding_path,
        "recovery_binding_file_sha256": binding_file_sha256,
        "recovery_binding_canonical_digest": binding.canonical_digest,
        "recovery_launch_record_path": launch_path,
        "recovery_launch_record_file_sha256": launch_file_sha256,
        "recovery_launch_record_canonical_digest": launch.canonical_digest,
    }
    _validate_recovery_model(result)
    return result


def recovery_model_provenance(
    recovery_binding: Any, recovery_launch_record: Any,
) -> dict[str, Any]:
    """Build the recovery model block used by Stage C provenance."""

    return normalize_recovery_model_context(
        recovery_binding, recovery_launch_record,
    )


def validate_stage_c_model_binding(
    value: Any, *, trusted_recovery_context: Any | None = None,
    require_trusted_recovery_context: bool = False,
) -> None:
    """Validate the historical or recovery Stage C model identity."""

    if type(require_trusted_recovery_context) is not bool:
        raise TypeError("require_trusted_recovery_context must be a boolean")
    candidate = value.to_dict() if (
        not isinstance(value, Mapping) and callable(getattr(value, "to_dict", None))
    ) else value
    if not isinstance(candidate, Mapping):
        raise TypeError("bindings.model must be a mapping")
    if not (set(candidate) & _RECOVERY_MODEL_MARKERS):
        _validate_historical_model(candidate)
        if trusted_recovery_context is not None:
            raise StageCProvenanceError(
                "historical model provenance does not match the trusted "
                "recovery context"
            )
        return

    if trusted_recovery_context is None:
        model = _validate_recovery_model(candidate)
        raise StageCProvenanceError(
            "recovery model provenance requires a trusted recovery context"
        )
    expected = normalize_recovery_model_context(trusted_recovery_context)
    model = _validate_recovery_model(candidate)
    mismatches = [
        field_name for field_name in sorted(_RECOVERY_MODEL_FIELDS)
        if model[field_name] != expected[field_name]
    ]
    if mismatches:
        raise StageCProvenanceError(
            "bindings.model does not match the trusted recovery context: "
            f"{mismatches}"
        )


def _is_pre_target_failure(
    status: Mapping[str, Any], failure: Mapping[str, Any] | None,
) -> bool:
    return (
        status.get("state") == "failed"
        and isinstance(failure, Mapping)
        and failure.get("phase") in PRE_TARGET_FAILURE_PHASES
    )


def _validate_input(
    value: Any,
    unit: Mapping[str, Any],
    status: Mapping[str, Any],
    failure: Mapping[str, Any] | None,
    target: Mapping[str, Any],
) -> None:
    item = _exact(
        value,
        {
            "schema_version", "sample_id", "image_id", "relative_path",
            "image_sha256", "size_bytes", "width", "height", "official_wnid",
            "imagenet1k_index", "label_source", "role", "audit",
            "selection_order", "source_selection_index", "selection_rule",
            "canonical_rgb_sha256", "normalized_tensor_sha256",
        },
        "bindings.input",
    )
    if type(item["schema_version"]) is not int or item["schema_version"] != 1:
        raise StageCProvenanceError("bindings.input.schema_version must be 1")
    if item["sample_id"] != unit["sample_id"]:
        raise StageCProvenanceError(
            "bindings.input.sample_id disagrees with unit_identity"
        )
    if not isinstance(item["sample_id"], str) or not _SAMPLE_ID.fullmatch(
        item["sample_id"]
    ):
        raise StageCProvenanceError("bindings.input.sample_id is invalid")
    image_id = _nonempty(item["image_id"], "bindings.input.image_id")
    if item["relative_path"] != f"{image_id}.JPEG":
        raise StageCProvenanceError(
            "bindings.input image_id/relative_path disagree"
        )
    _relative_path(item["relative_path"], "bindings.input.relative_path")
    _sha(item["image_sha256"], "bindings.input.image_sha256")
    for name in ("size_bytes", "width", "height"):
        if _strict_int(item[name], f"bindings.input.{name}", 1) < 1:
            raise StageCProvenanceError(
                f"bindings.input.{name} must be positive"
            )
    if not isinstance(item["official_wnid"], str) or not re.fullmatch(
        r"n[0-9]{8}", item["official_wnid"]
    ):
        raise StageCProvenanceError("bindings.input.official_wnid is invalid")
    class_index = _strict_int(
        item["imagenet1k_index"], "bindings.input.imagenet1k_index"
    )
    if class_index >= 1000:
        raise StageCProvenanceError(
            "bindings.input.imagenet1k_index must be in [0,999]"
        )
    if item["label_source"] != "official_devkit_val_gt":
        raise StageCProvenanceError(
            "bindings.input.label_source must be official_devkit_val_gt"
        )
    if item["role"] not in {"calibration", "holdout"}:
        raise StageCProvenanceError(
            "bindings.input.role must be calibration or holdout"
        )
    if type(item["audit"]) is not bool:
        raise TypeError("bindings.input.audit must be a boolean")
    if item["audit"] and item["role"] != "calibration":
        raise StageCProvenanceError("audit must be a calibration subset")
    order = _strict_int(
        item["selection_order"], "bindings.input.selection_order"
    )
    if order >= 20:
        raise StageCProvenanceError(
            "bindings.input.selection_order must be in [0,19]"
        )
    source_index = _strict_int(
        item["source_selection_index"],
        "bindings.input.source_selection_index",
    )
    if source_index >= 500:
        raise StageCProvenanceError(
            "bindings.input.source_selection_index must be in [0,499]"
        )
    if item["selection_rule"] != (
        "legacy_sorted_500_linspace20_then_fixed_role_positions_0_4_9_14_19_"
        "audit_0_19"
    ):
        raise StageCProvenanceError("bindings.input.selection_rule is not frozen")
    unavailable = target.get("status") == "unavailable"
    if unavailable:
        if not _is_pre_target_failure(status, failure):
            raise StageCProvenanceError(
                "unavailable input tensors are legal only for a pre-target failure"
            )
        hashes = (
            item["canonical_rgb_sha256"], item["normalized_tensor_sha256"]
        )
        if (hashes[0] is None) != (hashes[1] is None):
            raise StageCProvenanceError(
                "pre-target input tensor hashes must both be null or both present"
            )
        if hashes[0] is not None:
            _sha(hashes[0], "bindings.input.canonical_rgb_sha256")
            _sha(hashes[1], "bindings.input.normalized_tensor_sha256")
    else:
        _sha(
            item["canonical_rgb_sha256"],
            "bindings.input.canonical_rgb_sha256",
        )
        _sha(
            item["normalized_tensor_sha256"],
            "bindings.input.normalized_tensor_sha256",
        )


def _validate_external_bindings(bindings: Mapping[str, Any]) -> None:
    dependency = _exact(
        bindings["dependencies"],
        {"lock_path", "lock_sha256"},
        "bindings.dependencies",
    )
    _relative_path(dependency["lock_path"], "bindings.dependencies.lock_path")
    _sha(dependency["lock_sha256"], "bindings.dependencies.lock_sha256")

    sections = {
        "registry": {
            "path", "sha256", "metadata_path", "metadata_sha256",
            "source_manifest_path", "source_manifest_sha256",
            "source_manifest_metadata_path", "source_manifest_metadata_sha256",
        },
        "protocol": {"path", "sha256", "version"},
        "acceptance": {"path", "sha256"},
        "configuration": {
            "path", "sha256", "schema_path", "schema_sha256",
            "registry_schema_path", "registry_schema_sha256",
            "selection_schema_path", "selection_schema_sha256",
            "run_spec_sha256", "execution_plan_sha256",
        },
    }
    for section_name, fields in sections.items():
        section = _exact(
            bindings[section_name], fields, f"bindings.{section_name}"
        )
        for field, raw in section.items():
            context = f"bindings.{section_name}.{field}"
            if field.endswith("path"):
                _relative_path(raw, context)
            elif field.endswith("sha256") or field == "sha256":
                _sha(raw, context)
            else:
                _nonempty(raw, context)
    if bindings["protocol"]["version"] != "1.0.0":
        raise StageCProvenanceError("bindings.protocol.version must be 1.0.0")
    if (
        bindings["resources"]["preflight"].get("plan_digest")
        != bindings["configuration"]["execution_plan_sha256"]
    ):
        raise StageCProvenanceError(
            "preflight plan_digest disagrees with execution_plan_sha256"
        )


def _candidate_ref(value: Any, context: str) -> Mapping[str, Any]:
    ref = _exact(
        value,
        {"candidate_key", "block_index", "head_index", "depth_band"},
        context,
    )
    block = _strict_int(ref["block_index"], f"{context}.block_index")
    head = _strict_int(ref["head_index"], f"{context}.head_index")
    band = _strict_int(ref["depth_band"], f"{context}.depth_band")
    if block >= 12 or head >= 12 or band >= 4:
        raise StageCProvenanceError(f"{context} has an out-of-range field")
    if ref["candidate_key"] != f"block_{block:02d}_head_{head:02d}":
        raise StageCProvenanceError(f"{context}.candidate_key disagrees")
    if band != block // 3:
        raise StageCProvenanceError(f"{context}.depth_band disagrees")
    return ref


def _validate_selection(value: Any) -> None:
    selection = _exact(
        value,
        {
            "schema_version", "status", "path", "sha256", "primary",
            "depth_band_winners", "reason",
        },
        "bindings.selection",
    )
    if (
        type(selection["schema_version"]) is not int
        or selection["schema_version"] != 1
    ):
        raise StageCProvenanceError(
            "bindings.selection.schema_version must be 1"
        )
    _relative_path(selection["path"], "bindings.selection.path")
    if selection["status"] == SELECTION_NOT_APPLICABLE:
        if selection["sha256"] is not None or selection["primary"] is not None:
            raise StageCProvenanceError(
                "pre-C8 selection cannot contain a hash or primary"
            )
        if selection["depth_band_winners"] != []:
            raise StageCProvenanceError(
                "pre-C8 selection cannot contain depth-band winners"
            )
        _nonempty(selection["reason"], "bindings.selection.reason")
        return
    if selection["status"] != SELECTION_PRESENT:
        raise StageCProvenanceError(
            "selection status must be not_applicable_pre_c8 or present"
        )
    if selection["reason"] is not None:
        raise StageCProvenanceError("present selection must have reason=null")
    _sha(selection["sha256"], "bindings.selection.sha256")
    if selection["sha256"] == "0" * 64:
        raise StageCProvenanceError(
            "present selection cannot use a fabricated all-zero hash"
        )
    primary = _candidate_ref(
        selection["primary"], "bindings.selection.primary"
    )
    winners = selection["depth_band_winners"]
    if not isinstance(winners, list) or len(winners) != 4:
        raise StageCProvenanceError(
            "present selection requires four depth-band winners"
        )
    checked = [
        _candidate_ref(item, f"bindings.selection.depth_band_winners[{index}]")
        for index, item in enumerate(winners)
    ]
    if [item["depth_band"] for item in checked] != list(range(4)):
        raise StageCProvenanceError(
            "selection winners must be ordered by depth bands 0..3"
        )
    if dict(primary) not in [dict(item) for item in checked]:
        raise StageCProvenanceError(
            "selection primary must be one of the four winners"
        )


def _validate_selection_scope(
    selection: Mapping[str, Any], input_record: Mapping[str, Any],
    unit: Mapping[str, Any],
) -> None:
    if (
        input_record["role"] == "holdout"
        and selection["status"] != SELECTION_PRESENT
    ):
        raise StageCProvenanceError(
            "holdout provenance requires a present C8 selection"
        )
    if (
        input_record["role"] == "holdout"
        and unit["method_id"] in _PER_HEAD_METHODS
    ):
        selected = {
            (item["block_index"], item["head_index"])
            for item in selection["depth_band_winners"]
        }
        if (unit["block_index"], unit["head_index"]) not in selected:
            raise StageCProvenanceError(
                "holdout Value-path unit is outside the selected band winners"
            )


def _validate_method_target_map(
    bindings: Mapping[str, Any], unit: Mapping[str, Any],
    status: Mapping[str, Any], failure: Mapping[str, Any] | None,
) -> None:
    method = _exact(bindings["method"], _METHOD_FIELDS, "bindings.method")
    method_id = method["id"]
    if method_id != unit["method_id"] or method_id not in _METHODS:
        raise StageCProvenanceError(
            "bindings.method.id disagrees with unit_identity or is unregistered"
        )
    if (
        not isinstance(method["version"], str)
        or not _SEMVER.fullmatch(method["version"])
        or method["version"] != "1.0.0"
    ):
        raise StageCProvenanceError(
            "bindings.method.version must be an explicit semantic version"
        )
    registered = _METHODS[method_id]
    actual = (
        method["exact_name"],
        method["comparison_line"],
        method["explanation_target_ref"],
        method["fidelity_target_ref"],
        method["class_agnostic"],
        method["fidelity_eligible"],
        method["structural_reference_only"],
        method["native_map"],
    )
    if actual != registered[:8]:
        raise StageCProvenanceError(
            "bindings.method violates the registered method contract"
        )
    for field in (
        "class_agnostic", "fidelity_eligible", "structural_reference_only"
    ):
        if type(method[field]) is not bool:
            raise TypeError(f"bindings.method.{field} must be a boolean")
    map_record = _exact(
        bindings["map"],
        {
            "projection", "sign_semantics", "normalization",
            "metric_ready_dtype", "metric_ready_shape",
        },
        "bindings.map",
    )
    if (
        map_record["projection"] != registered[8]
        or map_record["sign_semantics"] != registered[9]
    ):
        raise StageCProvenanceError(
            "bindings.map projection/sign semantics disagree with the method"
        )
    if map_record["normalization"] != registered[10]:
        raise StageCProvenanceError(
            "bindings.map normalization disagrees with the method"
        )
    if (
        map_record["metric_ready_dtype"] != "float32"
        or map_record["metric_ready_shape"] != [14, 14]
    ):
        raise StageCProvenanceError(
            "bindings.map must describe a float32 14x14 metric-ready map"
        )

    target = _exact(
        bindings["target"],
        {
            "status", "reason", "comparison_line", "target_ref",
            "clean_input_normalized_sha256", "identity_sha256", "identity",
        },
        "bindings.target",
    )
    if target["comparison_line"] != method["comparison_line"]:
        raise StageCProvenanceError(
            "method and target comparison lines disagree"
        )
    if (
        target["target_ref"] != method["fidelity_target_ref"]
        or target["target_ref"] != unit["target_ref"]
    ):
        raise StageCProvenanceError(
            "target reference disagrees with method or unit identity"
        )
    if target["status"] == "unavailable":
        if status.get("state") != "failed" or not isinstance(failure, Mapping):
            raise StageCProvenanceError(
                "unavailable target is legal only for a failed execution"
            )
        if failure.get("phase") not in PRE_TARGET_FAILURE_PHASES:
            raise StageCProvenanceError(
                "unavailable target failure phase is not pre-target"
            )
        _nonempty(target["reason"], "bindings.target.reason")
        if any(
            target[key] is not None
            for key in (
                "clean_input_normalized_sha256", "identity_sha256", "identity"
            )
        ):
            raise StageCProvenanceError(
                "unavailable target cannot contain fabricated identity hashes"
            )
        return
    if target["status"] != "resolved" or target["reason"] is not None:
        raise StageCProvenanceError(
            "target status must be resolved with null reason or unavailable"
        )
    clean_hash = _sha(
        target["clean_input_normalized_sha256"],
        "bindings.target.clean_input_normalized_sha256",
    )
    if clean_hash != bindings["input"]["normalized_tensor_sha256"]:
        raise StageCProvenanceError(
            "target clean hash disagrees with the normalized input hash"
        )
    if target["target_ref"] is None:
        if target["identity"] is not None or target["identity_sha256"] is not None:
            raise StageCProvenanceError(
                "structural-only method must not invent a scalar target identity"
            )
        return
    identity = target["identity"]
    if not isinstance(identity, Mapping) or not identity:
        raise StageCProvenanceError(
            "bindings.target.identity must be a non-empty mapping"
        )
    _sha(target["identity_sha256"], "bindings.target.identity_sha256")
    if canonical_sha256(identity) != target["identity_sha256"]:
        raise StageCProvenanceError(
            "bindings.target.identity_sha256 does not match identity"
        )
    if (
        identity.get("comparison_line") != target["comparison_line"]
        or identity.get("target_ref") != target["target_ref"]
    ):
        raise StageCProvenanceError(
            "target identity crosses comparison line or target reference"
        )
    if identity.get("clean_input_sha256") != clean_hash:
        raise StageCProvenanceError(
            "target identity clean hash uses the wrong hash domain"
        )
    frozen = identity.get("frozen_target")
    if (
        not isinstance(frozen, Mapping)
        or frozen.get("clean_input_sha256") != clean_hash
    ):
        raise StageCProvenanceError(
            "frozen target clean hash uses the wrong hash domain"
        )
    if method["comparison_line"] == "value_path":
        if frozen.get("target_kind") != (
            "fixed_clean_attention_cls_head_direction"
        ):
            raise StageCProvenanceError(
                "Value-path unit requires the fixed head-direction target"
            )
        candidate = frozen.get("candidate")
        if (
            not isinstance(candidate, Mapping)
            or set(candidate) != {"block_index", "head_index"}
            or candidate["block_index"] != unit["block_index"]
            or candidate["head_index"] != unit["head_index"]
        ):
            raise StageCProvenanceError(
                "Value-path target candidate disagrees with unit identity"
            )
    elif frozen.get("target_kind") != "raw_logit":
        raise StageCProvenanceError(
            "class-logit unit requires a raw_logit target"
        )


def _validate_baseline_schedule_seeds(bindings: Mapping[str, Any]) -> None:
    baseline = _exact(
        bindings["baseline"],
        {
            "id", "normalized_value", "canonical_rgb_value", "channels",
            "normalized_tensor_sha256",
        },
        "bindings.baseline",
    )
    if (
        baseline["id"] != "zero_normalized"
        or type(baseline["normalized_value"]) is not float
        or baseline["normalized_value"] != 0.0
        or type(baseline["canonical_rgb_value"]) is not float
        or baseline["canonical_rgb_value"] != 0.5
        or type(baseline["channels"]) is not int
        or baseline["channels"] != 3
    ):
        raise StageCProvenanceError(
            "bindings.baseline is not the frozen normalized-zero baseline"
        )
    _sha(
        baseline["normalized_tensor_sha256"],
        "bindings.baseline.normalized_tensor_sha256",
    )
    schedule = _exact(
        bindings["ig_schedule"],
        {"quadrature", "steps_schedule", "step_chunk_size", "atol", "rtol"},
        "bindings.ig_schedule",
    )
    if dict(schedule) != {
        "quadrature": "gauss_legendre",
        "steps_schedule": [512, 1024, 2048, 4096, 8192],
        "step_chunk_size": 8,
        "atol": 0.0001,
        "rtol": 0.01,
    }:
        raise StageCProvenanceError(
            "bindings.ig_schedule differs from the frozen schedule"
        )
    seeds = _exact(
        bindings["seeds"],
        {
            "base_seed", "noise_seed", "noise_seed_rule",
            "fidelity_enabled", "fidelity_seed_namespace",
            "fidelity_pairing_context",
            "fidelity_seeds",
        },
        "bindings.seeds",
    )
    if seeds["base_seed"] != 20260821:
        raise StageCProvenanceError("bindings.seeds.base_seed is not frozen")
    expected_noise = 20260821 + bindings["input"]["selection_order"]
    if (
        seeds["noise_seed"] != expected_noise
        or seeds["noise_seed_rule"]
        != "base_seed_plus_registry_selection_order"
    ):
        raise StageCProvenanceError(
            "noise seed violates the registry-selection-order rule"
        )
    if seeds["fidelity_seed_namespace"] != (
        "stage-c-fidelity|20260821|sample_id|block|head|replicate"
    ):
        raise StageCProvenanceError("fidelity seed namespace is not frozen")
    fidelity_seeds = seeds["fidelity_seeds"]
    if not isinstance(fidelity_seeds, list):
        raise TypeError("bindings.seeds.fidelity_seeds must be a list")
    for index, seed in enumerate(fidelity_seeds):
        _strict_int(seed, f"bindings.seeds.fidelity_seeds[{index}]")
        if seed >= 2**64:
            raise StageCProvenanceError(
                "fidelity seeds must be unsigned 64-bit integers"
            )
    if len(fidelity_seeds) != len(set(fidelity_seeds)):
        raise StageCProvenanceError("fidelity seeds must be unique")
    if type(seeds["fidelity_enabled"]) is not bool:
        raise TypeError("bindings.seeds.fidelity_enabled must be boolean")
    method_id = bindings["method"]["id"]
    pairing = seeds["fidelity_pairing_context"]
    target_resolved = bindings["target"]["status"] == "resolved"
    if seeds["fidelity_enabled"] and not target_resolved:
        raise StageCProvenanceError(
            "fidelity cannot be enabled before target resolution"
        )
    if seeds["fidelity_enabled"]:
        if bindings["method"]["fidelity_eligible"] is not True:
            raise StageCProvenanceError(
                "non-fidelity method cannot enable fidelity seeds"
            )
        pair = _exact(
            pairing, {"block_index", "head_index"},
            "bindings.seeds.fidelity_pairing_context",
        )
        block = _strict_int(pair["block_index"], "fidelity pairing block_index")
        head = _strict_int(pair["head_index"], "fidelity pairing head_index")
        if block >= 12 or head >= 12:
            raise StageCProvenanceError("fidelity pairing indices must be in [0,11]")
        if method_id in _PER_HEAD_METHODS and (
            block != bindings["target"]["identity"]["frozen_target"]["candidate"]["block_index"]
            or head != bindings["target"]["identity"]["frozen_target"]["candidate"]["head_index"]
        ):
            raise StageCProvenanceError(
                "per-head fidelity pairing must match the unit candidate"
            )
        expected = [
            _fidelity_seed(bindings["input"]["sample_id"], block, head, repeat)
            for repeat in range(5)
        ]
        if fidelity_seeds != expected:
            raise StageCProvenanceError(
                "fidelity seeds violate the registered namespace/pairing context"
            )
    elif pairing is not None or fidelity_seeds:
        raise StageCProvenanceError(
            "disabled fidelity must not claim pairing or seeds"
        )


def _fidelity_seed(sample_id: str, block: int, head: int, repeat: int) -> int:
    namespace = (
        f"stage-c-fidelity|20260821|{sample_id}|{block}|{head}|{repeat}"
    )
    return int.from_bytes(
        hashlib.sha256(namespace.encode("utf-8")).digest()[:8], "big"
    )


def _validate_domain_hash(
    value: Any, domain: str, context: str, *, allow_null: bool = False,
) -> str | None:
    item = _exact(value, {"domain", "sha256"}, context)
    if item["domain"] != domain:
        raise StageCProvenanceError(
            f"{context} has the wrong hash domain (possible domain swap)"
        )
    if allow_null and item["sha256"] is None:
        return None
    return _sha(item["sha256"], f"{context}.sha256")


def _validate_perturbations(
    bindings: Mapping[str, Any], status: Mapping[str, Any],
    failure: Mapping[str, Any] | None,
) -> None:
    value = _exact(
        bindings["perturbations"],
        {"enabled", "application_domain", "normalized_domain", "clean", "items"},
        "bindings.perturbations",
    )
    if (
        value["application_domain"] != CANONICAL_RGB_HASH_DOMAIN
        or value["normalized_domain"] != NORMALIZED_INPUT_HASH_DOMAIN
    ):
        raise StageCProvenanceError(
            "perturbation hash domains are not the frozen domains"
        )
    clean = _exact(
        value["clean"],
        {"id", "canonical_rgb", "normalized_input"},
        "bindings.perturbations.clean",
    )
    if clean["id"] != "clean":
        raise StageCProvenanceError("clean perturbation id must be 'clean'")
    unavailable = (
        bindings["target"].get("status") == "unavailable"
        and _is_pre_target_failure(status, failure)
    )
    clean_canonical = _validate_domain_hash(
        clean["canonical_rgb"], CANONICAL_RGB_HASH_DOMAIN,
        "bindings.perturbations.clean.canonical_rgb", allow_null=unavailable,
    )
    clean_normalized = _validate_domain_hash(
        clean["normalized_input"], NORMALIZED_INPUT_HASH_DOMAIN,
        "bindings.perturbations.clean.normalized_input", allow_null=unavailable,
    )
    if clean_canonical != bindings["input"]["canonical_rgb_sha256"]:
        raise StageCProvenanceError(
            "clean canonical RGB hash disagrees with input canonical hash"
        )
    if clean_normalized != bindings["input"]["normalized_tensor_sha256"]:
        raise StageCProvenanceError(
            "clean normalized hash disagrees with input normalized hash"
        )
    if unavailable and (value["enabled"] is not False or value["items"] != []):
        raise StageCProvenanceError(
            "pre-target unavailable inputs require disabled perturbations with no items"
        )
    items = value["items"]
    if type(value["enabled"]) is not bool or not isinstance(items, list):
        raise TypeError("perturbations.enabled/items have invalid types")
    expected_count = 3 if value["enabled"] else 0
    if len(items) != expected_count:
        raise StageCProvenanceError(
            "enabled perturbations require exactly three records; disabled requires none"
        )
    for index, raw in enumerate(items):
        context = f"bindings.perturbations.items[{index}]"
        item = _exact(
            raw,
            {
                "id", "canonical_rgb", "normalized_input", "parameters",
                "shared_exact_tensor_across_methods",
            },
            context,
        )
        if item["id"] != _PERTURBATION_IDS[index]:
            raise StageCProvenanceError(
                "perturbations must appear in frozen order"
            )
        _validate_domain_hash(
            item["canonical_rgb"], CANONICAL_RGB_HASH_DOMAIN,
            f"{context}.canonical_rgb",
        )
        _validate_domain_hash(
            item["normalized_input"], NORMALIZED_INPUT_HASH_DOMAIN,
            f"{context}.normalized_input",
        )
        parameters = _exact(
            item["parameters"], _PERTURBATION_PARAMETER_FIELDS[item["id"]],
            f"{context}.parameters",
        )
        if item["id"] == "translate_right_down_4" and dict(parameters) != (
            _TRANSLATION_PARAMETERS
        ):
            raise StageCProvenanceError(
                "translation parameters differ from the frozen operator"
            )
        if item["id"] == "center_crop_216_resize_224" and dict(parameters) != (
            _SCALE_PARAMETERS
        ):
            raise StageCProvenanceError(
                "crop-scale parameters differ from the frozen operator"
            )
        if item["shared_exact_tensor_across_methods"] is not True:
            raise StageCProvenanceError(
                "perturbed tensors must be shared exactly across methods"
            )
        if item["id"] == "gaussian_rgb_std_0_01":
            if parameters["seed"] != bindings["seeds"]["noise_seed"]:
                raise StageCProvenanceError(
                    "noise perturbation seed disagrees with seeds record"
                )
            expected_noise = {
                "kind": "gaussian_noise", "mean": 0.0,
                "std_rgb_0_to_1": 0.01, "generator_device": "cpu",
                "generator_dtype": "float32", "base_seed": 20260821,
                "sample_seed_rule": "base_seed_plus_registry_selection_order",
                "seed": bindings["seeds"]["noise_seed"],
                "clamp_min": 0.0, "clamp_max": 1.0,
                "inverse_map": "identity",
                "valid_pixel_box_half_open": [0, 224, 0, 224],
                "fully_valid_patch_rows": list(range(14)),
                "fully_valid_patch_cols": list(range(14)),
            }
            if dict(parameters) != expected_noise:
                raise StageCProvenanceError(
                    "noise parameters differ from the frozen operator"
                )


def _validate_status_failure(status_value: Any, failure: Any) -> None:
    status = _exact(status_value, {"state", "terminal"}, "bindings.status")
    if status["state"] not in {"completed", "failed"}:
        raise StageCProvenanceError(
            "bindings.status.state must be completed or failed"
        )
    if status["terminal"] is not True:
        raise StageCProvenanceError(
            "provenance must describe an explicit terminal state"
        )
    if status["state"] == "completed":
        if failure is not None:
            raise StageCProvenanceError(
                "completed provenance must have failure=null"
            )
        return
    record = _exact(
        failure,
        {"reason", "phase", "type", "message"},
        "bindings.failure",
    )
    for key, item in record.items():
        _nonempty(item, f"bindings.failure.{key}")


def _validate_cuda(value: Any) -> None:
    cuda = _exact(
        value,
        {"status", "device", "peak_allocated_bytes", "peak_reserved_bytes"},
        "bindings.resources.actual.cuda",
    )
    if cuda["status"] == "not_applicable_cpu":
        if any(
            cuda[key] is not None
            for key in ("device", "peak_allocated_bytes", "peak_reserved_bytes")
        ):
            raise StageCProvenanceError(
                "CPU CUDA values must be null, not fabricated zeros"
            )
        return
    if cuda["status"] not in {"measured", "measurement_failed"}:
        raise StageCProvenanceError("invalid CUDA measurement status")
    _nonempty(cuda["device"], "bindings.resources.actual.cuda.device")
    for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
        if cuda[key] is None and cuda["status"] == "measured":
            raise StageCProvenanceError("measured CUDA peaks cannot be null")
        if cuda[key] is not None:
            _strict_int(cuda[key], f"bindings.resources.actual.cuda.{key}")


def _validate_resources(
    value: Any, status: Mapping[str, Any], failure: Mapping[str, Any] | None,
    expected_plan_digest: str,
) -> None:
    resources = _exact(value, {"preflight", "actual"}, "bindings.resources")
    preflight = _exact(
        resources["preflight"],
        {
            "schema_version", "document_type", "plan_digest", "counts",
            "storage", "compute", "caps", "violations", "within_budget",
            "canonical_digest",
        },
        "bindings.resources.preflight",
    )
    if (
        type(preflight["schema_version"]) is not int
        or preflight["schema_version"] != 1
        or preflight["document_type"] != "stage_c_preflight_estimate"
    ):
        raise StageCProvenanceError("preflight identity fields are invalid")
    _sha(preflight["plan_digest"], "bindings.resources.preflight.plan_digest")
    _sha(
        preflight["canonical_digest"],
        "bindings.resources.preflight.canonical_digest",
    )
    preflight_payload = {
        key: preflight[key] for key in preflight if key != "canonical_digest"
    }
    if canonical_sha256(preflight_payload) != preflight["canonical_digest"]:
        raise StageCProvenanceError("preflight canonical_digest mismatch")
    if preflight["plan_digest"] != expected_plan_digest:
        raise StageCProvenanceError(
            "preflight plan_digest disagrees with execution plan binding"
        )
    if preflight["within_budget"] is not True or preflight["violations"] != []:
        raise StageCProvenanceError(
            "provenance cannot claim an execution whose preflight did not pass"
        )
    for name in ("counts", "storage", "compute", "caps"):
        if not isinstance(preflight[name], Mapping):
            raise TypeError(f"bindings.resources.preflight.{name} must be a mapping")
    actual = _exact(
        resources["actual"],
        {
            "status", "wall_time_seconds", "cuda", "calls",
            "artifact_bytes", "artifact_bytes_scope", "artifact_root", "files",
            "exception", "instrumentation_errors",
        },
        "bindings.resources.actual",
    )
    expected = "completed" if status["state"] == "completed" else "failed"
    if actual["status"] != expected:
        raise StageCProvenanceError(
            "resource status disagrees with execution status"
        )
    wall = actual["wall_time_seconds"]
    if wall is not None and (
        type(wall) not in {int, float}
        or not math.isfinite(float(wall))
        or wall < 0
    ):
        raise StageCProvenanceError(
            "resource wall_time_seconds must be finite and non-negative or null"
        )
    _validate_cuda(actual["cuda"])
    calls = _exact(
        actual["calls"],
        {"accounting_unit", "forward_calls", "backward_calls"},
        "bindings.resources.actual.calls",
    )
    if calls["accounting_unit"] != "batched_model_invocation":
        raise StageCProvenanceError("resource call accounting unit is invalid")
    _strict_int(calls["forward_calls"], "resources.actual.calls.forward_calls")
    _strict_int(calls["backward_calls"], "resources.actual.calls.backward_calls")
    total = _strict_int(
        actual["artifact_bytes"], "bindings.resources.actual.artifact_bytes"
    )
    scope = actual["artifact_bytes_scope"]
    if scope not in {"explicit_payload_file_list", "exhaustive_regular_file_tree"}:
        raise StageCProvenanceError("invalid artifact byte measurement scope")
    if scope == "explicit_payload_file_list":
        if actual["artifact_root"] is not None:
            raise StageCProvenanceError(
                "explicit artifact measurement cannot name a root"
            )
    else:
        _nonempty(actual["artifact_root"], "resources.actual.artifact_root")
    _validate_measured_files(actual["files"], total)
    errors = actual["instrumentation_errors"]
    if not isinstance(errors, list):
        raise TypeError("resource instrumentation_errors must be a list")
    if status["state"] == "completed":
        if actual["exception"] is not None or errors:
            raise StageCProvenanceError(
                "completed resources cannot retain execution errors"
            )
    elif actual["exception"] is None and not errors:
        raise StageCProvenanceError(
            "failed resources must retain an exception or instrumentation error"
        )
    elif actual["exception"] is not None:
        exception = _exact(
            actual["exception"], {"type", "message"},
            "bindings.resources.actual.exception",
        )
        _nonempty(exception["type"], "resources.actual.exception.type")
        _nonempty(exception["message"], "resources.actual.exception.message")
        if isinstance(failure, Mapping) and (
            exception["type"] != failure["type"]
            or exception["message"] != failure["message"]
        ):
            raise StageCProvenanceError(
                "resource exception disagrees with the retained failure"
            )


def _validate_measured_files(value: Any, expected_total: int) -> None:
    if not isinstance(value, list):
        raise TypeError("bindings.resources.actual.files must be a list")
    measured = 0
    paths: set[str] = set()
    for index, raw in enumerate(value):
        item = _exact(raw, {"path", "bytes"}, f"resources.actual.files[{index}]")
        path = _nonempty(item["path"], f"resources.actual.files[{index}].path")
        if path in paths:
            raise StageCProvenanceError(
                "resource measurement contains duplicate file paths"
            )
        paths.add(path)
        measured += _strict_int(
            item["bytes"], f"resources.actual.files[{index}].bytes"
        )
    if measured != expected_total:
        raise StageCProvenanceError(
            "resource artifact_bytes does not equal the explicit file sum"
        )


def _validate_retention(value: Any, input_record: Mapping[str, Any]) -> None:
    retention = _exact(
        value,
        {
            "audit", "display_is_never_metric_input", "l1_storage_dtype",
            "l2_audit_only", "l3_display_only", "forbidden_granularities",
        },
        "bindings.retention",
    )
    if retention["audit"] != input_record["audit"]:
        raise StageCProvenanceError(
            "retention audit flag disagrees with the registry record"
        )
    if (
        retention["display_is_never_metric_input"] is not True
        or retention["l1_storage_dtype"] != "float32"
        or retention["l2_audit_only"] is not True
        or retention["l3_display_only"] is not True
    ):
        raise StageCProvenanceError(
            "retention record violates the frozen L1/L2/L3 policy"
        )
    forbidden = retention["forbidden_granularities"]
    if not isinstance(forbidden, list) or set(forbidden) != {
        "per_quadrature_step_files", "per_pixel_files",
        "per_dimension_files", "display_png_as_metric_input",
    }:
        raise StageCProvenanceError(
            "retention forbidden_granularities are incomplete"
        )


def _validate_inventory(value: Any) -> None:
    if not isinstance(value, list):
        raise TypeError("bindings.artifact_inventory must be a list")
    if not value:
        raise StageCProvenanceError(
            "terminal provenance requires a non-empty artifact inventory"
        )
    paths: set[str] = set()
    base_fields = {
        "level", "path", "sha256", "bytes", "kind", "dtype", "shape",
        "format",
    }
    for index, raw in enumerate(value):
        context = f"bindings.artifact_inventory[{index}]"
        if not isinstance(raw, Mapping):
            raise TypeError(f"{context} must be a mapping")
        level = raw.get("level")
        fields = set(base_fields)
        if level == "L3":
            fields |= {"display_only", "metric_input", "failure"}
        item = _exact(raw, fields, context)
        if level not in {"L0", "L1", "L2", "L3"}:
            raise StageCProvenanceError(f"{context}.level is invalid")
        path = _relative_path(item["path"], f"{context}.path")
        if path in paths:
            raise StageCProvenanceError(
                "artifact inventory contains duplicate paths"
            )
        paths.add(path)
        if (
            path in {"manifest.json", "status.json", "provenance.json"}
            or path.endswith("/provenance.json")
        ):
            raise StageCProvenanceError(
                "provenance inventory must exclude itself and completion controls"
            )
        _sha(item["sha256"], f"{context}.sha256")
        _strict_int(item["bytes"], f"{context}.bytes")
        _nonempty(item["kind"], f"{context}.kind")
        _nonempty(item["format"], f"{context}.format")
        if level == "L1" and (
            item["kind"] != "metric_ready"
            or item["dtype"] != "float32"
            or item["shape"] != [14, 14]
        ):
            raise StageCProvenanceError(
                "L1 entries must be float32 metric-ready 14x14 maps"
            )
        if level == "L3" and (
            item["kind"] != "display"
            or item["format"] != "png"
            or item["display_only"] is not True
            or item["metric_input"] is not False
            or type(item["failure"]) is not bool
        ):
            raise StageCProvenanceError(
                "L3 entries must be display-only and never metric input"
            )


def _reject_circular_hash_claims(value: Any, context: str = "bindings") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {
                "provenance_sha256", "manifest_sha256", "status_sha256"
            }:
                raise StageCProvenanceError(
                    f"{context}.{key} creates a circular provenance hash claim"
                )
            _reject_circular_hash_claims(item, f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_circular_hash_claims(item, f"{context}[{index}]")


def build_stage_c_provenance(
    *, unit_identity: Mapping[str, Any], code: Mapping[str, Any],
    model: Mapping[str, Any], input_record: Mapping[str, Any],
    dependencies: Mapping[str, Any], registry: Mapping[str, Any],
    protocol: Mapping[str, Any], acceptance: Mapping[str, Any],
    configuration: Mapping[str, Any], selection: Mapping[str, Any],
    method: Mapping[str, Any], target: Mapping[str, Any],
    map_record: Mapping[str, Any], baseline: Mapping[str, Any],
    ig_schedule: Mapping[str, Any], seeds: Mapping[str, Any],
    perturbations: Mapping[str, Any], status: Mapping[str, Any],
    failure: Mapping[str, Any] | None, resources: Mapping[str, Any],
    retention: Mapping[str, Any],
    artifact_inventory: Sequence[Mapping[str, Any]],
    trusted_recovery_context: Any | None = None,
    require_trusted_recovery_context: bool = False,
) -> StageCProvenance:
    """Build a validated document from runner-owned, JSON-safe records."""

    bindings = {
        "code": code,
        "model": model,
        "input": input_record,
        "dependencies": dependencies,
        "registry": registry,
        "protocol": protocol,
        "acceptance": acceptance,
        "configuration": configuration,
        "selection": selection,
        "method": method,
        "target": target,
        "map": map_record,
        "baseline": baseline,
        "ig_schedule": ig_schedule,
        "seeds": seeds,
        "perturbations": perturbations,
        "status": status,
        "failure": failure,
        "resources": resources,
        "retention": retention,
        "artifact_inventory": list(artifact_inventory),
    }
    return StageCProvenance(
        unit_identity=unit_identity, bindings=bindings,
        trusted_recovery_context=trusted_recovery_context,
        require_trusted_recovery_context=require_trusted_recovery_context,
    )


def validate_stage_c_provenance(
    value: StageCProvenance | Mapping[str, Any], *,
    expected_unit: Mapping[str, Any] | None = None,
    observed_git_state: Mapping[str, Any] | None = None,
    trusted_recovery_context: Any | None = None,
    require_trusted_recovery_context: bool = False,
) -> None:
    """Validate structure, hashes, domains, and cross-line relationships."""

    document = value.to_dict() if isinstance(value, StageCProvenance) else value
    if not isinstance(document, Mapping) and callable(
        getattr(document, "to_dict", None)
    ):
        document = document.to_dict()
    root = _exact(document, {
        "schema_version", "document_type", "unit_identity",
        "unit_identity_sha256", "bindings",
    }, "provenance")
    _json_safe(root)
    if (
        type(root["schema_version"]) is not int
        or root["schema_version"] != STAGE_C_PROVENANCE_SCHEMA_VERSION
    ):
        raise StageCProvenanceError("unexpected provenance schema_version")
    if root["document_type"] != STAGE_C_PROVENANCE_DOCUMENT_TYPE:
        raise StageCProvenanceError("unexpected provenance document_type")
    unit = _validate_unit_identity(root["unit_identity"])
    if root["unit_identity_sha256"] != unit["canonical_digest"]:
        raise StageCProvenanceError(
            "unit_identity_sha256 disagrees with canonical_digest"
        )
    if expected_unit is not None:
        expected = _validate_unit_identity(expected_unit)
        if _thaw(expected) != _thaw(unit):
            raise StageCProvenanceError(
                "provenance unit_identity does not match the expected unit"
            )
    bindings = _exact(root["bindings"], _BINDING_FIELDS, "bindings")
    _reject_circular_hash_claims(bindings)
    _validate_code(bindings["code"], observed_git_state)
    validate_stage_c_model_binding(
        bindings["model"],
        trusted_recovery_context=trusted_recovery_context,
        require_trusted_recovery_context=require_trusted_recovery_context,
    )
    _validate_input(
        bindings["input"], unit, bindings["status"], bindings["failure"],
        bindings["target"],
    )
    _validate_external_bindings(bindings)
    _validate_selection(bindings["selection"])
    _validate_selection_scope(
        bindings["selection"], bindings["input"], unit
    )
    _validate_method_target_map(
        bindings, unit, bindings["status"], bindings["failure"]
    )
    _validate_baseline_schedule_seeds(bindings)
    _validate_perturbations(
        bindings, bindings["status"], bindings["failure"]
    )
    _validate_status_failure(bindings["status"], bindings["failure"])
    _validate_resources(
        bindings["resources"], bindings["status"], bindings["failure"],
        bindings["configuration"]["execution_plan_sha256"],
    )
    _validate_retention(bindings["retention"], bindings["input"])
    _validate_inventory(bindings["artifact_inventory"])
    inventory = bindings["artifact_inventory"]
    inventory_pairs = {(item["path"], item["bytes"]) for item in inventory}
    resource_pairs = {
        (item["path"], item["bytes"])
        for item in bindings["resources"]["actual"]["files"]
    }
    if resource_pairs != inventory_pairs:
        raise StageCProvenanceError(
            "resource file measurements and artifact inventory disagree"
        )
    failure_records = [
        item for item in inventory
        if item["level"] == "L0" and item["kind"] == "failure"
    ]
    if bindings["status"]["state"] == "completed" and failure_records:
        raise StageCProvenanceError(
            "completed provenance cannot inventory a failure artifact"
        )
    if bindings["status"]["state"] == "failed" and len(failure_records) != 1:
        raise StageCProvenanceError(
            "failed provenance requires exactly one L0 failure artifact"
        )


__all__ = [
    "CANONICAL_RGB_HASH_DOMAIN",
    "NORMALIZED_INPUT_HASH_DOMAIN",
    "SELECTION_NOT_APPLICABLE",
    "SELECTION_PRESENT",
    "STAGE_C_PROVENANCE_DOCUMENT_TYPE",
    "STAGE_C_PROVENANCE_SCHEMA_VERSION",
    "StageCProvenance",
    "StageCProvenanceError",
    "build_stage_c_provenance",
    "build_unit_identity",
    "canonical_json_bytes",
    "canonical_sha256",
    "capture_git_state",
    "normalize_recovery_model_context",
    "recovery_model_provenance",
    "unit_identity_digest",
    "validate_stage_c_model_binding",
    "validate_stage_c_provenance",
]
