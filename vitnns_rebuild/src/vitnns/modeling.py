from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

import timm
import torch
from timm.data import create_transform, resolve_model_data_config
from timm.layers import set_fused_attn

from .constants import DEFAULT_STATE_DICT, MODEL_NAME, MODEL_SHA256
from .io_utils import sha256_file as _sha256_path

if TYPE_CHECKING:
    from .recovery_v2 import RecoveryV2Binding


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_RECOVERY_V2_MODEL_IDENTITY = "RECOVERY_V2_NEW_ARTIFACT_NOT_HISTORICAL_2E718"
_STATE_DICT_HASH_CHUNK_SIZE = 1024 * 1024


def sha256_file(
    source: Path | str | BinaryIO, chunk_size: int = _STATE_DICT_HASH_CHUNK_SIZE,
) -> str:
    """Hash a path or the already-open state-dict stream.

    Keeping this module-level seam preserves existing test monkeypatches while
    allowing verified model loads to hash and deserialize one open file.
    """

    if not hasattr(source, "read"):
        return _sha256_path(Path(source), chunk_size=chunk_size)
    digest = hashlib.sha256()
    for chunk in iter(lambda: source.read(chunk_size), b""):
        digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PreprocessSpec:
    input_size: tuple[int, int, int]
    resize_shorter_side: int
    center_crop: tuple[int, int]
    interpolation: str
    antialias: bool
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    crop_pct: float
    crop_mode: str
    color_mode: str = "RGB"

    def to_dict(self) -> dict:
        return asdict(self)


def canonical_data_config(model: torch.nn.Module) -> dict:
    return resolve_model_data_config(model)


def canonical_preprocess_spec(model: torch.nn.Module) -> PreprocessSpec:
    cfg = canonical_data_config(model)
    height, width = cfg["input_size"][-2:]
    resize = int(height / cfg["crop_pct"])
    return PreprocessSpec(
        input_size=tuple(cfg["input_size"]),
        resize_shorter_side=resize,
        center_crop=(height, width),
        interpolation=str(cfg["interpolation"]),
        antialias=True,
        mean=tuple(cfg["mean"]),
        std=tuple(cfg["std"]),
        crop_pct=float(cfg["crop_pct"]),
        crop_mode=str(cfg["crop_mode"]),
    )


def canonical_transform(model: torch.nn.Module):
    return create_transform(**canonical_data_config(model), is_training=False)


def _validate_explicit_model_identity(
    *,
    expected_state_dict_sha256: str | None,
    expected_model_name: str | None,
    expected_model_identity: str | None,
    expected_canonical_tensor_digest: str | None,
    allow_recovery_v2_identity: bool = False,
) -> None:
    if expected_state_dict_sha256 is not None and (
        not isinstance(expected_state_dict_sha256, str)
        or _SHA256_RE.fullmatch(expected_state_dict_sha256) is None
    ):
        raise ValueError("expected_state_dict_sha256 must be a lowercase SHA-256 digest")
    if expected_model_name is not None and (
        not isinstance(expected_model_name, str) or not expected_model_name.strip()
    ):
        raise ValueError("expected_model_name must be a non-empty string")
    if (
        (expected_model_identity is None)
        != (expected_canonical_tensor_digest is None)
    ):
        raise ValueError(
            "expected_model_identity and expected_canonical_tensor_digest "
            "must be provided together"
        )
    if expected_model_identity is not None:
        if (
            not isinstance(expected_model_identity, str)
            or not expected_model_identity.strip()
        ):
            raise ValueError("expected_model_identity must be a non-empty string")
        if (
            expected_model_identity == _RECOVERY_V2_MODEL_IDENTITY
            and not allow_recovery_v2_identity
        ):
            raise ValueError(
                "The recovery-v2 model identity requires a RecoveryV2Binding"
            )
    if expected_canonical_tensor_digest is not None:
        if (
            not isinstance(expected_canonical_tensor_digest, str)
            or _SHA256_RE.fullmatch(expected_canonical_tensor_digest) is None
        ):
            raise ValueError(
                "expected_canonical_tensor_digest must be a lowercase SHA-256 digest"
            )


def _resolve_model_identity(
    *,
    state_dict_path: Path | str | None,
    expected_state_dict_sha256: str | None,
    expected_model_name: str | None,
    expected_model_identity: str | None,
    expected_canonical_tensor_digest: str | None,
    recovery_binding: RecoveryV2Binding | None,
) -> tuple[Path, str, str, str | None, str | None]:
    _validate_explicit_model_identity(
        expected_state_dict_sha256=expected_state_dict_sha256,
        expected_model_name=expected_model_name,
        expected_model_identity=expected_model_identity,
        expected_canonical_tensor_digest=expected_canonical_tensor_digest,
        allow_recovery_v2_identity=recovery_binding is not None,
    )
    if recovery_binding is None:
        return (
            Path(
                DEFAULT_STATE_DICT if state_dict_path is None else state_dict_path
            ).resolve(),
            MODEL_SHA256
            if expected_state_dict_sha256 is None
            else expected_state_dict_sha256,
            MODEL_NAME if expected_model_name is None else expected_model_name,
            expected_model_identity,
            expected_canonical_tensor_digest,
        )

    # Import locally so ordinary historical loads do not acquire the recovery-v2
    # JSON-schema dependency at module import time.
    from .recovery_v2 import RecoveryV2Binding, load_recovery_v2_binding

    if not isinstance(recovery_binding, RecoveryV2Binding):
        raise TypeError("recovery_binding must be a RecoveryV2Binding")

    # RecoveryV2Binding is a public dataclass, so isinstance alone is not an
    # authenticity boundary.  Re-read its own binding and schema and require
    # live asset verification before consuming any claims from the object.
    revalidated = load_recovery_v2_binding(
        path=recovery_binding.path,
        schema_path=recovery_binding.schema_path,
        verify_assets=True,
    )
    conflicts = []
    if recovery_binding.document != revalidated.document:
        conflicts.append("document")
    for field in (
        "path",
        "schema_path",
        "model_path",
        "data_root",
        "devkit_path",
        "source_manifest_path",
    ):
        try:
            supplied_binding_path = Path(getattr(recovery_binding, field)).resolve()
        except (AttributeError, TypeError) as error:
            raise ValueError(
                f"recovery_binding.{field} must be a filesystem path"
            ) from error
        if supplied_binding_path != Path(getattr(revalidated, field)).resolve():
            conflicts.append(field)
    if conflicts:
        raise ValueError(
            "RecoveryV2Binding conflicts with its revalidated source: "
            + ", ".join(conflicts)
        )

    binding_model = revalidated.model
    binding_path = Path(revalidated.model_path).resolve()
    if not isinstance(binding_model, Mapping):
        raise TypeError("recovery_binding.model must be a mapping")
    try:
        binding_sha256 = binding_model["state_dict_sha256"]
        binding_name = binding_model["name"]
        binding_identity = binding_model["identity"]
        binding_tensor_digest = binding_model["canonical_tensor_digest"]
    except KeyError as error:
        raise ValueError(
            f"recovery_binding.model is missing required field: {error.args[0]}"
        ) from error
    _validate_explicit_model_identity(
        expected_state_dict_sha256=binding_sha256,
        expected_model_name=binding_name,
        expected_model_identity=binding_identity,
        expected_canonical_tensor_digest=binding_tensor_digest,
        allow_recovery_v2_identity=True,
    )
    if binding_identity != _RECOVERY_V2_MODEL_IDENTITY:
        raise ValueError(
            "recovery_binding.model.identity does not match the fixed "
            "recovery-v2 identity"
        )

    supplied_path = (
        binding_path if state_dict_path is None else Path(state_dict_path).resolve()
    )
    supplied = {
        "state_dict_path": supplied_path,
        "expected_state_dict_sha256": (
            binding_sha256
            if expected_state_dict_sha256 is None
            else expected_state_dict_sha256
        ),
        "expected_model_name": (
            binding_name if expected_model_name is None else expected_model_name
        ),
        "expected_model_identity": (
            binding_identity
            if expected_model_identity is None
            else expected_model_identity
        ),
        "expected_canonical_tensor_digest": (
            binding_tensor_digest
            if expected_canonical_tensor_digest is None
            else expected_canonical_tensor_digest
        ),
    }
    bound = {
        "state_dict_path": binding_path,
        "expected_state_dict_sha256": binding_sha256,
        "expected_model_name": binding_name,
        "expected_model_identity": binding_identity,
        "expected_canonical_tensor_digest": binding_tensor_digest,
    }
    conflicts = [key for key, value in supplied.items() if value != bound[key]]
    if conflicts:
        raise ValueError(
            "Explicit model identity conflicts with recovery_binding: "
            + ", ".join(conflicts)
        )
    return (
        binding_path, binding_sha256, binding_name, binding_identity,
        binding_tensor_digest,
    )


def _open_hash_and_load_state_dict(
    path: Path, expected_state_dict_sha256: str,
) -> Mapping:
    """Hash and deserialize one stable open file, rejecting live mutation."""

    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        actual_file_sha256 = sha256_file(handle)
        after_hash = os.fstat(handle.fileno())
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        try:
            named_after_hash = path.stat()
        except OSError as error:
            raise ValueError(
                "State-dict path changed while it was being hashed"
            ) from error
        if (
            any(
                getattr(before, field) != getattr(after_hash, field)
                for field in stable_fields
            )
            or (before.st_dev, before.st_ino)
            != (named_after_hash.st_dev, named_after_hash.st_ino)
        ):
            raise ValueError("State-dict file changed while it was being hashed")
        if actual_file_sha256 != expected_state_dict_sha256:
            raise ValueError(
                "State-dict SHA-256 mismatch: expected "
                f"{expected_state_dict_sha256}, got {actual_file_sha256}"
            )
        handle.seek(0)
        state = torch.load(handle, map_location="cpu", weights_only=True)
        after_load = os.fstat(handle.fileno())
        try:
            named_after_load = path.stat()
        except OSError as error:
            raise ValueError(
                "State-dict path changed while it was being loaded"
            ) from error
        if (
            any(
                getattr(before, field) != getattr(after_load, field)
                for field in stable_fields
            )
            or (before.st_dev, before.st_ino)
            != (named_after_load.st_dev, named_after_load.st_ino)
        ):
            raise ValueError("State-dict file changed while it was being loaded")
    return state


def load_canonical_model(
    state_dict_path: Path | str | None = None,
    *,
    device: str | torch.device = "cpu",
    verify_hash: bool = True,
    unfused_attention: bool = False,
    expected_state_dict_sha256: str | None = None,
    expected_model_name: str | None = None,
    expected_model_identity: str | None = None,
    expected_canonical_tensor_digest: str | None = None,
    recovery_binding: RecoveryV2Binding | None = None,
) -> torch.nn.Module:
    """Build a bound ViT-B model from a safe state dict.

    The defaults preserve the historical Stage A artifact identity.  A recovery
    binding supplies its path, architecture name, identity, file digest, and
    canonical tensor digest.  Explicit values may accompany a binding only when
    they agree.  ``verify_hash=False`` cannot disable file verification when a
    semantic identity is supplied.
    """
    force_file_verification = (
        expected_state_dict_sha256 is not None
        or expected_canonical_tensor_digest is not None
        or recovery_binding is not None
    )
    (
        path,
        expected_state_dict_sha256,
        expected_model_name,
        expected_model_identity,
        expected_canonical_tensor_digest,
    ) = _resolve_model_identity(
        state_dict_path=state_dict_path,
        expected_state_dict_sha256=expected_state_dict_sha256,
        expected_model_name=expected_model_name,
        expected_model_identity=expected_model_identity,
        expected_canonical_tensor_digest=expected_canonical_tensor_digest,
        recovery_binding=recovery_binding,
    )
    if not path.is_file():
        raise FileNotFoundError(f"State dict not found: {path}")
    if verify_hash or force_file_verification:
        state = _open_hash_and_load_state_dict(path, expected_state_dict_sha256)
    else:
        # Preserve the historical explicit opt-out, including path-based loading.
        state = torch.load(path, map_location="cpu", weights_only=True)
    if expected_canonical_tensor_digest is not None:
        # Local import avoids making the general model path depend on recovery-v2
        # JSON/schema machinery at module import time.
        from .recovery_v2 import canonical_state_dict_tensor_digest

        actual_tensor_digest = canonical_state_dict_tensor_digest(state)
        if actual_tensor_digest != expected_canonical_tensor_digest:
            raise ValueError(
                "Canonical state-dict tensor digest mismatch: expected "
                f"{expected_canonical_tensor_digest}, got {actual_tensor_digest}"
            )

    # This must happen before model construction; each Attention instance snapshots
    # the fused-attention setting in its constructor.
    set_fused_attn(not unfused_attention)
    model = timm.create_model(expected_model_name, pretrained=False)
    loaded = model.load_state_dict(state, strict=True)
    missing = tuple(getattr(loaded, "missing_keys", ()))
    unexpected = tuple(getattr(loaded, "unexpected_keys", ()))
    if missing or unexpected:
        raise RuntimeError(
            "Strict state-dict load reported "
            f"missing={list(missing)}, unexpected={list(unexpected)}"
        )
    model.eval()
    model.to(device)

    if unfused_attention:
        for block in model.blocks:
            block.attn.fused_attn = False
    _assert_model_contract(model)
    return model


def _assert_model_contract(model: torch.nn.Module) -> None:
    if len(model.blocks) != 12 or model.embed_dim != 768:
        raise RuntimeError("Unexpected ViT-B architecture")
    if tuple(model.pos_embed.shape) != (1, 197, 768):
        raise RuntimeError("Unexpected ViT-B token geometry")
    if any(block.attn.num_heads != 12 or block.attn.head_dim != 64 for block in model.blocks):
        raise RuntimeError("Unexpected attention geometry")
    if sum(parameter.numel() for parameter in model.parameters()) != 86_567_656:
        raise RuntimeError("Unexpected parameter count")

