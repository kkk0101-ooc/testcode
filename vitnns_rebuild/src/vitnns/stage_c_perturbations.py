"""Frozen Stage C input perturbations and inverse map geometry.

The operators in this module work in canonical, pre-normalization RGB space.
They intentionally do not run attribution methods or select targets.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .display import DisplayBoundaryError
from .retention import ArtifactArray, ArtifactKind, _deep_freeze, _immutable_array
from .stage_c_maps import raw_array
from .stage_c_class_logit import _tensor_sha256 as _canonical_tensor_sha256


CANONICAL_RGB_SHAPE = (1, 3, 224, 224)
PIXEL_SHAPE = (224, 224)
PATCH_SHAPE = (14, 14)
NOISE_BASE_SEED = 20260821
# Use the exact frozen JSON/protocol decimal (one binary64 ulp above Python's
# direct evaluation of the rational expression).
SCALE_INVERSE_MASS_JACOBIAN = 1.075445816186557
TRANSLATION_ID = "translate_right_down_4"
SCALE_ID = "center_crop_216_resize_224"
NOISE_ID = "gaussian_rgb_std_0_01"
PERTURBATION_IDS = (TRANSLATION_ID, SCALE_ID, NOISE_ID)
_PERTURBATION_ISSUER_SECRET = object()


class StageCPerturbationError(ValueError):
    """Raised when the frozen perturbation contract is violated."""


def tensor_sha256(value: torch.Tensor) -> str:
    """Use the same dtype/shape/bytes hash as frozen scalar evaluation."""

    return _canonical_tensor_sha256(value)


def _json_safe(value: Any, context: str = "metadata") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise StageCPerturbationError(f"{context} contains a non-finite number")
        return
    if isinstance(value, np.integer):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{context} keys must be strings")
            _json_safe(item, f"{context}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _json_safe(item, f"{context}[{index}]")
        return
    raise TypeError(f"{context} contains non-JSON-safe {type(value).__name__}")


def _canonical_rgb(value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError("canonical RGB input must be a torch.Tensor")
    if value.shape != CANONICAL_RGB_SHAPE:
        raise StageCPerturbationError(
            f"canonical RGB input must have shape {CANONICAL_RGB_SHAPE}"
        )
    if value.dtype is not torch.float32:
        raise TypeError("canonical RGB input must have dtype float32")
    if value.device.type != "cpu":
        raise StageCPerturbationError(
            "Stage C perturbations are generated once on CPU before normalization"
        )
    detached = value.detach()
    if not bool(torch.isfinite(detached).all().item()):
        raise StageCPerturbationError("canonical RGB input must be finite")
    if bool(((detached < 0.0) | (detached > 1.0)).any().item()):
        raise StageCPerturbationError(
            "canonical RGB input must be in pre-normalization [0,1] space"
        )
    return detached.clone(memory_format=torch.contiguous_format)


def _perturbed_rgb(value: Any) -> torch.Tensor:
    """Validate an owned perturbation tensor without reimposing input range.

    Bicubic interpolation can overshoot slightly; the frozen crop-scale
    operator does not prescribe clamping.
    """

    if not isinstance(value, torch.Tensor):
        raise TypeError("perturbation tensor must be a torch.Tensor")
    if value.shape != CANONICAL_RGB_SHAPE:
        raise StageCPerturbationError(
            f"perturbation tensor must have shape {CANONICAL_RGB_SHAPE}"
        )
    if value.dtype is not torch.float32:
        raise TypeError("perturbation tensor must have dtype float32")
    if value.device.type != "cpu":
        raise StageCPerturbationError("perturbation tensor must be generated on CPU")
    detached = value.detach()
    if not bool(torch.isfinite(detached).all().item()):
        raise StageCPerturbationError("perturbation tensor must be finite")
    return detached.clone(memory_format=torch.contiguous_format)


def _strict_selection_order(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise TypeError("selection_order must be a non-negative integer (not bool)")
    return value


def translate_right_down_4(clean_rgb: Any) -> torch.Tensor:
    clean = _canonical_rgb(clean_rgb)
    # Left/top reflection padding supplies the newly exposed area; cropping the
    # opposite edges moves every original pixel four positions right and down.
    return F.pad(clean, (4, 0, 4, 0), mode="reflect")[:, :, :224, :224]


def center_crop_216_resize_224(clean_rgb: Any) -> torch.Tensor:
    clean = _canonical_rgb(clean_rgb)
    return F.interpolate(
        clean[:, :, 4:220, 4:220],
        size=(224, 224),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )


def gaussian_rgb_std_0_01(clean_rgb: Any, *, selection_order: int) -> torch.Tensor:
    clean = _canonical_rgb(clean_rgb)
    order = _strict_selection_order(selection_order)
    generator = torch.Generator(device="cpu").manual_seed(NOISE_BASE_SEED + order)
    noise = torch.randn(
        clean.shape, generator=generator, device="cpu", dtype=torch.float32
    )
    return torch.clamp(clean + noise * 0.01, min=0.0, max=1.0)


def _pixel_mask(box: tuple[int, int, int, int]) -> np.ndarray:
    top, bottom, left, right = box
    mask = np.zeros(PIXEL_SHAPE, dtype=np.bool_)
    mask[top:bottom, left:right] = True
    return _immutable_array(mask)


def _patch_mask(indices: range) -> np.ndarray:
    mask = np.zeros(PATCH_SHAPE, dtype=np.bool_)
    selected = tuple(indices)
    mask[np.ix_(selected, selected)] = True
    return _immutable_array(mask)


def perturbation_masks(perturbation_id: str) -> tuple[np.ndarray, np.ndarray]:
    if perturbation_id == TRANSLATION_ID:
        return _pixel_mask((0, 220, 0, 220)), _patch_mask(range(13))
    if perturbation_id == SCALE_ID:
        return _pixel_mask((4, 220, 4, 220)), _patch_mask(range(1, 13))
    if perturbation_id == NOISE_ID:
        return _pixel_mask((0, 224, 0, 224)), _patch_mask(range(14))
    raise StageCPerturbationError(f"unknown Stage C perturbation: {perturbation_id!r}")


@dataclass(frozen=True, init=False)
class PerturbationResult:
    perturbation_id: str
    __shared_tensor: torch.Tensor = field(repr=False, compare=False)
    __canonical_bytes: bytes = field(repr=False, compare=False)
    tensor_sha256: str
    __normalized_tensor: torch.Tensor = field(repr=False, compare=False)
    normalized_tensor_sha256: str
    valid_pixel_mask: np.ndarray
    fully_valid_patch_mask: np.ndarray
    provenance: Mapping[str, Any]

    def __init__(
        self, *_args: Any, **_kwargs: Any,
    ) -> None:
        raise TypeError(
            "PerturbationResult can only be issued by "
            "generate_stage_c_perturbations"
        )

    @classmethod
    def _create(
        cls, *, perturbation_id: str, tensor: torch.Tensor,
        expected_tensor_sha256: str, valid_pixel_mask: np.ndarray,
        fully_valid_patch_mask: np.ndarray, provenance: Mapping[str, Any],
        issuer_secret: object,
    ) -> "PerturbationResult":
        if issuer_secret is not _PERTURBATION_ISSUER_SECRET:
            raise TypeError("perturbation results can only be issued by the generator")
        self = object.__new__(cls)
        if perturbation_id not in PERTURBATION_IDS:
            raise StageCPerturbationError("unregistered perturbation_id")
        owned = _perturbed_rgb(tensor)
        digest = tensor_sha256(owned)
        if expected_tensor_sha256 != digest:
            raise StageCPerturbationError("perturbation tensor SHA-256 mismatch")
        expected_pixel, expected_patch = perturbation_masks(perturbation_id)
        if not np.array_equal(valid_pixel_mask, expected_pixel):
            raise StageCPerturbationError("valid pixel mask violates frozen geometry")
        if not np.array_equal(fully_valid_patch_mask, expected_patch):
            raise StageCPerturbationError("valid patch mask violates frozen geometry")
        object.__setattr__(self, "perturbation_id", perturbation_id)
        object.__setattr__(self, "_PerturbationResult__shared_tensor", owned)
        object.__setattr__(
            self, "_PerturbationResult__canonical_bytes",
            owned.numpy().tobytes(order="C"),
        )
        object.__setattr__(self, "tensor_sha256", expected_tensor_sha256)
        normalized = (owned - 0.5) / 0.5
        object.__setattr__(
            self, "_PerturbationResult__normalized_tensor", normalized.clone()
        )
        object.__setattr__(
            self, "normalized_tensor_sha256", tensor_sha256(normalized)
        )
        object.__setattr__(self, "valid_pixel_mask", expected_pixel)
        object.__setattr__(self, "fully_valid_patch_mask", expected_patch)
        _json_safe(provenance, "provenance")
        object.__setattr__(self, "provenance", _deep_freeze(dict(provenance)))
        return self

    @property
    def tensor(self) -> torch.Tensor:
        """Return a defensive copy; use ``borrow_tensor`` for shared identity."""

        self.verify_integrity()
        return self.__shared_tensor.clone(memory_format=torch.contiguous_format)

    @property
    def normalized_tensor(self) -> torch.Tensor:
        self.verify_integrity()
        return self.__normalized_tensor.detach().clone()

    def verify_integrity(self) -> None:
        if tensor_sha256(self.__shared_tensor) != self.tensor_sha256:
            raise StageCPerturbationError(
                "shared perturbation tensor failed its frozen SHA-256 integrity check"
            )
        if tensor_sha256(self.__normalized_tensor) != self.normalized_tensor_sha256:
            raise StageCPerturbationError(
                "normalized perturbation tensor failed its SHA-256 integrity check"
            )

    def _restore_shared_tensor(self) -> None:
        restored = np.frombuffer(
            self.__canonical_bytes, dtype=np.float32
        ).reshape(CANONICAL_RGB_SHAPE).copy()
        self.__shared_tensor.copy_(torch.from_numpy(restored))

    def _verify_or_restore(self) -> None:
        if tensor_sha256(self.__shared_tensor) != self.tensor_sha256:
            self._restore_shared_tensor()
            raise StageCPerturbationError(
                "shared perturbation tensor was mutated; canonical bytes were restored"
            )

    @contextmanager
    def borrow_tensor(self):
        """Yield the one shared tensor and fail if a consumer mutates it.

        Sequential consumers observe the exact same Python/Torch object.  The
        tensor is validated both before and after use because Torch tensors do
        not support intrinsic read-only storage.
        """

        self._verify_or_restore()
        try:
            yield self.__shared_tensor
        finally:
            self._verify_or_restore()

    def consume(self, consumer: Any) -> Any:
        if not callable(consumer):
            raise TypeError("consumer must be callable")
        with self.borrow_tensor() as shared:
            result = consumer(shared)
        if _contains_tensor_alias(result, shared):
            raise StageCPerturbationError(
                "consumer must not return the shared perturbation tensor or an alias"
            )
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "perturbation_id": self.perturbation_id,
            "tensor_sha256": self.tensor_sha256,
            "normalized_tensor_sha256": self.normalized_tensor_sha256,
            "tensor_shape": list(self.__shared_tensor.shape),
            "tensor_dtype": "float32",
            "valid_pixel_count": int(self.valid_pixel_mask.sum()),
            "fully_valid_patch_count": int(self.fully_valid_patch_mask.sum()),
            "provenance": _thaw(self.provenance),
        }


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value.item() if isinstance(value, np.generic) else value


def _contains_tensor_alias(value: Any, shared: torch.Tensor, seen: set[int] | None = None) -> bool:
    visited = set() if seen is None else seen
    identity = id(value)
    if identity in visited:
        return False
    visited.add(identity)
    if isinstance(value, torch.Tensor):
        return (
            value is shared
            or value.untyped_storage().data_ptr()
            == shared.untyped_storage().data_ptr()
        )
    if isinstance(value, Mapping):
        return any(
            _contains_tensor_alias(key, shared, visited)
            or _contains_tensor_alias(item, shared, visited)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_tensor_alias(item, shared, visited) for item in value)
    return False


def generate_stage_c_perturbations(
    clean_rgb: Any, *, selection_order: int
) -> Mapping[str, PerturbationResult]:
    clean = _canonical_rgb(clean_rgb)
    order = _strict_selection_order(selection_order)
    values = {
        TRANSLATION_ID: translate_right_down_4(clean),
        SCALE_ID: center_crop_216_resize_224(clean),
        NOISE_ID: gaussian_rgb_std_0_01(clean, selection_order=order),
    }
    results: dict[str, PerturbationResult] = {}
    for perturbation_id in PERTURBATION_IDS:
        pixels, patches = perturbation_masks(perturbation_id)
        provenance: dict[str, Any] = {
            "application_domain": "canonical_224x224_rgb_float32_0_to_1_before_normalization",
            "shared_exact_tensor_across_methods": True,
        }
        if perturbation_id == NOISE_ID:
            provenance.update(
                generator_device="cpu", generator_dtype="float32",
                seed=NOISE_BASE_SEED + order, std_rgb_0_to_1=0.01,
            )
        results[perturbation_id] = PerturbationResult._create(
            perturbation_id=perturbation_id, tensor=values[perturbation_id],
            expected_tensor_sha256=tensor_sha256(values[perturbation_id]),
            valid_pixel_mask=pixels, fully_valid_patch_mask=patches,
            provenance=provenance, issuer_secret=_PERTURBATION_ISSUER_SECRET,
        )
    return MappingProxyType(results)


def _raw_pixel_map(value: Any) -> np.ndarray:
    kind = getattr(value, "kind", None)
    if kind is None:
        raise DisplayBoundaryError(
            "inverse geometry requires an explicitly tagged raw pixel map"
        )
    parsed = ArtifactKind.parse(kind)
    if parsed is not ArtifactKind.RAW:
        raise DisplayBoundaryError(
            f"inverse geometry requires raw input, got {parsed.value}"
        )
    array = np.asarray(value.data)
    if array.shape != PIXEL_SHAPE or array.dtype != np.float32:
        raise StageCPerturbationError(
            "inverse geometry map must be float32 with shape (224,224)"
        )
    if not np.isfinite(array).all():
        raise StageCPerturbationError("inverse geometry map must be finite")
    return array


def inverse_align_pixel_map(value: Any, perturbation_id: str) -> ArtifactArray:
    source = _raw_pixel_map(value)
    aligned = np.zeros(PIXEL_SHAPE, dtype=np.float32)
    if perturbation_id == TRANSLATION_ID:
        aligned[:220, :220] = source[4:224, 4:224]
    elif perturbation_id == SCALE_ID:
        tensor = torch.from_numpy(np.array(source, copy=True))[None, None]
        resized = F.interpolate(
            tensor, size=(216, 216), mode="bilinear", align_corners=False,
            antialias=True,
        )[0, 0]
        aligned[4:220, 4:220] = (
            resized * SCALE_INVERSE_MASS_JACOBIAN
        ).numpy()
    elif perturbation_id == NOISE_ID:
        aligned[...] = source
    else:
        raise StageCPerturbationError(
            f"unknown Stage C perturbation: {perturbation_id!r}"
        )
    return raw_array(
        aligned, metadata={"inverse_of": perturbation_id, "mass_map": True}
    )


__all__ = [
    "NOISE_ID", "PERTURBATION_IDS", "SCALE_ID",
    "SCALE_INVERSE_MASS_JACOBIAN", "StageCPerturbationError",
    "TRANSLATION_ID", "PerturbationResult",
    "center_crop_216_resize_224", "gaussian_rgb_std_0_01",
    "generate_stage_c_perturbations", "inverse_align_pixel_map",
    "perturbation_masks", "tensor_sha256", "translate_right_down_4",
]
