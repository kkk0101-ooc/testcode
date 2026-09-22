"""Outcome-blind N2 four-arm paired-image producer."""
from __future__ import annotations

import hashlib
import io
import math
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tf

from .core import canonical_digest


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _crop(image: Image.Image, size: int, interpolation: InterpolationMode) -> Image.Image:
    resized = tf.resize(image, 248, interpolation=interpolation, antialias=interpolation != InterpolationMode.NEAREST)
    return tf.center_crop(resized, [size, size])


def decode_record(repo: Path, archive: zipfile.ZipFile, row: Mapping[str, Any], crop_size: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    raw_image = (repo / row["image_local_path"]).read_bytes()
    if len(raw_image) != row["image_bytes"] or sha_bytes(raw_image) != row["image_sha256"]:
        raise ValueError("IMAGE_IDENTITY_MISMATCH")
    raw_mask = archive.read(row["annotation_archive_path"] )
    if sha_bytes(raw_mask) != row["annotation_sha256"]:
        raise ValueError("MASK_IDENTITY_MISMATCH")
    with Image.open(io.BytesIO(raw_image)) as image_source, Image.open(io.BytesIO(raw_mask)) as mask_source:
        if image_source.size != mask_source.size:
            raise ValueError("IMAGE_MASK_GEOMETRY_MISMATCH")
        image = np.asarray(_crop(image_source.convert("RGB"), crop_size, InterpolationMode.BICUBIC), dtype=np.float32) / 255.0
        mask_rgb = np.asarray(_crop(mask_source.convert("RGB"), crop_size, InterpolationMode.NEAREST), dtype=np.uint16)
    if np.any(mask_rgb[..., 2]):
        raise ValueError("MASK_BLUE_CHANNEL_NONZERO")
    labels = (mask_rgb[..., 0] + 256 * mask_rgb[..., 1]).astype(np.int32)
    mask = labels == int(row["annotation_label_id"])
    metadata = {
        "source_image_id": row["source_image_id"],
        "source_group_id": row["source_group_id"],
        "image_sha256": row["image_sha256"],
        "annotation_sha256": row["annotation_sha256"],
        "annotation_wnid": row["annotation_wnid"],
        "target_model_class_index": int(row["target_model_class_index"]),
        "mask_fraction": float(mask.mean()),
    }
    return np.ascontiguousarray(image), np.ascontiguousarray(mask), metadata


def remove_donor_foreground(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if image.shape[:2] != mask.shape or image.shape[2] != 3:
        raise ValueError("DONOR_GEOMETRY_MISMATCH")
    if mask.all():
        raise ValueError("DONOR_HAS_NO_BACKGROUND")
    import cv2
    expanded = ndimage.binary_dilation(mask, iterations=3)
    uint8 = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
    result = cv2.inpaint(uint8, expanded.astype(np.uint8) * 255, 3.0, cv2.INPAINT_TELEA)
    return np.ascontiguousarray(result.astype(np.float32) / 255.0)


def composite(recipient: np.ndarray, recipient_mask: np.ndarray, background: np.ndarray, radius: float) -> np.ndarray:
    alpha_image = Image.fromarray((recipient_mask.astype(np.uint8) * 255), mode="L").filter(ImageFilter.GaussianBlur(radius=radius))
    alpha = np.asarray(alpha_image, dtype=np.float32)[..., None] / 255.0
    result = alpha * recipient + (1.0 - alpha) * background
    return np.ascontiguousarray(np.clip(result, 0.0, 1.0), dtype=np.float32)


def low_level_metrics(base: np.ndarray, other: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    delta = other.astype(np.float64) - base.astype(np.float64)
    l1 = float(np.mean(np.abs(delta)))
    l2 = float(np.sqrt(np.mean(np.square(delta))))
    rgb = float(np.linalg.norm(delta.mean(axis=(0, 1))))
    grey = delta.mean(axis=2)
    highpass = float(np.mean(np.abs(grey - ndimage.gaussian_filter(grey, 2.0))))
    dilated = ndimage.binary_dilation(mask, iterations=3)
    eroded = ndimage.binary_erosion(mask, iterations=3)
    ring = dilated ^ eroded
    boundary = float(np.mean(np.abs(delta[ring]))) if ring.any() else 0.0
    return {"L1": l1, "L2": l2, "RGB_MEAN_SHIFT": rgb, "HIGHPASS_L1": highpass, "BOUNDARY_L1": boundary}


def matching_score(semantic: Mapping[str, float], null: Mapping[str, float]) -> tuple[float, float, dict[str, float]]:
    ratios = {}
    logs = []
    for key in semantic:
        a = max(float(semantic[key]), 1e-8); b = max(float(null[key]), 1e-8)
        ratios[key] = max(a, b) / min(a, b)
        logs.append(math.log(a / b))
    return float(math.sqrt(sum(value * value for value in logs))), float(max(ratios.values())), ratios


def array_digest(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode()); digest.update(b"\0")
    digest.update(str(tuple(value.shape)).encode()); digest.update(b"\0")
    digest.update(value.tobytes())
    return digest.hexdigest()
