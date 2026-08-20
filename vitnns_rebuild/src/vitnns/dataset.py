from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from .constants import IMAGENET_SYNSET_ORDER_SHA256, OFFICIAL_DEVKIT_SHA256
from .io_utils import sha256_file
from .labels import ImageNet1KMapping, VAL_PATTERN, validation_wnid_map

IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}


@dataclass(frozen=True)
class ManifestRow:
    sample_id: str
    relative_path: str
    image_sha256: str
    size_bytes: int
    width: int
    height: int
    wnid: str | None
    imagenet1k_index: int | None
    label_source: str


def build_manifest(
    root: Path,
    *,
    devkit_path: Path | None = None,
    require_labels: bool = True,
) -> list[ManifestRow]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    mapping = ImageNet1KMapping()
    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise ValueError(f"No images under {root}")
    flat = all(path.parent == root for path in paths)
    val_labels = validation_wnid_map(devkit_path) if flat and devkit_path else None
    if flat and require_labels and val_labels is None:
        raise ValueError("Flat validation images require the verified official ILSVRC2012 devkit")

    rows: list[ManifestRow] = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        if flat:
            match = VAL_PATTERN.match(path.name)
            if not match:
                raise ValueError(f"Malformed validation filename: {path.name}")
            image_number = int(match.group(1))
            wnid = val_labels[image_number] if val_labels is not None else None
            label_source = "official_devkit_val_gt" if wnid else "unlabeled"
        else:
            wnid = path.parent.name
            if not path.name.startswith(f"{wnid}_"):
                raise ValueError(f"Filename WNID does not match parent: {relative}")
            mapping.index_for_wnid(wnid)
            label_source = "parent_wnid_canonical_mapping"
        label = mapping.index_for_wnid(wnid) if wnid is not None else None
        with Image.open(path) as image:
            width, height = image.size
        rows.append(ManifestRow(
            sample_id=f"{root.name}/{relative}",
            relative_path=relative,
            image_sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
            width=width,
            height=height,
            wnid=wnid,
            imagenet1k_index=label,
            label_source=label_source,
        ))
    if len({row.sample_id for row in rows}) != len(rows):
        raise ValueError("Duplicate sample IDs in manifest")
    return rows


def write_manifest_csv(path: Path, rows: list[ManifestRow]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(rows[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def manifest_metadata(root: Path, rows: list[ManifestRow], devkit_path: Path | None) -> dict:
    return {
        "schema_version": 1,
        "dataset_root": str(Path(root).resolve()),
        "sample_count": len(rows),
        "labeled_count": sum(row.imagenet1k_index is not None for row in rows),
        "mapping": {
            "name": "timm ImageNetInfo imagenet-1k synset order",
            "synset_order_sha256": IMAGENET_SYNSET_ORDER_SHA256,
        },
        "official_devkit": None if devkit_path is None else {
            "sha256": OFFICIAL_DEVKIT_SHA256,
            "path": str(Path(devkit_path).resolve()),
        },
    }


class ManifestDataset(Dataset):
    def __init__(self, root: Path, rows: list[ManifestRow], transform: Optional[Callable] = None):
        self.root = Path(root).resolve()
        self.rows = tuple(rows)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor | Image.Image, int | None, ManifestRow]:
        row = self.rows[index]
        with Image.open(self.root / row.relative_path) as source:
            image = source.convert("RGB")
        value = self.transform(image) if self.transform else image
        return value, row.imagenet1k_index, row

